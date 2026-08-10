use super::*;

#[test]
fn training_interruption_is_visible_between_steps() {
    let interrupter = burn_train::Interrupter::new();
    assert_eq!(training_interruption_reason(&interrupter), None);

    interrupter.stop(Some("non-finite train loss"));

    assert_eq!(
        training_interruption_reason(&interrupter).as_deref(),
        Some("non-finite train loss")
    );
}

#[test]
fn policy_capability_feedback_uses_action_accuracy_for_exact_source() {
    let source_label = burn_dragon_universality::ruliad_source_capability_label(
        "formal_proof",
        "select_proof_action",
        2,
        "proof_action_step",
    );
    let result = RuliadPolicyRolloutProbeResult {
        summary: RuliadPolicyRolloutProbeSummary::default(),
        difficulty_summaries: BTreeMap::new(),
        source_summaries: BTreeMap::from([(
            source_label.clone(),
            RuliadPolicyRolloutProbeSummary {
                items: 4,
                solved: 3,
                steps: 10,
                valid_actions: 10,
                invalid_actions: 0,
                repeated_states: 0,
                backtracks: 0,
                scored_states: 10,
                scored_actions: 40,
                top1_expert_actions: 8,
                frontier_exhaustions: 0,
                solved_goals: 7,
                total_goals: 8,
            },
        )]),
    };

    let feedback = ruliad_policy_capability_feedback(&result);

    assert_eq!(feedback.len(), 1);
    assert_eq!(feedback[0].group_label, source_label);
    assert_eq!(feedback[0].item_count, 10);
    assert!((feedback[0].verifier_rate - 0.8).abs() < 1.0e-6);
    assert!((feedback[0].partial_credit_rate - 0.875).abs() < 1.0e-6);
    assert!((feedback[0].schema_valid_wrong_rate - 0.2).abs() < 1.0e-6);
    assert_eq!(feedback[0].completion_health_rate, 1.0);
}

#[test]
fn semantic_action_curriculum_never_mixes_free_and_policy_evidence() {
    let semantic_label = burn_dragon_universality::ruliad_source_capability_label(
        "formal_proof",
        "select_proof_action",
        0,
        "proof_action_step",
    );
    let category_label = burn_dragon_universality::ruliad_source_capability_label(
        "category",
        "verify_category_law",
        0,
        "ok,l,r",
    );
    let feedback = |group_label: String, verifier_rate: f32| {
        burn_dragon_universality::RuliadCapabilityFeedback {
            group_label,
            item_count: 8,
            verifier_rate,
            partial_credit_rate: verifier_rate,
            schema_valid_wrong_rate: 1.0 - verifier_rate,
            malformed_rate: 0.0,
            missing_rate: 0.0,
            completion_health_rate: 1.0,
        }
    };

    let between_policy_probes = merge_ruliad_policy_capability_feedback(
        vec![
            feedback(semantic_label.clone(), 0.0),
            feedback(category_label.clone(), 0.5),
        ],
        true,
        None,
    );
    assert_eq!(between_policy_probes.len(), 1);
    assert_eq!(between_policy_probes[0].group_label, category_label);

    let policy_result = RuliadPolicyRolloutProbeResult {
        source_summaries: BTreeMap::from([(
            semantic_label.clone(),
            RuliadPolicyRolloutProbeSummary {
                items: 2,
                scored_states: 10,
                top1_expert_actions: 8,
                valid_actions: 4,
                solved_goals: 3,
                total_goals: 4,
                ..Default::default()
            },
        )]),
        ..Default::default()
    };
    let on_policy_probe = merge_ruliad_policy_capability_feedback(
        vec![feedback(semantic_label.clone(), 0.0)],
        true,
        Some(&policy_result),
    );
    assert_eq!(on_policy_probe.len(), 1);
    assert_eq!(on_policy_probe[0].group_label, semantic_label);
    assert!((on_policy_probe[0].verifier_rate - 0.8).abs() < 1.0e-6);
}

#[test]
fn ruliad_policy_promotion_gate_requires_closed_loop_quality() {
    let gate = crate::config::RuliadPolicyPromotionGateConfig {
        enabled: true,
        ..Default::default()
    };
    let status = ruliad_policy_promotion_gate_status(
        RuliadPolicyRolloutProbeSummary {
            items: 16,
            solved: 7,
            steps: 160,
            valid_actions: 150,
            invalid_actions: 10,
            repeated_states: 60,
            backtracks: 80,
            scored_states: 150,
            scored_actions: 600,
            top1_expert_actions: 50,
            frontier_exhaustions: 2,
            solved_goals: 31,
            total_goals: 48,
        },
        gate,
    );

    assert!(!status.passed);
    assert!(
        status
            .reasons
            .iter()
            .any(|reason| reason.starts_with("solve_rate="))
    );
    assert!(
        status
            .reasons
            .iter()
            .any(|reason| reason.starts_with("repeated_state_rate="))
    );
    assert!(
        status
            .reasons
            .iter()
            .any(|reason| reason.starts_with("backtrack_rate="))
    );
}

#[test]
fn ruliad_policy_promotion_gate_accepts_solved_stable_trajectory() {
    let gate = crate::config::RuliadPolicyPromotionGateConfig {
        enabled: true,
        ..Default::default()
    };
    let status = ruliad_policy_promotion_gate_status(
        RuliadPolicyRolloutProbeSummary {
            items: 16,
            solved: 12,
            steps: 128,
            valid_actions: 128,
            invalid_actions: 0,
            repeated_states: 16,
            backtracks: 2,
            scored_states: 128,
            scored_actions: 512,
            top1_expert_actions: 96,
            frontier_exhaustions: 0,
            solved_goals: 45,
            total_goals: 48,
        },
        gate,
    );

    assert!(status.passed, "{:?}", status.reasons);
}

#[test]
fn constrained_correctness_scores_equivalent_actions_without_preferred_bias() {
    let job = RuliadCorrectnessConstrainedPolicyJob {
        presentations: vec![RuliadPolicyActionPresentation {
            rotation: 0,
            prompt_tokens: vec![1, 2],
            candidate_tokens: vec![vec![3], vec![4], vec![5]],
            answer_contract: Default::default(),
        }],
        prompt_contexts: Vec::new(),
        base_context: None,
        selected_index: 1,
        equivalent_indices: vec![1, 2],
    };
    let mut summary = RuliadCorrectnessConstrainedPolicySummary::default();
    record_ruliad_correctness_constrained_scores(
        &mut summary,
        &job,
        &[0.1f32.ln(), 0.5f32.ln(), 0.4f32.ln()],
    );
    record_ruliad_correctness_constrained_scores(
        &mut summary,
        &job,
        &[0.1f32.ln(), 0.4f32.ln(), 0.5f32.ln()],
    );

    assert_eq!(summary.items, 2);
    assert_eq!(summary.equivalent_top1, 2);
    assert_eq!(summary.preferred_top1, 1);
    assert!((summary.equivalent_nll_sum / 2.0 + 0.9f64.ln()).abs() < 1.0e-6);
    assert!(summary.valid_invalid_margin_sum > 0.0);
    assert_eq!(summary.valid_invalid_margin_items, 2);
}

#[test]
fn constrained_correctness_context_swap_detects_prompt_dependence() {
    let job = RuliadCorrectnessConstrainedPolicyJob {
        presentations: vec![RuliadPolicyActionPresentation {
            rotation: 0,
            prompt_tokens: vec![1, 2],
            candidate_tokens: vec![vec![3], vec![4], vec![5]],
            answer_contract: Default::default(),
        }],
        prompt_contexts: Vec::new(),
        base_context: None,
        selected_index: 1,
        equivalent_indices: vec![1],
    };
    let mut summary = RuliadCorrectnessConstrainedPolicySummary::default();
    record_ruliad_correctness_context_swap(
        &mut summary,
        &job,
        &[0.1f32.ln(), 0.8f32.ln(), 0.1f32.ln()],
        &[0.4f32.ln(), 0.2f32.ln(), 0.4f32.ln()],
    );

    assert_eq!(summary.context_swap_items, 1);
    assert_eq!(summary.context_swap_equivalent_top1, 0);
    assert_eq!(summary.context_swap_top1_changes, 1);
    assert!((summary.context_swap_equivalent_probability_drop_sum - 0.6).abs() < 1.0e-6);
    assert!(summary.context_swap_js_divergence_sum > 0.0);
}

#[test]
fn constrained_correctness_counterfactual_target_requires_preference_change() {
    let counterfactual_job = RuliadCorrectnessConstrainedPolicyJob {
        presentations: vec![RuliadPolicyActionPresentation {
            rotation: 0,
            prompt_tokens: vec![1, 2],
            candidate_tokens: vec![vec![3], vec![4], vec![5]],
            answer_contract: Default::default(),
        }],
        prompt_contexts: Vec::new(),
        base_context: None,
        selected_index: 2,
        equivalent_indices: vec![2],
    };
    let mut summary = RuliadCorrectnessConstrainedPolicySummary::default();
    record_ruliad_correctness_counterfactual_target(
        &mut summary,
        &counterfactual_job,
        &[0.1f32.ln(), 0.8f32.ln(), 0.1f32.ln()],
        &[0.1f32.ln(), 0.2f32.ln(), 0.7f32.ln()],
    );

    assert_eq!(summary.counterfactual_target_items, 1);
    assert_eq!(summary.counterfactual_target_equivalent_top1, 1);
    assert_eq!(summary.counterfactual_target_top1_changes, 1);
    assert!(summary.counterfactual_target_equivalent_probability_gain_sum > 0.5);
    assert!(summary.counterfactual_target_js_divergence_sum > 0.0);
}

#[test]
fn context_swap_changes_only_current_and_target_proof_state() {
    let config = burn_dragon_universality::ruliad::formal::RuliadFormalGeneratorConfig::default();
    let original_bundle =
        burn_dragon_universality::ruliad::formal::generate_formal_bundle(71, config)
            .expect("original bundle");
    let donor_bundle = burn_dragon_universality::ruliad::formal::generate_formal_bundle(73, config)
        .expect("donor bundle");
    let original = burn_dragon_universality::ruliad::oracle_proof_action_set(
        &original_bundle.problem,
        &original_bundle.certificate,
        0,
        4,
    )
    .expect("original actions");
    let donor = burn_dragon_universality::ruliad::oracle_proof_action_set(
        &donor_bundle.problem,
        &donor_bundle.certificate,
        0,
        4,
    )
    .expect("donor actions");
    let swapped = proof_action_set_with_swapped_state(&original, &donor);

    assert_eq!(swapped.goal, original.goal);
    assert_eq!(swapped.candidates, original.candidates);
    assert_eq!(swapped.selected_index, original.selected_index);
    assert_eq!(swapped.equivalent_indices, original.equivalent_indices);
    assert_eq!(swapped.current, donor.current);
    assert_eq!(swapped.target, donor.target);
}

#[test]
fn constrained_correctness_exposes_canonical_and_worst_orbit_behavior() {
    let job = RuliadCorrectnessConstrainedPolicyJob {
        presentations: vec![RuliadPolicyActionPresentation {
            rotation: 0,
            prompt_tokens: vec![1, 2],
            candidate_tokens: vec![vec![3], vec![4], vec![5]],
            answer_contract: Default::default(),
        }],
        prompt_contexts: Vec::new(),
        base_context: None,
        selected_index: 1,
        equivalent_indices: vec![1, 2],
    };
    let orbit = crate::train::ruliad_policy::semantic_action_orbit_summary(
        &[
            (0, vec![0.1f32.ln(), 0.6f32.ln(), 0.3f32.ln()]),
            (1, vec![0.3f32.ln(), 0.5f32.ln(), 0.2f32.ln()]),
            (2, vec![0.2f32.ln(), 0.55f32.ln(), 0.25f32.ln()]),
        ],
        3,
    )
    .expect("semantic orbit");
    let mut summary = RuliadCorrectnessConstrainedPolicySummary::default();
    record_ruliad_correctness_orbit_diagnostics(&mut summary, &job, &orbit);

    assert_eq!(summary.canonical_items, 1);
    assert_eq!(summary.canonical_equivalent_top1, 1);
    assert_eq!(summary.canonical_preferred_top1, 1);
    assert_eq!(summary.worst_presentation_items, 1);
    assert_eq!(summary.worst_presentation_equivalent_top1, 0);
    assert_eq!(summary.complete_orbit_items, 1);
    assert_eq!(summary.presentation_rows, 3);
    assert_eq!(summary.presentation_equivalent_top1, 2);
    assert_eq!(summary.presentation_preferred_top1, 1);
    assert!(summary.orbit_js_divergence_sum > 0.0);
    assert!((summary.orbit_top1_consensus_fraction_sum - 1.0 / 3.0).abs() < 1.0e-6);
    assert!((summary.worst_presentation_equivalent_nll_sum + 0.45f64.ln()).abs() < 1.0e-6);
    assert!(summary.worst_presentation_valid_invalid_margin_sum < 0.0);
}

#[test]
fn policy_probe_balances_candidate_presentation_without_changing_action() {
    let bundle = burn_dragon_universality::ruliad::formal::generate_formal_bundle(
        71,
        burn_dragon_universality::ruliad::formal::RuliadFormalGeneratorConfig::default(),
    )
    .expect("formal bundle");
    let actions = burn_dragon_universality::ruliad::oracle_proof_action_set(
        &bundle.problem,
        &bundle.certificate,
        0,
        4,
    )
    .expect("action set");
    let selected_step = actions.selected().expect("selected action").step.clone();
    for desired_index in 0..actions.candidates.len() {
        let rotated = apply_ruliad_policy_probe_candidate_symmetry(
            actions.clone(),
            crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation,
            desired_index,
        )
        .expect("balanced action set");
        assert_eq!(rotated.selected_index, desired_index);
        assert_eq!(
            rotated.selected().expect("rotated selected action").step,
            selected_step
        );
    }
    let orbit = ruliad_policy_action_presentations(
        &actions,
        crate::config::RuliadProofPolicyCandidateSymmetry::CyclicOrbitAverage,
        0,
    )
    .expect("cyclic orbit");
    assert_eq!(orbit.len(), actions.candidates.len());
    for original_index in 0..actions.candidates.len() {
        assert!(orbit.iter().all(|(rotation, presented)| {
            let presented_index =
                (original_index + actions.candidates.len() - rotation) % actions.candidates.len();
            presented.candidates[presented_index] == actions.candidates[original_index]
        }));
    }
    assert_eq!(
        crate::config::RuliadPolicyProbeConfig::default().candidate_symmetry,
        crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation
    );
}

#[test]
fn validation_primary_signal_follows_the_explicit_objective() {
    let report = DynamicValidationReport {
        loss: 9.0,
        stream_warm_loss: Some(2.5),
        ..DynamicValidationReport::default()
    };
    assert_eq!(report.primary_loss(), 9.0);

    let warm_report = DynamicValidationReport {
        objective: crate::config::TrainingValidationObjective::StreamWarm,
        loss: 9.0,
        stream_warm_loss: Some(2.5),
        ..DynamicValidationReport::default()
    };
    assert_eq!(warm_report.primary_loss(), 2.5);

    let non_finite = DynamicValidationReport {
        objective: crate::config::TrainingValidationObjective::StreamWarm,
        loss: 3.0,
        stream_warm_loss: Some(f64::NAN),
        ..DynamicValidationReport::default()
    };
    assert!(non_finite.primary_loss().is_nan());

    assert!(
        select_validation_objective_loss(
            crate::config::TrainingValidationObjective::StreamWarm,
            3.0,
            Some(2.0),
            None,
        )
        .is_err()
    );
}
use burn::data::dataloader::{DataLoaderIterator, Progress};
#[cfg(feature = "ddp")]
use burn::module::list_param_ids;
use burn::tensor::TensorData;
use burn_autodiff::Autodiff;
#[cfg(feature = "ddp")]
use burn_collective::reset_collective;
use burn_ndarray::NdArray;
use burn_train::checkpoint::CheckpointingAction;
#[cfg(feature = "ddp")]
use std::sync::{Mutex, OnceLock};
#[cfg(feature = "ddp")]
use tempfile::tempdir;

type TestBackend = Autodiff<NdArray<f32>>;
type TestValidBackend = ValidBackend<TestBackend>;
type TestForwardBackend = NdArray<f32>;

fn degeneracy_stats(
    entropy_bits: f64,
    mean_max_probability: f64,
    distinct_2_fraction: f64,
    repetition_fraction: f64,
) -> crate::train::steps::OutputDegeneracyStats {
    crate::train::steps::OutputDegeneracyStats {
        token_count: 128,
        entropy_bits,
        mean_max_probability,
        argmax_unique_fraction: 0.02,
        eos_fraction: 0.0,
        repetition_fraction,
        distinct_1_fraction: 0.02,
        distinct_2_fraction,
        period_2_fraction: 0.0,
        period_3_fraction: 0.0,
        max_period_2_to_16_fraction: 0.0,
        max_period_2_to_64_fraction: 0.0,
        dominant_period_2_to_64: 0,
        prompt_max_period_2_to_64_fraction: 0.0,
        prompt_dominant_period_2_to_64: 0,
        prompt_tokens: Vec::new(),
        generated_tokens: Vec::new(),
    }
}

fn ruliad_degeneracy_gates() -> burn_dragon_train::TrainingGatesConfig {
    burn_dragon_train::TrainingGatesConfig {
        degeneracy_entropy_min_bits: 2.0,
        degeneracy_max_probability_max: 0.90,
        degeneracy_argmax_unique_min_fraction: 0.08,
        degeneracy_distinct_2_min_fraction: 0.20,
        degeneracy_repetition_max_fraction: 0.75,
        degeneracy_period_2_max_fraction: 0.90,
        degeneracy_period_3_max_fraction: 0.90,
        degeneracy_period_2_to_16_max_fraction: 0.90,
        degeneracy_period_2_to_64_max_fraction: 0.90,
        degeneracy_patience: 1,
        ..burn_dragon_train::TrainingGatesConfig::default()
    }
}

fn free_run_training_with_gates(
    gates: burn_dragon_train::TrainingGatesConfig,
) -> TrainingHyperparameters {
    let mut training = tiny_training_hparams();
    training.gates = gates;
    training
}

fn policy_contract_training(
    contract: crate::config::RuliadCheckpointCapabilityContract,
) -> TrainingHyperparameters {
    let mut training = free_run_training_with_gates(ruliad_degeneracy_gates());
    training.ruliad_policy_probe.enabled = true;
    training.ruliad_policy_probe.checkpoint_capability_contract = contract;
    training.ruliad_policy_probe.promotion_gate.enabled = true;
    training
}

fn healthy_policy_rollout(solved: usize, solved_goals: usize) -> RuliadPolicyRolloutProbeResult {
    RuliadPolicyRolloutProbeResult {
        summary: RuliadPolicyRolloutProbeSummary {
            items: 32,
            solved,
            steps: 128,
            valid_actions: 128,
            invalid_actions: 0,
            repeated_states: 8,
            backtracks: 4,
            scored_states: 128,
            scored_actions: 512,
            top1_expert_actions: 96,
            frontier_exhaustions: 0,
            solved_goals,
            total_goals: 96,
        },
        ..Default::default()
    }
}

#[test]
fn policy_regression_requires_non_overlapping_wilson_intervals() {
    let z = 1.959_963_984_540_054;
    let best = BinomialObservation {
        successes: 18,
        trials: 32,
    };
    let noisy_current = BinomialObservation {
        successes: 15,
        trials: 32,
    };
    let collapsed_current = BinomialObservation {
        successes: 4,
        trials: 32,
    };

    assert!(!binomial_observation_materially_regressed(
        best,
        noisy_current,
        0.125,
        z
    ));
    assert!(binomial_observation_materially_regressed(
        best,
        collapsed_current,
        0.125,
        z
    ));

    let best_goal = BinomialObservation {
        successes: 66,
        trials: 118,
    };
    let noisy_goal = BinomialObservation {
        successes: 46,
        trials: 119,
    };
    assert!(!binomial_observation_materially_regressed(
        best_goal, noisy_goal, 0.25, z
    ));
}

#[test]
fn policy_best_observation_prefers_stronger_statistical_evidence() {
    let z = 1.959_963_984_540_054;
    let small_perfect = BinomialObservation {
        successes: 1,
        trials: 1,
    };
    let supported = BinomialObservation {
        successes: 24,
        trials: 32,
    };

    assert_eq!(
        small_perfect.prefer_stronger_evidence(supported, z),
        supported,
        "a single perfect observation must not anchor the continual-regression baseline"
    );
}

#[test]
fn policy_promotion_ineligibility_is_not_a_hard_quality_collapse() {
    let training = policy_contract_training(
        crate::config::RuliadCheckpointCapabilityContract::ClosedLoopPolicy,
    );
    let validation = DynamicValidationReport {
        ruliad_policy_rollout: Some(healthy_policy_rollout(7, 38)),
        ..Default::default()
    };

    let status = validation_capability_gate_status(&validation, &training);
    assert!(
        !status.passed,
        "sample is intentionally below promotion floors"
    );
    assert!(!validation_capability_quality_collapse(
        &validation,
        &training
    ));
}

fn ruliad_eval_report(
    verifier_accuracy: f32,
    semantic_accuracy: f32,
    mean_partial_progress: f32,
    certificate_prefix_coverage: f32,
) -> burn_dragon_universality::RuliadEvalReport {
    let item_count = 100usize;
    burn_dragon_universality::RuliadEvalReport {
        version: burn_dragon_universality::ruliad::RULIAD_EVAL_REPORT_VERSION,
        reasoning_score_version: burn_dragon_universality::ruliad::RULIAD_REASONING_SCORE_VERSION,
        dataset_name: "test".to_string(),
        item_count,
        scored_count: item_count,
        exact_match_count: 0,
        semantic_match_count: (semantic_accuracy.clamp(0.0, 1.0) * item_count as f32).round()
            as usize,
        verifier_match_count: (verifier_accuracy.clamp(0.0, 1.0) * item_count as f32).round()
            as usize,
        partial_credit_count: (mean_partial_progress.clamp(0.0, 1.0) * item_count as f32).round()
            as usize,
        schema_valid_wrong_count: 0,
        malformed_completion_count: 0,
        missing_completion_count: 0,
        unexpected_completion_count: 0,
        exact_accuracy: 0.0,
        semantic_accuracy,
        verifier_accuracy,
        partial_credit_rate: mean_partial_progress,
        mean_partial_progress,
        answer_field_correct_count: (mean_partial_progress.clamp(0.0, 1.0) * item_count as f32)
            .round() as usize,
        answer_field_expected_count: item_count,
        answer_field_accuracy: mean_partial_progress,
        answer_field_observed_count: item_count,
        answer_field_coverage: 1.0,
        answer_terminated_count: item_count,
        answer_termination_rate: 1.0,
        mean_completion_quality: 1.0,
        expected_answer_distinct_fraction: 1.0,
        actual_answer_distinct_fraction: 1.0,
        actual_answer_dominant_fraction: 0.01,
        expected_field_value_distinct_fraction: 1.0,
        actual_field_value_distinct_fraction: 1.0,
        field_value_distinct_ratio: 1.0,
        actual_field_value_dominant_fraction: 0.01,
        presented_action_expected_count: 0,
        presented_action_match_count: 0,
        presented_action_rate: 0.0,
        mean_certificate_prefix_coverage: certificate_prefix_coverage,
        mean_completion_tokens: 12.0,
        canary_count: 0,
        canary_semantic_match_count: 0,
        family_scores: Vec::new(),
        task_scores: Vec::new(),
        difficulty_scores: Vec::new(),
        answer_contract_scores: Vec::new(),
        source_scores: Vec::new(),
        math_domain_scores: Vec::new(),
        reasoning_mode_scores: Vec::new(),
        failures: Vec::new(),
    }
}

#[test]
fn checkpoint_promotion_rejects_loss_only_ruliad_progress_when_free_run_is_flat() {
    let gates = ruliad_degeneracy_gates();
    let training = free_run_training_with_gates(gates);
    let best_competence = ruliad_competence_key(&ruliad_eval_report(0.0, 0.0, 0.0, 0.0));
    let validation = DynamicValidationReport {
        objective: crate::config::TrainingValidationObjective::FixedHoldout,
        loss: 0.01,
        source_weighted_loss: None,
        stream_warm_loss: None,
        output_degeneracy: None,
        ruliad_eval_report: Some(ruliad_eval_report(0.0, 0.0, 0.0, 0.0)),
        ruliad_policy_rollout: None,
    };

    assert!(!should_promote_checkpoint(
        &validation,
        Some(1.0),
        best_competence,
        None,
        &training
    ));
}

#[test]
fn checkpoint_promotion_prefers_free_run_ruliad_competence_over_teacher_forced_loss() {
    let gates = ruliad_degeneracy_gates();
    let training = free_run_training_with_gates(gates);
    let best_competence = ruliad_competence_key(&ruliad_eval_report(0.0, 0.0, 0.0, 0.0));
    let validation = DynamicValidationReport {
        objective: crate::config::TrainingValidationObjective::FixedHoldout,
        loss: 1.5,
        source_weighted_loss: None,
        stream_warm_loss: None,
        output_degeneracy: None,
        ruliad_eval_report: Some(ruliad_eval_report(0.01, 0.01, 0.10, 0.10)),
        ruliad_policy_rollout: None,
    };

    assert!(should_promote_checkpoint(
        &validation,
        Some(1.0),
        best_competence,
        None,
        &training
    ));
}

#[test]
fn checkpoint_promotion_rejects_loss_only_when_capability_gate_fails() {
    let mut gates = ruliad_degeneracy_gates();
    gates.capability_schema_wrong_max_rate = 0.25;
    let mut report = ruliad_eval_report(0.25, 0.25, 0.25, 0.25);
    report.schema_valid_wrong_count = 40;
    let training = free_run_training_with_gates(gates);
    let validation = DynamicValidationReport {
        objective: crate::config::TrainingValidationObjective::FixedHoldout,
        loss: 0.01,
        source_weighted_loss: None,
        stream_warm_loss: None,
        output_degeneracy: None,
        ruliad_eval_report: Some(report),
        ruliad_policy_rollout: None,
    };

    assert!(!should_promote_checkpoint(
        &validation,
        None,
        None,
        None,
        &training
    ));
}

#[test]
fn closed_loop_checkpoint_contract_promotes_policy_with_zero_free_run_verifier() {
    let training = policy_contract_training(
        crate::config::RuliadCheckpointCapabilityContract::ClosedLoopPolicy,
    );
    let validation = DynamicValidationReport {
        loss: 3.0,
        ruliad_eval_report: Some(ruliad_eval_report(0.0, 0.0, 0.0, 0.0)),
        ruliad_policy_rollout: Some(healthy_policy_rollout(24, 90)),
        ..Default::default()
    };

    assert!(should_promote_checkpoint(
        &validation,
        Some(1.0),
        None,
        None,
        &training,
    ));
}

#[test]
fn joint_checkpoint_contract_rejects_zero_free_run_verifier() {
    let training =
        policy_contract_training(crate::config::RuliadCheckpointCapabilityContract::Joint);
    let validation = DynamicValidationReport {
        loss: 0.01,
        ruliad_eval_report: Some(ruliad_eval_report(0.0, 0.0, 0.0, 0.0)),
        ruliad_policy_rollout: Some(healthy_policy_rollout(24, 90)),
        ..Default::default()
    };

    assert!(!should_promote_checkpoint(
        &validation,
        Some(1.0),
        None,
        None,
        &training,
    ));
}

#[test]
fn closed_loop_checkpoint_contract_fails_closed_without_policy_probe() {
    let training = policy_contract_training(
        crate::config::RuliadCheckpointCapabilityContract::ClosedLoopPolicy,
    );
    let validation = DynamicValidationReport {
        loss: 0.01,
        ruliad_eval_report: Some(ruliad_eval_report(1.0, 1.0, 1.0, 1.0)),
        ruliad_policy_rollout: None,
        ..Default::default()
    };

    assert!(!should_promote_checkpoint(
        &validation,
        Some(1.0),
        None,
        None,
        &training,
    ));
}

#[test]
fn closed_loop_checkpoint_contract_rejects_policy_regression_despite_loss_gain() {
    let training = policy_contract_training(
        crate::config::RuliadCheckpointCapabilityContract::ClosedLoopPolicy,
    );
    let best = healthy_policy_rollout(24, 90);
    let validation = DynamicValidationReport {
        loss: 0.01,
        ruliad_eval_report: Some(ruliad_eval_report(0.0, 0.0, 0.0, 0.0)),
        ruliad_policy_rollout: Some(healthy_policy_rollout(20, 88)),
        ..Default::default()
    };

    assert!(!should_promote_checkpoint(
        &validation,
        Some(1.0),
        None,
        ruliad_policy_competence_key(&best),
        &training,
    ));
}

#[test]
fn closed_loop_recovery_checkpoint_tracks_policy_competence() {
    let training = policy_contract_training(
        crate::config::RuliadCheckpointCapabilityContract::ClosedLoopPolicy,
    );
    let validation = DynamicValidationReport {
        loss: 3.0,
        ruliad_eval_report: Some(ruliad_eval_report(0.0, 0.0, 0.0, 0.0)),
        ruliad_policy_rollout: Some(healthy_policy_rollout(24, 90)),
        ..Default::default()
    };
    let mut best_free = None;
    let mut best_policy = None;

    assert!(update_ruliad_recovery_competence(
        &validation,
        training.ruliad_policy_probe.checkpoint_capability_contract,
        training.ruliad_policy_probe.promotion_gate,
        &training.gates,
        &mut best_free,
        &mut best_policy,
    ));
    assert_eq!(best_free, None);
    assert_eq!(
        best_policy,
        validation
            .ruliad_policy_rollout
            .as_ref()
            .and_then(ruliad_policy_competence_key)
    );
}

#[test]
fn ruliad_correctness_progress_suppresses_loss_only_regression() {
    let state = ContinualLearningStabilityState {
        best_ruliad_verifier_accuracy: Some(0.1875),
        best_ruliad_partial_progress: Some(0.2917),
        ..Default::default()
    };
    let validation = DynamicValidationReport {
        loss: 0.454,
        ruliad_eval_report: Some(ruliad_eval_report(0.21875, 0.21875, 0.2917, 0.0)),
        ..Default::default()
    };

    assert!(validation_ruliad_capability_improved(
        &validation,
        &state,
        crate::config::RuliadCheckpointCapabilityContract::FreeRunText,
    ));
}

#[test]
fn flat_ruliad_correctness_does_not_suppress_loss_regression() {
    let state = ContinualLearningStabilityState {
        best_ruliad_verifier_accuracy: Some(0.1875),
        best_ruliad_partial_progress: Some(0.2917),
        ..Default::default()
    };
    let validation = DynamicValidationReport {
        loss: 0.454,
        ruliad_eval_report: Some(ruliad_eval_report(0.1875, 0.1875, 0.2917, 0.0)),
        ..Default::default()
    };

    assert!(!validation_ruliad_capability_improved(
        &validation,
        &state,
        crate::config::RuliadCheckpointCapabilityContract::FreeRunText,
    ));
}

#[test]
fn ruliad_correctness_regression_threshold_ignores_one_item_probe_noise() {
    assert!(!ruliad_metric_materially_regressed(
        0.21875, 0.1875, 32, 0.125
    ));
}

#[test]
fn ruliad_correctness_regression_threshold_flags_material_probe_drop() {
    assert!(ruliad_metric_materially_regressed(
        0.21875, 0.1875, 128, 0.125
    ));
}

#[test]
fn ruliad_capability_gate_status_flags_malformed_missing_and_output_collapse() {
    let mut gates = ruliad_degeneracy_gates();
    gates.capability_malformed_max_rate = 0.02;
    gates.capability_missing_max_rate = 0.02;
    gates.capability_completion_health_min_rate = 0.80;
    gates.capability_output_entropy_min_bits = 1.25;
    gates.capability_distinct_2_min_fraction = 0.30;
    let mut report = ruliad_eval_report(0.25, 0.25, 0.25, 0.25);
    report.malformed_completion_count = 5;
    report.missing_completion_count = 3;
    let stats = degeneracy_stats(0.5, 0.9, 0.1, 0.0);

    let status = ruliad_capability_gate_status(&report, Some(&stats), &gates);

    assert!(!status.passed);
    assert!(
        status
            .reasons
            .iter()
            .any(|reason| reason.starts_with("malformed_rate="))
    );
    assert!(
        status
            .reasons
            .iter()
            .any(|reason| reason.starts_with("missing_rate="))
    );
    assert!(
        status
            .reasons
            .iter()
            .any(|reason| reason.starts_with("output_entropy_bits="))
    );
    assert!(
        status
            .reasons
            .iter()
            .any(|reason| reason.starts_with("output_distinct2="))
    );
}

#[test]
fn ruliad_capability_gate_status_flags_low_quality_answer_collapse() {
    let mut gates = ruliad_degeneracy_gates();
    gates.capability_completion_health_min_rate = 0.80;
    gates.capability_answer_distinct_min_fraction = 0.30;
    let mut report = ruliad_eval_report(0.0, 0.0, 0.0, 0.0);
    report.mean_completion_quality = 0.0;
    report.actual_answer_distinct_fraction = 0.01;

    let status = ruliad_capability_gate_status(&report, None, &gates);

    assert!(!status.passed);
    assert!(
        status
            .reasons
            .iter()
            .any(|reason| reason.starts_with("completion_health=0.000<")),
        "{:?}",
        status.reasons
    );
    assert!(
        status
            .reasons
            .iter()
            .any(|reason| reason.starts_with("answer_distinct=0.010<")
                && reason.contains("expected_distinct=1.000")),
        "{:?}",
        status.reasons
    );
}

#[test]
fn ruliad_capability_gate_status_rejects_zero_verifier_even_when_format_is_healthy() {
    let mut gates = ruliad_degeneracy_gates();
    gates.capability_completion_health_min_rate = 0.10;
    gates.capability_answer_distinct_min_fraction = 0.20;
    gates.capability_field_value_distinct_ratio_min = 0.35;
    gates.capability_field_value_dominance_max = 0.85;
    let mut report = ruliad_eval_report(0.0, 0.0, 0.75, 0.75);
    report.mean_completion_quality = 1.0;
    report.actual_answer_distinct_fraction = 1.0;
    report.field_value_distinct_ratio = 1.0;
    report.actual_field_value_dominant_fraction = 0.01;

    let status = ruliad_capability_gate_status(&report, None, &gates);

    assert!(!status.passed);
    assert!(
        status
            .reasons
            .iter()
            .any(|reason| reason == "verifier_rate=0.000<=0"),
        "{:?}",
        status.reasons
    );
}

#[test]
fn ruliad_capability_gate_status_flags_field_value_collapse() {
    let mut gates = ruliad_degeneracy_gates();
    gates.capability_completion_health_min_rate = 0.0;
    gates.capability_answer_distinct_min_fraction = 0.20;
    gates.capability_field_value_distinct_ratio_min = 0.35;
    gates.capability_field_value_dominance_max = 0.85;
    let mut report = ruliad_eval_report(0.5, 0.5, 0.5, 0.5);
    report.actual_answer_distinct_fraction = 0.75;
    report.field_value_distinct_ratio = 0.10;
    report.actual_field_value_dominant_fraction = 0.95;

    let status = ruliad_capability_gate_status(&report, None, &gates);

    assert!(!status.passed);
    assert!(
        status
            .reasons
            .iter()
            .any(|reason| reason.starts_with("field_value_distinct_ratio=0.100<")),
        "{:?}",
        status.reasons
    );
    assert!(
        status
            .reasons
            .iter()
            .any(|reason| reason.starts_with("field_value_dominance=0.950>")),
        "{:?}",
        status.reasons
    );
    assert!(
        status
            .reasons
            .iter()
            .all(|reason| !reason.starts_with("answer_distinct=")),
        "{:?}",
        status.reasons
    );
}

#[test]
fn ruliad_capability_gate_does_not_flag_low_actual_diversity_when_targets_are_low_diversity() {
    let mut gates = ruliad_degeneracy_gates();
    gates.capability_completion_health_min_rate = 0.0;
    gates.capability_answer_distinct_min_fraction = 0.30;
    let mut report = ruliad_eval_report(0.0, 0.0, 0.0, 0.0);
    report.expected_answer_distinct_fraction = 0.05;
    report.actual_answer_distinct_fraction = 0.01;

    let status = ruliad_capability_gate_status(&report, None, &gates);

    assert!(
        status
            .reasons
            .iter()
            .all(|reason| !reason.starts_with("answer_distinct=")),
        "{:?}",
        status.reasons
    );
}

#[test]
fn ruliad_capability_gate_status_respects_disabled_gates() {
    let mut gates = ruliad_degeneracy_gates();
    gates.enabled = false;
    gates.capability_schema_wrong_max_rate = 0.0;
    let mut report = ruliad_eval_report(0.0, 0.0, 0.0, 0.0);
    report.schema_valid_wrong_count = report.item_count;

    let status = ruliad_capability_gate_status(&report, None, &gates);

    assert!(status.passed);
    assert!(status.reasons.is_empty());
}

#[test]
fn capability_run_control_warns_during_grace_then_recovers_after_first_pass_regression() {
    let dir = tempfile::tempdir().expect("tempdir");
    let run_dir = dir.path().join("run");
    let parallel_config = burn_dragon_train::ParallelConfig::default();
    let parallel_runtime =
        resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");
    let device = burn::tensor::Device::<TestBackend>::default();
    let valid_device = burn::tensor::Device::<TestValidBackend>::default();
    let mut training = tiny_training_hparams();
    training.events.flush_every_steps = 1;
    training.gates = burn_dragon_train::TrainingGatesConfig {
        capability_grace_epochs: 3,
        capability_regression_patience_epochs: 2,
        capability_required_after_first_pass: true,
        capability_schema_wrong_max_rate: 0.25,
        ..ruliad_degeneracy_gates()
    };
    let model_config = tiny_model_config();
    let devices = vec![device];
    let env = TrainEnvironment {
        parallel_runtime: &parallel_runtime,
        parallel_config: &parallel_config,
        run_dir: &run_dir,
        run_name: "capability-run-control-smoke",
        backend_name: "cpu",
        training: &training,
        resume_checkpoint_epoch: None,
        model_config: &model_config,
        device: &device,
        devices: &devices,
        train_dataset: None,
        valid_dataset: None,
        train_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 0],
            [2, 4],
        )])),
        valid_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<
            TestValidBackend,
        >(
            &valid_device,
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 1, 2, 2, 3, 3, 0],
            [2, 4],
        )])),
        source_selection_dataset: None,
        summary_event_token_ids: None,
        neuron_scaling_slot: None,
        epochs: 1,
        total_steps: 1,
        valid_steps: 1,
    };
    let handles = crate::train::events::build_training_event_handles(
        env.run_name,
        &run_dir,
        1,
        &training,
        None,
        None,
        None,
    )
    .expect("event handles");
    let bus = handles.metric_logger.bus();
    let mut state = ContinualLearningStabilityState::default();
    let mut bad_report = ruliad_eval_report(0.25, 0.25, 0.25, 0.25);
    bad_report.schema_valid_wrong_count = 80;
    let good_report = ruliad_eval_report(0.25, 0.25, 0.25, 0.25);

    apply_continual_learning_stability_policy(
        &env,
        DynamicValidationReport {
            objective: crate::config::TrainingValidationObjective::FixedHoldout,
            loss: 1.0,
            ruliad_eval_report: Some(bad_report.clone()),
            ..Default::default()
        },
        1,
        0,
        &mut state,
        &bus,
    );
    apply_continual_learning_stability_policy(
        &env,
        DynamicValidationReport {
            loss: 0.9,
            ruliad_eval_report: Some(good_report),
            ..Default::default()
        },
        4,
        1,
        &mut state,
        &bus,
    );
    apply_continual_learning_stability_policy(
        &env,
        DynamicValidationReport {
            loss: 0.9,
            ruliad_eval_report: Some(bad_report.clone()),
            ..Default::default()
        },
        5,
        2,
        &mut state,
        &bus,
    );
    apply_continual_learning_stability_policy(
        &env,
        DynamicValidationReport {
            loss: 0.9,
            ruliad_eval_report: Some(bad_report),
            ..Default::default()
        },
        6,
        3,
        &mut state,
        &bus,
    );
    let _ = bus.flush();
    drop(handles);

    let events = read_training_events(&run_dir);
    assert!(events.iter().any(|event| {
        event.get("type").and_then(|value| value.as_str()) == Some("gate")
            && event.get("gate").and_then(|value| value.as_str())
                == Some("continual_learning_capability_gate_grace")
    }));
    assert!(events.iter().any(|event| {
        event.get("type").and_then(|value| value.as_str()) == Some("dynamics_control")
            && event.get("mode").and_then(|value| value.as_str()) == Some("validation_recovery")
    }));
    assert_eq!(state.first_capability_pass_epoch, Some(4));
    assert_eq!(state.consecutive_capability_gate_failures, 2);
}

#[test]
fn policy_promotion_floor_failure_never_requests_recovery_by_itself() {
    let dir = tempfile::tempdir().expect("tempdir");
    let run_dir = dir.path().join("run");
    let parallel_config = burn_dragon_train::ParallelConfig::default();
    let parallel_runtime =
        resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");
    let device = burn::tensor::Device::<TestBackend>::default();
    let valid_device = burn::tensor::Device::<TestValidBackend>::default();
    let mut training = policy_contract_training(
        crate::config::RuliadCheckpointCapabilityContract::ClosedLoopPolicy,
    );
    training.events.flush_every_steps = 1;
    training.gates.capability_grace_epochs = 0;
    training.gates.capability_regression_patience_epochs = 2;
    training.gates.capability_required_after_first_pass = true;
    let model_config = tiny_model_config();
    let devices = vec![device];
    let env = TrainEnvironment {
        parallel_runtime: &parallel_runtime,
        parallel_config: &parallel_config,
        run_dir: &run_dir,
        run_name: "policy-promotion-ineligible-smoke",
        backend_name: "cpu",
        training: &training,
        resume_checkpoint_epoch: None,
        model_config: &model_config,
        device: &device,
        devices: &devices,
        train_dataset: None,
        valid_dataset: None,
        train_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 0],
            [2, 4],
        )])),
        valid_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<
            TestValidBackend,
        >(
            &valid_device,
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 1, 2, 2, 3, 3, 0],
            [2, 4],
        )])),
        source_selection_dataset: None,
        summary_event_token_ids: None,
        neuron_scaling_slot: None,
        epochs: 2,
        total_steps: 2,
        valid_steps: 1,
    };
    let handles = crate::train::events::build_training_event_handles(
        env.run_name,
        &run_dir,
        1,
        &training,
        None,
        None,
        None,
    )
    .expect("event handles");
    let bus = handles.metric_logger.bus();
    let mut state = ContinualLearningStabilityState {
        best_valid_loss: Some(0.5),
        best_ruliad_policy_observed_competence: ruliad_policy_competence_key(
            &healthy_policy_rollout(10, 48),
        ),
        best_ruliad_policy_solve_observation: Some(BinomialObservation {
            successes: 10,
            trials: 32,
        }),
        best_ruliad_policy_goal_observation: Some(BinomialObservation {
            successes: 48,
            trials: 96,
        }),
        best_ruliad_policy_valid_action_observation: Some(BinomialObservation {
            successes: 128,
            trials: 128,
        }),
        first_capability_pass_epoch: Some(1),
        last_capability_pass_epoch: Some(1),
        ..Default::default()
    };
    apply_continual_learning_stability_policy(
        &env,
        DynamicValidationReport {
            objective: crate::config::TrainingValidationObjective::FixedHoldout,
            loss: 0.60,
            ruliad_policy_rollout: Some(healthy_policy_rollout(7, 38)),
            ..Default::default()
        },
        2,
        1,
        &mut state,
        &bus,
    );
    apply_continual_learning_stability_policy(
        &env,
        DynamicValidationReport {
            objective: crate::config::TrainingValidationObjective::FixedHoldout,
            loss: 0.65,
            ruliad_policy_rollout: Some(healthy_policy_rollout(7, 38)),
            ..Default::default()
        },
        3,
        2,
        &mut state,
        &bus,
    );
    assert_eq!(state.consecutive_validation_regressions, 0);
    assert_eq!(state.consecutive_ruliad_correctness_regressions, 0);
    for (epoch, loss) in [(4, 0.49), (5, 0.48)] {
        apply_continual_learning_stability_policy(
            &env,
            DynamicValidationReport {
                objective: crate::config::TrainingValidationObjective::FixedHoldout,
                loss,
                ruliad_policy_rollout: Some(healthy_policy_rollout(0, 0)),
                ..Default::default()
            },
            epoch,
            epoch - 1,
            &mut state,
            &bus,
        );
    }
    let _ = bus.flush();
    drop(handles);

    assert_eq!(state.consecutive_capability_gate_failures, 4);
    assert_eq!(state.consecutive_validation_regressions, 0);
    assert_eq!(state.consecutive_ruliad_correctness_regressions, 2);
    let events = read_training_events(&run_dir);
    assert!(events.iter().any(|event| {
        event.get("type").and_then(|value| value.as_str()) == Some("gate")
            && event.get("gate").and_then(|value| value.as_str())
                == Some("continual_learning_checkpoint_promotion_ineligible")
    }));
    assert!(events.iter().any(|event| {
        event.get("type").and_then(|value| value.as_str()) == Some("gate")
            && event.get("gate").and_then(|value| value.as_str())
                == Some("continual_learning_validation_regression_suppressed_by_ruliad_progress")
            && event
                .get("message")
                .and_then(|value| value.as_str())
                .is_some_and(|message| message.contains("no statistically supported regression"))
    }));
    assert!(events.iter().all(|event| {
        event.get("type").and_then(|value| value.as_str()) != Some("dynamics_control")
            || event.get("epoch").and_then(|value| value.as_u64()) == Some(5)
    }));
    assert!(events.iter().any(|event| {
        event.get("type").and_then(|value| value.as_str()) == Some("gate")
            && event.get("gate").and_then(|value| value.as_str())
                == Some("continual_learning_ruliad_capability_regression")
            && event.get("epoch").and_then(|value| value.as_u64()) == Some(5)
    }));
}

#[test]
fn capability_quality_collapse_requests_source_capability_recovery_during_grace() {
    let dir = tempfile::tempdir().expect("tempdir");
    let run_dir = dir.path().join("run");
    let parallel_config = burn_dragon_train::ParallelConfig::default();
    let parallel_runtime =
        resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");
    let device = burn::tensor::Device::<TestBackend>::default();
    let valid_device = burn::tensor::Device::<TestValidBackend>::default();
    let mut training = tiny_training_hparams();
    training.events.flush_every_steps = 1;
    training.gates = burn_dragon_train::TrainingGatesConfig {
        capability_grace_epochs: 3,
        capability_completion_health_min_rate: 0.80,
        capability_distinct_2_min_fraction: 0.30,
        ..ruliad_degeneracy_gates()
    };
    let model_config = tiny_model_config();
    let devices = vec![device];
    let env = TrainEnvironment {
        parallel_runtime: &parallel_runtime,
        parallel_config: &parallel_config,
        run_dir: &run_dir,
        run_name: "capability-quality-collapse-recovery-smoke",
        backend_name: "cpu",
        training: &training,
        resume_checkpoint_epoch: None,
        model_config: &model_config,
        device: &device,
        devices: &devices,
        train_dataset: None,
        valid_dataset: None,
        train_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 0],
            [2, 4],
        )])),
        valid_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<
            TestValidBackend,
        >(
            &valid_device,
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 1, 2, 2, 3, 3, 0],
            [2, 4],
        )])),
        source_selection_dataset: None,
        summary_event_token_ids: None,
        neuron_scaling_slot: None,
        epochs: 1,
        total_steps: 1,
        valid_steps: 1,
    };
    let handles = crate::train::events::build_training_event_handles(
        env.run_name,
        &run_dir,
        1,
        &training,
        None,
        None,
        None,
    )
    .expect("event handles");
    let bus = handles.metric_logger.bus();
    let mut state = ContinualLearningStabilityState::default();
    let mut collapsed_report = ruliad_eval_report(0.0, 0.0, 0.0, 0.0);
    collapsed_report.mean_completion_quality = 0.0;
    collapsed_report.actual_answer_distinct_fraction = 0.01;

    apply_continual_learning_stability_policy(
        &env,
        DynamicValidationReport {
            objective: crate::config::TrainingValidationObjective::FixedHoldout,
            loss: 1.0,
            source_weighted_loss: None,
            stream_warm_loss: None,
            output_degeneracy: None,
            ruliad_eval_report: Some(collapsed_report),
            ruliad_policy_rollout: None,
        },
        1,
        0,
        &mut state,
        &bus,
    );
    let _ = bus.flush();
    drop(handles);

    let events = read_training_events(&run_dir);
    assert!(events.iter().any(|event| {
        event.get("type").and_then(|value| value.as_str()) == Some("gate")
            && event.get("gate").and_then(|value| value.as_str())
                == Some("continual_learning_capability_quality_recovery")
    }));
    assert!(events.iter().any(|event| {
        event.get("type").and_then(|value| value.as_str()) == Some("gate")
            && event.get("gate").and_then(|value| value.as_str())
                == Some("continual_learning_capability_gate_grace")
    }));
    let control = events
        .iter()
        .find(|event| {
            event.get("type").and_then(|value| value.as_str()) == Some("dynamics_control")
                && event.get("mode").and_then(|value| value.as_str())
                    == Some("source_capability_recovery")
        })
        .expect("quality collapse should request dynamics control");
    assert!(
        control
            .get("reason")
            .and_then(|value| value.as_str())
            .is_some_and(|reason| reason.contains("completion quality/diversity collapsed")),
        "{control:?}"
    );
    assert!(
        events.iter().all(|event| {
            event.get("type").and_then(|value| value.as_str()) != Some("dynamics_control")
                || event.get("mode").and_then(|value| value.as_str()) != Some("stable")
        }),
        "free-run capability collapse must not emit a contradictory stable control"
    );
    assert!(state.first_capability_pass_epoch.is_none());
}

#[test]
fn capability_field_value_collapse_requests_source_capability_recovery_during_grace() {
    let dir = tempfile::tempdir().expect("tempdir");
    let run_dir = dir.path().join("run");
    let parallel_config = burn_dragon_train::ParallelConfig::default();
    let parallel_runtime =
        resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");
    let device = burn::tensor::Device::<TestBackend>::default();
    let valid_device = burn::tensor::Device::<TestValidBackend>::default();
    let mut training = tiny_training_hparams();
    training.events.flush_every_steps = 1;
    training.gates = burn_dragon_train::TrainingGatesConfig {
        capability_grace_epochs: 3,
        capability_completion_health_min_rate: 0.10,
        capability_answer_distinct_min_fraction: 0.20,
        capability_field_value_distinct_ratio_min: 0.35,
        capability_field_value_dominance_max: 0.85,
        ..ruliad_degeneracy_gates()
    };
    let model_config = tiny_model_config();
    let devices = vec![device];
    let env = TrainEnvironment {
        parallel_runtime: &parallel_runtime,
        parallel_config: &parallel_config,
        run_dir: &run_dir,
        run_name: "capability-field-collapse-recovery-smoke",
        backend_name: "cpu",
        training: &training,
        resume_checkpoint_epoch: None,
        model_config: &model_config,
        device: &device,
        devices: &devices,
        train_dataset: None,
        valid_dataset: None,
        train_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 0],
            [2, 4],
        )])),
        valid_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<
            TestValidBackend,
        >(
            &valid_device,
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 1, 2, 2, 3, 3, 0],
            [2, 4],
        )])),
        source_selection_dataset: None,
        summary_event_token_ids: None,
        neuron_scaling_slot: None,
        epochs: 1,
        total_steps: 1,
        valid_steps: 1,
    };
    let handles = crate::train::events::build_training_event_handles(
        env.run_name,
        &run_dir,
        1,
        &training,
        None,
        None,
        None,
    )
    .expect("event handles");
    let bus = handles.metric_logger.bus();
    let mut state = ContinualLearningStabilityState::default();
    let mut collapsed_report = ruliad_eval_report(0.5, 0.5, 0.5, 0.5);
    collapsed_report.mean_completion_quality = 1.0;
    collapsed_report.actual_answer_distinct_fraction = 1.0;
    collapsed_report.field_value_distinct_ratio = 0.10;
    collapsed_report.actual_field_value_dominant_fraction = 0.95;

    apply_continual_learning_stability_policy(
        &env,
        DynamicValidationReport {
            objective: crate::config::TrainingValidationObjective::FixedHoldout,
            loss: 1.0,
            source_weighted_loss: None,
            stream_warm_loss: None,
            output_degeneracy: None,
            ruliad_eval_report: Some(collapsed_report),
            ruliad_policy_rollout: None,
        },
        1,
        0,
        &mut state,
        &bus,
    );
    let _ = bus.flush();
    drop(handles);

    let events = read_training_events(&run_dir);
    assert!(events.iter().any(|event| {
        event.get("type").and_then(|value| value.as_str()) == Some("gate")
            && event.get("gate").and_then(|value| value.as_str())
                == Some("continual_learning_capability_quality_recovery")
            && event
                .get("message")
                .and_then(|value| value.as_str())
                .is_some_and(|message| message.contains("field_value_distinct_ratio=0.100<"))
    }));
    assert!(events.iter().any(|event| {
        event.get("type").and_then(|value| value.as_str()) == Some("dynamics_control")
            && event.get("mode").and_then(|value| value.as_str())
                == Some("source_capability_recovery")
            && event
                .get("reason")
                .and_then(|value| value.as_str())
                .is_some_and(|reason| reason.contains("field_value_dominance=0.950>"))
    }));
}

#[test]
fn ruliad_correctness_regression_rolls_back_to_promoted_checkpoint() {
    let dir = tempfile::tempdir().expect("tempdir");
    let run_dir = dir.path().join("run");
    let parallel_config = burn_dragon_train::ParallelConfig::default();
    let parallel_runtime =
        resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");
    let device = burn::tensor::Device::<TestBackend>::default();
    let valid_device = burn::tensor::Device::<TestValidBackend>::default();
    let mut training = tiny_training_hparams();
    training.events.flush_every_steps = 1;
    training.gates = burn_dragon_train::TrainingGatesConfig {
        capability_regression_patience_epochs: 1,
        ..ruliad_degeneracy_gates()
    };
    let model_config = tiny_model_config();
    let devices = vec![device];
    let env = TrainEnvironment {
        parallel_runtime: &parallel_runtime,
        parallel_config: &parallel_config,
        run_dir: &run_dir,
        run_name: "ruliad-regression-rollback-target-smoke",
        backend_name: "cpu",
        training: &training,
        resume_checkpoint_epoch: None,
        model_config: &model_config,
        device: &device,
        devices: &devices,
        train_dataset: None,
        valid_dataset: None,
        train_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 0],
            [2, 4],
        )])),
        valid_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<
            TestValidBackend,
        >(
            &valid_device,
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 1, 2, 2, 3, 3, 0],
            [2, 4],
        )])),
        source_selection_dataset: None,
        summary_event_token_ids: None,
        neuron_scaling_slot: None,
        epochs: 1,
        total_steps: 1,
        valid_steps: 1,
    };
    let handles = crate::train::events::build_training_event_handles(
        env.run_name,
        &run_dir,
        1,
        &training,
        None,
        None,
        None,
    )
    .expect("event handles");
    let bus = handles.metric_logger.bus();
    let mut report = ruliad_eval_report(0.1328125, 0.1328125, 0.21875, 0.0);
    report.item_count = 128;
    report.scored_count = 128;
    let mut state = ContinualLearningStabilityState {
        best_valid_loss: Some(0.397696),
        best_checkpoint_epoch: Some(4),
        best_ruliad_verifier_accuracy: Some(0.203125),
        best_ruliad_partial_progress: Some(0.3125),
        ..Default::default()
    };
    apply_continual_learning_stability_policy(
        &env,
        DynamicValidationReport {
            objective: crate::config::TrainingValidationObjective::FixedHoldout,
            loss: 0.357596,
            source_weighted_loss: None,
            stream_warm_loss: None,
            output_degeneracy: None,
            ruliad_eval_report: Some(report),
            ruliad_policy_rollout: None,
        },
        5,
        2559,
        &mut state,
        &bus,
    );
    let _ = bus.flush();
    drop(handles);

    let control = read_training_events(&run_dir)
        .into_iter()
        .rev()
        .find(|event| {
            event.get("type").and_then(|value| value.as_str()) == Some("dynamics_control")
        })
        .expect("dynamics control event");
    assert_eq!(
        control.get("mode").and_then(|value| value.as_str()),
        Some("rollback_recovery")
    );
    assert_eq!(
        control
            .get("rollback_to_epoch")
            .and_then(|value| value.as_u64()),
        Some(4)
    );
}

#[test]
fn ruliad_correctness_regression_uses_capability_checkpoint_after_patience() {
    let dir = tempfile::tempdir().expect("tempdir");
    let run_dir = dir.path().join("run");
    let parallel_config = burn_dragon_train::ParallelConfig::default();
    let parallel_runtime =
        resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");
    let device = burn::tensor::Device::<TestBackend>::default();
    let valid_device = burn::tensor::Device::<TestValidBackend>::default();
    let mut training = tiny_training_hparams();
    training.events.flush_every_steps = 1;
    training.gates = burn_dragon_train::TrainingGatesConfig {
        capability_regression_patience_epochs: 2,
        capability_completion_health_min_rate: 0.0,
        capability_distinct_2_min_fraction: 0.0,
        ..ruliad_degeneracy_gates()
    };
    let model_config = tiny_model_config();
    let devices = vec![device];
    let env = TrainEnvironment {
        parallel_runtime: &parallel_runtime,
        parallel_config: &parallel_config,
        run_dir: &run_dir,
        run_name: "ruliad-regression-capability-checkpoint-smoke",
        backend_name: "cpu",
        training: &training,
        resume_checkpoint_epoch: None,
        model_config: &model_config,
        device: &device,
        devices: &devices,
        train_dataset: None,
        valid_dataset: None,
        train_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 0],
            [2, 4],
        )])),
        valid_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<
            TestValidBackend,
        >(
            &valid_device,
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 1, 2, 2, 3, 3, 0],
            [2, 4],
        )])),
        source_selection_dataset: None,
        summary_event_token_ids: None,
        neuron_scaling_slot: None,
        epochs: 1,
        total_steps: 1,
        valid_steps: 1,
    };
    let handles = crate::train::events::build_training_event_handles(
        env.run_name,
        &run_dir,
        1,
        &training,
        None,
        None,
        None,
    )
    .expect("event handles");
    let bus = handles.metric_logger.bus();
    let mut regressed_report = ruliad_eval_report(0.1328125, 0.1328125, 0.21875, 0.0);
    regressed_report.item_count = 128;
    regressed_report.scored_count = 128;
    let mut state = ContinualLearningStabilityState {
        best_valid_loss: Some(1.0),
        best_ruliad_recovery_competence: ruliad_competence_key(&ruliad_eval_report(
            0.203125, 0.203125, 0.3125, 0.0,
        )),
        best_ruliad_checkpoint_epoch: Some(3),
        best_ruliad_verifier_accuracy: Some(0.203125),
        best_ruliad_partial_progress: Some(0.3125),
        ..Default::default()
    };
    for (epoch, loss) in [(4, 0.90), (5, 0.80)] {
        apply_continual_learning_stability_policy(
            &env,
            DynamicValidationReport {
                objective: crate::config::TrainingValidationObjective::FixedHoldout,
                loss,
                source_weighted_loss: None,
                stream_warm_loss: None,
                output_degeneracy: None,
                ruliad_eval_report: Some(regressed_report.clone()),
                ruliad_policy_rollout: None,
            },
            epoch,
            epoch.saturating_mul(10),
            &mut state,
            &bus,
        );
    }
    let _ = bus.flush();
    drop(handles);

    assert_eq!(state.consecutive_ruliad_correctness_regressions, 2);
    let controls = read_training_events(&run_dir)
        .into_iter()
        .filter(|event| {
            event.get("type").and_then(|value| value.as_str()) == Some("dynamics_control")
                && event.get("mode").and_then(|value| value.as_str()) == Some("rollback_recovery")
        })
        .collect::<Vec<_>>();
    assert_eq!(controls.len(), 1, "{controls:?}");
    assert_eq!(
        controls[0]
            .get("rollback_to_epoch")
            .and_then(|value| value.as_u64()),
        Some(3)
    );
}

#[test]
fn output_degeneracy_policy_warns_on_low_confidence_argmax_loop() {
    let gates = ruliad_degeneracy_gates();
    let stats = degeneracy_stats(11.0, 0.03, 0.03, 0.94);

    assert!(uncertain_argmax_loop(&gates, &stats));
    assert!(output_degeneracy_tripped(&gates, &stats));
    assert!(!hard_output_collapse_for_gates(&gates, &stats));
}

#[test]
fn quiet_progress_renderer_is_default_for_training_efficiency() {
    assert!(quiet_progress_renderer_enabled_for("quiet"));
    assert!(quiet_progress_renderer_enabled_for("off"));
    assert!(quiet_progress_renderer_enabled_for(""));
    assert!(!quiet_progress_renderer_enabled_for("progress"));
    assert!(!quiet_progress_renderer_enabled_for("default"));
}

#[test]
fn output_degeneracy_policy_keeps_low_diversity_nonperiodic_output_soft() {
    let gates = ruliad_degeneracy_gates();
    let stats = degeneracy_stats(3.5, 0.52, 0.05, 0.04);

    assert!(!uncertain_argmax_loop(&gates, &stats));
    assert!(output_degeneracy_tripped(&gates, &stats));
    assert!(!hard_output_collapse_for_gates(&gates, &stats));
}

#[test]
fn output_degeneracy_policy_accepts_structured_low_entropy_output() {
    let gates = ruliad_degeneracy_gates();
    let mut stats = degeneracy_stats(1.35, 0.78, 0.38, 0.01);
    stats.argmax_unique_fraction = 0.13;
    stats.distinct_1_fraction = 0.13;
    stats.period_2_fraction = 0.02;
    stats.period_3_fraction = 0.24;
    stats.max_period_2_to_16_fraction = 0.36;
    stats.max_period_2_to_64_fraction = 0.36;
    stats.dominant_period_2_to_64 = 6;

    assert!(!output_degeneracy_tripped(&gates, &stats));
    assert!(!hard_output_collapse_for_gates(&gates, &stats));
}

#[test]
fn output_degeneracy_policy_ignores_periodic_but_diverse_structure() {
    let mut gates = ruliad_degeneracy_gates();
    gates.degeneracy_period_2_to_16_max_fraction = 0.40;
    gates.degeneracy_period_2_to_64_max_fraction = 0.40;
    let mut stats = degeneracy_stats(3.99, 0.07, 0.60, 0.20);
    stats.argmax_unique_fraction = 0.50;
    stats.distinct_1_fraction = 0.50;
    stats.period_2_fraction = 0.05;
    stats.period_3_fraction = 0.03;
    stats.max_period_2_to_16_fraction = 0.46;
    stats.max_period_2_to_64_fraction = 0.46;
    stats.dominant_period_2_to_64 = 11;

    assert!(!uncertain_argmax_loop(&gates, &stats));
    assert!(!output_degeneracy_tripped(&gates, &stats));
    assert!(!hard_output_collapse_for_gates(&gates, &stats));
}

#[test]
fn output_degeneracy_policy_flags_short_period_argmax_loop() {
    let mut gates = ruliad_degeneracy_gates();
    gates.degeneracy_period_2_to_16_max_fraction = 0.50;
    gates.degeneracy_period_2_to_64_max_fraction = 0.50;
    let mut stats = degeneracy_stats(3.20, 0.61, 0.55, 0.08);
    stats.argmax_unique_fraction = 0.45;
    stats.distinct_1_fraction = 0.45;
    stats.period_2_fraction = 0.04;
    stats.period_3_fraction = 0.05;
    stats.max_period_2_to_16_fraction = 0.58;
    stats.max_period_2_to_64_fraction = 0.58;
    stats.dominant_period_2_to_64 = 4;

    assert!(!uncertain_argmax_loop(&gates, &stats));
    assert!(output_degeneracy_tripped(&gates, &stats));
    assert!(!hard_output_collapse_for_gates(&gates, &stats));
}

#[test]
fn output_degeneracy_policy_keeps_low_alphabet_periodic_structure_soft() {
    let mut gates = ruliad_degeneracy_gates();
    gates.degeneracy_entropy_min_bits = 1.35;
    gates.degeneracy_max_probability_max = 0.82;
    gates.degeneracy_argmax_unique_min_fraction = 0.20;
    gates.degeneracy_distinct_2_min_fraction = 0.35;
    gates.degeneracy_repetition_max_fraction = 0.45;
    gates.degeneracy_period_2_max_fraction = 0.35;
    gates.degeneracy_period_3_max_fraction = 0.40;
    gates.degeneracy_period_2_to_16_max_fraction = 0.50;
    gates.degeneracy_period_2_to_64_max_fraction = 0.50;
    let mut stats = degeneracy_stats(2.293, 0.617, 0.319, 0.010);
    stats.argmax_unique_fraction = 0.172;
    stats.distinct_1_fraction = 0.172;
    stats.period_2_fraction = 0.0;
    stats.period_3_fraction = 0.005;
    stats.max_period_2_to_16_fraction = 0.573;
    stats.max_period_2_to_64_fraction = 0.573;
    stats.dominant_period_2_to_64 = 14;

    assert!(!uncertain_argmax_loop(&gates, &stats));
    assert!(output_degeneracy_tripped(&gates, &stats));
    assert!(!hard_output_collapse_for_gates(&gates, &stats));
}

#[test]
fn continual_learning_output_degeneracy_defers_recovery_to_ecs_without_rollback() {
    let dir = tempfile::tempdir().expect("tempdir");
    let run_dir = dir.path().join("run");
    let parallel_config = burn_dragon_train::ParallelConfig::default();
    let parallel_runtime =
        resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");
    let device = burn::tensor::Device::<TestBackend>::default();
    let valid_device = burn::tensor::Device::<TestValidBackend>::default();
    let mut training = tiny_training_hparams();
    training.events.flush_every_steps = 1;
    training.gates = ruliad_degeneracy_gates();
    let model_config = tiny_model_config();
    let devices = vec![device];
    let env = TrainEnvironment {
        parallel_runtime: &parallel_runtime,
        parallel_config: &parallel_config,
        run_dir: &run_dir,
        run_name: "output-degeneracy-ecs-recovery-smoke",
        backend_name: "cpu",
        training: &training,
        resume_checkpoint_epoch: None,
        model_config: &model_config,
        device: &device,
        devices: &devices,
        train_dataset: None,
        valid_dataset: None,
        train_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 0],
            [2, 4],
        )])),
        valid_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<
            TestValidBackend,
        >(
            &valid_device,
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 1, 2, 2, 3, 3, 0],
            [2, 4],
        )])),
        source_selection_dataset: None,
        summary_event_token_ids: None,
        neuron_scaling_slot: None,
        epochs: 1,
        total_steps: 1,
        valid_steps: 1,
    };
    let handles = crate::train::events::build_training_event_handles(
        env.run_name,
        &run_dir,
        1,
        &training,
        None,
        None,
        None,
    )
    .expect("event handles");
    let bus = handles.metric_logger.bus();
    let mut state = ContinualLearningStabilityState::default();
    apply_continual_learning_stability_policy(
        &env,
        DynamicValidationReport {
            objective: crate::config::TrainingValidationObjective::FixedHoldout,
            loss: 1.0,
            source_weighted_loss: None,
            stream_warm_loss: None,
            output_degeneracy: Some(degeneracy_stats(0.1, 0.99, 0.0, 1.0)),
            ruliad_eval_report: None,
            ruliad_policy_rollout: None,
        },
        1,
        0,
        &mut state,
        &bus,
    );
    let _ = bus.flush();
    drop(handles);

    let gate = read_training_events(&run_dir)
        .into_iter()
        .find(|event| {
            event.get("type").and_then(|value| value.as_str()) == Some("gate")
                && event.get("gate").and_then(|value| value.as_str())
                    == Some("continual_learning_output_degeneracy")
        })
        .expect("output degeneracy gate event");

    assert_eq!(
        gate.get("action").and_then(|value| value.as_str()),
        Some("alert")
    );
    assert_eq!(
        gate.get("severity").and_then(|value| value.as_str()),
        Some("warning")
    );
    let events = read_training_events(&run_dir);
    assert!(
        events.iter().all(|event| {
            event.get("type").and_then(|value| value.as_str()) != Some("dynamics_control")
        }),
        "output degeneracy recovery is emitted by the ECS dynamics plugin from the output-degeneracy sample, not duplicated by Dragon post-validation policy unless Dragon can add a rollback target"
    );
}

#[test]
fn disabled_dynamics_policy_emits_gate_without_recovery_control() {
    let dir = tempfile::tempdir().expect("tempdir");
    let run_dir = dir.path().join("run");
    let parallel_config = burn_dragon_train::ParallelConfig::default();
    let parallel_runtime =
        resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");
    let device = burn::tensor::Device::<TestBackend>::default();
    let valid_device = burn::tensor::Device::<TestValidBackend>::default();
    let mut training = tiny_training_hparams();
    training.events.flush_every_steps = 1;
    training.gates = ruliad_degeneracy_gates();
    training.dynamics.enabled = false;
    let model_config = tiny_model_config();
    let devices = vec![device];
    let env = TrainEnvironment {
        parallel_runtime: &parallel_runtime,
        parallel_config: &parallel_config,
        run_dir: &run_dir,
        run_name: "disabled-dynamics-no-recovery-control",
        backend_name: "cpu",
        training: &training,
        resume_checkpoint_epoch: None,
        model_config: &model_config,
        device: &device,
        devices: &devices,
        train_dataset: None,
        valid_dataset: None,
        train_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 0],
            [2, 4],
        )])),
        valid_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<
            TestValidBackend,
        >(
            &valid_device,
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 1, 2, 2, 3, 3, 0],
            [2, 4],
        )])),
        source_selection_dataset: None,
        summary_event_token_ids: None,
        neuron_scaling_slot: None,
        epochs: 1,
        total_steps: 1,
        valid_steps: 1,
    };
    let handles = crate::train::events::build_training_event_handles(
        env.run_name,
        &run_dir,
        1,
        &training,
        None,
        None,
        None,
    )
    .expect("event handles");
    let bus = handles.metric_logger.bus();
    let mut state = ContinualLearningStabilityState::default();
    apply_continual_learning_stability_policy(
        &env,
        DynamicValidationReport {
            objective: crate::config::TrainingValidationObjective::FixedHoldout,
            loss: 1.0,
            source_weighted_loss: None,
            stream_warm_loss: None,
            output_degeneracy: Some(degeneracy_stats(3.5, 0.52, 0.05, 0.04)),
            ruliad_eval_report: None,
            ruliad_policy_rollout: None,
        },
        1,
        0,
        &mut state,
        &bus,
    );
    let _ = bus.flush();
    drop(handles);

    let events = read_training_events(&run_dir);
    assert!(
        events.iter().any(|event| {
            event.get("type").and_then(|value| value.as_str()) == Some("gate")
                && event.get("gate").and_then(|value| value.as_str())
                    == Some("continual_learning_output_degeneracy")
        }),
        "degeneracy gate should still be visible when dynamics controls are disabled"
    );
    assert!(
        events.iter().all(|event| {
            event.get("type").and_then(|value| value.as_str()) != Some("dynamics_control")
        }),
        "disabled dynamics must not emit recovery controls"
    );
}

#[test]
fn ruliad_prompt_answer_keys_parse_answer_contract_line() {
    let prompt = "[R2 h..]\n?:eca^4\nA:xlen,xalpha,xcounts,xedge\n>trace\n!:";
    assert_eq!(
        ruliad_prompt_answer_keys(prompt),
        Some(vec![
            "xlen".to_string(),
            "xalpha".to_string(),
            "xcounts".to_string(),
            "xedge".to_string()
        ])
    );
}

#[test]
fn ruliad_correctness_metrics_emit_verifier_rates() {
    let dir = tempfile::tempdir().expect("tempdir");
    let run_dir = dir.path().join("run");
    let mut training = tiny_training_hparams();
    training.events.flush_every_steps = 1;
    let handles = crate::train::events::build_training_event_handles(
        "ruliad-correctness-metric-smoke",
        &run_dir,
        1,
        &training,
        None,
        None,
        None,
    )
    .expect("event handles");
    let bus = handles.metric_logger.bus();
    let report = burn_dragon_universality::RuliadEvalReport {
        version: burn_dragon_universality::ruliad::RULIAD_EVAL_REPORT_VERSION,
        reasoning_score_version: burn_dragon_universality::ruliad::RULIAD_REASONING_SCORE_VERSION,
        dataset_name: "test".to_string(),
        item_count: 4,
        scored_count: 4,
        exact_match_count: 1,
        semantic_match_count: 2,
        verifier_match_count: 2,
        partial_credit_count: 3,
        schema_valid_wrong_count: 1,
        malformed_completion_count: 1,
        missing_completion_count: 0,
        unexpected_completion_count: 0,
        exact_accuracy: 0.25,
        semantic_accuracy: 0.5,
        verifier_accuracy: 0.5,
        partial_credit_rate: 0.75,
        mean_partial_progress: 0.625,
        answer_field_correct_count: 5,
        answer_field_expected_count: 8,
        answer_field_accuracy: 0.625,
        answer_field_observed_count: 6,
        answer_field_coverage: 0.75,
        answer_terminated_count: 3,
        answer_termination_rate: 0.75,
        mean_completion_quality: 1.0,
        expected_answer_distinct_fraction: 1.0,
        actual_answer_distinct_fraction: 1.0,
        actual_answer_dominant_fraction: 0.25,
        expected_field_value_distinct_fraction: 1.0,
        actual_field_value_distinct_fraction: 1.0,
        field_value_distinct_ratio: 1.0,
        actual_field_value_dominant_fraction: 0.25,
        presented_action_expected_count: 4,
        presented_action_match_count: 2,
        presented_action_rate: 0.5,
        mean_certificate_prefix_coverage: 0.5,
        mean_completion_tokens: 12.0,
        canary_count: 0,
        canary_semantic_match_count: 0,
        family_scores: Vec::new(),
        task_scores: Vec::new(),
        difficulty_scores: vec![burn_dragon_universality::RuliadEvalGroupScore {
            label: "d7".to_string(),
            count: 4,
            exact_match_count: 1,
            semantic_match_count: 2,
            verifier_match_count: 2,
            partial_credit_count: 3,
            schema_valid_wrong_count: 1,
            malformed_completion_count: 1,
            missing_completion_count: 0,
            exact_accuracy: 0.25,
            semantic_accuracy: 0.5,
            verifier_accuracy: 0.5,
            partial_credit_rate: 0.75,
            mean_partial_progress: 0.625,
            answer_field_correct_count: 5,
            answer_field_expected_count: 8,
            answer_field_accuracy: 0.625,
            answer_field_observed_count: 6,
            answer_field_coverage: 0.75,
            answer_terminated_count: 3,
            answer_termination_rate: 0.75,
            mean_completion_quality: 1.0,
            expected_answer_distinct_fraction: 1.0,
            actual_answer_distinct_fraction: 1.0,
            actual_answer_dominant_fraction: 0.25,
            expected_field_value_distinct_fraction: 1.0,
            actual_field_value_distinct_fraction: 1.0,
            field_value_distinct_ratio: 1.0,
            actual_field_value_dominant_fraction: 0.25,
            presented_action_expected_count: 4,
            presented_action_match_count: 2,
            presented_action_rate: 0.5,
            formal_complexity: None,
        }],
        answer_contract_scores: Vec::new(),
        source_scores: Vec::new(),
        math_domain_scores: Vec::new(),
        reasoning_mode_scores: Vec::new(),
        failures: Vec::new(),
    };
    emit_ruliad_correctness_metrics("ruliad-correctness-metric-smoke", 3, 17, &report, &bus);
    let _ = bus.flush();
    drop(handles);

    let events = read_training_events(&run_dir);
    let metric_value = |name: &str| {
        events
            .iter()
            .find(|event| {
                event.get("type").and_then(|value| value.as_str()) == Some("metric")
                    && event.get("split").and_then(|value| value.as_str()) == Some("valid")
                    && event.get("name").and_then(|value| value.as_str()) == Some(name)
            })
            .and_then(|event| event.get("value"))
            .and_then(|value| value.as_f64())
            .unwrap_or_else(|| panic!("missing metric {name}"))
    };
    assert_eq!(metric_value("Ruliad Eval Items"), 4.0);
    assert_eq!(metric_value("Ruliad Verifier Accuracy"), 0.5);
    let competence_score = metric_value("Ruliad Competence Score");
    assert!(competence_score > 0.5);
    assert!(competence_score < 0.500001);
    assert_eq!(metric_value("Ruliad Competence Verifier PPM"), 500_000.0);
    assert_eq!(
        metric_value("Ruliad Expected Answer Distinct Fraction"),
        1.0
    );
    assert_eq!(metric_value("Ruliad Actual Answer Distinct Fraction"), 1.0);
    assert_eq!(metric_value("Ruliad Presented Action Rate"), 0.5);
    assert_eq!(metric_value("Ruliad Presented Action Items"), 4.0);
    assert_eq!(
        metric_value("Ruliad Competence Completion Health PPM"),
        375_000.0
    );
    assert_eq!(metric_value("Ruliad Answer Field Accuracy"), 0.625);
    assert_eq!(metric_value("Ruliad Answer Field Coverage"), 0.75);
    assert_eq!(metric_value("Ruliad Answer Termination Rate"), 0.75);
    assert_eq!(metric_value("Ruliad Malformed Completion Rate"), 0.25);
    let capability = events
        .iter()
        .find(|event| {
            event.get("type").and_then(|value| value.as_str()) == Some("capability_probe")
        })
        .expect("capability probe event");
    assert_eq!(
        capability
            .get("probe_name")
            .and_then(|value| value.as_str()),
        Some("ruliad_correctness")
    );
    assert_eq!(
        capability
            .get("achieved_difficulty_level")
            .and_then(|value| value.as_u64()),
        Some(7)
    );
    assert_eq!(
        capability
            .get("verifier_rate")
            .and_then(|value| value.as_f64()),
        Some(0.5)
    );
    assert_eq!(
        capability
            .get("answer_field_accuracy")
            .and_then(|value| value.as_f64()),
        Some(0.625)
    );
    assert_eq!(
        capability
            .get("answer_field_coverage")
            .and_then(|value| value.as_f64()),
        Some(0.75)
    );
    assert_eq!(
        capability
            .get("answer_termination_rate")
            .and_then(|value| value.as_f64()),
        Some(0.75)
    );
    assert_eq!(
        capability
            .get("expected_answer_distinct_fraction")
            .and_then(|value| value.as_f64()),
        Some(1.0)
    );
    assert_eq!(
        capability
            .get("actual_answer_distinct_fraction")
            .and_then(|value| value.as_f64()),
        Some(1.0)
    );
    assert_eq!(
        capability
            .get("actual_answer_dominant_fraction")
            .and_then(|value| value.as_f64()),
        Some(0.25)
    );
    assert_eq!(
        capability
            .get("field_value_distinct_ratio")
            .and_then(|value| value.as_f64()),
        Some(1.0)
    );
    assert_eq!(
        capability
            .get("field_value_dominant_fraction")
            .and_then(|value| value.as_f64()),
        Some(0.25)
    );
    let capability_jsonl = std::fs::read_to_string(run_dir.join("events/capability_probe.jsonl"))
        .expect("capability probe jsonl");
    assert!(capability_jsonl.contains("\"probe_name\":\"ruliad_correctness\""));
}

#[test]
fn ruliad_probe_examples_capture_mismatched_completion() {
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "hash-a".to_string(),
        sample_index: 7,
        split: burn_dragon_universality::SampleSplit::Validation,
        family: "proof_tree".to_string(),
        task_kind: "prove_theorem".to_string(),
        math_domains: vec!["category_theory".to_string()],
        reasoning_modes: vec!["equational_reasoning".to_string()],
        prompt: "[R2 hash-a v1 P/thm/proof]\nA:ok,l,r\n!:".to_string(),
        expected_answer: "ok=1;l=2;r=2".to_string(),
        difficulty_level: Some(3),
        spec: None,
    };
    let completion = burn_dragon_universality::RuliadCompletionRecord {
        oracle_hash: "hash-a".to_string(),
        completion: "!:ok=0;l=2;r=9\n[/R2]\n".to_string(),
    };

    let examples = ruliad_probe_examples(&[item], &[completion], 4);

    assert_eq!(examples.len(), 1);
    assert_eq!(examples[0].label, "proof_tree:prove_theorem");
    assert_eq!(examples[0].expected, "ok=1;l=2;r=2");
    assert_eq!(examples[0].actual.as_deref(), Some("ok=0;l=2;r=9"));
    assert_eq!(examples[0].status, "Partial");
    assert_eq!(examples[0].reason, "answer_mismatch");
    assert!(examples[0].prompt.contains("\\nA:ok,l,r\\n!:"));
    assert_eq!(examples[0].generated_tokens, 1);
}

#[test]
fn ruliad_completion_probe_records_write_raw_jsonl() {
    let dir = tempfile::tempdir().expect("tempdir");
    let run_dir = dir.path().join("run");
    let item = burn_dragon_universality::RuliadEvalItem {
        oracle_hash: "hash-a".to_string(),
        sample_index: 7,
        split: burn_dragon_universality::SampleSplit::Validation,
        family: "proof_tree".to_string(),
        task_kind: "prove_theorem".to_string(),
        math_domains: vec!["category_theory".to_string()],
        reasoning_modes: vec!["equational_reasoning".to_string()],
        prompt: "[R2 hash-a v1 P/thm/proof]\nA:ok,l,r\n!:".to_string(),
        expected_answer: "ok=1;l=2;r=2".to_string(),
        difficulty_level: Some(3),
        spec: None,
    };
    let completion = burn_dragon_universality::RuliadCompletionRecord {
        oracle_hash: "hash-a".to_string(),
        completion: "!:ok=1;l=2;r=2\n[/R2]\n".to_string(),
    };

    write_ruliad_completion_probe_records(
        &run_dir,
        RuliadProbeIdentity {
            run_name: "raw-probe-test",
            epoch: 5,
            absolute_step: 128,
            probe_name: "ruliad_correctness",
        },
        &[item],
        &[completion],
        &[vec![10, 11, 99, 77]],
        &[RuliadProbeGenerationBudget {
            max_new_tokens: 8,
            minimum_answer_tokens: 6,
            budget_sufficient: true,
            generation_hit_budget: false,
        }],
        Some(99),
    )
    .expect("write records");

    let path = run_dir.join("events/ruliad_completion_samples.jsonl");
    let contents = std::fs::read_to_string(path).expect("raw completion jsonl");
    let records = contents.lines().collect::<Vec<_>>();
    assert_eq!(records.len(), 1);
    let record: serde_json::Value = serde_json::from_str(records[0]).expect("valid sample json");
    assert_eq!(
        record.get("run_id").and_then(|value| value.as_str()),
        Some("raw-probe-test")
    );
    assert_eq!(
        record.get("probe_name").and_then(|value| value.as_str()),
        Some("ruliad_correctness")
    );
    assert_eq!(
        record.get("sample_index").and_then(|value| value.as_u64()),
        Some(7)
    );
    assert_eq!(
        record.get("epoch").and_then(|value| value.as_u64()),
        Some(5)
    );
    assert_eq!(
        record.get("absolute_step").and_then(|value| value.as_u64()),
        Some(128)
    );
    assert_eq!(
        record.get("version").and_then(|value| value.as_u64()),
        Some(3)
    );
    assert_eq!(
        record
            .get("generated_model_token_count")
            .and_then(|value| value.as_u64()),
        Some(3)
    );
    assert_eq!(
        record
            .get("generation_budget")
            .and_then(|value| value.as_u64()),
        Some(8)
    );
    assert_eq!(
        record
            .get("minimum_answer_tokens")
            .and_then(|value| value.as_u64()),
        Some(6)
    );
    assert_eq!(
        record
            .get("budget_sufficient")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        record
            .get("expected_answer")
            .and_then(|value| value.as_str()),
        Some("ok=1;l=2;r=2")
    );
    assert_eq!(
        record.get("actual_answer").and_then(|value| value.as_str()),
        Some("ok=1;l=2;r=2")
    );
    assert_eq!(
        record.get("status").and_then(|value| value.as_str()),
        Some("SemanticMatch")
    );
    assert_eq!(
        record
            .get("verifier_match")
            .and_then(|value| value.as_bool()),
        Some(false)
    );
    assert_eq!(
        record
            .get("semantic_match")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        record.get("completion").and_then(|value| value.as_str()),
        Some("!:ok=1;l=2;r=2\n[/R2]\n")
    );
}

#[test]
fn ruliad_completion_close_marker_uses_expected_dialect() {
    assert_eq!(
        canonicalize_ruliad_completion_close_marker(
            "!:certificate=x\n[/R2]\n".to_string(),
            "[/R3]",
        ),
        "!:certificate=x\n[/R3]\n"
    );
    assert_eq!(
        canonicalize_ruliad_completion_close_marker("!:ok=1\n[/R3]\n".to_string(), "[/R2]",),
        "!:ok=1\n[/R2]\n"
    );
}

#[test]
fn epoch_end_absolute_step_uses_completed_steps_for_partial_epochs() {
    assert_eq!(epoch_end_absolute_step(1, 256, 256), 255);
    assert_eq!(epoch_end_absolute_step(2, 256, 128), 383);
    assert_eq!(epoch_end_absolute_step(0, 256, 0), 0);
}

#[test]
fn ruliad_completion_degeneracy_summary_tracks_periodic_answers() {
    let summary =
        ruliad_completion_degeneracy_summary(&[vec![1, 2, 1, 2, 1, 2], vec![3, 4, 5, 6]], None)
            .expect("summary");

    assert_eq!(summary.sequence_count, 2);
    assert_eq!(summary.token_count, 10);
    assert!(summary.distinct_2_fraction < 1.0);
    assert_eq!(summary.dominant_period_2_to_64, 2);
    assert!(
        summary.max_period_2_to_64_fraction > 0.5,
        "{}",
        summary.max_period_2_to_64_fraction
    );
}

#[test]
fn ruliad_completion_degeneracy_summary_trims_after_close_token() {
    let summary = ruliad_completion_degeneracy_summary(
        &[vec![10, 11, 99, 7, 7, 7, 7, 7], vec![12, 13, 99, 8, 8, 8]],
        Some(99),
    )
    .expect("summary");

    assert_eq!(summary.sequence_count, 2);
    assert_eq!(summary.token_count, 6);
    assert!(summary.repetition_fraction < 0.1, "{summary:?}");
    assert!(summary.max_period_2_to_64_fraction < 0.1, "{summary:?}");
}

#[test]
fn ruliad_capability_gate_metrics_emit_failure_state() {
    let dir = tempfile::tempdir().expect("tempdir");
    let run_dir = dir.path().join("run");
    let mut training = tiny_training_hparams();
    training.events.flush_every_steps = 1;
    training.gates.capability_schema_wrong_max_rate = 0.25;
    training.gates.capability_malformed_max_rate = 0.02;
    training.gates.capability_completion_health_min_rate = 0.80;
    training.gates.capability_output_entropy_min_bits = 1.25;
    training.gates.capability_distinct_2_min_fraction = 0.30;
    let handles = crate::train::events::build_training_event_handles(
        "ruliad-capability-gate-metric-smoke",
        &run_dir,
        1,
        &training,
        None,
        None,
        None,
    )
    .expect("event handles");
    let bus = handles.metric_logger.bus();
    let mut report = ruliad_eval_report(0.25, 0.25, 0.25, 0.25);
    report.schema_valid_wrong_count = 40;
    report.malformed_completion_count = 5;
    let stats = degeneracy_stats(0.5, 0.9, 0.1, 0.0);
    emit_ruliad_capability_gate_metrics(
        "ruliad-capability-gate-metric-smoke",
        &report,
        Some(&stats),
        &training.gates,
        true,
        TrainingEventContext {
            epoch: 4,
            absolute_step: 19,
            bus: &bus,
        },
    );
    let _ = bus.flush();
    drop(handles);

    let events = read_training_events(&run_dir);
    let metric_value = |name: &str| {
        events
            .iter()
            .find(|event| {
                event.get("type").and_then(|value| value.as_str()) == Some("metric")
                    && event.get("split").and_then(|value| value.as_str()) == Some("valid")
                    && event.get("name").and_then(|value| value.as_str()) == Some(name)
            })
            .and_then(|event| event.get("value"))
            .and_then(|value| value.as_f64())
            .unwrap_or_else(|| panic!("missing metric {name}"))
    };
    assert_eq!(metric_value("Ruliad Capability Gate Passed"), 0.0);
    assert!(metric_value("Ruliad Capability Gate Failure Count") >= 4.0);
    assert!(events.iter().any(|event| {
        event.get("type").and_then(|value| value.as_str()) == Some("gate")
            && event.get("gate").and_then(|value| value.as_str())
                == Some("ruliad_capability_gate_failed")
    }));
}

#[test]
fn reused_training_serialization_probe_emits_distinct_identity_without_generation() {
    let dir = tempfile::tempdir().expect("tempdir");
    let run_dir = dir.path().join("run");
    let mut training = tiny_training_hparams();
    training.events.flush_every_steps = 1;
    let handles = crate::train::events::build_training_event_handles(
        "ruliad-reused-probe-smoke",
        &run_dir,
        1,
        &training,
        None,
        None,
        None,
    )
    .expect("event handles");
    let bus = handles.metric_logger.bus();
    let report = ruliad_eval_report(0.25, 0.5, 0.5, 0.5);
    emit_reused_ruliad_correctness_validation(
        "ruliad-reused-probe-smoke",
        2,
        31,
        &report,
        None,
        &bus,
    );
    let _ = bus.flush();
    drop(handles);

    let events = read_training_events(&run_dir);
    assert!(events.iter().any(|event| {
        event.get("type").and_then(|value| value.as_str()) == Some("capability_probe")
            && event.get("probe_name").and_then(|value| value.as_str())
                == Some("ruliad_correctness_training_serialization")
            && event.get("verifier_rate").and_then(|value| value.as_f64()) == Some(0.25)
    }));
    assert!(events.iter().any(|event| {
        event.get("type").and_then(|value| value.as_str()) == Some("metric")
            && event.get("name").and_then(|value| value.as_str())
                == Some("Ruliad Training Serialization Probe Reused Canonical Evaluation")
            && event.get("value").and_then(|value| value.as_f64()) == Some(1.0)
    }));
}

#[test]
fn latent_eval_step_sweep_sorts_and_deduplicates_steps() {
    let mut training = tiny_training_hparams();
    training.latent_reasoning.eval_step_sweep = vec![8, 1, 4, 1, 2];

    assert_eq!(latent_eval_step_sweep(&training), vec![1, 2, 4, 8]);
    assert_eq!(
        latent_eval_step_sweep_excluding(&training, Some(4)),
        vec![1, 2, 8]
    );
}

#[test]
fn ruliad_policy_probes_have_independent_validation_cadences() {
    let mut training = tiny_training_hparams();
    training.ruliad_policy_probe.enabled = true;
    training.ruliad_policy_probe.every_epochs = 2;

    assert!(!ruliad_constrained_policy_probe_due(&training, 1));
    assert!(ruliad_constrained_policy_probe_due(&training, 2));
    assert!(!ruliad_constrained_policy_probe_due(&training, 3));
    assert!(ruliad_constrained_policy_probe_due(&training, 4));
    assert!(!ruliad_closed_loop_policy_probe_due(&training, 1));
    assert!(ruliad_closed_loop_policy_probe_due(&training, 2));

    training.ruliad_policy_probe.closed_loop_every_epochs = Some(4);
    assert!(ruliad_constrained_policy_probe_due(&training, 2));
    assert!(!ruliad_closed_loop_policy_probe_due(&training, 2));
    assert!(ruliad_constrained_policy_probe_due(&training, 4));
    assert!(ruliad_closed_loop_policy_probe_due(&training, 4));
}

#[test]
fn ruliad_probe_generation_parallelism_is_memory_bounded() {
    assert_eq!(ruliad_probe_generation_in_flight_rows(1, 16, 128), 1);
    assert_eq!(ruliad_probe_generation_in_flight_rows(3, 16, 128), 3);
    assert_eq!(ruliad_probe_generation_in_flight_rows(64, 16, 128), 16);
    assert_eq!(ruliad_probe_generation_in_flight_rows(64, 16, 2), 2);
    assert_eq!(ruliad_probe_generation_in_flight_rows(64, 4, 128), 4);
}

#[test]
fn ruliad_probe_generation_batches_nearby_positions_and_preserves_output_indices() {
    let work = ruliad_probe_generation_work(&[5, 3, 5, 5, 3, 7], true, 2, 2, 64);
    assert_eq!(
        work,
        vec![
            RuliadProbeGenerationWork {
                probe_indices: vec![1, 4],
                batched: true,
            },
            RuliadProbeGenerationWork {
                probe_indices: vec![0, 2],
                batched: true,
            },
            RuliadProbeGenerationWork {
                probe_indices: vec![3, 5],
                batched: true,
            },
        ]
    );
    let mut output_indices = work
        .iter()
        .flat_map(|item| item.probe_indices.iter().copied())
        .collect::<Vec<_>>();
    output_indices.sort_unstable();
    assert_eq!(output_indices, (0..6).collect::<Vec<_>>());
}

#[test]
fn ruliad_probe_generation_bounds_ragged_prompt_position_span() {
    let work = ruliad_probe_generation_work(&[1, 2, 3, 70, 71, 200], true, 64, 2, 8);
    assert_eq!(work.len(), 3);
    assert_eq!(work[0].probe_indices, vec![0, 1, 2]);
    assert_eq!(work[1].probe_indices, vec![3, 4]);
    assert_eq!(work[2].probe_indices, vec![5]);
    assert!(!work[2].batched);
}

#[test]
fn ruliad_probe_generation_disable_keeps_every_row_independent() {
    let work = ruliad_probe_generation_work(&[3, 3, 3], false, 32, 2, 64);
    assert_eq!(work.len(), 3);
    assert!(work.iter().all(|item| !item.batched));
    assert_eq!(
        work.iter()
            .flat_map(|item| item.probe_indices.iter().copied())
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
}

#[test]
fn ruliad_probe_generation_waves_bound_total_live_rows() {
    let work = vec![
        RuliadProbeGenerationWork {
            probe_indices: vec![0, 1, 2],
            batched: true,
        },
        RuliadProbeGenerationWork {
            probe_indices: vec![3],
            batched: false,
        },
        RuliadProbeGenerationWork {
            probe_indices: vec![4, 5],
            batched: true,
        },
        RuliadProbeGenerationWork {
            probe_indices: vec![6],
            batched: false,
        },
    ];
    let waves = ruliad_probe_generation_waves(&work, 4);
    assert_eq!(waves.len(), 2);
    assert_eq!(
        waves
            .iter()
            .map(|wave| {
                wave.iter()
                    .map(|item| item.probe_indices.len())
                    .sum::<usize>()
            })
            .collect::<Vec<_>>(),
        vec![4, 3]
    );
}

#[test]
fn ruliad_correctness_eval_step_metrics_use_distinct_probe_name() {
    let dir = tempfile::tempdir().expect("tempdir");
    let run_dir = dir.path().join("run");
    let mut training = tiny_training_hparams();
    training.events.flush_every_steps = 1;
    let handles = crate::train::events::build_training_event_handles(
        "ruliad-eval-step-metric-smoke",
        &run_dir,
        1,
        &training,
        None,
        None,
        None,
    )
    .expect("event handles");
    let bus = handles.metric_logger.bus();
    let report = burn_dragon_universality::RuliadEvalReport {
        version: burn_dragon_universality::ruliad::RULIAD_EVAL_REPORT_VERSION,
        reasoning_score_version: burn_dragon_universality::ruliad::RULIAD_REASONING_SCORE_VERSION,
        dataset_name: "test".to_string(),
        item_count: 2,
        scored_count: 2,
        exact_match_count: 0,
        semantic_match_count: 1,
        verifier_match_count: 1,
        partial_credit_count: 1,
        schema_valid_wrong_count: 0,
        malformed_completion_count: 0,
        missing_completion_count: 0,
        unexpected_completion_count: 0,
        exact_accuracy: 0.0,
        semantic_accuracy: 0.5,
        verifier_accuracy: 0.5,
        partial_credit_rate: 0.5,
        mean_partial_progress: 0.5,
        answer_field_correct_count: 1,
        answer_field_expected_count: 2,
        answer_field_accuracy: 0.5,
        answer_field_observed_count: 1,
        answer_field_coverage: 0.5,
        answer_terminated_count: 2,
        answer_termination_rate: 1.0,
        mean_completion_quality: 1.0,
        expected_answer_distinct_fraction: 1.0,
        actual_answer_distinct_fraction: 1.0,
        actual_answer_dominant_fraction: 0.5,
        expected_field_value_distinct_fraction: 1.0,
        actual_field_value_distinct_fraction: 1.0,
        field_value_distinct_ratio: 1.0,
        actual_field_value_dominant_fraction: 0.5,
        presented_action_expected_count: 2,
        presented_action_match_count: 1,
        presented_action_rate: 0.5,
        mean_certificate_prefix_coverage: 0.5,
        mean_completion_tokens: 8.0,
        canary_count: 0,
        canary_semantic_match_count: 0,
        family_scores: Vec::new(),
        task_scores: Vec::new(),
        difficulty_scores: Vec::new(),
        answer_contract_scores: Vec::new(),
        source_scores: Vec::new(),
        math_domain_scores: Vec::new(),
        reasoning_mode_scores: Vec::new(),
        failures: Vec::new(),
    };
    emit_ruliad_correctness_metrics_with_labels(RuliadCorrectnessMetrics {
        identity: RuliadProbeIdentity {
            run_name: "ruliad-eval-step-metric-smoke",
            epoch: 2,
            absolute_step: 32,
            probe_name: "ruliad_correctness_eval_steps_8",
        },
        report: &report,
        bus: &bus,
        metric_prefix: Some("Ruliad Eval Steps 8"),
        output_degeneracy: None,
        examples: &[],
        schema_alignment: RuliadAnswerSchemaAlignmentSummary::default(),
        completion_degeneracy: None,
        generation_budget: None,
    });
    let _ = bus.flush();
    drop(handles);

    let events = read_training_events(&run_dir);
    assert!(events.iter().any(|event| {
        event.get("type").and_then(|value| value.as_str()) == Some("metric")
            && event.get("name").and_then(|value| value.as_str())
                == Some("Ruliad Eval Steps 8 Ruliad Verifier Accuracy")
    }));
    assert!(events.iter().any(|event| {
        event.get("type").and_then(|value| value.as_str()) == Some("capability_probe")
            && event.get("probe_name").and_then(|value| value.as_str())
                == Some("ruliad_correctness_eval_steps_8")
    }));
}

#[test]
fn ruliad_contract_probe_uses_tokenizer_valid_lowercase_alpha_values() {
    let field = RuliadAnswerFieldRange {
        key: "nfalpha".to_string(),
        value: "ABC".to_string(),
        start: 8,
        end: 11,
    };

    let allowed = ruliad_contract_value_allowed_chars(&field, 'A', &['A', 'B', 'C']);

    assert_eq!(allowed, vec!['a', 'b', 'c']);
}

#[test]
fn file_metric_best_strategy_tracks_best_value() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut strategy = FileMetricBestCheckpointingStrategy::new(
        dir.path(),
        &LossMetric::<TestValidBackend>::new(),
        burn_train::metric::store::Direction::Lowest,
        burn_train::metric::store::Split::Valid,
    );

    let previous_best = strategy.update_best_candidate(1, 3.5);

    assert_eq!(previous_best, None);
    assert_eq!(strategy.best_epoch, Some(1));
    assert_eq!(strategy.best_value, Some(3.5));
}

#[test]
fn file_metric_best_strategy_replaces_only_on_improvement() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut strategy = FileMetricBestCheckpointingStrategy::new(
        dir.path(),
        &LossMetric::<TestValidBackend>::new(),
        burn_train::metric::store::Direction::Lowest,
        burn_train::metric::store::Split::Valid,
    );
    strategy.best_epoch = Some(2);
    strategy.best_value = Some(3.2);

    let worse_previous_best = strategy.update_best_candidate(3, 3.3);
    assert_eq!(worse_previous_best, None);
    assert_eq!(strategy.best_epoch, Some(2));
    assert_eq!(strategy.best_value, Some(3.2));

    let better_previous_best = strategy.update_best_candidate(4, 3.1);
    assert_eq!(better_previous_best, Some(2));
    assert_eq!(strategy.best_epoch, Some(4));
    assert_eq!(strategy.best_value, Some(3.1));
}

fn write_metric_log(run_dir: &Path, split: &str, epoch: usize, values: &[f64]) {
    let epoch_dir = run_dir.join(split).join(format!("epoch-{epoch}"));
    fs::create_dir_all(&epoch_dir).expect("create epoch dir");
    let path = epoch_dir.join("Loss.log");
    let content = values
        .iter()
        .map(|value| format!("{value},1"))
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(path, content).expect("write metric log");
}

fn apply_checkpoint_actions(run_dir: &Path, epoch: usize, actions: &[CheckpointingAction]) {
    let checkpoint_dir = run_dir.join("checkpoint");
    fs::create_dir_all(&checkpoint_dir).expect("create checkpoint dir");
    for action in actions {
        match action {
            CheckpointingAction::Save => {
                for prefix in ["model", "optim", "scheduler"] {
                    fs::write(
                        checkpoint_dir.join(format!("{prefix}-{epoch}.bin")),
                        format!("{prefix}-{epoch}"),
                    )
                    .expect("write checkpoint file");
                }
            }
            CheckpointingAction::Delete(epoch) => {
                for prefix in ["model", "optim", "scheduler"] {
                    let path = checkpoint_dir.join(format!("{prefix}-{epoch}.bin"));
                    if path.exists() {
                        fs::remove_file(path).expect("remove checkpoint file");
                    }
                }
            }
        }
    }
}

fn retained_model_epochs(run_dir: &Path) -> Vec<usize> {
    let checkpoint_dir = run_dir.join("checkpoint");
    let mut epochs = fs::read_dir(&checkpoint_dir)
        .expect("read checkpoint dir")
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            let epoch = name
                .strip_prefix("model-")?
                .strip_suffix(".bin")?
                .parse::<usize>()
                .ok()?;
            Some(epoch)
        })
        .collect::<Vec<_>>();
    epochs.sort_unstable();
    epochs
}

fn write_dynamic_checkpoint_bundle(checkpoint_dir: &Path, epoch: usize) {
    for prefix in ["model", "optimizer", "scheduler"] {
        fs::write(
            checkpoint_dir.join(format!("{prefix}-{epoch}.bin")),
            format!("{prefix}-{epoch}"),
        )
        .expect("write dynamic checkpoint record");
    }
    for prefix in [
        "dynamics",
        "model-config",
        "stability",
        "training-ecs-state",
    ] {
        fs::write(
            checkpoint_dir.join(format!("{prefix}-{epoch}.json")),
            format!(r#"{{"epoch":{epoch}}}"#),
        )
        .expect("write dynamic checkpoint json");
    }
    fs::write(
        checkpoint_dir.join(format!("source-selection-state-{epoch}.json")),
        format!(r#"{{"epoch":{epoch}}}"#),
    )
    .expect("write source-selection state checkpoint");
}

fn append_validation_event(run_dir: &Path, epoch: usize, loss: f64) {
    let events_dir = run_dir.join("events");
    fs::create_dir_all(&events_dir).expect("create events dir");
    let path = events_dir.join("training_events.jsonl");
    let mut content = fs::read_to_string(&path).unwrap_or_default();
    content.push_str(&format!(
            r#"{{"type":"validation_finished","run_id":"test","epoch":{epoch},"absolute_step":{epoch},"loss":{loss}}}"#
        ));
    content.push('\n');
    fs::write(path, content).expect("append validation event");
}

#[test]
fn continual_learning_stability_checkpoint_roundtrips_and_exact_resume_fails_closed() {
    let directory = tempfile::tempdir().expect("temporary run directory");
    fs::create_dir_all(directory.path().join("checkpoint")).expect("checkpoint directory");
    let state = ContinualLearningStabilityState {
        best_valid_loss: Some(0.75),
        best_checkpoint_epoch: Some(3),
        best_ruliad_policy_competence: ruliad_policy_competence_key(&healthy_policy_rollout(
            24, 90,
        )),
        best_ruliad_policy_recovery_competence: ruliad_policy_competence_key(
            &healthy_policy_rollout(23, 88),
        ),
        best_ruliad_policy_observed_competence: ruliad_policy_competence_key(
            &healthy_policy_rollout(25, 92),
        ),
        best_ruliad_policy_solve_observation: Some(BinomialObservation {
            successes: 25,
            trials: 32,
        }),
        best_ruliad_policy_goal_observation: Some(BinomialObservation {
            successes: 92,
            trials: 96,
        }),
        best_ruliad_policy_valid_action_observation: Some(BinomialObservation {
            successes: 128,
            trials: 128,
        }),
        consecutive_validation_regressions: 2,
        consecutive_output_degeneracy: 1,
        ..ContinualLearningStabilityState::default()
    };
    save_continual_learning_stability_checkpoint(directory.path(), 4, &state)
        .expect("save stability state");

    assert_eq!(
        load_continual_learning_stability_checkpoint(directory.path(), 4, true)
            .expect("load stability state"),
        Some(state)
    );
    assert!(
        load_continual_learning_stability_checkpoint(directory.path(), 5, true)
            .expect_err("exact resume must require stability state")
            .to_string()
            .contains("exact resume requires continual-learning stability checkpoint")
    );
}

#[test]
fn continual_learning_stability_state_accepts_pre_observation_checkpoints() {
    let state: ContinualLearningStabilityState =
        serde_json::from_str(r#"{"best_valid_loss":0.75,"best_checkpoint_epoch":3}"#)
            .expect("older stability checkpoints should receive defaults for new evidence fields");

    assert_eq!(state.best_valid_loss, Some(0.75));
    assert_eq!(state.best_checkpoint_epoch, Some(3));
    assert_eq!(state.best_ruliad_policy_solve_observation, None);
    assert_eq!(state.best_ruliad_policy_goal_observation, None);
    assert_eq!(state.best_ruliad_policy_valid_action_observation, None);
}

#[test]
fn historical_best_validation_recovers_loss_and_available_checkpoint_epoch() {
    let dir = tempfile::tempdir().expect("tempdir");
    let checkpoint_dir = dir.path().join("checkpoint");
    fs::create_dir_all(&checkpoint_dir).expect("checkpoint dir");
    for epoch in [9, 10] {
        fs::write(
            checkpoint_dir.join(format!("model-{epoch}.bin")),
            format!("model-{epoch}"),
        )
        .expect("write checkpoint");
    }
    append_validation_event(dir.path(), 8, 0.789);
    append_validation_event(dir.path(), 9, 0.797);
    append_validation_event(dir.path(), 10, 0.821);
    append_validation_event(dir.path(), 11, 0.700);

    let historical = historical_best_validation(dir.path(), 10);

    assert_eq!(
        historical,
        HistoricalBestValidation {
            best_loss: Some(0.789),
            best_checkpoint_epoch: Some(9),
        }
    );
}

#[test]
fn historical_best_validation_keeps_true_best_checkpoint_when_present() {
    let dir = tempfile::tempdir().expect("tempdir");
    let checkpoint_dir = dir.path().join("checkpoint");
    fs::create_dir_all(&checkpoint_dir).expect("checkpoint dir");
    for epoch in [8, 9, 10] {
        fs::write(
            checkpoint_dir.join(format!("model-{epoch}.bin")),
            format!("model-{epoch}"),
        )
        .expect("write checkpoint");
    }
    append_validation_event(dir.path(), 8, 0.789);
    append_validation_event(dir.path(), 9, 0.797);
    append_validation_event(dir.path(), 10, 0.821);

    let historical = historical_best_validation(dir.path(), 10);

    assert_eq!(
        historical,
        HistoricalBestValidation {
            best_loss: Some(0.789),
            best_checkpoint_epoch: Some(8),
        }
    );
}

#[test]
fn file_metric_best_strategy_preserves_old_best_outside_keep_last_window() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut strategy = FileMetricBestCheckpointingStrategy::new(
        dir.path(),
        &LossMetric::<TestValidBackend>::new(),
        burn_train::metric::store::Direction::Lowest,
        burn_train::metric::store::Split::Valid,
    );

    let means = [
        2.0, 1.9, 1.8, 1.7, 1.6, 1.55, 1.53, 1.52, 1.515, 1.51, 1.509, 1.508, 1.507, 1.506, 1.505,
        1.504, 1.503, 1.502, 1.497, 1.501, 1.510, 1.512, 1.511, 1.499, 1.513, 1.514, 1.502, 1.520,
        1.506, 1.530,
    ];

    for (index, mean) in means.iter().enumerate() {
        let epoch = index + 1;
        write_metric_log(dir.path(), "valid", epoch, &[*mean]);
        let actions = strategy.actions_for_epoch(epoch);
        apply_checkpoint_actions(dir.path(), epoch, &actions);
    }

    assert_eq!(strategy.best_epoch, Some(19));
    assert_eq!(retained_model_epochs(dir.path()), vec![19, 29, 30]);
}

#[test]
fn dynamic_scheduler_checkpoint_pruning_keeps_recent_and_best() {
    let dir = tempfile::tempdir().expect("tempdir");
    let checkpoint_dir = dir.path().join("checkpoint");
    fs::create_dir_all(&checkpoint_dir).expect("checkpoint dir");
    for epoch in 1..=6 {
        write_dynamic_checkpoint_bundle(&checkpoint_dir, epoch);
    }

    prune_dragon_model_checkpoints(dir.path(), 6, &[Some(2), Some(3)]).expect("prune checkpoints");

    assert_eq!(retained_model_epochs(dir.path()), vec![2, 3, 5, 6]);
    for kept_epoch in [2, 3, 5, 6] {
        for file in [
            format!("model-{kept_epoch}.bin"),
            format!("optimizer-{kept_epoch}.bin"),
            format!("scheduler-{kept_epoch}.bin"),
            format!("dynamics-{kept_epoch}.json"),
            format!("model-config-{kept_epoch}.json"),
            format!("source-selection-state-{kept_epoch}.json"),
            format!("stability-{kept_epoch}.json"),
            format!("training-ecs-state-{kept_epoch}.json"),
        ] {
            assert!(
                checkpoint_dir.join(file).is_file(),
                "expected checkpoint bundle artifact for kept epoch {kept_epoch}"
            );
        }
    }
    for pruned_epoch in [1, 4] {
        for file in [
            format!("model-{pruned_epoch}.bin"),
            format!("optimizer-{pruned_epoch}.bin"),
            format!("scheduler-{pruned_epoch}.bin"),
            format!("dynamics-{pruned_epoch}.json"),
            format!("model-config-{pruned_epoch}.json"),
            format!("source-selection-state-{pruned_epoch}.json"),
            format!("stability-{pruned_epoch}.json"),
            format!("training-ecs-state-{pruned_epoch}.json"),
        ] {
            assert!(
                !checkpoint_dir.join(file).exists(),
                "expected checkpoint bundle artifact to be pruned for epoch {pruned_epoch}"
            );
        }
    }
}

#[test]
fn file_metric_best_strategy_deletes_old_best_after_replacement() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut strategy = FileMetricBestCheckpointingStrategy::new(
        dir.path(),
        &LossMetric::<TestValidBackend>::new(),
        burn_train::metric::store::Direction::Lowest,
        burn_train::metric::store::Split::Valid,
    );

    for (epoch, mean) in [(1, 3.0), (2, 2.0), (3, 2.5), (4, 1.5)] {
        write_metric_log(dir.path(), "valid", epoch, &[mean]);
        let actions = strategy.actions_for_epoch(epoch);
        apply_checkpoint_actions(dir.path(), epoch, &actions);
    }

    assert_eq!(strategy.best_epoch, Some(4));
    assert_eq!(retained_model_epochs(dir.path()), vec![3, 4]);
}

#[test]
fn file_metric_best_strategy_rehydrates_history_when_resuming() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut strategy = FileMetricBestCheckpointingStrategy::new(
        dir.path(),
        &LossMetric::<TestValidBackend>::new(),
        burn_train::metric::store::Direction::Lowest,
        burn_train::metric::store::Split::Valid,
    );

    for (epoch, mean) in [(1, 3.0), (2, 1.5), (3, 2.0), (4, 2.1), (5, 2.2), (6, 2.3)] {
        write_metric_log(dir.path(), "valid", epoch, &[mean]);
    }
    for epoch in [2, 5, 6] {
        apply_checkpoint_actions(dir.path(), epoch, &[CheckpointingAction::Save]);
    }

    write_metric_log(dir.path(), "valid", 7, &[2.4]);
    let actions = strategy.actions_for_epoch(7);
    apply_checkpoint_actions(dir.path(), 7, &actions);

    assert_eq!(strategy.best_epoch, Some(2));
    assert_eq!(retained_model_epochs(dir.path()), vec![2, 6, 7]);
}

#[test]
fn file_metric_best_strategy_recomputes_history_when_new_best_log_arrives_late() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut strategy = FileMetricBestCheckpointingStrategy::new(
        dir.path(),
        &LossMetric::<TestValidBackend>::new(),
        burn_train::metric::store::Direction::Lowest,
        burn_train::metric::store::Split::Valid,
    );

    for epoch in 1..=23 {
        let mean = if epoch == 23 {
            1.50
        } else {
            2.0 + epoch as f64 * 0.01
        };
        write_metric_log(dir.path(), "valid", epoch, &[mean]);
        let actions = strategy.actions_for_epoch(epoch);
        apply_checkpoint_actions(dir.path(), epoch, &actions);
    }

    for epoch in 24..=28 {
        write_metric_log(dir.path(), "valid", epoch, &[1.60 + epoch as f64 * 0.001]);
        let actions = strategy.actions_for_epoch(epoch);
        apply_checkpoint_actions(dir.path(), epoch, &actions);
    }

    let actions = strategy.actions_for_epoch(29);
    apply_checkpoint_actions(dir.path(), 29, &actions);
    write_metric_log(dir.path(), "valid", 29, &[1.48]);

    write_metric_log(dir.path(), "valid", 30, &[1.49]);
    let actions = strategy.actions_for_epoch(30);
    apply_checkpoint_actions(dir.path(), 30, &actions);

    assert_eq!(strategy.best_epoch, Some(29));
    assert_eq!(retained_model_epochs(dir.path()), vec![29, 30]);
}

#[derive(Clone)]
struct StaticSequenceLoader<B: BackendTrait> {
    items: Vec<SequenceBatch<B>>,
}

impl<B: BackendTrait> StaticSequenceLoader<B> {
    fn new(items: Vec<SequenceBatch<B>>) -> Self {
        Self { items }
    }
}

struct StaticSequenceIterator<B: BackendTrait> {
    items: Vec<SequenceBatch<B>>,
    index: usize,
}

impl<B: BackendTrait> Iterator for StaticSequenceIterator<B> {
    type Item = SequenceBatch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.items.get(self.index).cloned();
        if item.is_some() {
            self.index += 1;
        }
        item
    }
}

impl<B: BackendTrait> DataLoaderIterator<SequenceBatch<B>> for StaticSequenceIterator<B> {
    fn progress(&self) -> Progress {
        Progress::new(self.index, self.items.len())
    }
}

impl<B> DataLoader<B, SequenceBatch<B>> for StaticSequenceLoader<B>
where
    B: BackendTrait + 'static,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<SequenceBatch<B>> + 'a> {
        Box::new(StaticSequenceIterator {
            items: self.items.clone(),
            index: 0,
        })
    }

    fn num_items(&self) -> usize {
        self.items.len()
    }

    fn to_device(&self, _device: &B::Device) -> Arc<dyn DataLoader<B, SequenceBatch<B>>> {
        Arc::new(self.clone())
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, SequenceBatch<B>>> {
        let len = self.items.len();
        let start = start.min(len);
        let end = end.min(len);
        Arc::new(Self {
            items: self.items[start..end].to_vec(),
        })
    }
}

fn make_batch<B: BackendTrait>(
    device: &B::Device,
    inputs: &[i64],
    targets: &[i64],
    shape: [usize; 2],
) -> SequenceBatch<B> {
    SequenceBatch::new(
        Tensor::<B, 2, Int>::from_data(TensorData::new(inputs.to_vec(), shape), device),
        Tensor::<B, 2, Int>::from_data(TensorData::new(targets.to_vec(), shape), device),
        None,
    )
}

fn tensor_values<B: BackendTrait, const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("tensor values")
}

fn max_abs_diff(lhs: Vec<f32>, rhs: Vec<f32>) -> f32 {
    assert_eq!(lhs.len(), rhs.len(), "tensor length mismatch");
    lhs.into_iter()
        .zip(rhs)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max)
}

fn tiny_model_config() -> DragonConfig {
    DragonConfig {
        n_layer: 1,
        n_embd: 8,
        n_head: 1,
        mlp_internal_dim_multiplier: 1,
        dropout: 0.0,
        vocab_size: 16,
        ..Default::default()
    }
}

#[test]
fn proof_action_batch_scores_match_independent_variable_length_rows() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 41);
    let model = DragonModel::<TestBackend>::new(tiny_model_config(), &device).valid();
    let prompts = vec![vec![1, 2, 3], vec![4, 3, 2, 1, 0]];
    let candidates = vec![
        vec![vec![7, 5, 9], vec![7, 6, 9]],
        vec![vec![8, 3, 10], vec![8, 4, 10], vec![8, 5, 10]],
    ];

    let batched = crate::train::ruliad_policy::constrained_completion_log_probs_batch(
        &model,
        &prompts,
        &candidates,
        &device,
    )
    .expect("batched action scores");
    for (row, (prompt, candidate_group)) in prompts.iter().zip(&candidates).enumerate() {
        let independent = crate::train::ruliad_policy::constrained_completion_log_probs(
            &model,
            prompt,
            candidate_group,
            &device,
        )
        .expect("independent action scores");
        assert_eq!(batched[row].len(), independent.len());
        for (actual, expected) in batched[row].iter().zip(independent) {
            assert!(
                (actual - expected).abs() <= 1.0e-5,
                "batched score {actual} differs from independent score {expected}"
            );
        }
    }
}

#[test]
fn proof_action_scoring_chunks_bound_rows_and_padded_tokens() {
    let lengths = [100, 200, 100, 1_000];
    assert_eq!(bounded_padded_batch_end(&lengths, 0, 2, 1_000), 2);
    assert_eq!(bounded_padded_batch_end(&lengths, 0, 64, 350), 1);
    assert_eq!(bounded_padded_batch_end(&lengths, 1, 64, 450), 3);
    assert_eq!(bounded_padded_batch_end(&lengths, 3, 64, 10), 4);

    let mut summary = RuliadPolicyScoringSummary::default();
    summary.record_batch(&[10, 20]);
    summary.record_batch(&[30]);
    summary.record_pipeline_depth(2);
    summary.record_pipeline_depth(1);
    assert_eq!(summary.batches, 2);
    assert_eq!(summary.rows, 3);
    assert_eq!(summary.unpadded_tokens, 60);
    assert_eq!(summary.padded_tokens, 70);
    assert_eq!(summary.maximum_batch_rows, 2);
    assert_eq!(summary.maximum_pipeline_depth, 2);
}

fn tiny_training_hparams() -> TrainingHyperparameters {
    TrainingHyperparameters {
        algorithm: TrainingAlgorithm::Auto,
        block_size: 4,
        tbptt_chunk_size: None,
        tbptt_credit_window_chunks: 1,
        tbptt_persist_across_steps: false,
        sequence_batching: Default::default(),
        retain_ephemeral_terminal_sequence_state: false,
        min_logical_block_size: None,
        batch_size: 2,
        seed: 1337,
        gradient_accumulation_steps: 1,
        target_effective_batch_size: None,
        epochs: Some(1),
        max_iters: 2,
        checkpoint_interval_iters: 2000,
        log_frequency: 1,
        launch_mode: burn_dragon_train::train::pipeline::TrainingLaunchMode::Fresh,
        resume_run_dir: None,
        resume_checkpoint_epoch: None,
        resume_horizon_extension: Default::default(),
        init_checkpoint_path: None,
        init_checkpoint_epoch: None,
        source_selection_state_path: None,
        init_transfer: Default::default(),
        continual_backprop: Default::default(),
        input_corruption: Default::default(),
        logit_entropy_floor: Default::default(),
        repeat_unlikelihood: Default::default(),
        greedy_rollout_unlikelihood: Default::default(),
        dynamics_anchor: Default::default(),
        predictive_coding: Default::default(),
        local_predictive_coding: Default::default(),
        predictive_context_routing: Default::default(),
        latent_reasoning: Default::default(),
        ruliad_supervision: Default::default(),
        ruliad_probe_generation: Default::default(),
        ruliad_policy_probe: Default::default(),
        module_lr_scales: Vec::new(),
        context_strategy: ContextStrategyConfig::Infinite,
        sequence_kernel_override: None,
        objective: Default::default(),
        gdpo: None,
        events: Default::default(),
        validation: Default::default(),
        sequence_state_probe: Default::default(),
        gates: Default::default(),
        dynamics: Default::default(),
        neuron_scaling: Default::default(),
        auto_batch_size: Default::default(),
    }
}

fn tiny_training_hparams_with_epochs(
    epochs: usize,
    resume_checkpoint_epoch: Option<usize>,
) -> TrainingHyperparameters {
    let mut training = tiny_training_hparams();
    training.epochs = Some(epochs);
    training.resume_checkpoint_epoch = resume_checkpoint_epoch;
    training
}

#[test]
fn persistent_tbptt_uses_stream_loss_metric_name() {
    let mut training = tiny_training_hparams();
    training.tbptt_persist_across_steps = true;
    training.log_frequency = 7;
    training.events.source_selection_every_steps = 2;

    assert_eq!(train_loss_metric_name(&training), METRIC_STREAM_WARM_LOSS);
    assert_eq!(
        crate::train::events::train_loss_metric_frequency(&training, None),
        7
    );
    assert!(!source_selection_telemetry_due_for(&training, None, 0));
}

#[test]
fn predictive_coding_state_only_control_disables_optimizer_steps() {
    let mut training = tiny_training_hparams();
    assert!(parameter_updates_enabled(&training));

    training.predictive_coding.enabled = true;
    assert!(parameter_updates_enabled(&training));

    training.predictive_coding.parameter_update = PredictiveCodingParameterUpdate::StateOnlyControl;
    assert!(!parameter_updates_enabled(&training));
}

fn objective_training_hparams(objective: TrainingObjectiveConfig) -> TrainingHyperparameters {
    let mut training = tiny_training_hparams();
    training.objective = objective;
    training
}

fn tiny_language_optimizer(
    training: &TrainingHyperparameters,
    model_config: &DragonConfig,
    device: &burn::tensor::Device<TestBackend>,
) -> crate::train::continual_backprop::LanguageOptimizer<TestBackend> {
    let optimizer_cfg = OptimizerConfig {
        name: OptimizerKind::Adamw,
        learning_rate: 1e-3,
        weight_decay: 0.0,
        weight_decay_final: None,
        lr_schedule: None,
        schedule_mode: OptimizerScheduleMode::DragonReference,
        grad_clip_norm: None,
        grad_clip_value: None,
        eggroll: burn_eggroll::EggrollConfig::default(),
        eggroll_population_execution: Default::default(),
        eggroll_auto_population: Default::default(),
        predictive_coding: Default::default(),
    };
    let fresh_model = DragonModel::<TestBackend>::new(model_config.clone(), device);
    crate::train::continual_backprop::resolve_dragon_language_optimizer::<TestBackend>(
        training,
        &optimizer_cfg,
        1,
        fresh_model,
    )
    .expect("optimizer")
}

#[test]
fn eggroll_chunk_autotune_candidates_are_even_bounded_and_include_configured() {
    let optimizer_cfg = OptimizerConfig {
        name: OptimizerKind::Eggroll,
        learning_rate: 1e-2,
        weight_decay: 0.0,
        weight_decay_final: None,
        lr_schedule: None,
        schedule_mode: OptimizerScheduleMode::DragonReference,
        grad_clip_norm: None,
        grad_clip_value: None,
        eggroll: burn_eggroll::EggrollConfig {
            population: burn_eggroll::PopulationConfig {
                population_size: 512,
                population_chunk_size: 64,
                rank: 1,
                seed: 7,
                matrix_noise: burn_eggroll::MatrixNoiseMode::default(),
            },
            ..burn_eggroll::EggrollConfig::default()
        },
        eggroll_population_execution: Default::default(),
        eggroll_auto_population: burn_dragon_train::EggrollAutoPopulationConfig {
            enabled: true,
            chunk_autotune: burn_dragon_train::EggrollChunkAutotuneConfig {
                enabled: true,
                candidates: vec![32, 128, 256],
                max_probe_population_size: 128,
            },
            ..Default::default()
        },
        predictive_coding: Default::default(),
    };

    let candidates = super::resolve_eggroll_chunk_autotune_candidates(&optimizer_cfg);

    assert_eq!(candidates, vec![32, 64, 128]);
}

#[test]
fn eggroll_population_execution_stacked_tensorized_matches_manual_shared_lowrank_members() {
    let device = burn::tensor::Device::<TestForwardBackend>::default();
    TestForwardBackend::seed(&device, 17);
    let model = LanguageTrainModel::new(DragonModel::<TestForwardBackend>::new(
        tiny_model_config(),
        &device,
    ));
    let batch = make_batch::<TestForwardBackend>(
        &device,
        &[0, 1, 2, 3, 4, 5, 6, 7],
        &[1, 2, 3, 4, 5, 6, 7, 8],
        [2, 4],
    );
    let eggroll = burn_eggroll::EggrollConfig {
        sigma: 1.0e-3,
        population: burn_eggroll::PopulationConfig {
            population_size: 4,
            population_chunk_size: 4,
            rank: 2,
            seed: 99,
            matrix_noise: burn_eggroll::MatrixNoiseMode::default(),
        },
        ..burn_eggroll::EggrollConfig::default()
    };
    let optimizer_cfg = OptimizerConfig {
        name: OptimizerKind::Eggroll,
        learning_rate: 1.0e-4,
        weight_decay: 0.0,
        weight_decay_final: None,
        lr_schedule: None,
        schedule_mode: OptimizerScheduleMode::DragonReference,
        grad_clip_norm: None,
        grad_clip_value: None,
        eggroll: eggroll.clone(),
        eggroll_population_execution: burn_dragon_train::EggrollPopulationExecutionConfig {
            perturbation_scope: EggrollPerturbationScope::DragonCoreProjection,
            backend: EggrollPopulationExecutionBackend::Reference,
            population_tile_size: None,
        },
        eggroll_auto_population: Default::default(),
        predictive_coding: Default::default(),
    };
    let plan = resolve_eggroll_population_execution_plan(&optimizer_cfg, &model)
        .expect("stacked tensorized plan");
    assert_eq!(plan.executor_name(), "stacked_tensorized");

    let pair_count = 2;
    let lowrank = build_shared_lowrank_population_weights(&model, &eggroll, 3, 0, pair_count);
    let lowrank_factors =
        build_shared_lowrank_population_factors(&model, &eggroll, 3, 0, pair_count);
    let base_weights = model.model.shared_lowrank_weights();
    let stacked_logits = model
        .model
        .forward_with_shared_lowrank_population(batch.inputs.clone(), lowrank.clone());
    let factorized_logits = model.model.forward_with_shared_lowrank_population_factors(
        batch.inputs.clone(),
        lowrank_factors.clone(),
    );
    let factorized_logit_diff = max_abs_diff(
        tensor_values(stacked_logits.clone()),
        tensor_values(factorized_logits.clone()),
    );
    assert!(
        factorized_logit_diff <= 1.0e-4,
        "materialized and factorized population logits drifted by {factorized_logit_diff}"
    );
    let [base_batch, _time] = batch.inputs.shape().dims::<2>();
    let mut manual_losses = Vec::with_capacity(pair_count * 2);
    for member in 0..pair_count * 2 {
        let member_weights = SharedLowrankWeights {
            encoder: lowrank
                .encoder
                .clone()
                .slice_dim(0, member..member + 1)
                .reshape(base_weights.encoder.shape().dims::<3>()),
            encoder_v: lowrank
                .encoder_v
                .clone()
                .slice_dim(0, member..member + 1)
                .reshape(base_weights.encoder_v.shape().dims::<3>()),
            decoder: lowrank
                .decoder
                .clone()
                .slice_dim(0, member..member + 1)
                .reshape(base_weights.decoder.shape().dims::<2>()),
        };
        let manual_model = model
            .clone()
            .map_model(|dragon| dragon.with_shared_lowrank_weights(member_weights));
        let manual_logits = manual_model.model.forward(batch.inputs.clone());
        let single_lowrank = SharedLowrankPopulationWeights {
            encoder: lowrank.encoder.clone().slice_dim(0, member..member + 1),
            encoder_v: lowrank.encoder_v.clone().slice_dim(0, member..member + 1),
            decoder: lowrank.decoder.clone().slice_dim(0, member..member + 1),
        };
        let single_logits = model
            .model
            .forward_with_shared_lowrank_population(batch.inputs.clone(), single_lowrank);
        let single_logit_diff = max_abs_diff(
            tensor_values(manual_logits.clone()),
            tensor_values(single_logits),
        );
        assert!(
            single_logit_diff <= 1.0e-5,
            "manual and single tensorized member {member} logits drifted by {single_logit_diff}"
        );
        let stacked_member_logits = stacked_logits
            .clone()
            .slice_dim(0, member * base_batch..(member + 1) * base_batch);
        let stacked_logit_diff = max_abs_diff(
            tensor_values(manual_logits.clone()),
            tensor_values(stacked_member_logits),
        );
        assert!(
            stacked_logit_diff <= 1.0e-5,
            "manual and stacked tensorized member {member} logits drifted by {stacked_logit_diff}"
        );
        let factorized_member_logits = factorized_logits
            .clone()
            .slice_dim(0, member * base_batch..(member + 1) * base_batch);
        let factorized_member_diff = max_abs_diff(
            tensor_values(manual_logits),
            tensor_values(factorized_member_logits),
        );
        assert!(
            factorized_member_diff <= 1.0e-4,
            "manual and factorized member {member} logits drifted by {factorized_member_diff}"
        );
        manual_losses.push(eggroll_batch_loss_tensor(&manual_model, batch.clone()));
    }
    let stacked_tensorized =
        evaluate_eggroll_population_chunk(&plan, &model, batch.clone(), &eggroll, 3, 0, pair_count)
            .expect("stacked tensorized losses");
    let factorized_optimizer_cfg = OptimizerConfig {
        eggroll_population_execution: burn_dragon_train::EggrollPopulationExecutionConfig {
            backend: EggrollPopulationExecutionBackend::Factorized,
            perturbation_scope: EggrollPerturbationScope::DragonCoreProjection,
            population_tile_size: None,
        },
        ..optimizer_cfg
    };
    let factorized_plan =
        resolve_eggroll_population_execution_plan(&factorized_optimizer_cfg, &model)
            .expect("factorized tensorized plan");
    assert_eq!(factorized_plan.executor_name(), "factorized_tensorized");
    let factorized_tensorized = evaluate_eggroll_population_chunk(
        &factorized_plan,
        &model,
        batch,
        &eggroll,
        3,
        0,
        pair_count,
    )
    .expect("factorized tensorized losses");
    let manual = scalar_values_from_loss_tensors(manual_losses);

    assert_eq!(stacked_tensorized.len(), manual.len());
    assert_eq!(factorized_tensorized.len(), manual.len());
    assert!(stacked_tensorized.iter().all(|loss| loss.is_finite()));
    assert!(factorized_tensorized.iter().all(|loss| loss.is_finite()));
    for ((expected, actual), factorized) in manual
        .iter()
        .zip(stacked_tensorized.iter())
        .zip(factorized_tensorized.iter())
    {
        assert!(
            (expected - actual).abs() <= 1.0e-4,
            "manual={expected} stacked={actual}"
        );
        assert!(
            (expected - factorized).abs() <= 1.0e-4,
            "manual={expected} factorized={factorized}"
        );
    }
}

#[test]
fn eggroll_training_dynamics_are_bounded_against_adamw() {
    let dir = tempfile::tempdir().expect("tempdir");
    let report = crate::train::optimizer_dynamics::run_optimizer_dynamics_suite(
        &crate::train::optimizer_dynamics::OptimizerDynamicsConfig::default(),
        &[17, 29, 53],
        dir.path(),
    )
    .expect("optimizer dynamics suite");

    eprintln!("optimizer dynamics suite: {report:#?}");
    for pair in &report.pairs {
        assert!(pair.adamw.initial_train_loss.is_finite());
        assert!(pair.adamw.final_train_loss.is_finite());
        assert!(pair.adamw.initial_loss.is_finite());
        assert!(pair.adamw.final_loss.is_finite());
        assert!(pair.eggroll.initial_train_loss.is_finite());
        assert!(pair.eggroll.final_train_loss.is_finite());
        assert!(pair.eggroll.initial_loss.is_finite());
        assert!(pair.eggroll.final_loss.is_finite());
        assert!(
            pair.adamw.loss_delta() > 0.0,
            "AdamW should learn the deterministic comparison task: {pair:?}"
        );
        assert!(
            pair.eggroll.train_loss_delta() > 0.0,
            "EGGROLL should reduce train loss in the deterministic comparison task: {pair:?}"
        );
        assert!(
            pair.eggroll.final_loss <= pair.eggroll.initial_loss + 0.05,
            "EGGROLL should not severely regress the deterministic comparison task: {pair:?}"
        );
        assert!(
            pair.eggroll.evaluations_per_second() >= pair.adamw.evaluations_per_second() * 0.02,
            "EGGROLL eval throughput is pathologically low: {pair:?}"
        );
    }
    let min_mean_eggroll_loss_delta = 0.01;
    assert!(
        report.mean_eggroll_loss_delta() > min_mean_eggroll_loss_delta,
        "tensorized EGGROLL should learn a positive average signal: {report:#?}"
    );
    assert!(
        report.mean_eggroll_train_loss_delta() > 0.015,
        "tensorized EGGROLL should reduce train loss by a measurable average signal: {report:#?}"
    );
    let min_adamw_fraction = 0.005;
    assert!(
        report.mean_eggroll_loss_delta() >= report.mean_adamw_loss_delta() * min_adamw_fraction,
        "tensorized EGGROLL should retain a positive fraction of AdamW quality on the deterministic comparison task: {report:#?}"
    );
}

#[test]
fn manual_adamw_loop_matches_learner_dynamics() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = crate::train::optimizer_dynamics::OptimizerDynamicsConfig {
        epochs: 4,
        max_iters: 16,
        log_frequency: 4,
        seed: 29,
        ..crate::train::optimizer_dynamics::OptimizerDynamicsConfig::default()
    };
    let learner = crate::train::optimizer_dynamics::run_optimizer_dynamics(
        crate::train::optimizer_dynamics::OptimizerDynamicsKind::AdamW,
        &config,
        &dir.path().join("learner"),
    )
    .expect("learner adamw dynamics");
    let manual = crate::train::optimizer_dynamics::run_manual_adamw_optimizer_dynamics(&config)
        .expect("manual adamw dynamics");

    eprintln!("adamw learner={learner:#?} manual={manual:#?}");
    assert!(
        manual.loss_delta() > 0.0,
        "manual AdamW loop should learn the deterministic comparison task: {manual:?}"
    );
    assert!(
        manual.loss_delta() >= learner.loss_delta() * 0.60,
        "manual AdamW loop should be in the same quality regime as burn_train::Learner: learner={learner:?} manual={manual:?}"
    );
}

#[test]
fn eggroll_update_preserves_train_step_gradients() {
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 29);
    let model_config = tiny_model_config();
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        model_config.clone(),
        &device,
    ));
    let batch = make_batch::<TestBackend>(
        &device,
        &[0, 1, 2, 3, 4, 5, 6, 7],
        &[1, 2, 3, 4, 5, 6, 7, 8],
        [2, 4],
    );
    let eggroll = burn_eggroll::EggrollConfig {
        sigma: 0.0025,
        update: burn_eggroll::EggrollUpdateConfig {
            learning_rate: 1.0e-8,
            ..burn_eggroll::EggrollUpdateConfig::default()
        },
        ..burn_eggroll::EggrollConfig::default()
    };
    let mut eggroll_state = burn_dragon_eggroll::EggrollModuleOptimizerState::<TestBackend>::new();
    let (updated, _metrics) = burn_dragon_eggroll::apply_antithetic_update(
        model,
        &eggroll,
        0,
        &[burn_dragon_eggroll::AntitheticFitness {
            pair_index: 0,
            plus: 0.0,
            minus: 1.0,
        }],
        &mut eggroll_state,
    )
    .expect("eggroll update");
    let item = burn_train::TrainStep::step(&updated, batch);
    let raw_gradient_count = item.grads.len();
    let mut accumulator = GradientsAccumulator::new();
    accumulator.accumulate(&updated, item.grads);
    let grads = accumulator.grads();
    let accumulated_gradient_count = grads.len();

    eprintln!(
        "eggroll-updated gradient counts raw={raw_gradient_count} accumulated={accumulated_gradient_count}"
    );
    assert!(
        raw_gradient_count > 0,
        "EGGROLL-updated model should expose train-step gradients"
    );
    assert!(
        accumulated_gradient_count > 0,
        "EGGROLL-updated model should expose accumulated gradients"
    );
}

#[test]
fn eggroll_forward_only_trains_on_plain_backend() {
    let dir = tempfile::tempdir().expect("tempdir");
    let run_dir = dir.path().join("eggroll-forward-only");
    let parallel_config = burn_dragon_train::ParallelConfig::default();
    let parallel_runtime =
        resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");
    let device = burn::tensor::Device::<TestForwardBackend>::default();
    TestForwardBackend::seed(&device, 41);
    let training = tiny_training_hparams();
    let model_config = tiny_model_config();
    let optimizer_cfg = OptimizerConfig {
        name: OptimizerKind::Eggroll,
        learning_rate: 1.0e-6,
        weight_decay: 0.0,
        weight_decay_final: None,
        lr_schedule: None,
        schedule_mode: OptimizerScheduleMode::DragonReference,
        grad_clip_norm: None,
        grad_clip_value: None,
        eggroll: burn_eggroll::EggrollConfig {
            sigma: 2.5e-3,
            population: burn_eggroll::PopulationConfig {
                population_size: 2,
                population_chunk_size: 2,
                rank: 1,
                seed: 41,
                matrix_noise: burn_eggroll::MatrixNoiseMode::default(),
            },
            update: burn_eggroll::EggrollUpdateConfig {
                learning_rate: 1.0e-6,
                ..burn_eggroll::EggrollUpdateConfig::default()
            },
            ..burn_eggroll::EggrollConfig::default()
        },
        eggroll_population_execution: Default::default(),
        eggroll_auto_population: Default::default(),
        predictive_coding: Default::default(),
    };
    let env = ForwardEggrollTrainEnvironment {
        parallel_runtime: &parallel_runtime,
        run_dir: &run_dir,
        run_name: "eggroll-forward-only-smoke",
        backend_name: "cpu",
        training: &training,
        resume_checkpoint_epoch: None,
        model_config: &model_config,
        device: &device,
        train_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<
            TestForwardBackend,
        >(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 8],
            [2, 4],
        )])),
        valid_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<
            TestForwardBackend,
        >(
            &device,
            &[1, 2, 3, 4, 5, 6, 7, 8],
            &[2, 3, 4, 5, 6, 7, 8, 9],
            [2, 4],
        )])),
        source_selection_dataset: None,
        summary_event_token_ids: None,
        epochs: 1,
    };
    let model = LanguageTrainModel::new(DragonModel::<TestForwardBackend>::new(
        model_config.clone(),
        &device,
    ));
    let trained = train_with_eggroll_forward_only(&env, &optimizer_cfg, model)
        .expect("forward-only EGGROLL training should not require autodiff");
    let probe = make_batch::<TestForwardBackend>(
        &device,
        &[1, 2, 3, 4, 5, 6, 7, 8],
        &[2, 3, 4, 5, 6, 7, 8, 9],
        [2, 4],
    );
    let loss =
        language_model_loss::<TestForwardBackend>(trained.forward(probe.inputs), probe.targets)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("loss vec")[0];
    assert!(
        loss.is_finite(),
        "forward-only EGGROLL loss should be finite"
    );
    assert!(
        run_dir.join("checkpoint/model-1.bin").is_file(),
        "forward-only EGGROLL should save plain-backend checkpoints"
    );
}

#[test]
fn eggroll_interval_reduces_population_evaluations() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = crate::train::optimizer_dynamics::OptimizerDynamicsConfig {
        epochs: 4,
        max_iters: 16,
        log_frequency: 4,
        seed: 29,
        eggroll_learning_rate: 1.0e-2,
        eggroll_interval_steps: 4,
        ..crate::train::optimizer_dynamics::OptimizerDynamicsConfig::default()
    };
    let report = crate::train::optimizer_dynamics::run_optimizer_dynamics(
        crate::train::optimizer_dynamics::OptimizerDynamicsKind::Eggroll,
        &config,
        dir.path(),
    )
    .expect("interval eggroll dynamics");
    let total_steps = config.epochs * 4;
    let eggroll_steps = total_steps.div_ceil(config.eggroll_interval_steps);
    let expected_forward_evaluations = eggroll_steps * config.eggroll_population_size;

    eprintln!("interval eggroll report={report:#?}");
    assert_eq!(report.forward_evaluations, expected_forward_evaluations);
    assert!(
        report.final_loss.is_finite(),
        "interval EGGROLL should produce finite validation loss: {report:?}"
    );
}

#[test]
fn eggroll_baseline_is_reasonable_against_nearby_variants() {
    let dir = tempfile::tempdir().expect("tempdir");
    let base = crate::train::optimizer_dynamics::OptimizerDynamicsConfig {
        epochs: 16,
        max_iters: 64,
        log_frequency: 16,
        seed: 29,
        ..crate::train::optimizer_dynamics::OptimizerDynamicsConfig::default()
    };
    let mut center = base.clone();
    center.eggroll_fitness_normalization = burn_eggroll::FitnessNormalization::Center;
    let mut zscore = base.clone();
    zscore.eggroll_fitness_normalization = burn_eggroll::FitnessNormalization::ZScore;
    let mut adamw_update = base.clone();
    adamw_update.eggroll_update_kind = burn_eggroll::EggrollUpdateKind::Adamw;
    adamw_update.eggroll_learning_rate = 1.0e-3;
    let mut smaller_population = base.clone();
    smaller_population.eggroll_population_size = 128;
    let mut larger_population = base.clone();
    larger_population.eggroll_population_size = 512;

    let report = crate::train::optimizer_dynamics::run_eggroll_dynamics_sweep(
        &[
            crate::train::optimizer_dynamics::EggrollDynamicsCandidate::new(
                "rank_sgd_pop256_rank4",
                base.clone(),
            ),
            crate::train::optimizer_dynamics::EggrollDynamicsCandidate::new(
                "center_sgd_pop256_rank4",
                center,
            ),
            crate::train::optimizer_dynamics::EggrollDynamicsCandidate::new(
                "zscore_sgd_pop256_rank4",
                zscore,
            ),
            crate::train::optimizer_dynamics::EggrollDynamicsCandidate::new(
                "rank_adamw_pop256_rank4",
                adamw_update,
            ),
            crate::train::optimizer_dynamics::EggrollDynamicsCandidate::new(
                "rank_sgd_pop128_rank4",
                smaller_population,
            ),
            crate::train::optimizer_dynamics::EggrollDynamicsCandidate::new(
                "rank_sgd_pop512_rank4",
                larger_population,
            ),
        ],
        dir.path(),
    )
    .expect("eggroll dynamics sweep");

    eprintln!("eggroll candidate sweep: {report:#?}");
    let baseline = report
        .get("rank_sgd_pop256_rank4")
        .expect("baseline candidate");
    let best_quality = report.best_by_loss_delta().expect("quality candidate");
    let best_train = report
        .best_by_train_loss_delta()
        .expect("train-loss candidate");
    for candidate in &report.candidates {
        assert!(candidate.report.initial_train_loss.is_finite());
        assert!(candidate.report.final_train_loss.is_finite());
        assert!(candidate.report.initial_loss.is_finite());
        assert!(candidate.report.final_loss.is_finite());
    }
    assert!(
        baseline.report.final_loss <= baseline.report.initial_loss + 0.05,
        "baseline EGGROLL should not regress in the candidate sweep: {report:#?}"
    );
    assert!(
        baseline.report.train_loss_delta() > 0.0,
        "baseline EGGROLL should reduce train loss in the candidate sweep: {report:#?}"
    );
    assert!(
        best_quality.report.loss_delta() > 0.01,
        "at least one tensorized EGGROLL candidate should learn a measurable signal in the candidate sweep: {report:#?}"
    );
    assert!(
        best_train.report.train_loss_delta() > 0.02,
        "at least one tensorized EGGROLL candidate should reduce train loss by a measurable signal in the candidate sweep: {report:#?}"
    );
    assert!(
        baseline.report.loss_delta() >= best_quality.report.loss_delta() * -0.5,
        "tensorized baseline EGGROLL should not be badly dominated on quality by nearby candidates: best={best_quality:?} report={report:#?}"
    );
    assert!(
        baseline.report.evaluations_per_second()
            >= best_quality.report.evaluations_per_second() * 0.25,
        "tensorized baseline EGGROLL should remain throughput-reasonable against nearby candidates: best={best_quality:?} report={report:#?}"
    );
}

fn single_device_scheduler_smoke(objective: TrainingObjectiveConfig, run_name: &str) -> f32 {
    let dir = tempfile::tempdir().expect("tempdir");
    let run_dir = dir.path().join("run");
    let parallel_config = burn_dragon_train::ParallelConfig::default();
    let parallel_runtime =
        resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");

    let primary_device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&primary_device, 11);
    let valid_device = burn::tensor::Device::<TestValidBackend>::default();
    let train_batches = vec![
        make_batch::<TestBackend>(
            &primary_device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 0],
            [2, 4],
        ),
        make_batch::<TestBackend>(
            &primary_device,
            &[7, 6, 5, 4, 3, 2, 1, 0],
            &[6, 5, 4, 3, 2, 1, 0, 7],
            [2, 4],
        ),
    ];
    let valid_batches = vec![make_batch::<TestValidBackend>(
        &valid_device,
        &[0, 0, 1, 1, 2, 2, 3, 3],
        &[0, 1, 1, 2, 2, 3, 3, 0],
        [2, 4],
    )];

    let training = objective_training_hparams(objective.clone());
    let model_config = tiny_model_config();
    let devices = vec![primary_device];
    let env = TrainEnvironment {
        parallel_runtime: &parallel_runtime,
        parallel_config: &parallel_config,
        run_dir: &run_dir,
        run_name,
        backend_name: "cpu",
        training: &training,
        resume_checkpoint_epoch: None,
        model_config: &model_config,
        device: &primary_device,
        devices: &devices,
        train_dataset: None,
        valid_dataset: None,
        train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
        valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
        source_selection_dataset: None,
        summary_event_token_ids: None,
        neuron_scaling_slot: None,
        epochs: 1,
        total_steps: 2,
        valid_steps: 1,
    };
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        model_config.clone(),
        &primary_device,
    ))
    .with_training_objective(objective);
    let optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<TestBackend, LanguageTrainModel<TestBackend>>();

    let trained =
        train_with_scheduler(&env, model, optimizer, 1e-3).expect("objective scheduler train");
    assert!(run_dir.join("checkpoint").join("model-1.bin").is_file());

    let probe = make_batch::<TestValidBackend>(
        &valid_device,
        &[1, 2, 3, 4, 4, 3, 2, 1],
        &[2, 3, 4, 5, 3, 2, 1, 0],
        [2, 4],
    );
    language_model_loss::<TestValidBackend>(trained.forward(probe.inputs), probe.targets)
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("loss vec")[0]
}

#[test]
fn train_with_scheduler_accepts_next_token_objective_toggle() {
    let loss = single_device_scheduler_smoke(
        TrainingObjectiveConfig::NextToken,
        "single-next-token-objective-smoke",
    );
    assert!(loss.is_finite(), "next_token smoke loss must be finite");
}

#[test]
fn train_with_scheduler_accepts_local_predictive_coding_algorithm() {
    let dir = tempfile::tempdir().expect("tempdir");
    let run_dir = dir.path().join("single-pc-optimizer-smoke");
    let parallel_config = burn_dragon_train::ParallelConfig::default();
    let parallel_runtime =
        resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");

    let primary_device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&primary_device, 19);
    let valid_device = burn::tensor::Device::<TestValidBackend>::default();
    let train_batches = vec![
        make_batch::<TestBackend>(
            &primary_device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 0],
            [2, 4],
        ),
        make_batch::<TestBackend>(
            &primary_device,
            &[3, 4, 5, 6, 7, 0, 1, 2],
            &[4, 5, 6, 7, 0, 1, 2, 3],
            [2, 4],
        ),
    ];
    let valid_batches = vec![make_batch::<TestValidBackend>(
        &valid_device,
        &[0, 0, 1, 1, 2, 2, 3, 3],
        &[0, 1, 1, 2, 2, 3, 3, 0],
        [2, 4],
    )];

    let mut training = tiny_training_hparams();
    training.algorithm = TrainingAlgorithm::PredictiveCoding;
    training.local_predictive_coding.inference.steps = 2;
    training.local_predictive_coding.inference.step_size = 0.05;
    let mut model_config = tiny_model_config();
    model_config.dropout = 0.0;
    model_config.sequence_kernel = SequenceKernelConfig::dense_score_short_context();
    model_config.fused_kernels.rotary_embedding = burn_dragon_core::RotaryEmbedding::Alibi;
    let devices = vec![primary_device];
    let env = TrainEnvironment {
        parallel_runtime: &parallel_runtime,
        parallel_config: &parallel_config,
        run_dir: &run_dir,
        run_name: "single-pc-optimizer-smoke",
        backend_name: "cpu",
        training: &training,
        resume_checkpoint_epoch: None,
        model_config: &model_config,
        device: &primary_device,
        devices: &devices,
        train_dataset: None,
        valid_dataset: None,
        train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
        valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
        source_selection_dataset: None,
        summary_event_token_ids: None,
        neuron_scaling_slot: None,
        epochs: 1,
        total_steps: 2,
        valid_steps: 1,
    };
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        model_config.clone(),
        &primary_device,
    ))
    .with_training_algorithm(training.algorithm)
    .with_local_predictive_coding(training.local_predictive_coding.clone());
    let optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<TestBackend, LanguageTrainModel<TestBackend>>();

    let trained = train_with_scheduler(&env, model, optimizer, 1e-3).expect("local PC train");
    assert!(run_dir.join("checkpoint").join("model-1.bin").is_file());
    let events = std::fs::read_to_string(run_dir.join("events/training_events.jsonl"))
        .expect("local PC training events");
    assert!(events.contains("local_factor_vjp_v1"));
    assert!(events.contains("\"global_backward_calls\":0"));

    let probe = make_batch::<TestValidBackend>(
        &valid_device,
        &[1, 2, 3, 4, 4, 3, 2, 1],
        &[2, 3, 4, 5, 3, 2, 1, 0],
        [2, 4],
    );
    let loss =
        language_model_loss::<TestValidBackend>(trained.forward(probe.inputs), probe.targets)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("loss vec")[0];
    assert!(loss.is_finite(), "PC optimizer smoke loss must be finite");
}

#[test]
fn train_with_scheduler_accepts_sdft_objective_toggle() {
    let loss = single_device_scheduler_smoke(
        TrainingObjectiveConfig::Sdft(SdftObjectiveConfig {
            max_completion_tokens: 2,
            top_k: Some(1),
            generate_from_teacher: true,
            num_loss_tokens_to_skip: 1,
            ..Default::default()
        }),
        "single-sdft-objective-smoke",
    );
    assert!(loss.is_finite(), "SDFT smoke loss must be finite");
}

#[test]
fn train_with_scheduler_accepts_sdpo_objective_toggle() {
    let loss = single_device_scheduler_smoke(
        TrainingObjectiveConfig::Sdpo(SdpoObjectiveConfig {
            group_size: 2,
            max_completion_tokens: 2,
            top_k: Some(1),
            ..Default::default()
        }),
        "single-sdpo-objective-smoke",
    );
    assert!(loss.is_finite(), "SDPO smoke loss must be finite");
}

#[test]
fn train_with_scheduler_accepts_composite_sdft_sdpo_objective_toggle() {
    let loss = single_device_scheduler_smoke(
        TrainingObjectiveConfig::SdftSdpo(SdftSdpoObjectiveConfig {
            sdft: SdftObjectiveConfig {
                max_completion_tokens: 2,
                top_k: Some(1),
                ..Default::default()
            },
            sdpo: SdpoObjectiveConfig {
                group_size: 2,
                max_completion_tokens: 2,
                top_k: Some(1),
                ..Default::default()
            },
            ..Default::default()
        }),
        "single-sdft-sdpo-objective-smoke",
    );
    assert!(
        loss.is_finite(),
        "composite SDFT/SDPO smoke loss must be finite"
    );
}

#[test]
fn dynamic_neuron_scale_widens_model_in_process() {
    let dir = tempfile::tempdir().expect("tempdir");
    let run_dir = dir.path().join("run");
    let parallel_config = burn_dragon_train::ParallelConfig::default();
    let parallel_runtime =
        resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 7);
    let valid_device = burn::tensor::Device::<TestValidBackend>::default();
    let mut training = tiny_training_hparams();
    training.neuron_scaling.enabled = true;
    training.neuron_scaling.max_latent_total = 16;
    training.neuron_scaling.stabilization.freeze_base_steps = 1;
    training.neuron_scaling.stabilization.unfreeze_ramp_steps = 1;
    let model_config = tiny_model_config();
    let devices = vec![device];
    let train_batches = vec![make_batch::<TestBackend>(
        &device,
        &[0, 1, 2, 3, 4, 5, 6, 7],
        &[1, 2, 3, 4, 5, 6, 7, 0],
        [2, 4],
    )];
    let valid_batches = vec![make_batch::<TestValidBackend>(
        &valid_device,
        &[0, 0, 1, 1, 2, 2, 3, 3],
        &[0, 1, 1, 2, 2, 3, 3, 0],
        [2, 4],
    )];
    let env = TrainEnvironment {
        parallel_runtime: &parallel_runtime,
        parallel_config: &parallel_config,
        run_dir: &run_dir,
        run_name: "dynamic-scale-smoke",
        backend_name: "cpu",
        training: &training,
        resume_checkpoint_epoch: None,
        model_config: &model_config,
        device: &device,
        devices: &devices,
        train_dataset: None,
        valid_dataset: None,
        train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
        valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
        source_selection_dataset: None,
        summary_event_token_ids: None,
        neuron_scaling_slot: None,
        epochs: 1,
        total_steps: 1,
        valid_steps: 1,
    };
    let mut model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        model_config.clone(),
        &device,
    ))
    .with_gradient_scale_schedule(&training, 1);
    let mut optimizer = tiny_language_optimizer(&training, &model_config, &device);
    let handles = crate::train::events::build_training_event_handles(
        "dynamic-scale-smoke",
        &run_dir,
        1,
        &training,
        None,
        None,
        None,
    )
    .expect("event handles");
    let bus = handles.metric_logger.bus();
    let mut current_model_config = model_config.clone();
    let mut scale_generation = 0usize;

    let scale_result = apply_dynamic_neuron_scale(
        &env,
        DynamicNeuronScaleState {
            model: &mut model,
            optimizer: &mut optimizer,
            model_config: &mut current_model_config,
            scale_generation: &mut scale_generation,
            batch_size: training.batch_size,
            gradient_accumulation_steps: training.gradient_accumulation_steps,
        },
        ModelScaleRequest {
            run_id: "dynamic-scale-smoke".to_string().into(),
            epoch: Some(1),
            absolute_step: Some(0),
            from_capacity_units: 8,
            to_capacity_units: 16,
            reason: "test plateau".to_string(),
        },
        TrainingEventContext {
            epoch: 1,
            absolute_step: 0,
            bus: &bus,
        },
    )
    .expect("apply scale");

    let _ = bus.flush();
    assert_eq!(scale_result, Some((8, 16)));
    assert_eq!(model.model.latent_total_capacity(), 16);
    assert_eq!(current_model_config.latent_total(), 16);
    assert_eq!(scale_generation, 1);
}

#[test]
fn dynamic_neuron_scaling_scheduler_consumes_request_in_process() {
    let dir = tempfile::tempdir().expect("tempdir");
    let run_dir = dir.path().join("run");
    let parallel_config = burn_dragon_train::ParallelConfig::default();
    let parallel_runtime =
        resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 13);
    let valid_device = burn::tensor::Device::<TestValidBackend>::default();
    let mut training = tiny_training_hparams();
    training.neuron_scaling.enabled = true;
    training.neuron_scaling.max_latent_total = 16;
    let model_config = tiny_model_config();
    let devices = vec![device];
    let request_slot = crate::train::neuron_scaling::NeuronScaleRequestSlot::default();
    assert!(request_slot.set_if_empty(ModelScaleRequest {
        run_id: "dynamic-scale-loop-smoke".to_string().into(),
        epoch: Some(1),
        absolute_step: Some(0),
        from_capacity_units: 8,
        to_capacity_units: 16,
        reason: "test plateau".to_string(),
    }));
    let train_batches = vec![make_batch::<TestBackend>(
        &device,
        &[0, 1, 2, 3, 4, 5, 6, 7],
        &[1, 2, 3, 4, 5, 6, 7, 0],
        [2, 4],
    )];
    let valid_batches = vec![make_batch::<TestValidBackend>(
        &valid_device,
        &[0, 0, 1, 1, 2, 2, 3, 3],
        &[0, 1, 1, 2, 2, 3, 3, 0],
        [2, 4],
    )];
    let env = TrainEnvironment {
        parallel_runtime: &parallel_runtime,
        parallel_config: &parallel_config,
        run_dir: &run_dir,
        run_name: "dynamic-scale-loop-smoke",
        backend_name: "cpu",
        training: &training,
        resume_checkpoint_epoch: None,
        model_config: &model_config,
        device: &device,
        devices: &devices,
        train_dataset: None,
        valid_dataset: None,
        train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
        valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
        source_selection_dataset: None,
        summary_event_token_ids: None,
        neuron_scaling_slot: Some(request_slot.clone()),
        epochs: 1,
        total_steps: 1,
        valid_steps: 1,
    };
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        model_config.clone(),
        &device,
    ))
    .with_gradient_scale_schedule(&training, 1);
    let optimizer = tiny_language_optimizer(&training, &model_config, &device);

    let trained = train_with_dynamic_neuron_scaling_scheduler(&env, model, optimizer, 1e-3)
        .expect("dynamic scaling train");

    assert_eq!(trained.latent_total_capacity(), 16);
    assert!(request_slot.take().is_none());
    assert!(run_dir.join("checkpoint").join("model-1.bin").is_file());
}

#[test]
fn dynamic_scheduler_throttles_train_metric_events() {
    let dir = tempfile::tempdir().expect("tempdir");
    let run_dir = dir.path().join("run");
    let parallel_config = burn_dragon_train::ParallelConfig::default();
    let parallel_runtime =
        resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 19);
    let valid_device = burn::tensor::Device::<TestValidBackend>::default();
    let mut training = tiny_training_hparams();
    training.log_frequency = 2;
    training.events.flush_every_steps = 1;
    training.events.degeneracy_probe_every_epochs = usize::MAX;
    let model_config = tiny_model_config();
    let devices = vec![device];
    let train_batches = vec![
        make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 0],
            [2, 4],
        ),
        make_batch::<TestBackend>(
            &device,
            &[1, 2, 3, 4, 5, 6, 7, 0],
            &[2, 3, 4, 5, 6, 7, 0, 1],
            [2, 4],
        ),
        make_batch::<TestBackend>(
            &device,
            &[2, 3, 4, 5, 6, 7, 0, 1],
            &[3, 4, 5, 6, 7, 0, 1, 2],
            [2, 4],
        ),
    ];
    let valid_batches = vec![make_batch::<TestValidBackend>(
        &valid_device,
        &[0, 0, 1, 1, 2, 2, 3, 3],
        &[0, 1, 1, 2, 2, 3, 3, 0],
        [2, 4],
    )];
    let env = TrainEnvironment {
        parallel_runtime: &parallel_runtime,
        parallel_config: &parallel_config,
        run_dir: &run_dir,
        run_name: "dynamic-metric-throttle-smoke",
        backend_name: "cpu",
        training: &training,
        resume_checkpoint_epoch: None,
        model_config: &model_config,
        device: &device,
        devices: &devices,
        train_dataset: None,
        valid_dataset: None,
        train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
        valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
        source_selection_dataset: None,
        summary_event_token_ids: None,
        neuron_scaling_slot: None,
        epochs: 1,
        total_steps: 3,
        valid_steps: 1,
    };
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        model_config.clone(),
        &device,
    ))
    .with_gradient_scale_schedule(&training, 3);
    let optimizer = tiny_language_optimizer(&training, &model_config, &device);

    let _trained = train_with_dynamic_neuron_scaling_scheduler(&env, model, optimizer, 1e-3)
        .expect("dynamic scheduler train");

    let events = read_training_events(&run_dir);
    let train_loss_steps = events
        .iter()
        .filter(|event| {
            event.get("type").and_then(|value| value.as_str()) == Some("metric")
                && event.get("split").and_then(|value| value.as_str()) == Some("train")
                && event.get("name").and_then(|value| value.as_str()) == Some("Loss")
        })
        .map(|event| {
            event
                .get("step_in_epoch")
                .and_then(|value| value.as_u64())
                .expect("train loss step") as usize
        })
        .collect::<Vec<_>>();

    assert_eq!(train_loss_steps, vec![2, 3]);
}

#[test]
fn dynamic_scheduler_emits_canonical_local_predictive_coding_contract() {
    let dir = tempfile::tempdir().expect("tempdir");
    let run_dir = dir.path().join("run");
    let parallel_config = burn_dragon_train::ParallelConfig::default();
    let parallel_runtime =
        resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 31);
    let valid_device = burn::tensor::Device::<TestValidBackend>::default();
    let mut training = tiny_training_hparams();
    training.algorithm = TrainingAlgorithm::PredictiveCoding;
    training.log_frequency = 2;
    training.local_predictive_coding.inference.steps = 2;
    training.local_predictive_coding.inference.step_size = 0.05;
    training.events.flush_every_steps = 1;
    training.events.degeneracy_probe_every_epochs = usize::MAX;
    let mut model_config = tiny_model_config();
    model_config.sequence_kernel = SequenceKernelConfig::dense_score_short_context();
    model_config.fused_kernels.rotary_embedding = burn_dragon_core::RotaryEmbedding::Alibi;
    let devices = vec![device];
    let train_batches = vec![
        make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 0],
            [2, 4],
        ),
        make_batch::<TestBackend>(
            &device,
            &[3, 4, 5, 6, 7, 0, 1, 2],
            &[4, 5, 6, 7, 0, 1, 2, 3],
            [2, 4],
        ),
    ];
    let valid_batches = vec![make_batch::<TestValidBackend>(
        &valid_device,
        &[0, 0, 1, 1, 2, 2, 3, 3],
        &[0, 1, 1, 2, 2, 3, 3, 0],
        [2, 4],
    )];
    let env = TrainEnvironment {
        parallel_runtime: &parallel_runtime,
        parallel_config: &parallel_config,
        run_dir: &run_dir,
        run_name: "dynamic-local-pc-smoke",
        backend_name: "cpu",
        training: &training,
        resume_checkpoint_epoch: None,
        model_config: &model_config,
        device: &device,
        devices: &devices,
        train_dataset: None,
        valid_dataset: None,
        train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
        valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
        source_selection_dataset: None,
        summary_event_token_ids: None,
        neuron_scaling_slot: None,
        epochs: 1,
        total_steps: 2,
        valid_steps: 1,
    };
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        model_config.clone(),
        &device,
    ))
    .with_training_configuration(&training, 2);
    let optimizer = tiny_language_optimizer(&training, &model_config, &device);

    let _trained = train_with_dynamic_neuron_scaling_scheduler(&env, model, optimizer, 1e-3)
        .expect("dynamic local-PC train");

    let run_manifest: burn_pc::PcCheckpointManifest = serde_json::from_slice(
        &std::fs::read(run_dir.join("predictive-coding-program.json"))
            .expect("dynamic scheduler PC run manifest"),
    )
    .expect("parse dynamic scheduler PC run manifest");
    let checkpoint_manifest: burn_pc::PcCheckpointManifest = serde_json::from_slice(
        &std::fs::read(
            run_dir
                .join("checkpoint")
                .join("predictive-coding-manifest-1.json"),
        )
        .expect("dynamic scheduler PC checkpoint manifest"),
    )
    .expect("parse dynamic scheduler PC checkpoint manifest");
    assert_eq!(run_manifest, checkpoint_manifest);

    let events = read_training_events(&run_dir);
    let sample = events
        .iter()
        .find(|event| {
            event.get("type").and_then(|value| value.as_str()) == Some("predictive_coding")
        })
        .expect("local-PC telemetry event");
    assert_eq!(
        sample
            .get("learning_contract")
            .and_then(|value| value.as_str()),
        Some("local_factor_vjp_v1")
    );
    assert_eq!(
        sample
            .get("global_autodiff_graph")
            .and_then(|value| value.as_bool()),
        Some(false)
    );
    assert_eq!(
        sample
            .get("global_backward_calls")
            .and_then(|value| value.as_u64()),
        Some(0)
    );
    assert!(
        sample
            .get("local_vjp_calls")
            .and_then(|value| value.as_u64())
            .is_some_and(|calls| calls > 0)
    );
    assert_eq!(
        sample
            .get("gradient_tensors")
            .and_then(|value| value.as_u64()),
        Some(18)
    );
}

#[test]
fn dynamic_scheduler_defers_validation_and_emits_unpromoted_checkpoint() {
    let dir = tempfile::tempdir().expect("tempdir");
    let run_dir = dir.path().join("run");
    let parallel_config = burn_dragon_train::ParallelConfig::default();
    let parallel_runtime =
        resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 23);
    let valid_device = burn::tensor::Device::<TestValidBackend>::default();
    let mut training = tiny_training_hparams();
    training.validation.execution = crate::config::TrainingValidationExecution::ExternalEvaluator;
    training.gates.enabled = false;
    training.dynamics.enabled = false;
    training.events.ruliad_correctness_probe_items = 0;
    training.events.source_weighted_validation_batches = 0;
    training.ruliad_policy_probe.enabled = false;
    let model_config = tiny_model_config();
    let devices = vec![device];
    let train_batches = vec![make_batch::<TestBackend>(
        &device,
        &[0, 1, 2, 3, 4, 5, 6, 7],
        &[1, 2, 3, 4, 5, 6, 7, 0],
        [2, 4],
    )];
    let valid_batches = vec![make_batch::<TestValidBackend>(
        &valid_device,
        &[0, 0, 1, 1, 2, 2, 3, 3],
        &[0, 1, 1, 2, 2, 3, 3, 0],
        [2, 4],
    )];
    let env = TrainEnvironment {
        parallel_runtime: &parallel_runtime,
        parallel_config: &parallel_config,
        run_dir: &run_dir,
        run_name: "external-evaluator-loop-smoke",
        backend_name: "cpu",
        training: &training,
        resume_checkpoint_epoch: None,
        model_config: &model_config,
        device: &device,
        devices: &devices,
        train_dataset: None,
        valid_dataset: None,
        train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
        valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
        source_selection_dataset: None,
        summary_event_token_ids: None,
        neuron_scaling_slot: None,
        epochs: 1,
        total_steps: 1,
        valid_steps: 1,
    };
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        model_config.clone(),
        &device,
    ))
    .with_gradient_scale_schedule(&training, 1);
    let optimizer = tiny_language_optimizer(&training, &model_config, &device);

    let _trained = train_with_dynamic_neuron_scaling_scheduler(&env, model, optimizer, 1e-3)
        .expect("external evaluator scheduler train");

    let events = read_training_events(&run_dir);
    assert!(events.iter().all(|event| {
        event.get("type").and_then(|value| value.as_str()) != Some("validation_finished")
    }));
    assert!(
        events
            .iter()
            .all(|event| { event.get("split").and_then(|value| value.as_str()) != Some("valid") })
    );
    let checkpoint = events
        .iter()
        .find(|event| event.get("type").and_then(|value| value.as_str()) == Some("checkpoint"))
        .expect("checkpoint event");
    assert_eq!(
        checkpoint.get("promoted").and_then(|value| value.as_bool()),
        Some(false)
    );
    assert!(run_dir.join("checkpoint").join("model-1.bin").is_file());
}

#[test]
fn dynamic_scheduler_recovery_control_scales_continual_backprop_in_training_loop() {
    let dir = tempfile::tempdir().expect("tempdir");
    let weak_events = run_recovery_cbp_scale_smoke(
        &dir.path().join("weak"),
        "dynamic-recovery-cbp-weak",
        0.5,
        1.25,
    );
    let strong_events = run_recovery_cbp_scale_smoke(
        &dir.path().join("strong"),
        "dynamic-recovery-cbp-strong",
        2.0,
        0.75,
    );

    let weak_control_scale =
        latest_dynamics_control_scale(&weak_events).expect("weak recovery control");
    let strong_control_scale =
        latest_dynamics_control_scale(&strong_events).expect("strong recovery control");
    let strong_control_max_replacements = latest_dynamics_control_max_replacements(&strong_events)
        .expect("strong recovery max replacements");
    let weak_effective_scale =
        latest_continual_backprop_replacement_scale(&weak_events).expect("weak CBP telemetry");
    let strong_effective_scale =
        latest_continual_backprop_replacement_scale(&strong_events).expect("strong CBP telemetry");
    let strong_effective_max_replacements =
        latest_continual_backprop_max_replacements(&strong_events)
            .expect("strong CBP max replacements telemetry");

    assert_eq!(weak_control_scale, 0.5);
    assert_eq!(weak_effective_scale, 0.5);
    assert_eq!(strong_control_scale, 2.0);
    assert_eq!(strong_control_max_replacements, 3);
    assert_eq!(strong_effective_scale, 2.0);
    assert_eq!(strong_effective_max_replacements, 3);
    assert!(
        strong_effective_scale > weak_effective_scale,
        "strong recovery should increase realized CBP plasticity relative to weak baseline"
    );
}

fn run_recovery_cbp_scale_smoke(
    run_dir: &Path,
    run_name: &str,
    recovery_cbp_scale: f64,
    recovery_source_pressure: f64,
) -> Vec<serde_json::Value> {
    let parallel_config = burn_dragon_train::ParallelConfig::default();
    let parallel_runtime =
        resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");
    let device = burn::tensor::Device::<TestBackend>::default();
    TestBackend::seed(&device, 17);
    let valid_device = burn::tensor::Device::<TestValidBackend>::default();
    let mut training = tiny_training_hparams_with_epochs(2, None);
    training.continual_backprop.enabled = true;
    training.continual_backprop.warmup_steps = 0;
    training.continual_backprop.maturity_steps = 0;
    training.continual_backprop.cooldown_steps = 0;
    training.continual_backprop.sample_interval_steps = 1;
    training.continual_backprop.replace_interval_steps = 1;
    training.continual_backprop.replacement_rate = 0.1;
    training.continual_backprop.max_replacements_per_interval = 1;
    training.continual_backprop.lr_coupling = burn_dragon_train::ContinualBackpropLrCoupling::None;
    training.events.continual_backprop_every_steps = 1;
    training.events.degeneracy_probe_every_epochs = 1;
    training.events.degeneracy_probe_tokens = 8;
    training.gates.degeneracy_entropy_min_bits = 128.0;
    training.gates.degeneracy_max_probability_max = 2.0;
    training.gates.degeneracy_argmax_unique_min_fraction = 1.0;
    training.gates.degeneracy_distinct_2_min_fraction = 1.0;
    training.gates.degeneracy_repetition_max_fraction = 0.0;
    training.gates.degeneracy_period_2_max_fraction = 0.0;
    training.gates.degeneracy_period_3_max_fraction = 0.0;
    training.gates.degeneracy_period_2_to_16_max_fraction = 0.0;
    training.gates.degeneracy_period_2_to_64_max_fraction = 0.0;
    training.dynamics.soft_recovery_continual_backprop_scale = recovery_cbp_scale;
    training
        .dynamics
        .validation_recovery_continual_backprop_scale = recovery_cbp_scale;
    training.dynamics.hard_recovery_continual_backprop_scale = recovery_cbp_scale;
    training
        .dynamics
        .soft_recovery_max_replacements_per_interval = Some(3);
    training
        .dynamics
        .validation_recovery_max_replacements_per_interval = Some(3);
    training
        .dynamics
        .hard_recovery_max_replacements_per_interval = Some(3);
    training.dynamics.recovery_source_difficulty_pressure = recovery_source_pressure;
    let model_config = tiny_model_config();
    let devices = vec![device];
    let train_batches = vec![make_batch::<TestBackend>(
        &device,
        &[0, 1, 2, 3, 4, 5, 6, 7],
        &[1, 2, 3, 4, 5, 6, 7, 0],
        [2, 4],
    )];
    let valid_batches = vec![make_batch::<TestValidBackend>(
        &valid_device,
        &[0, 0, 1, 1, 2, 2, 3, 3],
        &[0, 1, 1, 2, 2, 3, 3, 0],
        [2, 4],
    )];
    let env = TrainEnvironment {
        parallel_runtime: &parallel_runtime,
        parallel_config: &parallel_config,
        run_dir,
        run_name,
        backend_name: "cpu",
        training: &training,
        resume_checkpoint_epoch: None,
        model_config: &model_config,
        device: &device,
        devices: &devices,
        train_dataset: None,
        valid_dataset: None,
        train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
        valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
        source_selection_dataset: None,
        summary_event_token_ids: None,
        neuron_scaling_slot: None,
        epochs: 2,
        total_steps: 2,
        valid_steps: 1,
    };
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        model_config.clone(),
        &device,
    ))
    .with_continual_backprop(&training.continual_backprop);
    let optimizer = tiny_language_optimizer(&training, &model_config, &device);

    let trained = train_with_dynamic_neuron_scaling_scheduler(&env, model, optimizer, 1e-3)
        .expect("dynamic recovery train");

    assert_eq!(trained.latent_total_capacity(), model_config.latent_total());
    let events = read_training_events(run_dir);
    assert!(
        events
            .iter()
            .any(|event| event.get("type").and_then(|value| value.as_str())
                == Some("dynamics_control")
                && event
                    .get("mode")
                    .and_then(|value| value.as_str())
                    .is_some_and(is_recovery_mode)
                && event
                    .get("continual_backprop_scale")
                    .and_then(|value| value.as_f64())
                    .is_some_and(|scale| (scale - recovery_cbp_scale).abs() < 1.0e-9)
                && event
                    .get("max_replacements_per_interval")
                    .and_then(|value| value.as_u64())
                    == Some(3)
                && event
                    .get("source_difficulty_pressure")
                    .and_then(|value| value.as_f64())
                    .is_some_and(|pressure| (pressure - recovery_source_pressure).abs() < 1.0e-9)),
        "training events should include a plasticity recovery control: {events:#?}"
    );
    assert!(
        events
            .iter()
            .any(|event| event.get("type").and_then(|value| value.as_str())
                == Some("continual_backprop")
                && event
                    .get("replacement_rate_scale")
                    .and_then(|value| value.as_f64())
                    .is_some_and(|scale| (scale - recovery_cbp_scale).abs() < 1.0e-6)
                && event
                    .get("effective_max_replacements_per_interval")
                    .and_then(|value| value.as_u64())
                    == Some(3)),
        "epoch after recovery should emit CBP telemetry using recovery scale: {events:#?}"
    );
    events
}

fn latest_dynamics_control_scale(events: &[serde_json::Value]) -> Option<f64> {
    events.iter().rev().find_map(|event| {
        (event.get("type").and_then(|value| value.as_str()) == Some("dynamics_control")
            && event
                .get("mode")
                .and_then(|value| value.as_str())
                .is_some_and(is_recovery_mode))
        .then(|| {
            event
                .get("continual_backprop_scale")
                .and_then(|value| value.as_f64())
        })
        .flatten()
    })
}

fn latest_continual_backprop_replacement_scale(events: &[serde_json::Value]) -> Option<f64> {
    events.iter().rev().find_map(|event| {
        (event.get("type").and_then(|value| value.as_str()) == Some("continual_backprop"))
            .then(|| {
                event
                    .get("replacement_rate_scale")
                    .and_then(|value| value.as_f64())
            })
            .flatten()
    })
}

fn latest_dynamics_control_max_replacements(events: &[serde_json::Value]) -> Option<usize> {
    events.iter().rev().find_map(|event| {
        (event.get("type").and_then(|value| value.as_str()) == Some("dynamics_control")
            && event
                .get("mode")
                .and_then(|value| value.as_str())
                .is_some_and(is_recovery_mode))
        .then(|| {
            event
                .get("max_replacements_per_interval")
                .and_then(|value| value.as_u64())
                .map(|value| value as usize)
        })
        .flatten()
    })
}

fn is_recovery_mode(mode: &str) -> bool {
    matches!(
        mode,
        "plasticity_recovery"
            | "validation_recovery"
            | "rollback_recovery"
            | "hard_recovery"
            | "hard_collapse"
    )
}

fn latest_continual_backprop_max_replacements(events: &[serde_json::Value]) -> Option<usize> {
    events.iter().rev().find_map(|event| {
        (event.get("type").and_then(|value| value.as_str()) == Some("continual_backprop"))
            .then(|| {
                event
                    .get("effective_max_replacements_per_interval")
                    .and_then(|value| value.as_u64())
                    .map(|value| value as usize)
            })
            .flatten()
    })
}

fn read_training_events(run_dir: &Path) -> Vec<serde_json::Value> {
    let path = run_dir.join("events").join("training_events.jsonl");
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("training event json"))
        .collect()
}

#[cfg(feature = "ddp")]
fn collective_test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[cfg(feature = "ddp")]
fn flatten_gradients_in_module_order<B, M>(module: &M, mut grads: GradientsParams) -> Vec<f32>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    #[derive(Default)]
    struct GradientCollector {
        values: Vec<f32>,
    }

    struct GradientCollectorVisitor<'a> {
        collector: &'a mut GradientCollector,
        grads: &'a mut GradientsParams,
    }

    impl<B: AutodiffBackend> burn::module::ModuleVisitor<B> for GradientCollectorVisitor<'_> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            let grad = self
                .grads
                .remove::<B::InnerBackend, D>(param.id)
                .unwrap_or_else(|| param.val().inner().zeros_like());
            let values = grad
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("gradient data");
            self.collector.values.extend(values);
        }
    }

    let mut collector = GradientCollector::default();
    let mut visitor = GradientCollectorVisitor {
        collector: &mut collector,
        grads: &mut grads,
    };
    module.visit(&mut visitor);
    collector.values
}

#[cfg(feature = "ddp")]
fn mean_abs_diff(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .sum::<f32>()
        / left.len().max(1) as f32
}

#[cfg(feature = "ddp")]
fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

#[cfg(feature = "ddp")]
fn stage_split_surrogate_gradients(
    split_model: LanguageTrainModel<TestBackend>,
    plan: &PipelinePlan,
    batch: SequenceBatch<TestBackend>,
) -> Vec<f32> {
    let [batch_size, _] = batch.inputs.shape().dims();
    let ranges = split_microbatch_ranges(batch_size, plan.microbatches).expect("ranges");
    let chunk_inputs = ranges
        .iter()
        .cloned()
        .map(|range| slice_batch_int(batch.inputs.clone(), range))
        .collect::<Vec<_>>();
    let chunk_targets = ranges
        .iter()
        .cloned()
        .map(|range| slice_batch_int(batch.targets.clone(), range))
        .collect::<Vec<_>>();
    let chunk_masks = ranges
        .iter()
        .cloned()
        .map(|range| {
            batch
                .summary_event_mask
                .clone()
                .map(|mask| slice_batch_int(mask, range))
        })
        .collect::<Vec<_>>();
    let mut chunk_states = (0..plan.microbatches)
        .map(|_| split_model.model.init_state())
        .collect::<Vec<_>>();
    let mut accumulator = GradientsAccumulator::new();
    let last_virtual_stage_id = plan.total_virtual_stages.saturating_sub(1);

    for microbatch_id in 0..plan.microbatches {
        let stage0_output = split_model
            .model
            .forward_language_pipeline_stage_with_state(
                split_model
                    .model
                    .begin_language_pipeline(chunk_inputs[microbatch_id].clone()),
                &mut chunk_states[microbatch_id],
                plan.assignment(0).layer_range.clone(),
                chunk_masks[microbatch_id].clone(),
            );
        let stage1_input = attach_pipeline_state_require_grad::<TestBackend>(
            detach_pipeline_state_to_inner(&stage0_output),
        );
        let stage1_input_for_grad = stage1_input.clone();
        let stage1_output = split_model
            .model
            .forward_language_pipeline_stage_with_state(
                stage1_input,
                &mut chunk_states[microbatch_id],
                plan.assignment(last_virtual_stage_id).layer_range.clone(),
                chunk_masks[microbatch_id].clone(),
            );
        let hidden = split_model
            .model
            .finish_language_pipeline_hidden_with_state(
                stage1_output,
                &mut chunk_states[microbatch_id],
            );
        let weight = ranges[microbatch_id].len() as f32 / batch_size as f32;
        let loss = split_model
            .model
            .language_loss_from_hidden(hidden, chunk_targets[microbatch_id].clone())
            .mul_scalar(weight);
        let mut stage1_grads = loss.backward();
        let grad_to_stage0 = pipeline_input_grad_state(&stage1_input_for_grad, &mut stage1_grads);
        accumulator.accumulate(
            &split_model,
            GradientsParams::from_grads(stage1_grads, &split_model),
        );

        let stage0_surrogate = pipeline_surrogate_loss(&stage0_output, grad_to_stage0);
        accumulator.accumulate(
            &split_model,
            GradientsParams::from_grads(stage0_surrogate.backward(), &split_model),
        );
    }

    flatten_gradients_in_module_order::<TestBackend, _>(&split_model, accumulator.grads())
}

#[cfg(feature = "ddp")]
#[test]
fn train_with_scheduler_executes_local_ddp_on_ndarray() {
    let dir = tempdir().expect("tempdir");
    let run_dir = dir.path().join("run");

    let parallel_config = burn_dragon_train::ParallelConfig {
        mode: ParallelismKind::Ddp,
        world_size: 2,
        data: burn_dragon_train::ParallelDataConfig {
            size: 2,
            ..Default::default()
        },
        ..Default::default()
    };
    let parallel_runtime =
        resolve_parallel_runtime(&parallel_config).expect("resolve local ddp runtime");

    let primary_device = burn::tensor::Device::<TestBackend>::default();
    let devices =
        resolve_training_devices::<TestBackend>(&parallel_runtime, &primary_device).unwrap();
    assert_eq!(devices.len(), 2, "expected 2 local replicas");

    let valid_device = burn::tensor::Device::<TestValidBackend>::default();
    let train_batches = vec![
        make_batch::<TestBackend>(
            &primary_device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 0],
            [2, 4],
        ),
        make_batch::<TestBackend>(
            &primary_device,
            &[7, 6, 5, 4, 3, 2, 1, 0],
            &[6, 5, 4, 3, 2, 1, 0, 7],
            [2, 4],
        ),
    ];
    let valid_batches = vec![make_batch::<TestValidBackend>(
        &valid_device,
        &[0, 0, 1, 1, 2, 2, 3, 3],
        &[0, 1, 1, 2, 2, 3, 3, 0],
        [2, 4],
    )];

    let training = tiny_training_hparams();
    let model_config = tiny_model_config();
    let env = TrainEnvironment {
        parallel_runtime: &parallel_runtime,
        parallel_config: &parallel_config,
        run_dir: &run_dir,
        run_name: "ddp-ndarray-smoke",
        backend_name: "cpu",
        training: &training,
        resume_checkpoint_epoch: None,
        model_config: &model_config,
        device: &primary_device,
        devices: &devices,
        train_dataset: None,
        valid_dataset: None,
        train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
        valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
        source_selection_dataset: None,
        summary_event_token_ids: None,
        neuron_scaling_slot: None,
        epochs: 1,
        total_steps: 2,
        valid_steps: 1,
    };

    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        model_config.clone(),
        &primary_device,
    ));
    let optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<TestBackend, LanguageTrainModel<TestBackend>>();

    let trained = train_with_scheduler(&env, model, optimizer, 1e-3).expect("ddp train");
    let probe = make_batch::<TestValidBackend>(
        &valid_device,
        &[1, 2, 3, 4, 4, 3, 2, 1],
        &[2, 3, 4, 5, 3, 2, 1, 0],
        [2, 4],
    );
    let loss =
        language_model_loss::<TestValidBackend>(trained.forward(probe.inputs), probe.targets)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("loss vec")[0];

    assert!(loss.is_finite(), "ddp smoke loss must be finite");
}

#[test]
fn train_with_scheduler_retains_best_valid_and_last_checkpoints() {
    let dir = tempfile::tempdir().expect("tempdir");
    let run_dir = dir.path().join("run");

    let parallel_config = burn_dragon_train::ParallelConfig::default();
    let parallel_runtime =
        resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");

    let primary_device = burn::tensor::Device::<TestBackend>::default();
    let valid_device = burn::tensor::Device::<TestValidBackend>::default();
    let train_batches = vec![
        make_batch::<TestBackend>(
            &primary_device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 0],
            [2, 4],
        ),
        make_batch::<TestBackend>(
            &primary_device,
            &[7, 6, 5, 4, 3, 2, 1, 0],
            &[6, 5, 4, 3, 2, 1, 0, 7],
            [2, 4],
        ),
    ];
    let valid_batches = vec![make_batch::<TestValidBackend>(
        &valid_device,
        &[0, 0, 1, 1, 2, 2, 3, 3],
        &[0, 1, 1, 2, 2, 3, 3, 0],
        [2, 4],
    )];

    let training = tiny_training_hparams_with_epochs(4, None);
    let model_config = tiny_model_config();
    let devices = vec![primary_device];
    let env = TrainEnvironment {
        parallel_runtime: &parallel_runtime,
        parallel_config: &parallel_config,
        run_dir: &run_dir,
        run_name: "single-retention-smoke",
        backend_name: "cpu",
        training: &training,
        resume_checkpoint_epoch: None,
        model_config: &model_config,
        device: &primary_device,
        devices: &devices,
        train_dataset: None,
        valid_dataset: None,
        train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
        valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
        source_selection_dataset: None,
        summary_event_token_ids: None,
        neuron_scaling_slot: None,
        epochs: 4,
        total_steps: 8,
        valid_steps: 1,
    };
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        model_config.clone(),
        &primary_device,
    ));
    let optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<TestBackend, LanguageTrainModel<TestBackend>>();

    let _trained = train_with_scheduler(&env, model, optimizer, 1e-3).expect("single-device train");

    let retained = retained_model_epochs(&run_dir);
    assert!(
        retained.contains(&3),
        "third epoch should be kept as recent"
    );
    assert!(retained.contains(&4), "last epoch should be kept as recent");
    assert!(
        retained.len() <= CHECKPOINT_KEEP_LAST + 1,
        "retention should keep the recent window plus at most one older best checkpoint"
    );
    assert!(
        retained.iter().all(|epoch| (1..=4).contains(epoch)),
        "retained epochs must come from completed checkpoints"
    );
}

#[cfg(feature = "ddp")]
#[test]
fn shard_bounds_evenly_distribute_remainder_steps() {
    assert_eq!(shard_bounds(5, 0, 2).expect("rank0"), (0, 3));
    assert_eq!(shard_bounds(5, 1, 2).expect("rank1"), (3, 5));
    assert!(shard_bounds(1, 1, 2).is_err());
}

#[cfg(feature = "ddp")]
#[test]
fn gradient_mean_matches_combined_batch_reference_in_module_order() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let config = tiny_model_config();
    let reference = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device));
    let combined_model = reference.clone();
    let shard_a_model = reference.clone();
    let shard_b_model = reference;

    let shard_a = make_batch::<TestBackend>(
        &device,
        &[0, 1, 2, 3, 4, 5, 6, 7],
        &[1, 2, 3, 4, 5, 6, 7, 0],
        [2, 4],
    );
    let shard_b = make_batch::<TestBackend>(
        &device,
        &[7, 6, 5, 4, 3, 2, 1, 0],
        &[6, 5, 4, 3, 2, 1, 0, 7],
        [2, 4],
    );
    let combined = make_batch::<TestBackend>(
        &device,
        &[0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0],
        &[1, 2, 3, 4, 5, 6, 7, 0, 6, 5, 4, 3, 2, 1, 0, 7],
        [4, 4],
    );

    let combined_grads = flatten_gradients_in_module_order::<TestBackend, _>(
        &combined_model,
        burn_train::TrainStep::step(&combined_model, combined).grads,
    );
    let shard_a_grads = flatten_gradients_in_module_order::<TestBackend, _>(
        &shard_a_model,
        burn_train::TrainStep::step(&shard_a_model, shard_a).grads,
    );
    let shard_b_grads = flatten_gradients_in_module_order::<TestBackend, _>(
        &shard_b_model,
        burn_train::TrainStep::step(&shard_b_model, shard_b).grads,
    );

    assert_eq!(combined_grads.len(), shard_a_grads.len());
    assert_eq!(combined_grads.len(), shard_b_grads.len());

    let averaged_shards = shard_a_grads
        .iter()
        .zip(shard_b_grads.iter())
        .map(|(lhs, rhs)| (lhs + rhs) * 0.5)
        .collect::<Vec<_>>();

    let mean_abs = mean_abs_diff(&combined_grads, &averaged_shards);
    let combined_norm = l2_norm(&combined_grads);
    let averaged_norm = l2_norm(&averaged_shards);

    assert!(
        mean_abs <= 1.0e-5,
        "combined-batch reference and mean rank-local gradients drifted: mean_abs_diff={mean_abs}"
    );
    assert!(
        (combined_norm - averaged_norm).abs() <= 1.0e-5,
        "gradient norms drifted: combined_norm={combined_norm} averaged_norm={averaged_norm}"
    );
}

#[cfg(feature = "ddp")]
#[test]
fn train_with_collective_scheduler_runs_single_rank_and_writes_checkpoint() {
    let _lock = collective_test_lock().lock().expect("collective lock");
    reset_collective::<TestValidBackend>();

    let dir = tempdir().expect("tempdir");
    let run_dir = dir.path().join("run");
    let parallel_config = burn_dragon_train::ParallelConfig {
        mode: ParallelismKind::Ddp,
        world_size: 1,
        data: burn_dragon_train::ParallelDataConfig {
            size: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let parallel_runtime = ParallelRuntime {
        mode: ParallelismKind::Ddp,
        world_size: 1,
        global_rank: 0,
        local_rank: 0,
        data_parallel_size: 1,
        local_data_parallel_size: 1,
        tensor_parallel_size: 1,
        process_group_launch: false,
    };

    let primary_device = burn::tensor::Device::<TestBackend>::default();
    let valid_device = burn::tensor::Device::<TestValidBackend>::default();
    let train_batches = vec![
        make_batch::<TestBackend>(
            &primary_device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 0],
            [2, 4],
        ),
        make_batch::<TestBackend>(
            &primary_device,
            &[7, 6, 5, 4, 3, 2, 1, 0],
            &[6, 5, 4, 3, 2, 1, 0, 7],
            [2, 4],
        ),
    ];
    let valid_batches = vec![make_batch::<TestValidBackend>(
        &valid_device,
        &[0, 0, 1, 1, 2, 2, 3, 3],
        &[0, 1, 1, 2, 2, 3, 3, 0],
        [2, 4],
    )];

    let training = tiny_training_hparams();
    let model_config = tiny_model_config();
    let devices = vec![primary_device.clone()];
    let env = TrainEnvironment {
        parallel_runtime: &parallel_runtime,
        parallel_config: &parallel_config,
        run_dir: &run_dir,
        run_name: "collective-single-rank",
        backend_name: "cpu",
        training: &training,
        resume_checkpoint_epoch: None,
        model_config: &model_config,
        device: &primary_device,
        devices: &devices,
        train_dataset: None,
        valid_dataset: None,
        train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
        valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
        source_selection_dataset: None,
        summary_event_token_ids: None,
        neuron_scaling_slot: None,
        epochs: 1,
        total_steps: 2,
        valid_steps: 1,
    };
    let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        model_config.clone(),
        &primary_device,
    ));
    let optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<TestBackend, LanguageTrainModel<TestBackend>>();
    let collective =
        resolve_collective_config(&parallel_runtime, &parallel_config).expect("collective");

    let trained =
        train_with_collective_scheduler(&env, model, optimizer, 1e-3, collective, 0.into())
            .expect("collective train");
    let probe = make_batch::<TestValidBackend>(
        &valid_device,
        &[1, 2, 3, 4, 4, 3, 2, 1],
        &[2, 3, 4, 5, 3, 2, 1, 0],
        [2, 4],
    );
    let loss =
        language_model_loss::<TestValidBackend>(trained.forward(probe.inputs), probe.targets)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("loss vec")[0];

    assert!(loss.is_finite());
    assert!(run_dir.join("checkpoint").join("model-1.bin").is_file());

    reset_collective::<TestValidBackend>();
}

#[cfg(feature = "ddp")]
#[test]
fn train_with_collective_scheduler_resumes_from_checkpoint_family() {
    let _lock = collective_test_lock().lock().expect("collective lock");
    reset_collective::<TestValidBackend>();

    let dir = tempdir().expect("tempdir");
    let run_dir = dir.path().join("run");
    let parallel_config = burn_dragon_train::ParallelConfig {
        mode: ParallelismKind::Ddp,
        world_size: 1,
        data: burn_dragon_train::ParallelDataConfig {
            size: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let parallel_runtime = ParallelRuntime {
        mode: ParallelismKind::Ddp,
        world_size: 1,
        global_rank: 0,
        local_rank: 0,
        data_parallel_size: 1,
        local_data_parallel_size: 1,
        tensor_parallel_size: 1,
        process_group_launch: false,
    };

    let primary_device = burn::tensor::Device::<TestBackend>::default();
    let valid_device = burn::tensor::Device::<TestValidBackend>::default();
    let train_loader: Arc<dyn DataLoader<TestBackend, SequenceBatch<TestBackend>>> =
        Arc::new(StaticSequenceLoader::new(vec![
            make_batch::<TestBackend>(
                &primary_device,
                &[0, 1, 2, 3, 4, 5, 6, 7],
                &[1, 2, 3, 4, 5, 6, 7, 0],
                [2, 4],
            ),
            make_batch::<TestBackend>(
                &primary_device,
                &[7, 6, 5, 4, 3, 2, 1, 0],
                &[6, 5, 4, 3, 2, 1, 0, 7],
                [2, 4],
            ),
        ]));
    let valid_loader: Arc<dyn DataLoader<TestValidBackend, SequenceBatch<TestValidBackend>>> =
        Arc::new(StaticSequenceLoader::new(vec![make_batch::<
            TestValidBackend,
        >(
            &valid_device,
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 1, 2, 2, 3, 3, 0],
            [2, 4],
        )]));
    let devices = vec![primary_device.clone()];
    let model_config = tiny_model_config();
    let collective =
        resolve_collective_config(&parallel_runtime, &parallel_config).expect("collective");

    let training_first = tiny_training_hparams_with_epochs(1, None);
    let env_first = TrainEnvironment {
        parallel_runtime: &parallel_runtime,
        parallel_config: &parallel_config,
        run_dir: &run_dir,
        run_name: "collective-resume",
        backend_name: "cpu",
        training: &training_first,
        resume_checkpoint_epoch: None,
        model_config: &model_config,
        device: &primary_device,
        devices: &devices,
        train_dataset: None,
        valid_dataset: None,
        train_loader: Arc::clone(&train_loader),
        valid_loader: Arc::clone(&valid_loader),
        source_selection_dataset: None,
        summary_event_token_ids: None,
        neuron_scaling_slot: None,
        epochs: 1,
        total_steps: 2,
        valid_steps: 1,
    };
    let model_first = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        model_config.clone(),
        &primary_device,
    ));
    let optimizer_first = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<TestBackend, LanguageTrainModel<TestBackend>>();
    train_with_collective_scheduler(
        &env_first,
        model_first,
        optimizer_first,
        1e-3,
        collective.clone(),
        0.into(),
    )
    .expect("first collective train");
    assert!(run_dir.join("checkpoint").join("model-1.bin").is_file());

    reset_collective::<TestValidBackend>();

    let training_resume = tiny_training_hparams_with_epochs(2, Some(1));
    let env_resume = TrainEnvironment {
        parallel_runtime: &parallel_runtime,
        parallel_config: &parallel_config,
        run_dir: &run_dir,
        run_name: "collective-resume",
        backend_name: "cpu",
        training: &training_resume,
        resume_checkpoint_epoch: Some(1),
        model_config: &model_config,
        device: &primary_device,
        devices: &devices,
        train_dataset: None,
        valid_dataset: None,
        train_loader,
        valid_loader,
        source_selection_dataset: None,
        summary_event_token_ids: None,
        neuron_scaling_slot: None,
        epochs: 2,
        total_steps: 4,
        valid_steps: 1,
    };
    let model_resume = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
        model_config.clone(),
        &primary_device,
    ));
    let optimizer_resume = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<TestBackend, LanguageTrainModel<TestBackend>>();
    let resumed = train_with_collective_scheduler(
        &env_resume,
        model_resume,
        optimizer_resume,
        1e-3,
        collective,
        0.into(),
    )
    .expect("resumed collective train");

    let probe = make_batch::<TestValidBackend>(
        &valid_device,
        &[1, 2, 3, 4, 4, 3, 2, 1],
        &[2, 3, 4, 5, 3, 2, 1, 0],
        [2, 4],
    );
    let loss =
        language_model_loss::<TestValidBackend>(resumed.forward(probe.inputs), probe.targets)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("loss vec")[0];

    assert!(loss.is_finite());
    assert!(run_dir.join("checkpoint").join("model-2.bin").is_file());

    reset_collective::<TestValidBackend>();
}

#[cfg(feature = "ddp")]
#[test]
fn pipeline_stage_surrogate_backward_matches_full_pipeline_gradients() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut config = tiny_model_config();
    config.n_layer = 2;
    let pipeline = burn_dragon_train::ParallelPipelineConfig {
        enabled: true,
        stage_count: 2,
        virtual_stages_per_rank: 1,
        schedule: burn_dragon_train::PipelineScheduleKind::Interleaved1f1b,
        microbatches: 2,
        ..Default::default()
    };
    let plan = build_pipeline_plan(config.n_layer, &pipeline).expect("plan");
    let reference_model =
        LanguageTrainModel::new(DragonModel::<TestBackend>::new(config.clone(), &device))
            .with_pipeline_plan(Some(plan.clone()));
    let split_model = reference_model.clone();

    let batch = make_batch::<TestBackend>(
        &device,
        &[0, 1, 2, 3, 7, 6, 5, 4],
        &[1, 2, 3, 4, 6, 5, 4, 3],
        [2, 4],
    );
    let reference_grads = flatten_gradients_in_module_order::<TestBackend, _>(
        &reference_model,
        burn_train::TrainStep::step(&reference_model, batch.clone()).grads,
    );
    let split_grads = stage_split_surrogate_gradients(split_model, &plan, batch);
    let mean_abs = mean_abs_diff(&reference_grads, &split_grads);
    let reference_norm = l2_norm(&reference_grads);
    let split_norm = l2_norm(&split_grads);

    assert!(
        mean_abs <= 1.0e-5,
        "surrogate split pipeline gradients drifted from full pipeline reference: mean_abs_diff={mean_abs}"
    );
    assert!(
        (reference_norm - split_norm).abs() <= 1.0e-5,
        "split pipeline gradient norm drifted from reference: reference_norm={reference_norm} split_norm={split_norm}"
    );
}

#[cfg(feature = "ddp")]
#[test]
fn pipeline_stage_surrogate_mean_across_replicas_matches_full_batch_gradients() {
    let device = burn::tensor::Device::<TestBackend>::default();
    let mut config = tiny_model_config();
    config.n_layer = 2;
    let pipeline = burn_dragon_train::ParallelPipelineConfig {
        enabled: true,
        stage_count: 2,
        virtual_stages_per_rank: 1,
        schedule: burn_dragon_train::PipelineScheduleKind::Interleaved1f1b,
        microbatches: 2,
        ..Default::default()
    };
    let plan = build_pipeline_plan(config.n_layer, &pipeline).expect("plan");
    let reference_model =
        LanguageTrainModel::new(DragonModel::<TestBackend>::new(config.clone(), &device))
            .with_pipeline_plan(Some(plan.clone()));

    let replica_a = make_batch::<TestBackend>(
        &device,
        &[0, 1, 2, 3, 4, 5, 6, 7],
        &[1, 2, 3, 4, 5, 6, 7, 0],
        [2, 4],
    );
    let replica_b = make_batch::<TestBackend>(
        &device,
        &[7, 6, 5, 4, 3, 2, 1, 0],
        &[6, 5, 4, 3, 2, 1, 0, 7],
        [2, 4],
    );
    let combined = make_batch::<TestBackend>(
        &device,
        &[0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0],
        &[1, 2, 3, 4, 5, 6, 7, 0, 6, 5, 4, 3, 2, 1, 0, 7],
        [4, 4],
    );

    let combined_grads = flatten_gradients_in_module_order::<TestBackend, _>(
        &reference_model,
        burn_train::TrainStep::step(&reference_model, combined).grads,
    );
    let replica_a_grads =
        stage_split_surrogate_gradients(reference_model.clone(), &plan, replica_a);
    let replica_b_grads =
        stage_split_surrogate_gradients(reference_model.clone(), &plan, replica_b);
    let averaged_grads = replica_a_grads
        .iter()
        .zip(replica_b_grads.iter())
        .map(|(lhs, rhs)| (lhs + rhs) * 0.5)
        .collect::<Vec<_>>();

    let mean_abs = mean_abs_diff(&combined_grads, &averaged_grads);
    let combined_norm = l2_norm(&combined_grads);
    let averaged_norm = l2_norm(&averaged_grads);

    assert!(
        mean_abs <= 1.0e-5,
        "replica-averaged split pipeline gradients drifted from combined-batch reference: mean_abs_diff={mean_abs}"
    );
    assert!(
        (combined_norm - averaged_norm).abs() <= 1.0e-5,
        "replica-averaged split pipeline gradient norm drifted from combined-batch reference: combined_norm={combined_norm} averaged_norm={averaged_norm}"
    );
}

#[cfg(feature = "ddp")]
#[test]
fn process_group_peer_id_uses_global_rank() {
    let runtime = ParallelRuntime {
        mode: ParallelismKind::Ddp,
        world_size: 4,
        global_rank: 3,
        local_rank: 1,
        data_parallel_size: 4,
        local_data_parallel_size: 1,
        tensor_parallel_size: 1,
        process_group_launch: true,
    };

    assert_eq!(process_group_peer_id(&runtime), 3usize.into());
}

#[cfg(feature = "ddp")]
#[test]
fn process_group_data_shard_uses_data_parallel_rank_when_pipeline_enabled() {
    let runtime = ParallelRuntime {
        mode: ParallelismKind::Ddp,
        world_size: 4,
        global_rank: 3,
        local_rank: 1,
        data_parallel_size: 2,
        local_data_parallel_size: 1,
        tensor_parallel_size: 1,
        process_group_launch: true,
    };
    let config = burn_dragon_train::ParallelConfig {
        mode: ParallelismKind::Ddp,
        world_size: 4,
        data: burn_dragon_train::ParallelDataConfig {
            size: 2,
            ..Default::default()
        },
        pipeline: burn_dragon_train::ParallelPipelineConfig {
            enabled: true,
            stage_count: 2,
            virtual_stages_per_rank: 1,
            ..Default::default()
        },
        ..Default::default()
    };

    let (shard_index, shard_count, assignment, layout) =
        process_group_data_shard(&runtime, &config).expect("pipeline shard");

    assert_eq!(shard_index, 1);
    assert_eq!(shard_count, 2);
    let assignment = assignment.expect("rank assignment");
    let layout = layout.expect("layout");
    assert_eq!(assignment.pipeline_stage_id, 1);
    assert_eq!(assignment.data_parallel_rank, 1);
    assert_eq!(assignment.pipeline_group_ranks, vec![2, 3]);
    assert_eq!(assignment.data_parallel_group_ranks, vec![1, 3]);
    assert_eq!(
        layout.summary(),
        "pipeline_layout=replica_major stage_count=2 virtual_stages_per_rank=1 data_parallel_size=2 world_size=4"
    );
}

#[cfg(feature = "ddp")]
#[test]
fn fresh_models_use_random_param_ids_but_stable_module_traversal_shapes() {
    #[derive(Default)]
    struct ShapeCollector {
        shapes: Vec<Vec<usize>>,
    }

    impl<B: BackendTrait> burn::module::ModuleVisitor<B> for ShapeCollector {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            self.shapes.push(param.val().shape().dims::<D>().into());
        }
    }

    let device = burn::tensor::Device::<TestBackend>::default();
    let config = tiny_model_config();
    let model_a = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config.clone(), &device));
    let model_b = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device));

    let ids_a = list_param_ids(&model_a);
    let ids_b = list_param_ids(&model_b);
    let mut shapes_a = ShapeCollector::default();
    let mut shapes_b = ShapeCollector::default();
    model_a.visit(&mut shapes_a);
    model_b.visit(&mut shapes_b);

    assert_eq!(ids_a.len(), ids_b.len());
    assert_ne!(
        ids_a, ids_b,
        "fresh models should not rely on matching ParamIds"
    );
    assert_eq!(shapes_a.shapes, shapes_b.shapes);
}
