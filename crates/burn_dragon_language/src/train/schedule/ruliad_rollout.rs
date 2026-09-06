//! Ruliad rollout promotion gates and generation budgets.

use super::*;

const RULIAD_POLICY_CERTIFICATE_BUDGET_MULTIPLIER: usize = 4;

fn ruliad_policy_rollout_limit(
    configured_max_steps: usize,
    certificate_steps: usize,
) -> Result<usize> {
    let oracle_steps = certificate_steps.max(1);
    let automatic_limit = oracle_steps
        .saturating_mul(RULIAD_POLICY_CERTIFICATE_BUDGET_MULTIPLIER)
        .max(1);
    if configured_max_steps == 0 {
        return Ok(automatic_limit);
    }
    if configured_max_steps < oracle_steps {
        return Err(anyhow!(
            "Ruliad policy probe max_steps={configured_max_steps} is shorter than the verifier certificate ({oracle_steps} steps); use max_steps=0 for an automatic proof-relative budget"
        ));
    }
    Ok(configured_max_steps.min(automatic_limit))
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct RuliadPolicyPromotionGateStatus {
    pub(super) passed: bool,
    pub(super) reasons: Vec<String>,
}

pub(super) fn ruliad_policy_promotion_gate_status(
    summary: RuliadPolicyRolloutProbeSummary,
    gate: crate::config::RuliadPolicyPromotionGateConfig,
) -> RuliadPolicyPromotionGateStatus {
    if !gate.enabled {
        return RuliadPolicyPromotionGateStatus {
            passed: true,
            reasons: Vec::new(),
        };
    }
    let attempted_actions = summary
        .valid_actions
        .saturating_add(summary.invalid_actions);
    let solve_rate = ratio_usize(summary.solved, summary.items);
    let goal_completion_rate = ratio_usize(summary.solved_goals, summary.total_goals);
    let valid_action_rate = ratio_usize(summary.valid_actions, attempted_actions);
    let invalid_action_rate = ratio_usize(summary.invalid_actions, attempted_actions);
    let repeated_state_rate = ratio_usize(summary.repeated_states, summary.valid_actions);
    let backtrack_rate = ratio_usize(
        summary.backtracks,
        summary.valid_actions.saturating_add(summary.backtracks),
    );
    let mut reasons = Vec::new();
    if summary.items < gate.minimum_items {
        reasons.push(format!("items={}<{}", summary.items, gate.minimum_items));
    }
    if solve_rate < gate.minimum_solve_rate {
        reasons.push(format!(
            "solve_rate={solve_rate:.3}<{}",
            gate.minimum_solve_rate
        ));
    }
    if goal_completion_rate < gate.minimum_goal_completion_rate {
        reasons.push(format!(
            "goal_completion_rate={goal_completion_rate:.3}<{}",
            gate.minimum_goal_completion_rate
        ));
    }
    if valid_action_rate < gate.minimum_valid_action_rate {
        reasons.push(format!(
            "valid_action_rate={valid_action_rate:.3}<{}",
            gate.minimum_valid_action_rate
        ));
    }
    if invalid_action_rate > gate.maximum_invalid_action_rate {
        reasons.push(format!(
            "invalid_action_rate={invalid_action_rate:.3}>{}",
            gate.maximum_invalid_action_rate
        ));
    }
    if repeated_state_rate > gate.maximum_repeated_state_rate {
        reasons.push(format!(
            "repeated_state_rate={repeated_state_rate:.3}>{}",
            gate.maximum_repeated_state_rate
        ));
    }
    if backtrack_rate > gate.maximum_backtrack_rate {
        reasons.push(format!(
            "backtrack_rate={backtrack_rate:.3}>{}",
            gate.maximum_backtrack_rate
        ));
    }
    RuliadPolicyPromotionGateStatus {
        passed: reasons.is_empty(),
        reasons,
    }
}

pub(super) fn ruliad_policy_context_binding_gate_status(
    constrained: Option<&RuliadCorrectnessConstrainedPolicyResult>,
    gate: crate::config::RuliadPolicyPromotionGateConfig,
) -> RuliadPolicyPromotionGateStatus {
    if !gate.enabled || !gate.require_context_binding {
        return RuliadPolicyPromotionGateStatus {
            passed: true,
            reasons: Vec::new(),
        };
    }
    let Some(constrained) = constrained else {
        return RuliadPolicyPromotionGateStatus {
            passed: false,
            reasons: vec!["context_binding_probe=missing".to_string()],
        };
    };
    let summary = &constrained.summary;
    let context_change_rate = ratio_usize(
        summary.context_swap_top1_changes,
        summary.context_swap_items,
    );
    let context_probability_drop = summary.context_swap_equivalent_probability_drop_sum
        / summary.context_swap_items.max(1) as f64;
    let target_change_rate = ratio_usize(
        summary.counterfactual_target_top1_changes,
        summary.counterfactual_target_items,
    );
    let target_probability_gain = summary.counterfactual_target_equivalent_probability_gain_sum
        / summary.counterfactual_target_items.max(1) as f64;
    let mut reasons = Vec::new();
    if summary.context_swap_items < gate.minimum_conditioning_items {
        reasons.push(format!(
            "context_swap_items={}<{}",
            summary.context_swap_items, gate.minimum_conditioning_items
        ));
    }
    if summary.counterfactual_target_items < gate.minimum_conditioning_items {
        reasons.push(format!(
            "counterfactual_target_items={}<{}",
            summary.counterfactual_target_items, gate.minimum_conditioning_items
        ));
    }
    if context_change_rate < gate.minimum_context_swap_top1_change_rate {
        reasons.push(format!(
            "context_swap_top1_change_rate={context_change_rate:.3}<{}",
            gate.minimum_context_swap_top1_change_rate
        ));
    }
    if context_probability_drop < gate.minimum_context_swap_equivalent_probability_drop {
        reasons.push(format!(
            "context_swap_equivalent_probability_drop={context_probability_drop:.6}<{}",
            gate.minimum_context_swap_equivalent_probability_drop
        ));
    }
    if target_change_rate < gate.minimum_counterfactual_target_top1_change_rate {
        reasons.push(format!(
            "counterfactual_target_top1_change_rate={target_change_rate:.3}<{}",
            gate.minimum_counterfactual_target_top1_change_rate
        ));
    }
    if target_probability_gain < gate.minimum_counterfactual_target_equivalent_probability_gain {
        reasons.push(format!(
            "counterfactual_target_equivalent_probability_gain={target_probability_gain:.6}<{}",
            gate.minimum_counterfactual_target_equivalent_probability_gain
        ));
    }
    RuliadPolicyPromotionGateStatus {
        passed: reasons.is_empty(),
        reasons,
    }
}

pub(super) fn ruliad_policy_capability_bucket(
    difficulty_level: usize,
    summary: RuliadPolicyRolloutProbeSummary,
) -> CapabilityProbeGroupMetric {
    let attempted_actions = summary
        .valid_actions
        .saturating_add(summary.invalid_actions);
    let solve_rate = ratio_usize(summary.solved, summary.items);
    let goal_completion_rate = ratio_usize(summary.solved_goals, summary.total_goals);
    let invalid_action_rate = ratio_usize(summary.invalid_actions, attempted_actions);
    CapabilityProbeGroupMetric {
        label: format!("difficulty:d{difficulty_level}"),
        item_count: summary.items,
        exact_rate: solve_rate,
        semantic_rate: solve_rate,
        verifier_rate: solve_rate,
        partial_credit_rate: goal_completion_rate,
        schema_valid_wrong_rate: invalid_action_rate,
        malformed_rate: invalid_action_rate,
        missing_rate: 0.0,
        mean_partial_progress: goal_completion_rate,
        answer_field_accuracy: ratio_usize(summary.valid_actions, attempted_actions),
        answer_field_coverage: 1.0,
        answer_termination_rate: 1.0,
    }
}

pub(super) fn ruliad_policy_capability_feedback(
    result: &RuliadPolicyRolloutProbeResult,
) -> Vec<burn_dragon_universality::RuliadCapabilityFeedback> {
    result
        .source_summaries
        .iter()
        .map(|(group_label, summary)| {
            let attempted_actions = summary
                .valid_actions
                .saturating_add(summary.invalid_actions);
            let top1_expert_rate = ratio_usize(summary.top1_expert_actions, summary.scored_states)
                .clamp(0.0, 1.0) as f32;
            let malformed_rate =
                ratio_usize(summary.invalid_actions, attempted_actions).clamp(0.0, 1.0) as f32;
            let missing_rate = if summary.scored_states == 0 {
                1.0
            } else {
                ratio_usize(summary.frontier_exhaustions, summary.items).clamp(0.0, 1.0) as f32
            };
            burn_dragon_universality::RuliadCapabilityFeedback {
                group_label: group_label.clone(),
                item_count: summary.scored_states.max(summary.items),
                verifier_rate: top1_expert_rate,
                partial_credit_rate: ratio_usize(summary.solved_goals, summary.total_goals)
                    .clamp(0.0, 1.0) as f32,
                schema_valid_wrong_rate: if summary.scored_states == 0 {
                    0.0
                } else {
                    1.0 - top1_expert_rate
                },
                malformed_rate,
                missing_rate,
                completion_health_rate: ((1.0 - malformed_rate) * (1.0 - missing_rate))
                    .clamp(0.0, 1.0),
            }
        })
        .collect()
}

pub(super) fn merge_ruliad_policy_capability_feedback(
    mut free_run_feedback: Vec<burn_dragon_universality::RuliadCapabilityFeedback>,
    policy_probe_enabled: bool,
    policy_result: Option<&RuliadPolicyRolloutProbeResult>,
) -> Vec<burn_dragon_universality::RuliadCapabilityFeedback> {
    if policy_probe_enabled {
        free_run_feedback
            .retain(|feedback| !is_semantic_proof_action_source_feedback(&feedback.group_label));
    }
    if let Some(policy_result) = policy_result {
        free_run_feedback.extend(ruliad_policy_capability_feedback(policy_result));
    }
    free_run_feedback
}

pub(super) fn is_semantic_proof_action_source_feedback(group_label: &str) -> bool {
    group_label.starts_with("source:formal_proof:select_proof_action@d")
        && group_label.ends_with("#proof_action_step")
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_ruliad_policy_rollout_probe<B>(
    run_name: &str,
    dataset: &Dataset,
    model: &LanguageTrainModel<B>,
    epoch: usize,
    absolute_step: usize,
    device: &B::Device,
    training: &TrainingHyperparameters,
    probe_items: &[crate::dataset::RuliadValidationProbeItem],
    bus: &TrainingEventBus,
) -> Result<RuliadPolicyRolloutProbeResult>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    let config = training.ruliad_policy_probe;
    eprintln!(
        "ruliad policy probe start run={run_name} epoch={epoch} requested_items={} max_steps={} beam_width={} scoring={:?} scoring_batch_rows={} scoring_pipeline_depth={}",
        config.items,
        if config.max_steps == 0 {
            "auto(certificate_x4)".to_string()
        } else {
            config.max_steps.to_string()
        },
        config.beam_width,
        config.scoring,
        config.scoring_batch_rows,
        config.scoring_pipeline_depth,
    );
    let probe_started = burn_dragon_time::Instant::now();
    let mut summary = RuliadPolicyRolloutProbeSummary::default();
    let mut seen_problems = BTreeSet::new();
    let mut examples = Vec::new();
    let mut difficulty_summaries = BTreeMap::<usize, RuliadPolicyRolloutProbeSummary>::new();
    let mut source_summaries = BTreeMap::<String, RuliadPolicyRolloutProbeSummary>::new();
    let mut scoring_summary = RuliadPolicyScoringSummary::default();
    let mut searches = Vec::<RuliadPolicyProbeSearch>::new();
    for probe in probe_items {
        if searches.len() >= config.items {
            break;
        }
        let Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
            problem,
            certificate,
            action_answer_contract,
            task: burn_dragon_universality::RuliadTaskKind::SelectProofAction,
            ..
        }) = probe.item.spec.as_ref()
        else {
            continue;
        };
        let difficulty_level = probe.item.difficulty_level.unwrap_or(0);
        if !seen_problems.insert(certificate.problem_hash.clone()) {
            continue;
        }
        let initial_state = burn_dragon_universality::ruliad::RuliadProofPolicyState::new(problem);
        let initial_node = RuliadPolicyBeamNode {
            state: initial_state,
            log_probability: 0.0,
            steps: 0,
        };
        let mut best_state_scores = BTreeMap::<String, f32>::new();
        best_state_scores.insert(initial_node.state.canonical_state_key(problem)?, 0.0);
        let rollout_limit =
            ruliad_policy_rollout_limit(config.max_steps, certificate.step_count())?;
        let item_summary = RuliadPolicyRolloutProbeSummary {
            items: 1,
            total_goals: initial_node.state.total_goals(),
            ..Default::default()
        };
        searches.push(RuliadPolicyProbeSearch {
            problem: problem.clone(),
            certificate_hash: certificate.problem_hash.clone(),
            answer_contract: *action_answer_contract,
            difficulty_level,
            rollout_limit,
            beam: vec![initial_node.clone()],
            best_node: initial_node,
            best_state_scores,
            summary: item_summary,
            done: false,
        });
    }

    let maximum_depth = searches
        .iter()
        .map(|search| search.rollout_limit)
        .max()
        .unwrap_or_default();
    for depth in 0..maximum_depth {
        if searches.iter().all(|search| search.done) {
            break;
        }
        let prepare_started = burn_dragon_time::Instant::now();
        let expansion_groups = searches
            .par_iter_mut()
            .enumerate()
            .map(|(search_index, search)| {
                prepare_ruliad_policy_search_expansions(
                    dataset,
                    config,
                    search_index,
                    search,
                    depth,
                )
            })
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let expansions = expansion_groups.into_iter().flatten().collect::<Vec<_>>();
        scoring_summary.cpu_prepare_ms += prepare_started.elapsed().as_micros() as f64 / 1_000.0;
        if expansions.is_empty() {
            continue;
        }

        let scoring_started = burn_dragon_time::Instant::now();
        let mut children = (0..searches.len())
            .map(|_| BTreeMap::<String, RuliadPolicyBeamNode>::new())
            .collect::<Vec<_>>();
        let scoring_presentations = expansions
            .iter()
            .enumerate()
            .flat_map(|(expansion_index, expansion)| {
                expansion
                    .presentations
                    .iter()
                    .map(move |presentation| (expansion_index, presentation))
            })
            .collect::<Vec<_>>();
        let scoring_sequence_lengths = scoring_presentations
            .iter()
            .map(|(_, presentation)| {
                presentation.prompt_tokens.len().saturating_add(
                    presentation
                        .candidate_tokens
                        .first()
                        .map(Vec::len)
                        .unwrap_or_default(),
                )
            })
            .collect::<Vec<_>>();
        let mut scoring_order = (0..scoring_presentations.len()).collect::<Vec<_>>();
        scoring_order.sort_by_key(|index| (scoring_sequence_lengths[*index], *index));
        let sorted_sequence_lengths = scoring_order
            .iter()
            .map(|index| scoring_sequence_lengths[*index])
            .collect::<Vec<_>>();
        let mut scores_by_expansion = (0..expansions.len())
            .map(|_| Vec::<(usize, Vec<f32>)>::new())
            .collect::<Vec<_>>();
        let mut pending = std::collections::VecDeque::<(
            Vec<usize>,
            crate::train::ruliad_policy::DeferredProofActionCompletionScores<B>,
        )>::new();
        let mut start = 0usize;
        while start < scoring_order.len() {
            let end = bounded_padded_batch_end(
                &sorted_sequence_lengths,
                start,
                config.scoring_batch_rows,
                config.scoring_token_budget,
            );
            let batch_indices = &scoring_order[start..end];
            let prompts = batch_indices
                .iter()
                .map(|index| scoring_presentations[*index].1.prompt_tokens.clone())
                .collect::<Vec<_>>();
            let candidates = batch_indices
                .iter()
                .map(|index| scoring_presentations[*index].1.candidate_tokens.clone())
                .collect::<Vec<_>>();
            scoring_summary.record_batch(&sorted_sequence_lengths[start..end]);
            let answer_contract = batch_indices
                .first()
                .map(|index| scoring_presentations[*index].1.answer_contract)
                .ok_or_else(|| anyhow!("proof-action scoring batch is empty"))?;
            if batch_indices
                .iter()
                .any(|index| scoring_presentations[*index].1.answer_contract != answer_contract)
            {
                return Err(anyhow!(
                    "proof-action scoring batch mixes incompatible answer contracts"
                ));
            }
            let deferred =
                crate::train::ruliad_policy::enqueue_proof_action_scores_batch_with_normalization(
                    &model.model,
                    &prompts,
                    &candidates,
                    answer_contract,
                    config.scoring,
                    config.normalization,
                    device,
                )?;
            pending.push_back((batch_indices.to_vec(), deferred));
            scoring_summary.record_pipeline_depth(pending.len());
            if pending.len() >= config.scoring_pipeline_depth.max(1)
                && let Some((indices, deferred)) = pending.pop_front()
            {
                for (index, scores) in indices.into_iter().zip(deferred.resolve()?) {
                    let (expansion_index, presentation) = scoring_presentations[index];
                    scores_by_expansion[expansion_index].push((presentation.rotation, scores));
                }
            }
            start = end;
        }
        while let Some((indices, deferred)) = pending.pop_front() {
            for (index, scores) in indices.into_iter().zip(deferred.resolve()?) {
                let (expansion_index, presentation) = scoring_presentations[index];
                scores_by_expansion[expansion_index].push((presentation.rotation, scores));
            }
        }
        scoring_summary.model_scoring_ms += scoring_started.elapsed().as_micros() as f64 / 1_000.0;

        let transition_started = burn_dragon_time::Instant::now();
        drop(scoring_presentations);
        let mut scored_by_search = (0..searches.len())
            .map(|_| Vec::<RuliadPolicyScoredExpansion>::new())
            .collect::<Vec<_>>();
        for (expansion, presentation_scores) in expansions.into_iter().zip(scores_by_expansion) {
            if presentation_scores.is_empty() {
                continue;
            }
            let scores = crate::train::ruliad_policy::semantic_action_log_probs(
                &presentation_scores,
                expansion.actions.candidates.len(),
            )?;
            scored_by_search[expansion.search_index]
                .push(RuliadPolicyScoredExpansion { expansion, scores });
        }
        searches
            .par_iter_mut()
            .zip(children.par_iter_mut())
            .zip(scored_by_search.into_par_iter())
            .try_for_each(|((search, children), scored_expansions)| {
                apply_ruliad_policy_scored_expansions(
                    search,
                    children,
                    scored_expansions,
                    config,
                    depth,
                )
            })?;
        scoring_summary.cpu_transition_ms +=
            transition_started.elapsed().as_micros() as f64 / 1_000.0;
    }

    for mut search in searches {
        let answer_contract = match search.answer_contract {
            burn_dragon_universality::RuliadProofActionAnswerContract::PresentationIndex => {
                "action_index"
            }
            burn_dragon_universality::RuliadProofActionAnswerContract::SemanticStep => {
                "proof_action_step"
            }
        };
        let source_label = burn_dragon_universality::ruliad_source_capability_label(
            burn_dragon_universality::RuliadFamilyKind::FormalProof.label(),
            burn_dragon_universality::RuliadTaskKind::SelectProofAction.label(),
            search.difficulty_level,
            answer_contract,
        );
        search.summary.solved = usize::from(search.best_node.state.solved());
        search.summary.solved_goals = search.best_node.state.solved_goals();
        search.summary.steps = search.best_node.steps;
        search.summary.valid_actions = search.best_node.steps;
        let item_summary = search.summary;
        let valid_actions = item_summary.valid_actions;
        let invalid_actions = item_summary.invalid_actions;
        let repeated_states = item_summary.repeated_states;
        let backtracks = item_summary.backtracks;
        let scored_states = item_summary.scored_states;
        let scored_actions = item_summary.scored_actions;
        let top1_expert_actions = item_summary.top1_expert_actions;
        let frontier_exhaustions = item_summary.frontier_exhaustions;
        summary.accumulate(item_summary);
        difficulty_summaries
            .entry(search.difficulty_level)
            .or_default()
            .accumulate(item_summary);
        source_summaries
            .entry(source_label)
            .or_default()
            .accumulate(item_summary);
        examples.push(CapabilityProbeExample {
            label: format!(
                "d{}:{}:{}",
                search.difficulty_level,
                search.problem.domain.label(),
                &search.certificate_hash[..8]
            ),
            prompt: String::new(),
            expected: format!("solve=1;goals={}", search.best_node.state.total_goals()),
            actual: Some(format!(
                "solve={};goals={};valid={valid_actions};invalid={invalid_actions};loops={repeated_states};backtracks={backtracks};scored_states={scored_states};scored_actions={scored_actions};top1={top1_expert_actions};exhausted={frontier_exhaustions};beam={}",
                usize::from(search.best_node.state.solved()),
                search.best_node.state.solved_goals(),
                config.beam_width,
            )),
            completion: String::new(),
            status: if search.best_node.state.solved() {
                "VerifierMatch".to_string()
            } else if search.best_node.state.solved_goals() > 0 {
                "Partial".to_string()
            } else {
                "SchemaValidWrong".to_string()
            },
            reason: if invalid_actions > 0 {
                "invalid_or_malformed_action".to_string()
            } else if repeated_states > 0 {
                "repeated_proof_state".to_string()
            } else if search.best_node.state.solved() {
                String::new()
            } else {
                "step_budget_exhausted".to_string()
            },
            generated_tokens: valid_actions,
        });
    }

    scoring_summary.elapsed_ms = probe_started.elapsed().as_millis() as f64;
    eprintln!(
        "ruliad policy probe complete run={run_name} epoch={epoch} items={} scored_states={} elapsed_ms={:.0} cpu_prepare_ms={:.0} model_scoring_ms={:.0} cpu_transition_ms={:.0}",
        summary.items,
        summary.scored_states,
        scoring_summary.elapsed_ms,
        scoring_summary.cpu_prepare_ms,
        scoring_summary.model_scoring_ms,
        scoring_summary.cpu_transition_ms,
    );
    emit_ruliad_policy_rollout_metrics(
        run_name,
        summary,
        config,
        scoring_summary,
        &difficulty_summaries,
        &examples,
        TrainingEventContext {
            epoch,
            absolute_step,
            bus,
        },
    );
    Ok(RuliadPolicyRolloutProbeResult {
        summary,
        difficulty_summaries,
        source_summaries,
    })
}

#[cfg(test)]
mod rollout_budget_tests {
    use super::ruliad_policy_rollout_limit;

    #[test]
    fn automatic_budget_tracks_finite_certificate_work() {
        assert_eq!(ruliad_policy_rollout_limit(0, 13).unwrap(), 52);
        assert_eq!(ruliad_policy_rollout_limit(0, 0).unwrap(), 4);
    }

    #[test]
    fn fixed_budget_retains_a_recovery_ceiling() {
        assert_eq!(ruliad_policy_rollout_limit(32, 13).unwrap(), 32);
        assert_eq!(ruliad_policy_rollout_limit(128, 13).unwrap(), 52);
    }

    #[test]
    fn impossible_fixed_budget_fails_closed() {
        let error = ruliad_policy_rollout_limit(12, 13).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("shorter than the verifier certificate")
        );
    }
}

pub(super) fn emit_ruliad_policy_rollout_metrics(
    run_name: &str,
    summary: RuliadPolicyRolloutProbeSummary,
    config: crate::config::RuliadPolicyProbeConfig,
    scoring: RuliadPolicyScoringSummary,
    difficulty_summaries: &BTreeMap<usize, RuliadPolicyRolloutProbeSummary>,
    examples: &[CapabilityProbeExample],
    event: TrainingEventContext<'_>,
) {
    let TrainingEventContext {
        epoch,
        absolute_step,
        bus,
    } = event;
    let attempted_actions = summary
        .valid_actions
        .saturating_add(summary.invalid_actions);
    let metrics = [
        ("Ruliad Policy Rollout Items", summary.items as f64),
        (
            "Ruliad Policy Rollout Solve Rate",
            ratio_usize(summary.solved, summary.items),
        ),
        (
            "Ruliad Policy Rollout Goal Completion Rate",
            ratio_usize(summary.solved_goals, summary.total_goals),
        ),
        (
            "Ruliad Policy Rollout Valid Action Rate",
            ratio_usize(summary.valid_actions, attempted_actions),
        ),
        (
            "Ruliad Policy Rollout Invalid Action Rate",
            ratio_usize(summary.invalid_actions, attempted_actions),
        ),
        (
            "Ruliad Policy Rollout Repeated State Rate",
            ratio_usize(summary.repeated_states, summary.valid_actions),
        ),
        (
            "Ruliad Policy Rollout Backtrack Rate",
            ratio_usize(
                summary.backtracks,
                summary.valid_actions.saturating_add(summary.backtracks),
            ),
        ),
        (
            "Ruliad Policy Rollout Mean Backtracks",
            ratio_usize(summary.backtracks, summary.items),
        ),
        (
            "Ruliad Policy Model Top-1 Expert Rate",
            ratio_usize(summary.top1_expert_actions, summary.scored_states),
        ),
        (
            "Ruliad Policy Candidate Symmetry Balanced",
            usize::from(!matches!(
                config.candidate_symmetry,
                crate::config::RuliadProofPolicyCandidateSymmetry::Canonical
            )) as f64,
        ),
        (
            "Ruliad Policy Candidate Symmetry Orbit Averaged",
            usize::from(matches!(
                config.candidate_symmetry,
                crate::config::RuliadProofPolicyCandidateSymmetry::CyclicOrbitAverage
            )) as f64,
        ),
        (
            "Ruliad Policy Mean Scored States",
            ratio_usize(summary.scored_states, summary.items),
        ),
        (
            "Ruliad Policy Mean Scored Actions",
            ratio_usize(summary.scored_actions, summary.items),
        ),
        (
            "Ruliad Policy Frontier Exhaustions Per Item",
            ratio_usize(summary.frontier_exhaustions, summary.items),
        ),
        (
            "Ruliad Policy Rollout Mean Steps",
            ratio_usize(summary.steps, summary.items),
        ),
        ("Ruliad Policy Scoring Batches", scoring.batches as f64),
        (
            "Ruliad Policy Scoring Mean Batch Rows",
            ratio_usize(scoring.rows, scoring.batches),
        ),
        (
            "Ruliad Policy Scoring Maximum Batch Rows",
            scoring.maximum_batch_rows as f64,
        ),
        (
            "Ruliad Policy Scoring Maximum Pipeline Depth",
            scoring.maximum_pipeline_depth as f64,
        ),
        (
            "Ruliad Policy Scoring Padding Utilization",
            ratio_usize(scoring.unpadded_tokens, scoring.padded_tokens),
        ),
        ("Ruliad Policy Probe Elapsed MS", scoring.elapsed_ms),
        ("Ruliad Policy Probe CPU Prepare MS", scoring.cpu_prepare_ms),
        (
            "Ruliad Policy Probe Model Scoring MS",
            scoring.model_scoring_ms,
        ),
        (
            "Ruliad Policy Probe CPU Transition MS",
            scoring.cpu_transition_ms,
        ),
        (
            "Ruliad Policy Probe Scored States Per Second",
            if scoring.elapsed_ms > 0.0 {
                summary.scored_states as f64 * 1_000.0 / scoring.elapsed_ms
            } else {
                0.0
            },
        ),
        (
            "Ruliad Policy Probe Scored Actions Per Second",
            if scoring.elapsed_ms > 0.0 {
                summary.scored_actions as f64 * 1_000.0 / scoring.elapsed_ms
            } else {
                0.0
            },
        ),
        (
            "Ruliad Policy Probe Padded Tokens Per Second",
            if scoring.elapsed_ms > 0.0 {
                scoring.padded_tokens as f64 * 1_000.0 / scoring.elapsed_ms
            } else {
                0.0
            },
        ),
    ];
    for (name, value) in metrics {
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: run_name.to_string().into(),
            split: TrainingMetricSplit::Valid,
            epoch,
            step_in_epoch: 0,
            absolute_step,
            name: name.to_string(),
            value,
            running_value: value,
        });
    }
    let solve_rate = ratio_usize(summary.solved, summary.items);
    let goal_completion_rate = ratio_usize(summary.solved_goals, summary.total_goals);
    let valid_action_rate = ratio_usize(summary.valid_actions, attempted_actions);
    let repeated_state_rate = ratio_usize(summary.repeated_states, summary.valid_actions);
    let gate = config.promotion_gate;
    let gate_status = ruliad_policy_promotion_gate_status(summary, gate);
    for (name, value) in [
        (
            "Ruliad Policy Promotion Gate Passed",
            if gate_status.passed { 1.0 } else { 0.0 },
        ),
        (
            "Ruliad Policy Promotion Gate Failure Count",
            gate_status.reasons.len() as f64,
        ),
    ] {
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: run_name.to_string().into(),
            split: TrainingMetricSplit::Valid,
            epoch,
            step_in_epoch: 0,
            absolute_step,
            name: name.to_string(),
            value,
            running_value: value,
        });
    }
    if gate.enabled && !gate_status.passed {
        let _ = bus.send_gate_event(TrainingGateEvent {
            run_id: run_name.to_string().into(),
            gate: "ruliad_proof_policy_promotion_gate_failed".to_string(),
            action: TrainingGateAction::Alert,
            severity: TrainingGateSeverity::Warning,
            epoch: Some(epoch),
            absolute_step: Some(absolute_step),
            message: format!(
                "ruliad proof-policy promotion gate failed: {}",
                gate_status.reasons.join(", ")
            ),
        });
    }
    let group_buckets = difficulty_summaries
        .iter()
        .map(|(difficulty_level, summary)| {
            let bucket = ruliad_policy_capability_bucket(*difficulty_level, *summary);
            for (suffix, value) in [
                ("Solve Rate", bucket.verifier_rate),
                ("Goal Completion Rate", bucket.partial_credit_rate),
                ("Valid Action Rate", bucket.answer_field_accuracy),
                (
                    "Repeated State Rate",
                    ratio_usize(summary.repeated_states, summary.valid_actions),
                ),
                (
                    "Backtrack Rate",
                    ratio_usize(
                        summary.backtracks,
                        summary.valid_actions.saturating_add(summary.backtracks),
                    ),
                ),
                (
                    "Model Top-1 Expert Rate",
                    ratio_usize(summary.top1_expert_actions, summary.scored_states),
                ),
            ] {
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: run_name.to_string().into(),
                    split: TrainingMetricSplit::Valid,
                    epoch,
                    step_in_epoch: 0,
                    absolute_step,
                    name: format!("Ruliad Policy d{difficulty_level} {suffix}"),
                    value,
                    running_value: value,
                });
            }
            bucket
        })
        .collect::<Vec<_>>();
    let _ = bus.send_capability_probe_sample(CapabilityProbeSample {
        run_id: run_name.to_string().into(),
        split: TrainingMetricSplit::Valid,
        epoch,
        absolute_step,
        probe_name: "ruliad_proof_policy_rollout".to_string(),
        item_count: summary.items,
        scored_count: summary.items,
        exact_rate: solve_rate,
        semantic_rate: solve_rate,
        verifier_rate: solve_rate,
        partial_credit_rate: goal_completion_rate,
        schema_valid_wrong_rate: ratio_usize(summary.invalid_actions, attempted_actions),
        malformed_rate: ratio_usize(summary.invalid_actions, attempted_actions),
        missing_rate: 0.0,
        certificate_rate: solve_rate,
        completion_health_rate: valid_action_rate * (1.0 - repeated_state_rate),
        mean_partial_progress: goal_completion_rate,
        answer_field_accuracy: valid_action_rate,
        answer_field_coverage: 1.0,
        answer_termination_rate: 1.0,
        expected_answer_distinct_fraction: 0.0,
        actual_answer_distinct_fraction: 0.0,
        actual_answer_dominant_fraction: None,
        field_value_distinct_ratio: None,
        field_value_dominant_fraction: None,
        mean_completion_tokens: ratio_usize(summary.steps, summary.items),
        achieved_difficulty_level: None,
        output_entropy_bits: None,
        output_distinct_2_fraction: None,
        completion_repetition_fraction: Some(repeated_state_rate),
        completion_distinct_1_fraction: None,
        completion_distinct_2_fraction: None,
        completion_max_period_2_to_16_fraction: None,
        completion_max_period_2_to_64_fraction: None,
        completion_dominant_period_2_to_64: None,
        group_buckets,
        examples: examples.to_vec(),
    });
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RuliadProbeDecodeMode {
    FreeRun,
    PromptSchemaContract,
    FixedContract,
}

pub(super) fn ruliad_probe_generation_in_flight_rows(
    training_batch_size: usize,
    configured_maximum: usize,
    item_count: usize,
) -> usize {
    training_batch_size
        .max(1)
        .min(configured_maximum.max(1))
        .min(item_count.max(1))
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct RuliadProbeGenerationStats {
    pub(super) prompt_position_groups: usize,
    pub(super) largest_prompt_position_group: usize,
    pub(super) batched_rows: usize,
    pub(super) batched_forwards: usize,
    pub(super) independent_rows: usize,
    pub(super) maximum_in_flight_rows: usize,
    pub(super) maximum_batch_rows: usize,
    pub(super) mean_batch_rows: f64,
    pub(super) maximum_batch_prompt_position_span: usize,
    pub(super) mean_batch_prompt_position_span: f64,
    pub(super) device_buffer_tokens: usize,
    pub(super) profile: crate::generation::GenerationProfileSnapshot,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct RuliadProbeGenerationWork {
    pub(super) probe_indices: Vec<usize>,
    pub(super) batched: bool,
}

pub(super) fn ruliad_probe_generation_work(
    prompt_lengths: &[usize],
    enabled: bool,
    max_batch_rows: usize,
    minimum_batch_rows: usize,
    maximum_prompt_position_span: usize,
) -> Vec<RuliadProbeGenerationWork> {
    if !enabled {
        return (0..prompt_lengths.len())
            .map(|probe_index| RuliadProbeGenerationWork {
                probe_indices: vec![probe_index],
                batched: false,
            })
            .collect();
    }

    let max_batch_rows = max_batch_rows.max(1);
    let maximum_prompt_position_span = maximum_prompt_position_span.max(1);
    let mut sorted_indices = (0..prompt_lengths.len()).collect::<Vec<_>>();
    sorted_indices.sort_by_key(|index| (prompt_lengths[*index], *index));

    let mut cohorts = Vec::<Vec<usize>>::new();
    let mut current = Vec::<usize>::new();
    let mut first_prompt_position = 0usize;
    for probe_index in sorted_indices {
        let prompt_position = prompt_lengths[probe_index];
        let exceeds_span = !current.is_empty()
            && prompt_position.saturating_sub(first_prompt_position) > maximum_prompt_position_span;
        if !current.is_empty() && (current.len() >= max_batch_rows || exceeds_span) {
            cohorts.push(std::mem::take(&mut current));
        }
        if current.is_empty() {
            first_prompt_position = prompt_position;
        }
        current.push(probe_index);
    }
    if !current.is_empty() {
        cohorts.push(current);
    }

    cohorts
        .into_iter()
        .flat_map(|chunk| {
            if chunk.len() >= minimum_batch_rows.max(1) {
                vec![RuliadProbeGenerationWork {
                    probe_indices: chunk,
                    batched: true,
                }]
            } else {
                chunk
                    .into_iter()
                    .map(|probe_index| RuliadProbeGenerationWork {
                        probe_indices: vec![probe_index],
                        batched: false,
                    })
                    .collect()
            }
        })
        .collect()
}

pub(super) fn ruliad_probe_generation_waves(
    work: &[RuliadProbeGenerationWork],
    maximum_in_flight_rows: usize,
) -> Vec<Vec<RuliadProbeGenerationWork>> {
    let maximum_in_flight_rows = maximum_in_flight_rows.max(1);
    let mut waves = Vec::<Vec<RuliadProbeGenerationWork>>::new();
    let mut current = Vec::new();
    let mut current_rows = 0usize;
    for item in work {
        let item_rows = item.probe_indices.len().max(1);
        if !current.is_empty() && current_rows.saturating_add(item_rows) > maximum_in_flight_rows {
            waves.push(std::mem::take(&mut current));
            current_rows = 0;
        }
        current.push(item.clone());
        current_rows = current_rows.saturating_add(item_rows);
        if current_rows >= maximum_in_flight_rows {
            waves.push(std::mem::take(&mut current));
            current_rows = 0;
        }
    }
    if !current.is_empty() {
        waves.push(current);
    }
    waves
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct RuliadProbeGenerationBudget {
    pub(super) max_new_tokens: usize,
    pub(super) minimum_answer_tokens: usize,
    pub(super) budget_sufficient: bool,
    pub(super) generation_hit_budget: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct RuliadProbeGenerationBudgetSummary {
    pub(super) mean_max_new_tokens: f64,
    pub(super) mean_minimum_answer_tokens: f64,
    pub(super) sufficient_fraction: f64,
    pub(super) hit_budget_fraction: f64,
}

pub(super) struct RuliadProbeGenerator<'a, B: BackendTrait> {
    pub(super) dataset: &'a Dataset,
    pub(super) model: &'a LanguageTrainModel<B>,
    pub(super) training: &'a TrainingHyperparameters,
    pub(super) device: &'a B::Device,
    pub(super) close_token_id: Option<i64>,
    pub(super) decode_mode: RuliadProbeDecodeMode,
    pub(super) context_router: Option<&'a crate::train::PredictiveContextValidationRouter<B>>,
}

impl<B> RuliadProbeGenerator<'_, B>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    fn generate(
        &self,
        probe: &crate::dataset::RuliadValidationProbeItem,
        generation_budget: RuliadProbeGenerationBudget,
    ) -> Result<Vec<i64>> {
        let max_new_tokens = generation_budget.max_new_tokens;
        let generation_settings = crate::generation::GenerationSettings {
            max_new_tokens: Some(max_new_tokens),
            temperature: 1.0,
            top_k: Some(1),
            strategy: crate::generation::resolve_context_strategy(
                &self.training.context_strategy,
                self.training.block_size,
            ),
            stop_on_token: self.close_token_id,
        };
        let prompt_len = probe.prompt_tokens.len();
        match self.decode_mode {
            RuliadProbeDecodeMode::FreeRun => {
                let full_tokens = if let Some(router) = self.context_router {
                    let route = router.select(self.model, &probe.prompt_tokens)?;
                    crate::generation::generate_greedy_tokens_with_subnetwork_masks(
                        &self.model.model,
                        probe.prompt_tokens.clone(),
                        self.device,
                        generation_settings,
                        route.masks.neuron,
                        route.masks.activity,
                    )?
                } else {
                    crate::generation::generate_tokens(
                        &self.model.model,
                        probe.prompt_tokens.clone(),
                        self.device,
                        generation_settings,
                        None,
                    )?
                };
                Ok(full_tokens
                    .get(prompt_len..)
                    .map(|tokens| tokens.to_vec())
                    .unwrap_or_default())
            }
            RuliadProbeDecodeMode::FixedContract => Ok(ruliad_fixed_contract_completion_tokens(
                self.dataset,
                &self.model.model,
                probe.prompt_tokens.clone(),
                &probe.item.expected_answer,
                probe.item.document_close_marker(),
                max_new_tokens,
                self.device,
            )
            .unwrap_or_default()),
            RuliadProbeDecodeMode::PromptSchemaContract => {
                Ok(ruliad_prompt_schema_completion_tokens(
                    self.dataset,
                    &self.model.model,
                    probe.prompt_tokens.clone(),
                    &probe.item.prompt,
                    max_new_tokens,
                    self.device,
                )
                .unwrap_or_default())
            }
        }
    }
}

pub(super) fn generate_ruliad_probe_rows<B>(
    generator: &RuliadProbeGenerator<'_, B>,
    probe_items: &[crate::dataset::RuliadValidationProbeItem],
    generation_budgets: &[RuliadProbeGenerationBudget],
    training_batch_size: usize,
) -> Result<(Vec<Vec<i64>>, RuliadProbeGenerationStats)>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    crate::generation::generation_profile_reset();
    let config = generator.training.ruliad_probe_generation;
    let prompt_lengths = probe_items
        .iter()
        .map(|probe| probe.prompt_tokens.len())
        .collect::<Vec<_>>();
    let work = ruliad_probe_generation_work(
        &prompt_lengths,
        config.enabled
            && generator.decode_mode == RuliadProbeDecodeMode::FreeRun
            && generator.context_router.is_none(),
        config.max_batch_rows.min(training_batch_size.max(1)),
        config.minimum_batch_rows,
        config.maximum_prompt_position_span,
    );
    let mut generated_rows = (0..probe_items.len())
        .map(|_| None)
        .collect::<Vec<Option<Vec<i64>>>>();
    let mut stats = RuliadProbeGenerationStats {
        prompt_position_groups: prompt_lengths
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len(),
        largest_prompt_position_group: prompt_lengths
            .iter()
            .copied()
            .fold(BTreeMap::<usize, usize>::new(), |mut counts, prompt_len| {
                *counts.entry(prompt_len).or_default() += 1;
                counts
            })
            .into_values()
            .max()
            .unwrap_or_default(),
        device_buffer_tokens: config.device_buffer_tokens,
        ..RuliadProbeGenerationStats::default()
    };
    stats.batched_rows = work
        .iter()
        .filter(|item| item.batched)
        .map(|item| item.probe_indices.len())
        .sum();
    stats.batched_forwards = work.iter().filter(|item| item.batched).count();
    stats.independent_rows = work
        .iter()
        .filter(|item| !item.batched)
        .map(|item| item.probe_indices.len())
        .sum();
    stats.maximum_batch_rows = work
        .iter()
        .map(|item| item.probe_indices.len())
        .max()
        .unwrap_or_default();
    let batch_prompt_position_spans = work
        .iter()
        .filter(|item| item.batched)
        .map(|item| {
            let minimum = item
                .probe_indices
                .iter()
                .map(|index| prompt_lengths[*index])
                .min()
                .unwrap_or_default();
            let maximum = item
                .probe_indices
                .iter()
                .map(|index| prompt_lengths[*index])
                .max()
                .unwrap_or_default();
            maximum.saturating_sub(minimum)
        })
        .collect::<Vec<_>>();
    stats.maximum_batch_prompt_position_span = batch_prompt_position_spans
        .iter()
        .copied()
        .max()
        .unwrap_or_default();
    stats.mean_batch_prompt_position_span = if batch_prompt_position_spans.is_empty() {
        0.0
    } else {
        batch_prompt_position_spans.iter().sum::<usize>() as f64
            / batch_prompt_position_spans.len() as f64
    };
    let in_flight_row_budget = ruliad_probe_generation_in_flight_rows(
        training_batch_size,
        config.max_in_flight_rows,
        probe_items.len(),
    );
    let waves = ruliad_probe_generation_waves(&work, in_flight_row_budget);
    stats.maximum_in_flight_rows = waves
        .iter()
        .map(|wave| {
            wave.iter()
                .map(|item| item.probe_indices.len())
                .sum::<usize>()
        })
        .max()
        .unwrap_or_default();
    let execute = |item: &RuliadProbeGenerationWork| {
        let rows = if item.batched {
            let prompts = item
                .probe_indices
                .iter()
                .map(|probe_index| probe_items[*probe_index].prompt_tokens.clone())
                .collect::<Vec<_>>();
            let budgets = item
                .probe_indices
                .iter()
                .map(|probe_index| generation_budgets[*probe_index].max_new_tokens)
                .collect::<Vec<_>>();
            crate::generation::generate_greedy_batch_ragged(
                &generator.model.model,
                &prompts,
                &budgets,
                generator.device,
                crate::generation::resolve_context_strategy(
                    &generator.training.context_strategy,
                    generator.training.block_size,
                ),
                generator.close_token_id,
                config.device_buffer_tokens,
            )?
        } else {
            let probe_index = item.probe_indices[0];
            vec![generator.generate(&probe_items[probe_index], generation_budgets[probe_index])?]
        };
        Ok::<_, anyhow::Error>((item.probe_indices.clone(), rows))
    };
    for wave in waves {
        let wave_results = if wave.len() == 1 {
            wave.iter().map(&execute).collect::<Result<Vec<_>>>()?
        } else {
            wave.par_iter().map(&execute).collect::<Result<Vec<_>>>()?
        };
        for (probe_indices, rows) in wave_results {
            for (probe_index, row) in probe_indices.into_iter().zip(rows) {
                generated_rows[probe_index] = Some(row);
            }
        }
    }
    let forward_groups = stats
        .batched_forwards
        .saturating_add(stats.independent_rows);
    stats.mean_batch_rows = if forward_groups > 0 {
        probe_items.len() as f64 / forward_groups as f64
    } else {
        0.0
    };
    stats.profile = crate::generation::generation_profile_snapshot();

    generated_rows
        .into_iter()
        .enumerate()
        .map(|(probe_index, row)| {
            row.ok_or_else(|| anyhow!("missing generated probe row {probe_index}"))
        })
        .collect::<Result<Vec<_>>>()
        .map(|rows| (rows, stats))
}

pub(super) fn ruliad_probe_generation_budget(
    dataset: &Dataset,
    item: &burn_dragon_universality::RuliadEvalItem,
    training: &TrainingHyperparameters,
) -> RuliadProbeGenerationBudget {
    let base = training.events.ruliad_correctness_probe_tokens.max(1);
    let hard_cap = training
        .events
        .ruliad_correctness_probe_hard_token_cap
        .max(base);
    let expected_completion = format!("{}\n{}", item.expected_answer, item.document_close_marker());
    let minimum_answer_tokens = dataset
        .encode_ruliad_payload_tokens(&expected_completion)
        .map(|tokens| tokens.len())
        .unwrap_or(base);
    let max_new_tokens = if training
        .events
        .ruliad_correctness_probe_adaptive_answer_budget
    {
        base.max(minimum_answer_tokens).min(hard_cap)
    } else {
        base
    };
    RuliadProbeGenerationBudget {
        max_new_tokens,
        minimum_answer_tokens,
        budget_sufficient: max_new_tokens >= minimum_answer_tokens,
        generation_hit_budget: false,
    }
}

pub(super) fn ruliad_probe_generation_budget_summary(
    budgets: &[RuliadProbeGenerationBudget],
) -> Option<RuliadProbeGenerationBudgetSummary> {
    if budgets.is_empty() {
        return None;
    }
    let count = budgets.len() as f64;
    Some(RuliadProbeGenerationBudgetSummary {
        mean_max_new_tokens: budgets
            .iter()
            .map(|budget| budget.max_new_tokens)
            .sum::<usize>() as f64
            / count,
        mean_minimum_answer_tokens: budgets
            .iter()
            .map(|budget| budget.minimum_answer_tokens)
            .sum::<usize>() as f64
            / count,
        sufficient_fraction: budgets
            .iter()
            .filter(|budget| budget.budget_sufficient)
            .count() as f64
            / count,
        hit_budget_fraction: budgets
            .iter()
            .filter(|budget| budget.generation_hit_budget)
            .count() as f64
            / count,
    })
}
