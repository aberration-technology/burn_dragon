use burn::tensor::{
    Int, Tensor, TensorData,
    backend::{AutodiffBackend, Backend},
};
use burn_dragon_core::DragonModel;
use std::collections::HashSet;

use crate::config::{
    LocalPredictiveCodingTerminalCriterion, RuliadProofPolicyNormalization,
    RuliadProofPolicyTrainingConfig,
};

use super::criterion::LocalPcTerminalCriterion;

#[derive(Debug, Clone)]
pub(crate) struct PreparedRuliadVerifierTerminal<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub criterion: LocalPcTerminalCriterion<B>,
    pub semantic_states: usize,
    pub decision_rows: usize,
    pub stats: RuliadVerifierPanelStats,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RuliadVerifierPanelStats {
    pub answer_contract: &'static str,
    pub configured_mode: &'static str,
    pub effective_mode: &'static str,
    pub available_sample_groups: usize,
    pub sample_groups: usize,
    pub nonzero_start_trajectories: usize,
    pub start_step_sum: usize,
    pub semantic_states: usize,
    pub base_semantic_states: usize,
    pub counterfactual_semantic_states: usize,
    pub counterfactual_target_shortfall: usize,
    pub static_expert_states: usize,
    pub dagger_expert_states: usize,
    pub model_visited_states: usize,
    pub model_scoring_batches: usize,
    pub model_valid_actions: usize,
    pub model_invalid_actions: usize,
    pub model_expert_equivalent_actions: usize,
    pub model_off_expert_actions: usize,
    pub repeated_states: usize,
    pub backtracks: usize,
    pub solved_proofs: usize,
    pub rollout_depth_reached: usize,
}

pub(crate) fn lift_ruliad_verifier_terminal<B: AutodiffBackend>(
    prepared: PreparedRuliadVerifierTerminal<B::InnerBackend>,
) -> PreparedRuliadVerifierTerminal<B> {
    let criterion = match prepared.criterion {
        LocalPcTerminalCriterion::CategoricalSetAtPositions {
            positions,
            support_action_mask,
            valid_action_mask,
            row_weights,
            eps,
        } => LocalPcTerminalCriterion::CategoricalSetAtPositions {
            positions: Tensor::from_inner(positions),
            support_action_mask: Tensor::from_inner(support_action_mask),
            valid_action_mask: Tensor::from_inner(valid_action_mask),
            row_weights: Tensor::from_inner(row_weights),
            eps,
        },
        LocalPcTerminalCriterion::SequenceEnergySetAtPositions {
            prompt_positions,
            terminal_positions,
            valid_action_mask,
            row_weights,
            candidates_per_group,
            eps,
        } => LocalPcTerminalCriterion::SequenceEnergySetAtPositions {
            prompt_positions: Tensor::from_inner(prompt_positions),
            terminal_positions: Tensor::from_inner(terminal_positions),
            valid_action_mask: Tensor::from_inner(valid_action_mask),
            row_weights: Tensor::from_inner(row_weights),
            candidates_per_group,
            eps,
        },
        _ => unreachable!("Ruliad verifier preparation only emits sparse-position criteria"),
    };
    PreparedRuliadVerifierTerminal {
        inputs: Tensor::from_inner(prepared.inputs),
        criterion,
        semantic_states: prepared.semantic_states,
        decision_rows: prepared.decision_rows,
        stats: prepared.stats,
    }
}

#[derive(Debug)]
enum VerifierDecisionRow {
    Prefix {
        inputs: Vec<i64>,
        position: usize,
        support_tokens: Vec<i64>,
        valid_tokens: Vec<i64>,
        weight: f32,
    },
    SequenceEnergy {
        inputs: Vec<Vec<i64>>,
        prompt_position: usize,
        terminal_positions: Vec<usize>,
        valid_indices: Vec<usize>,
        weight: f32,
    },
}

struct VerifierTrajectory {
    sample_index: usize,
    is_dagger: bool,
    max_depth: usize,
    answer_contract: burn_dragon_universality::ruliad::RuliadProofActionAnswerContract,
    state: burn_dragon_universality::ruliad::RuliadProofPolicyState,
}

struct PreparedVerifierState {
    canonical_prompt: Vec<i64>,
    rows: Vec<VerifierDecisionRow>,
    rotations: Vec<usize>,
    request: crate::train::ruliad_policy::EncodedRuliadProofActionRequest,
}

struct VerifierExpansion {
    trajectory_index: usize,
    actions: burn_dragon_universality::ruliad::RuliadProofActionSet,
    request: crate::train::ruliad_policy::EncodedRuliadProofActionRequest,
}

pub(crate) fn verifier_terminal_due(
    terminal: LocalPredictiveCodingTerminalCriterion,
    policy: RuliadProofPolicyTrainingConfig,
    absolute_step: usize,
) -> bool {
    matches!(
        terminal,
        LocalPredictiveCodingTerminalCriterion::RuliadVerifierSet
    ) && policy.enabled
        && policy.every_steps > 0
        && absolute_step >= policy.start_after_steps
        && absolute_step.is_multiple_of(policy.every_steps)
}

/// Materialize a bounded static-expert verifier panel as sparse terminal rows.
///
/// Only actual decision points in the candidate trie become model rows;
/// deterministic action syntax is context. Every semantic state has unit total
/// weight regardless of presentation count or trie depth.
pub(crate) fn prepare_ruliad_verifier_terminal<B: Backend + Clone + 'static>(
    policy_batch: &crate::dataset::RuliadPolicyBatch,
    config: RuliadProofPolicyTrainingConfig,
    block_size: usize,
    vocab: usize,
    device: &B::Device,
) -> Option<PreparedRuliadVerifierTerminal<B>>
where
    B::Device: Clone,
{
    let mut static_config = config;
    static_config.mode = crate::config::RuliadProofPolicyTrainingMode::StaticExpert;
    prepare_ruliad_verifier_terminal_at_step::<B>(
        None,
        policy_batch,
        static_config,
        block_size,
        vocab,
        0,
        device,
    )
}

/// Build the exact sparse verifier terminal used by both the global-backprop
/// control and local predictive coding. Dynamic modes use detached model
/// decisions to visit states, while every resulting row is still labelled by
/// the formal expert.
pub(crate) fn prepare_ruliad_verifier_terminal_at_step<B: Backend + Clone + 'static>(
    sampling_model: Option<&DragonModel<B>>,
    policy_batch: &crate::dataset::RuliadPolicyBatch,
    config: RuliadProofPolicyTrainingConfig,
    block_size: usize,
    vocab: usize,
    absolute_step: usize,
    device: &B::Device,
) -> Option<PreparedRuliadVerifierTerminal<B>>
where
    B::Device: Clone,
{
    let tokenizer = burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer::from_config(
        &policy_batch.tokenization,
    )
    .ok()?;
    let completion_budget = config
        .max_completion_tokens
        .max(1)
        .min(block_size.saturating_sub(1).max(1));
    let effective_mode = config.effective_mode(absolute_step);
    let plan = crate::train::steps::RuliadProofPolicyBatchPlan::new(
        effective_mode,
        config.base_semantic_rows_per_update(),
        config.rollout_steps,
        config.stratified_difficulty_levels,
    );
    let base_state_budget = config.base_semantic_rows_per_update().max(1);
    let row_budget = config.max_presentation_rows_per_update.max(1);
    let mut rows = Vec::<VerifierDecisionRow>::new();
    let mut semantic_states = 0usize;
    let mut base_semantic_states = 0usize;
    let mut counterfactual_semantic_states = 0usize;
    let mut counterfactual_target_shortfall = 0usize;
    let mut visited_prompts = HashSet::<Vec<i64>>::new();
    let mut trajectories = Vec::<VerifierTrajectory>::new();
    let mut available_sample_groups = 0usize;
    let mut nonzero_start_trajectories = 0usize;
    let mut start_step_sum = 0usize;
    let mut static_expert_states = 0usize;
    let mut dagger_expert_states = 0usize;
    let mut model_visited_states = 0usize;
    let mut model_scoring_batches = 0usize;
    let mut model_valid_actions = 0usize;
    let mut model_invalid_actions = 0usize;
    let mut model_expert_equivalent_actions = 0usize;
    let mut model_off_expert_actions = 0usize;
    let mut repeated_states = 0usize;
    let mut backtracks = 0usize;
    let mut rollout_depth_reached = 0usize;

    let prepare_state =
        |problem: &burn_dragon_universality::ruliad::RuliadProofProblem,
         actions: &burn_dragon_universality::ruliad::RuliadProofActionSet,
         answer_contract: burn_dragon_universality::ruliad::RuliadProofActionAnswerContract,
         presentation_index: usize,
         base_rotations: Option<&[usize]>|
         -> Option<PreparedVerifierState> {
            let rotations = crate::train::ruliad_policy::target_group_presentation_rotations(
                config.candidate_symmetry,
                actions.selected_index,
                actions.candidates.len(),
                presentation_index,
                base_rotations,
            )
            .ok()?;
            let canonical_prompt = tokenizer
                .encode_payload(
                    &burn_dragon_universality::ruliad::ruliad_proof_action_prompt(problem, actions)
                        .ok()?,
                )
                .into_iter()
                .map(i64::from)
                .collect::<Vec<_>>();
            let mut state_rows = Vec::<VerifierDecisionRow>::new();
            let mut presentations = Vec::with_capacity(rotations.len());
            for rotation in &rotations {
                let presented = actions.rotate_left(*rotation).ok()?;
                let prompt = burn_dragon_universality::ruliad::ruliad_proof_action_prompt(
                    problem, &presented,
                )
                .ok()?;
                let prompt_tokens = tokenizer
                    .encode_payload(&prompt)
                    .into_iter()
                    .map(i64::from)
                    .collect::<Vec<_>>();
                let candidates = (0..presented.candidates.len())
                .map(|candidate_index| {
                    let answer = burn_dragon_universality::ruliad::proof_action_answer(
                        &presented,
                        candidate_index,
                        answer_contract,
                    )
                    .ok()?;
                    let mut tokens = tokenizer
                        .encode_payload(&answer)
                        .into_iter()
                        .map(i64::from)
                        .collect::<Vec<_>>();
                    if answer_contract
                        == burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep
                        && let Some(stop_token_id) = policy_batch.stop_token_id
                        && tokens.last().copied() != Some(stop_token_id)
                    {
                        tokens.push(stop_token_id);
                    }
                    (!tokens.is_empty() && tokens.len() <= completion_budget).then_some(tokens)
                })
                .collect::<Option<Vec<_>>>()?;
                let max_completion = candidates.iter().map(Vec::len).max()?.max(1);
                let max_prompt = block_size.saturating_sub(max_completion).max(1);
                let prompt_tokens = if prompt_tokens.len() > max_prompt {
                    prompt_tokens[prompt_tokens.len() - max_prompt..].to_vec()
                } else {
                    prompt_tokens
                };
                match config.scoring {
                    crate::config::RuliadProofPolicyScoring::CompletionLikelihood => {
                        let branches =
                            crate::train::ruliad_policy::semantic_candidate_trie_branches(
                                &candidates,
                                &presented.equivalent_indices,
                            )
                            .ok()?;
                        let branch_weight = config.weight
                            / rotations.len().max(1) as f32
                            / branches.len().max(1) as f32;
                        for branch in branches {
                            let max_prompt = block_size.saturating_sub(branch.prefix.len()).max(1);
                            let prompt_start = prompt_tokens.len().saturating_sub(max_prompt);
                            let mut inputs = prompt_tokens[prompt_start..].to_vec();
                            inputs.extend_from_slice(&branch.prefix);
                            if inputs.is_empty() || inputs.len() > block_size {
                                return None;
                            }
                            let support_tokens = match config.normalization {
                                RuliadProofPolicyNormalization::VocabularyMarginal => Vec::new(),
                                RuliadProofPolicyNormalization::CandidateConditional
                                | RuliadProofPolicyNormalization::PrefixConditional => {
                                    branch.candidate_tokens
                                }
                            };
                            state_rows.push(VerifierDecisionRow::Prefix {
                                position: inputs.len() - 1,
                                inputs,
                                support_tokens,
                                valid_tokens: branch.equivalent_tokens,
                                weight: branch_weight,
                            });
                        }
                    }
                    crate::config::RuliadProofPolicyScoring::SemanticEnergy => {
                        if prompt_tokens.is_empty() || presented.equivalent_indices.is_empty() {
                            return None;
                        }
                        let prompt_position = prompt_tokens.len() - 1;
                        let mut sequence_inputs = Vec::with_capacity(candidates.len());
                        let mut terminal_positions = Vec::with_capacity(candidates.len());
                        for candidate in &candidates {
                            let mut inputs = prompt_tokens.clone();
                            inputs.extend_from_slice(candidate);
                            if inputs.len() > block_size {
                                return None;
                            }
                            terminal_positions.push(inputs.len().saturating_sub(1));
                            sequence_inputs.push(inputs);
                        }
                        state_rows.push(VerifierDecisionRow::SequenceEnergy {
                            inputs: sequence_inputs,
                            prompt_position,
                            terminal_positions,
                            valid_indices: presented.equivalent_indices.clone(),
                            weight: config.weight / rotations.len().max(1) as f32,
                        });
                    }
                    crate::config::RuliadProofPolicyScoring::ResidualEnergy => return None,
                }
                presentations.push(
                    crate::train::ruliad_policy::EncodedRuliadProofActionPresentation {
                        rotation: *rotation,
                        prompt_tokens,
                        candidate_tokens: candidates,
                    },
                );
            }
            Some(PreparedVerifierState {
                canonical_prompt,
                rows: state_rows,
                rotations,
                request: crate::train::ruliad_policy::EncodedRuliadProofActionRequest {
                    presentations,
                    answer_contract,
                },
            })
        };

    for (sample_index, sample) in policy_batch.samples.iter().enumerate() {
        let Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
            problem,
            certificate,
            proof_step_index,
            action_answer_contract,
            task: burn_dragon_universality::RuliadTaskKind::SelectProofAction,
            ..
        }) = sample.item.spec.as_ref()
        else {
            continue;
        };
        available_sample_groups = available_sample_groups.saturating_add(1);
        if trajectories.len() >= plan.trajectory_budget() {
            continue;
        }
        let start_step = proof_step_index.unwrap_or_default();
        let Ok(state) =
            burn_dragon_universality::ruliad::RuliadProofPolicyState::from_certificate_prefix(
                problem,
                certificate,
                start_step,
            )
        else {
            continue;
        };
        nonzero_start_trajectories =
            nonzero_start_trajectories.saturating_add(usize::from(start_step > 0));
        start_step_sum = start_step_sum.saturating_add(start_step);
        let trajectory_index = trajectories.len();
        let (is_dagger, max_depth) = if trajectory_index < plan.static_row_budget {
            (false, 1)
        } else {
            let dagger_index = trajectory_index.saturating_sub(plan.static_row_budget);
            (true, plan.dagger_depth(dagger_index))
        };
        trajectories.push(VerifierTrajectory {
            sample_index,
            is_dagger,
            max_depth,
            answer_contract: *action_answer_contract,
            state,
        });
    }

    for rollout_depth in 0..plan.rollout_steps {
        if base_semantic_states >= base_state_budget
            || trajectories.iter().all(|trajectory| {
                rollout_depth >= trajectory.max_depth || trajectory.state.solved()
            })
        {
            break;
        }
        let mut expansions = Vec::<VerifierExpansion>::new();
        for (trajectory_index, trajectory) in trajectories.iter_mut().enumerate() {
            if rollout_depth >= trajectory.max_depth
                || trajectory.state.solved()
                || base_semantic_states >= base_state_budget
            {
                continue;
            }
            let sample = &policy_batch.samples[trajectory.sample_index];
            let Some(burn_dragon_universality::RuliadSampleSpec::FormalProof { problem, .. }) =
                sample.item.spec.as_ref()
            else {
                continue;
            };
            let actions = match trajectory.state.action_set(problem, config.candidates) {
                Ok(actions) => actions,
                Err(_) => {
                    backtracks =
                        backtracks.saturating_add(usize::from(trajectory.state.backtrack()));
                    continue;
                }
            };
            let Some(original) = prepare_state(
                problem,
                &actions,
                trajectory.answer_contract,
                semantic_states,
                None,
            ) else {
                continue;
            };
            let rollout_request = original.request.clone();
            let target_group_rotations = original.rotations.clone();
            let mut prepared_states = vec![original];
            let counterfactual_indices =
                crate::train::ruliad_policy::counterfactual_candidate_indices(
                    &actions,
                    config.counterfactual_targets_per_state,
                    actions
                        .selected_index
                        .saturating_add(base_semantic_states)
                        .saturating_add(1),
                );
            let mut group_shortfall = config
                .counterfactual_targets_per_state
                .saturating_sub(counterfactual_indices.len());
            for candidate_index in counterfactual_indices {
                let Some((counterfactual_problem, counterfactual_actions)) =
                    burn_dragon_universality::ruliad::counterfactual_proof_action_target(
                        problem,
                        &actions,
                        candidate_index,
                    )
                    .ok()
                else {
                    group_shortfall = group_shortfall.saturating_add(1);
                    continue;
                };
                let Some(counterfactual) = prepare_state(
                    &counterfactual_problem,
                    &counterfactual_actions,
                    trajectory.answer_contract,
                    semantic_states.saturating_add(prepared_states.len()),
                    Some(&target_group_rotations),
                ) else {
                    group_shortfall = group_shortfall.saturating_add(1);
                    continue;
                };
                prepared_states.push(counterfactual);
            }
            counterfactual_target_shortfall =
                counterfactual_target_shortfall.saturating_add(group_shortfall);
            let complete_target_group =
                group_shortfall == 0 && prepared_states.len() == config.target_variants_per_state();
            let group_rows = prepared_states
                .iter()
                .map(|prepared| prepared.rows.len())
                .sum::<usize>();
            let unique_target_group = complete_target_group
                && prepared_states
                    .iter()
                    .all(|prepared| !visited_prompts.contains(&prepared.canonical_prompt));
            if !unique_target_group
                || group_rows == 0
                || rows.len().saturating_add(group_rows) > row_budget
            {
                continue;
            }
            let variants_added = prepared_states.len();
            for prepared in prepared_states {
                visited_prompts.insert(prepared.canonical_prompt);
                rows.extend(prepared.rows);
            }
            semantic_states = semantic_states.saturating_add(variants_added);
            base_semantic_states = base_semantic_states.saturating_add(1);
            counterfactual_semantic_states =
                counterfactual_semantic_states.saturating_add(variants_added.saturating_sub(1));
            static_expert_states = static_expert_states
                .saturating_add(variants_added.saturating_mul(usize::from(!trajectory.is_dagger)));
            dagger_expert_states = dagger_expert_states
                .saturating_add(variants_added.saturating_mul(usize::from(trajectory.is_dagger)));
            model_visited_states = model_visited_states.saturating_add(
                variants_added
                    .saturating_mul(usize::from(trajectory.is_dagger && rollout_depth > 0)),
            );
            rollout_depth_reached = rollout_depth_reached.max(rollout_depth.saturating_add(1));
            if trajectory.is_dagger && rollout_depth.saturating_add(1) < trajectory.max_depth {
                expansions.push(VerifierExpansion {
                    trajectory_index,
                    actions,
                    request: rollout_request,
                });
            }
        }
        if base_semantic_states >= base_state_budget || expansions.is_empty() {
            break;
        }
        let Some(sampling_model) = sampling_model else {
            break;
        };
        model_scoring_batches = model_scoring_batches.saturating_add(1);
        let requests = expansions
            .iter()
            .map(|expansion| expansion.request.clone())
            .collect::<Vec<_>>();
        let Ok(decisions) =
            crate::train::ruliad_policy::select_ruliad_proof_actions_batch_with_contract(
                sampling_model,
                &requests,
                config.max_presentation_rows_per_update.max(1),
                config.scoring,
                config.normalization,
                device,
            )
        else {
            break;
        };
        for (expansion, decision) in expansions.into_iter().zip(decisions) {
            if expansion
                .actions
                .is_equivalent_index(decision.selected_semantic_index)
            {
                model_expert_equivalent_actions = model_expert_equivalent_actions.saturating_add(1);
            } else {
                model_off_expert_actions = model_off_expert_actions.saturating_add(1);
            }
            match trajectories[expansion.trajectory_index]
                .state
                .apply(&expansion.actions, decision.selected_semantic_index)
            {
                Ok(repeated) => {
                    model_valid_actions = model_valid_actions.saturating_add(1);
                    repeated_states = repeated_states.saturating_add(usize::from(repeated));
                }
                Err(_) => model_invalid_actions = model_invalid_actions.saturating_add(1),
            }
        }
    }
    if rows.is_empty() {
        return None;
    }

    let row_count = rows.len();
    let answer_contract = trajectories
        .first()
        .map(|trajectory| trajectory.answer_contract.label())
        .unwrap_or_default();
    let configured_mode = match config.mode {
        crate::config::RuliadProofPolicyTrainingMode::StaticExpert => "static_expert",
        crate::config::RuliadProofPolicyTrainingMode::Dagger => "dagger",
        crate::config::RuliadProofPolicyTrainingMode::StaticThenPairedDagger => {
            "static_then_paired_dagger"
        }
    };
    let effective_mode = match effective_mode {
        crate::config::RuliadProofPolicyEffectiveMode::StaticExpert => "static_expert",
        crate::config::RuliadProofPolicyEffectiveMode::Dagger => "dagger",
        crate::config::RuliadProofPolicyEffectiveMode::PairedDagger => "paired_dagger",
    };
    let stats = RuliadVerifierPanelStats {
        answer_contract,
        configured_mode,
        effective_mode,
        available_sample_groups,
        sample_groups: trajectories.len(),
        nonzero_start_trajectories,
        start_step_sum,
        semantic_states,
        base_semantic_states,
        counterfactual_semantic_states,
        counterfactual_target_shortfall,
        static_expert_states,
        dagger_expert_states,
        model_visited_states,
        model_scoring_batches,
        model_valid_actions,
        model_invalid_actions,
        model_expert_equivalent_actions,
        model_off_expert_actions,
        repeated_states,
        backtracks,
        solved_proofs: trajectories
            .iter()
            .filter(|trajectory| trajectory.state.solved())
            .count(),
        rollout_depth_reached,
    };
    let (inputs, criterion) = match config.scoring {
        crate::config::RuliadProofPolicyScoring::CompletionLikelihood => {
            let sequence_len = rows
                .iter()
                .filter_map(|row| match row {
                    VerifierDecisionRow::Prefix { inputs, .. } => Some(inputs.len()),
                    VerifierDecisionRow::SequenceEnergy { .. } => None,
                })
                .max()?
                .max(1);
            let mut input_values = vec![0_i64; row_count * sequence_len];
            let mut positions = Vec::with_capacity(row_count);
            let mut support = vec![0.0_f32; row_count * vocab];
            let mut valid = vec![0.0_f32; row_count * vocab];
            let mut weights = Vec::with_capacity(row_count);
            for (row_index, row) in rows.into_iter().enumerate() {
                let VerifierDecisionRow::Prefix {
                    inputs,
                    position,
                    support_tokens,
                    valid_tokens,
                    weight,
                } = row
                else {
                    return None;
                };
                let offset = row_index * sequence_len;
                input_values[offset..offset + inputs.len()].copy_from_slice(&inputs);
                positions.push(i64::try_from(position).ok()?);
                weights.push(weight);
                if support_tokens.is_empty() {
                    support[row_index * vocab..(row_index + 1) * vocab].fill(1.0);
                } else {
                    for token in support_tokens {
                        let token = usize::try_from(token).ok()?;
                        if token >= vocab {
                            return None;
                        }
                        support[row_index * vocab + token] = 1.0;
                    }
                }
                for token in valid_tokens {
                    let token = usize::try_from(token).ok()?;
                    if token >= vocab || support[row_index * vocab + token] == 0.0 {
                        return None;
                    }
                    valid[row_index * vocab + token] = 1.0;
                }
            }
            (
                Tensor::from_data(
                    TensorData::new(input_values, [row_count, sequence_len]),
                    device,
                ),
                LocalPcTerminalCriterion::CategoricalSetAtPositions {
                    positions: Tensor::from_data(TensorData::new(positions, [row_count]), device),
                    support_action_mask: Tensor::from_data(
                        TensorData::new(support, [row_count, vocab]),
                        device,
                    ),
                    valid_action_mask: Tensor::from_data(
                        TensorData::new(valid, [row_count, vocab]),
                        device,
                    ),
                    row_weights: Tensor::from_data(TensorData::new(weights, [row_count]), device),
                    eps: 1.0e-12,
                },
            )
        }
        crate::config::RuliadProofPolicyScoring::SemanticEnergy => {
            let candidates_per_group = rows.first().and_then(|row| match row {
                VerifierDecisionRow::SequenceEnergy { inputs, .. } => Some(inputs.len()),
                VerifierDecisionRow::Prefix { .. } => None,
            })?;
            if candidates_per_group < 2 {
                return None;
            }
            let sequence_len = rows
                .iter()
                .filter_map(|row| match row {
                    VerifierDecisionRow::SequenceEnergy { inputs, .. } => {
                        inputs.iter().map(Vec::len).max()
                    }
                    VerifierDecisionRow::Prefix { .. } => None,
                })
                .max()?
                .max(1);
            let flat_rows = row_count.saturating_mul(candidates_per_group);
            let mut input_values = vec![0_i64; flat_rows.saturating_mul(sequence_len)];
            let mut prompt_positions = Vec::with_capacity(flat_rows);
            let mut terminal_positions = Vec::with_capacity(flat_rows);
            let mut valid = vec![0.0_f32; row_count.saturating_mul(candidates_per_group)];
            let mut weights = Vec::with_capacity(row_count);
            for (group_index, row) in rows.into_iter().enumerate() {
                let VerifierDecisionRow::SequenceEnergy {
                    inputs,
                    prompt_position,
                    terminal_positions: row_terminal_positions,
                    valid_indices,
                    weight,
                } = row
                else {
                    return None;
                };
                if inputs.len() != candidates_per_group
                    || row_terminal_positions.len() != candidates_per_group
                {
                    return None;
                }
                for (candidate_index, input) in inputs.into_iter().enumerate() {
                    let flat_index = group_index * candidates_per_group + candidate_index;
                    let offset = flat_index * sequence_len;
                    input_values[offset..offset + input.len()].copy_from_slice(&input);
                    prompt_positions.push(i64::try_from(prompt_position).ok()?);
                    terminal_positions
                        .push(i64::try_from(row_terminal_positions[candidate_index]).ok()?);
                }
                for candidate_index in valid_indices {
                    if candidate_index >= candidates_per_group {
                        return None;
                    }
                    valid[group_index * candidates_per_group + candidate_index] = 1.0;
                }
                weights.push(weight);
            }
            (
                Tensor::from_data(
                    TensorData::new(input_values, [flat_rows, sequence_len]),
                    device,
                ),
                LocalPcTerminalCriterion::SequenceEnergySetAtPositions {
                    prompt_positions: Tensor::from_data(
                        TensorData::new(prompt_positions, [flat_rows]),
                        device,
                    ),
                    terminal_positions: Tensor::from_data(
                        TensorData::new(terminal_positions, [flat_rows]),
                        device,
                    ),
                    valid_action_mask: Tensor::from_data(
                        TensorData::new(valid, [row_count, candidates_per_group]),
                        device,
                    ),
                    row_weights: Tensor::from_data(TensorData::new(weights, [row_count]), device),
                    candidates_per_group,
                    eps: 1.0e-12,
                },
            )
        }
        crate::config::RuliadProofPolicyScoring::ResidualEnergy => return None,
    };

    Some(PreparedRuliadVerifierTerminal {
        inputs,
        criterion,
        semantic_states,
        decision_rows: row_count,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::AutodiffModule;
    use burn_autodiff::Autodiff;
    use burn_dragon_core::{DragonConfig, SequenceTrainingExecutor};
    use burn_ndarray::NdArray;
    use std::path::PathBuf;

    type TestBackend = NdArray<f32>;
    type AutodiffTestBackend = Autodiff<TestBackend>;

    fn verifier_model_with_layers<B: Backend>(
        device: &B::Device,
        n_layer: usize,
    ) -> DragonModel<B> {
        let mut config = DragonConfig {
            n_layer,
            n_embd: 8,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 272,
            dropout: 0.0,
            ..DragonConfig::default()
        };
        config.sequence_kernel.executor = SequenceTrainingExecutor::DenseScoreShortContext;
        config.fused_kernels.rotary_embedding = burn_dragon_core::RotaryEmbedding::Alibi;
        // Gradient-contract tests should not depend on the subgradient chosen
        // exactly at ReLU zero. Keep their tiny deterministic fixture in the
        // smooth, active branch; kernel tests cover sparse and zero support.
        config.fused_kernels.relu_threshold = -0.25;
        DragonModel::new(config, device)
    }

    fn verifier_model<B: Backend>(device: &B::Device) -> DragonModel<B> {
        verifier_model_with_layers(device, 2)
    }

    fn verifier_model_with_score_head<B: Backend>(
        device: &B::Device,
        n_layer: usize,
    ) -> DragonModel<B> {
        let mut config = DragonConfig {
            n_layer,
            n_embd: 8,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 272,
            dropout: 0.0,
            ..DragonConfig::default()
        };
        config.sequence_kernel.executor = SequenceTrainingExecutor::DenseScoreShortContext;
        config.fused_kernels.rotary_embedding = burn_dragon_core::RotaryEmbedding::Alibi;
        config.fused_kernels.relu_threshold = -0.25;
        config.sequence_score_head.enabled = true;
        config.sequence_score_head.projection_dim = 6;
        DragonModel::new(config, device)
    }

    fn formal_policy_batch(
        answer_contract: burn_dragon_universality::ruliad::RuliadProofActionAnswerContract,
    ) -> crate::dataset::RuliadPolicyBatch {
        let bundle = burn_dragon_universality::ruliad::formal::generate_formal_bundle(
            29,
            burn_dragon_universality::ruliad::formal::RuliadFormalGeneratorConfig {
                rewrite_depth: 2,
                leaf_count: 3,
                context_depth: 1,
                distractor_axioms: 1,
                ..Default::default()
            },
        )
        .expect("formal bundle");
        let proof_step_index = 1.min(bundle.certificate.step_count().saturating_sub(1));
        let actions = burn_dragon_universality::ruliad::oracle_proof_action_set(
            &bundle.problem,
            &bundle.certificate,
            proof_step_index,
            4,
        )
        .expect("oracle action set");
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: bundle.problem.canonical_hash().expect("problem hash"),
            sample_index: 29,
            split: burn_dragon_universality::SampleSplit::Train,
            family: "formal_proof".to_string(),
            task_kind: burn_dragon_universality::RuliadTaskKind::SelectProofAction
                .label()
                .to_string(),
            math_domains: vec!["formal_proof".to_string()],
            reasoning_modes: vec!["proof_construction".to_string()],
            prompt: burn_dragon_universality::ruliad::ruliad_proof_action_prompt(
                &bundle.problem,
                &actions,
            )
            .expect("policy prompt"),
            expected_answer: format!("c={}", actions.selected_index),
            difficulty_level: Some(0),
            spec: Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
                problem: bundle.problem,
                certificate: bundle.certificate,
                candidate: None,
                proof_step_index: Some(proof_step_index),
                action_presentation_rotation: Some(0),
                action_answer_contract: answer_contract,
                task: burn_dragon_universality::RuliadTaskKind::SelectProofAction,
            }),
        };
        crate::dataset::RuliadPolicyBatch {
            samples: vec![crate::dataset::RuliadPolicySample {
                item,
                prompt_tokens: vec![1],
            }],
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::StructuredSymbolic {
                vocab_size: 272,
                eos_id: Some(271),
            },
            stop_token_id: Some(271),
        }
    }

    fn policy(normalization: RuliadProofPolicyNormalization) -> RuliadProofPolicyTrainingConfig {
        RuliadProofPolicyTrainingConfig {
            enabled: true,
            mode: crate::config::RuliadProofPolicyTrainingMode::StaticExpert,
            scoring: crate::config::RuliadProofPolicyScoring::CompletionLikelihood,
            gradient_scope: crate::config::RuliadProofPolicyGradientScope::FullModel,
            normalization,
            candidate_symmetry:
                crate::config::RuliadProofPolicyCandidateSymmetry::CyclicOrbitAverage,
            presentation_risk: crate::config::RuliadProofPolicyPresentationRisk::Mean,
            weight: 1.0,
            every_steps: 1,
            start_after_steps: 0,
            max_rows_per_update: 1,
            max_presentation_rows_per_update: 64,
            candidates: 4,
            max_completion_tokens: 128,
            ..RuliadProofPolicyTrainingConfig::default()
        }
    }

    #[test]
    fn verifier_schedule_is_explicit_and_periodic() {
        let policy = RuliadProofPolicyTrainingConfig {
            enabled: true,
            every_steps: 4,
            start_after_steps: 8,
            ..RuliadProofPolicyTrainingConfig::default()
        };
        for step in 0..16 {
            assert_eq!(
                verifier_terminal_due(
                    LocalPredictiveCodingTerminalCriterion::RuliadVerifierSet,
                    policy,
                    step,
                ),
                matches!(step, 8 | 12)
            );
        }
        assert!(!verifier_terminal_due(
            LocalPredictiveCodingTerminalCriterion::NextToken,
            policy,
            8,
        ));
    }

    #[test]
    fn semantic_verifier_panel_materializes_sparse_normalized_trie_rows() {
        let device = Default::default();
        let batch = formal_policy_batch(
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
        );
        let prepared = prepare_ruliad_verifier_terminal::<TestBackend>(
            &batch,
            policy(RuliadProofPolicyNormalization::CandidateConditional),
            512,
            272,
            &device,
        )
        .expect("semantic verifier terminal");
        assert_eq!(prepared.semantic_states, 1);
        assert!(
            prepared.decision_rows > 1,
            "semantic answers should form a trie"
        );
        let [rows, time] = prepared.inputs.shape().dims::<2>();
        assert_eq!(rows, prepared.decision_rows);
        assert!(time <= 512);

        let LocalPcTerminalCriterion::CategoricalSetAtPositions {
            positions,
            support_action_mask,
            valid_action_mask,
            row_weights,
            ..
        } = prepared.criterion
        else {
            panic!("verifier panel must use sparse positions");
        };
        let positions = positions.into_data().to_vec::<i64>().expect("positions");
        assert!(
            positions
                .iter()
                .all(|position| *position >= 0 && *position < time as i64)
        );
        let support = support_action_mask
            .into_data()
            .to_vec::<f32>()
            .expect("support mask");
        let valid = valid_action_mask
            .into_data()
            .to_vec::<f32>()
            .expect("valid mask");
        let weights = row_weights.into_data().to_vec::<f32>().expect("weights");
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
        for row in 0..rows {
            let range = row * 272..(row + 1) * 272;
            let support_count = support[range.clone()].iter().filter(|v| **v > 0.0).count();
            let valid_count = valid[range.clone()].iter().filter(|v| **v > 0.0).count();
            assert!(support_count >= valid_count && valid_count >= 1);
            assert!(
                support_count < 272,
                "conditional support must stay candidate-local"
            );
            assert!(
                valid[range]
                    .iter()
                    .zip(&support[row * 272..(row + 1) * 272])
                    .all(|(valid, support)| *valid <= *support)
            );
        }
    }

    #[test]
    fn counterfactual_verifier_panel_materializes_complete_paired_target_groups() {
        let device = Default::default();
        let batch = formal_policy_batch(
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
        );
        let mut config = policy(RuliadProofPolicyNormalization::PrefixConditional);
        config.candidate_symmetry =
            crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation;
        config.counterfactual_targets_per_state = 1;
        config.max_rows_per_update = 2;
        config.max_presentation_rows_per_update = 64;
        config.weight = 0.25;

        let prepared =
            prepare_ruliad_verifier_terminal::<TestBackend>(&batch, config, 512, 272, &device)
                .expect("paired counterfactual verifier terminal");

        assert_eq!(prepared.semantic_states, 2);
        assert_eq!(prepared.stats.base_semantic_states, 1);
        assert_eq!(prepared.stats.counterfactual_semantic_states, 1);
        assert_eq!(prepared.stats.counterfactual_target_shortfall, 0);
        assert!(prepared.decision_rows >= 2);
        let LocalPcTerminalCriterion::CategoricalSetAtPositions { row_weights, .. } =
            prepared.criterion
        else {
            panic!("counterfactual verifier panel must use sparse positions");
        };
        let total_weight = row_weights
            .into_data()
            .to_vec::<f32>()
            .expect("row weights")
            .into_iter()
            .sum::<f32>();
        assert!((total_weight - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn counterfactual_verifier_panel_fixed_prediction_matches_global_backpropagation() {
        let device = burn::tensor::Device::<AutodiffTestBackend>::default();
        let model = crate::train::test_support::deterministic_matrix_parameters(
            verifier_model_with_layers::<AutodiffTestBackend>(&device, 4),
        );
        let batch = formal_policy_batch(
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
        );
        let mut policy = policy(RuliadProofPolicyNormalization::PrefixConditional);
        policy.candidate_symmetry =
            crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation;
        policy.counterfactual_targets_per_state = 1;
        policy.max_rows_per_update = 2;
        policy.max_presentation_rows_per_update = 64;
        let prepared =
            prepare_ruliad_verifier_terminal::<TestBackend>(&batch, policy, 512, 272, &device)
                .expect("counterfactual verifier terminal");
        assert_eq!(prepared.semantic_states, 2);

        let report = super::super::diagnostics::local_predictive_coding_verifier_gradient_fidelity(
            &model,
            prepared,
            &crate::config::LocalPredictiveCodingConfig {
                solver: crate::config::LocalPredictiveCodingSolver::FixedPrediction,
                factor_reduction: crate::config::PredictiveCodingFactorReduction::Sum,
                ..crate::config::LocalPredictiveCodingConfig::default()
            },
        )
        .expect("real verifier-panel gradient fidelity");

        assert!(report.loss_absolute_error < 1.0e-6, "{report:?}");
        assert_eq!(report.pc_step.global_backward_calls, 0);
        assert_eq!(report.pc_gradient_tensors, 9);
        assert!(
            report.global.cosine.is_some_and(|cosine| cosine > 0.999_99),
            "{report:?}"
        );
        assert!(
            report
                .global
                .relative_l2_error
                .is_some_and(|error| error < 1.0e-4),
            "{report:?}"
        );
        for family in &report.parameter_families {
            if family.reference_norm > 1.0e-8 {
                assert!(
                    family.cosine.is_some_and(|cosine| cosine > 0.999_9),
                    "{family:?}"
                );
                assert!(
                    family.relative_l2_error.is_some_and(|error| error < 1.0e-3),
                    "{family:?}"
                );
            }
        }
    }

    #[test]
    fn semantic_energy_verifier_panel_fixed_prediction_matches_global_backpropagation() {
        let device = burn::tensor::Device::<AutodiffTestBackend>::default();
        let model = crate::train::test_support::deterministic_matrix_parameters(
            verifier_model_with_score_head::<AutodiffTestBackend>(&device, 4),
        );
        let batch = formal_policy_batch(
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
        );
        let mut policy = policy(RuliadProofPolicyNormalization::CandidateConditional);
        policy.scoring = crate::config::RuliadProofPolicyScoring::SemanticEnergy;
        policy.candidate_symmetry =
            crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation;
        policy.counterfactual_targets_per_state = 1;
        policy.max_rows_per_update = 2;
        policy.max_presentation_rows_per_update = 8;
        let prepared =
            prepare_ruliad_verifier_terminal::<TestBackend>(&batch, policy, 512, 272, &device)
                .expect("semantic-energy counterfactual verifier terminal");
        assert_eq!(prepared.semantic_states, 2);
        assert_eq!(prepared.decision_rows, 2);
        assert_eq!(prepared.inputs.shape().dims::<2>()[0], 8);
        let LocalPcTerminalCriterion::SequenceEnergySetAtPositions {
            candidates_per_group,
            valid_action_mask,
            row_weights,
            ..
        } = &prepared.criterion
        else {
            panic!("semantic verifier panel must use sequence-energy rows");
        };
        assert_eq!(*candidates_per_group, 4);
        assert_eq!(valid_action_mask.shape().dims::<2>(), [2, 4]);
        assert_eq!(row_weights.shape().dims::<1>(), [2]);

        let report = super::super::diagnostics::local_predictive_coding_verifier_gradient_fidelity(
            &model,
            prepared,
            &crate::config::LocalPredictiveCodingConfig {
                solver: crate::config::LocalPredictiveCodingSolver::FixedPrediction,
                factor_reduction: crate::config::PredictiveCodingFactorReduction::Sum,
                ..crate::config::LocalPredictiveCodingConfig::default()
            },
        )
        .expect("semantic-energy verifier gradient fidelity");

        assert!(report.loss_absolute_error < 2.0e-6, "{report:?}");
        assert_eq!(report.pc_step.global_backward_calls, 0);
        assert_eq!(report.pc_gradient_tensors, 15);
        assert_eq!(report.parameter_families.len(), 15);
        assert!(
            report.global.cosine.is_some_and(|cosine| cosine > 0.999_98),
            "{report:?}"
        );
        assert!(
            report
                .global
                .relative_l2_error
                .is_some_and(|error| error < 2.0e-4),
            "{report:?}"
        );
        for family in &report.parameter_families {
            if family.parameter_family == "sequence_score_bias" {
                assert!(family.reference_norm < 1.0e-6, "{family:?}");
                assert!(family.pc_norm < 1.0e-6, "{family:?}");
            } else if family.reference_norm > 1.0e-6 {
                assert!(
                    family.cosine.is_some_and(|cosine| cosine > 0.999_8),
                    "{family:?}"
                );
                assert!(
                    family.relative_l2_error.is_some_and(|error| error < 2.0e-3),
                    "{family:?}"
                );
            }
        }
    }

    #[test]
    fn dynamic_dagger_panel_is_deterministic_and_executes_without_global_backward() {
        let device = burn::tensor::Device::<AutodiffTestBackend>::default();
        let model = verifier_model::<AutodiffTestBackend>(&device);
        let sampling_model = model.valid();
        let batch = formal_policy_batch(
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
        );
        let mut config = policy(RuliadProofPolicyNormalization::PrefixConditional);
        config.mode = crate::config::RuliadProofPolicyTrainingMode::Dagger;
        config.rollout_steps = 2;
        config.counterfactual_targets_per_state = 1;
        config.max_rows_per_update = 4;
        config.max_presentation_rows_per_update = 64;

        let first = prepare_ruliad_verifier_terminal_at_step::<TestBackend>(
            Some(&sampling_model),
            &batch,
            config,
            512,
            272,
            0,
            &device,
        )
        .expect("dynamic verifier panel");
        let second = prepare_ruliad_verifier_terminal_at_step::<TestBackend>(
            Some(&sampling_model),
            &batch,
            config,
            512,
            272,
            0,
            &device,
        )
        .expect("repeat dynamic verifier panel");
        assert_eq!(first.semantic_states, second.semantic_states);
        assert_eq!(first.decision_rows, second.decision_rows);
        assert_eq!(first.inputs.to_data(), second.inputs.to_data());
        assert_eq!(first.stats.base_semantic_states, 2);
        assert_eq!(first.stats.counterfactual_semantic_states, 2);
        assert_eq!(first.stats.counterfactual_target_shortfall, 0);

        let step = super::super::local_predictive_coding_verifier_train_step(
            &model,
            first,
            &crate::config::LocalPredictiveCodingConfig {
                solver: crate::config::LocalPredictiveCodingSolver::FixedPrediction,
                ..crate::config::LocalPredictiveCodingConfig::default()
            },
            &super::super::LocalPredictiveCodingProfile::default(),
        );
        assert_eq!(step.report.global_backward_calls, 0);
        assert!(burn_pc::diagnostic_scalar_f32(step.loss.inner()).is_finite());
    }

    #[test]
    fn vocabulary_marginal_panel_exposes_the_full_vocabulary_support() {
        let device = Default::default();
        let batch = formal_policy_batch(
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::PresentationIndex,
        );
        let prepared = prepare_ruliad_verifier_terminal::<TestBackend>(
            &batch,
            policy(RuliadProofPolicyNormalization::VocabularyMarginal),
            512,
            272,
            &device,
        )
        .expect("vocabulary-marginal verifier terminal");
        let LocalPcTerminalCriterion::CategoricalSetAtPositions {
            support_action_mask,
            ..
        } = prepared.criterion
        else {
            panic!("verifier panel must use sparse positions");
        };
        let support = support_action_mask
            .into_data()
            .to_vec::<f32>()
            .expect("support mask");
        assert!(
            support
                .iter()
                .all(|value| (*value - 1.0).abs() < f32::EPSILON)
        );
    }

    #[test]
    fn local_pc_profile_materializes_source_selected_verifier_rows() {
        use crate::dataset::{DatasetSplit, TokenSequenceDataset};

        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let profile =
            root.join("config/language/experiments/predictive_coding/local-pc-verifier-1m.toml");
        let mut training =
            crate::config::load_training_config(&[profile]).expect("load local PC profile");
        if let crate::config::DatasetSourceConfig::UniversalityRuliad { config } =
            &mut training.dataset.source
        {
            *config = root.join(&*config);
        } else {
            panic!("local PC profile must use the Ruliad corpus");
        }
        let datasets = crate::train::utils::prepare_datasets(&training.dataset, &training.training)
            .expect("prepare local PC datasets");
        let vocab = datasets.train.tokenizer().len();
        let batch = TokenSequenceDataset::source_selected_ruliad_policy_batch(
            datasets.train.as_ref(),
            DatasetSplit::Train,
            0,
            0,
            32,
            4,
        )
        .expect("source-selected proof-policy batch");
        assert!(batch.samples.iter().all(|sample| matches!(
            sample.item.spec,
            Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
                task: burn_dragon_universality::RuliadTaskKind::SelectProofAction,
                ..
            })
        )));

        let mut config = policy(RuliadProofPolicyNormalization::PrefixConditional);
        config.candidate_symmetry =
            crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation;
        config.max_rows_per_update = 8;
        config.max_presentation_rows_per_update = 64;
        config.stratified_difficulty_levels = 4;
        let prepared = prepare_ruliad_verifier_terminal::<TestBackend>(
            &batch,
            config,
            training.training.block_size,
            vocab,
            &Default::default(),
        )
        .expect("real source-selected batch must yield verifier rows");
        assert!(prepared.semantic_states > 0);
        assert!(prepared.decision_rows <= config.max_presentation_rows_per_update);
    }
}
