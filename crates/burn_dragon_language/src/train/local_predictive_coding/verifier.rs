use burn::tensor::{
    Int, Tensor, TensorData,
    backend::{AutodiffBackend, Backend},
};
use burn_dragon_core::DragonModel;
use std::collections::{BTreeMap, HashSet};

use crate::config::{
    LocalPredictiveCodingTerminalCriterion, RuliadProofPolicyNormalization,
    RuliadProofPolicyTrainingConfig,
};
use crate::train::ruliad_objective_fingerprint::{
    RuliadObjectivePanelFingerprint, RuliadObjectiveSequenceKind,
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct RuliadVerifierPanelStats {
    pub policy_batch_fingerprint: u64,
    /// Stable identity of the fully prepared verifier objective. Unlike the
    /// source-batch fingerprint, this includes model-visited DAgger states,
    /// candidate supports, verifier labels, target groups, and row weights.
    pub objective_panel_fingerprint: u64,
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
    pub target_group_conditional_groups: usize,
    pub target_group_conditional_rows: usize,
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
    pub supervised_action_tokens: usize,
    pub candidate_target_tokens: usize,
    pub equivalent_target_tokens: usize,
    pub prefix_branch_rows: usize,
    pub prefix_candidate_tokens: usize,
    pub prefix_equivalent_tokens: usize,
    pub original_prompt_tokens: usize,
    pub retained_prompt_tokens: usize,
    pub maximum_original_prompt_tokens: usize,
    pub maximum_retained_prompt_tokens: usize,
    pub truncated_presentations: usize,
    pub difficulty_sample_groups: BTreeMap<usize, usize>,
    pub difficulty_visited_states: BTreeMap<usize, usize>,
    pub difficulty_expert_rows: BTreeMap<usize, usize>,
    pub expert_selected_index_histogram: BTreeMap<usize, usize>,
    pub expert_equivalent_index_histogram: BTreeMap<usize, usize>,
    pub model_selected_index_histogram: BTreeMap<usize, usize>,
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
            propagate_hidden_gradient,
            eps,
        } => LocalPcTerminalCriterion::CategoricalSetAtPositions {
            positions: Tensor::from_inner(positions),
            support_action_mask: Tensor::from_inner(support_action_mask),
            valid_action_mask: Tensor::from_inner(valid_action_mask),
            row_weights: Tensor::from_inner(row_weights),
            propagate_hidden_gradient,
            eps,
        },
        LocalPcTerminalCriterion::SequenceCompletionSet {
            targets,
            token_mask,
            support_action_mask,
            valid_action_mask,
            row_weights,
            candidates_per_group,
            propagate_hidden_gradient,
            eps,
        } => LocalPcTerminalCriterion::SequenceCompletionSet {
            targets: Tensor::from_inner(targets),
            token_mask: Tensor::from_inner(token_mask),
            support_action_mask: Tensor::from_inner(support_action_mask),
            valid_action_mask: Tensor::from_inner(valid_action_mask),
            row_weights: Tensor::from_inner(row_weights),
            candidates_per_group,
            propagate_hidden_gradient,
            eps,
        },
        LocalPcTerminalCriterion::SequenceEnergySetAtPositions {
            prompt_positions,
            terminal_positions,
            support_action_mask,
            valid_action_mask,
            target_action_weights,
            row_weights,
            candidates_per_group,
            propagate_hidden_gradient,
            eps,
        } => LocalPcTerminalCriterion::SequenceEnergySetAtPositions {
            prompt_positions: Tensor::from_inner(prompt_positions),
            terminal_positions: Tensor::from_inner(terminal_positions),
            support_action_mask: Tensor::from_inner(support_action_mask),
            valid_action_mask: Tensor::from_inner(valid_action_mask),
            target_action_weights: target_action_weights.map(Tensor::from_inner),
            row_weights: Tensor::from_inner(row_weights),
            candidates_per_group,
            propagate_hidden_gradient,
            eps,
        },
        LocalPcTerminalCriterion::SequenceResidualEnergySetAtPositions {
            targets,
            token_mask,
            prompt_positions,
            terminal_positions,
            support_action_mask,
            valid_action_mask,
            target_action_weights,
            row_weights,
            candidates_per_group,
            propagate_hidden_gradient,
            propagate_language_prior_gradient,
            eps,
        } => LocalPcTerminalCriterion::SequenceResidualEnergySetAtPositions {
            targets: Tensor::from_inner(targets),
            token_mask: Tensor::from_inner(token_mask),
            prompt_positions: Tensor::from_inner(prompt_positions),
            terminal_positions: Tensor::from_inner(terminal_positions),
            support_action_mask: Tensor::from_inner(support_action_mask),
            valid_action_mask: Tensor::from_inner(valid_action_mask),
            target_action_weights: target_action_weights.map(Tensor::from_inner),
            row_weights: Tensor::from_inner(row_weights),
            candidates_per_group,
            propagate_hidden_gradient,
            propagate_language_prior_gradient,
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
    SequenceCompletion {
        prompt: Vec<i64>,
        candidates: Vec<Vec<i64>>,
        valid_indices: Vec<usize>,
        target_group: usize,
        weight: f32,
    },
    SequenceEnergy {
        inputs: Vec<Vec<i64>>,
        prompt_position: usize,
        terminal_positions: Vec<usize>,
        valid_indices: Vec<usize>,
        target_action_weights: Option<Vec<f32>>,
        target_group: usize,
        weight: f32,
    },
}

fn verifier_decision_rows_fingerprint(
    rows: &[VerifierDecisionRow],
    scoring: crate::config::RuliadProofPolicyScoring,
) -> Option<u64> {
    let sequence_kind = match scoring {
        crate::config::RuliadProofPolicyScoring::CompletionLikelihood => {
            RuliadObjectiveSequenceKind::CompletionLikelihood
        }
        crate::config::RuliadProofPolicyScoring::SemanticEnergy => {
            RuliadObjectiveSequenceKind::SemanticEnergy
        }
        crate::config::RuliadProofPolicyScoring::ResidualEnergy => {
            RuliadObjectiveSequenceKind::ResidualEnergy
        }
    };
    let mut panel = RuliadObjectivePanelFingerprint::new(rows.len());
    for row in rows {
        match row {
            VerifierDecisionRow::Prefix {
                inputs,
                position,
                support_tokens,
                valid_tokens,
                weight,
            } => {
                panel.push_prefix(inputs, *position, support_tokens, valid_tokens, *weight);
            }
            VerifierDecisionRow::SequenceCompletion {
                prompt,
                candidates,
                valid_indices,
                target_group,
                weight,
            } => {
                panel.push_sequence(
                    RuliadObjectiveSequenceKind::CompletionLikelihood,
                    prompt,
                    candidates,
                    valid_indices,
                    None,
                    *target_group,
                    *weight,
                );
            }
            VerifierDecisionRow::SequenceEnergy {
                inputs,
                prompt_position,
                terminal_positions,
                valid_indices,
                target_action_weights,
                target_group,
                weight,
            } => {
                let first = inputs.first()?;
                let prompt_len = prompt_position.checked_add(1)?;
                let prompt = first.get(..prompt_len)?;
                let mut candidates = Vec::with_capacity(inputs.len());
                for input in inputs {
                    if !input.starts_with(prompt) {
                        return None;
                    }
                    candidates.push(input[prompt_len..].to_vec());
                }
                if terminal_positions.len() != inputs.len()
                    || terminal_positions
                        .iter()
                        .zip(inputs)
                        .any(|(position, input)| *position != input.len().saturating_sub(1))
                {
                    return None;
                }
                panel.push_sequence(
                    sequence_kind,
                    prompt,
                    &candidates,
                    valid_indices,
                    target_action_weights.as_deref(),
                    *target_group,
                    *weight,
                );
            }
        }
    }
    panel.finish()
}

#[derive(Clone)]
struct VerifierTrajectory {
    sample_index: usize,
    difficulty_level: usize,
    is_dagger: bool,
    max_depth: usize,
    start_step: usize,
    answer_contract: burn_dragon_universality::ruliad::RuliadProofActionAnswerContract,
    state: burn_dragon_universality::ruliad::RuliadProofPolicyState,
}

struct PreparedVerifierState {
    canonical_prompt: Vec<i64>,
    rows: Vec<VerifierDecisionRow>,
    rotations: Vec<usize>,
    request: crate::train::ruliad_policy::EncodedRuliadProofActionRequest,
    original_prompt_tokens: usize,
    retained_prompt_tokens: usize,
    maximum_original_prompt_tokens: usize,
    maximum_retained_prompt_tokens: usize,
    truncated_presentations: usize,
    selected_indices: Vec<usize>,
    equivalent_indices: Vec<Vec<usize>>,
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
            | LocalPredictiveCodingTerminalCriterion::RuliadVerifierSetJoint
    ) && policy.enabled
        && policy.every_steps > 0
        && absolute_step >= policy.start_after_steps
        && absolute_step.is_multiple_of(policy.every_steps)
}

pub(crate) fn verifier_terminal_preserves_primary(
    terminal: LocalPredictiveCodingTerminalCriterion,
) -> bool {
    matches!(
        terminal,
        LocalPredictiveCodingTerminalCriterion::RuliadVerifierSetJoint
    )
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
    let mut original_prompt_tokens = 0usize;
    let mut retained_prompt_tokens = 0usize;
    let mut maximum_original_prompt_tokens = 0usize;
    let mut maximum_retained_prompt_tokens = 0usize;
    let mut truncated_presentations = 0usize;
    let mut difficulty_sample_groups = BTreeMap::<usize, usize>::new();
    let mut difficulty_visited_states = BTreeMap::<usize, usize>::new();
    let mut difficulty_expert_rows = BTreeMap::<usize, usize>::new();
    let mut expert_selected_index_histogram = BTreeMap::<usize, usize>::new();
    let mut expert_equivalent_index_histogram = BTreeMap::<usize, usize>::new();
    let mut model_selected_index_histogram = BTreeMap::<usize, usize>::new();

    let prepare_state =
        |problem: &burn_dragon_universality::ruliad::RuliadProofProblem,
         actions: &burn_dragon_universality::ruliad::RuliadProofActionSet,
         answer_contract: burn_dragon_universality::ruliad::RuliadProofActionAnswerContract,
         presentation_index: usize,
         target_group_offset: usize,
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
                    &crate::train::ruliad_policy::ruliad_proof_policy_prompt(
                        config.prompt_context,
                        problem,
                        actions,
                    )
                    .ok()?,
                )
                .into_iter()
                .map(i64::from)
                .collect::<Vec<_>>();
            let request =
                crate::train::ruliad_policy::encode_ruliad_proof_action_request_with_rotations(
                    problem,
                    actions,
                    answer_contract,
                    config.prompt_context,
                    &rotations,
                    &policy_batch.tokenization,
                    policy_batch.stop_token_id,
                    block_size,
                    completion_budget,
                )
                .ok()?;
            let mut state_rows = Vec::<VerifierDecisionRow>::new();
            let mut selected_indices = Vec::with_capacity(request.presentations.len());
            let mut equivalent_indices = Vec::with_capacity(request.presentations.len());
            for (presentation_slot, presentation) in request.presentations.iter().enumerate() {
                let target_group = target_group_offset.saturating_add(presentation_slot);
                let presented = actions.rotate_left(presentation.rotation).ok()?;
                selected_indices.push(presented.selected_index);
                equivalent_indices.push(presented.equivalent_indices.clone());
                let prompt_tokens = &presentation.prompt_tokens;
                let candidates = &presentation.candidate_tokens;
                match config.scoring {
                    crate::config::RuliadProofPolicyScoring::CompletionLikelihood => {
                        if answer_contract
                            == burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep
                            && config.normalization
                                == RuliadProofPolicyNormalization::CandidateConditional
                        {
                            state_rows.push(VerifierDecisionRow::SequenceCompletion {
                                prompt: prompt_tokens.clone(),
                                candidates: candidates.clone(),
                                valid_indices: presented.equivalent_indices.clone(),
                                target_group,
                                weight: config.weight / rotations.len().max(1) as f32,
                            });
                        } else {
                            let branches =
                                crate::train::ruliad_policy::semantic_candidate_trie_branches(
                                    candidates,
                                    &presented.equivalent_indices,
                                )
                                .ok()?;
                            let branch_weight = config.weight
                                / rotations.len().max(1) as f32
                                / branches.len().max(1) as f32;
                            for branch in branches {
                                let max_prompt =
                                    block_size.saturating_sub(branch.prefix.len()).max(1);
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
                    }
                    crate::config::RuliadProofPolicyScoring::SemanticEnergy
                    | crate::config::RuliadProofPolicyScoring::ResidualEnergy => {
                        if prompt_tokens.is_empty() || presented.equivalent_indices.is_empty() {
                            return None;
                        }
                        let prompt_position = prompt_tokens.len() - 1;
                        let mut sequence_inputs = Vec::with_capacity(candidates.len());
                        let mut terminal_positions = Vec::with_capacity(candidates.len());
                        for candidate in candidates {
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
                            target_action_weights: (config.target
                                == crate::config::RuliadProofPolicyTarget::VerifiedProgressDistribution)
                                .then(|| {
                                    presented
                                        .candidate_progress_units()
                                        .into_iter()
                                        .map(|weight| weight as f32)
                                        .collect()
                                }),
                            target_group,
                            weight: config.weight / rotations.len().max(1) as f32,
                        });
                    }
                }
            }
            let state_original_prompt_tokens = request
                .presentations
                .iter()
                .map(|presentation| presentation.original_prompt_token_count)
                .sum::<usize>();
            let state_retained_prompt_tokens = request
                .presentations
                .iter()
                .map(|presentation| presentation.prompt_tokens.len())
                .sum::<usize>();
            let state_maximum_original_prompt_tokens = request
                .presentations
                .iter()
                .map(|presentation| presentation.original_prompt_token_count)
                .max()
                .unwrap_or_default();
            let state_maximum_retained_prompt_tokens = request
                .presentations
                .iter()
                .map(|presentation| presentation.prompt_tokens.len())
                .max()
                .unwrap_or_default();
            let state_truncated_presentations = request
                .presentations
                .iter()
                .filter(|presentation| {
                    presentation.prompt_tokens.len() < presentation.original_prompt_token_count
                })
                .count();
            Some(PreparedVerifierState {
                canonical_prompt,
                rows: state_rows,
                rotations,
                request,
                original_prompt_tokens: state_original_prompt_tokens,
                retained_prompt_tokens: state_retained_prompt_tokens,
                maximum_original_prompt_tokens: state_maximum_original_prompt_tokens,
                maximum_retained_prompt_tokens: state_maximum_retained_prompt_tokens,
                truncated_presentations: state_truncated_presentations,
                selected_indices,
                equivalent_indices,
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
        let difficulty_level = sample.item.difficulty_level.unwrap_or_default();
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
            difficulty_level,
            is_dagger,
            max_depth,
            start_step,
            answer_contract: *action_answer_contract,
            state,
        });
        *difficulty_sample_groups
            .entry(difficulty_level)
            .or_default() += 1;
    }

    let source_trajectory_count = trajectories.len();
    let desired_dagger_trajectories = plan.dagger_trajectories_for_samples(source_trajectory_count);
    let existing_dagger_trajectories = trajectories
        .iter()
        .filter(|trajectory| trajectory.is_dagger)
        .count();
    let missing_dagger_trajectories =
        desired_dagger_trajectories.saturating_sub(existing_dagger_trajectories);
    let paired_reuse = trajectories
        .iter()
        .filter(|trajectory| !trajectory.is_dagger)
        .take(missing_dagger_trajectories)
        .cloned()
        .collect::<Vec<_>>();
    for mut trajectory in paired_reuse {
        trajectory.is_dagger = true;
        trajectory.max_depth = 1;
        nonzero_start_trajectories =
            nonzero_start_trajectories.saturating_add(usize::from(trajectory.start_step > 0));
        start_step_sum = start_step_sum.saturating_add(trajectory.start_step);
        *difficulty_sample_groups
            .entry(trajectory.difficulty_level)
            .or_default() += 1;
        trajectories.push(trajectory);
    }
    let dagger_trajectory_count = trajectories
        .iter()
        .filter(|trajectory| trajectory.is_dagger)
        .count();
    for (dagger_index, trajectory) in trajectories
        .iter_mut()
        .filter(|trajectory| trajectory.is_dagger)
        .enumerate()
    {
        trajectory.max_depth = plan.dagger_depth_for_count(dagger_index, dagger_trajectory_count);
    }
    let effective_rollout_steps = plan.rollout_steps_for_dagger_count(dagger_trajectory_count);

    for rollout_depth in 0..effective_rollout_steps {
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
                base_semantic_states.saturating_mul(config.presentations_per_state()),
                None,
            ) else {
                continue;
            };
            *difficulty_visited_states
                .entry(trajectory.difficulty_level)
                .or_default() += 1;
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
                    base_semantic_states.saturating_mul(config.presentations_per_state()),
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
            // Counterfactual supervision is a paired experiment. Retaining only the base row when
            // an alternate target cannot be materialized creates a target-frequency shortcut and
            // makes the independent and conditional halves of the joint objective incomparable.
            let requires_complete_target_group = config.counterfactual_targets_per_state > 0;
            let usable_target_group = !requires_complete_target_group || complete_target_group;
            let unique_target_group = usable_target_group
                && prepared_states
                    .iter()
                    .all(|prepared| !visited_prompts.contains(&prepared.canonical_prompt));
            if group_rows == 0 || rows.len().saturating_add(group_rows) > row_budget {
                continue;
            }
            if unique_target_group {
                let variants_added = prepared_states.len();
                for prepared in prepared_states {
                    visited_prompts.insert(prepared.canonical_prompt);
                    for selected_index in prepared.selected_indices {
                        *expert_selected_index_histogram
                            .entry(selected_index)
                            .or_default() += 1;
                    }
                    for equivalent_indices in prepared.equivalent_indices {
                        for equivalent_index in equivalent_indices {
                            *expert_equivalent_index_histogram
                                .entry(equivalent_index)
                                .or_default() += 1;
                        }
                    }
                    original_prompt_tokens =
                        original_prompt_tokens.saturating_add(prepared.original_prompt_tokens);
                    retained_prompt_tokens =
                        retained_prompt_tokens.saturating_add(prepared.retained_prompt_tokens);
                    maximum_original_prompt_tokens =
                        maximum_original_prompt_tokens.max(prepared.maximum_original_prompt_tokens);
                    maximum_retained_prompt_tokens =
                        maximum_retained_prompt_tokens.max(prepared.maximum_retained_prompt_tokens);
                    truncated_presentations =
                        truncated_presentations.saturating_add(prepared.truncated_presentations);
                    rows.extend(prepared.rows);
                }
                semantic_states = semantic_states.saturating_add(variants_added);
                base_semantic_states = base_semantic_states.saturating_add(1);
                counterfactual_semantic_states =
                    counterfactual_semantic_states.saturating_add(variants_added.saturating_sub(1));
                static_expert_states = static_expert_states.saturating_add(
                    variants_added.saturating_mul(usize::from(!trajectory.is_dagger)),
                );
                dagger_expert_states = dagger_expert_states.saturating_add(
                    variants_added.saturating_mul(usize::from(trajectory.is_dagger)),
                );
                model_visited_states =
                    model_visited_states
                        .saturating_add(variants_added.saturating_mul(usize::from(
                            trajectory.is_dagger && rollout_depth > 0,
                        )));
                *difficulty_expert_rows
                    .entry(trajectory.difficulty_level)
                    .or_default() += variants_added;
                rollout_depth_reached = rollout_depth_reached.max(rollout_depth.saturating_add(1));
            }
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
            *model_selected_index_histogram
                .entry(decision.selected_semantic_index)
                .or_default() += 1;
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
    let mut supervised_action_tokens = 0usize;
    let mut candidate_target_tokens = 0usize;
    let mut equivalent_target_tokens = 0usize;
    let mut prefix_branch_rows = 0usize;
    let mut prefix_candidate_tokens = 0usize;
    let mut prefix_equivalent_tokens = 0usize;
    for row in &rows {
        match row {
            VerifierDecisionRow::SequenceCompletion {
                candidates,
                valid_indices,
                ..
            } => {
                let candidate_tokens = candidates.iter().map(Vec::len).sum::<usize>();
                let equivalent_tokens = valid_indices
                    .iter()
                    .filter_map(|index| candidates.get(*index))
                    .map(Vec::len)
                    .sum::<usize>();
                supervised_action_tokens =
                    supervised_action_tokens.saturating_add(candidate_tokens);
                candidate_target_tokens = candidate_target_tokens.saturating_add(candidate_tokens);
                equivalent_target_tokens =
                    equivalent_target_tokens.saturating_add(equivalent_tokens);
            }
            VerifierDecisionRow::Prefix {
                support_tokens,
                valid_tokens,
                ..
            } => {
                supervised_action_tokens = supervised_action_tokens.saturating_add(1);
                candidate_target_tokens =
                    candidate_target_tokens.saturating_add(support_tokens.len());
                equivalent_target_tokens =
                    equivalent_target_tokens.saturating_add(valid_tokens.len());
                prefix_branch_rows = prefix_branch_rows.saturating_add(1);
                prefix_candidate_tokens =
                    prefix_candidate_tokens.saturating_add(support_tokens.len());
                prefix_equivalent_tokens =
                    prefix_equivalent_tokens.saturating_add(valid_tokens.len());
            }
            VerifierDecisionRow::SequenceEnergy { .. } => {}
        }
    }
    let answer_contract = trajectories
        .first()
        .map(|trajectory| trajectory.answer_contract.label())
        .unwrap_or_default();
    let objective_panel_fingerprint = verifier_decision_rows_fingerprint(&rows, config.scoring)?;
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
        policy_batch_fingerprint: policy_batch.fingerprint(),
        objective_panel_fingerprint,
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
        target_group_conditional_groups: if config
            .counterfactual_objective
            .uses_target_group_support()
        {
            base_semantic_states.saturating_mul(config.presentations_per_state())
        } else {
            0
        },
        target_group_conditional_rows: if config
            .counterfactual_objective
            .uses_target_group_support()
        {
            row_count
        } else {
            0
        },
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
        supervised_action_tokens,
        candidate_target_tokens,
        equivalent_target_tokens,
        prefix_branch_rows,
        prefix_candidate_tokens,
        prefix_equivalent_tokens,
        original_prompt_tokens,
        retained_prompt_tokens,
        maximum_original_prompt_tokens,
        maximum_retained_prompt_tokens,
        truncated_presentations,
        difficulty_sample_groups,
        difficulty_visited_states,
        difficulty_expert_rows,
        expert_selected_index_histogram,
        expert_equivalent_index_histogram,
        model_selected_index_histogram,
    };
    let (inputs, criterion) = match config.scoring {
        crate::config::RuliadProofPolicyScoring::CompletionLikelihood => {
            if matches!(
                rows.first(),
                Some(VerifierDecisionRow::SequenceCompletion { .. })
            ) {
                let candidates_per_group = rows.first().and_then(|row| match row {
                    VerifierDecisionRow::SequenceCompletion { candidates, .. } => {
                        Some(candidates.len())
                    }
                    VerifierDecisionRow::Prefix { .. }
                    | VerifierDecisionRow::SequenceEnergy { .. } => None,
                })?;
                if candidates_per_group < 2 {
                    return None;
                }
                let sequence_len = rows
                    .iter()
                    .filter_map(|row| match row {
                        VerifierDecisionRow::SequenceCompletion {
                            prompt, candidates, ..
                        } => candidates
                            .iter()
                            .map(|candidate| {
                                prompt
                                    .len()
                                    .saturating_add(candidate.len())
                                    .saturating_sub(1)
                            })
                            .max(),
                        VerifierDecisionRow::Prefix { .. }
                        | VerifierDecisionRow::SequenceEnergy { .. } => None,
                    })
                    .max()?
                    .max(1);
                let flat_rows = row_count.saturating_mul(candidates_per_group);
                let mut input_values = vec![0_i64; flat_rows.saturating_mul(sequence_len)];
                let mut target_values = vec![0_i64; flat_rows.saturating_mul(sequence_len)];
                let mut token_mask = vec![0.0_f32; flat_rows.saturating_mul(sequence_len)];
                let mut valid = vec![0.0_f32; row_count.saturating_mul(candidates_per_group)];
                let mut target_groups = Vec::with_capacity(row_count);
                let mut weights = Vec::with_capacity(row_count);
                for (group_index, row) in rows.into_iter().enumerate() {
                    let VerifierDecisionRow::SequenceCompletion {
                        prompt,
                        candidates,
                        valid_indices,
                        target_group,
                        weight,
                    } = row
                    else {
                        return None;
                    };
                    if prompt.is_empty() || candidates.len() != candidates_per_group {
                        return None;
                    }
                    for (candidate_index, candidate) in candidates.into_iter().enumerate() {
                        if candidate.is_empty() {
                            return None;
                        }
                        let flat_index = group_index * candidates_per_group + candidate_index;
                        let offset = flat_index * sequence_len;
                        let input_len = prompt
                            .len()
                            .saturating_add(candidate.len())
                            .saturating_sub(1);
                        if input_len == 0 || input_len > sequence_len {
                            return None;
                        }
                        input_values[offset..offset + prompt.len()].copy_from_slice(&prompt);
                        if candidate.len() > 1 {
                            input_values[offset + prompt.len()..offset + input_len]
                                .copy_from_slice(&candidate[..candidate.len() - 1]);
                        }
                        for (candidate_position, target) in candidate.into_iter().enumerate() {
                            let position = prompt
                                .len()
                                .saturating_sub(1)
                                .saturating_add(candidate_position);
                            target_values[offset + position] = target;
                            token_mask[offset + position] = 1.0;
                        }
                    }
                    for candidate_index in valid_indices {
                        if candidate_index >= candidates_per_group {
                            return None;
                        }
                        valid[group_index * candidates_per_group + candidate_index] = 1.0;
                    }
                    target_groups.push(target_group);
                    weights.push(weight);
                }
                let support = crate::train::ruliad_policy::target_group_candidate_support_mask(
                    config.counterfactual_objective.uses_target_group_support(),
                    &valid,
                    &target_groups,
                    candidates_per_group,
                    config.target_variants_per_state(),
                )?;
                return Some(PreparedRuliadVerifierTerminal {
                    inputs: Tensor::from_data(
                        TensorData::new(input_values, [flat_rows, sequence_len]),
                        device,
                    ),
                    criterion: LocalPcTerminalCriterion::SequenceCompletionSet {
                        targets: Tensor::from_data(
                            TensorData::new(target_values, [flat_rows, sequence_len]),
                            device,
                        ),
                        token_mask: Tensor::from_data(
                            TensorData::new(token_mask, [flat_rows, sequence_len]),
                            device,
                        ),
                        support_action_mask: Tensor::from_data(
                            TensorData::new(support, [row_count, candidates_per_group]),
                            device,
                        ),
                        valid_action_mask: Tensor::from_data(
                            TensorData::new(valid, [row_count, candidates_per_group]),
                            device,
                        ),
                        row_weights: Tensor::from_data(
                            TensorData::new(weights, [row_count]),
                            device,
                        ),
                        candidates_per_group,
                        propagate_hidden_gradient: config.gradient_scope
                            != crate::config::RuliadProofPolicyGradientScope::LanguageHeadOnly,
                        eps: 1.0e-12,
                    },
                    semantic_states,
                    decision_rows: row_count,
                    stats,
                });
            }
            let sequence_len = rows
                .iter()
                .filter_map(|row| match row {
                    VerifierDecisionRow::Prefix { inputs, .. } => Some(inputs.len()),
                    VerifierDecisionRow::SequenceCompletion { .. }
                    | VerifierDecisionRow::SequenceEnergy { .. } => None,
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
                    propagate_hidden_gradient: config.gradient_scope
                        != crate::config::RuliadProofPolicyGradientScope::LanguageHeadOnly,
                    eps: 1.0e-12,
                },
            )
        }
        crate::config::RuliadProofPolicyScoring::SemanticEnergy
        | crate::config::RuliadProofPolicyScoring::ResidualEnergy => {
            let candidates_per_group = rows.first().and_then(|row| match row {
                VerifierDecisionRow::SequenceEnergy { inputs, .. } => Some(inputs.len()),
                VerifierDecisionRow::Prefix { .. }
                | VerifierDecisionRow::SequenceCompletion { .. } => None,
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
                    VerifierDecisionRow::Prefix { .. }
                    | VerifierDecisionRow::SequenceCompletion { .. } => None,
                })
                .max()?
                .max(1);
            let flat_rows = row_count.saturating_mul(candidates_per_group);
            let mut input_values = vec![0_i64; flat_rows.saturating_mul(sequence_len)];
            let mut target_values = vec![0_i64; flat_rows.saturating_mul(sequence_len)];
            let mut token_mask = vec![0.0_f32; flat_rows.saturating_mul(sequence_len)];
            let mut prompt_positions = Vec::with_capacity(flat_rows);
            let mut terminal_positions = Vec::with_capacity(flat_rows);
            let mut valid = vec![0.0_f32; row_count.saturating_mul(candidates_per_group)];
            let mut target_weights = (config.target
                == crate::config::RuliadProofPolicyTarget::VerifiedProgressDistribution)
                .then(|| vec![0.0_f32; row_count.saturating_mul(candidates_per_group)]);
            let mut target_groups = Vec::with_capacity(row_count);
            let mut weights = Vec::with_capacity(row_count);
            for (group_index, row) in rows.into_iter().enumerate() {
                let VerifierDecisionRow::SequenceEnergy {
                    inputs,
                    prompt_position,
                    terminal_positions: row_terminal_positions,
                    valid_indices,
                    target_action_weights: row_target_weights,
                    target_group,
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
                    let terminal_position = row_terminal_positions[candidate_index];
                    if config.scoring == crate::config::RuliadProofPolicyScoring::ResidualEnergy {
                        if terminal_position <= prompt_position || terminal_position >= input.len()
                        {
                            return None;
                        }
                        for position in prompt_position..terminal_position {
                            target_values[offset + position] = input[position + 1];
                            token_mask[offset + position] = 1.0;
                        }
                    }
                    prompt_positions.push(i64::try_from(prompt_position).ok()?);
                    terminal_positions.push(i64::try_from(terminal_position).ok()?);
                }
                for candidate_index in valid_indices {
                    if candidate_index >= candidates_per_group {
                        return None;
                    }
                    valid[group_index * candidates_per_group + candidate_index] = 1.0;
                }
                match (&mut target_weights, row_target_weights) {
                    (Some(target_weights), Some(row_target_weights))
                        if row_target_weights.len() == candidates_per_group =>
                    {
                        let start = group_index * candidates_per_group;
                        target_weights[start..start + candidates_per_group]
                            .copy_from_slice(&row_target_weights);
                    }
                    (None, None) => {}
                    _ => return None,
                }
                target_groups.push(target_group);
                weights.push(weight);
            }
            let support = crate::train::ruliad_policy::target_group_candidate_support_mask(
                config.counterfactual_objective.uses_target_group_support(),
                &valid,
                &target_groups,
                candidates_per_group,
                config.target_variants_per_state(),
            )?;
            let support_action_mask = Tensor::from_data(
                TensorData::new(support, [row_count, candidates_per_group]),
                device,
            );
            let valid_action_mask = Tensor::from_data(
                TensorData::new(valid, [row_count, candidates_per_group]),
                device,
            );
            let target_action_weights = target_weights.map(|weights| {
                Tensor::from_data(
                    TensorData::new(weights, [row_count, candidates_per_group]),
                    device,
                )
            });
            let row_weights = Tensor::from_data(TensorData::new(weights, [row_count]), device);
            let prompt_positions =
                Tensor::from_data(TensorData::new(prompt_positions, [flat_rows]), device);
            let terminal_positions =
                Tensor::from_data(TensorData::new(terminal_positions, [flat_rows]), device);
            let criterion = match config.scoring {
                crate::config::RuliadProofPolicyScoring::SemanticEnergy => {
                    LocalPcTerminalCriterion::SequenceEnergySetAtPositions {
                        prompt_positions,
                        terminal_positions,
                        support_action_mask,
                        valid_action_mask,
                        target_action_weights,
                        row_weights,
                        candidates_per_group,
                        propagate_hidden_gradient: config.gradient_scope
                            != crate::config::RuliadProofPolicyGradientScope::ScoreHeadOnly,
                        eps: 1.0e-12,
                    }
                }
                crate::config::RuliadProofPolicyScoring::ResidualEnergy => {
                    LocalPcTerminalCriterion::SequenceResidualEnergySetAtPositions {
                        targets: Tensor::from_data(
                            TensorData::new(target_values, [flat_rows, sequence_len]),
                            device,
                        ),
                        token_mask: Tensor::from_data(
                            TensorData::new(token_mask, [flat_rows, sequence_len]),
                            device,
                        ),
                        prompt_positions,
                        terminal_positions,
                        support_action_mask,
                        valid_action_mask,
                        target_action_weights,
                        row_weights,
                        candidates_per_group,
                        propagate_hidden_gradient: config.gradient_scope
                            != crate::config::RuliadProofPolicyGradientScope::ScoreHeadOnly,
                        propagate_language_prior_gradient: config.gradient_scope
                            == crate::config::RuliadProofPolicyGradientScope::FullModel,
                        eps: 1.0e-12,
                    }
                }
                crate::config::RuliadProofPolicyScoring::CompletionLikelihood => unreachable!(),
            };
            (
                Tensor::from_data(
                    TensorData::new(input_values, [flat_rows, sequence_len]),
                    device,
                ),
                criterion,
            )
        }
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

    #[test]
    fn verifier_objective_panel_fingerprint_is_stable_and_label_sensitive() {
        let row = |valid_indices| VerifierDecisionRow::SequenceEnergy {
            inputs: vec![vec![1, 2, 3], vec![1, 2, 4]],
            prompt_position: 1,
            terminal_positions: vec![2, 2],
            valid_indices,
            target_action_weights: Some(vec![1.0, 0.0]),
            target_group: 7,
            weight: 0.5,
        };
        let first = verifier_decision_rows_fingerprint(
            &[row(vec![0])],
            crate::config::RuliadProofPolicyScoring::ResidualEnergy,
        );
        let repeated = verifier_decision_rows_fingerprint(
            &[row(vec![0])],
            crate::config::RuliadProofPolicyScoring::ResidualEnergy,
        );
        let relabelled = verifier_decision_rows_fingerprint(
            &[row(vec![1])],
            crate::config::RuliadProofPolicyScoring::ResidualEnergy,
        );

        assert_eq!(first, repeated);
        assert_ne!(first, relabelled);
    }

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
                action_candidate_count: Some(actions.candidates.len()),
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
            sampling_metadata: None,
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
            assert_eq!(
                verifier_terminal_due(
                    LocalPredictiveCodingTerminalCriterion::RuliadVerifierSetJoint,
                    policy,
                    step,
                ),
                matches!(step, 8 | 12)
            );
        }
        assert!(!verifier_terminal_preserves_primary(
            LocalPredictiveCodingTerminalCriterion::RuliadVerifierSet
        ));
        assert!(verifier_terminal_preserves_primary(
            LocalPredictiveCodingTerminalCriterion::RuliadVerifierSetJoint
        ));
        assert!(!verifier_terminal_due(
            LocalPredictiveCodingTerminalCriterion::NextToken,
            policy,
            8,
        ));
    }

    #[test]
    fn semantic_candidate_completion_panel_materializes_complete_sequence_rows() {
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
        assert_eq!(prepared.decision_rows, 4);
        let [rows, time] = prepared.inputs.shape().dims::<2>();
        assert_eq!(rows, prepared.decision_rows * 4);
        assert!(time <= 512);

        let LocalPcTerminalCriterion::SequenceCompletionSet {
            targets,
            token_mask,
            valid_action_mask,
            row_weights,
            candidates_per_group,
            ..
        } = prepared.criterion
        else {
            panic!("candidate-conditional verifier panel must use complete sequence rows");
        };
        assert_eq!(candidates_per_group, 4);
        assert_eq!(targets.shape().dims::<2>(), [rows, time]);
        let mask = token_mask.into_data().to_vec::<f32>().expect("token mask");
        assert!(mask.chunks(time).all(|row| row.iter().sum::<f32>() > 0.0));
        let valid = valid_action_mask
            .into_data()
            .to_vec::<f32>()
            .expect("valid mask");
        let weights = row_weights.into_data().to_vec::<f32>().expect("weights");
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
        for group in valid.chunks(candidates_per_group) {
            assert!(group.iter().any(|value| *value > 0.0));
        }
    }

    #[test]
    fn semantic_candidate_completion_fixed_prediction_matches_global_backpropagation() {
        let device = burn::tensor::Device::<AutodiffTestBackend>::default();
        let model = crate::train::test_support::deterministic_matrix_parameters(
            verifier_model_with_layers::<AutodiffTestBackend>(&device, 4),
        );
        let batch = formal_policy_batch(
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
        );
        let mut policy = policy(RuliadProofPolicyNormalization::CandidateConditional);
        policy.candidate_symmetry =
            crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation;
        policy.max_presentation_rows_per_update = 8;
        let prepared =
            prepare_ruliad_verifier_terminal::<TestBackend>(&batch, policy, 512, 272, &device)
                .expect("candidate-completion verifier terminal");
        assert_eq!(prepared.semantic_states, 1);
        assert_eq!(prepared.decision_rows, 1);
        assert_eq!(prepared.inputs.shape().dims::<2>()[0], 4);

        let report = super::super::diagnostics::local_predictive_coding_verifier_gradient_fidelity(
            &model,
            prepared,
            &crate::config::LocalPredictiveCodingConfig {
                solver: crate::config::LocalPredictiveCodingSolver::FixedPrediction,
                factor_reduction: crate::config::PredictiveCodingFactorReduction::Sum,
                ..crate::config::LocalPredictiveCodingConfig::default()
            },
        )
        .expect("candidate-completion verifier gradient fidelity");

        assert!(report.loss_absolute_error < 2.0e-6, "{report:?}");
        assert_eq!(report.pc_step.global_backward_calls, 0);
        assert_eq!(report.pc_gradient_tensors, 9);
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
    }

    #[test]
    fn semantic_candidate_completion_inferred_solvers_cover_all_parameter_families() {
        let device = burn::tensor::Device::<AutodiffTestBackend>::default();
        let model = crate::train::test_support::deterministic_matrix_parameters(
            verifier_model_with_layers::<AutodiffTestBackend>(&device, 4),
        );
        let batch = formal_policy_batch(
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
        );

        for solver in [
            crate::config::LocalPredictiveCodingSolver::SynchronousEquilibrium,
            crate::config::LocalPredictiveCodingSolver::ReverseGaussSeidel,
            crate::config::LocalPredictiveCodingSolver::LayerLocalPrediction,
            crate::config::LocalPredictiveCodingSolver::ErrorEquilibrium,
            crate::config::LocalPredictiveCodingSolver::AugmentedLagrangian,
        ] {
            let mut policy = policy(RuliadProofPolicyNormalization::CandidateConditional);
            policy.candidate_symmetry =
                crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation;
            policy.max_presentation_rows_per_update = 8;
            let prepared =
                prepare_ruliad_verifier_terminal::<TestBackend>(&batch, policy, 512, 272, &device)
                    .expect("candidate-completion verifier terminal");
            let mut config = crate::config::LocalPredictiveCodingConfig {
                solver,
                factor_reduction: if matches!(
                    solver,
                    crate::config::LocalPredictiveCodingSolver::LayerLocalPrediction
                ) {
                    crate::config::PredictiveCodingFactorReduction::Mean
                } else {
                    crate::config::PredictiveCodingFactorReduction::Sum
                },
                ..crate::config::LocalPredictiveCodingConfig::default()
            };
            config.inference.steps = if matches!(
                solver,
                crate::config::LocalPredictiveCodingSolver::SynchronousEquilibrium
            ) {
                5
            } else {
                1
            };
            config.inference.step_size = 0.1;
            config.prediction_precision = if matches!(
                solver,
                crate::config::LocalPredictiveCodingSolver::ErrorEquilibrium
            ) {
                10.0
            } else {
                1.0
            };
            // Finite-inference ALM propagates terminal credit one local factor at a time.
            // Four shared-depth uses therefore need a fifth primal step to reach embedding.
            config.augmented_lagrangian.steps = 5;

            let report =
                super::super::diagnostics::local_predictive_coding_verifier_gradient_fidelity(
                    &model, prepared, &config,
                )
                .unwrap_or_else(|error| {
                    panic!("{solver:?} candidate-completion fidelity: {error}")
                });
            assert!(
                report.loss_absolute_error < 2.0e-6,
                "{solver:?}: {report:?}"
            );
            assert_eq!(report.pc_step.global_backward_calls, 0);
            assert_eq!(report.pc_gradient_tensors, 9, "{solver:?}: {report:?}");
            assert_eq!(report.parameter_families.len(), 9, "{solver:?}: {report:?}");
            assert!(
                report
                    .parameter_families
                    .iter()
                    .filter(|family| family.reference_norm > 1.0e-8)
                    .all(|family| family.pc_norm > 1.0e-8),
                "{solver:?}: {:?}",
                report.parameter_families
            );
        }
    }

    #[test]
    fn target_group_conditional_completion_matches_global_backpropagation() {
        let device = burn::tensor::Device::<AutodiffTestBackend>::default();
        let model = crate::train::test_support::deterministic_matrix_parameters(
            verifier_model_with_layers::<AutodiffTestBackend>(&device, 4),
        );
        let batch = formal_policy_batch(
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
        );
        let mut policy = policy(RuliadProofPolicyNormalization::CandidateConditional);
        policy.candidate_symmetry =
            crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation;
        policy.counterfactual_targets_per_state = 1;
        policy.counterfactual_objective =
            crate::config::RuliadProofPolicyCounterfactualObjective::TargetGroupConditional;
        policy.max_rows_per_update = 2;
        policy.max_presentation_rows_per_update = 8;
        let prepared =
            prepare_ruliad_verifier_terminal::<TestBackend>(&batch, policy, 512, 272, &device)
                .expect("counterfactual candidate-completion terminal");
        assert_eq!(prepared.semantic_states, 2);
        assert_eq!(prepared.stats.base_semantic_states, 1);
        assert_eq!(prepared.stats.counterfactual_semantic_states, 1);
        assert_eq!(prepared.stats.counterfactual_target_shortfall, 0);
        assert_eq!(prepared.decision_rows, 2);
        assert_eq!(prepared.inputs.shape().dims::<2>()[0], 8);
        assert!(matches!(
            &prepared.criterion,
            LocalPcTerminalCriterion::SequenceCompletionSet { .. }
        ));
        let LocalPcTerminalCriterion::SequenceCompletionSet {
            support_action_mask,
            valid_action_mask,
            ..
        } = &prepared.criterion
        else {
            unreachable!()
        };
        let support = support_action_mask
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("target-group support");
        let valid = valid_action_mask
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("target-group labels");
        assert_eq!(&support[..4], &support[4..]);
        assert_eq!(support.iter().filter(|value| **value > 0.0).count(), 4);
        assert_ne!(&valid[..4], &valid[4..]);

        let report = super::super::diagnostics::local_predictive_coding_verifier_gradient_fidelity(
            &model,
            prepared,
            &crate::config::LocalPredictiveCodingConfig {
                solver: crate::config::LocalPredictiveCodingSolver::FixedPrediction,
                factor_reduction: crate::config::PredictiveCodingFactorReduction::Sum,
                ..crate::config::LocalPredictiveCodingConfig::default()
            },
        )
        .expect("counterfactual candidate-completion gradient fidelity");

        assert!(report.loss_absolute_error < 2.0e-6, "{report:?}");
        assert_eq!(report.pc_step.global_backward_calls, 0);
        assert_eq!(report.pc_gradient_tensors, 9);
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
    fn counterfactual_shortfall_rejects_unbalanced_training_groups() {
        let device = Default::default();
        let batch = formal_policy_batch(
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
        );
        let mut independent = policy(RuliadProofPolicyNormalization::PrefixConditional);
        independent.candidate_symmetry =
            crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation;
        independent.counterfactual_targets_per_state = 8;
        independent.counterfactual_objective =
            crate::config::RuliadProofPolicyCounterfactualObjective::Independent;
        independent.max_rows_per_update = 9;
        independent.max_presentation_rows_per_update = 64;

        assert!(
            prepare_ruliad_verifier_terminal::<TestBackend>(
                &batch,
                independent,
                512,
                272,
                &device,
            )
            .is_none(),
            "paired full-menu updates must not retain a target-frequency shortcut"
        );

        let mut conditional = independent;
        conditional.counterfactual_objective =
            crate::config::RuliadProofPolicyCounterfactualObjective::TargetGroupConditional;
        assert!(
            prepare_ruliad_verifier_terminal::<TestBackend>(
                &batch,
                conditional,
                512,
                272,
                &device,
            )
            .is_none(),
            "target-group conditioning must reject an incomplete matched group"
        );
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
        assert_eq!(report.pc_gradient_tensors, 14);
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
    fn semantic_energy_verifier_panel_supports_tensorized_layer_local_credit() {
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
                .expect("semantic-energy layer-local verifier terminal");

        let report = super::super::diagnostics::local_predictive_coding_verifier_gradient_fidelity(
            &model,
            prepared,
            &crate::config::LocalPredictiveCodingConfig {
                solver: crate::config::LocalPredictiveCodingSolver::LayerLocalPrediction,
                factor_reduction: crate::config::PredictiveCodingFactorReduction::Mean,
                sync_diagnostics: false,
                ..crate::config::LocalPredictiveCodingConfig::default()
            },
        )
        .expect("semantic-energy layer-local verifier derivatives");

        assert!(report.loss_absolute_error < 2.0e-6, "{report:?}");
        assert_eq!(report.pc_step.global_backward_calls, 0);
        assert_eq!(report.pc_step.local_vjp_calls, 3);
        assert_eq!(report.pc_gradient_tensors, 14);
        for required in [
            "embedding",
            "shared_encoder",
            "shared_value_encoder",
            "shared_decoder",
            "sequence_query_weight",
            "sequence_candidate_weight",
            "sequence_score_weight",
        ] {
            let family = report
                .parameter_families
                .iter()
                .find(|family| family.parameter_family == required)
                .unwrap_or_else(|| panic!("missing {required}: {report:?}"));
            assert!(family.pc_norm > 1.0e-8, "{family:?}");
        }
    }

    #[test]
    fn semantic_energy_score_head_only_pc_matches_isolated_global_backpropagation() {
        let device = burn::tensor::Device::<AutodiffTestBackend>::default();
        let model = crate::train::test_support::deterministic_matrix_parameters(
            verifier_model_with_score_head::<AutodiffTestBackend>(&device, 4),
        );
        let batch = formal_policy_batch(
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
        );
        let mut policy = policy(RuliadProofPolicyNormalization::CandidateConditional);
        policy.scoring = crate::config::RuliadProofPolicyScoring::SemanticEnergy;
        policy.gradient_scope = crate::config::RuliadProofPolicyGradientScope::ScoreHeadOnly;
        policy.candidate_symmetry =
            crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation;
        policy.counterfactual_targets_per_state = 1;
        policy.counterfactual_objective =
            crate::config::RuliadProofPolicyCounterfactualObjective::TargetGroupConditional;
        policy.max_rows_per_update = 2;
        policy.max_presentation_rows_per_update = 8;
        let prepared =
            prepare_ruliad_verifier_terminal::<TestBackend>(&batch, policy, 512, 272, &device)
                .expect("score-head-only semantic verifier terminal");
        let LocalPcTerminalCriterion::SequenceEnergySetAtPositions {
            propagate_hidden_gradient,
            ..
        } = &prepared.criterion
        else {
            panic!("semantic verifier panel must use sequence-energy rows");
        };
        assert!(!propagate_hidden_gradient);

        let report = super::super::diagnostics::local_predictive_coding_verifier_gradient_fidelity(
            &model,
            prepared,
            &crate::config::LocalPredictiveCodingConfig {
                solver: crate::config::LocalPredictiveCodingSolver::FixedPrediction,
                factor_reduction: crate::config::PredictiveCodingFactorReduction::Sum,
                ..crate::config::LocalPredictiveCodingConfig::default()
            },
        )
        .expect("score-head-only semantic gradient fidelity");

        assert!(report.loss_absolute_error < 2.0e-6, "{report:?}");
        assert_eq!(report.pc_step.global_backward_calls, 0);
        for family in &report.parameter_families {
            if family.parameter_family.starts_with("sequence_") {
                if family.reference_norm > 1.0e-6 {
                    assert!(
                        family.cosine.is_some_and(|cosine| cosine > 0.999_8),
                        "{family:?}"
                    );
                    assert!(
                        family.relative_l2_error.is_some_and(|error| error < 2.0e-3),
                        "{family:?}"
                    );
                }
            } else {
                assert!(family.reference_norm < 1.0e-7, "{family:?}");
                assert!(family.pc_norm < 1.0e-7, "{family:?}");
            }
        }
    }

    #[test]
    fn residual_energy_verifier_panel_fixed_prediction_matches_global_backpropagation() {
        let device = burn::tensor::Device::<AutodiffTestBackend>::default();
        let model = crate::train::test_support::deterministic_matrix_parameters(
            verifier_model_with_score_head::<AutodiffTestBackend>(&device, 4),
        );
        let batch = formal_policy_batch(
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
        );
        let mut policy = policy(RuliadProofPolicyNormalization::CandidateConditional);
        policy.scoring = crate::config::RuliadProofPolicyScoring::ResidualEnergy;
        policy.candidate_symmetry =
            crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation;
        policy.counterfactual_targets_per_state = 1;
        policy.counterfactual_objective =
            crate::config::RuliadProofPolicyCounterfactualObjective::Independent;
        policy.max_rows_per_update = 2;
        policy.max_presentation_rows_per_update = 8;
        let prepared =
            prepare_ruliad_verifier_terminal::<TestBackend>(&batch, policy, 512, 272, &device)
                .expect("residual-energy counterfactual verifier terminal");
        let LocalPcTerminalCriterion::SequenceResidualEnergySetAtPositions {
            token_mask,
            propagate_hidden_gradient,
            ..
        } = &prepared.criterion
        else {
            panic!("residual verifier panel must use residual-energy rows");
        };
        assert!(*propagate_hidden_gradient);
        assert!(burn_pc::diagnostic_scalar_f32(token_mask.clone().sum()) > 0.0);
        let layer_local_prepared = prepared.clone();

        let report = super::super::diagnostics::local_predictive_coding_verifier_gradient_fidelity(
            &model,
            prepared,
            &crate::config::LocalPredictiveCodingConfig {
                solver: crate::config::LocalPredictiveCodingSolver::FixedPrediction,
                factor_reduction: crate::config::PredictiveCodingFactorReduction::Sum,
                ..crate::config::LocalPredictiveCodingConfig::default()
            },
        )
        .expect("residual-energy verifier gradient fidelity");

        assert!(report.loss_absolute_error < 2.0e-6, "{report:?}");
        assert_eq!(report.pc_step.global_backward_calls, 0);
        assert_eq!(report.pc_gradient_tensors, 15);
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

        let layer_local =
            super::super::diagnostics::local_predictive_coding_verifier_gradient_fidelity(
                &model,
                layer_local_prepared,
                &crate::config::LocalPredictiveCodingConfig {
                    solver: crate::config::LocalPredictiveCodingSolver::LayerLocalPrediction,
                    factor_reduction: crate::config::PredictiveCodingFactorReduction::Mean,
                    sync_diagnostics: false,
                    ..crate::config::LocalPredictiveCodingConfig::default()
                },
            )
            .expect("residual-energy layer-local verifier derivatives");
        assert!(layer_local.loss_absolute_error < 2.0e-6, "{layer_local:?}");
        assert_eq!(layer_local.pc_step.global_backward_calls, 0);
        assert_eq!(layer_local.pc_step.local_vjp_calls, 3);
        assert_eq!(layer_local.pc_gradient_tensors, 15);
        for required in [
            "embedding",
            "shared_encoder",
            "shared_value_encoder",
            "shared_decoder",
            "language_head",
            "sequence_query_weight",
            "sequence_candidate_weight",
            "sequence_score_weight",
        ] {
            let family = layer_local
                .parameter_families
                .iter()
                .find(|family| family.parameter_family == required)
                .unwrap_or_else(|| panic!("missing {required}: {layer_local:?}"));
            assert!(family.pc_norm > 1.0e-8, "{family:?}");
        }
    }

    #[test]
    fn verified_progress_residual_terminal_matches_global_backpropagation() {
        let device = burn::tensor::Device::<AutodiffTestBackend>::default();
        let model = crate::train::test_support::deterministic_matrix_parameters(
            verifier_model_with_score_head::<AutodiffTestBackend>(&device, 4),
        );
        let batch = formal_policy_batch(
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
        );
        let mut policy = policy(RuliadProofPolicyNormalization::CandidateConditional);
        policy.scoring = crate::config::RuliadProofPolicyScoring::ResidualEnergy;
        policy.target = crate::config::RuliadProofPolicyTarget::VerifiedProgressDistribution;
        policy.candidate_symmetry =
            crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation;
        policy.counterfactual_targets_per_state = 1;
        policy.max_rows_per_update = 2;
        policy.max_presentation_rows_per_update = 8;
        let prepared =
            prepare_ruliad_verifier_terminal::<TestBackend>(&batch, policy, 512, 272, &device)
                .expect("verified-progress residual terminal");
        let LocalPcTerminalCriterion::SequenceResidualEnergySetAtPositions {
            target_action_weights,
            ..
        } = &prepared.criterion
        else {
            panic!("verified progress requires residual-energy rows");
        };
        assert!(target_action_weights.is_some());

        let report = super::super::diagnostics::local_predictive_coding_verifier_gradient_fidelity(
            &model,
            prepared,
            &crate::config::LocalPredictiveCodingConfig {
                solver: crate::config::LocalPredictiveCodingSolver::FixedPrediction,
                factor_reduction: crate::config::PredictiveCodingFactorReduction::Sum,
                ..crate::config::LocalPredictiveCodingConfig::default()
            },
        )
        .expect("verified-progress gradient fidelity");

        assert!(report.loss_absolute_error < 2.0e-6, "{report:?}");
        assert_eq!(report.pc_step.global_backward_calls, 0);
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
    }

    #[test]
    fn residual_energy_score_head_only_pc_matches_isolated_global_backpropagation() {
        let device = burn::tensor::Device::<AutodiffTestBackend>::default();
        let model = crate::train::test_support::deterministic_matrix_parameters(
            verifier_model_with_score_head::<AutodiffTestBackend>(&device, 4),
        );
        let batch = formal_policy_batch(
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
        );
        let mut policy = policy(RuliadProofPolicyNormalization::CandidateConditional);
        policy.scoring = crate::config::RuliadProofPolicyScoring::ResidualEnergy;
        policy.gradient_scope = crate::config::RuliadProofPolicyGradientScope::ScoreHeadOnly;
        policy.candidate_symmetry =
            crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation;
        policy.counterfactual_targets_per_state = 1;
        policy.counterfactual_objective =
            crate::config::RuliadProofPolicyCounterfactualObjective::Independent;
        policy.max_rows_per_update = 2;
        policy.max_presentation_rows_per_update = 8;
        let prepared =
            prepare_ruliad_verifier_terminal::<TestBackend>(&batch, policy, 512, 272, &device)
                .expect("score-head-only residual verifier terminal");
        let LocalPcTerminalCriterion::SequenceResidualEnergySetAtPositions {
            propagate_hidden_gradient,
            ..
        } = &prepared.criterion
        else {
            panic!("residual verifier panel must use residual-energy rows");
        };
        assert!(!propagate_hidden_gradient);

        let report = super::super::diagnostics::local_predictive_coding_verifier_gradient_fidelity(
            &model,
            prepared,
            &crate::config::LocalPredictiveCodingConfig {
                solver: crate::config::LocalPredictiveCodingSolver::FixedPrediction,
                factor_reduction: crate::config::PredictiveCodingFactorReduction::Sum,
                ..crate::config::LocalPredictiveCodingConfig::default()
            },
        )
        .expect("score-head-only residual gradient fidelity");

        assert!(report.loss_absolute_error < 2.0e-6, "{report:?}");
        assert_eq!(report.pc_step.global_backward_calls, 0);
        for family in &report.parameter_families {
            if family.parameter_family.starts_with("sequence_") {
                if family.reference_norm > 1.0e-6 {
                    assert!(
                        family.cosine.is_some_and(|cosine| cosine > 0.999_8),
                        "{family:?}"
                    );
                    assert!(
                        family.relative_l2_error.is_some_and(|error| error < 2.0e-3),
                        "{family:?}"
                    );
                }
            } else {
                assert!(family.reference_norm < 1.0e-7, "{family:?}");
                assert!(family.pc_norm < 1.0e-7, "{family:?}");
            }
        }
    }

    #[test]
    fn residual_energy_policy_path_pc_matches_decoder_isolated_global_backpropagation() {
        let device = burn::tensor::Device::<AutodiffTestBackend>::default();
        let model = crate::train::test_support::deterministic_matrix_parameters(
            verifier_model_with_score_head::<AutodiffTestBackend>(&device, 4),
        );
        let batch = formal_policy_batch(
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
        );
        let mut policy = policy(RuliadProofPolicyNormalization::CandidateConditional);
        policy.scoring = crate::config::RuliadProofPolicyScoring::ResidualEnergy;
        policy.gradient_scope = crate::config::RuliadProofPolicyGradientScope::PolicyPath;
        policy.candidate_symmetry =
            crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation;
        policy.counterfactual_targets_per_state = 1;
        policy.counterfactual_objective =
            crate::config::RuliadProofPolicyCounterfactualObjective::TargetGroupConditional;
        policy.max_rows_per_update = 2;
        policy.max_presentation_rows_per_update = 8;
        let prepared =
            prepare_ruliad_verifier_terminal::<TestBackend>(&batch, policy, 512, 272, &device)
                .expect("policy-path residual verifier terminal");
        let LocalPcTerminalCriterion::SequenceResidualEnergySetAtPositions {
            propagate_hidden_gradient,
            propagate_language_prior_gradient,
            ..
        } = &prepared.criterion
        else {
            panic!("policy-path panel must use residual-energy rows");
        };
        assert!(*propagate_hidden_gradient);
        assert!(!propagate_language_prior_gradient);

        let report = super::super::diagnostics::local_predictive_coding_verifier_gradient_fidelity(
            &model,
            prepared,
            &crate::config::LocalPredictiveCodingConfig {
                solver: crate::config::LocalPredictiveCodingSolver::FixedPrediction,
                factor_reduction: crate::config::PredictiveCodingFactorReduction::Sum,
                ..crate::config::LocalPredictiveCodingConfig::default()
            },
        )
        .expect("policy-path residual gradient fidelity");

        assert!(report.loss_absolute_error < 2.0e-6, "{report:?}");
        assert_eq!(report.pc_step.global_backward_calls, 0);
        assert!(
            report.global.cosine.is_some_and(|cosine| cosine > 0.999_98),
            "{report:?}"
        );
        assert!(
            report
                .global
                .relative_l2_error
                .is_some_and(|error| error < 5.0e-4),
            "{report:?}"
        );
        let language_head = report
            .parameter_families
            .iter()
            .find(|family| family.parameter_family == "language_head")
            .expect("language-head diagnostics");
        assert!(language_head.reference_norm < 1.0e-7, "{language_head:?}");
        assert!(language_head.pc_norm < 1.0e-7, "{language_head:?}");
        assert!(
            report.parameter_families.iter().any(|family| {
                !family.parameter_family.starts_with("sequence_")
                    && family.parameter_family != "language_head"
                    && family.reference_norm > 1.0e-6
                    && family.pc_norm > 1.0e-6
            }),
            "policy-path must propagate residual credit into Dragon: {report:?}"
        );
    }

    #[test]
    fn semantic_energy_inferred_solvers_register_score_head_derivatives() {
        let device = burn::tensor::Device::<AutodiffTestBackend>::default();
        let model = crate::train::test_support::deterministic_matrix_parameters(
            verifier_model_with_score_head::<AutodiffTestBackend>(&device, 4),
        );
        let batch = formal_policy_batch(
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
        );

        for solver in [
            crate::config::LocalPredictiveCodingSolver::SynchronousEquilibrium,
            crate::config::LocalPredictiveCodingSolver::ReverseGaussSeidel,
            crate::config::LocalPredictiveCodingSolver::ErrorEquilibrium,
            crate::config::LocalPredictiveCodingSolver::AugmentedLagrangian,
        ] {
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
            let mut config = crate::config::LocalPredictiveCodingConfig {
                solver,
                factor_reduction: crate::config::PredictiveCodingFactorReduction::Sum,
                ..crate::config::LocalPredictiveCodingConfig::default()
            };
            config.inference.steps = if matches!(
                solver,
                crate::config::LocalPredictiveCodingSolver::SynchronousEquilibrium
            ) {
                5
            } else {
                1
            };
            config.inference.step_size = 0.1;
            config.prediction_precision = if matches!(
                solver,
                crate::config::LocalPredictiveCodingSolver::ErrorEquilibrium
            ) {
                10.0
            } else {
                1.0
            };
            config.augmented_lagrangian.steps = 2;

            let report =
                super::super::diagnostics::local_predictive_coding_verifier_gradient_fidelity(
                    &model, prepared, &config,
                )
                .unwrap_or_else(|error| panic!("{solver:?} semantic-energy fidelity: {error}"));
            assert!(
                report.loss_absolute_error < 2.0e-6,
                "{solver:?}: {report:?}"
            );
            assert_eq!(report.pc_step.global_backward_calls, 0);
            assert_eq!(report.pc_gradient_tensors, 14, "{solver:?}: {report:?}");
            let score_families = report
                .parameter_families
                .iter()
                .filter(|family| family.parameter_family.starts_with("sequence_"))
                .collect::<Vec<_>>();
            assert_eq!(score_families.len(), 6, "{solver:?}: {report:?}");
            assert!(
                score_families
                    .iter()
                    .filter(|family| family.reference_norm > 1.0e-8)
                    .all(|family| family.pc_norm > 1.0e-8),
                "{solver:?}: {score_families:?}"
            );
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
        assert_eq!(first.stats, second.stats);
        assert_eq!(
            first.stats.difficulty_sample_groups.values().sum::<usize>(),
            first.stats.sample_groups
        );
        assert_eq!(
            first.stats.difficulty_expert_rows.values().sum::<usize>(),
            first.stats.semantic_states
        );
        assert!(
            first
                .stats
                .difficulty_visited_states
                .values()
                .sum::<usize>()
                >= first.stats.base_semantic_states
        );
        assert!(
            first
                .stats
                .expert_selected_index_histogram
                .values()
                .sum::<usize>()
                > 0
        );
        assert!(
            first
                .stats
                .model_selected_index_histogram
                .values()
                .sum::<usize>()
                > 0
        );

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
    fn paired_dagger_reuses_a_single_sample_for_model_visited_supervision() {
        let device = burn::tensor::Device::<AutodiffTestBackend>::default();
        let sampling_model = verifier_model::<AutodiffTestBackend>(&device).valid();
        let batch = formal_policy_batch(
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
        );
        let mut config = policy(RuliadProofPolicyNormalization::PrefixConditional);
        config.mode = crate::config::RuliadProofPolicyTrainingMode::StaticThenPairedDagger;
        config.dagger_start_after_steps = 0;
        config.rollout_steps = 4;
        config.counterfactual_targets_per_state = 1;
        config.max_rows_per_update = 32;
        config.max_presentation_rows_per_update = 128;

        let prepared = prepare_ruliad_verifier_terminal_at_step::<TestBackend>(
            Some(&sampling_model),
            &batch,
            config,
            512,
            272,
            0,
            &device,
        )
        .expect("batch-one paired DAgger verifier panel");

        assert_eq!(prepared.stats.available_sample_groups, 1);
        assert_eq!(prepared.stats.sample_groups, 2);
        assert!(prepared.stats.static_expert_states > 0);
        assert!(prepared.stats.dagger_expert_states > 0);
        assert!(prepared.stats.model_scoring_batches > 0);
        assert!(prepared.stats.model_visited_states > 0);
        assert!(prepared.stats.model_valid_actions > 0);
        assert!(prepared.stats.rollout_depth_reached > 1);
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
