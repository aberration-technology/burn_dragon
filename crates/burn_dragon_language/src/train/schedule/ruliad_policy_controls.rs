//! Checkpoint-only controls for verifier-enumerated action menus.
//!
//! These measure shortcuts within an oracle menu, not unconstrained proof generation.

use burn_dragon_universality::ruliad::{
    RuliadProofActionSet,
    kernel::{RuliadGoalTransitionKernel, RuliadKernelLimits, replay_certificate},
    policy::ruliad_term_distance,
};

use super::*;
use crate::train::ruliad_policy::{EncodedRuliadProofActionRequest, RuliadProofActionDecision};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RuliadPolicyControlMode {
    Disabled,
    Checkpoint,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct RuliadPolicyControlItem {
    pub oracle_hash: String,
    pub difficulty_level: usize,
    pub source_label: String,
    pub candidates: usize,
    pub equivalent_candidates: usize,
    pub uniform_expected_accuracy: f64,
    pub first_canonical_correct: bool,
    pub first_presented_correct: bool,
    /// Expected accuracy under uniform tie breaking; no label-based tie breaking.
    pub shortest_expected_accuracy: f64,
    /// One-step symbolic execution and structural distance, without a certificate.
    pub greedy_distance_expected_accuracy: f64,
    pub greedy_distance_ties: usize,
    pub model_correct: bool,
    pub no_context_correct: bool,
    pub model_equivalent_probability: f64,
    pub no_context_equivalent_probability: f64,
}

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct RuliadPolicyControlSummary {
    pub items: usize,
    pub uniform_expected_accuracy: f64,
    pub first_canonical_accuracy: f64,
    pub first_presented_accuracy: f64,
    pub shortest_expected_accuracy: f64,
    pub greedy_distance_expected_accuracy: f64,
    pub model_accuracy: f64,
    pub no_context_accuracy: f64,
    pub model_minus_chance: f64,
    pub model_minus_greedy_distance: f64,
    pub model_minus_no_context: f64,
    pub context_helped_items: usize,
    pub context_harmed_items: usize,
    pub context_equivalent_probability_gain: f64,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct RuliadPolicyControlEvaluation {
    pub version: u32,
    /// Explicitly distinguishes reference-certificate menus from deployable proposals.
    pub candidate_source: String,
    pub no_context_prompt: String,
    /// Replays cached outcomes/labels through the existing kernel, not a second verifier.
    pub kernel_audited_candidates: usize,
    pub summary: RuliadPolicyControlSummary,
    pub by_difficulty: BTreeMap<usize, RuliadPolicyControlSummary>,
    pub by_source: BTreeMap<String, RuliadPolicyControlSummary>,
    pub items: Vec<RuliadPolicyControlItem>,
}

const NO_CONTEXT_PROMPT: &str = "!:";

pub(super) fn no_context_action_requests(
    dataset: &Dataset,
    requests: &[EncodedRuliadProofActionRequest],
) -> Result<Vec<EncodedRuliadProofActionRequest>> {
    let prompt = dataset
        .encode_ruliad_payload_tokens(NO_CONTEXT_PROMPT)
        .filter(|tokens| !tokens.is_empty())
        .ok_or_else(|| anyhow!("cannot encode no-context action control"))?
        .into_iter()
        .map(i64::from)
        .collect::<Vec<_>>();
    Ok(requests
        .iter()
        .map(|request| replace_action_context(request, &prompt))
        .collect())
}

fn replace_action_context(
    request: &EncodedRuliadProofActionRequest,
    prompt: &[i64],
) -> EncodedRuliadProofActionRequest {
    let mut request = request.clone();
    for presentation in &mut request.presentations {
        presentation.prompt_tokens = prompt.to_vec();
        presentation.original_prompt_token_count = prompt.len();
    }
    request
}

fn expected_accuracy(indices: &[usize], actions: &RuliadProofActionSet) -> f64 {
    indices
        .iter()
        .filter(|index| actions.is_equivalent_index(**index))
        .count() as f64
        / indices.len().max(1) as f64
}

fn minimum_indices(values: &[usize]) -> Vec<usize> {
    let minimum = values.iter().min();
    values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (Some(value) == minimum).then_some(index))
        .collect()
}

fn audit_transitions(
    item: &burn_dragon_universality::RuliadEvalItem,
    context: &RuliadPolicyActionPromptContext,
) -> Result<Vec<usize>> {
    let Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
        problem,
        certificate,
        proof_step_index,
        ..
    }) = item.spec.as_ref()
    else {
        return Err(anyhow!("policy control requires a formal proof item"));
    };
    let actions = &context.actions;
    anyhow::ensure!(
        problem == &context.problem,
        "policy control problem mismatch"
    );
    anyhow::ensure!(
        !actions.candidates.is_empty()
            && actions.selected_index < actions.candidates.len()
            && !actions.equivalent_indices.is_empty(),
        "invalid control action set"
    );
    let limits = RuliadKernelLimits::default();
    anyhow::ensure!(
        replay_certificate(problem, certificate, limits).accepted,
        "policy control reference certificate does not verify"
    );
    let step_index = proof_step_index.unwrap_or_default();
    let (goal, expert_step) = certificate
        .step_at(step_index)
        .ok_or_else(|| anyhow!("missing control reference step"))?;
    anyhow::ensure!(goal == actions.goal, "control goal mismatch");
    let kernel = RuliadGoalTransitionKernel::new(problem, goal, limits)
        .map_err(|failure| anyhow!("control kernel: {}", failure.message))?;
    let prefix = certificate
        .prefix_before(step_index)
        .ok_or_else(|| anyhow!("missing control reference prefix"))?;
    let steps = prefix
        .goals
        .iter()
        .find(|candidate| candidate.goal == goal)
        .map(|candidate| candidate.steps.as_slice())
        .unwrap_or_default();
    let current = kernel
        .replay_prefix(steps)
        .map_err(|failure| anyhow!("control prefix: {}", failure.message))?;
    anyhow::ensure!(
        current == actions.current && kernel.target() == &actions.target,
        "cached control state differs from replay"
    );
    let expert_outcome = kernel
        .apply(&current, expert_step)
        .map_err(|failure| anyhow!("control expert: {}", failure.message))?;
    let mut equivalents = Vec::new();
    let mut distances = Vec::with_capacity(actions.candidates.len());
    for (index, candidate) in actions.candidates.iter().enumerate() {
        let outcome = kernel
            .apply(&current, &candidate.step)
            .map_err(|failure| anyhow!("control candidate: {}", failure.message))?;
        let distance = ruliad_term_distance(&outcome, kernel.target());
        anyhow::ensure!(
            candidate.outcome.as_ref() == Some(&outcome)
                && candidate.distance_to_goal == Some(distance),
            "cached candidate outcome/distance differs from replay"
        );
        if outcome == expert_outcome {
            equivalents.push(index);
        }
        // The heuristic consumes only executed transitions and the public goal.
        distances.push(distance);
    }
    let mut cached = actions.equivalent_indices.clone();
    cached.sort_unstable();
    anyhow::ensure!(
        equivalents == cached && equivalents.contains(&actions.selected_index),
        "cached candidate equivalence labels differ from replay"
    );
    Ok(distances)
}

fn equivalent_probability(
    decision: &RuliadProofActionDecision,
    actions: &RuliadProofActionSet,
) -> Result<f64> {
    let scores = &decision.orbit.averaged_log_probs;
    anyhow::ensure!(
        scores.len() == actions.candidates.len()
            && decision.selected_semantic_index < scores.len()
            && scores.iter().all(|score| score.is_finite()),
        "invalid control model decision"
    );
    let mass = scores
        .iter()
        .map(|score| f64::from(*score).exp())
        .sum::<f64>();
    anyhow::ensure!(
        (mass - 1.0).abs() < 1.0e-4,
        "control probabilities are not normalized"
    );
    Ok(actions
        .equivalent_indices
        .iter()
        .map(|index| f64::from(scores[*index]).exp())
        .sum())
}

fn summarize(items: &[&RuliadPolicyControlItem]) -> RuliadPolicyControlSummary {
    let mut result = RuliadPolicyControlSummary {
        items: items.len(),
        ..Default::default()
    };
    let denominator = items.len().max(1) as f64;
    for item in items {
        result.uniform_expected_accuracy += item.uniform_expected_accuracy / denominator;
        result.first_canonical_accuracy += f64::from(item.first_canonical_correct) / denominator;
        result.first_presented_accuracy += f64::from(item.first_presented_correct) / denominator;
        result.shortest_expected_accuracy += item.shortest_expected_accuracy / denominator;
        result.greedy_distance_expected_accuracy +=
            item.greedy_distance_expected_accuracy / denominator;
        result.model_accuracy += f64::from(item.model_correct) / denominator;
        result.no_context_accuracy += f64::from(item.no_context_correct) / denominator;
        result.context_helped_items += usize::from(item.model_correct && !item.no_context_correct);
        result.context_harmed_items += usize::from(!item.model_correct && item.no_context_correct);
        result.context_equivalent_probability_gain += (item.model_equivalent_probability
            - item.no_context_equivalent_probability)
            / denominator;
    }
    result.model_minus_chance = result.model_accuracy - result.uniform_expected_accuracy;
    result.model_minus_greedy_distance =
        result.model_accuracy - result.greedy_distance_expected_accuracy;
    result.model_minus_no_context = result.model_accuracy - result.no_context_accuracy;
    result
}

pub(super) fn evaluate_ruliad_policy_controls(
    items: &[burn_dragon_universality::RuliadEvalItem],
    jobs: &[RuliadCorrectnessConstrainedPolicyJob],
    decisions: &[RuliadProofActionDecision],
    no_context_decisions: &[RuliadProofActionDecision],
) -> Result<RuliadPolicyControlEvaluation> {
    anyhow::ensure!(
        items.len() == jobs.len()
            && jobs.len() == decisions.len()
            && decisions.len() == no_context_decisions.len(),
        "policy control item/decision alignment mismatch"
    );
    let mut rows = Vec::with_capacity(items.len());
    for (((item, job), decision), no_context) in items
        .iter()
        .zip(jobs)
        .zip(decisions)
        .zip(no_context_decisions)
    {
        let context = job
            .base_context
            .as_ref()
            .ok_or_else(|| anyhow!("control job lacks canonical state"))?;
        let actions = &context.actions;
        let distances = audit_transitions(item, context)?;
        let lengths =
            (0..actions.candidates.len())
                .map(|index| {
                    burn_dragon_universality::ruliad::proof_action_answer(
                actions, index,
                burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
            ).map(|answer| answer.len())
                })
                .collect::<Result<Vec<_>>>()?;
        let greedy = minimum_indices(&distances);
        let first_rotation = job
            .presentations
            .first()
            .ok_or_else(|| anyhow!("control job has no presentation"))?
            .rotation;
        rows.push(RuliadPolicyControlItem {
            oracle_hash: item.oracle_hash.clone(),
            difficulty_level: job.difficulty_level,
            source_label: job.source_label.clone(),
            candidates: actions.candidates.len(),
            equivalent_candidates: actions.equivalent_indices.len(),
            uniform_expected_accuracy: actions.equivalent_indices.len() as f64
                / actions.candidates.len() as f64,
            first_canonical_correct: actions.is_equivalent_index(0),
            first_presented_correct: actions
                .is_equivalent_index(actions.original_index_after_rotation(0, first_rotation)?),
            shortest_expected_accuracy: expected_accuracy(&minimum_indices(&lengths), actions),
            greedy_distance_expected_accuracy: expected_accuracy(&greedy, actions),
            greedy_distance_ties: greedy.len(),
            model_correct: actions.is_equivalent_index(decision.selected_semantic_index),
            no_context_correct: actions.is_equivalent_index(no_context.selected_semantic_index),
            model_equivalent_probability: equivalent_probability(decision, actions)?,
            no_context_equivalent_probability: equivalent_probability(no_context, actions)?,
        });
    }
    let mut difficulties = BTreeMap::<usize, Vec<_>>::new();
    let mut sources = BTreeMap::<String, Vec<_>>::new();
    for row in &rows {
        difficulties
            .entry(row.difficulty_level)
            .or_default()
            .push(row);
        sources
            .entry(row.source_label.clone())
            .or_default()
            .push(row);
    }
    Ok(RuliadPolicyControlEvaluation {
        version: 1,
        candidate_source: "reference_certificate_oracle_menu".into(),
        no_context_prompt: NO_CONTEXT_PROMPT.into(),
        kernel_audited_candidates: rows.iter().map(|row| row.candidates).sum(),
        summary: summarize(&rows.iter().collect::<Vec<_>>()),
        by_difficulty: difficulties
            .into_iter()
            .map(|(key, rows)| (key, summarize(&rows)))
            .collect(),
        by_source: sources
            .into_iter()
            .map(|(key, rows)| (key, summarize(&rows)))
            .collect(),
        items: rows,
    })
}

#[cfg(test)]
mod tests;
