//! Verifier-backed proof-state actions and closed-loop policy evaluation.

use std::collections::BTreeSet;

use anyhow::{Result, anyhow};

use crate::ruliad::ir::{
    RuliadProofCertificate, RuliadProofProblem, RuliadProofSource, RuliadProofStep,
    RuliadRewriteDirection, RuliadTerm,
};
use crate::ruliad::kernel::{RuliadGoalTransitionKernel, RuliadKernelLimits};

pub const DEFAULT_PROOF_ACTION_CANDIDATES: usize = 4;
const MAX_ENUMERATED_SOURCES: usize = 8;
const MAX_ENUMERATED_PATHS: usize = 12;
const MAX_TRIAL_ACTIONS: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuliadProofActionCandidate {
    pub step: RuliadProofStep,
    pub outcome: Option<RuliadTerm>,
    pub distance_to_goal: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuliadProofActionSet {
    pub goal: usize,
    pub current: RuliadTerm,
    pub target: RuliadTerm,
    pub candidates: Vec<RuliadProofActionCandidate>,
    pub selected_index: usize,
    pub equivalent_indices: Vec<usize>,
}

impl RuliadProofActionSet {
    pub fn selected(&self) -> Option<&RuliadProofActionCandidate> {
        self.candidates.get(self.selected_index)
    }

    pub fn is_equivalent_index(&self, index: usize) -> bool {
        self.equivalent_indices.contains(&index)
    }

    pub fn candidate_progress_ppm(&self, index: usize) -> usize {
        const PPM: usize = 1_000_000;
        let Some(candidate) = self.candidates.get(index) else {
            return 0;
        };
        if self.is_equivalent_index(index) || candidate.outcome.as_ref() == Some(&self.target) {
            return PPM;
        }
        let current_distance = ruliad_term_distance(&self.current, &self.target);
        let Some(next_distance) = candidate.distance_to_goal else {
            return 0;
        };
        current_distance
            .saturating_sub(next_distance)
            .saturating_mul(PPM)
            .checked_div(current_distance)
            .unwrap_or(0)
    }

    /// Rotate the candidate presentation to the left while preserving proof-action semantics.
    ///
    /// The returned indices are expressed in the rotated presentation. Use
    /// [`Self::original_index_after_rotation`] to map a model choice back to this action set.
    pub fn rotate_left(&self, rotation: usize) -> Result<Self> {
        let candidate_count = self.candidates.len();
        if candidate_count == 0 || self.selected_index >= candidate_count {
            return Err(anyhow!(
                "cannot rotate an empty or invalid proof action set"
            ));
        }
        if self
            .equivalent_indices
            .iter()
            .any(|index| *index >= candidate_count)
        {
            return Err(anyhow!(
                "cannot rotate a proof action set with an invalid equivalent index"
            ));
        }
        let rotation = rotation % candidate_count;
        let remap = |index: usize| (index + candidate_count - rotation) % candidate_count;
        let mut candidates = self.candidates.clone();
        candidates.rotate_left(rotation);
        let mut equivalent_indices = self
            .equivalent_indices
            .iter()
            .copied()
            .map(remap)
            .collect::<Vec<_>>();
        equivalent_indices.sort_unstable();
        Ok(Self {
            goal: self.goal,
            current: self.current.clone(),
            target: self.target.clone(),
            candidates,
            selected_index: remap(self.selected_index),
            equivalent_indices,
        })
    }

    /// Map an index from a left-rotated presentation back to this action set.
    pub fn original_index_after_rotation(
        &self,
        presented_index: usize,
        rotation: usize,
    ) -> Result<usize> {
        let candidate_count = self.candidates.len();
        if candidate_count == 0 || presented_index >= candidate_count {
            return Err(anyhow!(
                "presented proof action index {presented_index} exceeds {candidate_count} candidates"
            ));
        }
        Ok((presented_index + rotation % candidate_count) % candidate_count)
    }

    /// Rotate the candidate presentation while preserving the represented proof action.
    ///
    /// Candidate indices are serialization details rather than proof semantics. Training can use
    /// this group action to prevent an on-policy state distribution from turning a preferred
    /// menu position into a shortcut. The selected and verifier-equivalent indices are remapped
    /// with the candidates, so applying the returned action set reaches the same proof state.
    pub fn rotate_selected_to(&self, desired_index: usize) -> Result<Self> {
        let candidate_count = self.candidates.len();
        if candidate_count == 0 || self.selected_index >= candidate_count {
            return Err(anyhow!(
                "cannot rotate an empty or invalid proof action set"
            ));
        }
        if desired_index >= candidate_count {
            return Err(anyhow!(
                "desired proof action index {desired_index} exceeds {candidate_count} candidates"
            ));
        }
        let rotation = (self.selected_index + candidate_count - desired_index) % candidate_count;
        self.rotate_left(rotation)
    }
}

/// Retarget a proof-policy state to one verifier-valid alternative outcome.
///
/// The formal laws, current state, and candidate steps remain unchanged. Only the selected goal's
/// target and target-dependent action metadata are replaced. The result is therefore a genuine
/// supervised policy state: applying the requested candidate proves the counterfactual goal in one
/// verifier step, while a scorer that ignores the target cannot satisfy both labels reliably.
pub fn counterfactual_proof_action_target(
    problem: &RuliadProofProblem,
    actions: &RuliadProofActionSet,
    candidate_index: usize,
) -> Result<(RuliadProofProblem, RuliadProofActionSet)> {
    let goal = problem
        .goals
        .get(actions.goal)
        .ok_or_else(|| anyhow!("proof action goal {} is out of range", actions.goal))?;
    if goal.claim.rhs != actions.target {
        return Err(anyhow!(
            "proof action target does not match goal {} in the formal problem",
            actions.goal
        ));
    }
    if actions.is_equivalent_index(candidate_index) {
        return Err(anyhow!(
            "counterfactual proof target must select a non-equivalent action"
        ));
    }
    let target = actions
        .candidates
        .get(candidate_index)
        .ok_or_else(|| anyhow!("proof action candidate {candidate_index} is out of range"))?
        .outcome
        .clone()
        .ok_or_else(|| anyhow!("counterfactual proof action is not verifier-valid"))?;
    if target == actions.current {
        return Err(anyhow!(
            "counterfactual proof action must advance beyond the current state"
        ));
    }

    let mut retargeted_problem = problem.clone();
    retargeted_problem.goals[actions.goal].claim.rhs = target.clone();

    let mut retargeted_actions = actions.clone();
    retargeted_actions.target = target.clone();
    retargeted_actions.selected_index = candidate_index;
    retargeted_actions.equivalent_indices = retargeted_actions
        .candidates
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| {
            (candidate.outcome.as_ref() == Some(&target)).then_some(index)
        })
        .collect();
    for candidate in &mut retargeted_actions.candidates {
        candidate.distance_to_goal = candidate
            .outcome
            .as_ref()
            .map(|outcome| ruliad_term_distance(outcome, &target));
    }

    let transition = RuliadGoalTransitionKernel::new(
        &retargeted_problem,
        actions.goal,
        RuliadKernelLimits::default(),
    )
    .map_err(|failure| anyhow!("invalid counterfactual proof target: {}", failure.message))?;
    let selected = &retargeted_actions.candidates[candidate_index];
    let verified = transition
        .apply(&retargeted_actions.current, &selected.step)
        .map_err(|failure| anyhow!("counterfactual proof action failed: {}", failure.message))?;
    if verified != target {
        return Err(anyhow!(
            "counterfactual proof action does not reach the requested target"
        ));
    }

    Ok((retargeted_problem, retargeted_actions))
}

pub fn parse_proof_action_index(answer: &str) -> Option<usize> {
    let mut fields = answer.trim().split(';');
    let field = fields.next()?;
    if fields.next().is_some() {
        return None;
    }
    field.strip_prefix("c=")?.parse().ok()
}

/// Encode a candidate using the configured model/verifier contract.
pub fn proof_action_answer(
    actions: &RuliadProofActionSet,
    candidate_index: usize,
    contract: crate::ruliad::config::RuliadProofActionAnswerContract,
) -> Result<String> {
    let candidate = actions
        .candidates
        .get(candidate_index)
        .ok_or_else(|| anyhow!("proof action candidate {candidate_index} is out of range"))?;
    Ok(match contract {
        crate::ruliad::config::RuliadProofActionAnswerContract::PresentationIndex => {
            format!("c={candidate_index}")
        }
        crate::ruliad::config::RuliadProofActionAnswerContract::SemanticStep => {
            crate::ruliad::wire::encode_model_proof_step(actions.goal, &candidate.step)
        }
    })
}

/// Resolve a contract-bound answer to the represented candidate.
pub fn resolve_proof_action_answer(
    actions: &RuliadProofActionSet,
    answer: &str,
    contract: crate::ruliad::config::RuliadProofActionAnswerContract,
) -> Option<usize> {
    match contract {
        crate::ruliad::config::RuliadProofActionAnswerContract::PresentationIndex => {
            let index = parse_proof_action_index(answer)?;
            (index < actions.candidates.len()).then_some(index)
        }
        crate::ruliad::config::RuliadProofActionAnswerContract::SemanticStep => {
            let (goal, step) = crate::ruliad::wire::decode_model_proof_step(answer)?;
            (goal == actions.goal)
                .then(|| {
                    actions
                        .candidates
                        .iter()
                        .position(|candidate| candidate.step == step)
                })
                .flatten()
        }
    }
}

/// Construct a deterministic action menu for one oracle transition.
///
/// Every menu contains the oracle action and, where the proof state permits,
/// verifier-valid hard negatives. Candidate order is derived from the problem
/// hash and transition index, so the correct label has no fixed position.
pub fn oracle_proof_action_set(
    problem: &RuliadProofProblem,
    certificate: &RuliadProofCertificate,
    step_index: usize,
    maximum_candidates: usize,
) -> Result<RuliadProofActionSet> {
    let maximum_candidates = maximum_candidates.max(2);
    let prefix = certificate
        .prefix_before(step_index)
        .ok_or_else(|| anyhow!("proof action step index {step_index} is out of bounds"))?;
    let (goal, oracle_step) = certificate
        .step_at(step_index)
        .ok_or_else(|| anyhow!("proof action step index {step_index} is out of bounds"))?;
    let local_steps = prefix
        .goals
        .iter()
        .find(|candidate| candidate.goal == goal)
        .map(|candidate| candidate.steps.as_slice())
        .unwrap_or_default();
    proof_action_set_for_state(
        problem,
        goal,
        local_steps,
        Some(oracle_step),
        maximum_candidates,
        action_seed(&certificate.problem_hash, step_index),
        None,
    )
}

fn proof_action_set_for_state(
    problem: &RuliadProofProblem,
    goal: usize,
    local_steps: &[RuliadProofStep],
    preferred_step: Option<&RuliadProofStep>,
    maximum_candidates: usize,
    seed: u64,
    excluded_state_keys: Option<&BTreeSet<String>>,
) -> Result<RuliadProofActionSet> {
    let transition = RuliadGoalTransitionKernel::new(problem, goal, RuliadKernelLimits::default())
        .map_err(|failure| anyhow!("invalid proof policy problem: {}", failure.message))?;
    let current = transition
        .replay_prefix(local_steps)
        .map_err(|failure| anyhow!("invalid proof policy state: {}", failure.message))?;
    let target = transition.target().clone();

    let mut candidates = enumerate_action_candidates(
        problem,
        &transition,
        goal,
        &current,
        &target,
        preferred_step,
        seed,
    );
    candidates.retain(|candidate| {
        candidate.outcome.as_ref().is_some_and(|outcome| {
            outcome != &current
                && excluded_state_keys
                    .is_none_or(|excluded| !excluded.contains(&proof_state_key(goal, outcome)))
        })
    });
    if candidates.is_empty() {
        return Err(anyhow!("proof policy state has no candidate actions"));
    }

    let preferred_outcome = preferred_step.and_then(|preferred| {
        candidates
            .iter()
            .find(|candidate| candidate.step == *preferred)
            .and_then(|candidate| candidate.outcome.clone())
    });
    select_hard_candidate_subset(&mut candidates, preferred_step, maximum_candidates, seed);

    let selected_index = preferred_step
        .and_then(|preferred| {
            candidates
                .iter()
                .position(|candidate| candidate.step == *preferred)
        })
        .unwrap_or_else(|| {
            candidates
                .iter()
                .enumerate()
                .min_by_key(|(_, candidate)| candidate.distance_to_goal.unwrap_or(usize::MAX))
                .map(|(index, _)| index)
                .unwrap_or(0)
        });
    let selected_outcome = preferred_outcome
        .or_else(|| {
            candidates
                .get(selected_index)
                .and_then(|candidate| candidate.outcome.clone())
        })
        .ok_or_else(|| anyhow!("selected proof action is not verifier-valid"))?;
    let equivalent_indices = candidates
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| {
            (candidate.outcome.as_ref() == Some(&selected_outcome)).then_some(index)
        })
        .collect();

    Ok(RuliadProofActionSet {
        goal,
        current,
        target,
        candidates,
        selected_index,
        equivalent_indices,
    })
}

fn enumerate_action_candidates(
    problem: &RuliadProofProblem,
    transition: &RuliadGoalTransitionKernel<'_>,
    goal: usize,
    current: &RuliadTerm,
    target: &RuliadTerm,
    preferred_step: Option<&RuliadProofStep>,
    seed: u64,
) -> Vec<RuliadProofActionCandidate> {
    let Some(goal_spec) = problem.goals.get(goal) else {
        return Vec::new();
    };
    let mut sources = Vec::new();
    if let Some(step) = preferred_step {
        sources.push(step.source.clone());
    }
    sources.extend(
        goal_spec
            .dependencies
            .iter()
            .copied()
            .map(|goal| RuliadProofSource::Lemma { goal }),
    );
    sources.extend(problem.axioms.iter().map(|axiom| RuliadProofSource::Axiom {
        id: axiom.id.clone(),
    }));
    dedup_preserving_order(&mut sources);
    sources.truncate(MAX_ENUMERATED_SOURCES);

    let mut paths = Vec::new();
    if let Some(step) = preferred_step {
        paths.push(step.path.clone());
    }
    paths.push(first_difference_path(current, target));
    collect_term_paths(current, &mut Vec::new(), &mut paths, MAX_ENUMERATED_PATHS);
    dedup_preserving_order(&mut paths);
    paths.truncate(MAX_ENUMERATED_PATHS);

    let mut steps = Vec::new();
    if let Some(step) = preferred_step {
        steps.push(step.clone());
    }
    for source in sources {
        for direction in [
            RuliadRewriteDirection::Forward,
            RuliadRewriteDirection::Reverse,
        ] {
            for path in &paths {
                steps.push(RuliadProofStep {
                    source: source.clone(),
                    path: path.clone(),
                    direction,
                });
            }
        }
    }
    dedup_preserving_order(&mut steps);
    if preferred_step.is_some() && steps.len() > 1 {
        let mut tail = steps.split_off(1);
        deterministic_shuffle(&mut tail, seed ^ 0xe703_7ed1_a0b4_28db);
        tail.truncate(MAX_TRIAL_ACTIONS.saturating_sub(1));
        steps.extend(tail);
    } else {
        deterministic_shuffle(&mut steps, seed ^ 0xe703_7ed1_a0b4_28db);
    }

    steps
        .into_iter()
        .map(|step| {
            let outcome = transition.apply(current, &step).ok();
            let distance_to_goal = outcome
                .as_ref()
                .map(|outcome| ruliad_term_distance(outcome, target));
            RuliadProofActionCandidate {
                step,
                outcome,
                distance_to_goal,
            }
        })
        .collect()
}

fn first_difference_path(current: &RuliadTerm, target: &RuliadTerm) -> Vec<usize> {
    let mut path = Vec::new();
    let (mut current, mut target) = (current, target);
    while let (
        RuliadTerm::Apply {
            operator: current_operator,
            arguments: current_arguments,
        },
        RuliadTerm::Apply {
            operator: target_operator,
            arguments: target_arguments,
        },
    ) = (current, target)
    {
        if current_operator != target_operator || current_arguments.len() != target_arguments.len()
        {
            break;
        }
        let Some((index, (next_current, next_target))) = current_arguments
            .iter()
            .zip(target_arguments)
            .enumerate()
            .find(|(_, (current, target))| current != target)
        else {
            break;
        };
        path.push(index);
        current = next_current;
        target = next_target;
    }
    path
}

fn select_hard_candidate_subset(
    candidates: &mut Vec<RuliadProofActionCandidate>,
    preferred_step: Option<&RuliadProofStep>,
    maximum_candidates: usize,
    seed: u64,
) {
    let preferred = preferred_step.and_then(|preferred| {
        candidates
            .iter()
            .position(|candidate| candidate.step == *preferred)
            .map(|index| candidates.remove(index))
    });
    candidates.sort_by_key(|candidate| {
        (
            candidate.outcome.is_none(),
            candidate.distance_to_goal.unwrap_or(usize::MAX),
            action_sort_key(&candidate.step, seed),
        )
    });

    let mut selected = Vec::with_capacity(maximum_candidates);
    if let Some(preferred) = preferred {
        selected.push(preferred);
    }
    let preferred_outcome = selected
        .first()
        .and_then(|candidate| candidate.outcome.clone());
    for candidate in candidates
        .iter()
        .filter(|candidate| candidate.outcome.is_some() && candidate.outcome != preferred_outcome)
    {
        if selected.len() >= maximum_candidates {
            break;
        }
        selected.push(candidate.clone());
    }
    for candidate in candidates.iter() {
        if selected.len() >= maximum_candidates {
            break;
        }
        if !selected
            .iter()
            .any(|existing| existing.step == candidate.step)
        {
            selected.push(candidate.clone());
        }
    }

    deterministic_shuffle(&mut selected, seed ^ 0xa076_1d64_78bd_642f);
    *candidates = selected;
}

fn deterministic_shuffle<T>(items: &mut [T], mut state: u64) {
    for upper in (1..items.len()).rev() {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut mixed = state;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        mixed ^= mixed >> 31;
        items.swap(upper, mixed as usize % (upper + 1));
    }
}

fn collect_term_paths(
    term: &RuliadTerm,
    path: &mut Vec<usize>,
    output: &mut Vec<Vec<usize>>,
    limit: usize,
) {
    if output.len() >= limit {
        return;
    }
    output.push(path.clone());
    let RuliadTerm::Apply { arguments, .. } = term else {
        return;
    };
    for (index, argument) in arguments.iter().enumerate() {
        if output.len() >= limit {
            break;
        }
        path.push(index);
        collect_term_paths(argument, path, output, limit);
        path.pop();
    }
}

fn dedup_preserving_order<T: PartialEq>(items: &mut Vec<T>) {
    let mut index = 0;
    while index < items.len() {
        if items[..index].contains(&items[index]) {
            items.remove(index);
        } else {
            index += 1;
        }
    }
}

fn action_seed(problem_hash: &str, step_index: usize) -> u64 {
    problem_hash
        .bytes()
        .fold(step_index as u64 ^ 0x9e37_79b9_7f4a_7c15, |state, byte| {
            state.rotate_left(7) ^ u64::from(byte)
        })
}

fn action_sort_key(step: &RuliadProofStep, seed: u64) -> u64 {
    let source = match &step.source {
        RuliadProofSource::Axiom { id } => id
            .bytes()
            .fold(0u64, |acc, byte| acc.rotate_left(5) ^ u64::from(byte)),
        RuliadProofSource::Lemma { goal } => (*goal as u64).wrapping_mul(0x517c_c1b7_2722_0a95),
    };
    let path = step.path.iter().fold(0u64, |acc, index| {
        acc.rotate_left(9) ^ (*index as u64).wrapping_add(1)
    });
    source
        ^ path
        ^ seed
        ^ match step.direction {
            RuliadRewriteDirection::Forward => 0,
            RuliadRewriteDirection::Reverse => u64::MAX,
        }
}

pub fn ruliad_term_distance(left: &RuliadTerm, right: &RuliadTerm) -> usize {
    if left == right {
        return 0;
    }
    match (left, right) {
        (
            RuliadTerm::Apply {
                operator: left_operator,
                arguments: left_arguments,
            },
            RuliadTerm::Apply {
                operator: right_operator,
                arguments: right_arguments,
            },
        ) if left_operator == right_operator => {
            left_arguments.len().abs_diff(right_arguments.len())
                + left_arguments
                    .iter()
                    .zip(right_arguments)
                    .map(|(left, right)| ruliad_term_distance(left, right))
                    .sum::<usize>()
        }
        _ => term_node_count(left).saturating_add(term_node_count(right)),
    }
}

fn term_node_count(term: &RuliadTerm) -> usize {
    match term {
        RuliadTerm::Variable { .. } | RuliadTerm::Atom { .. } => 1,
        RuliadTerm::Apply { arguments, .. } => {
            1usize.saturating_add(arguments.iter().map(term_node_count).sum::<usize>())
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuliadProofPolicyState {
    goal_order: Vec<usize>,
    goal_cursor: usize,
    local_steps: Vec<RuliadProofStep>,
    visited: BTreeSet<String>,
}

impl RuliadProofPolicyState {
    pub fn new(problem: &RuliadProofProblem) -> Self {
        Self {
            goal_order: problem.required_goal_indices(),
            goal_cursor: 0,
            local_steps: Vec::new(),
            visited: BTreeSet::new(),
        }
    }

    /// Reconstruct the verifier state immediately before a certificate step. This is used to
    /// start policy training from the same randomly selected proof position represented by a
    /// generated sample rather than silently resetting every trajectory to step zero.
    pub fn from_certificate_prefix(
        problem: &RuliadProofProblem,
        certificate: &RuliadProofCertificate,
        step_index: usize,
    ) -> Result<Self> {
        if step_index > certificate.step_count() {
            return Err(anyhow!(
                "proof policy prefix step {step_index} exceeds certificate length {}",
                certificate.step_count()
            ));
        }
        let mut state = Self::new(problem);
        let mut remaining_steps = step_index;
        for goal_certificate in &certificate.goals {
            if remaining_steps == 0 {
                break;
            }
            let expected_goal = state
                .goal_order
                .get(state.goal_cursor)
                .copied()
                .ok_or_else(|| {
                    anyhow!("certificate prefix continues after every goal is solved")
                })?;
            if goal_certificate.goal != expected_goal {
                return Err(anyhow!(
                    "certificate goal {} does not match policy goal {expected_goal}",
                    goal_certificate.goal
                ));
            }
            let transition = RuliadGoalTransitionKernel::new(
                problem,
                expected_goal,
                RuliadKernelLimits::default(),
            )
            .map_err(|failure| anyhow!("invalid proof policy problem: {}", failure.message))?;
            let prefix_len = remaining_steps.min(goal_certificate.steps.len());
            let mut current = transition.initial();
            for step in goal_certificate.steps.iter().take(prefix_len) {
                current = transition.apply(&current, step).map_err(|failure| {
                    anyhow!("invalid proof policy prefix: {}", failure.message)
                })?;
                state.local_steps.push(step.clone());
                state
                    .visited
                    .insert(proof_state_key(expected_goal, &current));
            }
            remaining_steps = remaining_steps.saturating_sub(prefix_len);
            if prefix_len < goal_certificate.steps.len() {
                break;
            }
            if &current != transition.target() {
                return Err(anyhow!(
                    "certificate goal {expected_goal} does not reach its target"
                ));
            }
            state.goal_cursor = state.goal_cursor.saturating_add(1);
            state.local_steps.clear();
            state.visited.clear();
        }
        if remaining_steps > 0 {
            return Err(anyhow!(
                "certificate prefix is missing {remaining_steps} requested steps"
            ));
        }
        Ok(state)
    }

    pub fn solved(&self) -> bool {
        self.goal_cursor >= self.goal_order.len()
    }

    pub fn solved_goals(&self) -> usize {
        self.goal_cursor.min(self.goal_order.len())
    }

    pub fn total_goals(&self) -> usize {
        self.goal_order.len()
    }

    pub fn canonical_state_key(&self, problem: &RuliadProofProblem) -> Result<String> {
        if self.solved() {
            return Ok("solved".to_string());
        }
        let goal = self.goal_order[self.goal_cursor];
        let transition =
            RuliadGoalTransitionKernel::new(problem, goal, RuliadKernelLimits::default())
                .map_err(|failure| anyhow!("invalid proof policy problem: {}", failure.message))?;
        let current = transition
            .replay_prefix(&self.local_steps)
            .map_err(|failure| anyhow!("invalid proof policy state: {}", failure.message))?;
        Ok(proof_state_key(goal, &current))
    }

    /// Restore the previous state in the current goal while retaining the closed set of states
    /// already explored. This turns policy execution into bounded graph search rather than a
    /// single irreversible trajectory.
    pub fn backtrack(&mut self) -> bool {
        self.local_steps.pop().is_some()
    }

    pub fn action_set(
        &self,
        problem: &RuliadProofProblem,
        maximum_candidates: usize,
    ) -> Result<RuliadProofActionSet> {
        let goal = *self
            .goal_order
            .get(self.goal_cursor)
            .ok_or_else(|| anyhow!("proof policy is already solved"))?;
        let seed = action_seed(
            &problem.canonical_hash()?,
            self.local_steps
                .len()
                .saturating_add(self.goal_cursor << 20),
        );
        proof_action_set_for_state(
            problem,
            goal,
            &self.local_steps,
            None,
            maximum_candidates.max(2),
            seed,
            Some(&self.visited),
        )
    }

    pub fn apply(
        &mut self,
        action_set: &RuliadProofActionSet,
        candidate_index: usize,
    ) -> Result<bool> {
        let expected_goal = self
            .goal_order
            .get(self.goal_cursor)
            .copied()
            .ok_or_else(|| anyhow!("proof policy is already solved"))?;
        if action_set.goal != expected_goal {
            return Err(anyhow!("proof action set belongs to a different goal"));
        }
        let candidate = action_set
            .candidates
            .get(candidate_index)
            .ok_or_else(|| anyhow!("proof action candidate {candidate_index} is out of range"))?;
        let outcome = candidate
            .outcome
            .as_ref()
            .ok_or_else(|| anyhow!("proof action candidate is verifier-invalid"))?;
        self.local_steps.push(candidate.step.clone());
        let state_key = proof_state_key(expected_goal, outcome);
        let repeated = !self.visited.insert(state_key);
        if outcome == &action_set.target {
            self.goal_cursor = self.goal_cursor.saturating_add(1);
            self.local_steps.clear();
            self.visited.clear();
        }
        Ok(repeated)
    }
}

fn proof_state_key(goal: usize, term: &RuliadTerm) -> String {
    format!("{}:{}", goal, term.canonical_text())
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RuliadProofRolloutReport {
    pub solved: bool,
    pub steps: usize,
    pub valid_actions: usize,
    pub invalid_actions: usize,
    pub repeated_states: usize,
    pub backtracks: usize,
    pub solved_goals: usize,
    pub total_goals: usize,
}

pub fn rollout_proof_policy<F>(
    problem: &RuliadProofProblem,
    maximum_steps: usize,
    maximum_candidates: usize,
    mut select: F,
) -> RuliadProofRolloutReport
where
    F: FnMut(&RuliadProofPolicyState, &RuliadProofActionSet) -> Option<usize>,
{
    let mut state = RuliadProofPolicyState::new(problem);
    let mut report = RuliadProofRolloutReport {
        total_goals: state.total_goals(),
        ..RuliadProofRolloutReport::default()
    };
    for _ in 0..maximum_steps {
        if state.solved() {
            break;
        }
        let actions = match state.action_set(problem, maximum_candidates) {
            Ok(actions) => actions,
            Err(_) if state.backtrack() => {
                report.backtracks = report.backtracks.saturating_add(1);
                continue;
            }
            Err(_) => {
                report.invalid_actions = report.invalid_actions.saturating_add(1);
                break;
            }
        };
        let Some(index) = select(&state, &actions) else {
            report.invalid_actions = report.invalid_actions.saturating_add(1);
            break;
        };
        report.steps = report.steps.saturating_add(1);
        match state.apply(&actions, index) {
            Ok(repeated) => {
                report.valid_actions = report.valid_actions.saturating_add(1);
                report.repeated_states =
                    report.repeated_states.saturating_add(usize::from(repeated));
            }
            Err(_) => {
                report.invalid_actions = report.invalid_actions.saturating_add(1);
                break;
            }
        }
    }
    report.solved = state.solved();
    report.solved_goals = state.solved_goals();
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ruliad::formal::{RuliadFormalGeneratorConfig, generate_formal_bundle};

    fn bundle() -> crate::ruliad::ir::RuliadProofBundle {
        generate_formal_bundle(
            41,
            RuliadFormalGeneratorConfig {
                rewrite_depth: 3,
                leaf_count: 4,
                context_depth: 2,
                distractor_axioms: 2,
                ..RuliadFormalGeneratorConfig::default()
            },
        )
        .expect("formal bundle")
    }

    #[test]
    fn oracle_action_menu_contains_valid_hard_negatives_and_varied_labels() {
        let bundle = bundle();
        let mut labels = BTreeSet::new();
        for step_index in 0..bundle.certificate.step_count() {
            let actions = oracle_proof_action_set(
                &bundle.problem,
                &bundle.certificate,
                step_index,
                DEFAULT_PROOF_ACTION_CANDIDATES,
            )
            .expect("action set");
            assert!(actions.candidates.len() >= 2);
            assert!(
                actions
                    .candidates
                    .iter()
                    .all(|candidate| candidate.outcome.is_some())
            );
            assert!(
                actions
                    .selected()
                    .is_some_and(|candidate| candidate.outcome.is_some())
            );
            assert!(actions.is_equivalent_index(actions.selected_index));
            assert!(actions.candidates.iter().any(|candidate| {
                candidate.outcome.is_some()
                    && !actions.is_equivalent_index(
                        actions
                            .candidates
                            .iter()
                            .position(|item| item == candidate)
                            .expect("candidate position"),
                    )
            }));
            labels.insert(actions.selected_index);
        }
        assert!(
            labels.len() > 1,
            "correct action position must not be fixed"
        );
    }

    #[test]
    fn candidate_rotation_preserves_semantics_and_can_balance_every_label() {
        let bundle = bundle();
        let actions = oracle_proof_action_set(
            &bundle.problem,
            &bundle.certificate,
            0,
            DEFAULT_PROOF_ACTION_CANDIDATES,
        )
        .expect("action set");
        let selected_step = actions.selected().expect("selected action").step.clone();
        let equivalent_steps = actions
            .equivalent_indices
            .iter()
            .map(|index| actions.candidates[*index].step.clone())
            .collect::<Vec<_>>();

        for desired_index in 0..actions.candidates.len() {
            let rotated = actions
                .rotate_selected_to(desired_index)
                .expect("rotated action set");
            assert_eq!(rotated.selected_index, desired_index);
            assert_eq!(
                rotated.selected().expect("rotated selected action").step,
                selected_step
            );
            let rotated_equivalent_steps = rotated
                .equivalent_indices
                .iter()
                .map(|index| rotated.candidates[*index].step.clone())
                .collect::<Vec<_>>();
            assert_eq!(rotated_equivalent_steps.len(), equivalent_steps.len());
            assert!(
                rotated_equivalent_steps
                    .iter()
                    .all(|step| equivalent_steps.contains(step))
            );
        }

        for rotation in 0..actions.candidates.len() {
            let rotated = actions.rotate_left(rotation).expect("cyclic presentation");
            for presented_index in 0..rotated.candidates.len() {
                let original_index = actions
                    .original_index_after_rotation(presented_index, rotation)
                    .expect("original index");
                assert_eq!(
                    rotated.candidates[presented_index],
                    actions.candidates[original_index]
                );
                assert_eq!(
                    rotated.is_equivalent_index(presented_index),
                    actions.is_equivalent_index(original_index)
                );
            }
        }
    }

    #[test]
    fn counterfactual_target_preserves_laws_and_makes_alternate_action_exact() {
        let bundle = bundle();
        let actions = oracle_proof_action_set(
            &bundle.problem,
            &bundle.certificate,
            0,
            DEFAULT_PROOF_ACTION_CANDIDATES,
        )
        .expect("action set");
        let alternate_index = actions
            .candidates
            .iter()
            .enumerate()
            .find_map(|(index, candidate)| {
                (candidate.outcome.is_some() && !actions.is_equivalent_index(index))
                    .then_some(index)
            })
            .expect("verifier-valid alternate action");
        let alternate_target = actions.candidates[alternate_index]
            .outcome
            .clone()
            .expect("alternate outcome");

        let (retargeted_problem, retargeted_actions) =
            counterfactual_proof_action_target(&bundle.problem, &actions, alternate_index)
                .expect("counterfactual target");

        assert_eq!(retargeted_problem.axioms, bundle.problem.axioms);
        assert_eq!(
            retargeted_problem.goals[actions.goal].claim.lhs,
            bundle.problem.goals[actions.goal].claim.lhs
        );
        assert_eq!(
            retargeted_problem.goals[actions.goal].dependencies,
            bundle.problem.goals[actions.goal].dependencies
        );
        assert_eq!(
            retargeted_problem.goals[actions.goal].claim.rhs,
            alternate_target
        );
        assert_ne!(
            retargeted_problem
                .canonical_hash()
                .expect("retargeted hash"),
            bundle.problem.canonical_hash().expect("original hash")
        );
        assert_eq!(retargeted_actions.current, actions.current);
        assert_eq!(retargeted_actions.target, alternate_target);
        assert_eq!(retargeted_actions.selected_index, alternate_index);
        assert_eq!(
            retargeted_actions
                .candidates
                .iter()
                .map(|candidate| (&candidate.step, &candidate.outcome))
                .collect::<Vec<_>>(),
            actions
                .candidates
                .iter()
                .map(|candidate| (&candidate.step, &candidate.outcome))
                .collect::<Vec<_>>()
        );
        assert!(retargeted_actions.is_equivalent_index(alternate_index));
        assert!(retargeted_actions.equivalent_indices.iter().all(|index| {
            retargeted_actions.candidates[*index].outcome.as_ref()
                == Some(&retargeted_actions.target)
        }));
        assert_eq!(
            retargeted_actions.candidates[alternate_index].distance_to_goal,
            Some(0)
        );

        let transition = RuliadGoalTransitionKernel::new(
            &retargeted_problem,
            actions.goal,
            RuliadKernelLimits::default(),
        )
        .expect("counterfactual transition");
        assert_eq!(
            transition
                .apply(
                    &retargeted_actions.current,
                    &retargeted_actions.candidates[alternate_index].step,
                )
                .expect("apply alternate action"),
            retargeted_actions.target
        );
    }

    #[test]
    fn counterfactual_target_rejects_the_existing_equivalence_class() {
        let bundle = bundle();
        let actions = oracle_proof_action_set(
            &bundle.problem,
            &bundle.certificate,
            0,
            DEFAULT_PROOF_ACTION_CANDIDATES,
        )
        .expect("action set");

        let error =
            counterfactual_proof_action_target(&bundle.problem, &actions, actions.selected_index)
                .expect_err("existing target is not counterfactual");
        assert!(error.to_string().contains("non-equivalent"));
    }

    #[test]
    fn semantic_action_answer_is_rotation_invariant_and_resolves_to_executable_step() {
        let bundle = bundle();
        let actions = oracle_proof_action_set(
            &bundle.problem,
            &bundle.certificate,
            0,
            DEFAULT_PROOF_ACTION_CANDIDATES,
        )
        .expect("action set");
        let contract = crate::ruliad::config::RuliadProofActionAnswerContract::SemanticStep;
        let expected = proof_action_answer(&actions, actions.selected_index, contract)
            .expect("semantic answer");

        for rotation in 0..actions.candidates.len() {
            let rotated = actions.rotate_left(rotation).expect("rotated actions");
            let answer = proof_action_answer(&rotated, rotated.selected_index, contract)
                .expect("rotated semantic answer");
            assert_eq!(answer, expected);
            assert_eq!(
                resolve_proof_action_answer(&rotated, &answer, contract),
                Some(rotated.selected_index)
            );
            assert_eq!(
                resolve_proof_action_answer(&rotated, "c=0", contract),
                None,
                "semantic contract must not accept a presentation-relative shortcut"
            );
        }
    }

    #[test]
    fn greedy_distance_policy_solves_generated_proof_closed_loop() {
        let bundle = bundle();
        let report = rollout_proof_policy(
            &bundle.problem,
            bundle.certificate.step_count().saturating_mul(4),
            DEFAULT_PROOF_ACTION_CANDIDATES,
            |_state, actions| {
                actions
                    .candidates
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, candidate)| candidate.distance_to_goal.unwrap_or(usize::MAX))
                    .map(|(index, _)| index)
            },
        );
        assert!(report.solved, "{report:?}");
        assert_eq!(report.invalid_actions, 0);
        assert_eq!(report.repeated_states, 0);
        assert_eq!(report.solved_goals, report.total_goals);
    }

    #[test]
    fn certificate_prefix_reconstruction_matches_oracle_transition_state() {
        let bundle = bundle();
        let mut reference = RuliadProofPolicyState::new(&bundle.problem);
        for step_index in 0..=bundle.certificate.step_count() {
            let state = RuliadProofPolicyState::from_certificate_prefix(
                &bundle.problem,
                &bundle.certificate,
                step_index,
            )
            .expect("certificate prefix state");
            assert_eq!(state, reference, "prefix step {step_index}");
            if step_index < bundle.certificate.step_count() {
                let oracle = oracle_proof_action_set(
                    &bundle.problem,
                    &bundle.certificate,
                    step_index,
                    DEFAULT_PROOF_ACTION_CANDIDATES,
                )
                .expect("oracle transition");
                reference
                    .apply(&oracle, oracle.selected_index)
                    .expect("apply oracle transition");
            }
        }
    }

    #[test]
    fn runtime_action_menu_prunes_current_and_visited_states() {
        let bundle = bundle();
        let mut state = RuliadProofPolicyState::new(&bundle.problem);
        for _ in 0..bundle.certificate.step_count().saturating_mul(4) {
            if state.solved() {
                break;
            }
            let actions = state
                .action_set(&bundle.problem, DEFAULT_PROOF_ACTION_CANDIDATES)
                .expect("non-repeating action set");
            assert!(actions.candidates.iter().all(|candidate| {
                candidate.outcome.as_ref().is_none_or(|outcome| {
                    outcome != &actions.current
                        && !state
                            .visited
                            .contains(&proof_state_key(actions.goal, outcome))
                })
            }));
            assert!(
                !state
                    .apply(&actions, actions.selected_index)
                    .expect("apply selected action")
            );
        }
        assert!(state.solved());
    }

    #[test]
    fn backtracking_restores_parent_and_keeps_explored_branch_closed() {
        let bundle = bundle();
        let mut state = RuliadProofPolicyState::new(&bundle.problem);
        let actions = state
            .action_set(&bundle.problem, DEFAULT_PROOF_ACTION_CANDIDATES)
            .expect("initial actions");
        let parent = actions.current.clone();
        let branch_index = actions
            .candidates
            .iter()
            .enumerate()
            .find_map(|(index, candidate)| {
                candidate
                    .outcome
                    .as_ref()
                    .filter(|outcome| *outcome != &actions.target)
                    .map(|_| index)
            })
            .expect("non-terminal branch");
        let explored = actions.candidates[branch_index]
            .outcome
            .clone()
            .expect("verifier-valid branch");
        assert!(!state.apply(&actions, branch_index).expect("apply branch"));
        assert!(state.backtrack());

        let restored = state
            .action_set(&bundle.problem, DEFAULT_PROOF_ACTION_CANDIDATES)
            .expect("restored parent actions");
        assert_eq!(restored.current, parent);
        assert!(
            restored
                .candidates
                .iter()
                .all(|candidate| candidate.outcome.as_ref() != Some(&explored))
        );
    }

    #[test]
    fn invalid_policy_is_rejected_without_corrupting_state() {
        let bundle = bundle();
        let report = rollout_proof_policy(
            &bundle.problem,
            8,
            DEFAULT_PROOF_ACTION_CANDIDATES,
            |_state, actions| Some(actions.candidates.len()),
        );
        assert!(!report.solved);
        assert_eq!(report.invalid_actions, 1);
        assert_eq!(report.valid_actions, 0);
    }
}
