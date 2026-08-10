use anyhow::{Result, anyhow};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData};
use burn_dragon_core::DragonModel;

pub(crate) fn counterfactual_candidate_indices(
    actions: &burn_dragon_universality::ruliad::RuliadProofActionSet,
    maximum: usize,
    offset: usize,
) -> Vec<usize> {
    let candidate_count = actions.candidates.len();
    if maximum == 0 || candidate_count == 0 {
        return Vec::new();
    }
    let mut outcomes = Vec::new();
    let mut indices = Vec::new();
    for step in 0..candidate_count {
        let index = (offset + step) % candidate_count;
        let Some(outcome) = actions
            .candidates
            .get(index)
            .and_then(|candidate| candidate.outcome.as_ref())
        else {
            continue;
        };
        if actions.is_equivalent_index(index)
            || outcome == &actions.current
            || outcomes.contains(outcome)
        {
            continue;
        }
        outcomes.push(outcome.clone());
        indices.push(index);
        if indices.len() >= maximum {
            break;
        }
    }
    indices
}

pub(crate) fn candidate_presentation_rotations(
    symmetry: crate::config::RuliadProofPolicyCandidateSymmetry,
    selected_index: usize,
    candidate_count: usize,
    presentation_index: usize,
) -> Result<Vec<usize>> {
    if candidate_count < 2 || selected_index >= candidate_count {
        return Err(anyhow!(
            "proof-action presentation requires at least two candidates and a valid selected index"
        ));
    }
    Ok(match symmetry {
        crate::config::RuliadProofPolicyCandidateSymmetry::Canonical => vec![0],
        crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation => {
            let desired_index = presentation_index % candidate_count;
            vec![(selected_index + candidate_count - desired_index) % candidate_count]
        }
        crate::config::RuliadProofPolicyCandidateSymmetry::CyclicOrbitAverage => {
            (0..candidate_count).collect()
        }
    })
}

/// Choose candidate-menu rotations for one target variant in a paired supervision group.
///
/// The base target establishes the presentation. Counterfactual targets must reuse that exact
/// rotation so the requested goal is the only changed input; independently balancing each target
/// would make presentation position a label shortcut.
pub(crate) fn target_group_presentation_rotations(
    symmetry: crate::config::RuliadProofPolicyCandidateSymmetry,
    selected_index: usize,
    candidate_count: usize,
    presentation_index: usize,
    base_rotations: Option<&[usize]>,
) -> Result<Vec<usize>> {
    if let Some(base_rotations) = base_rotations {
        if candidate_count < 2
            || base_rotations.is_empty()
            || base_rotations
                .iter()
                .any(|rotation| *rotation >= candidate_count)
        {
            return Err(anyhow!(
                "paired proof-action presentation requires valid non-empty base rotations"
            ));
        }
        return Ok(base_rotations.to_vec());
    }
    candidate_presentation_rotations(
        symmetry,
        selected_index,
        candidate_count,
        presentation_index,
    )
}

#[derive(Clone, Debug, PartialEq)]
pub struct SemanticActionOrbitSummary {
    pub averaged_log_probs: Vec<f32>,
    pub presentation_log_probs: Vec<(usize, Vec<f32>)>,
    pub js_divergence: f64,
    pub top1_consensus_fraction: f64,
    pub complete_cyclic_orbit: bool,
}

fn semantic_presentation_log_probs(
    rotation: usize,
    scores: &[f32],
    candidate_count: usize,
) -> Result<Vec<f32>> {
    if scores.len() != candidate_count || scores.iter().any(|score| !score.is_finite()) {
        return Err(anyhow!(
            "proof-action presentation scores do not match the semantic candidate set"
        ));
    }
    let mut semantic = vec![f32::NEG_INFINITY; candidate_count];
    for (presented_index, score) in scores.iter().copied().enumerate() {
        let original_index = (presented_index + rotation % candidate_count) % candidate_count;
        semantic[original_index] = score;
    }
    Ok(semantic)
}

fn log_prob_entropy(log_probs: &[f32]) -> f64 {
    log_probs
        .iter()
        .map(|log_probability| {
            let probability = f64::from(*log_probability).exp();
            -probability * f64::from(*log_probability)
        })
        .sum()
}

/// Map presentation-indexed probabilities back to semantic candidates and summarize the orbit.
///
/// Averaging probabilities, rather than logits, is the finite-group Reynolds operator. The
/// generalized Jensen-Shannon divergence additionally exposes whether individual presentations
/// agree before averaging hides their disagreement.
pub(crate) fn semantic_action_orbit_summary(
    presentation_scores: &[(usize, Vec<f32>)],
    candidate_count: usize,
) -> Result<SemanticActionOrbitSummary> {
    if presentation_scores.is_empty() || candidate_count < 2 {
        return Err(anyhow!(
            "semantic proof-action scoring requires a non-empty candidate orbit"
        ));
    }
    let mut probabilities = vec![0.0f64; candidate_count];
    let mut semantic_presentations = Vec::with_capacity(presentation_scores.len());
    let mut rotations = std::collections::BTreeSet::new();
    for (rotation, scores) in presentation_scores {
        let semantic = semantic_presentation_log_probs(*rotation, scores, candidate_count)?;
        for (index, score) in semantic.iter().enumerate() {
            probabilities[index] += f64::from(*score).exp();
        }
        rotations.insert(rotation % candidate_count);
        semantic_presentations.push((*rotation % candidate_count, semantic));
    }
    let divisor = presentation_scores.len() as f64;
    for probability in &mut probabilities {
        *probability /= divisor;
    }
    let normalizer = probabilities.iter().sum::<f64>();
    if !normalizer.is_finite() || normalizer <= 0.0 {
        return Err(anyhow!(
            "semantic proof-action probability orbit is non-finite"
        ));
    }
    let averaged_log_probs = probabilities
        .into_iter()
        .map(|probability| (probability / normalizer).max(1.0e-30).ln() as f32)
        .collect::<Vec<_>>();
    let mean_presentation_entropy = semantic_presentations
        .iter()
        .map(|(_, scores)| log_prob_entropy(scores))
        .sum::<f64>()
        / semantic_presentations.len() as f64;
    let js_divergence =
        (log_prob_entropy(&averaged_log_probs) - mean_presentation_entropy).max(0.0);
    let averaged_top1 = best_candidate_index(&averaged_log_probs);
    let top1_consensus_fraction = averaged_top1.map_or(0.0, |averaged_top1| {
        semantic_presentations
            .iter()
            .filter(|(_, scores)| best_candidate_index(scores) == Some(averaged_top1))
            .count() as f64
            / semantic_presentations.len() as f64
    });
    Ok(SemanticActionOrbitSummary {
        averaged_log_probs,
        presentation_log_probs: semantic_presentations,
        js_divergence,
        top1_consensus_fraction,
        complete_cyclic_orbit: presentation_scores.len() == candidate_count
            && rotations.len() == candidate_count,
    })
}

/// One encoded presentation of a verifier-enumerated proof-action menu.
///
/// `rotation` maps the presented candidate order back to the request's semantic order. Prompt
/// and candidate tokens use the same tokenizer as the Dragon language head.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedRuliadProofActionPresentation {
    pub rotation: usize,
    pub prompt_tokens: Vec<i64>,
    pub candidate_tokens: Vec<Vec<i64>>,
}

/// A typed proof-policy query, potentially carrying a complete finite presentation orbit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedRuliadProofActionRequest {
    pub presentations: Vec<EncodedRuliadProofActionPresentation>,
    pub answer_contract: burn_dragon_universality::ruliad::RuliadProofActionAnswerContract,
}

/// A semantic proof-policy decision with a deterministic surface rendering.
///
/// The selected completion is copied from the verifier-provided candidate set. Consequently,
/// callers never need to parse an unconstrained autoregressive completion to apply the action.
#[derive(Clone, Debug, PartialEq)]
pub struct RuliadProofActionDecision {
    pub selected_semantic_index: usize,
    pub selected_completion_tokens: Vec<i64>,
    pub orbit: SemanticActionOrbitSummary,
}

fn validate_encoded_action_request(request: &EncodedRuliadProofActionRequest) -> Result<usize> {
    let candidate_count = request
        .presentations
        .first()
        .map(|presentation| presentation.candidate_tokens.len())
        .ok_or_else(|| anyhow!("typed proof-action request has no presentations"))?;
    if candidate_count < 2 {
        return Err(anyhow!(
            "typed proof-action request requires at least two candidates"
        ));
    }
    if request.presentations.iter().any(|presentation| {
        presentation.prompt_tokens.is_empty()
            || presentation.candidate_tokens.len() != candidate_count
            || presentation.candidate_tokens.iter().any(Vec::is_empty)
    }) {
        return Err(anyhow!(
            "typed proof-action presentations require non-empty prompts and a stable non-empty candidate set"
        ));
    }
    Ok(candidate_count)
}

fn selected_completion_tokens(
    presentation: &EncodedRuliadProofActionPresentation,
    semantic_index: usize,
) -> Result<Vec<i64>> {
    let candidate_count = presentation.candidate_tokens.len();
    if semantic_index >= candidate_count {
        return Err(anyhow!(
            "typed proof-action semantic index exceeds the candidate set"
        ));
    }
    let presented_index = (semantic_index + candidate_count
        - presentation.rotation % candidate_count)
        % candidate_count;
    presentation
        .candidate_tokens
        .get(presented_index)
        .cloned()
        .ok_or_else(|| anyhow!("typed proof-action presentation mapping is inconsistent"))
}

/// Score and select verifier-enumerated proof actions with bounded tensorized forwards.
///
/// Requests may use canonical, balanced, or complete cyclic-orbit presentations. Scores are
/// mapped back to semantic candidate order and averaged as probabilities before selection. The
/// row bound is a hard launch/memory bound; it does not change the resulting decision.
pub fn select_ruliad_proof_actions_batch<B>(
    model: &DragonModel<B>,
    requests: &[EncodedRuliadProofActionRequest],
    max_batch_rows: usize,
    device: &B::Device,
) -> Result<Vec<RuliadProofActionDecision>>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    select_ruliad_proof_actions_batch_with_scoring(
        model,
        requests,
        max_batch_rows,
        crate::config::RuliadProofPolicyScoring::CompletionLikelihood,
        device,
    )
}

/// Score typed proof actions with an explicit completion-likelihood or sequence-energy contract.
pub fn select_ruliad_proof_actions_batch_with_scoring<B>(
    model: &DragonModel<B>,
    requests: &[EncodedRuliadProofActionRequest],
    max_batch_rows: usize,
    scoring: crate::config::RuliadProofPolicyScoring,
    device: &B::Device,
) -> Result<Vec<RuliadProofActionDecision>>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    select_ruliad_proof_actions_batch_with_contract(
        model,
        requests,
        max_batch_rows,
        scoring,
        crate::config::RuliadProofPolicyNormalization::CandidateConditional,
        device,
    )
}

/// Score typed proof actions with the same score and normalization contract used for training.
///
/// In particular, semantic-step completion policies trained with `PrefixConditional` are ranked
/// by their constrained action-trie probability. Deterministic serialization tokens are omitted
/// from both training and inference, so syntax likelihood cannot dilute a semantic branch.
pub fn select_ruliad_proof_actions_batch_with_contract<B>(
    model: &DragonModel<B>,
    requests: &[EncodedRuliadProofActionRequest],
    max_batch_rows: usize,
    scoring: crate::config::RuliadProofPolicyScoring,
    normalization: crate::config::RuliadProofPolicyNormalization,
    device: &B::Device,
) -> Result<Vec<RuliadProofActionDecision>>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    if requests.is_empty() {
        return Ok(Vec::new());
    }
    if max_batch_rows == 0 {
        return Err(anyhow!(
            "typed proof-action scoring requires a positive row bound"
        ));
    }
    let candidate_counts = requests
        .iter()
        .map(validate_encoded_action_request)
        .collect::<Result<Vec<_>>>()?;
    let flat_presentations = requests
        .iter()
        .enumerate()
        .flat_map(|(request_index, request)| {
            request
                .presentations
                .iter()
                .map(move |presentation| (request_index, request.answer_contract, presentation))
        })
        .collect::<Vec<_>>();
    let mut scores_by_request = requests
        .iter()
        .map(|request| Vec::with_capacity(request.presentations.len()))
        .collect::<Vec<Vec<(usize, Vec<f32>)>>>();

    let mut offset = 0usize;
    while offset < flat_presentations.len() {
        let answer_contract = flat_presentations[offset].1;
        let mut end = offset;
        while end < flat_presentations.len()
            && end - offset < max_batch_rows
            && flat_presentations[end].1 == answer_contract
        {
            end = end.saturating_add(1);
        }
        let chunk = &flat_presentations[offset..end];
        let prompts = chunk
            .iter()
            .map(|(_, _, presentation)| presentation.prompt_tokens.clone())
            .collect::<Vec<_>>();
        let candidates = chunk
            .iter()
            .map(|(_, _, presentation)| presentation.candidate_tokens.clone())
            .collect::<Vec<_>>();
        let scores = proof_action_scores_batch_with_normalization(
            model,
            &prompts,
            &candidates,
            answer_contract,
            scoring,
            normalization,
            device,
        )?;
        for ((request_index, _, presentation), scores) in chunk.iter().zip(scores) {
            scores_by_request[*request_index].push((presentation.rotation, scores));
        }
        offset = end;
    }

    requests
        .iter()
        .zip(candidate_counts)
        .zip(scores_by_request)
        .map(|((request, candidate_count), presentation_scores)| {
            let orbit = semantic_action_orbit_summary(&presentation_scores, candidate_count)?;
            let selected_semantic_index = best_candidate_index(&orbit.averaged_log_probs)
                .ok_or_else(|| anyhow!("typed proof-action scores have no finite candidate"))?;
            let selected_completion_tokens = selected_completion_tokens(
                request
                    .presentations
                    .first()
                    .expect("validated request has a presentation"),
                selected_semantic_index,
            )?;
            Ok(RuliadProofActionDecision {
                selected_semantic_index,
                selected_completion_tokens,
                orbit,
            })
        })
        .collect()
}

pub(crate) fn semantic_action_log_probs(
    presentation_scores: &[(usize, Vec<f32>)],
    candidate_count: usize,
) -> Result<Vec<f32>> {
    Ok(semantic_action_orbit_summary(presentation_scores, candidate_count)?.averaged_log_probs)
}

pub(crate) struct DeferredConstrainedCompletionScores<B: Backend> {
    logits: Tensor<B, 2>,
    branch_tokens: Vec<Vec<i64>>,
    vocab: usize,
}

pub(crate) struct DeferredSequenceCompletionScores<B: Backend> {
    scores: Tensor<B, 1>,
    group_sizes: Vec<usize>,
}

pub(crate) struct DeferredTrieConditionalScores<B: Backend> {
    logits: Tensor<B, 2>,
    branches: Vec<SemanticCandidateTrieBranch>,
    group_branch_counts: Vec<usize>,
    group_candidate_counts: Vec<usize>,
    vocab: usize,
}

pub(crate) struct SequenceCompletionScoreTensor<B: Backend> {
    pub mean_log_scores: Tensor<B, 1>,
    pub sum_log_scores: Tensor<B, 1>,
    pub group_sizes: Vec<usize>,
}

type CandidateContinuationScoreGroup<B> = (usize, Tensor<B, 1>, Tensor<B, 1>);

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SemanticCandidateTrieBranch {
    pub prefix: Vec<i64>,
    pub candidate_tokens: Vec<i64>,
    pub equivalent_tokens: Vec<i64>,
    candidate_indices_by_token: Vec<Vec<usize>>,
}

/// Enumerate verifier-relevant decision points in a semantic-action token trie.
///
/// Deterministic syntax is skipped. At each actual branch, the target is the probability mass of
/// every child that can still reach a verifier-equivalent action. Recursion follows only those
/// children, so late branches cannot compensate for choosing an incorrect early goal or source.
pub(crate) fn semantic_candidate_trie_branches(
    candidates: &[Vec<i64>],
    equivalent_indices: &[usize],
) -> Result<Vec<SemanticCandidateTrieBranch>> {
    if candidates.len() < 2
        || candidates.iter().any(Vec::is_empty)
        || equivalent_indices.is_empty()
        || equivalent_indices
            .iter()
            .any(|index| *index >= candidates.len())
    {
        return Err(anyhow!(
            "semantic candidate trie requires non-empty candidates and valid equivalent indices"
        ));
    }
    let equivalent = equivalent_indices
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let mut branches = Vec::new();
    visit_semantic_candidate_trie(
        candidates,
        &equivalent,
        &(0..candidates.len()).collect::<Vec<_>>(),
        0,
        &mut Vec::new(),
        &mut branches,
    )?;
    if branches.is_empty() {
        return Err(anyhow!(
            "semantic candidate trie contains no verifier-relevant decision"
        ));
    }
    Ok(branches)
}

fn visit_semantic_candidate_trie(
    candidates: &[Vec<i64>],
    equivalent: &std::collections::BTreeSet<usize>,
    active: &[usize],
    depth: usize,
    prefix: &mut Vec<i64>,
    branches: &mut Vec<SemanticCandidateTrieBranch>,
) -> Result<()> {
    if !active.iter().any(|index| equivalent.contains(index)) {
        return Ok(());
    }
    let mut children = std::collections::BTreeMap::<i64, Vec<usize>>::new();
    for index in active {
        let token = candidates[*index].get(depth).copied().ok_or_else(|| {
            anyhow!("semantic candidate terminates before another candidate at a shared prefix")
        })?;
        children.entry(token).or_default().push(*index);
    }
    if children.len() > 1 {
        let equivalent_tokens = children
            .iter()
            .filter_map(|(token, indices)| {
                indices
                    .iter()
                    .any(|index| equivalent.contains(index))
                    .then_some(*token)
            })
            .collect::<Vec<_>>();
        if equivalent_tokens.is_empty() {
            return Err(anyhow!(
                "semantic candidate trie branch lost every equivalent action"
            ));
        }
        branches.push(SemanticCandidateTrieBranch {
            prefix: prefix.clone(),
            candidate_tokens: children.keys().copied().collect(),
            equivalent_tokens,
            candidate_indices_by_token: children.values().cloned().collect(),
        });
    }
    for (token, indices) in children {
        if !indices.iter().any(|index| equivalent.contains(index)) {
            continue;
        }
        prefix.push(token);
        let terminal = indices
            .iter()
            .all(|index| candidates[*index].len() == depth.saturating_add(1));
        if !terminal {
            visit_semantic_candidate_trie(
                candidates,
                equivalent,
                &indices,
                depth.saturating_add(1),
                prefix,
                branches,
            )?;
        }
        prefix.pop();
    }
    Ok(())
}

pub(crate) enum DeferredProofActionCompletionScores<B: Backend> {
    PresentationIndex(DeferredConstrainedCompletionScores<B>),
    SemanticStep(DeferredSequenceCompletionScores<B>),
    TrieConditional(DeferredTrieConditionalScores<B>),
    SemanticEnergy(DeferredSequenceCompletionScores<B>),
    ResidualEnergy(DeferredSequenceCompletionScores<B>),
}

/// Run the sequence model once and decode only the requested causal positions.
///
/// Proof-action candidates branch at one token. Decoding every prompt position creates a full
/// `[batch, time, vocab]` tensor that is immediately discarded, so gather hidden states before
/// applying the language head instead.
pub(crate) fn logits_at_sequence_positions<B>(
    model: &DragonModel<B>,
    inputs: Tensor<B, 2, Int>,
    positions: &[usize],
    device: &B::Device,
) -> Result<Tensor<B, 2>>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    let [batch_size, sequence_len] = inputs.shape().dims::<2>();
    if batch_size == 0 || positions.len() != batch_size {
        return Err(anyhow!(
            "branch-logit gather requires one sequence position per non-empty batch row"
        ));
    }
    if positions.iter().any(|position| *position >= sequence_len) {
        return Err(anyhow!(
            "branch-logit gather position exceeds the input sequence"
        ));
    }
    let hidden = model.forward_hidden(inputs);
    let [_, _, hidden_size] = hidden.shape().dims::<3>();
    let mut gather_values = Vec::with_capacity(batch_size.saturating_mul(hidden_size));
    for position in positions {
        let position = i64::try_from(*position)
            .map_err(|_| anyhow!("proof-action branch position exceeds i64"))?;
        gather_values.extend(std::iter::repeat_n(position, hidden_size));
    }
    let gather_indices = Tensor::<B, 3, Int>::from_data(
        TensorData::new(gather_values, [batch_size, 1, hidden_size]),
        device,
    );
    let branch_hidden = hidden.gather(1, gather_indices);
    let logits = model.logits_from_hidden(branch_hidden);
    let [_, _, vocab] = logits.shape().dims::<3>();
    Ok(logits.reshape([batch_size, vocab]))
}

impl<B> DeferredConstrainedCompletionScores<B>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    pub(crate) fn resolve(self) -> Result<Vec<Vec<f32>>> {
        let values = self
            .logits
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .map_err(|error| anyhow!("proof-action logits could not be read: {error:?}"))?;
        self.branch_tokens
            .iter()
            .enumerate()
            .map(|(row, tokens)| {
                normalize_candidate_scores(
                    &values[row * self.vocab..(row + 1) * self.vocab],
                    tokens,
                )
            })
            .collect()
    }
}

impl<B> DeferredSequenceCompletionScores<B>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    pub(crate) fn resolve(self) -> Result<Vec<Vec<f32>>> {
        let scores = self
            .scores
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .map_err(|error| {
                anyhow!("proof-action sequence scores could not be read: {error:?}")
            })?;
        let mut offset = 0usize;
        self.group_sizes
            .into_iter()
            .map(|group_size| {
                let end = offset.saturating_add(group_size);
                let group = scores.get(offset..end).ok_or_else(|| {
                    anyhow!("proof-action sequence score grouping is inconsistent")
                })?;
                offset = end;
                normalize_log_scores(group)
            })
            .collect::<Result<Vec<_>>>()
            .and_then(|groups| {
                (offset == scores.len())
                    .then_some(groups)
                    .ok_or_else(|| anyhow!("proof-action sequence scores contain trailing rows"))
            })
    }
}

impl<B> DeferredTrieConditionalScores<B>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    pub(crate) fn resolve(self) -> Result<Vec<Vec<f32>>> {
        let values = self
            .logits
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .map_err(|error| anyhow!("proof-action trie logits could not be read: {error:?}"))?;
        trie_conditional_log_scores(
            &values,
            self.vocab,
            &self.branches,
            &self.group_branch_counts,
            &self.group_candidate_counts,
        )
    }
}

fn trie_conditional_log_scores(
    values: &[f32],
    vocab: usize,
    branches: &[SemanticCandidateTrieBranch],
    group_branch_counts: &[usize],
    group_candidate_counts: &[usize],
) -> Result<Vec<Vec<f32>>> {
    if vocab == 0
        || group_branch_counts.len() != group_candidate_counts.len()
        || values.len() != branches.len().saturating_mul(vocab)
    {
        return Err(anyhow!(
            "proof-action trie score dimensions are inconsistent"
        ));
    }
    let mut branch_offset = 0usize;
    let groups = group_branch_counts
        .iter()
        .copied()
        .zip(group_candidate_counts.iter().copied())
        .map(|(branch_count, candidate_count)| {
            let mut candidate_scores = vec![0.0f32; candidate_count];
            for branch_index in branch_offset..branch_offset.saturating_add(branch_count) {
                let branch = branches
                    .get(branch_index)
                    .ok_or_else(|| anyhow!("proof-action trie branch grouping is inconsistent"))?;
                let row = &values[branch_index * vocab..(branch_index + 1) * vocab];
                let branch_log_probs = normalize_candidate_scores(row, &branch.candidate_tokens)?;
                for (log_probability, candidate_indices) in branch_log_probs
                    .into_iter()
                    .zip(&branch.candidate_indices_by_token)
                {
                    for candidate_index in candidate_indices {
                        let score =
                            candidate_scores.get_mut(*candidate_index).ok_or_else(|| {
                                anyhow!("proof-action trie references an invalid candidate")
                            })?;
                        *score += log_probability;
                    }
                }
            }
            branch_offset = branch_offset.saturating_add(branch_count);
            normalize_log_scores(&candidate_scores)
        })
        .collect::<Result<Vec<_>>>()?;
    if branch_offset != branches.len() {
        return Err(anyhow!(
            "proof-action trie scores contain trailing branches"
        ));
    }
    Ok(groups)
}

impl<B> DeferredProofActionCompletionScores<B>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    pub(crate) fn resolve(self) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::PresentationIndex(scores) => scores.resolve(),
            Self::SemanticStep(scores)
            | Self::SemanticEnergy(scores)
            | Self::ResidualEnergy(scores) => scores.resolve(),
            Self::TrieConditional(scores) => scores.resolve(),
        }
    }
}

/// Score grammar-constrained completions at their first distinguishing token.
///
/// Formal action completions share a fixed wrapper (`c=` and the document close marker). The
/// wrapper is teacher-forced once; only the action-bearing token competes. This keeps training and
/// deployment on the language head while avoiding autoregressive syntax generation and parser
/// failures for a finite verifier-provided action set.
pub(crate) fn constrained_completion_log_probs<B>(
    model: &DragonModel<B>,
    prompt_tokens: &[i64],
    candidate_tokens: &[Vec<i64>],
    device: &B::Device,
) -> Result<Vec<f32>>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    constrained_completion_log_probs_batch(
        model,
        &[prompt_tokens.to_vec()],
        &[candidate_tokens.to_vec()],
        device,
    )?
    .into_iter()
    .next()
    .ok_or_else(|| anyhow!("proof-action scorer returned no rows"))
}

/// Tensorized variant used by beam search. Right padding is consumed only after each row's
/// branch-logit position, so causal outputs at the gathered positions are identical to scoring
/// each variable-length prompt independently.
pub(crate) fn constrained_completion_log_probs_batch<B>(
    model: &DragonModel<B>,
    prompt_tokens: &[Vec<i64>],
    candidate_tokens: &[Vec<Vec<i64>>],
    device: &B::Device,
) -> Result<Vec<Vec<f32>>>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    enqueue_constrained_completion_log_probs_batch(model, prompt_tokens, candidate_tokens, device)?
        .resolve()
}

pub(crate) fn proof_action_completion_log_probs_batch<B>(
    model: &DragonModel<B>,
    prompt_tokens: &[Vec<i64>],
    candidate_tokens: &[Vec<Vec<i64>>],
    contract: burn_dragon_universality::ruliad::RuliadProofActionAnswerContract,
    device: &B::Device,
) -> Result<Vec<Vec<f32>>>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    enqueue_proof_action_completion_log_probs_batch(
        model,
        prompt_tokens,
        candidate_tokens,
        contract,
        device,
    )?
    .resolve()
}

pub(crate) fn proof_action_scores_batch<B>(
    model: &DragonModel<B>,
    prompt_tokens: &[Vec<i64>],
    candidate_tokens: &[Vec<Vec<i64>>],
    contract: burn_dragon_universality::ruliad::RuliadProofActionAnswerContract,
    scoring: crate::config::RuliadProofPolicyScoring,
    device: &B::Device,
) -> Result<Vec<Vec<f32>>>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    proof_action_scores_batch_with_normalization(
        model,
        prompt_tokens,
        candidate_tokens,
        contract,
        scoring,
        crate::config::RuliadProofPolicyNormalization::CandidateConditional,
        device,
    )
}

pub(crate) fn proof_action_scores_batch_with_normalization<B>(
    model: &DragonModel<B>,
    prompt_tokens: &[Vec<i64>],
    candidate_tokens: &[Vec<Vec<i64>>],
    contract: burn_dragon_universality::ruliad::RuliadProofActionAnswerContract,
    scoring: crate::config::RuliadProofPolicyScoring,
    normalization: crate::config::RuliadProofPolicyNormalization,
    device: &B::Device,
) -> Result<Vec<Vec<f32>>>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    enqueue_proof_action_scores_batch_with_normalization(
        model,
        prompt_tokens,
        candidate_tokens,
        contract,
        scoring,
        normalization,
        device,
    )?
    .resolve()
}

pub(crate) fn enqueue_proof_action_scores_batch<B>(
    model: &DragonModel<B>,
    prompt_tokens: &[Vec<i64>],
    candidate_tokens: &[Vec<Vec<i64>>],
    contract: burn_dragon_universality::ruliad::RuliadProofActionAnswerContract,
    scoring: crate::config::RuliadProofPolicyScoring,
    device: &B::Device,
) -> Result<DeferredProofActionCompletionScores<B>>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    enqueue_proof_action_scores_batch_with_normalization(
        model,
        prompt_tokens,
        candidate_tokens,
        contract,
        scoring,
        crate::config::RuliadProofPolicyNormalization::CandidateConditional,
        device,
    )
}

pub(crate) fn enqueue_proof_action_scores_batch_with_normalization<B>(
    model: &DragonModel<B>,
    prompt_tokens: &[Vec<i64>],
    candidate_tokens: &[Vec<Vec<i64>>],
    contract: burn_dragon_universality::ruliad::RuliadProofActionAnswerContract,
    scoring: crate::config::RuliadProofPolicyScoring,
    normalization: crate::config::RuliadProofPolicyNormalization,
    device: &B::Device,
) -> Result<DeferredProofActionCompletionScores<B>>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    match scoring {
        crate::config::RuliadProofPolicyScoring::CompletionLikelihood => {
            if contract
                == burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep
                && normalization == crate::config::RuliadProofPolicyNormalization::PrefixConditional
            {
                return Ok(DeferredProofActionCompletionScores::TrieConditional(
                    enqueue_trie_conditional_scores_batch(
                        model,
                        prompt_tokens,
                        candidate_tokens,
                        device,
                    )?,
                ));
            }
            enqueue_proof_action_completion_log_probs_batch(
                model,
                prompt_tokens,
                candidate_tokens,
                contract,
                device,
            )
        }
        crate::config::RuliadProofPolicyScoring::SemanticEnergy => {
            Ok(DeferredProofActionCompletionScores::SemanticEnergy(
                enqueue_sequence_energy_scores_batch(
                    model,
                    prompt_tokens,
                    candidate_tokens,
                    device,
                )?,
            ))
        }
        crate::config::RuliadProofPolicyScoring::ResidualEnergy => {
            Ok(DeferredProofActionCompletionScores::ResidualEnergy(
                enqueue_sequence_residual_energy_scores_batch(
                    model,
                    prompt_tokens,
                    candidate_tokens,
                    device,
                )?,
            ))
        }
    }
}

/// Queue every discriminative branch of each semantic-action trie in one model forward.
///
/// Candidate probabilities are products of legal-token conditional probabilities along their
/// paths. Shared deterministic syntax contributes neither score nor compute after the branch
/// prefixes are assembled, matching the prefix-conditional training terminal exactly.
fn enqueue_trie_conditional_scores_batch<B>(
    model: &DragonModel<B>,
    prompt_tokens: &[Vec<i64>],
    candidate_tokens: &[Vec<Vec<i64>>],
    device: &B::Device,
) -> Result<DeferredTrieConditionalScores<B>>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    let group_candidate_counts =
        validate_sequence_completion_inputs(prompt_tokens, candidate_tokens)?;
    let mut sequences = Vec::<Vec<i64>>::new();
    let mut branches = Vec::<SemanticCandidateTrieBranch>::new();
    let mut group_branch_counts = Vec::with_capacity(prompt_tokens.len());
    for (prompt, candidates) in prompt_tokens.iter().zip(candidate_tokens) {
        let all_candidates = (0..candidates.len()).collect::<Vec<_>>();
        let group_branches = semantic_candidate_trie_branches(candidates, &all_candidates)?;
        group_branch_counts.push(group_branches.len());
        for branch in group_branches {
            let mut sequence = prompt.clone();
            sequence.extend_from_slice(&branch.prefix);
            sequences.push(sequence);
            branches.push(branch);
        }
    }
    let maximum_len = sequences.iter().map(Vec::len).max().unwrap_or_default();
    if sequences.is_empty() || maximum_len == 0 {
        return Err(anyhow!("proof-action trie scorer has no branch inputs"));
    }
    let row_count = sequences.len();
    let mut values = vec![0i64; row_count.saturating_mul(maximum_len)];
    let mut positions = Vec::with_capacity(row_count);
    for (row, sequence) in sequences.into_iter().enumerate() {
        if sequence.is_empty() {
            return Err(anyhow!("proof-action trie branch has no causal input"));
        }
        let length = sequence.len();
        values[row * maximum_len..row * maximum_len + length].copy_from_slice(&sequence);
        positions.push(length.saturating_sub(1));
    }
    let inputs =
        Tensor::<B, 2, Int>::from_data(TensorData::new(values, [row_count, maximum_len]), device);
    let logits = logits_at_sequence_positions(model, inputs, &positions, device)?;
    let [_, vocab] = logits.shape().dims::<2>();
    if vocab == 0 {
        return Err(anyhow!(
            "proof-action trie scorer produced an empty vocabulary"
        ));
    }
    Ok(DeferredTrieConditionalScores {
        logits,
        branches,
        group_branch_counts,
        group_candidate_counts,
        vocab,
    })
}

pub(crate) fn enqueue_proof_action_completion_log_probs_batch<B>(
    model: &DragonModel<B>,
    prompt_tokens: &[Vec<i64>],
    candidate_tokens: &[Vec<Vec<i64>>],
    contract: burn_dragon_universality::ruliad::RuliadProofActionAnswerContract,
    device: &B::Device,
) -> Result<DeferredProofActionCompletionScores<B>>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    Ok(match contract {
        burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::PresentationIndex => {
            DeferredProofActionCompletionScores::PresentationIndex(
                enqueue_constrained_completion_log_probs_batch(
                    model,
                    prompt_tokens,
                    candidate_tokens,
                    device,
                )?,
            )
        }
        burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep => {
            DeferredProofActionCompletionScores::SemanticStep(
                enqueue_sequence_completion_log_probs_batch(
                    model,
                    prompt_tokens,
                    candidate_tokens,
                    device,
                )?,
            )
        }
    })
}

/// Score complete variable-length semantic actions with one tensorized model forward.
///
/// Mean token log-probability removes the otherwise strong preference for shorter path/source
/// encodings. The resulting candidate scores are normalized within each verifier-provided menu.
fn enqueue_sequence_completion_log_probs_batch<B>(
    model: &DragonModel<B>,
    prompt_tokens: &[Vec<i64>],
    candidate_tokens: &[Vec<Vec<i64>>],
    device: &B::Device,
) -> Result<DeferredSequenceCompletionScores<B>>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    let scores = sequence_completion_score_tensor_with_prefix_reuse(
        model,
        prompt_tokens,
        candidate_tokens,
        device,
    )?;
    Ok(DeferredSequenceCompletionScores {
        scores: scores.mean_log_scores,
        group_sizes: scores.group_sizes,
    })
}

/// Score complete semantic candidates from one encoding of each shared prompt.
///
/// Candidate rows inherit the exact recurrent state at the end of their prompt. Every continuation
/// is then gathered at its true terminal position, so right padding never enters the score. This
/// keeps the scalar energy head independent from the language projection without multiplying the
/// much longer formal prompt by the number of verifier-enumerated actions.
fn enqueue_sequence_energy_scores_batch<B>(
    model: &DragonModel<B>,
    prompt_tokens: &[Vec<i64>],
    candidate_tokens: &[Vec<Vec<i64>>],
    device: &B::Device,
) -> Result<DeferredSequenceCompletionScores<B>>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    let (scores, group_sizes) = sequence_energy_score_tensor_with_prefix_reuse(
        model,
        prompt_tokens,
        candidate_tokens,
        device,
    )?;
    Ok(DeferredSequenceCompletionScores {
        scores,
        group_sizes,
    })
}

/// Score candidates with the language model as a normalized prior and the sequence head as a
/// learned residual energy. Both terms are produced from one prompt-prefix/candidate forward.
fn enqueue_sequence_residual_energy_scores_batch<B>(
    model: &DragonModel<B>,
    prompt_tokens: &[Vec<i64>],
    candidate_tokens: &[Vec<Vec<i64>>],
    device: &B::Device,
) -> Result<DeferredSequenceCompletionScores<B>>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    let (scores, group_sizes) = sequence_residual_energy_score_tensor_with_prefix_reuse(
        model,
        prompt_tokens,
        candidate_tokens,
        device,
    )?;
    Ok(DeferredSequenceCompletionScores {
        scores,
        group_sizes,
    })
}

fn sequence_energy_score_tensor_with_prefix_reuse<B>(
    model: &DragonModel<B>,
    prompt_tokens: &[Vec<i64>],
    candidate_tokens: &[Vec<Vec<i64>>],
    device: &B::Device,
) -> Result<(Tensor<B, 1>, Vec<usize>)>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    let group_sizes = validate_sequence_completion_inputs(prompt_tokens, candidate_tokens)?;
    if !model.sequence_score_head_enabled() {
        return Err(anyhow!(
            "semantic-energy proof-action scoring requires an enabled sequence score head"
        ));
    }
    let score_groups = score_ragged_prompt_prefixes(
        model,
        prompt_tokens,
        device,
        |prompt_groups, prompt_last_hidden, prefix_state| {
            score_candidate_energies_from_prefix(
                model,
                prompt_groups,
                candidate_tokens,
                prompt_last_hidden,
                prefix_state,
                device,
            )
        },
    )?;
    let scores = Tensor::cat(
        score_groups
            .into_iter()
            .map(|group| group.expect("validated energy prompt group must be scored"))
            .collect(),
        0,
    );
    Ok((scores, group_sizes))
}

fn sequence_residual_energy_score_tensor_with_prefix_reuse<B>(
    model: &DragonModel<B>,
    prompt_tokens: &[Vec<i64>],
    candidate_tokens: &[Vec<Vec<i64>>],
    device: &B::Device,
) -> Result<(Tensor<B, 1>, Vec<usize>)>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    let group_sizes = validate_sequence_completion_inputs(prompt_tokens, candidate_tokens)?;
    if !model.sequence_score_head_enabled() {
        return Err(anyhow!(
            "residual-energy proof-action scoring requires an enabled sequence score head"
        ));
    }
    let score_groups = score_ragged_prompt_prefixes(
        model,
        prompt_tokens,
        device,
        |prompt_groups, prompt_last_hidden, prefix_state| {
            score_candidate_residual_energies_from_prefix(
                model,
                prompt_groups,
                candidate_tokens,
                prompt_last_hidden,
                prefix_state,
                device,
            )
        },
    )?;
    let scores = Tensor::cat(
        score_groups
            .into_iter()
            .map(|group| group.expect("validated residual-energy prompt group must be scored"))
            .collect(),
        0,
    );
    Ok((scores, group_sizes))
}

pub(crate) fn sequence_energy_score_tensor<B>(
    model: &DragonModel<B>,
    prompt_tokens: &[Vec<i64>],
    candidate_tokens: &[Vec<Vec<i64>>],
    device: &B::Device,
) -> Result<(Tensor<B, 1>, Vec<usize>)>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    let group_sizes = validate_sequence_completion_inputs(prompt_tokens, candidate_tokens)?;
    let scores = sequence_energy_score_tensor_dense(
        model,
        prompt_tokens,
        candidate_tokens,
        false,
        device,
        |inputs| model.forward_hidden(inputs),
    )?;
    Ok((scores, group_sizes))
}

pub(crate) fn sequence_energy_score_tensor_with_gradient_scope<B>(
    model: &DragonModel<B>,
    prompt_tokens: &[Vec<i64>],
    candidate_tokens: &[Vec<Vec<i64>>],
    gradient_scope: crate::config::RuliadProofPolicyGradientScope,
    device: &B::Device,
) -> Result<(Tensor<B, 1>, Vec<usize>)>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let group_sizes = validate_sequence_completion_inputs(prompt_tokens, candidate_tokens)?;
    let scores = match gradient_scope {
        crate::config::RuliadProofPolicyGradientScope::FullModel => {
            sequence_energy_score_tensor_dense(
                model,
                prompt_tokens,
                candidate_tokens,
                false,
                device,
                |inputs| model.forward_hidden(inputs),
            )?
        }
        crate::config::RuliadProofPolicyGradientScope::ScoreHeadOnly => {
            sequence_energy_score_tensor_dense(
                model,
                prompt_tokens,
                candidate_tokens,
                true,
                device,
                |inputs| model.forward_hidden_deterministic_auxiliary(inputs),
            )?
        }
        crate::config::RuliadProofPolicyGradientScope::LanguageHeadOnly => {
            return Err(anyhow!(
                "language_head_only gradient scope is unavailable for semantic-energy scoring"
            ));
        }
    };
    Ok((scores, group_sizes))
}

/// Differentiable residual-EBM candidate scores.
///
/// The autoregressive term is a fixed prior under `ScoreHeadOnly`; only the residual sequence head
/// receives policy gradients. `FullModel` keeps both paths differentiable for a controlled global
/// backpropagation ablation.
pub(crate) fn sequence_residual_energy_score_tensor_with_gradient_scope<B>(
    model: &DragonModel<B>,
    prompt_tokens: &[Vec<i64>],
    candidate_tokens: &[Vec<Vec<i64>>],
    gradient_scope: crate::config::RuliadProofPolicyGradientScope,
    device: &B::Device,
) -> Result<(Tensor<B, 1>, Vec<usize>)>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let group_sizes = validate_sequence_completion_inputs(prompt_tokens, candidate_tokens)?;
    if !model.sequence_score_head_enabled() {
        return Err(anyhow!(
            "residual-energy proof-action scoring requires an enabled sequence score head"
        ));
    }
    let scores = match gradient_scope {
        crate::config::RuliadProofPolicyGradientScope::FullModel => {
            sequence_residual_energy_score_tensor_dense(
                model,
                prompt_tokens,
                candidate_tokens,
                false,
                device,
                |inputs| model.forward_hidden(inputs),
            )?
        }
        crate::config::RuliadProofPolicyGradientScope::ScoreHeadOnly => {
            sequence_residual_energy_score_tensor_dense(
                model,
                prompt_tokens,
                candidate_tokens,
                true,
                device,
                |inputs| model.forward_hidden_deterministic_auxiliary(inputs),
            )?
        }
        crate::config::RuliadProofPolicyGradientScope::LanguageHeadOnly => {
            return Err(anyhow!(
                "language_head_only gradient scope is unavailable for residual-energy scoring"
            ));
        }
    };
    Ok((scores, group_sizes))
}

fn sequence_residual_energy_score_tensor_dense<B, F>(
    score_model: &DragonModel<B>,
    prompt_tokens: &[Vec<i64>],
    candidate_tokens: &[Vec<Vec<i64>>],
    detach_base: bool,
    device: &B::Device,
    forward_hidden: F,
) -> Result<Tensor<B, 1>>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
    F: FnOnce(Tensor<B, 2, Int>) -> Tensor<B, 3>,
{
    let group_sizes = validate_sequence_completion_inputs(prompt_tokens, candidate_tokens)?;
    let row_count = group_sizes.iter().sum::<usize>();
    let maximum_len = prompt_tokens
        .iter()
        .zip(candidate_tokens)
        .flat_map(|(prompt, candidates)| {
            candidates
                .iter()
                .map(move |candidate| prompt.len().saturating_add(candidate.len()))
        })
        .max()
        .unwrap_or_default();
    if maximum_len == 0 {
        return Err(anyhow!(
            "residual-energy sequence scorer has no causal input"
        ));
    }

    let mut input_values = vec![0i64; row_count.saturating_mul(maximum_len)];
    let mut target_values = vec![0i64; row_count.saturating_mul(maximum_len)];
    let mut mask_values = vec![0.0f32; row_count.saturating_mul(maximum_len)];
    let mut prompt_positions = Vec::with_capacity(row_count);
    let mut terminal_positions = Vec::with_capacity(row_count);
    let mut lengths = Vec::with_capacity(row_count);
    let mut row = 0usize;
    for (prompt, candidates) in prompt_tokens.iter().zip(candidate_tokens) {
        for candidate in candidates {
            let row_offset = row.saturating_mul(maximum_len);
            let sequence_len = prompt.len().saturating_add(candidate.len());
            input_values[row_offset..row_offset + prompt.len()].copy_from_slice(prompt);
            input_values[row_offset + prompt.len()..row_offset + sequence_len]
                .copy_from_slice(candidate);
            for (candidate_index, target) in candidate.iter().copied().enumerate() {
                let position = prompt
                    .len()
                    .saturating_sub(1)
                    .saturating_add(candidate_index);
                target_values[row_offset + position] = target;
                mask_values[row_offset + position] = 1.0;
            }
            prompt_positions.push(prompt.len().saturating_sub(1));
            terminal_positions.push(sequence_len.saturating_sub(1));
            lengths.push(candidate.len() as f32);
            row = row.saturating_add(1);
        }
    }

    let inputs = Tensor::<B, 2, Int>::from_data(
        TensorData::new(input_values, [row_count, maximum_len]),
        device,
    );
    let targets = Tensor::<B, 2, Int>::from_data(
        TensorData::new(target_values, [row_count, maximum_len]),
        device,
    );
    let mask = Tensor::<B, 2>::from_data(
        TensorData::new(mask_values, [row_count, maximum_len]),
        device,
    );
    let hidden = forward_hidden(inputs);
    let hidden_base = if detach_base {
        hidden.clone().detach()
    } else {
        hidden.clone()
    };
    let logits = score_model.logits_from_hidden(hidden_base);
    let logits = if detach_base { logits.detach() } else { logits };
    let selected = burn_dragon_core::objective::selected_token_log_probs(
        burn_dragon_core::objective::log_probs_from_logits(logits),
        targets,
    );
    let lengths = Tensor::<B, 1>::from_data(TensorData::new(lengths, [row_count]), device);
    let mean_log_scores = (selected * mask).sum_dim(1).reshape([row_count]) / lengths;

    let hidden = if detach_base { hidden.detach() } else { hidden };
    let [_, _, hidden_size] = hidden.shape().dims::<3>();
    let prompt_gather = Tensor::<B, 3, Int>::from_data(
        TensorData::new(
            prompt_positions
                .into_iter()
                .flat_map(|position| std::iter::repeat_n(position as i64, hidden_size))
                .collect::<Vec<_>>(),
            [row_count, 1, hidden_size],
        ),
        device,
    );
    let terminal_gather = Tensor::<B, 3, Int>::from_data(
        TensorData::new(
            terminal_positions
                .into_iter()
                .flat_map(|position| std::iter::repeat_n(position as i64, hidden_size))
                .collect::<Vec<_>>(),
            [row_count, 1, hidden_size],
        ),
        device,
    );
    let prompt_hidden = hidden.clone().gather(1, prompt_gather);
    let terminal_hidden = hidden.gather(1, terminal_gather);
    let residual = score_model
        .sequence_scores_from_hidden_pair(prompt_hidden, terminal_hidden)
        .map(|scores| scores.reshape([row_count]))
        .ok_or_else(|| anyhow!("residual-energy sequence score head is unavailable"))?;
    Ok(mean_log_scores + residual)
}

/// Keep every candidate row in one dense differentiable training launch.
///
/// Autodiff policy batches are small compared with the language batch. Reusing ragged prefix states
/// here reduces arithmetic but fragments each optimizer update into many short GPU launches. Dense
/// replication has the better training duty cycle and preserves the exact gradient from every
/// candidate score through its prompt.
fn sequence_energy_score_tensor_dense<B, F>(
    score_model: &DragonModel<B>,
    prompt_tokens: &[Vec<i64>],
    candidate_tokens: &[Vec<Vec<i64>>],
    detach_hidden: bool,
    device: &B::Device,
    forward_hidden: F,
) -> Result<Tensor<B, 1>>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
    F: FnOnce(Tensor<B, 2, Int>) -> Tensor<B, 3>,
{
    let group_sizes = validate_sequence_completion_inputs(prompt_tokens, candidate_tokens)?;
    let row_count = group_sizes.iter().sum::<usize>();
    let maximum_len = prompt_tokens
        .iter()
        .zip(candidate_tokens)
        .flat_map(|(prompt, candidates)| {
            candidates
                .iter()
                .map(move |candidate| prompt.len().saturating_add(candidate.len()))
        })
        .max()
        .unwrap_or_default();
    let mut values = vec![0i64; row_count.saturating_mul(maximum_len)];
    let mut prompt_positions = Vec::with_capacity(row_count);
    let mut terminal_positions = Vec::with_capacity(row_count);
    for (row, (prompt, candidate)) in prompt_tokens
        .iter()
        .zip(candidate_tokens)
        .flat_map(|(prompt, candidates)| {
            candidates.iter().map(move |candidate| (prompt, candidate))
        })
        .enumerate()
    {
        let row_offset = row * maximum_len;
        let length = prompt.len().saturating_add(candidate.len());
        values[row_offset..row_offset + prompt.len()].copy_from_slice(prompt);
        values[row_offset + prompt.len()..row_offset + length].copy_from_slice(candidate);
        prompt_positions.push(prompt.len().saturating_sub(1));
        terminal_positions.push(length.saturating_sub(1));
    }
    let inputs =
        Tensor::<B, 2, Int>::from_data(TensorData::new(values, [row_count, maximum_len]), device);
    let hidden = forward_hidden(inputs);
    let [_, _, hidden_size] = hidden.shape().dims::<3>();
    let prompt_gather_values = prompt_positions
        .into_iter()
        .flat_map(|position| std::iter::repeat_n(position as i64, hidden_size))
        .collect::<Vec<_>>();
    let prompt_gather_indices = Tensor::<B, 3, Int>::from_data(
        TensorData::new(prompt_gather_values, [row_count, 1, hidden_size]),
        device,
    );
    let gather_values = terminal_positions
        .into_iter()
        .flat_map(|position| std::iter::repeat_n(position as i64, hidden_size))
        .collect::<Vec<_>>();
    let gather_indices = Tensor::<B, 3, Int>::from_data(
        TensorData::new(gather_values, [row_count, 1, hidden_size]),
        device,
    );
    let prompt_hidden = hidden.clone().gather(1, prompt_gather_indices);
    let terminal_hidden = hidden.gather(1, gather_indices);
    let (prompt_hidden, terminal_hidden) = if detach_hidden {
        (prompt_hidden.detach(), terminal_hidden.detach())
    } else {
        (prompt_hidden, terminal_hidden)
    };
    score_model
        .sequence_scores_from_hidden_pair(prompt_hidden, terminal_hidden)
        .map(|scores| scores.reshape([row_count]))
        .ok_or_else(|| anyhow!("semantic-energy sequence score head is unavailable"))
}

/// Score semantic continuations while encoding each shared prompt exactly once.
///
/// Ragged prompts advance together until the shortest active row ends. Its exact recurrent state
/// is removed, the remaining rows advance to the next boundary, and so on. This avoids padding
/// state corruption and the one-small-forward-per-prompt-length behavior that otherwise starves
/// accelerators during proof-policy search.
#[allow(clippy::single_range_in_vec_init)] // Burn's 1-D slice API requires one range per dimension.
fn sequence_completion_score_tensor_with_prefix_reuse<B>(
    model: &DragonModel<B>,
    prompt_tokens: &[Vec<i64>],
    candidate_tokens: &[Vec<Vec<i64>>],
    device: &B::Device,
) -> Result<SequenceCompletionScoreTensor<B>>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    let group_sizes = validate_sequence_completion_inputs(prompt_tokens, candidate_tokens)?;
    let score_groups = score_ragged_prompt_prefixes(
        model,
        prompt_tokens,
        device,
        |prompt_groups, prompt_last_hidden, prefix_state| {
            score_candidate_continuations_from_prefix(
                model,
                prompt_groups,
                candidate_tokens,
                prompt_last_hidden,
                prefix_state,
                device,
            )
            .map(|groups| {
                groups
                    .into_iter()
                    .map(|(group_index, mean_scores, sum_scores)| {
                        (group_index, (mean_scores, sum_scores))
                    })
                    .collect()
            })
        },
    )?;
    let (mean_groups, sum_groups): (Vec<_>, Vec<_>) = score_groups
        .into_iter()
        .map(|group| group.expect("validated completion prompt group must be scored"))
        .unzip();

    Ok(SequenceCompletionScoreTensor {
        mean_log_scores: Tensor::cat(mean_groups, 0),
        sum_log_scores: Tensor::cat(sum_groups, 0),
        group_sizes,
    })
}

type PromptPrefixScoreGroup<T> = (usize, T);

/// Advance ragged prompt rows in length order and hand exact terminal states to a scorer.
#[allow(clippy::single_range_in_vec_init)] // Burn's 1-D slice API requires one range per dimension.
fn score_ragged_prompt_prefixes<B, T, F>(
    model: &DragonModel<B>,
    prompt_tokens: &[Vec<i64>],
    device: &B::Device,
    mut score_completed: F,
) -> Result<Vec<Option<T>>>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
    F: FnMut(
        &[usize],
        Tensor<B, 3>,
        burn_dragon_core::ModelState<B>,
    ) -> Result<Vec<PromptPrefixScoreGroup<T>>>,
{
    let mut score_groups = (0..prompt_tokens.len()).map(|_| None).collect::<Vec<_>>();
    let mut active_groups = (0..prompt_tokens.len()).collect::<Vec<_>>();
    active_groups.sort_by_key(|group_index| (prompt_tokens[*group_index].len(), *group_index));
    let mut processed_tokens = 0usize;
    let mut prefix_state = model.init_state();
    while !active_groups.is_empty() {
        let next_prompt_len = prompt_tokens[active_groups[0]].len();
        let segment_len = next_prompt_len.saturating_sub(processed_tokens);
        if segment_len == 0 {
            return Err(anyhow!(
                "ragged proof-action prompt scheduler made no forward progress"
            ));
        }
        let active_rows = active_groups.len();
        let prompt_values = active_groups
            .iter()
            .flat_map(|group_index| {
                prompt_tokens[*group_index][processed_tokens..next_prompt_len]
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        let prompt_input = Tensor::<B, 2, Int>::from_data(
            TensorData::new(prompt_values, [active_rows, segment_len]),
            device,
        );
        let prompt_hidden = model.forward_hidden_with_state(prompt_input, &mut prefix_state);
        let [_, _, hidden_size] = prompt_hidden.shape().dims::<3>();
        let completed_rows = active_groups
            .iter()
            .take_while(|group_index| prompt_tokens[**group_index].len() == next_prompt_len)
            .count();
        let completed_groups = &active_groups[..completed_rows];
        let completed_last_hidden = prompt_hidden.slice([
            0..completed_rows,
            segment_len.saturating_sub(1)..segment_len,
            0..hidden_size,
        ]);
        let completed_state = if completed_rows == active_rows {
            prefix_state.clone()
        } else {
            let completed_indices = Tensor::<B, 1, Int>::from_data(
                TensorData::new(
                    (0..completed_rows).map(|row| row as i64).collect(),
                    [completed_rows],
                ),
                device,
            );
            prefix_state.select_batch(completed_indices)
        };
        for (group_index, scores) in
            score_completed(completed_groups, completed_last_hidden, completed_state)?
        {
            let slot = score_groups.get_mut(group_index).ok_or_else(|| {
                anyhow!("ragged proof-action scorer returned an invalid prompt group")
            })?;
            if slot.is_some() {
                return Err(anyhow!(
                    "ragged proof-action scorer returned a prompt group more than once"
                ));
            }
            *slot = Some(scores);
        }

        if completed_rows < active_rows {
            let remaining_indices = Tensor::<B, 1, Int>::from_data(
                TensorData::new(
                    (completed_rows..active_rows)
                        .map(|row| row as i64)
                        .collect(),
                    [active_rows - completed_rows],
                ),
                device,
            );
            prefix_state = prefix_state.select_batch(remaining_indices);
        }
        active_groups.drain(..completed_rows);
        processed_tokens = next_prompt_len;
    }
    if score_groups.iter().any(Option::is_none) {
        return Err(anyhow!(
            "ragged proof-action scorer did not return every prompt group"
        ));
    }
    Ok(score_groups)
}

#[allow(clippy::single_range_in_vec_init)] // Burn's 1-D slice API requires one range per dimension.
fn score_candidate_energies_from_prefix<B>(
    model: &DragonModel<B>,
    prompt_groups: &[usize],
    candidate_tokens: &[Vec<Vec<i64>>],
    prompt_last_hidden: Tensor<B, 3>,
    prefix_state: burn_dragon_core::ModelState<B>,
    device: &B::Device,
) -> Result<Vec<PromptPrefixScoreGroup<Tensor<B, 1>>>>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    let row_count = prompt_groups
        .iter()
        .map(|group_index| candidate_tokens[*group_index].len())
        .sum::<usize>();
    let maximum_len = prompt_groups
        .iter()
        .flat_map(|group_index| candidate_tokens[*group_index].iter())
        .map(Vec::len)
        .max()
        .unwrap_or_default();
    if row_count == 0 || maximum_len == 0 {
        return Err(anyhow!(
            "semantic-energy sequence scorer has no continuation"
        ));
    }
    let mut values = vec![0i64; row_count.saturating_mul(maximum_len)];
    let mut terminal_positions = Vec::with_capacity(row_count);
    for (row, candidate) in prompt_groups
        .iter()
        .flat_map(|group_index| candidate_tokens[*group_index].iter())
        .enumerate()
    {
        let row_offset = row * maximum_len;
        values[row_offset..row_offset + candidate.len()].copy_from_slice(candidate);
        terminal_positions.push(candidate.len().saturating_sub(1));
    }
    let inputs =
        Tensor::<B, 2, Int>::from_data(TensorData::new(values, [row_count, maximum_len]), device);
    let parent_rows = prompt_groups
        .iter()
        .enumerate()
        .flat_map(|(prompt_row, group_index)| {
            std::iter::repeat_n(prompt_row as i64, candidate_tokens[*group_index].len())
        })
        .collect::<Vec<_>>();
    let parent_rows =
        Tensor::<B, 1, Int>::from_data(TensorData::new(parent_rows, [row_count]), device);
    let repeated_prompt_hidden = prompt_last_hidden.select(0, parent_rows.clone());
    let mut candidate_state = prefix_state.select_batch(parent_rows);
    let hidden = model.forward_hidden_with_state(inputs, &mut candidate_state);
    let [_, _, hidden_size] = hidden.shape().dims::<3>();
    let gather_values = terminal_positions
        .into_iter()
        .flat_map(|position| std::iter::repeat_n(position as i64, hidden_size))
        .collect::<Vec<_>>();
    let gather_indices = Tensor::<B, 3, Int>::from_data(
        TensorData::new(gather_values, [row_count, 1, hidden_size]),
        device,
    );
    let terminal_hidden = hidden.gather(1, gather_indices);
    let scores = model
        .sequence_scores_from_hidden_pair(repeated_prompt_hidden, terminal_hidden)
        .ok_or_else(|| anyhow!("semantic-energy sequence score head is unavailable"))?
        .reshape([row_count]);
    let mut row_offset = 0usize;
    Ok(prompt_groups
        .iter()
        .map(|group_index| {
            let group_size = candidate_tokens[*group_index].len();
            let end = row_offset.saturating_add(group_size);
            let group = (*group_index, scores.clone().slice([row_offset..end]));
            row_offset = end;
            group
        })
        .collect())
}

/// Evaluate the autoregressive prior and semantic residual from one continuation pass.
#[allow(clippy::single_range_in_vec_init)] // Burn's 1-D slice API requires one range per dimension.
fn score_candidate_residual_energies_from_prefix<B>(
    model: &DragonModel<B>,
    prompt_groups: &[usize],
    candidate_tokens: &[Vec<Vec<i64>>],
    prompt_last_hidden: Tensor<B, 3>,
    prefix_state: burn_dragon_core::ModelState<B>,
    device: &B::Device,
) -> Result<Vec<PromptPrefixScoreGroup<Tensor<B, 1>>>>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    let row_count = prompt_groups
        .iter()
        .map(|group_index| candidate_tokens[*group_index].len())
        .sum::<usize>();
    let maximum_len = prompt_groups
        .iter()
        .flat_map(|group_index| candidate_tokens[*group_index].iter())
        .map(Vec::len)
        .max()
        .unwrap_or_default();
    if row_count == 0 || maximum_len == 0 {
        return Err(anyhow!(
            "residual-energy sequence scorer has no continuation"
        ));
    }

    let parent_rows = prompt_groups
        .iter()
        .enumerate()
        .flat_map(|(prompt_row, group_index)| {
            std::iter::repeat_n(prompt_row as i64, candidate_tokens[*group_index].len())
        })
        .collect::<Vec<_>>();
    let parent_rows =
        Tensor::<B, 1, Int>::from_data(TensorData::new(parent_rows, [row_count]), device);
    let repeated_prompt_hidden = prompt_last_hidden.select(0, parent_rows.clone());
    let first_targets = Tensor::<B, 2, Int>::from_data(
        TensorData::new(
            prompt_groups
                .iter()
                .flat_map(|group_index| {
                    candidate_tokens[*group_index]
                        .iter()
                        .map(|candidate| candidate[0])
                })
                .collect::<Vec<_>>(),
            [row_count, 1],
        ),
        device,
    );
    let first_log_probs = burn_dragon_core::objective::log_probs_from_logits(
        model.logits_from_hidden(repeated_prompt_hidden.clone()),
    );
    let mut sum_log_scores =
        burn_dragon_core::objective::selected_token_log_probs(first_log_probs, first_targets)
            .reshape([row_count]);

    let mut input_values = vec![0i64; row_count.saturating_mul(maximum_len)];
    let mut tail_targets = vec![0i64; row_count.saturating_mul(maximum_len)];
    let mut tail_masks = vec![0.0f32; row_count.saturating_mul(maximum_len)];
    let mut terminal_positions = Vec::with_capacity(row_count);
    let mut lengths = Vec::with_capacity(row_count);
    for (row, candidate) in prompt_groups
        .iter()
        .flat_map(|group_index| candidate_tokens[*group_index].iter())
        .enumerate()
    {
        let row_offset = row.saturating_mul(maximum_len);
        input_values[row_offset..row_offset + candidate.len()].copy_from_slice(candidate);
        for (position, target) in candidate.iter().copied().enumerate().skip(1) {
            tail_targets[row_offset + position - 1] = target;
            tail_masks[row_offset + position - 1] = 1.0;
        }
        terminal_positions.push(candidate.len().saturating_sub(1));
        lengths.push(candidate.len() as f32);
    }
    let inputs = Tensor::<B, 2, Int>::from_data(
        TensorData::new(input_values, [row_count, maximum_len]),
        device,
    );
    let targets = Tensor::<B, 2, Int>::from_data(
        TensorData::new(tail_targets, [row_count, maximum_len]),
        device,
    );
    let mask = Tensor::<B, 2>::from_data(
        TensorData::new(tail_masks, [row_count, maximum_len]),
        device,
    );
    let mut candidate_state = prefix_state.select_batch(parent_rows);
    let hidden = model.forward_hidden_with_state(inputs, &mut candidate_state);
    let tail_log_probs = burn_dragon_core::objective::log_probs_from_logits(
        model.logits_from_hidden(hidden.clone()),
    );
    let tail_selected =
        burn_dragon_core::objective::selected_token_log_probs(tail_log_probs, targets);
    sum_log_scores = sum_log_scores + (tail_selected * mask).sum_dim(1).reshape([row_count]);
    let lengths = Tensor::<B, 1>::from_data(TensorData::new(lengths, [row_count]), device);
    let mean_log_scores = sum_log_scores / lengths;

    let [_, _, hidden_size] = hidden.shape().dims::<3>();
    let gather_indices = Tensor::<B, 3, Int>::from_data(
        TensorData::new(
            terminal_positions
                .into_iter()
                .flat_map(|position| std::iter::repeat_n(position as i64, hidden_size))
                .collect::<Vec<_>>(),
            [row_count, 1, hidden_size],
        ),
        device,
    );
    let terminal_hidden = hidden.gather(1, gather_indices);
    let residual = model
        .sequence_scores_from_hidden_pair(repeated_prompt_hidden, terminal_hidden)
        .map(|scores| scores.reshape([row_count]))
        .ok_or_else(|| anyhow!("residual-energy sequence score head is unavailable"))?;
    let scores = mean_log_scores + residual;

    let mut row_offset = 0usize;
    Ok(prompt_groups
        .iter()
        .map(|group_index| {
            let group_size = candidate_tokens[*group_index].len();
            let end = row_offset.saturating_add(group_size);
            let group = (*group_index, scores.clone().slice([row_offset..end]));
            row_offset = end;
            group
        })
        .collect())
}

#[allow(clippy::single_range_in_vec_init)] // Burn's 1-D slice API requires one range per dimension.
fn score_candidate_continuations_from_prefix<B>(
    model: &DragonModel<B>,
    prompt_groups: &[usize],
    candidate_tokens: &[Vec<Vec<i64>>],
    prompt_last_hidden: Tensor<B, 3>,
    prefix_state: burn_dragon_core::ModelState<B>,
    device: &B::Device,
) -> Result<Vec<CandidateContinuationScoreGroup<B>>>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    let row_count = prompt_groups
        .iter()
        .map(|group_index| candidate_tokens[*group_index].len())
        .sum::<usize>();
    let parent_rows = prompt_groups
        .iter()
        .enumerate()
        .flat_map(|(prompt_row, group_index)| {
            std::iter::repeat_n(prompt_row as i64, candidate_tokens[*group_index].len())
        })
        .collect::<Vec<_>>();
    let parent_rows =
        Tensor::<B, 1, Int>::from_data(TensorData::new(parent_rows, [row_count]), device);
    let first_logits = model
        .logits_from_hidden(prompt_last_hidden)
        .select(0, parent_rows.clone());
    let first_targets = Tensor::<B, 2, Int>::from_data(
        TensorData::new(
            prompt_groups
                .iter()
                .flat_map(|group_index| {
                    candidate_tokens[*group_index]
                        .iter()
                        .map(|candidate| candidate[0])
                })
                .collect::<Vec<_>>(),
            [row_count, 1],
        ),
        device,
    );
    let first_log_probs = burn_dragon_core::objective::log_probs_from_logits(first_logits);
    let mut sum_log_scores =
        burn_dragon_core::objective::selected_token_log_probs(first_log_probs, first_targets)
            .reshape([row_count]);

    let maximum_tail_len = prompt_groups
        .iter()
        .flat_map(|group_index| candidate_tokens[*group_index].iter())
        .map(|candidate| candidate.len().saturating_sub(1))
        .max()
        .unwrap_or_default();
    if maximum_tail_len > 0 {
        let mut inputs = vec![0i64; row_count.saturating_mul(maximum_tail_len)];
        let mut targets = vec![0i64; row_count.saturating_mul(maximum_tail_len)];
        let mut masks = vec![0.0f32; row_count.saturating_mul(maximum_tail_len)];
        for (row, candidate) in prompt_groups
            .iter()
            .flat_map(|group_index| candidate_tokens[*group_index].iter())
            .enumerate()
        {
            let tail_len = candidate.len().saturating_sub(1);
            let row_offset = row * maximum_tail_len;
            if tail_len > 0 {
                inputs[row_offset..row_offset + tail_len].copy_from_slice(&candidate[..tail_len]);
                targets[row_offset..row_offset + tail_len].copy_from_slice(&candidate[1..]);
                masks[row_offset..row_offset + tail_len].fill(1.0);
            }
        }
        let inputs = Tensor::<B, 2, Int>::from_data(
            TensorData::new(inputs, [row_count, maximum_tail_len]),
            device,
        );
        let targets = Tensor::<B, 2, Int>::from_data(
            TensorData::new(targets, [row_count, maximum_tail_len]),
            device,
        );
        let mask = Tensor::<B, 2>::from_data(
            TensorData::new(masks, [row_count, maximum_tail_len]),
            device,
        );
        let mut candidate_state = prefix_state.select_batch(parent_rows);
        let continuation_hidden = model.forward_hidden_with_state(inputs, &mut candidate_state);
        let continuation_log_probs = burn_dragon_core::objective::log_probs_from_logits(
            model.logits_from_hidden(continuation_hidden),
        );
        let continuation_selected =
            burn_dragon_core::objective::selected_token_log_probs(continuation_log_probs, targets);
        sum_log_scores = sum_log_scores
            + (continuation_selected * mask)
                .sum_dim(1)
                .reshape([row_count]);
    }

    let lengths = Tensor::<B, 1>::from_data(
        TensorData::new(
            prompt_groups
                .iter()
                .flat_map(|group_index| {
                    candidate_tokens[*group_index]
                        .iter()
                        .map(|candidate| candidate.len() as f32)
                })
                .collect::<Vec<_>>(),
            [row_count],
        ),
        device,
    );
    let mean_log_scores = sum_log_scores.clone().div(lengths);
    let mut row_offset = 0usize;
    Ok(prompt_groups
        .iter()
        .map(|group_index| {
            let group_size = candidate_tokens[*group_index].len();
            let end = row_offset.saturating_add(group_size);
            let scores = (
                *group_index,
                mean_log_scores.clone().slice([row_offset..end]),
                sum_log_scores.clone().slice([row_offset..end]),
            );
            row_offset = end;
            scores
        })
        .collect())
}

fn validate_sequence_completion_inputs(
    prompt_tokens: &[Vec<i64>],
    candidate_tokens: &[Vec<Vec<i64>>],
) -> Result<Vec<usize>> {
    if prompt_tokens.is_empty() || prompt_tokens.len() != candidate_tokens.len() {
        return Err(anyhow!(
            "proof-action sequence scoring requires matching non-empty prompt and candidate groups"
        ));
    }
    let group_sizes = candidate_tokens.iter().map(Vec::len).collect::<Vec<_>>();
    if group_sizes.iter().any(|size| *size < 2)
        || prompt_tokens.iter().any(Vec::is_empty)
        || candidate_tokens
            .iter()
            .flatten()
            .any(|candidate| candidate.is_empty())
    {
        return Err(anyhow!(
            "proof-action sequence scoring requires non-empty prompts and at least two non-empty candidates per group"
        ));
    }
    Ok(group_sizes)
}

pub(crate) fn sequence_completion_score_tensor<B>(
    model: &DragonModel<B>,
    prompt_tokens: &[Vec<i64>],
    candidate_tokens: &[Vec<Vec<i64>>],
    device: &B::Device,
) -> Result<SequenceCompletionScoreTensor<B>>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    sequence_completion_score_tensor_dense(prompt_tokens, candidate_tokens, device, |inputs| {
        model.forward(inputs)
    })
}

pub(crate) fn sequence_completion_score_tensor_with_gradient_scope<B>(
    model: &DragonModel<B>,
    prompt_tokens: &[Vec<i64>],
    candidate_tokens: &[Vec<Vec<i64>>],
    gradient_scope: crate::config::RuliadProofPolicyGradientScope,
    device: &B::Device,
) -> Result<SequenceCompletionScoreTensor<B>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    match gradient_scope {
        crate::config::RuliadProofPolicyGradientScope::FullModel => {
            sequence_completion_score_tensor_dense(
                prompt_tokens,
                candidate_tokens,
                device,
                |inputs| model.forward(inputs),
            )
        }
        crate::config::RuliadProofPolicyGradientScope::LanguageHeadOnly => {
            sequence_completion_score_tensor_dense(
                prompt_tokens,
                candidate_tokens,
                device,
                |inputs| {
                    let hidden = model
                        .forward_hidden_deterministic_auxiliary(inputs)
                        .detach();
                    model.logits_from_hidden(hidden)
                },
            )
        }
        crate::config::RuliadProofPolicyGradientScope::ScoreHeadOnly => Err(anyhow!(
            "score_head_only gradient scope is unavailable for completion-likelihood scoring"
        )),
    }
}

fn sequence_completion_score_tensor_dense<B, F>(
    prompt_tokens: &[Vec<i64>],
    candidate_tokens: &[Vec<Vec<i64>>],
    device: &B::Device,
    forward_logits: F,
) -> Result<SequenceCompletionScoreTensor<B>>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
    F: FnOnce(Tensor<B, 2, Int>) -> Tensor<B, 3>,
{
    let group_sizes = validate_sequence_completion_inputs(prompt_tokens, candidate_tokens)?;

    let row_count = group_sizes.iter().sum::<usize>();
    let maximum_len = prompt_tokens
        .iter()
        .zip(candidate_tokens)
        .flat_map(|(prompt, candidates)| {
            candidates.iter().map(move |candidate| {
                prompt
                    .len()
                    .saturating_add(candidate.len())
                    .saturating_sub(1)
            })
        })
        .max()
        .unwrap_or_default();
    if maximum_len == 0 {
        return Err(anyhow!("proof-action sequence scorer has no causal input"));
    }

    let mut inputs = vec![0i64; row_count.saturating_mul(maximum_len)];
    let mut targets = vec![0i64; row_count.saturating_mul(maximum_len)];
    let mut masks = vec![0.0f32; row_count.saturating_mul(maximum_len)];
    let mut lengths = Vec::with_capacity(row_count);
    let mut row = 0usize;
    for (prompt, candidates) in prompt_tokens.iter().zip(candidate_tokens) {
        for candidate in candidates {
            let row_offset = row * maximum_len;
            let input_len = prompt
                .len()
                .saturating_add(candidate.len())
                .saturating_sub(1);
            inputs[row_offset..row_offset + prompt.len()].copy_from_slice(prompt);
            if candidate.len() > 1 {
                inputs[row_offset + prompt.len()..row_offset + input_len]
                    .copy_from_slice(&candidate[..candidate.len() - 1]);
            }
            for (candidate_index, target) in candidate.iter().copied().enumerate() {
                let position = prompt
                    .len()
                    .saturating_sub(1)
                    .saturating_add(candidate_index);
                targets[row_offset + position] = target;
                masks[row_offset + position] = 1.0;
            }
            lengths.push(candidate.len() as f32);
            row = row.saturating_add(1);
        }
    }

    let inputs =
        Tensor::<B, 2, Int>::from_data(TensorData::new(inputs, [row_count, maximum_len]), device);
    let targets =
        Tensor::<B, 2, Int>::from_data(TensorData::new(targets, [row_count, maximum_len]), device);
    let mask = Tensor::<B, 2>::from_data(TensorData::new(masks, [row_count, maximum_len]), device);
    let lengths = Tensor::<B, 1>::from_data(TensorData::new(lengths, [row_count]), device);
    let log_probs = burn_dragon_core::objective::log_probs_from_logits(forward_logits(inputs));
    let selected = burn_dragon_core::objective::selected_token_log_probs(log_probs, targets);
    let sum_log_scores = (selected * mask).sum_dim(1).reshape([row_count]);
    let mean_log_scores = sum_log_scores.clone().div(lengths);
    Ok(SequenceCompletionScoreTensor {
        mean_log_scores,
        sum_log_scores,
        group_sizes,
    })
}

/// Queue a proof-action scoring forward without synchronizing logits back to the host. Callers can
/// enqueue a bounded number of independent chunks before resolving them to overlap CUDA work and
/// readback while retaining an explicit in-flight memory bound.
pub(crate) fn enqueue_constrained_completion_log_probs_batch<B>(
    model: &DragonModel<B>,
    prompt_tokens: &[Vec<i64>],
    candidate_tokens: &[Vec<Vec<i64>>],
    device: &B::Device,
) -> Result<DeferredConstrainedCompletionScores<B>>
where
    B: Backend + Clone + 'static,
    B::Device: Clone,
{
    if prompt_tokens.is_empty() || prompt_tokens.len() != candidate_tokens.len() {
        return Err(anyhow!(
            "proof-action batch requires matching non-empty prompts and candidate groups"
        ));
    }
    let mut sequences = Vec::with_capacity(prompt_tokens.len());
    let mut branch_tokens = Vec::with_capacity(prompt_tokens.len());
    for (prompt, candidates) in prompt_tokens.iter().zip(candidate_tokens) {
        if prompt.is_empty() {
            return Err(anyhow!("proof-action prompt is empty"));
        }
        let (prefix_len, tokens) = candidate_branch_tokens(candidates)?;
        let mut sequence = prompt.clone();
        sequence.extend(candidates[0].iter().copied().take(prefix_len));
        sequences.push(sequence);
        branch_tokens.push(tokens);
    }
    let maximum_len = sequences.iter().map(Vec::len).max().unwrap_or_default();
    if maximum_len == 0 {
        return Err(anyhow!("proof-action scorer has no input tokens"));
    }
    let batch_size = sequences.len();
    let mut values = vec![0i64; batch_size.saturating_mul(maximum_len)];
    let mut branch_positions = Vec::with_capacity(batch_size);
    for (row, sequence) in sequences.into_iter().enumerate() {
        let length = sequence.len();
        values[row * maximum_len..row * maximum_len + length].copy_from_slice(&sequence);
        branch_positions.push(length.saturating_sub(1));
    }
    let inputs =
        Tensor::<B, 2, Int>::from_data(TensorData::new(values, [batch_size, maximum_len]), device);
    let logits = logits_at_sequence_positions(model, inputs, &branch_positions, device)?;
    let [_, vocab] = logits.shape().dims::<2>();
    if vocab == 0 {
        return Err(anyhow!("proof-action scorer produced an empty vocabulary"));
    }
    Ok(DeferredConstrainedCompletionScores {
        logits,
        branch_tokens,
        vocab,
    })
}

fn candidate_branch_tokens(candidate_tokens: &[Vec<i64>]) -> Result<(usize, Vec<i64>)> {
    if candidate_tokens.len() < 2 || candidate_tokens.iter().any(Vec::is_empty) {
        return Err(anyhow!(
            "proof-action scoring requires at least two non-empty candidates"
        ));
    }
    let prefix_len = common_prefix_len(candidate_tokens);
    if candidate_tokens
        .iter()
        .any(|candidate| prefix_len >= candidate.len())
    {
        return Err(anyhow!(
            "proof-action candidates must differ before one candidate terminates"
        ));
    }
    let mut distinguishing_tokens = std::collections::BTreeSet::new();
    let mut tokens = Vec::with_capacity(candidate_tokens.len());
    for candidate in candidate_tokens {
        let token = candidate[prefix_len];
        if !distinguishing_tokens.insert(token) {
            return Err(anyhow!(
                "proof-action tokenizer does not distinguish candidates at the first branch token"
            ));
        }
        tokens.push(token);
    }
    Ok((prefix_len, tokens))
}

fn normalize_candidate_scores(values: &[f32], tokens: &[i64]) -> Result<Vec<f32>> {
    let mut scores = Vec::with_capacity(tokens.len());
    for token in tokens {
        let index = usize::try_from(*token)
            .map_err(|_| anyhow!("proof-action token id {token} is negative"))?;
        let score = *values
            .get(index)
            .ok_or_else(|| anyhow!("proof-action token id {token} exceeds model vocabulary"))?;
        if !score.is_finite() {
            return Err(anyhow!("proof-action score is non-finite"));
        }
        scores.push(score);
    }
    normalize_log_scores(&scores)
}

fn normalize_log_scores(scores: &[f32]) -> Result<Vec<f32>> {
    if scores.len() < 2 || scores.iter().any(|score| !score.is_finite()) {
        return Err(anyhow!(
            "proof-action candidate scores require at least two finite values"
        ));
    }
    let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_normalizer = maximum
        + scores
            .iter()
            .map(|score| (*score - maximum).exp())
            .sum::<f32>()
            .ln();
    Ok(scores.iter().map(|score| score - log_normalizer).collect())
}

pub(crate) fn best_candidate_index(log_probs: &[f32]) -> Option<usize> {
    log_probs
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
}

pub(crate) fn candidate_branch_index(candidate_tokens: &[Vec<i64>]) -> Result<usize> {
    candidate_branch_tokens(candidate_tokens).map(|(prefix_len, _)| prefix_len)
}

fn common_prefix_len(candidates: &[Vec<i64>]) -> usize {
    let minimum = candidates.iter().map(Vec::len).min().unwrap_or_default();
    (0..minimum)
        .take_while(|index| {
            let expected = candidates[0][*index];
            candidates
                .iter()
                .skip(1)
                .all(|candidate| candidate[*index] == expected)
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::AutodiffModule;
    use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
    use burn::tensor::activation;
    use burn_autodiff::Autodiff;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;
    type TrainBackend = Autodiff<TestBackend>;

    fn tensor_values<const D: usize>(tensor: Tensor<TestBackend, D>) -> Vec<f32> {
        tensor
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("tensor values")
    }

    fn maximum_abs_difference(left: &[f32], right: &[f32]) -> f32 {
        assert_eq!(left.len(), right.len());
        left.iter()
            .zip(right)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f32::max)
    }

    fn assert_confident_target_conditioning(
        label: &str,
        initial_loss: Option<f32>,
        final_loss: f32,
    ) {
        let initial_loss = initial_loss.expect("initial target-conditioned loss");
        assert!(
            final_loss.is_finite() && final_loss < initial_loss,
            "{label} did not reduce target NLL: initial={initial_loss}, final={final_loss}"
        );
        assert!(
            final_loss < std::f32::consts::LN_2,
            "{label} target NLL does not imply geometric-mean target probability above 50%: final={final_loss}"
        );
    }

    #[test]
    fn common_prefix_stops_at_action_bearing_token() {
        assert_eq!(
            common_prefix_len(&[vec![10, 11, 20, 30], vec![10, 11, 21, 30]]),
            2
        );
    }

    #[test]
    fn best_candidate_ignores_non_finite_scores() {
        assert_eq!(best_candidate_index(&[f32::NAN, -2.0, -0.5]), Some(2));
    }

    #[test]
    fn counterfactual_candidates_are_valid_distinct_and_deterministic() {
        use burn_dragon_universality::ruliad::{
            RuliadProofActionCandidate, RuliadProofActionSet, RuliadProofSource, RuliadProofStep,
            RuliadRewriteDirection, RuliadTerm,
        };

        let step = |id: &str| RuliadProofStep {
            source: RuliadProofSource::Axiom { id: id.to_string() },
            path: Vec::new(),
            direction: RuliadRewriteDirection::Forward,
        };
        let candidate = |id: &str, outcome: Option<RuliadTerm>| RuliadProofActionCandidate {
            step: step(id),
            outcome,
            distance_to_goal: Some(1),
        };
        let actions = RuliadProofActionSet {
            goal: 0,
            current: RuliadTerm::atom("current"),
            target: RuliadTerm::atom("target"),
            candidates: vec![
                candidate("expert", Some(RuliadTerm::atom("target"))),
                candidate("invalid", None),
                candidate("alternate-a", Some(RuliadTerm::atom("a"))),
                candidate("duplicate-a", Some(RuliadTerm::atom("a"))),
                candidate("current", Some(RuliadTerm::atom("current"))),
                candidate("alternate-b", Some(RuliadTerm::atom("b"))),
            ],
            selected_index: 0,
            equivalent_indices: vec![0],
        };

        assert_eq!(counterfactual_candidate_indices(&actions, 3, 1), vec![2, 5]);
        assert_eq!(counterfactual_candidate_indices(&actions, 1, 5), vec![5]);
        assert!(counterfactual_candidate_indices(&actions, 0, 0).is_empty());
    }

    #[test]
    fn candidate_branch_index_identifies_only_competing_token() {
        assert_eq!(
            candidate_branch_index(&[vec![10, 11, 20, 30], vec![10, 11, 21, 30]])
                .expect("branch index"),
            2
        );
    }

    #[test]
    fn presentation_rotations_balance_or_cover_the_exact_cyclic_orbit() {
        use crate::config::RuliadProofPolicyCandidateSymmetry::{
            BalancedRotation, Canonical, CyclicOrbitAverage,
        };

        assert_eq!(
            candidate_presentation_rotations(Canonical, 2, 4, 17).expect("canonical"),
            vec![0]
        );
        assert_eq!(
            candidate_presentation_rotations(BalancedRotation, 2, 4, 3).expect("balanced"),
            vec![3]
        );
        assert_eq!(
            candidate_presentation_rotations(CyclicOrbitAverage, 2, 4, 3).expect("orbit"),
            vec![0, 1, 2, 3]
        );
    }

    #[test]
    fn counterfactual_target_reuses_base_candidate_presentation() {
        use crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation;

        let base = target_group_presentation_rotations(BalancedRotation, 2, 4, 3, None)
            .expect("base rotations");
        assert_eq!(base, vec![3]);

        let paired = target_group_presentation_rotations(BalancedRotation, 0, 4, 4, Some(&base))
            .expect("paired target rotations");
        assert_eq!(paired, base);
        assert_ne!(
            paired,
            candidate_presentation_rotations(BalancedRotation, 0, 4, 4)
                .expect("independently balanced rotations")
        );
        assert!(
            target_group_presentation_rotations(BalancedRotation, 0, 4, 4, Some(&[4])).is_err()
        );
    }

    #[test]
    fn semantic_scores_remove_pure_position_bias_over_the_orbit() {
        let position_biased = (0..4)
            .map(|rotation| (rotation, vec![-0.1, -3.0, -3.0, -3.0]))
            .collect::<Vec<_>>();
        let summary = semantic_action_orbit_summary(&position_biased, 4).expect("orbit scores");
        for score in summary.averaged_log_probs.iter().skip(1) {
            assert!(
                (score - summary.averaged_log_probs[0]).abs() < 1.0e-6,
                "{:?}",
                summary.averaged_log_probs
            );
        }
        assert!(summary.complete_cyclic_orbit);
        assert!(summary.js_divergence > 0.5, "{summary:?}");
        assert!(summary.top1_consensus_fraction <= 0.25, "{summary:?}");
    }

    #[test]
    fn semantic_scores_preserve_a_rotation_equivariant_preference() {
        let semantic = [0.7f32, 0.1, 0.15, 0.05];
        let presentations = (0..semantic.len())
            .map(|rotation| {
                let scores = (0..semantic.len())
                    .map(|presented| {
                        let original = (presented + rotation) % semantic.len();
                        semantic[original].ln()
                    })
                    .collect::<Vec<_>>();
                (rotation, scores)
            })
            .collect::<Vec<_>>();
        let summary =
            semantic_action_orbit_summary(&presentations, semantic.len()).expect("scores");
        assert_eq!(best_candidate_index(&summary.averaged_log_probs), Some(0));
        for (score, expected) in summary.averaged_log_probs.iter().zip(semantic) {
            assert!(
                (score.exp() - expected).abs() < 1.0e-6,
                "{:?}",
                summary.averaged_log_probs
            );
        }
        assert!(summary.complete_cyclic_orbit);
        assert!(summary.js_divergence < 1.0e-6, "{summary:?}");
        assert!((summary.top1_consensus_fraction - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn semantic_summary_marks_an_incomplete_orbit() {
        let summary = semantic_action_orbit_summary(
            &[
                (0, vec![-0.2, -2.0, -2.0, -2.0]),
                (2, vec![-2.0, -2.0, -0.2, -2.0]),
            ],
            4,
        )
        .expect("partial orbit");
        assert!(!summary.complete_cyclic_orbit);
    }

    #[test]
    fn typed_policy_decision_is_batch_bound_invariant_and_renders_semantic_action() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 59);
        let model = DragonModel::<TestBackend>::new(
            burn_dragon_core::DragonConfig {
                n_layer: 1,
                n_embd: 16,
                n_head: 2,
                mlp_internal_dim_multiplier: 2,
                vocab_size: 32,
                dropout: 0.0,
                ..Default::default()
            },
            &device,
        );
        let canonical = vec![vec![10, 11], vec![12, 13, 14], vec![15, 16]];
        let request = EncodedRuliadProofActionRequest {
            answer_contract:
                burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
            presentations: [1, 2, 0]
                .into_iter()
                .map(|rotation| {
                    let mut candidate_tokens = canonical.clone();
                    candidate_tokens.rotate_left(rotation);
                    EncodedRuliadProofActionPresentation {
                        rotation,
                        prompt_tokens: vec![1, 2, 3, 4],
                        candidate_tokens,
                    }
                })
                .collect(),
        };

        let serialized =
            select_ruliad_proof_actions_batch(&model, std::slice::from_ref(&request), 1, &device)
                .expect("serialized typed decision")
                .remove(0);
        let tensorized = select_ruliad_proof_actions_batch(&model, &[request], 64, &device)
            .expect("tensorized typed decision")
            .remove(0);

        assert_eq!(serialized, tensorized);
        assert!(tensorized.orbit.complete_cyclic_orbit);
        assert_eq!(
            tensorized.selected_completion_tokens,
            canonical[tensorized.selected_semantic_index]
        );
    }

    #[test]
    fn typed_policy_rejects_zero_row_bound_and_inconsistent_candidates() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let model = DragonModel::<TestBackend>::new(
            burn_dragon_core::DragonConfig {
                n_layer: 1,
                n_embd: 8,
                n_head: 1,
                mlp_internal_dim_multiplier: 2,
                vocab_size: 16,
                dropout: 0.0,
                ..Default::default()
            },
            &device,
        );
        let request = EncodedRuliadProofActionRequest {
            answer_contract:
                burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
            presentations: vec![
                EncodedRuliadProofActionPresentation {
                    rotation: 0,
                    prompt_tokens: vec![1, 2],
                    candidate_tokens: vec![vec![3], vec![4]],
                },
                EncodedRuliadProofActionPresentation {
                    rotation: 1,
                    prompt_tokens: vec![1, 2],
                    candidate_tokens: vec![vec![4], vec![3], vec![5]],
                },
            ],
        };
        assert!(
            select_ruliad_proof_actions_batch(&model, std::slice::from_ref(&request), 0, &device,)
                .is_err()
        );
        assert!(select_ruliad_proof_actions_batch(&model, &[request], 4, &device).is_err());
    }

    #[test]
    fn semantic_sequence_scorer_handles_shared_branch_tokens_and_candidate_permutations() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 47);
        let model = DragonModel::<TestBackend>::new(
            burn_dragon_core::DragonConfig {
                n_layer: 1,
                n_embd: 16,
                n_head: 2,
                mlp_internal_dim_multiplier: 2,
                vocab_size: 32,
                dropout: 0.0,
                ..Default::default()
            },
            &device,
        );
        let prompts = vec![vec![1, 2, 3]];
        let candidates = vec![vec![vec![10, 11], vec![10, 12, 13], vec![10, 12, 14]]];
        assert!(candidate_branch_tokens(&candidates[0]).is_err());

        let scores = proof_action_completion_log_probs_batch(
            &model,
            &prompts,
            &candidates,
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
            &device,
        )
        .expect("semantic sequence scores")
        .remove(0);
        assert!((scores.iter().map(|score| score.exp()).sum::<f32>() - 1.0).abs() < 1.0e-5);

        let permuted = vec![vec![
            candidates[0][2].clone(),
            candidates[0][0].clone(),
            candidates[0][1].clone(),
        ]];
        let permuted_scores = proof_action_completion_log_probs_batch(
            &model,
            &prompts,
            &permuted,
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
            &device,
        )
        .expect("permuted semantic sequence scores")
        .remove(0);
        assert!((scores[2] - permuted_scores[0]).abs() < 1.0e-5);
        assert!((scores[0] - permuted_scores[1]).abs() < 1.0e-5);
        assert!((scores[1] - permuted_scores[2]).abs() < 1.0e-5);
    }

    #[test]
    fn semantic_energy_scorer_is_normalized_and_candidate_permutation_equivariant() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 71);
        let mut config = burn_dragon_core::DragonConfig {
            n_layer: 1,
            n_embd: 16,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 32,
            dropout: 0.0,
            ..Default::default()
        };
        config.sequence_score_head.enabled = true;
        let model = DragonModel::<TestBackend>::new(config, &device);
        let prompts = vec![vec![1, 2, 3]];
        let candidates = vec![vec![vec![10, 11], vec![12], vec![13, 14, 15]]];
        let scores = proof_action_scores_batch(
            &model,
            &prompts,
            &candidates,
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
            crate::config::RuliadProofPolicyScoring::SemanticEnergy,
            &device,
        )
        .expect("semantic energy scores")
        .remove(0);
        assert!((scores.iter().map(|score| score.exp()).sum::<f32>() - 1.0).abs() < 1.0e-5);

        let permuted = vec![vec![
            candidates[0][2].clone(),
            candidates[0][0].clone(),
            candidates[0][1].clone(),
        ]];
        let permuted_scores = proof_action_scores_batch(
            &model,
            &prompts,
            &permuted,
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
            crate::config::RuliadProofPolicyScoring::SemanticEnergy,
            &device,
        )
        .expect("permuted semantic energy scores")
        .remove(0);
        assert!((scores[2] - permuted_scores[0]).abs() < 1.0e-5);
        assert!((scores[0] - permuted_scores[1]).abs() < 1.0e-5);
        assert!((scores[1] - permuted_scores[2]).abs() < 1.0e-5);
    }

    #[test]
    fn residual_energy_scorer_matches_lm_prior_plus_semantic_energy() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 711);
        let mut config = burn_dragon_core::DragonConfig {
            n_layer: 1,
            n_embd: 16,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 32,
            dropout: 0.0,
            ..Default::default()
        };
        config.sequence_score_head.enabled = true;
        let model = DragonModel::<TestBackend>::new(config, &device);
        let prompts = vec![vec![1, 2, 3]];
        let candidates = vec![vec![vec![10, 11], vec![12], vec![13, 14, 15]]];

        let completion = sequence_completion_score_tensor(&model, &prompts, &candidates, &device)
            .expect("completion prior");
        let (energy, group_sizes) =
            sequence_energy_score_tensor(&model, &prompts, &candidates, &device)
                .expect("semantic residual");
        assert_eq!(completion.group_sizes, group_sizes);
        let expected = normalize_log_scores(&tensor_values(completion.mean_log_scores + energy))
            .expect("normalized residual posterior");
        let actual = proof_action_scores_batch(
            &model,
            &prompts,
            &candidates,
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
            crate::config::RuliadProofPolicyScoring::ResidualEnergy,
            &device,
        )
        .expect("residual-energy scores")
        .remove(0);

        assert_eq!(actual.len(), expected.len());
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (*actual - expected).abs() < 2.0e-4),
            "fused residual scorer must equal the explicit LM-plus-energy contract: {actual:?}"
        );
    }

    #[test]
    fn semantic_energy_pair_head_changes_candidate_margin_with_prompt() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 72);
        let mut config = burn_dragon_core::DragonConfig {
            n_layer: 1,
            n_embd: 8,
            n_head: 1,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 16,
            dropout: 0.0,
            ..Default::default()
        };
        config.sequence_score_head.enabled = true;
        let model = DragonModel::<TestBackend>::new(config, &device);
        let terminal_hidden = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // candidate A
                    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // candidate B
                ],
                [2, 1, 8],
            ),
            &device,
        );
        let prompt_a = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                vec![
                    1.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0,
                ],
                [2, 1, 8],
            ),
            &device,
        );
        let prompt_b = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                vec![
                    0.25, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0,
                ],
                [2, 1, 8],
            ),
            &device,
        );

        let scores_a = tensor_values(
            model
                .sequence_scores_from_hidden_pair(prompt_a, terminal_hidden.clone())
                .expect("enabled pair score head"),
        );
        let scores_b = tensor_values(
            model
                .sequence_scores_from_hidden_pair(prompt_b, terminal_hidden)
                .expect("enabled pair score head"),
        );
        let margin_a = scores_a[0] - scores_a[1];
        let margin_b = scores_b[0] - scores_b[1];
        assert!(
            (margin_a - margin_b).abs() > 1.0e-5,
            "prompt-conditioned margins should differ: a={margin_a}, b={margin_b}"
        );
    }

    #[test]
    fn semantic_energy_prefix_scorer_changes_candidate_margin_with_prompt_tokens() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 74);
        let mut config = burn_dragon_core::DragonConfig {
            n_layer: 2,
            n_embd: 16,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 32,
            dropout: 0.0,
            ..Default::default()
        };
        config.sequence_score_head.enabled = true;
        let model = DragonModel::<TestBackend>::new(config, &device);
        let prompts = vec![vec![1, 2, 3, 4], vec![1, 2, 9, 4]];
        let menu = vec![vec![10, 11], vec![12, 13], vec![14, 15]];
        let candidates = vec![menu.clone(), menu];
        let scores = proof_action_scores_batch(
            &model,
            &prompts,
            &candidates,
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
            crate::config::RuliadProofPolicyScoring::SemanticEnergy,
            &device,
        )
        .expect("prompt-conditioned semantic energy scores");
        let margin_a = scores[0][0] - scores[0][1];
        let margin_b = scores[1][0] - scores[1][1];
        assert!(
            (margin_a - margin_b).abs() > 1.0e-6,
            "token-level prompt change should alter the candidate margin: a={margin_a}, b={margin_b}"
        );
    }

    #[test]
    fn semantic_energy_pair_head_overfits_opposite_targets_for_one_candidate_menu() {
        let device = burn::tensor::Device::<TrainBackend>::default();
        TrainBackend::seed(&device, 75);
        let mut config = burn_dragon_core::DragonConfig {
            n_layer: 1,
            n_embd: 16,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 32,
            dropout: 0.0,
            ..Default::default()
        };
        config.sequence_score_head.enabled = true;
        let mut model = DragonModel::<TrainBackend>::new(config, &device);
        let mut optimizer = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<TrainBackend, DragonModel<TrainBackend>>();
        let prompts = vec![vec![1, 2, 3, 4], vec![1, 2, 9, 4]];
        let menu = vec![vec![10, 11], vec![12, 13], vec![14, 15]];
        let candidates = vec![menu.clone(), menu];
        let targets = Tensor::<TrainBackend, 2, Int>::from_data(
            TensorData::new(vec![0_i64, 1], [2, 1]),
            &device,
        );
        let mut initial_loss = None;
        let mut final_loss = 0.0f32;

        for _ in 0..256 {
            let (scores, group_sizes) =
                sequence_energy_score_tensor(&model, &prompts, &candidates, &device)
                    .expect("paired-target energy scores");
            assert_eq!(group_sizes, vec![3, 3]);
            let loss = activation::log_softmax(scores.reshape([2, 3]), 1)
                .gather(1, targets.clone())
                .mean()
                .neg()
                .reshape([1]);
            let scalar = loss
                .clone()
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("paired-target loss")[0];
            initial_loss.get_or_insert(scalar);
            final_loss = scalar;
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(1.0e-2, model, grads);
        }

        let valid = model.valid();
        let scores = proof_action_scores_batch(
            &valid,
            &prompts,
            &candidates,
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
            crate::config::RuliadProofPolicyScoring::SemanticEnergy,
            &burn::tensor::Device::<TestBackend>::default(),
        )
        .expect("trained paired-target semantic scores");
        assert_eq!(best_candidate_index(&scores[0]), Some(0), "{scores:?}");
        assert_eq!(best_candidate_index(&scores[1]), Some(1), "{scores:?}");
        assert!(
            final_loss < initial_loss.expect("initial loss") * 0.1,
            "paired-target loss did not converge: initial={initial_loss:?}, final={final_loss}"
        );
    }

    #[test]
    fn semantic_energy_score_head_only_update_preserves_language_logits() {
        let device = burn::tensor::Device::<TrainBackend>::default();
        TrainBackend::seed(&device, 751);
        let mut config = burn_dragon_core::DragonConfig {
            n_layer: 1,
            n_embd: 16,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 32,
            dropout: 0.0,
            ..Default::default()
        };
        config.sequence_score_head.enabled = true;
        let mut model = DragonModel::<TrainBackend>::new(config, &device);
        let mut optimizer = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<TrainBackend, DragonModel<TrainBackend>>();
        let prompts = vec![vec![1, 2, 3, 4], vec![1, 2, 9, 4]];
        let menu = vec![vec![10, 11], vec![12, 13], vec![14, 15]];
        let candidates = vec![menu.clone(), menu];
        let targets = Tensor::<TrainBackend, 2, Int>::from_data(
            TensorData::new(vec![0_i64, 1], [2, 1]),
            &device,
        );
        let probe = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4, 5, 6], [2, 3]),
            &burn::tensor::Device::<TestBackend>::default(),
        );
        let before_valid = model.valid();
        let before_logits = tensor_values(before_valid.forward(probe.clone()));
        let before_scores = proof_action_scores_batch(
            &before_valid,
            &prompts,
            &candidates,
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
            crate::config::RuliadProofPolicyScoring::SemanticEnergy,
            &burn::tensor::Device::<TestBackend>::default(),
        )
        .expect("initial semantic scores")
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

        let (scores, group_sizes) = sequence_energy_score_tensor_with_gradient_scope(
            &model,
            &prompts,
            &candidates,
            crate::config::RuliadProofPolicyGradientScope::ScoreHeadOnly,
            &device,
        )
        .expect("head-only semantic scores");
        assert_eq!(group_sizes, vec![3, 3]);
        let loss = activation::log_softmax(scores.reshape([2, 3]), 1)
            .gather(1, targets)
            .mean()
            .neg()
            .reshape([1]);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.1, model, grads);

        let after_valid = model.valid();
        let after_logits = tensor_values(after_valid.forward(probe));
        let after_scores = proof_action_scores_batch(
            &after_valid,
            &prompts,
            &candidates,
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
            crate::config::RuliadProofPolicyScoring::SemanticEnergy,
            &burn::tensor::Device::<TestBackend>::default(),
        )
        .expect("updated semantic scores")
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

        assert_eq!(
            maximum_abs_difference(&before_logits, &after_logits),
            0.0,
            "head-only policy update changed Dragon language logits"
        );
        assert!(
            maximum_abs_difference(&before_scores, &after_scores) > 1.0e-5,
            "head-only policy update did not change semantic scores"
        );
    }

    #[test]
    fn residual_energy_score_head_only_update_preserves_language_logits() {
        let device = burn::tensor::Device::<TrainBackend>::default();
        TrainBackend::seed(&device, 752);
        let mut config = burn_dragon_core::DragonConfig {
            n_layer: 1,
            n_embd: 16,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 32,
            dropout: 0.0,
            ..Default::default()
        };
        config.sequence_score_head.enabled = true;
        let mut model = DragonModel::<TrainBackend>::new(config, &device);
        let mut optimizer = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<TrainBackend, DragonModel<TrainBackend>>();
        let prompts = vec![vec![1, 2, 3, 4], vec![1, 2, 9, 4]];
        let menu = vec![vec![10, 11], vec![12, 13], vec![14, 15]];
        let candidates = vec![menu.clone(), menu];
        let targets = Tensor::<TrainBackend, 2, Int>::from_data(
            TensorData::new(vec![0_i64, 1], [2, 1]),
            &device,
        );
        let probe = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4, 5, 6], [2, 3]),
            &burn::tensor::Device::<TestBackend>::default(),
        );
        let before_valid = model.valid();
        let before_logits = tensor_values(before_valid.forward(probe.clone()));
        let before_scores = proof_action_scores_batch(
            &before_valid,
            &prompts,
            &candidates,
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
            crate::config::RuliadProofPolicyScoring::ResidualEnergy,
            &burn::tensor::Device::<TestBackend>::default(),
        )
        .expect("initial residual scores")
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

        let (scores, group_sizes) = sequence_residual_energy_score_tensor_with_gradient_scope(
            &model,
            &prompts,
            &candidates,
            crate::config::RuliadProofPolicyGradientScope::ScoreHeadOnly,
            &device,
        )
        .expect("head-only residual scores");
        assert_eq!(group_sizes, vec![3, 3]);
        let loss = activation::log_softmax(scores.reshape([2, 3]), 1)
            .gather(1, targets)
            .mean()
            .neg()
            .reshape([1]);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(0.1, model, grads);

        let after_valid = model.valid();
        let after_logits = tensor_values(after_valid.forward(probe));
        let after_scores = proof_action_scores_batch(
            &after_valid,
            &prompts,
            &candidates,
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
            crate::config::RuliadProofPolicyScoring::ResidualEnergy,
            &burn::tensor::Device::<TestBackend>::default(),
        )
        .expect("updated residual scores")
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

        assert_eq!(
            maximum_abs_difference(&before_logits, &after_logits),
            0.0,
            "head-only residual policy update changed Dragon language logits"
        );
        assert!(
            maximum_abs_difference(&before_scores, &after_scores) > 1.0e-5,
            "head-only policy update did not change residual scores"
        );
    }

    #[test]
    fn completion_language_head_only_update_preserves_dragon_hidden_states() {
        let device = burn::tensor::Device::<TrainBackend>::default();
        TrainBackend::seed(&device, 753);
        let config = burn_dragon_core::DragonConfig {
            n_layer: 1,
            n_embd: 16,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 32,
            dropout: 0.0,
            tie_input_output_embeddings: false,
            ..Default::default()
        };
        let mut model = DragonModel::<TrainBackend>::new(config, &device);
        let mut optimizer = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<TrainBackend, DragonModel<TrainBackend>>();
        let prompts = vec![vec![1, 2, 3, 4], vec![1, 2, 9, 4]];
        let menu = vec![vec![10, 11], vec![12, 13], vec![14, 15]];
        let candidates = vec![menu.clone(), menu];
        let targets = Tensor::<TrainBackend, 2, Int>::from_data(
            TensorData::new(vec![0_i64, 1], [2, 1]),
            &device,
        );
        let probe = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4, 5, 6], [2, 3]),
            &burn::tensor::Device::<TestBackend>::default(),
        );
        let before_valid = model.valid();
        let before_hidden = tensor_values(before_valid.forward_hidden(probe.clone()));
        let before_logits = tensor_values(before_valid.forward(probe.clone()));

        let mut initial_loss = None;
        let mut final_loss = 0.0f32;
        for _ in 0..256 {
            let scores = sequence_completion_score_tensor_with_gradient_scope(
                &model,
                &prompts,
                &candidates,
                crate::config::RuliadProofPolicyGradientScope::LanguageHeadOnly,
                &device,
            )
            .expect("language-head-only paired-target completion scores");
            assert_eq!(scores.group_sizes, vec![3, 3]);
            let loss = activation::log_softmax(scores.mean_log_scores.reshape([2, 3]), 1)
                .gather(1, targets.clone())
                .mean()
                .neg()
                .reshape([1]);
            let scalar = loss
                .clone()
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("language-head-only paired-target loss")[0];
            initial_loss.get_or_insert(scalar);
            final_loss = scalar;
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(3.0e-2, model, grads);
        }

        let after_valid = model.valid();
        let after_hidden = tensor_values(after_valid.forward_hidden(probe.clone()));
        let after_logits = tensor_values(after_valid.forward(probe));
        assert_eq!(
            maximum_abs_difference(&before_hidden, &after_hidden),
            0.0,
            "language-head-only policy update changed Dragon hidden states"
        );
        assert!(
            maximum_abs_difference(&before_logits, &after_logits) > 1.0e-5,
            "language-head-only policy update did not change completion logits"
        );
        let scores = proof_action_scores_batch(
            &after_valid,
            &prompts,
            &candidates,
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
            crate::config::RuliadProofPolicyScoring::CompletionLikelihood,
            &burn::tensor::Device::<TestBackend>::default(),
        )
        .expect("trained language-head-only paired-target scores");
        assert_eq!(best_candidate_index(&scores[0]), Some(0), "{scores:?}");
        assert_eq!(best_candidate_index(&scores[1]), Some(1), "{scores:?}");
        assert_confident_target_conditioning(
            "paired-target language head",
            initial_loss,
            final_loss,
        );
    }

    #[test]
    fn semantic_energy_score_head_only_forward_is_deterministic_and_rng_recoverable() {
        let device = burn::tensor::Device::<TrainBackend>::default();
        TrainBackend::seed(&device, 752);
        let mut config = burn_dragon_core::DragonConfig {
            n_layer: 1,
            n_embd: 16,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 32,
            dropout: 0.25,
            ..Default::default()
        };
        config.sequence_score_head.enabled = true;
        let model = DragonModel::<TrainBackend>::new(config, &device);
        let prompts = vec![vec![1, 2, 3, 4], vec![1, 2, 9, 4]];
        let menu = vec![vec![10, 11], vec![12, 13], vec![14, 15]];
        let candidates = vec![menu.clone(), menu];

        let random_probe = || {
            tensor_values(
                Tensor::<TrainBackend, 1>::random(
                    [32],
                    burn::tensor::Distribution::Uniform(-1.0, 1.0),
                    &device,
                )
                .inner(),
            )
        };

        TrainBackend::seed(&device, 9_001);
        let (first_scores, _) = sequence_energy_score_tensor_with_gradient_scope(
            &model,
            &prompts,
            &candidates,
            crate::config::RuliadProofPolicyGradientScope::ScoreHeadOnly,
            &device,
        )
        .expect("deterministic head-only semantic scores");
        let first_scores = tensor_values(first_scores.inner());
        let _noise = random_probe();
        let (second_scores, _) = sequence_energy_score_tensor_with_gradient_scope(
            &model,
            &prompts,
            &candidates,
            crate::config::RuliadProofPolicyGradientScope::ScoreHeadOnly,
            &device,
        )
        .expect("repeated deterministic head-only semantic scores");
        let second_scores = tensor_values(second_scores.inner());
        assert_eq!(
            maximum_abs_difference(&first_scores, &second_scores),
            0.0,
            "head-only policy scores depend on the ambient training RNG"
        );

        TrainBackend::seed(&device, 9_002);
        let expected = random_probe();
        let _scores = sequence_energy_score_tensor_with_gradient_scope(
            &model,
            &prompts,
            &candidates,
            crate::config::RuliadProofPolicyGradientScope::ScoreHeadOnly,
            &device,
        )
        .expect("head-only semantic scores before stream recovery");
        TrainBackend::seed(&device, 9_002);
        let actual = random_probe();
        assert_eq!(
            maximum_abs_difference(&expected, &actual),
            0.0,
            "explicit stochastic-substream reset did not recover the expected training RNG"
        );
    }

    #[test]
    fn semantic_energy_query_key_head_overfits_formal_counterfactual_target_pair() {
        use burn_dragon_universality::ruliad::formal::generate_formal_bundle;
        use burn_dragon_universality::ruliad::tokenize::RuliadByteTokenizer;
        use burn_dragon_universality::ruliad::{
            RuliadFormalGeneratorConfig, RuliadProofActionAnswerContract, RuliadTokenizationConfig,
            counterfactual_proof_action_target, oracle_proof_action_set, proof_action_answer,
            ruliad_proof_action_prompt,
        };

        let device = burn::tensor::Device::<TrainBackend>::default();
        TrainBackend::seed(&device, 76);
        let bundle = generate_formal_bundle(
            76,
            RuliadFormalGeneratorConfig {
                rewrite_depth: 2,
                leaf_count: 3,
                context_depth: 1,
                distractor_axioms: 1,
                ..Default::default()
            },
        )
        .expect("formal paired-target fixture");
        let actions = oracle_proof_action_set(&bundle.problem, &bundle.certificate, 0, 4)
            .expect("base action menu");
        let alternate_index = counterfactual_candidate_indices(&actions, 1, 1)
            .into_iter()
            .next()
            .expect("verifier-valid alternate target");
        let (counterfactual_problem, counterfactual_actions) =
            counterfactual_proof_action_target(&bundle.problem, &actions, alternate_index)
                .expect("counterfactual target");
        let rotation = candidate_presentation_rotations(
            crate::config::RuliadProofPolicyCandidateSymmetry::BalancedRotation,
            actions.selected_index,
            actions.candidates.len(),
            2,
        )
        .expect("base presentation")[0];
        let actions = actions.rotate_left(rotation).expect("presented base menu");
        let counterfactual_actions = counterfactual_actions
            .rotate_left(rotation)
            .expect("same presented counterfactual menu");
        let tokenizer =
            RuliadByteTokenizer::from_config(&RuliadTokenizationConfig::StructuredSymbolic {
                vocab_size: 512,
                eos_id: None,
            })
            .expect("structured tokenizer");
        let encode = |text: &str| {
            tokenizer
                .encode_payload(text)
                .into_iter()
                .map(i64::from)
                .collect::<Vec<_>>()
        };
        let mut prompts = vec![
            encode(
                &ruliad_proof_action_prompt(&bundle.problem, &actions).expect("base proof prompt"),
            ),
            encode(
                &ruliad_proof_action_prompt(&counterfactual_problem, &counterfactual_actions)
                    .expect("counterfactual proof prompt"),
            ),
        ];
        for prompt in &mut prompts {
            if prompt.len() > 128 {
                *prompt = prompt.split_off(prompt.len() - 128);
            }
        }
        let candidate_menu = |actions: &burn_dragon_universality::ruliad::RuliadProofActionSet| {
            (0..actions.candidates.len())
                .map(|index| {
                    encode(
                        &proof_action_answer(
                            actions,
                            index,
                            RuliadProofActionAnswerContract::SemanticStep,
                        )
                        .expect("semantic action"),
                    )
                })
                .collect::<Vec<_>>()
        };
        let candidates = vec![
            candidate_menu(&actions),
            candidate_menu(&counterfactual_actions),
        ];
        assert_eq!(
            candidates[0], candidates[1],
            "candidate menu must be held fixed"
        );
        assert_ne!(
            prompts[0], prompts[1],
            "only the requested target should change"
        );
        let prompt_diff_positions = prompts[0]
            .iter()
            .zip(&prompts[1])
            .enumerate()
            .filter_map(|(index, (left, right))| (left != right).then_some(index))
            .collect::<Vec<_>>();
        assert!(
            prompt_diff_positions
                .last()
                .is_some_and(|index| *index + 32 >= prompts[0].len()),
            "counterfactual target signal must remain near the decode boundary: {prompt_diff_positions:?}"
        );
        let targets = Tensor::<TrainBackend, 2, Int>::from_data(
            TensorData::new(
                vec![
                    actions.selected_index as i64,
                    counterfactual_actions.selected_index as i64,
                ],
                [2, 1],
            ),
            &device,
        );
        assert_ne!(
            actions.selected_index,
            counterfactual_actions.selected_index
        );

        let mut config = burn_dragon_core::DragonConfig {
            n_layer: 1,
            n_embd: 16,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 512,
            dropout: 0.0,
            ..Default::default()
        };
        config.sequence_score_head.enabled = true;
        config.sequence_score_head.projection_dim = 16;
        TrainBackend::seed(&device, 76);
        let mut model = DragonModel::<TrainBackend>::new(config, &device);
        let mut optimizer = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<TrainBackend, DragonModel<TrainBackend>>();
        let mut initial_loss = None;
        let mut final_loss = f32::INFINITY;
        for _ in 0..512 {
            let (scores, group_sizes) =
                sequence_energy_score_tensor(&model, &prompts, &candidates, &device)
                    .expect("formal paired-target scores");
            assert_eq!(group_sizes, vec![4, 4]);
            let loss = activation::log_softmax(scores.reshape([2, 4]), 1)
                .gather(1, targets.clone())
                .mean()
                .neg()
                .reshape([1]);
            let scalar = loss
                .clone()
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("formal paired-target loss")[0];
            initial_loss.get_or_insert(scalar);
            final_loss = scalar;
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(1.0e-2, model, grads);
        }

        let valid = model.valid();
        let scores = proof_action_scores_batch(
            &valid,
            &prompts,
            &candidates,
            RuliadProofActionAnswerContract::SemanticStep,
            crate::config::RuliadProofPolicyScoring::SemanticEnergy,
            &burn::tensor::Device::<TestBackend>::default(),
        )
        .expect("trained formal target-conditioned scores");
        assert_eq!(
            best_candidate_index(&scores[0]),
            Some(actions.selected_index),
            "scores={scores:?} prompt_diff_positions={prompt_diff_positions:?}"
        );
        assert_eq!(
            best_candidate_index(&scores[1]),
            Some(counterfactual_actions.selected_index),
            "scores={scores:?} prompt_diff_positions={prompt_diff_positions:?}"
        );
        assert_confident_target_conditioning(
            "formal paired-target energy head",
            initial_loss,
            final_loss,
        );
    }

    #[test]
    fn semantic_energy_prefix_reuse_matches_dense_full_sequence_reference() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 73);
        let mut config = burn_dragon_core::DragonConfig {
            n_layer: 2,
            n_embd: 16,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 32,
            dropout: 0.0,
            ..Default::default()
        };
        config.sequence_score_head.enabled = true;
        let model = DragonModel::<TestBackend>::new(config, &device);
        let prompts = vec![vec![1, 2, 3], vec![4, 5, 6, 7, 8], vec![9, 10, 11]];
        let candidates = vec![
            vec![vec![12], vec![13, 14, 15]],
            vec![vec![16, 17], vec![18, 19, 20], vec![21]],
            vec![vec![22, 23, 24, 25], vec![26, 27]],
        ];

        let dense = tensor_values(
            sequence_energy_score_tensor_dense(
                &model,
                &prompts,
                &candidates,
                false,
                &device,
                |inputs| model.forward_hidden(inputs),
            )
            .expect("dense reference scores"),
        );
        let (reused, group_sizes) =
            sequence_energy_score_tensor_with_prefix_reuse(&model, &prompts, &candidates, &device)
                .expect("prefix-reused scores");
        let reused = tensor_values(reused);

        assert_eq!(group_sizes, vec![2, 3, 2]);
        assert_eq!(dense.len(), reused.len());
        for (index, (dense, reused)) in dense.iter().zip(&reused).enumerate() {
            assert!(
                (dense - reused).abs() < 2.0e-4,
                "score {index} differs: dense={dense}, reused={reused}"
            );
        }
    }

    #[test]
    fn semantic_energy_scorer_requires_the_explicit_model_head() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let model = DragonModel::<TestBackend>::new(
            burn_dragon_core::DragonConfig {
                n_layer: 1,
                n_embd: 8,
                n_head: 1,
                mlp_internal_dim_multiplier: 2,
                vocab_size: 16,
                dropout: 0.0,
                ..Default::default()
            },
            &device,
        );
        let result = proof_action_scores_batch(
            &model,
            &[vec![1, 2]],
            &[vec![vec![3], vec![4]]],
            burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep,
            crate::config::RuliadProofPolicyScoring::SemanticEnergy,
            &device,
        );
        assert!(result.is_err());
    }

    #[test]
    fn semantic_candidate_trie_supervises_each_equivalent_decision_prefix() {
        let candidates = vec![
            b"g0|a:r1|f|1"
                .iter()
                .map(|token| i64::from(*token))
                .collect(),
            b"g1|a:r1|f|1"
                .iter()
                .map(|token| i64::from(*token))
                .collect(),
            b"g1|a:r2|f|1"
                .iter()
                .map(|token| i64::from(*token))
                .collect(),
            b"g1|a:r2|r|1"
                .iter()
                .map(|token| i64::from(*token))
                .collect(),
        ];
        let branches = semantic_candidate_trie_branches(&candidates, &[2]).expect("trie");
        assert_eq!(branches.len(), 3, "{branches:?}");
        assert_eq!(
            branches[0].prefix,
            b"g".iter()
                .map(|token| i64::from(*token))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            branches[0].candidate_tokens,
            vec![i64::from(b'0'), i64::from(b'1')]
        );
        assert_eq!(branches[0].equivalent_tokens, vec![i64::from(b'1')]);
        assert_eq!(branches[1].equivalent_tokens, vec![i64::from(b'2')]);
        assert_eq!(branches[2].equivalent_tokens, vec![i64::from(b'f')]);

        let equivalent = semantic_candidate_trie_branches(&candidates, &[2, 3]).expect("trie");
        assert_eq!(equivalent.len(), 3);
        assert_eq!(
            equivalent[2].equivalent_tokens,
            vec![i64::from(b'f'), i64::from(b'r')]
        );
    }

    #[test]
    fn trie_conditional_scores_form_the_exact_leaf_distribution() {
        let candidates = vec![
            b"g0|a:r1|f|1"
                .iter()
                .map(|token| i64::from(*token))
                .collect(),
            b"g1|a:r1|f|1"
                .iter()
                .map(|token| i64::from(*token))
                .collect(),
            b"g1|a:r2|f|1"
                .iter()
                .map(|token| i64::from(*token))
                .collect(),
            b"g1|a:r2|r|1"
                .iter()
                .map(|token| i64::from(*token))
                .collect(),
        ];
        let branches = semantic_candidate_trie_branches(&candidates, &[0, 1, 2, 3])
            .expect("complete scoring trie");
        assert_eq!(branches.len(), 3, "{branches:?}");
        let vocab = 128;
        let mut logits = vec![-12.0f32; branches.len() * vocab];
        let branch_probabilities = [[0.2f32, 0.8], [0.25, 0.75], [0.6, 0.4]];
        for (branch_index, (branch, probabilities)) in
            branches.iter().zip(branch_probabilities).enumerate()
        {
            for (token, probability) in branch.candidate_tokens.iter().zip(probabilities) {
                logits[branch_index * vocab + *token as usize] = probability.ln();
            }
            // A huge score on an illegal syntax token must not affect the constrained policy.
            logits[branch_index * vocab + 127] = 1_000.0;
        }
        let scores = trie_conditional_log_scores(
            &logits,
            vocab,
            &branches,
            &[branches.len()],
            &[candidates.len()],
        )
        .expect("trie scores")
        .pop()
        .expect("score group");
        let expected = [0.2f32, 0.2, 0.36, 0.24];
        for (score, expected) in scores.iter().zip(expected) {
            assert!((score.exp() - expected).abs() < 1.0e-6, "{scores:?}");
        }
        assert!((scores.iter().map(|score| score.exp()).sum::<f32>() - 1.0).abs() < 1.0e-6);
        assert_eq!(best_candidate_index(&scores), Some(2));
    }

    #[test]
    fn trie_conditional_scores_are_candidate_permutation_equivariant() {
        let canonical = vec![
            b"g0|a:r1|f|1"
                .iter()
                .map(|token| i64::from(*token))
                .collect(),
            b"g1|a:r1|f|1"
                .iter()
                .map(|token| i64::from(*token))
                .collect(),
            b"g1|a:r2|f|1"
                .iter()
                .map(|token| i64::from(*token))
                .collect(),
            b"g1|a:r2|r|1"
                .iter()
                .map(|token| i64::from(*token))
                .collect(),
        ];
        let score = |candidates: &[Vec<i64>]| {
            let branches = semantic_candidate_trie_branches(
                candidates,
                &(0..candidates.len()).collect::<Vec<_>>(),
            )
            .expect("complete scoring trie");
            let vocab = 128;
            let mut logits = vec![-12.0f32; branches.len() * vocab];
            for (branch_index, branch) in branches.iter().enumerate() {
                let probabilities = match branch.prefix.as_slice() {
                    prefix if prefix == b"g".iter().map(|v| i64::from(*v)).collect::<Vec<_>>() => {
                        [0.2f32, 0.8]
                    }
                    prefix
                        if prefix
                            == b"g1|a:r".iter().map(|v| i64::from(*v)).collect::<Vec<_>>() =>
                    {
                        [0.25, 0.75]
                    }
                    _ => [0.6, 0.4],
                };
                for (token, probability) in branch.candidate_tokens.iter().zip(probabilities) {
                    logits[branch_index * vocab + *token as usize] = probability.ln();
                }
            }
            trie_conditional_log_scores(
                &logits,
                vocab,
                &branches,
                &[branches.len()],
                &[candidates.len()],
            )
            .expect("trie scores")
            .pop()
            .expect("score group")
        };
        let canonical_scores = score(&canonical);
        let permutation = [2usize, 0, 3, 1];
        let permuted = permutation
            .iter()
            .map(|index| canonical[*index].clone())
            .collect::<Vec<_>>();
        let permuted_scores = score(&permuted);
        for (permuted_index, canonical_index) in permutation.into_iter().enumerate() {
            assert!(
                (permuted_scores[permuted_index] - canonical_scores[canonical_index]).abs()
                    < 1.0e-6,
                "canonical={canonical_scores:?} permuted={permuted_scores:?}"
            );
        }
    }

    #[test]
    fn prefix_reuse_sequence_scores_match_dense_teacher_forcing() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 53);
        let model = DragonModel::<TestBackend>::new(
            burn_dragon_core::DragonConfig {
                n_layer: 2,
                n_embd: 16,
                n_head: 2,
                mlp_internal_dim_multiplier: 2,
                vocab_size: 32,
                dropout: 0.0,
                ..Default::default()
            },
            &device,
        );
        let prompts = vec![vec![1, 2, 3, 4], vec![7, 8, 9], vec![4, 3, 2, 1]];
        let candidates = vec![
            vec![vec![10], vec![10, 11], vec![10, 12, 13], vec![14, 15]],
            vec![vec![16, 17, 18], vec![19], vec![16, 20]],
            vec![vec![21, 22], vec![23], vec![21, 24, 25]],
        ];
        let dense = sequence_completion_score_tensor(&model, &prompts, &candidates, &device)
            .expect("dense scores");
        let reused = sequence_completion_score_tensor_with_prefix_reuse(
            &model,
            &prompts,
            &candidates,
            &device,
        )
        .expect("prefix-reused scores");
        assert_eq!(dense.group_sizes, reused.group_sizes);

        let dense_mean = tensor_values(dense.mean_log_scores);
        let reused_mean = tensor_values(reused.mean_log_scores);
        let dense_sum = tensor_values(dense.sum_log_scores);
        let reused_sum = tensor_values(reused.sum_log_scores);
        let maximum_error = dense_mean
            .iter()
            .chain(&dense_sum)
            .zip(reused_mean.iter().chain(&reused_sum))
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0f32, f32::max);
        assert!(maximum_error <= 1.0e-5, "maximum_error={maximum_error}");
    }

    #[test]
    fn branch_only_decoder_matches_full_sequence_logits() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 43);
        let model = DragonModel::<TestBackend>::new(
            burn_dragon_core::DragonConfig {
                n_layer: 1,
                n_embd: 16,
                n_head: 2,
                mlp_internal_dim_multiplier: 2,
                vocab_size: 32,
                dropout: 0.0,
                ..Default::default()
            },
            &device,
        );
        let inputs = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4, 5, 6, 7, 8], [2, 4]),
            &device,
        );
        let full = model.forward(inputs.clone());
        let [_, _, vocab] = full.shape().dims::<3>();
        let positions = [1usize, 3usize];
        let indices = Tensor::<TestBackend, 3, Int>::from_data(
            TensorData::new(
                positions
                    .iter()
                    .flat_map(|position| std::iter::repeat_n(*position as i64, vocab))
                    .collect(),
                [2, 1, vocab],
            ),
            &device,
        );
        let expected = full.gather(1, indices).reshape([2, vocab]);
        let actual = logits_at_sequence_positions(&model, inputs, &positions, &device)
            .expect("branch logits");
        let maximum_error = tensor_values(expected)
            .into_iter()
            .zip(tensor_values(actual))
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0f32, f32::max);
        assert!(maximum_error <= 1.0e-6, "maximum_error={maximum_error}");
    }
}
