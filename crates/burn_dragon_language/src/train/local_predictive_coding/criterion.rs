use burn::tensor::{Int, Tensor, backend::Backend};
use burn_dragon_core::{DragonModel, DragonPredictiveCodingSequenceScoreHeadVjp};

/// Criterion data clamped at Dragon's terminal factor.
///
/// A typed tensor enum permits criterion changes without dynamic callbacks in
/// the accelerator path. The categorical-set form is verifier-native: valid
/// actions are marginalized inside an explicitly enumerated legal support set.
#[derive(Debug, Clone)]
pub(crate) enum LocalPcTerminalCriterion<B: Backend> {
    NextToken {
        targets: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
    },
    CategoricalSet {
        support_action_mask: Tensor<B, 3>,
        valid_action_mask: Tensor<B, 3>,
        row_weights: Tensor<B, 2>,
        eps: f32,
    },
    CategoricalSetAtPositions {
        positions: Tensor<B, 1, Int>,
        support_action_mask: Tensor<B, 2>,
        valid_action_mask: Tensor<B, 2>,
        row_weights: Tensor<B, 1>,
        propagate_hidden_gradient: bool,
        eps: f32,
    },
    SequenceCompletionSet {
        targets: Tensor<B, 2, Int>,
        token_mask: Tensor<B, 2>,
        support_action_mask: Tensor<B, 2>,
        valid_action_mask: Tensor<B, 2>,
        row_weights: Tensor<B, 1>,
        candidates_per_group: usize,
        propagate_hidden_gradient: bool,
        eps: f32,
    },
    SequenceEnergySetAtPositions {
        prompt_positions: Tensor<B, 1, Int>,
        terminal_positions: Tensor<B, 1, Int>,
        support_action_mask: Tensor<B, 2>,
        valid_action_mask: Tensor<B, 2>,
        target_action_weights: Option<Tensor<B, 2>>,
        row_weights: Tensor<B, 1>,
        candidates_per_group: usize,
        propagate_hidden_gradient: bool,
        eps: f32,
    },
    SequenceResidualEnergySetAtPositions {
        targets: Tensor<B, 2, Int>,
        token_mask: Tensor<B, 2>,
        prompt_positions: Tensor<B, 1, Int>,
        terminal_positions: Tensor<B, 1, Int>,
        support_action_mask: Tensor<B, 2>,
        valid_action_mask: Tensor<B, 2>,
        target_action_weights: Option<Tensor<B, 2>>,
        row_weights: Tensor<B, 1>,
        candidates_per_group: usize,
        propagate_hidden_gradient: bool,
        propagate_language_prior_gradient: bool,
        eps: f32,
    },
}

impl<B: Backend> LocalPcTerminalCriterion<B> {
    /// Repeat the complete terminal contract in depth-major order.
    ///
    /// Dragon's layer-local solver folds shared layer uses into the batch
    /// axis. Repeating every criterion tensor as one contiguous block per
    /// layer preserves candidate-group boundaries and aligns each block with
    /// the corresponding retained activity.
    pub(super) fn repeat_batch(&self, repeats: usize) -> Self {
        assert!(
            repeats > 0,
            "terminal criterion repeat count must be positive"
        );
        match self {
            Self::NextToken { targets, loss_mask } => Self::NextToken {
                targets: targets.clone().repeat_dim(0, repeats),
                loss_mask: loss_mask.clone().map(|mask| mask.repeat_dim(0, repeats)),
            },
            Self::CategoricalSet {
                support_action_mask,
                valid_action_mask,
                row_weights,
                eps,
            } => Self::CategoricalSet {
                support_action_mask: support_action_mask.clone().repeat_dim(0, repeats),
                valid_action_mask: valid_action_mask.clone().repeat_dim(0, repeats),
                row_weights: row_weights.clone().repeat_dim(0, repeats),
                eps: *eps,
            },
            Self::CategoricalSetAtPositions {
                positions,
                support_action_mask,
                valid_action_mask,
                row_weights,
                propagate_hidden_gradient,
                eps,
            } => Self::CategoricalSetAtPositions {
                positions: positions.clone().repeat_dim(0, repeats),
                support_action_mask: support_action_mask.clone().repeat_dim(0, repeats),
                valid_action_mask: valid_action_mask.clone().repeat_dim(0, repeats),
                row_weights: row_weights.clone().repeat_dim(0, repeats),
                propagate_hidden_gradient: *propagate_hidden_gradient,
                eps: *eps,
            },
            Self::SequenceCompletionSet {
                targets,
                token_mask,
                support_action_mask,
                valid_action_mask,
                row_weights,
                candidates_per_group,
                propagate_hidden_gradient,
                eps,
            } => Self::SequenceCompletionSet {
                targets: targets.clone().repeat_dim(0, repeats),
                token_mask: token_mask.clone().repeat_dim(0, repeats),
                support_action_mask: support_action_mask.clone().repeat_dim(0, repeats),
                valid_action_mask: valid_action_mask.clone().repeat_dim(0, repeats),
                row_weights: row_weights.clone().repeat_dim(0, repeats),
                candidates_per_group: *candidates_per_group,
                propagate_hidden_gradient: *propagate_hidden_gradient,
                eps: *eps,
            },
            Self::SequenceEnergySetAtPositions {
                prompt_positions,
                terminal_positions,
                support_action_mask,
                valid_action_mask,
                target_action_weights,
                row_weights,
                candidates_per_group,
                propagate_hidden_gradient,
                eps,
            } => Self::SequenceEnergySetAtPositions {
                prompt_positions: prompt_positions.clone().repeat_dim(0, repeats),
                terminal_positions: terminal_positions.clone().repeat_dim(0, repeats),
                support_action_mask: support_action_mask.clone().repeat_dim(0, repeats),
                valid_action_mask: valid_action_mask.clone().repeat_dim(0, repeats),
                target_action_weights: target_action_weights
                    .clone()
                    .map(|weights| weights.repeat_dim(0, repeats)),
                row_weights: row_weights.clone().repeat_dim(0, repeats),
                candidates_per_group: *candidates_per_group,
                propagate_hidden_gradient: *propagate_hidden_gradient,
                eps: *eps,
            },
            Self::SequenceResidualEnergySetAtPositions {
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
            } => Self::SequenceResidualEnergySetAtPositions {
                targets: targets.clone().repeat_dim(0, repeats),
                token_mask: token_mask.clone().repeat_dim(0, repeats),
                prompt_positions: prompt_positions.clone().repeat_dim(0, repeats),
                terminal_positions: terminal_positions.clone().repeat_dim(0, repeats),
                support_action_mask: support_action_mask.clone().repeat_dim(0, repeats),
                valid_action_mask: valid_action_mask.clone().repeat_dim(0, repeats),
                target_action_weights: target_action_weights
                    .clone()
                    .map(|weights| weights.repeat_dim(0, repeats)),
                row_weights: row_weights.clone().repeat_dim(0, repeats),
                candidates_per_group: *candidates_per_group,
                propagate_hidden_gradient: *propagate_hidden_gradient,
                propagate_language_prior_gradient: *propagate_language_prior_gradient,
                eps: *eps,
            },
        }
    }
}

fn apply_hidden_gradient_scope<B: Backend, const D: usize>(
    gradient: Tensor<B, D>,
    propagate_hidden_gradient: bool,
) -> Tensor<B, D> {
    if propagate_hidden_gradient {
        gradient
    } else {
        gradient.zeros_like()
    }
}

#[derive(Debug, Clone)]
pub(super) struct LocalPcTerminalActivityFactor<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub grad_hidden: Tensor<B, 3>,
    pub normalization: Tensor<B, 1>,
    pub verifier_probability_mass: Option<Tensor<B, 1>>,
}

#[derive(Debug, Clone)]
pub(super) struct LocalPcTerminalParameterFactor<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub grad_hidden: Tensor<B, 3>,
    /// `None` means the language head is outside this objective's gradient
    /// graph. Preserve that distinction at the optimizer boundary: an explicit
    /// zero would still advance Adam moments and can apply decoupled decay.
    pub grad_lm_head: Option<Tensor<B, 2>>,
    pub grad_sequence_score_head: Option<DragonPredictiveCodingSequenceScoreHeadVjp<B>>,
    pub supervised_tokens: Tensor<B, 1>,
    pub verifier_probability_mass: Option<Tensor<B, 1>>,
}

struct LocalPcSequenceCompletionFactor<B: Backend> {
    loss: Tensor<B, 1>,
    grad_logits: Tensor<B, 3>,
    normalization: Tensor<B, 1>,
    conditional_probability_mass: Tensor<B, 1>,
}

struct LocalPcSequenceEnergyFactor<B: Backend> {
    loss: Tensor<B, 1>,
    grad_hidden: Tensor<B, 3>,
    grad_candidate_scores: Tensor<B, 2>,
    grad_sequence_score_head: DragonPredictiveCodingSequenceScoreHeadVjp<B>,
    normalization: Tensor<B, 1>,
    conditional_probability_mass: Tensor<B, 1>,
}

struct LocalPcCategoricalFactor<B: Backend> {
    loss: Tensor<B, 1>,
    grad_logits: Tensor<B, 2>,
    normalization: Tensor<B, 1>,
    target_probability: Tensor<B, 1>,
}

fn categorical_target_factor<B: Backend>(
    scores: Tensor<B, 2>,
    support_action_mask: Tensor<B, 2>,
    valid_action_mask: Tensor<B, 2>,
    target_action_weights: Option<Tensor<B, 2>>,
    row_weights: Tensor<B, 1>,
    eps: f32,
) -> LocalPcCategoricalFactor<B> {
    match target_action_weights {
        Some(target_action_weights) => {
            let factor = burn_pc::categorical_conditional_distribution_cross_entropy(
                scores,
                support_action_mask,
                target_action_weights,
                row_weights,
                eps,
            );
            LocalPcCategoricalFactor {
                loss: factor.loss,
                grad_logits: factor.grad_logits,
                normalization: factor.normalization,
                target_probability: factor.target_probability,
            }
        }
        None => {
            let factor = burn_pc::categorical_conditional_set_nll(
                scores,
                support_action_mask,
                valid_action_mask,
                row_weights,
                eps,
            );
            LocalPcCategoricalFactor {
                loss: factor.loss,
                grad_logits: factor.grad_logits,
                normalization: factor.normalization,
                target_probability: factor.conditional_probability_mass,
            }
        }
    }
}

fn categorical_target_autodiff_loss<B: Backend>(
    scores: Tensor<B, 2>,
    support_action_mask: Tensor<B, 2>,
    valid_action_mask: Tensor<B, 2>,
    target_action_weights: Option<Tensor<B, 2>>,
    row_weights: Tensor<B, 1>,
    eps: f32,
) -> Tensor<B, 1> {
    let row_losses = match target_action_weights {
        Some(target_action_weights) => {
            burn_pc::categorical_conditional_distribution_cross_entropy_rows(
                scores,
                support_action_mask,
                target_action_weights,
                eps,
            )
        }
        None => burn_pc::categorical_conditional_set_log_probabilities(
            scores,
            support_action_mask,
            valid_action_mask,
            eps,
        )
        .mul_scalar(-1.0),
    };
    let normalization = row_weights.clone().sum().reshape([1]).clamp_min(eps);
    (row_losses * row_weights).sum().reshape([1]) / normalization
}

fn sequence_mean_log_scores<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    token_mask: Tensor<B, 2>,
    candidates_per_group: usize,
    eps: f32,
) -> (Tensor<B, 2>, Tensor<B, 3>, Tensor<B, 1>) {
    let [rows, time, _vocab] = logits.shape().dims::<3>();
    let candidates = candidates_per_group.max(1);
    assert!(rows.is_multiple_of(candidates));
    let groups = rows / candidates;
    assert_eq!(targets.shape().dims::<2>(), [rows, time]);
    assert_eq!(token_mask.shape().dims::<2>(), [rows, time]);

    let lengths = token_mask.clone().sum_dim(1).reshape([rows]).clamp_min(eps);
    let log_probs = burn_dragon_core::objective::log_probs_from_logits(logits);
    let selected =
        burn_dragon_core::objective::selected_token_log_probs(log_probs.clone(), targets);
    let scores = (selected * token_mask)
        .sum_dim(1)
        .reshape([rows])
        .div(lengths.clone())
        .reshape([groups, candidates]);
    (scores, log_probs, lengths)
}

fn sequence_mean_log_score_gradient<B: Backend>(
    log_probs: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    token_mask: Tensor<B, 2>,
    candidate_gradient: Tensor<B, 2>,
    lengths: Tensor<B, 1>,
) -> Tensor<B, 3> {
    let [rows, time, vocab] = log_probs.shape().dims::<3>();
    let target_log_probability_gradient = targets.one_hot::<3>(vocab).float() - log_probs.exp();
    target_log_probability_gradient
        * token_mask.reshape([rows, time, 1])
        * candidate_gradient
            .reshape([rows, 1, 1])
            .div(lengths.reshape([rows, 1, 1]))
}

#[allow(clippy::too_many_arguments)]
fn sequence_completion_factor<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    token_mask: Tensor<B, 2>,
    support_action_mask: Tensor<B, 2>,
    valid_action_mask: Tensor<B, 2>,
    row_weights: Tensor<B, 1>,
    candidates_per_group: usize,
    eps: f32,
) -> LocalPcSequenceCompletionFactor<B> {
    let [rows, time, _vocab] = logits.shape().dims::<3>();
    let candidates = candidates_per_group.max(1);
    assert!(rows.is_multiple_of(candidates));
    let groups = rows / candidates;
    assert_eq!(targets.shape().dims::<2>(), [rows, time]);
    assert_eq!(token_mask.shape().dims::<2>(), [rows, time]);
    assert_eq!(
        support_action_mask.shape().dims::<2>(),
        [groups, candidates]
    );
    assert_eq!(valid_action_mask.shape().dims::<2>(), [groups, candidates]);
    assert_eq!(row_weights.shape().dims::<1>(), [groups]);

    let (candidate_scores, log_probs, lengths) =
        sequence_mean_log_scores(logits, targets.clone(), token_mask.clone(), candidates, eps);
    let candidate_factor = burn_pc::categorical_conditional_set_nll(
        candidate_scores,
        support_action_mask,
        valid_action_mask,
        row_weights,
        eps,
    );

    let grad_logits = sequence_mean_log_score_gradient(
        log_probs,
        targets,
        token_mask,
        candidate_factor.grad_logits.clone(),
        lengths,
    );
    LocalPcSequenceCompletionFactor {
        loss: candidate_factor.loss,
        grad_logits,
        normalization: candidate_factor.normalization,
        conditional_probability_mass: candidate_factor.conditional_probability_mass,
    }
}

#[allow(clippy::too_many_arguments)]
fn sequence_energy_factor<B: Backend>(
    model: &DragonModel<B>,
    hidden: Tensor<B, 3>,
    prompt_positions: Tensor<B, 1, Int>,
    terminal_positions: Tensor<B, 1, Int>,
    candidate_prior: Option<Tensor<B, 2>>,
    support_action_mask: Tensor<B, 2>,
    valid_action_mask: Tensor<B, 2>,
    target_action_weights: Option<Tensor<B, 2>>,
    row_weights: Tensor<B, 1>,
    candidates_per_group: usize,
    propagate_hidden_gradient: bool,
    eps: f32,
) -> LocalPcSequenceEnergyFactor<B> {
    let [rows, time, dim] = hidden.shape().dims::<3>();
    let candidates = candidates_per_group.max(1);
    assert!(rows.is_multiple_of(candidates));
    let groups = rows / candidates;
    assert_eq!(prompt_positions.shape().dims::<1>(), [rows]);
    assert_eq!(terminal_positions.shape().dims::<1>(), [rows]);
    assert_eq!(
        support_action_mask.shape().dims::<2>(),
        [groups, candidates]
    );
    assert_eq!(valid_action_mask.shape().dims::<2>(), [groups, candidates]);
    assert_eq!(row_weights.shape().dims::<1>(), [groups]);

    let prompt_hidden = hidden
        .clone()
        .gather(
            1,
            prompt_positions
                .clone()
                .reshape([rows, 1, 1])
                .repeat_dim(2, dim),
        )
        .reshape([rows, dim]);
    let terminal_hidden = hidden
        .gather(
            1,
            terminal_positions
                .clone()
                .reshape([rows, 1, 1])
                .repeat_dim(2, dim),
        )
        .reshape([rows, dim]);
    let mut scores = model
        .predictive_coding_sequence_scores(prompt_hidden.clone(), terminal_hidden.clone())
        .expect("validated sequence score head")
        .reshape([groups, candidates]);
    if let Some(candidate_prior) = candidate_prior {
        assert_eq!(candidate_prior.shape().dims::<2>(), [groups, candidates]);
        scores = scores + candidate_prior;
    }
    let factor = categorical_target_factor(
        scores,
        support_action_mask,
        valid_action_mask,
        target_action_weights,
        row_weights,
        eps,
    );
    let grad_sequence_score_head = model
        .predictive_coding_sequence_score_vjp(
            prompt_hidden,
            terminal_hidden,
            factor.grad_logits.clone().reshape([rows]),
        )
        .expect("validated sequence score VJP");
    let prompt_mask = prompt_positions
        .one_hot::<2>(time)
        .float()
        .reshape([rows, time, 1]);
    let terminal_mask = terminal_positions
        .one_hot::<2>(time)
        .float()
        .reshape([rows, time, 1]);
    let grad_hidden = apply_hidden_gradient_scope(
        grad_sequence_score_head
            .grad_prompt_hidden
            .clone()
            .reshape([rows, 1, dim])
            * prompt_mask
            + grad_sequence_score_head
                .grad_terminal_hidden
                .clone()
                .reshape([rows, 1, dim])
                * terminal_mask,
        propagate_hidden_gradient,
    );
    LocalPcSequenceEnergyFactor {
        loss: factor.loss,
        grad_hidden,
        grad_candidate_scores: factor.grad_logits,
        grad_sequence_score_head,
        normalization: factor.normalization,
        conditional_probability_mass: factor.target_probability,
    }
}

impl<B: Backend> LocalPcTerminalCriterion<B>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    fn propagate_hidden_gradient(&self) -> bool {
        match self {
            Self::NextToken { .. } | Self::CategoricalSet { .. } => true,
            Self::CategoricalSetAtPositions {
                propagate_hidden_gradient,
                ..
            }
            | Self::SequenceCompletionSet {
                propagate_hidden_gradient,
                ..
            }
            | Self::SequenceEnergySetAtPositions {
                propagate_hidden_gradient,
                ..
            }
            | Self::SequenceResidualEnergySetAtPositions {
                propagate_hidden_gradient,
                ..
            } => *propagate_hidden_gradient,
        }
    }

    pub(super) fn next_token(
        targets: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
    ) -> Self {
        Self::NextToken { targets, loss_mask }
    }

    /// Evaluate the typed verifier criterion through backend autodiff.
    ///
    /// This is the matched global-backpropagation control for the analytic
    /// local VJP below. Both paths consume the same rows, masks, weights, and
    /// normalization; only the credit-assignment mechanism differs.
    pub(crate) fn verifier_autodiff_loss(&self, logits: Tensor<B, 3>) -> Option<Tensor<B, 1>> {
        match self {
            Self::NextToken { .. } => None,
            Self::CategoricalSet {
                support_action_mask,
                valid_action_mask,
                row_weights,
                eps,
            } => {
                let [batch, time, vocab] = logits.shape().dims::<3>();
                let rows = batch * time;
                let weights = row_weights.clone().reshape([rows]);
                let log_probability = burn_pc::categorical_conditional_set_log_probabilities(
                    logits.reshape([rows, vocab]),
                    support_action_mask.clone().reshape([rows, vocab]),
                    valid_action_mask.clone().reshape([rows, vocab]),
                    *eps,
                );
                let normalization = weights.clone().sum().reshape([1]).clamp_min(*eps);
                Some(
                    (log_probability * weights)
                        .sum()
                        .reshape([1])
                        .mul_scalar(-1.0)
                        / normalization,
                )
            }
            Self::CategoricalSetAtPositions {
                positions,
                support_action_mask,
                valid_action_mask,
                row_weights,
                propagate_hidden_gradient: _,
                eps,
            } => {
                let [rows, time, vocab] = logits.shape().dims::<3>();
                let selected = logits
                    .gather(
                        1,
                        positions.clone().reshape([rows, 1, 1]).repeat_dim(2, vocab),
                    )
                    .reshape([rows, vocab]);
                let log_probability = burn_pc::categorical_conditional_set_log_probabilities(
                    selected,
                    support_action_mask.clone(),
                    valid_action_mask.clone(),
                    *eps,
                );
                let normalization = row_weights.clone().sum().reshape([1]).clamp_min(*eps);
                debug_assert!(time > 0);
                Some(
                    (log_probability * row_weights.clone())
                        .sum()
                        .reshape([1])
                        .mul_scalar(-1.0)
                        / normalization,
                )
            }
            Self::SequenceCompletionSet {
                targets,
                token_mask,
                support_action_mask,
                valid_action_mask,
                row_weights,
                candidates_per_group,
                propagate_hidden_gradient: _,
                eps,
            } => {
                let [rows, time, _vocab] = logits.shape().dims::<3>();
                let candidates = (*candidates_per_group).max(1);
                assert!(rows.is_multiple_of(candidates));
                let groups = rows / candidates;
                let lengths = token_mask
                    .clone()
                    .sum_dim(1)
                    .reshape([rows])
                    .clamp_min(*eps);
                let log_probs = burn_dragon_core::objective::log_probs_from_logits(logits);
                let selected = burn_dragon_core::objective::selected_token_log_probs(
                    log_probs,
                    targets.clone(),
                );
                let scores = (selected * token_mask.clone())
                    .sum_dim(1)
                    .reshape([rows])
                    .div(lengths)
                    .reshape([groups, candidates]);
                let log_probability = burn_pc::categorical_conditional_set_log_probabilities(
                    scores,
                    support_action_mask.clone(),
                    valid_action_mask.clone(),
                    *eps,
                );
                let normalization = row_weights.clone().sum().reshape([1]).clamp_min(*eps);
                debug_assert!(time > 0);
                Some(
                    (log_probability * row_weights.clone())
                        .sum()
                        .reshape([1])
                        .mul_scalar(-1.0)
                        / normalization,
                )
            }
            Self::SequenceEnergySetAtPositions { .. }
            | Self::SequenceResidualEnergySetAtPositions { .. } => None,
        }
    }

    /// Evaluate a verifier objective from the shared hidden trajectory.
    ///
    /// Sequence energy is defined before the vocabulary projection, while
    /// completion likelihood reuses the ordinary token-logit objective. This
    /// method is diagnostic-only for PC and the global-backprop training arm;
    /// exact PC uses the analytic VJP below.
    pub(crate) fn verifier_autodiff_loss_from_hidden(
        &self,
        model: &DragonModel<B>,
        hidden: Tensor<B, 3>,
    ) -> Option<Tensor<B, 1>> {
        let hidden = if self.propagate_hidden_gradient() {
            hidden
        } else {
            hidden.detach()
        };
        match self {
            Self::SequenceEnergySetAtPositions {
                prompt_positions,
                terminal_positions,
                support_action_mask,
                valid_action_mask,
                target_action_weights,
                row_weights,
                candidates_per_group,
                propagate_hidden_gradient: _,
                eps,
            } => {
                let [rows, _time, dim] = hidden.shape().dims::<3>();
                let candidates = (*candidates_per_group).max(1);
                assert!(rows.is_multiple_of(candidates));
                let groups = rows / candidates;
                let prompt_hidden = hidden.clone().gather(
                    1,
                    prompt_positions
                        .clone()
                        .reshape([rows, 1, 1])
                        .repeat_dim(2, dim),
                );
                let terminal_hidden = hidden.gather(
                    1,
                    terminal_positions
                        .clone()
                        .reshape([rows, 1, 1])
                        .repeat_dim(2, dim),
                );
                let scores = model
                    .sequence_scores_from_hidden_pair(prompt_hidden, terminal_hidden)?
                    .reshape([groups, candidates]);
                Some(categorical_target_autodiff_loss(
                    scores,
                    support_action_mask.clone(),
                    valid_action_mask.clone(),
                    target_action_weights.clone(),
                    row_weights.clone(),
                    *eps,
                ))
            }
            Self::SequenceResidualEnergySetAtPositions {
                targets,
                token_mask,
                prompt_positions,
                terminal_positions,
                support_action_mask,
                valid_action_mask,
                target_action_weights,
                row_weights,
                candidates_per_group,
                propagate_hidden_gradient: _,
                propagate_language_prior_gradient,
                eps,
            } => {
                let [rows, _time, dim] = hidden.shape().dims::<3>();
                let candidates = (*candidates_per_group).max(1);
                assert!(rows.is_multiple_of(candidates));
                let groups = rows / candidates;
                let logits = model.predictive_coding_logits(hidden.clone());
                let logits = if *propagate_language_prior_gradient {
                    logits
                } else {
                    logits.detach()
                };
                let (completion_scores, _, _) = sequence_mean_log_scores(
                    logits,
                    targets.clone(),
                    token_mask.clone(),
                    candidates,
                    *eps,
                );
                let prompt_hidden = hidden.clone().gather(
                    1,
                    prompt_positions
                        .clone()
                        .reshape([rows, 1, 1])
                        .repeat_dim(2, dim),
                );
                let terminal_hidden = hidden.gather(
                    1,
                    terminal_positions
                        .clone()
                        .reshape([rows, 1, 1])
                        .repeat_dim(2, dim),
                );
                let residual_scores = model
                    .sequence_scores_from_hidden_pair(prompt_hidden, terminal_hidden)?
                    .reshape([groups, candidates]);
                Some(categorical_target_autodiff_loss(
                    completion_scores + residual_scores,
                    support_action_mask.clone(),
                    valid_action_mask.clone(),
                    target_action_weights.clone(),
                    row_weights.clone(),
                    *eps,
                ))
            }
            _ => self.verifier_autodiff_loss(model.predictive_coding_logits(hidden)),
        }
    }

    pub(super) fn activity_factor(
        &self,
        model: &DragonModel<B>,
        hidden: Tensor<B, 3>,
    ) -> LocalPcTerminalActivityFactor<B> {
        match self {
            Self::NextToken { targets, loss_mask } => {
                let factor = model.predictive_coding_head_activity_vjp(
                    hidden,
                    targets.clone(),
                    loss_mask.clone(),
                );
                LocalPcTerminalActivityFactor {
                    loss: factor.loss,
                    grad_hidden: factor.grad_hidden,
                    normalization: factor.normalization.reshape([1]),
                    verifier_probability_mass: None,
                }
            }
            Self::CategoricalSet {
                support_action_mask,
                valid_action_mask,
                row_weights,
                eps,
            } => {
                let [batch, time, _dim] = hidden.shape().dims::<3>();
                let logits = model.predictive_coding_logits(hidden);
                let vocab = logits.shape().dims::<3>()[2];
                assert_eq!(
                    support_action_mask.shape().dims::<3>(),
                    [batch, time, vocab]
                );
                assert_eq!(valid_action_mask.shape().dims::<3>(), [batch, time, vocab]);
                assert_eq!(row_weights.shape().dims::<2>(), [batch, time]);
                let factor = burn_pc::categorical_conditional_set_nll(
                    logits.reshape([batch * time, vocab]),
                    support_action_mask.clone().reshape([batch * time, vocab]),
                    valid_action_mask.clone().reshape([batch * time, vocab]),
                    row_weights.clone().reshape([batch * time]),
                    *eps,
                );
                let grad_hidden = model.predictive_coding_logits_activity_vjp(
                    factor.grad_logits.reshape([batch, time, vocab]),
                );
                LocalPcTerminalActivityFactor {
                    loss: factor.loss,
                    grad_hidden,
                    normalization: factor.normalization,
                    verifier_probability_mass: Some(factor.conditional_probability_mass),
                }
            }
            Self::CategoricalSetAtPositions {
                positions,
                support_action_mask,
                valid_action_mask,
                row_weights,
                propagate_hidden_gradient,
                eps,
            } => {
                let [rows, time, dim] = hidden.shape().dims::<3>();
                assert_eq!(positions.shape().dims::<1>(), [rows]);
                let head = model
                    .predictive_coding_head_weight()
                    .expect("validated flat PC head");
                let vocab = head.shape().dims::<2>()[1];
                assert_eq!(support_action_mask.shape().dims::<2>(), [rows, vocab]);
                assert_eq!(valid_action_mask.shape().dims::<2>(), [rows, vocab]);
                assert_eq!(row_weights.shape().dims::<1>(), [rows]);
                let gather = positions.clone().reshape([rows, 1, 1]).repeat_dim(2, dim);
                let selected_hidden = hidden.gather(1, gather).reshape([rows, dim]);
                let factor = burn_pc::categorical_conditional_set_nll(
                    selected_hidden.matmul(head.clone()),
                    support_action_mask.clone(),
                    valid_action_mask.clone(),
                    row_weights.clone(),
                    *eps,
                );
                let selected_gradient = factor.grad_logits.matmul(head.transpose());
                let position_mask = positions
                    .clone()
                    .one_hot::<2>(time)
                    .float()
                    .reshape([rows, time, 1]);
                let grad_hidden = apply_hidden_gradient_scope(
                    selected_gradient.reshape([rows, 1, dim]) * position_mask,
                    *propagate_hidden_gradient,
                );
                LocalPcTerminalActivityFactor {
                    loss: factor.loss,
                    grad_hidden,
                    normalization: factor.normalization,
                    verifier_probability_mass: Some(factor.conditional_probability_mass),
                }
            }
            Self::SequenceCompletionSet {
                targets,
                token_mask,
                support_action_mask,
                valid_action_mask,
                row_weights,
                candidates_per_group,
                propagate_hidden_gradient,
                eps,
            } => {
                let factor = sequence_completion_factor(
                    model.predictive_coding_logits(hidden),
                    targets.clone(),
                    token_mask.clone(),
                    support_action_mask.clone(),
                    valid_action_mask.clone(),
                    row_weights.clone(),
                    *candidates_per_group,
                    *eps,
                );
                let grad_hidden = apply_hidden_gradient_scope(
                    model.predictive_coding_logits_activity_vjp(factor.grad_logits),
                    *propagate_hidden_gradient,
                );
                LocalPcTerminalActivityFactor {
                    loss: factor.loss,
                    grad_hidden,
                    normalization: factor.normalization,
                    verifier_probability_mass: Some(factor.conditional_probability_mass),
                }
            }
            Self::SequenceEnergySetAtPositions {
                prompt_positions,
                terminal_positions,
                support_action_mask,
                valid_action_mask,
                target_action_weights,
                row_weights,
                candidates_per_group,
                propagate_hidden_gradient,
                eps,
            } => {
                let factor = sequence_energy_factor(
                    model,
                    hidden,
                    prompt_positions.clone(),
                    terminal_positions.clone(),
                    None,
                    support_action_mask.clone(),
                    valid_action_mask.clone(),
                    target_action_weights.clone(),
                    row_weights.clone(),
                    *candidates_per_group,
                    *propagate_hidden_gradient,
                    *eps,
                );
                LocalPcTerminalActivityFactor {
                    loss: factor.loss,
                    grad_hidden: factor.grad_hidden,
                    normalization: factor.normalization,
                    verifier_probability_mass: Some(factor.conditional_probability_mass),
                }
            }
            Self::SequenceResidualEnergySetAtPositions {
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
            } => {
                let logits = model.predictive_coding_logits(hidden.clone());
                let (completion_scores, log_probs, lengths) = sequence_mean_log_scores(
                    logits,
                    targets.clone(),
                    token_mask.clone(),
                    *candidates_per_group,
                    *eps,
                );
                let factor = sequence_energy_factor(
                    model,
                    hidden.clone(),
                    prompt_positions.clone(),
                    terminal_positions.clone(),
                    Some(completion_scores),
                    support_action_mask.clone(),
                    valid_action_mask.clone(),
                    target_action_weights.clone(),
                    row_weights.clone(),
                    *candidates_per_group,
                    *propagate_hidden_gradient,
                    *eps,
                );
                let grad_hidden = if *propagate_hidden_gradient {
                    if *propagate_language_prior_gradient {
                        let grad_logits = sequence_mean_log_score_gradient(
                            log_probs,
                            targets.clone(),
                            token_mask.clone(),
                            factor.grad_candidate_scores,
                            lengths,
                        );
                        factor.grad_hidden
                            + model.predictive_coding_logits_activity_vjp(grad_logits)
                    } else {
                        factor.grad_hidden
                    }
                } else {
                    hidden.zeros_like()
                };
                LocalPcTerminalActivityFactor {
                    loss: factor.loss,
                    grad_hidden,
                    normalization: factor.normalization,
                    verifier_probability_mass: Some(factor.conditional_probability_mass),
                }
            }
        }
    }

    pub(super) fn parameter_factor(
        &self,
        model: &DragonModel<B>,
        hidden: Tensor<B, 3>,
    ) -> LocalPcTerminalParameterFactor<B> {
        match self {
            Self::NextToken { targets, loss_mask } => {
                let factor =
                    model.predictive_coding_head_vjp(hidden, targets.clone(), loss_mask.clone());
                LocalPcTerminalParameterFactor {
                    loss: factor.loss,
                    grad_hidden: factor.grad_hidden,
                    grad_lm_head: Some(factor.grad_lm_head),
                    grad_sequence_score_head: None,
                    supervised_tokens: factor.supervised_tokens,
                    verifier_probability_mass: None,
                }
            }
            Self::CategoricalSet {
                support_action_mask,
                valid_action_mask,
                row_weights,
                eps,
            } => {
                let [batch, time, _dim] = hidden.shape().dims::<3>();
                let logits = model.predictive_coding_logits(hidden.clone());
                let vocab = logits.shape().dims::<3>()[2];
                assert_eq!(
                    support_action_mask.shape().dims::<3>(),
                    [batch, time, vocab]
                );
                assert_eq!(valid_action_mask.shape().dims::<3>(), [batch, time, vocab]);
                assert_eq!(row_weights.shape().dims::<2>(), [batch, time]);
                let factor = burn_pc::categorical_conditional_set_nll(
                    logits.reshape([batch * time, vocab]),
                    support_action_mask.clone().reshape([batch * time, vocab]),
                    valid_action_mask.clone().reshape([batch * time, vocab]),
                    row_weights.clone().reshape([batch * time]),
                    *eps,
                );
                let vjp = model.predictive_coding_logits_vjp(
                    hidden,
                    factor.grad_logits.reshape([batch, time, vocab]),
                );
                LocalPcTerminalParameterFactor {
                    loss: factor.loss,
                    grad_hidden: vjp.grad_hidden,
                    grad_lm_head: Some(vjp.grad_lm_head),
                    grad_sequence_score_head: None,
                    supervised_tokens: factor.normalization,
                    verifier_probability_mass: Some(factor.conditional_probability_mass),
                }
            }
            Self::CategoricalSetAtPositions {
                positions,
                support_action_mask,
                valid_action_mask,
                row_weights,
                propagate_hidden_gradient,
                eps,
            } => {
                let [rows, time, dim] = hidden.shape().dims::<3>();
                assert_eq!(positions.shape().dims::<1>(), [rows]);
                let head = model
                    .predictive_coding_head_weight()
                    .expect("validated flat PC head");
                let vocab = head.shape().dims::<2>()[1];
                assert_eq!(support_action_mask.shape().dims::<2>(), [rows, vocab]);
                assert_eq!(valid_action_mask.shape().dims::<2>(), [rows, vocab]);
                assert_eq!(row_weights.shape().dims::<1>(), [rows]);
                let gather = positions.clone().reshape([rows, 1, 1]).repeat_dim(2, dim);
                let selected_hidden = hidden.gather(1, gather).reshape([rows, dim]);
                let factor = burn_pc::categorical_conditional_set_nll(
                    selected_hidden.clone().matmul(head.clone()),
                    support_action_mask.clone(),
                    valid_action_mask.clone(),
                    row_weights.clone(),
                    *eps,
                );
                let grad_logits = factor.grad_logits;
                let selected_gradient = grad_logits.clone().matmul(head.transpose());
                let position_mask = positions
                    .clone()
                    .one_hot::<2>(time)
                    .float()
                    .reshape([rows, time, 1]);
                let grad_hidden = apply_hidden_gradient_scope(
                    selected_gradient.reshape([rows, 1, dim]) * position_mask,
                    *propagate_hidden_gradient,
                );
                let grad_lm_head = selected_hidden.transpose().matmul(grad_logits);
                LocalPcTerminalParameterFactor {
                    loss: factor.loss,
                    grad_hidden,
                    grad_lm_head: Some(grad_lm_head),
                    grad_sequence_score_head: None,
                    supervised_tokens: factor.normalization,
                    verifier_probability_mass: Some(factor.conditional_probability_mass),
                }
            }
            Self::SequenceCompletionSet {
                targets,
                token_mask,
                support_action_mask,
                valid_action_mask,
                row_weights,
                candidates_per_group,
                propagate_hidden_gradient,
                eps,
            } => {
                let factor = sequence_completion_factor(
                    model.predictive_coding_logits(hidden.clone()),
                    targets.clone(),
                    token_mask.clone(),
                    support_action_mask.clone(),
                    valid_action_mask.clone(),
                    row_weights.clone(),
                    *candidates_per_group,
                    *eps,
                );
                let vjp = model.predictive_coding_logits_vjp(hidden, factor.grad_logits);
                LocalPcTerminalParameterFactor {
                    loss: factor.loss,
                    grad_hidden: apply_hidden_gradient_scope(
                        vjp.grad_hidden,
                        *propagate_hidden_gradient,
                    ),
                    grad_lm_head: Some(vjp.grad_lm_head),
                    grad_sequence_score_head: None,
                    supervised_tokens: factor.normalization,
                    verifier_probability_mass: Some(factor.conditional_probability_mass),
                }
            }
            Self::SequenceEnergySetAtPositions {
                prompt_positions,
                terminal_positions,
                support_action_mask,
                valid_action_mask,
                target_action_weights,
                row_weights,
                candidates_per_group,
                propagate_hidden_gradient,
                eps,
            } => {
                let factor = sequence_energy_factor(
                    model,
                    hidden,
                    prompt_positions.clone(),
                    terminal_positions.clone(),
                    None,
                    support_action_mask.clone(),
                    valid_action_mask.clone(),
                    target_action_weights.clone(),
                    row_weights.clone(),
                    *candidates_per_group,
                    *propagate_hidden_gradient,
                    *eps,
                );
                LocalPcTerminalParameterFactor {
                    loss: factor.loss,
                    grad_hidden: factor.grad_hidden,
                    grad_lm_head: None,
                    grad_sequence_score_head: Some(factor.grad_sequence_score_head),
                    supervised_tokens: factor.normalization,
                    verifier_probability_mass: Some(factor.conditional_probability_mass),
                }
            }
            Self::SequenceResidualEnergySetAtPositions {
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
            } => {
                let logits = model.predictive_coding_logits(hidden.clone());
                let (completion_scores, log_probs, lengths) = sequence_mean_log_scores(
                    logits,
                    targets.clone(),
                    token_mask.clone(),
                    *candidates_per_group,
                    *eps,
                );
                let factor = sequence_energy_factor(
                    model,
                    hidden.clone(),
                    prompt_positions.clone(),
                    terminal_positions.clone(),
                    Some(completion_scores),
                    support_action_mask.clone(),
                    valid_action_mask.clone(),
                    target_action_weights.clone(),
                    row_weights.clone(),
                    *candidates_per_group,
                    *propagate_hidden_gradient,
                    *eps,
                );
                let (grad_hidden, grad_lm_head) =
                    if *propagate_hidden_gradient && *propagate_language_prior_gradient {
                        let grad_logits = sequence_mean_log_score_gradient(
                            log_probs,
                            targets.clone(),
                            token_mask.clone(),
                            factor.grad_candidate_scores,
                            lengths,
                        );
                        let lm_vjp = model.predictive_coding_logits_vjp(hidden, grad_logits);
                        (
                            factor.grad_hidden + lm_vjp.grad_hidden,
                            Some(lm_vjp.grad_lm_head),
                        )
                    } else if *propagate_hidden_gradient {
                        (factor.grad_hidden, None)
                    } else {
                        (hidden.zeros_like(), None)
                    };
                LocalPcTerminalParameterFactor {
                    loss: factor.loss,
                    grad_hidden,
                    grad_lm_head,
                    grad_sequence_score_head: Some(factor.grad_sequence_score_head),
                    supervised_tokens: factor.normalization,
                    verifier_probability_mass: Some(factor.conditional_probability_mass),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn_autodiff::Autodiff;
    use burn_dragon_core::{DragonConfig, RotaryEmbedding, SequenceTrainingExecutor};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;
    type AutodiffTestBackend = Autodiff<TestBackend>;

    fn model(device: &burn::tensor::Device<TestBackend>) -> DragonModel<TestBackend> {
        let mut config = DragonConfig {
            n_layer: 2,
            n_embd: 8,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 16,
            dropout: 0.0,
            ..DragonConfig::default()
        };
        config.sequence_kernel.executor = SequenceTrainingExecutor::DenseScoreShortContext;
        config.fused_kernels.rotary_embedding = RotaryEmbedding::Alibi;
        DragonModel::new(config, device)
    }

    fn max_abs<const D: usize>(left: Tensor<TestBackend, D>, right: Tensor<TestBackend, D>) -> f32 {
        burn_pc::diagnostic_scalar_f32((left - right).abs().max())
    }

    #[test]
    fn singleton_verifier_set_matches_next_token_terminal_vjp() {
        let device = Default::default();
        let model = model(&device);
        let hidden = Tensor::<TestBackend, 3>::random(
            [1, 2, 8],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );
        let targets = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![2_i64, 7], [1, 2]),
            &device,
        );
        let next_token = LocalPcTerminalCriterion::next_token(targets.clone(), None)
            .parameter_factor(&model, hidden.clone());
        let verifier = LocalPcTerminalCriterion::CategoricalSet {
            support_action_mask: Tensor::ones([1, 2, 16], &device),
            valid_action_mask: targets.one_hot::<3>(16).float(),
            row_weights: Tensor::ones([1, 2], &device),
            eps: 1.0e-8,
        }
        .parameter_factor(&model, hidden.clone());
        let verifier_activity = LocalPcTerminalCriterion::CategoricalSet {
            support_action_mask: Tensor::ones([1, 2, 16], &device),
            valid_action_mask: Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(vec![2_i64, 7], [1, 2]),
                &device,
            )
            .one_hot::<3>(16)
            .float(),
            row_weights: Tensor::ones([1, 2], &device),
            eps: 1.0e-8,
        }
        .activity_factor(&model, hidden);

        assert!(max_abs(next_token.loss, verifier.loss) < 2.0e-6);
        assert!(max_abs(next_token.grad_hidden, verifier.grad_hidden.clone()) < 2.0e-6);
        assert!(
            max_abs(
                next_token.grad_lm_head.expect("next-token head gradient"),
                verifier.grad_lm_head.expect("verifier head gradient"),
            ) < 2.0e-6
        );
        assert!(max_abs(verifier.grad_hidden, verifier_activity.grad_hidden) < 2.0e-6);
        assert!(verifier.verifier_probability_mass.is_some());
        assert!(verifier_activity.verifier_probability_mass.is_some());
        assert_eq!(
            burn_pc::diagnostic_scalar_f32(verifier_activity.normalization),
            2.0
        );
        assert!(burn_pc::diagnostic_scalar_f32(verifier_activity.loss).is_finite());
    }

    #[test]
    fn sparse_position_set_matches_selected_next_token_rows() {
        let device = Default::default();
        let model = model(&device);
        let hidden = Tensor::<TestBackend, 3>::random(
            [2, 3, 8],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        );
        let positions =
            Tensor::<TestBackend, 1, Int>::from_data(TensorData::new(vec![1_i64, 2], [2]), &device);
        let targets = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![2_i64, 7], [2, 1]),
            &device,
        );
        let selected_hidden = hidden
            .clone()
            .gather(1, positions.clone().reshape([2, 1, 1]).repeat_dim(2, 8))
            .reshape([2, 1, 8]);
        let expected = model.predictive_coding_head_vjp(selected_hidden, targets.clone(), None);
        let sparse = LocalPcTerminalCriterion::CategoricalSetAtPositions {
            positions,
            support_action_mask: Tensor::ones([2, 16], &device),
            valid_action_mask: targets.reshape([2]).one_hot::<2>(16).float(),
            row_weights: Tensor::ones([2], &device),
            propagate_hidden_gradient: true,
            eps: 1.0e-8,
        }
        .parameter_factor(&model, hidden);

        assert!(max_abs(expected.loss, sparse.loss) < 2.0e-6);
        assert!(
            max_abs(
                expected.grad_lm_head,
                sparse.grad_lm_head.expect("sparse head gradient"),
            ) < 2.0e-6
        );
        let sparse_selected = sparse
            .grad_hidden
            .clone()
            .slice([0..1, 1..2, 0..8])
            .reshape([1, 8]);
        let expected_first = expected
            .grad_hidden
            .slice([0..1, 0..1, 0..8])
            .reshape([1, 8]);
        assert!(max_abs(sparse_selected, expected_first) < 2.0e-6);
        let off_position_mass = sparse.grad_hidden.slice([0..1, 0..1, 0..8]).abs().max();
        assert!(burn_pc::diagnostic_scalar_f32(off_position_mass) < 1.0e-8);
    }

    #[test]
    fn sparse_verifier_autodiff_control_matches_the_analytic_logit_vjp() {
        let device = Default::default();
        let logits = Tensor::<AutodiffTestBackend, 3>::from_floats(
            [
                [[0.0, 0.1, 0.2, 0.3], [0.5, -0.2, 0.8, 0.1], [0.0; 4]],
                [[0.0; 4], [0.0; 4], [-0.4, 0.7, 0.2, 0.9]],
            ],
            &device,
        )
        .require_grad();
        let positions = Tensor::<AutodiffTestBackend, 1, Int>::from_data(
            TensorData::new(vec![1_i64, 2], [2]),
            &device,
        );
        let inner_positions =
            Tensor::<TestBackend, 1, Int>::from_data(TensorData::new(vec![1_i64, 2], [2]), &device);
        let support = Tensor::<AutodiffTestBackend, 2>::from_floats(
            [[1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0]],
            &device,
        );
        let valid = Tensor::<AutodiffTestBackend, 2>::from_floats(
            [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            &device,
        );
        let weights = Tensor::<AutodiffTestBackend, 1>::from_floats([0.25, 0.75], &device);
        let selected = logits
            .clone()
            .gather(1, positions.clone().reshape([2, 1, 1]).repeat_dim(2, 4))
            .reshape([2, 4]);
        let analytic = burn_pc::categorical_conditional_set_nll(
            selected,
            support.clone(),
            valid.clone(),
            weights.clone(),
            1.0e-8,
        );
        let loss = LocalPcTerminalCriterion::CategoricalSetAtPositions {
            positions: positions.clone(),
            support_action_mask: support,
            valid_action_mask: valid,
            row_weights: weights,
            propagate_hidden_gradient: true,
            eps: 1.0e-8,
        }
        .verifier_autodiff_loss(logits.clone())
        .expect("verifier loss");
        let loss_residual = burn_pc::diagnostic_scalar_f32(
            (loss.clone() - analytic.loss).abs().max().detach().inner(),
        );
        let grads = loss.backward();
        let autodiff = logits
            .grad(&grads)
            .expect("logits gradient")
            .gather(1, inner_positions.reshape([2, 1, 1]).repeat_dim(2, 4))
            .reshape([2, 4]);
        let gradient_residual = burn_pc::diagnostic_scalar_f32(
            (autodiff - analytic.grad_logits.detach().inner())
                .abs()
                .max(),
        );

        assert!(loss_residual < 1.0e-6, "loss residual={loss_residual}");
        assert!(
            gradient_residual < 2.0e-6,
            "gradient residual={gradient_residual}"
        );
    }
}
