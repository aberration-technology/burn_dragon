//! Teacher-forced diagnostics for Ruliad prompt-to-answer binding.

use super::*;

#[derive(Clone, Copy, Debug, Default, Serialize, PartialEq)]
pub struct RuliadTeacherForcedEvaluation {
    /// Version 2 scores complete prompts/answers using bounded recurrent chunks.
    pub version: u32,
    pub items: usize,
    pub completion_tokens: usize,
    pub mean_nll: f64,
    pub mean_sequence_nll: f64,
    pub token_accuracy: f64,
    pub first_token_accuracy: f64,
    pub sequence_accuracy: f64,
    pub context_swap_items: usize,
    pub context_swap_mean_nll: f64,
    /// Positive values mean the same target completion is less likely under a
    /// mismatched prompt than under its oracle prompt.
    pub context_binding_nll_gain: f64,
}

#[derive(Clone, Debug)]
struct RuliadTeacherForcedRow {
    inputs: Vec<i64>,
    targets: Vec<i64>,
    mask: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct RuliadTeacherForcedRowScore {
    token_count: usize,
    nll_sum: f64,
    correct_tokens: usize,
    first_token_correct: bool,
    sequence_correct: bool,
}

fn ruliad_teacher_forced_completion_tokens(
    dataset: &Dataset,
    item: &burn_dragon_universality::RuliadEvalItem,
) -> Option<Vec<i64>> {
    let completion = dataset
        .encode_ruliad_payload_tokens(&format!(
            "{}\n{}",
            item.expected_answer.trim(),
            item.document_close_marker()
        ))?
        .into_iter()
        .map(i64::from)
        .collect::<Vec<_>>();
    (!completion.is_empty()).then_some(completion)
}

fn ruliad_teacher_forced_row<B: BackendTrait>(
    prompt_tokens: &[i64],
    completion_tokens: &[i64],
) -> Option<RuliadTeacherForcedRow> {
    if prompt_tokens.is_empty() || completion_tokens.is_empty() {
        return None;
    }
    let (inputs, targets, mask) = LanguageTrainModel::<B>::ruliad_policy_row_from_completion(
        prompt_tokens,
        completion_tokens,
    )?;
    Some(RuliadTeacherForcedRow {
        inputs,
        targets,
        mask,
    })
}

#[cfg(test)]
fn score_ruliad_teacher_forced_row(
    selected_log_probabilities: &[f32],
    predictions: &[i64],
    targets: &[i64],
    mask: &[f32],
) -> RuliadTeacherForcedRowScore {
    let mut score = RuliadTeacherForcedRowScore {
        sequence_correct: true,
        ..RuliadTeacherForcedRowScore::default()
    };
    for index in 0..mask.len() {
        if mask[index] <= f32::EPSILON {
            continue;
        }
        let correct = predictions.get(index) == targets.get(index);
        if score.token_count == 0 {
            score.first_token_correct = correct;
        }
        score.token_count = score.token_count.saturating_add(1);
        score.correct_tokens = score.correct_tokens.saturating_add(usize::from(correct));
        score.sequence_correct &= correct;
        score.nll_sum -= f64::from(
            selected_log_probabilities
                .get(index)
                .copied()
                .unwrap_or(f32::NEG_INFINITY),
        );
    }
    if score.token_count == 0 {
        score.sequence_correct = false;
    }
    score
}

#[cfg(test)]
fn summarize_ruliad_teacher_forced_rows(
    selected_log_probabilities: &[f32],
    predictions: &[i64],
    targets: &[i64],
    masks: &[f32],
    row_count: usize,
    time: usize,
    matched_rows: usize,
    swap_source_rows: &[usize],
) -> RuliadTeacherForcedEvaluation {
    let scores = (0..row_count)
        .map(|row| {
            let start = row.saturating_mul(time);
            let end = start.saturating_add(time);
            score_ruliad_teacher_forced_row(
                selected_log_probabilities
                    .get(start..end)
                    .unwrap_or_default(),
                predictions.get(start..end).unwrap_or_default(),
                targets.get(start..end).unwrap_or_default(),
                masks.get(start..end).unwrap_or_default(),
            )
        })
        .collect::<Vec<_>>();
    summarize_scores(&scores, matched_rows, swap_source_rows)
}

fn summarize_scores(
    scores: &[RuliadTeacherForcedRowScore],
    matched_rows: usize,
    swap_source_rows: &[usize],
) -> RuliadTeacherForcedEvaluation {
    let matched = scores.iter().take(matched_rows);
    let matched_tokens = matched
        .clone()
        .map(|score| score.token_count)
        .sum::<usize>();
    let matched_nll = matched.clone().map(|score| score.nll_sum).sum::<f64>();
    let matched_correct = matched
        .clone()
        .map(|score| score.correct_tokens)
        .sum::<usize>();
    let first_correct = matched
        .clone()
        .filter(|score| score.first_token_correct)
        .count();
    let sequence_correct = matched.filter(|score| score.sequence_correct).count();

    let swap_scores = scores.iter().skip(matched_rows);
    let swap_tokens = swap_scores
        .clone()
        .map(|score| score.token_count)
        .sum::<usize>();
    let swap_nll = swap_scores.map(|score| score.nll_sum).sum::<f64>();
    let paired_matched_nll = swap_source_rows
        .iter()
        .filter_map(|source| scores.get(*source))
        .map(|score| score.nll_sum)
        .sum::<f64>();
    let paired_tokens = swap_source_rows
        .iter()
        .filter_map(|source| scores.get(*source))
        .map(|score| score.token_count)
        .sum::<usize>();

    RuliadTeacherForcedEvaluation {
        version: 2,
        items: matched_rows,
        completion_tokens: matched_tokens,
        mean_nll: matched_nll / matched_tokens.max(1) as f64,
        mean_sequence_nll: matched_nll / matched_rows.max(1) as f64,
        token_accuracy: matched_correct as f64 / matched_tokens.max(1) as f64,
        first_token_accuracy: first_correct as f64 / matched_rows.max(1) as f64,
        sequence_accuracy: sequence_correct as f64 / matched_rows.max(1) as f64,
        context_swap_items: swap_source_rows.len(),
        context_swap_mean_nll: swap_nll / swap_tokens.max(1) as f64,
        context_binding_nll_gain: swap_nll / swap_tokens.max(1) as f64
            - paired_matched_nll / paired_tokens.max(1) as f64,
    }
}

/// Retain full causal context without allocating an all-panel activation tensor.
/// Device-side reductions make readback cost proportional to rows, not tokens.
fn score_rows<B: BackendTrait>(
    model: &LanguageTrainModel<B>,
    rows: &[RuliadTeacherForcedRow],
    chunk_size: usize,
    batch_rows: usize,
    device: &B::Device,
) -> Result<Vec<RuliadTeacherForcedRowScore>> {
    anyhow::ensure!(
        chunk_size > 0 && batch_rows > 0,
        "teacher-forced scoring limits must be positive"
    );
    let mut scores = Vec::with_capacity(rows.len());
    let width = chunk_size.min(rows.iter().map(|row| row.inputs.len()).max().unwrap_or(1));
    for batch in rows.chunks(batch_rows) {
        let count = batch.len();
        let time = batch.iter().map(|row| row.inputs.len()).max().unwrap_or(0);
        let first_tokens = batch
            .iter()
            .map(|row| row.mask.iter().position(|mask| *mask > 0.0))
            .collect::<Vec<_>>();
        let mut state = model.model.init_state();
        let mut sums = Tensor::<B, 2>::zeros([count, 3], device).cast(burn::tensor::DType::F32);
        // Pad the final causal chunk, not the prefix. Reusing a stable shape
        // avoids compiling a separate CUDA specialization for every remainder.
        for start in (0..time).step_by(width) {
            let mut inputs = vec![0_i64; count * width];
            let mut targets = vec![0_i64; count * width];
            let mut masks = vec![0_f32; count * width];
            let mut first_masks = vec![0_f32; count * width];
            for (index, row) in batch.iter().enumerate() {
                let offset = index * width;
                let end = (start + width).min(row.inputs.len());
                if start >= end {
                    continue;
                }
                let target = offset..offset + end - start;
                inputs[target.clone()].copy_from_slice(&row.inputs[start..end]);
                targets[target.clone()].copy_from_slice(&row.targets[start..end]);
                masks[target].copy_from_slice(&row.mask[start..end]);
                if let Some(first) = first_tokens[index]
                    && (start..end).contains(&first)
                {
                    first_masks[offset + first - start] = 1.0;
                }
            }
            let summary = crate::summary_events::summary_event_mask_tensor::<B>(
                &inputs,
                count,
                width,
                model.model.summary_memory_write_trigger_token_ids(),
                device,
            );
            let inputs =
                Tensor::<B, 2, Int>::from_data(TensorData::new(inputs, [count, width]), device);
            let targets =
                Tensor::<B, 2, Int>::from_data(TensorData::new(targets, [count, width]), device);
            let masks = Tensor::<B, 2>::from_data(TensorData::new(masks, [count, width]), device)
                .cast(burn::tensor::DType::F32);
            let first_masks =
                Tensor::<B, 2>::from_data(TensorData::new(first_masks, [count, width]), device)
                    .cast(burn::tensor::DType::F32);
            let logits = match summary {
                Some(mask) => model
                    .model
                    .forward_with_state_and_summary_event_mask(inputs, mask, &mut state),
                None => model.model.forward_with_state(inputs, &mut state),
            }
            .cast(burn::tensor::DType::F32);
            let correct = logits
                .clone()
                .argmax(2)
                .reshape([count, width])
                .equal(targets.clone())
                .float()
                .cast(burn::tensor::DType::F32);
            let nll = burn::tensor::activation::log_softmax(logits, 2)
                .gather(2, targets.reshape([count, width, 1]))
                .reshape([count, width])
                .mul_scalar(-1.0);
            sums = sums
                + Tensor::cat(
                    vec![
                        (nll * masks.clone()).sum_dim(1),
                        (correct.clone() * masks).sum_dim(1),
                        (correct * first_masks).sum_dim(1),
                    ],
                    1,
                );
        }
        let values = sums
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .map_err(|error| anyhow!("reading teacher-forced row reductions: {error}"))?;
        anyhow::ensure!(
            values.iter().all(|value| value.is_finite()),
            "non-finite teacher-forced row reduction"
        );
        for (row, values) in batch.iter().zip(values.chunks_exact(3)) {
            let token_count = row.mask.iter().filter(|mask| **mask > 0.0).count();
            let correct_tokens = values[1].round() as usize;
            scores.push(RuliadTeacherForcedRowScore {
                token_count,
                nll_sum: f64::from(values[0]),
                correct_tokens,
                first_token_correct: values[2] > 0.5,
                sequence_correct: token_count > 0 && token_count == correct_tokens,
            });
        }
    }
    Ok(scores)
}

pub(super) fn evaluate_ruliad_teacher_forced_context<B>(
    dataset: &Dataset,
    model: &LanguageTrainModel<B>,
    training: &TrainingHyperparameters,
    probe_items: &[crate::dataset::RuliadValidationProbeItem],
    batch_rows: usize,
    device: &B::Device,
) -> Result<RuliadTeacherForcedEvaluation>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    let completion_rows = probe_items
        .iter()
        .map(|probe| ruliad_teacher_forced_completion_tokens(dataset, &probe.item))
        .collect::<Vec<_>>();
    let mut matched = Vec::with_capacity(probe_items.len());
    let mut matched_probe_indices = Vec::with_capacity(probe_items.len());
    for (probe_index, (probe, completion)) in
        probe_items.iter().zip(completion_rows.iter()).enumerate()
    {
        let Some(completion) = completion.as_deref() else {
            continue;
        };
        let Some(row) = ruliad_teacher_forced_row::<B>(&probe.prompt_tokens, completion) else {
            continue;
        };
        matched.push(row);
        matched_probe_indices.push(probe_index);
    }
    anyhow::ensure!(
        matched.len() == probe_items.len(),
        "teacher-forced panel contains an unencodable or empty prompt/answer"
    );
    if matched.is_empty() {
        return Ok(RuliadTeacherForcedEvaluation {
            version: 2,
            ..Default::default()
        });
    }

    // Pair each target with a different-answer prompt. Same-answer swaps are
    // uninformative because both contexts may legitimately support the target.
    let mut swapped = Vec::new();
    let mut swap_source_rows = Vec::new();
    for (source_row, source_probe_index) in matched_probe_indices.iter().copied().enumerate() {
        let source_probe = &probe_items[source_probe_index];
        let Some(completion) = completion_rows[source_probe_index].as_deref() else {
            continue;
        };
        let replacement = (1..probe_items.len()).find_map(|offset| {
            let candidate_index = (source_probe_index + offset) % probe_items.len();
            let candidate = &probe_items[candidate_index];
            (candidate.item.expected_answer != source_probe.item.expected_answer
                && candidate.prompt_tokens != source_probe.prompt_tokens)
                .then_some(candidate)
        });
        let Some(replacement) = replacement else {
            continue;
        };
        let Some(row) = ruliad_teacher_forced_row::<B>(&replacement.prompt_tokens, completion)
        else {
            continue;
        };
        swapped.push(row);
        swap_source_rows.push(source_row);
    }

    let matched_rows = matched.len();
    let mut rows = matched;
    rows.extend(swapped);
    let chunk_size = training
        .tbptt_chunk_size
        .unwrap_or(training.block_size)
        .max(1);
    let batch_rows = batch_rows
        .min(training.ruliad_probe_generation.max_batch_rows.max(1))
        .max(1);
    let scores = score_rows(model, &rows, chunk_size, batch_rows, device)?;
    Ok(summarize_scores(&scores, matched_rows, &swap_source_rows))
}

pub(super) fn emit_ruliad_teacher_forced_metrics(
    identity: RuliadProbeIdentity<'_>,
    metric_prefix: Option<&str>,
    evaluation: RuliadTeacherForcedEvaluation,
    bus: &TrainingEventBus,
) {
    let scope = metric_prefix.unwrap_or("Ruliad");
    for (name, value) in [
        ("Teacher Forced Items", evaluation.items as f64),
        (
            "Teacher Forced Completion Tokens",
            evaluation.completion_tokens as f64,
        ),
        ("Teacher Forced NLL", evaluation.mean_nll),
        ("Teacher Forced Sequence NLL", evaluation.mean_sequence_nll),
        ("Teacher Forced Token Accuracy", evaluation.token_accuracy),
        (
            "Teacher Forced First Token Accuracy",
            evaluation.first_token_accuracy,
        ),
        (
            "Teacher Forced Sequence Accuracy",
            evaluation.sequence_accuracy,
        ),
        (
            "Teacher Forced Context-Swap Items",
            evaluation.context_swap_items as f64,
        ),
        (
            "Teacher Forced Context-Swap NLL",
            evaluation.context_swap_mean_nll,
        ),
        (
            "Teacher Forced Context-Binding NLL Gain",
            evaluation.context_binding_nll_gain,
        ),
    ] {
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: identity.run_name.to_string().into(),
            split: TrainingMetricSplit::Valid,
            epoch: identity.epoch,
            step_in_epoch: 0,
            absolute_step: identity.absolute_step,
            name: format!("{scope} {name}"),
            value,
            running_value: value,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestBackend = burn_ndarray::NdArray<f32>;

    #[test]
    fn teacher_forced_rows_preserve_the_complete_prompt_and_answer() {
        let prompt = vec![1; 31];
        let answer = vec![2; 43];
        let row = ruliad_teacher_forced_row::<TestBackend>(&prompt, &answer).unwrap();
        assert_eq!(row.inputs.len(), 73);
        assert_eq!(&row.inputs[..31], &prompt);
        assert_eq!(row.mask.iter().filter(|mask| **mask > 0.0).count(), 43);
        assert_eq!(row.targets.last(), Some(&2));
        assert!(ruliad_teacher_forced_row::<TestBackend>(&[], &answer).is_none());
    }

    #[test]
    fn teacher_forced_recurrent_chunks_and_batching_match_full_context() {
        let device = Default::default();
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            DragonConfig {
                n_layer: 2,
                n_embd: 8,
                n_head: 1,
                mlp_internal_dim_multiplier: 2,
                vocab_size: 16,
                dropout: 0.0,
                ..Default::default()
            },
            &device,
        ));
        crate::train::model_identity::materialize_model_parameters::<TestBackend, _>(&model.model);
        let rows = vec![
            ruliad_teacher_forced_row::<TestBackend>(&[1, 2, 3, 4, 5, 6], &[7, 8, 9, 10]).unwrap(),
            ruliad_teacher_forced_row::<TestBackend>(&[11, 12], &[13, 14]).unwrap(),
        ];
        let reference = score_rows(&model, &rows, 32, 1, &device).unwrap();
        for chunk in [1, 2, 4, 32] {
            for batch in [1, 2] {
                let actual = score_rows(&model, &rows, chunk, batch, &device).unwrap();
                for (actual, expected) in actual.iter().zip(&reference) {
                    assert_eq!(actual.token_count, expected.token_count);
                    assert_eq!(actual.correct_tokens, expected.correct_tokens);
                    assert_eq!(actual.first_token_correct, expected.first_token_correct);
                    assert!(
                        (actual.nll_sum - expected.nll_sum).abs() < 1.0e-4,
                        "chunk={chunk} batch={batch}: {actual:?} versus {expected:?}"
                    );
                }
            }
        }
        assert!(score_rows(&model, &rows, 0, 1, &device).is_err());
        assert!(score_rows(&model, &rows, 1, 0, &device).is_err());
    }

    #[test]
    fn teacher_forced_summary_separates_context_binding_from_token_accuracy() {
        // Rows 0/1 are matched; rows 2/3 use the same targets under swapped prompts.
        let log_probabilities = [
            -0.1, -0.2, 0.0, -0.4, -0.3, 0.0, -1.1, -1.2, 0.0, -1.4, -1.3, 0.0,
        ];
        let predictions = [1, 2, 0, 4, 9, 0, 8, 8, 0, 8, 8, 0];
        let targets = [1, 2, 0, 4, 5, 0, 1, 2, 0, 4, 5, 0];
        let masks = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let summary = summarize_ruliad_teacher_forced_rows(
            &log_probabilities,
            &predictions,
            &targets,
            &masks,
            4,
            3,
            2,
            &[0, 1],
        );
        assert_eq!(summary.items, 2);
        assert_eq!(summary.completion_tokens, 4);
        assert!((summary.mean_nll - 0.25).abs() < 1.0e-6);
        assert_eq!(summary.version, 2);
        assert!((summary.mean_sequence_nll - 0.5).abs() < 1.0e-6);
        assert_eq!(summary.token_accuracy, 0.75);
        assert_eq!(summary.first_token_accuracy, 1.0);
        assert_eq!(summary.sequence_accuracy, 0.5);
        assert_eq!(summary.context_swap_items, 2);
        assert!((summary.context_swap_mean_nll - 1.25).abs() < 1.0e-6);
        assert!((summary.context_binding_nll_gain - 1.0).abs() < 1.0e-6);
    }
}
