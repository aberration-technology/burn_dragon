//! Free-running Ruliad generation and verifier evaluation.

use super::*;

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct RuliadModelEvaluation {
    pub report: burn_dragon_universality::RuliadEvalReport,
    pub item_count: usize,
    pub elapsed_ms: f64,
    pub generation_mean_batch_rows: f64,
    pub generation_maximum_batch_rows: usize,
    pub generation_maximum_in_flight_rows: usize,
    pub generation_batched_row_fraction: f64,
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_ruliad_model_free_run<B>(
    dataset: &Dataset,
    model: &LanguageTrainModel<B>,
    training: &TrainingHyperparameters,
    sampler_epoch: usize,
    sampler_step: usize,
    item_count: usize,
    training_batch_size: usize,
    dataset_name: &str,
    device: &B::Device,
) -> Result<Option<RuliadModelEvaluation>>
where
    B: burn::tensor::backend::Backend + Clone + 'static,
    B::Device: Clone,
{
    let probe_items =
        dataset.sample_ruliad_validation_probe_items(sampler_epoch, sampler_step, item_count);
    if probe_items.is_empty() {
        return Ok(None);
    }
    let evaluation = evaluate_ruliad_correctness_validation_for_items_core(
        None,
        None,
        dataset,
        model,
        sampler_epoch,
        sampler_step,
        device,
        training,
        &probe_items,
        training_batch_size,
        dataset_name,
        None,
        None,
        None,
        None,
        RuliadProbeDecodeMode::FreeRun,
        None,
    )?;
    Ok(Some(RuliadModelEvaluation {
        report: evaluation.report,
        item_count: evaluation.item_count,
        elapsed_ms: evaluation.elapsed_ms,
        generation_mean_batch_rows: evaluation.generation_stats.mean_batch_rows,
        generation_maximum_batch_rows: evaluation.generation_stats.maximum_batch_rows,
        generation_maximum_in_flight_rows: evaluation.generation_stats.maximum_in_flight_rows,
        generation_batched_row_fraction: ratio_usize(
            evaluation.generation_stats.batched_rows,
            evaluation.item_count,
        ),
    }))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_ruliad_correctness_validation_for_items<B>(
    run_name: &str,
    run_dir: &Path,
    dataset: &Dataset,
    model: &LanguageTrainModel<B>,
    epoch: usize,
    absolute_step: usize,
    device: &B::Device,
    training: &TrainingHyperparameters,
    probe_items: &[crate::dataset::RuliadValidationProbeItem],
    training_batch_size: usize,
    dataset_name: &str,
    probe_name: &str,
    metric_prefix: Option<&str>,
    output_degeneracy: Option<&crate::train::steps::OutputDegeneracyStats>,
    bus: &TrainingEventBus,
    decode_mode: RuliadProbeDecodeMode,
) -> Result<burn_dragon_universality::RuliadEvalReport>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    Ok(evaluate_ruliad_correctness_validation_for_items_core(
        Some(run_name),
        Some(run_dir),
        dataset,
        model,
        epoch,
        absolute_step,
        device,
        training,
        probe_items,
        training_batch_size,
        dataset_name,
        Some(probe_name),
        metric_prefix,
        output_degeneracy,
        Some(bus),
        decode_mode,
        None,
    )?
    .report)
}

pub(super) struct RuliadCorrectnessEvaluation {
    pub(super) report: burn_dragon_universality::RuliadEvalReport,
    pub(super) item_count: usize,
    pub(super) elapsed_ms: f64,
    pub(super) generation_stats: RuliadProbeGenerationStats,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_ruliad_correctness_validation_for_items_core<B>(
    run_name: Option<&str>,
    run_dir: Option<&Path>,
    dataset: &Dataset,
    model: &LanguageTrainModel<B>,
    epoch: usize,
    absolute_step: usize,
    device: &B::Device,
    training: &TrainingHyperparameters,
    probe_items: &[crate::dataset::RuliadValidationProbeItem],
    training_batch_size: usize,
    dataset_name: &str,
    probe_name: Option<&str>,
    metric_prefix: Option<&str>,
    output_degeneracy: Option<&crate::train::steps::OutputDegeneracyStats>,
    bus: Option<&TrainingEventBus>,
    decode_mode: RuliadProbeDecodeMode,
    context_router: Option<&crate::train::PredictiveContextValidationRouter<B>>,
) -> Result<RuliadCorrectnessEvaluation>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    let probe_started = burn_dragon_time::Instant::now();
    let mut items = Vec::with_capacity(probe_items.len());
    let mut completions = Vec::with_capacity(probe_items.len());
    let mut generated_token_rows = Vec::with_capacity(probe_items.len());
    let mut generation_budgets = probe_items
        .iter()
        .map(|probe| ruliad_probe_generation_budget(dataset, &probe.item, training))
        .collect::<Vec<_>>();
    let close_token_id = dataset.ruliad_document_end_token_id().map(i64::from);
    if let (Some(run_name), Some(probe_name)) = (run_name, probe_name) {
        eprintln!(
            "ruliad correctness probe start run={run_name} epoch={epoch} probe={probe_name} items={} grouped_batching={} max_batch_rows={} minimum_batch_rows={} maximum_prompt_position_span={} device_buffer_tokens={}",
            probe_items.len(),
            training.ruliad_probe_generation.enabled
                && decode_mode == RuliadProbeDecodeMode::FreeRun,
            training.ruliad_probe_generation.max_batch_rows,
            training.ruliad_probe_generation.minimum_batch_rows,
            training
                .ruliad_probe_generation
                .maximum_prompt_position_span,
            training.ruliad_probe_generation.device_buffer_tokens,
        );
    }

    let generator = RuliadProbeGenerator {
        dataset,
        model,
        training,
        device,
        close_token_id,
        decode_mode,
        context_router,
    };
    let (generated_rows, generation_stats) = generate_ruliad_probe_rows(
        &generator,
        probe_items,
        &generation_budgets,
        training_batch_size,
    )?;
    let generation_mode = match (
        generation_stats.batched_rows > 0,
        generation_stats.independent_rows > 0,
    ) {
        (true, false) => "ragged_recurrent_batched_greedy",
        (true, true) => "hybrid_ragged_recurrent_batched_greedy",
        (false, true) if generation_stats.maximum_in_flight_rows > 1 => {
            "bounded_parallel_independent_recurrent"
        }
        _ => "independent_recurrent",
    };

    for (probe_index, (probe, generated_tokens)) in
        probe_items.iter().cloned().zip(generated_rows).enumerate()
    {
        let max_new_tokens = generation_budgets[probe_index].max_new_tokens;
        generation_budgets[probe_index].generation_hit_budget = generated_tokens.len()
            >= max_new_tokens
            && close_token_id.is_none_or(|stop| !generated_tokens.contains(&stop));
        let completion = dataset
            .decode_ruliad_payload_tokens(&generated_tokens, true)
            .unwrap_or_else(|| dataset.decode(&generated_tokens));
        let completion = canonicalize_ruliad_completion_close_marker(
            completion,
            probe.item.document_close_marker(),
        );
        generated_token_rows.push(generated_tokens);
        completions.push(burn_dragon_universality::RuliadCompletionRecord {
            oracle_hash: probe.item.oracle_hash.clone(),
            completion,
        });
        items.push(probe.item);
    }

    let report = burn_dragon_universality::evaluate_completions(dataset_name, &items, &completions);
    let elapsed_ms = probe_started.elapsed().as_millis() as f64;
    if let (Some(run_name), Some(run_dir), Some(probe_name), Some(bus)) =
        (run_name, run_dir, probe_name, bus)
    {
        let schema_alignment = ruliad_answer_schema_alignment_summary(&items, &completions);
        let completion_degeneracy =
            ruliad_completion_degeneracy_summary(&generated_token_rows, close_token_id);
        let examples = ruliad_probe_examples(
            &items,
            &completions,
            training.events.capability_probe_example_count,
        );
        write_ruliad_completion_probe_records(
            run_dir,
            RuliadProbeIdentity {
                run_name,
                epoch,
                absolute_step,
                probe_name,
            },
            &items,
            &completions,
            &generated_token_rows,
            &generation_budgets,
            close_token_id,
        )?;
        emit_ruliad_correctness_metrics_with_labels(RuliadCorrectnessMetrics {
            identity: RuliadProbeIdentity {
                run_name,
                epoch,
                absolute_step,
                probe_name,
            },
            report: &report,
            bus,
            metric_prefix,
            output_degeneracy,
            examples: &examples,
            schema_alignment,
            completion_degeneracy,
            generation_budget: ruliad_probe_generation_budget_summary(&generation_budgets),
        });
        let metric_scope = metric_prefix.unwrap_or("Ruliad");
        for (name, value) in [
            (format!("{metric_scope} Probe Elapsed MS"), elapsed_ms),
            (
                format!("{metric_scope} Probe Items Per Second"),
                if elapsed_ms > 0.0 {
                    items.len() as f64 * 1_000.0 / elapsed_ms
                } else {
                    0.0
                },
            ),
            (
                format!("{metric_scope} Probe Generation Mean Batch Rows"),
                generation_stats.mean_batch_rows,
            ),
            (
                format!("{metric_scope} Probe Generation Maximum Batch Rows"),
                generation_stats.maximum_batch_rows as f64,
            ),
            (
                format!("{metric_scope} Probe Generation In Flight Rows"),
                generation_stats.maximum_in_flight_rows as f64,
            ),
            (
                format!("{metric_scope} Probe Generation Mean Prompt Position Span"),
                generation_stats.mean_batch_prompt_position_span,
            ),
            (
                format!("{metric_scope} Probe Generation Maximum Prompt Position Span"),
                generation_stats.maximum_batch_prompt_position_span as f64,
            ),
            (
                format!("{metric_scope} Probe Generation Batched Row Fraction"),
                ratio_usize(generation_stats.batched_rows, probe_items.len()),
            ),
            (
                format!("{metric_scope} Probe Prompt Position Groups"),
                generation_stats.prompt_position_groups as f64,
            ),
            (
                format!("{metric_scope} Probe Largest Prompt Position Group"),
                generation_stats.largest_prompt_position_group as f64,
            ),
            (
                format!("{metric_scope} Probe Generation Device Buffer Tokens"),
                generation_stats.device_buffer_tokens as f64,
            ),
            (
                format!("{metric_scope} Probe Generation Prefill Forward MS"),
                generation_stats.profile.prefill_forward_ns as f64 / 1_000_000.0,
            ),
            (
                format!("{metric_scope} Probe Generation Token Forward MS"),
                generation_stats.profile.token_forward_ns as f64 / 1_000_000.0,
            ),
            (
                format!("{metric_scope} Probe Generation Host Transfer MS"),
                generation_stats.profile.sample_host_transfer_ns as f64 / 1_000_000.0,
            ),
            (
                format!("{metric_scope} Probe Generation Host Sync Points"),
                generation_stats.profile.host_sync_points as f64,
            ),
            (
                format!("{metric_scope} Probe Generation Token Steps"),
                generation_stats.profile.token_steps as f64,
            ),
        ] {
            let _ = bus.send_metric_sample(TrainingMetricSample {
                run_id: run_name.to_string().into(),
                split: TrainingMetricSplit::Valid,
                epoch,
                step_in_epoch: 0,
                absolute_step,
                name,
                value,
                running_value: value,
            });
        }
        eprintln!(
            "ruliad correctness probe complete run={run_name} epoch={epoch} probe={probe_name} items={} generation_mode={generation_mode} mean_batch_rows={:.2} max_batch_rows={} mean_prompt_position_span={:.2} max_prompt_position_span={} batched_row_fraction={:.3} prompt_position_groups={} largest_prompt_position_group={} maximum_in_flight_rows={} prefill_ms={:.0} token_forward_ms={:.0} host_transfer_ms={:.0} host_sync_points={} elapsed_ms={elapsed_ms:.0}",
            items.len(),
            generation_stats.mean_batch_rows,
            generation_stats.maximum_batch_rows,
            generation_stats.mean_batch_prompt_position_span,
            generation_stats.maximum_batch_prompt_position_span,
            ratio_usize(generation_stats.batched_rows, probe_items.len()),
            generation_stats.prompt_position_groups,
            generation_stats.largest_prompt_position_group,
            generation_stats.maximum_in_flight_rows,
            generation_stats.profile.prefill_forward_ns as f64 / 1_000_000.0,
            generation_stats.profile.token_forward_ns as f64 / 1_000_000.0,
            generation_stats.profile.sample_host_transfer_ns as f64 / 1_000_000.0,
            generation_stats.profile.host_sync_points,
        );
    }
    Ok(RuliadCorrectnessEvaluation {
        report,
        item_count: items.len(),
        elapsed_ms,
        generation_stats,
    })
}

#[derive(Clone, Debug)]
pub(super) struct RuliadAnswerFieldRange {
    pub(super) key: String,
    pub(super) value: String,
    pub(super) start: usize,
    pub(super) end: usize,
}

pub(super) fn ruliad_prompt_schema_completion_tokens<B>(
    dataset: &Dataset,
    model: &DragonModel<B>,
    prompt_tokens: Vec<i64>,
    prompt: &str,
    max_new_tokens: usize,
    device: &B::Device,
) -> Option<Vec<i64>>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    if prompt_tokens.is_empty() || max_new_tokens == 0 {
        return None;
    }
    let keys = ruliad_prompt_answer_keys(prompt)?;
    if keys.is_empty() {
        return None;
    }
    let value_tokens = ruliad_prompt_schema_value_token_ids(dataset);
    if value_tokens.is_empty() {
        return None;
    }
    let close_marker = if prompt.trim_start().starts_with("[R3") {
        burn_dragon_universality::ruliad::RULIAD_V3_DOCUMENT_CLOSE_MARKER
    } else {
        burn_dragon_universality::ruliad::RULIAD_V2_DOCUMENT_CLOSE_MARKER
    };
    let mut decoder = PromptSchemaDecoder::new(keys, value_tokens, close_marker);
    let (mut state, mut last_logits) =
        crate::generation::prefill_state(model, &prompt_tokens, device).ok()?;
    let mut generated = Vec::with_capacity(max_new_tokens);
    for _ in 0..max_new_tokens {
        let allowed = decoder.allowed_tokens(dataset)?;
        if allowed.is_empty() {
            break;
        }
        let token = ruliad_argmax_allowed_token(last_logits.clone(), &allowed)?;
        generated.push(token);
        if decoder.observe_token(dataset, token).is_none() {
            break;
        }
        if decoder.finished {
            break;
        }
        if Some(token) == dataset.ruliad_document_end_token_id().map(i64::from) {
            break;
        }
        let next = Tensor::<B, 2, Int>::from_data(TensorData::new(vec![token], [1, 1]), device);
        let logits = model.forward_with_state(next, &mut state);
        let [_, time, vocab] = logits.shape().dims::<3>();
        if time == 0 || vocab == 0 {
            break;
        }
        last_logits = logits.slice_dim(1, (time - 1)..time).reshape([vocab]);
    }
    Some(generated)
}

#[derive(Clone, Debug)]
pub(super) enum PromptSchemaPhase {
    Key { prefix: Vec<char>, offset: usize },
    Value { len: usize },
    Close { suffix: Vec<char>, offset: usize },
}

#[derive(Clone, Debug)]
pub(super) struct PromptSchemaDecoder {
    pub(super) keys: Vec<String>,
    pub(super) field_index: usize,
    pub(super) value_tokens: Vec<i64>,
    pub(super) close_marker: &'static str,
    pub(super) phase: PromptSchemaPhase,
    pub(super) finished: bool,
}

impl PromptSchemaDecoder {
    fn new(keys: Vec<String>, value_tokens: Vec<i64>, close_marker: &'static str) -> Self {
        let prefix = ruliad_prompt_schema_key_prefix(&keys[0]);
        Self {
            keys,
            field_index: 0,
            value_tokens,
            close_marker,
            phase: PromptSchemaPhase::Key { prefix, offset: 0 },
            finished: false,
        }
    }

    fn allowed_tokens(&self, dataset: &Dataset) -> Option<Vec<i64>> {
        if self.finished {
            return Some(Vec::new());
        }
        match &self.phase {
            PromptSchemaPhase::Key { prefix, offset } => {
                let ch = *prefix.get(*offset)?;
                ruliad_single_char_token(dataset, ch).map(|token| vec![token])
            }
            PromptSchemaPhase::Value { len } => {
                let mut allowed = self.value_tokens.clone();
                if *len > 0 {
                    let separator = if self.field_index + 1 < self.keys.len() {
                        ';'
                    } else {
                        '\n'
                    };
                    if let Some(token) = ruliad_single_char_token(dataset, separator) {
                        allowed.push(token);
                    }
                }
                allowed.sort_unstable();
                allowed.dedup();
                Some(allowed)
            }
            PromptSchemaPhase::Close { suffix, offset } => {
                let ch = *suffix.get(*offset)?;
                ruliad_single_char_token(dataset, ch).map(|token| vec![token])
            }
        }
    }

    fn observe_token(&mut self, dataset: &Dataset, token: i64) -> Option<()> {
        let ch = ruliad_token_single_char(dataset, token)?;
        match &mut self.phase {
            PromptSchemaPhase::Key { prefix, offset } => {
                if prefix.get(*offset).copied()? != ch {
                    return None;
                }
                *offset += 1;
                if *offset >= prefix.len() {
                    self.phase = PromptSchemaPhase::Value { len: 0 };
                }
            }
            PromptSchemaPhase::Value { len } => {
                if self.field_index + 1 < self.keys.len() && ch == ';' && *len > 0 {
                    self.field_index += 1;
                    self.phase = PromptSchemaPhase::Key {
                        prefix: ruliad_prompt_schema_key_prefix(&self.keys[self.field_index]),
                        offset: 0,
                    };
                } else if self.field_index + 1 == self.keys.len() && ch == '\n' && *len > 0 {
                    self.phase = PromptSchemaPhase::Close {
                        suffix: self.close_marker.chars().collect(),
                        offset: 0,
                    };
                } else {
                    *len = len.saturating_add(1);
                }
            }
            PromptSchemaPhase::Close { suffix, offset } => {
                if suffix.get(*offset).copied()? != ch {
                    return None;
                }
                *offset += 1;
                if *offset >= suffix.len() {
                    self.finished = true;
                }
            }
        }
        Some(())
    }
}

pub(super) fn ruliad_prompt_schema_key_prefix(key: &str) -> Vec<char> {
    format!("{key}=").chars().collect()
}

pub(super) fn ruliad_prompt_answer_keys(prompt: &str) -> Option<Vec<String>> {
    let answer_line = prompt
        .lines()
        .find_map(|line| line.strip_prefix("A:"))
        .or_else(|| prompt.split('\n').find_map(|line| line.strip_prefix("A:")))?;
    let keys = answer_line
        .split(',')
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    (!keys.is_empty()).then_some(keys)
}

pub(super) fn ruliad_prompt_schema_value_token_ids(dataset: &Dataset) -> Vec<i64> {
    let mut chars = Vec::new();
    chars.extend('0'..='9');
    chars.extend('a'..='z');
    chars.extend('A'..='Z');
    chars.extend([',', '-', '+', '.']);
    let mut tokens = chars
        .into_iter()
        .filter_map(|ch| ruliad_single_char_token(dataset, ch))
        .collect::<Vec<_>>();
    tokens.sort_unstable();
    tokens.dedup();
    tokens
}

pub(super) fn ruliad_token_single_char(dataset: &Dataset, token: i64) -> Option<char> {
    u32::try_from(token).ok()?;
    let decoded = dataset.decode_ruliad_payload_tokens(&[token], false)?;
    let mut chars = decoded.chars();
    let ch = chars.next()?;
    chars.next().is_none().then_some(ch)
}

pub(super) fn ruliad_fixed_contract_completion_tokens<B>(
    dataset: &Dataset,
    model: &DragonModel<B>,
    prompt_tokens: Vec<i64>,
    expected_answer: &str,
    close_marker: &str,
    max_new_tokens: usize,
    device: &B::Device,
) -> Option<Vec<i64>>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    if prompt_tokens.is_empty() || max_new_tokens == 0 {
        return None;
    }
    let allowed = ruliad_fixed_contract_allowed_tokens(
        dataset,
        expected_answer,
        close_marker,
        max_new_tokens,
    )?;
    if allowed.is_empty() {
        return None;
    }
    let (mut state, mut last_logits) =
        crate::generation::prefill_state(model, &prompt_tokens, device).ok()?;
    let mut generated = Vec::with_capacity(allowed.len());
    for allowed_tokens in allowed {
        let token = ruliad_argmax_allowed_token(last_logits.clone(), &allowed_tokens)?;
        generated.push(token);
        if Some(token) == dataset.ruliad_document_end_token_id().map(i64::from) {
            break;
        }
        let next = Tensor::<B, 2, Int>::from_data(TensorData::new(vec![token], [1, 1]), device);
        let logits = model.forward_with_state(next, &mut state);
        let [_, time, vocab] = logits.shape().dims::<3>();
        if time == 0 || vocab == 0 {
            break;
        }
        last_logits = logits.slice_dim(1, (time - 1)..time).reshape([vocab]);
    }
    Some(generated)
}

pub(super) fn ruliad_argmax_allowed_token<B>(logits: Tensor<B, 1>, allowed: &[i64]) -> Option<i64>
where
    B: BackendTrait,
{
    let values = logits.to_data().convert::<f32>().into_vec::<f32>().ok()?;
    if values.is_empty() {
        return None;
    }
    let mut best = None::<(i64, f32)>;
    for token in allowed.iter().copied() {
        let index = usize::try_from(token).ok()?;
        let value = *values.get(index)?;
        if !value.is_finite() {
            continue;
        }
        if best.is_none_or(|(_, best_value)| value > best_value) {
            best = Some((token, value));
        }
    }
    best.map(|(token, _)| token)
}

pub(super) fn ruliad_fixed_contract_allowed_tokens(
    dataset: &Dataset,
    expected_answer: &str,
    close_marker: &str,
    max_new_tokens: usize,
) -> Option<Vec<Vec<i64>>> {
    let answer = expected_answer.trim();
    if answer.is_empty() || max_new_tokens == 0 {
        return None;
    }
    let field_ranges = ruliad_answer_field_ranges(answer);
    let uppercase_alphabet = ruliad_answer_uppercase_alphabet(&field_ranges);
    let mut allowed = Vec::<Vec<i64>>::new();
    for (byte_index, ch) in answer.char_indices() {
        let field = field_ranges
            .iter()
            .find(|field| byte_index >= field.start && byte_index < field.end);
        let chars = field
            .map(|field| ruliad_contract_value_allowed_chars(field, ch, &uppercase_alphabet))
            .unwrap_or_else(|| vec![ch]);
        let mut token_ids = chars
            .into_iter()
            .filter_map(|candidate| ruliad_single_char_token(dataset, candidate))
            .collect::<Vec<_>>();
        token_ids.sort_unstable();
        token_ids.dedup();
        if token_ids.is_empty() {
            return None;
        }
        allowed.push(token_ids);
        if allowed.len() >= max_new_tokens {
            return Some(allowed);
        }
    }
    for token in dataset.encode_ruliad_payload_tokens(&format!("\n{close_marker}"))? {
        allowed.push(vec![i64::from(token)]);
        if allowed.len() >= max_new_tokens {
            break;
        }
    }
    Some(allowed)
}

pub(super) fn ruliad_single_char_token(dataset: &Dataset, ch: char) -> Option<i64> {
    let mut text = String::new();
    text.push(ch);
    let tokens = dataset.encode_ruliad_payload_tokens(&text)?;
    match tokens.as_slice() {
        [token] => Some(i64::from(*token)),
        _ => None,
    }
}

pub(super) fn ruliad_answer_field_ranges(answer: &str) -> Vec<RuliadAnswerFieldRange> {
    let mut ranges = Vec::new();
    let mut field_start = 0usize;
    while field_start < answer.len() {
        let field_end = answer[field_start..]
            .find(';')
            .map(|offset| field_start + offset)
            .unwrap_or(answer.len());
        let field = &answer[field_start..field_end];
        if let Some(eq_offset) = field.find('=') {
            let key = field[..eq_offset].trim().to_ascii_lowercase();
            let value_start = field_start + eq_offset + 1;
            if !key.is_empty() && value_start < field_end {
                ranges.push(RuliadAnswerFieldRange {
                    key,
                    value: answer[value_start..field_end].to_string(),
                    start: value_start,
                    end: field_end,
                });
            }
        }
        field_start = field_end.saturating_add(1);
    }
    ranges
}

pub(super) fn ruliad_answer_uppercase_alphabet(fields: &[RuliadAnswerFieldRange]) -> Vec<char> {
    let mut alphabet = fields
        .iter()
        .find(|field| field.key.ends_with("alpha"))
        .map(|field| {
            field
                .value
                .chars()
                .filter(char::is_ascii_uppercase)
                .collect()
        })
        .unwrap_or_else(|| {
            fields
                .iter()
                .flat_map(|field| field.value.chars())
                .filter(char::is_ascii_uppercase)
                .collect::<Vec<_>>()
        });
    if alphabet.is_empty() {
        alphabet.extend(['A', 'B', 'C']);
    }
    alphabet.sort_unstable();
    alphabet.dedup();
    alphabet
}

pub(super) fn ruliad_contract_value_allowed_chars(
    field: &RuliadAnswerFieldRange,
    ch: char,
    uppercase_alphabet: &[char],
) -> Vec<char> {
    if ch == ',' {
        return vec![ch];
    }
    if ch.is_ascii_digit() {
        if matches!(field.key.as_str(), "ok" | "acc")
            || (field.key.ends_with("edge")
                && field
                    .value
                    .chars()
                    .all(|candidate| matches!(candidate, '0' | '1')))
        {
            return vec!['0', '1'];
        }
        return ('0'..='9').collect();
    }
    if ch.is_ascii_uppercase() {
        return uppercase_alphabet
            .iter()
            .map(|candidate| candidate.to_ascii_lowercase())
            .collect();
    }
    vec![ch]
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct RuliadAnswerSchemaAlignmentSummary {
    pub(super) key_match_rate: f64,
    pub(super) mean_key_overlap: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct RuliadCompletionDegeneracySummary {
    pub(super) sequence_count: usize,
    pub(super) token_count: usize,
    pub(super) repetition_fraction: f64,
    pub(super) distinct_1_fraction: f64,
    pub(super) distinct_2_fraction: f64,
    pub(super) max_period_2_to_16_fraction: f64,
    pub(super) max_period_2_to_64_fraction: f64,
    pub(super) dominant_period_2_to_64: usize,
}

pub(super) fn ruliad_answer_schema_alignment_summary(
    items: &[burn_dragon_universality::RuliadEvalItem],
    completions: &[burn_dragon_universality::RuliadCompletionRecord],
) -> RuliadAnswerSchemaAlignmentSummary {
    if items.is_empty() {
        return RuliadAnswerSchemaAlignmentSummary::default();
    }
    let completion_by_hash = completions
        .iter()
        .map(|completion| {
            (
                completion.oracle_hash.as_str(),
                completion.completion.as_str(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut exact_matches = 0usize;
    let mut overlap_ppm_sum = 0usize;
    for item in items {
        let answer = completion_by_hash
            .get(item.oracle_hash.as_str())
            .copied()
            .and_then(burn_dragon_universality::ruliad::extract_ruliad_answer);
        let alignment = burn_dragon_universality::ruliad::ruliad_answer_key_alignment(
            &item.expected_answer,
            answer.as_deref(),
        );
        exact_matches += usize::from(alignment.exact_key_match);
        overlap_ppm_sum = overlap_ppm_sum.saturating_add(alignment.overlap_ppm);
    }
    let item_count = items.len().max(1) as f64;
    RuliadAnswerSchemaAlignmentSummary {
        key_match_rate: exact_matches as f64 / item_count,
        mean_key_overlap: overlap_ppm_sum as f64 / (item_count * 1_000_000.0),
    }
}

pub(super) fn ruliad_completion_degeneracy_summary(
    completions: &[Vec<i64>],
    stop_on_token: Option<i64>,
) -> Option<RuliadCompletionDegeneracySummary> {
    let trimmed_rows = completions
        .iter()
        .map(|tokens| ruliad_completion_tokens_until_stop(tokens, stop_on_token))
        .filter(|tokens| !tokens.is_empty())
        .collect::<Vec<_>>();
    let rows = trimmed_rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
    if rows.is_empty() {
        return None;
    }
    let token_count = rows.iter().map(|tokens| tokens.len()).sum::<usize>();
    Some(RuliadCompletionDegeneracySummary {
        sequence_count: rows.len(),
        token_count,
        repetition_fraction: repeated_token_fraction(&rows),
        distinct_1_fraction: row_weighted_distinct_n_fraction(&rows, 1),
        distinct_2_fraction: row_weighted_distinct_n_fraction(&rows, 2),
        max_period_2_to_16_fraction: row_weighted_max_period_fraction(&rows, 2..=16).1,
        max_period_2_to_64_fraction: row_weighted_max_period_fraction(&rows, 2..=64).1,
        dominant_period_2_to_64: row_weighted_max_period_fraction(&rows, 2..=64).0,
    })
}

pub(super) fn ruliad_completion_tokens_until_stop(
    tokens: &[i64],
    stop_on_token: Option<i64>,
) -> Vec<i64> {
    let Some(stop) = stop_on_token else {
        return tokens.to_vec();
    };
    match tokens.iter().position(|token| *token == stop) {
        Some(index) => tokens[..=index].to_vec(),
        None => tokens.to_vec(),
    }
}

pub(super) fn repeated_token_fraction(rows: &[&[i64]]) -> f64 {
    let mut repeats = 0usize;
    let mut comparisons = 0usize;
    for tokens in rows {
        for pair in tokens.windows(2) {
            comparisons = comparisons.saturating_add(1);
            repeats += usize::from(pair[0] == pair[1]);
        }
    }
    ratio_usize(repeats, comparisons)
}

pub(super) fn row_weighted_distinct_n_fraction(rows: &[&[i64]], n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let mut distinct_sum = 0usize;
    let mut window_sum = 0usize;
    for tokens in rows.iter().copied().filter(|tokens| tokens.len() >= n) {
        let window_count = tokens.len() + 1 - n;
        window_sum = window_sum.saturating_add(window_count);
        distinct_sum = distinct_sum.saturating_add(
            tokens
                .windows(n)
                .map(|window| window.to_vec())
                .collect::<HashSet<_>>()
                .len(),
        );
    }
    ratio_usize(distinct_sum, window_sum)
}

pub(super) fn row_weighted_max_period_fraction(
    rows: &[&[i64]],
    periods: impl IntoIterator<Item = usize>,
) -> (usize, f64) {
    periods
        .into_iter()
        .map(|period| (period, row_weighted_period_fraction(rows, period)))
        .max_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or((0, 0.0))
}

pub(super) fn row_weighted_period_fraction(rows: &[&[i64]], period: usize) -> f64 {
    if period == 0 {
        return 0.0;
    }
    let mut matches = 0usize;
    let mut comparisons = 0usize;
    for tokens in rows
        .iter()
        .copied()
        .filter(|tokens| tokens.len() >= period.saturating_mul(2))
    {
        comparisons = comparisons.saturating_add(tokens.len() - period);
        matches = matches.saturating_add(
            (period..tokens.len())
                .filter(|idx| tokens[*idx] == tokens[*idx - period])
                .count(),
        );
    }
    ratio_usize(matches, comparisons)
}

pub(super) fn ruliad_probe_examples(
    items: &[burn_dragon_universality::RuliadEvalItem],
    completions: &[burn_dragon_universality::RuliadCompletionRecord],
    limit: usize,
) -> Vec<CapabilityProbeExample> {
    if limit == 0 {
        return Vec::new();
    }
    let completion_by_hash = completions
        .iter()
        .map(|completion| {
            (
                completion.oracle_hash.as_str(),
                completion.completion.as_str(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut examples = Vec::with_capacity(limit.min(items.len()));
    for item in items {
        let completion = completion_by_hash.get(item.oracle_hash.as_str()).copied();
        let score =
            burn_dragon_universality::ruliad::score_ruliad_item_completion(item, completion);
        if score.verifier_match() {
            continue;
        }
        let extracted = completion.map(burn_dragon_universality::ruliad::extract_ruliad_completion);
        let actual = extracted
            .as_ref()
            .and_then(|completion| completion.answer.clone());
        examples.push(CapabilityProbeExample {
            label: format!("{}:{}", item.family, item.task_kind),
            prompt: compact_probe_example_text(&item.prompt, 512),
            expected: compact_probe_example_text(&item.expected_answer, 256),
            actual: actual.map(|answer| compact_probe_example_text(&answer, 256)),
            completion: compact_probe_example_text(completion.unwrap_or_default(), 512),
            status: format!("{:?}", score.status),
            reason: if completion.is_none() {
                "missing_completion".to_string()
            } else if extracted
                .as_ref()
                .is_none_or(|completion| completion.answer.is_none())
            {
                "malformed_completion".to_string()
            } else {
                "answer_mismatch".to_string()
            },
            generated_tokens: extracted
                .as_ref()
                .map(|completion| completion.generated_token_count)
                .unwrap_or_default(),
        });
        if examples.len() >= limit {
            break;
        }
    }
    examples
}

#[derive(Debug, Serialize)]
pub(super) struct RuliadCompletionProbeRecord {
    pub(super) version: u32,
    pub(super) run_id: String,
    pub(super) epoch: usize,
    pub(super) absolute_step: usize,
    pub(super) probe_name: String,
    pub(super) sample_index: usize,
    pub(super) oracle_hash: String,
    pub(super) split: String,
    pub(super) family: String,
    pub(super) task_kind: String,
    pub(super) difficulty_level: Option<usize>,
    pub(super) math_domains: Vec<String>,
    pub(super) reasoning_modes: Vec<String>,
    pub(super) prompt: String,
    pub(super) expected_answer: String,
    pub(super) completion: String,
    pub(super) actual_answer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) presented_action_match: Option<bool>,
    pub(super) status: String,
    pub(super) verifier_match: bool,
    pub(super) semantic_match: bool,
    pub(super) partial_credit: bool,
    pub(super) partial_progress_ppm: usize,
    pub(super) correct_field_count: usize,
    pub(super) expected_field_count: usize,
    pub(super) certificate_prefix_ppm: usize,
    pub(super) completion_quality_ppm: usize,
    /// Legacy evaluator count based on whitespace-delimited payload segments.
    pub(super) generated_token_count: usize,
    pub(super) generated_model_token_count: usize,
    pub(super) generation_budget: usize,
    pub(super) minimum_answer_tokens: usize,
    pub(super) budget_sufficient: bool,
    pub(super) generation_hit_budget: bool,
    pub(super) answer_terminated: bool,
    pub(super) hash_canary: bool,
}

#[derive(Clone, Copy)]
pub(super) struct RuliadProbeIdentity<'a> {
    pub(super) run_name: &'a str,
    pub(super) epoch: usize,
    pub(super) absolute_step: usize,
    pub(super) probe_name: &'a str,
}

pub(super) fn write_ruliad_completion_probe_records(
    run_dir: &Path,
    identity: RuliadProbeIdentity<'_>,
    items: &[burn_dragon_universality::RuliadEvalItem],
    completions: &[burn_dragon_universality::RuliadCompletionRecord],
    generated_token_rows: &[Vec<i64>],
    generation_budgets: &[RuliadProbeGenerationBudget],
    stop_on_token: Option<i64>,
) -> Result<()> {
    let RuliadProbeIdentity {
        run_name,
        epoch,
        absolute_step,
        probe_name,
    } = identity;
    if items.is_empty() || completions.is_empty() {
        return Ok(());
    }
    let events_dir = run_dir.join("events");
    fs::create_dir_all(&events_dir)
        .with_context(|| format!("failed to create events directory {}", events_dir.display()))?;
    let path = events_dir.join("ruliad_completion_samples.jsonl");
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("opening {}", path.display()))?;
    for (((item, completion), generated_tokens), generation_budget) in items
        .iter()
        .zip(completions.iter())
        .zip(generated_token_rows.iter())
        .zip(generation_budgets.iter())
    {
        let score = burn_dragon_universality::ruliad::score_ruliad_item_completion(
            item,
            Some(&completion.completion),
        );
        let extracted =
            burn_dragon_universality::ruliad::extract_ruliad_completion(&completion.completion);
        let presented_action_match =
            burn_dragon_universality::ruliad::ruliad_presented_action_match(
                item,
                extracted.answer.as_deref(),
            );
        let record = RuliadCompletionProbeRecord {
            version: 3,
            run_id: run_name.to_string(),
            epoch,
            absolute_step,
            probe_name: probe_name.to_string(),
            sample_index: item.sample_index,
            oracle_hash: item.oracle_hash.clone(),
            split: format!("{:?}", item.split),
            family: item.family.clone(),
            task_kind: item.task_kind.clone(),
            difficulty_level: item.difficulty_level,
            math_domains: item.math_domains.clone(),
            reasoning_modes: item.reasoning_modes.clone(),
            prompt: item.prompt.clone(),
            expected_answer: item.expected_answer.clone(),
            completion: completion.completion.clone(),
            actual_answer: extracted.answer,
            presented_action_match,
            status: format!("{:?}", score.status),
            verifier_match: score.verifier_match(),
            semantic_match: matches!(
                score.status,
                burn_dragon_universality::ruliad::RuliadAnswerStatus::VerifierMatch
                    | burn_dragon_universality::ruliad::RuliadAnswerStatus::SemanticMatch
            ),
            partial_credit: score.partial_credit(),
            partial_progress_ppm: score.partial_progress_ppm,
            correct_field_count: score.correct_field_count,
            expected_field_count: score.expected_field_count,
            certificate_prefix_ppm: score.certificate_prefix_ppm,
            completion_quality_ppm: score.completion_quality_ppm,
            generated_token_count: score.generated_token_count,
            generated_model_token_count: ruliad_completion_tokens_until_stop(
                generated_tokens,
                stop_on_token,
            )
            .len(),
            generation_budget: generation_budget.max_new_tokens,
            minimum_answer_tokens: generation_budget.minimum_answer_tokens,
            budget_sufficient: generation_budget.budget_sufficient,
            generation_hit_budget: generation_budget.generation_hit_budget,
            answer_terminated: score.answer_terminated,
            hash_canary: score.hash_canary,
        };
        let line =
            serde_json::to_string(&record).context("serializing ruliad completion probe record")?;
        writeln!(file, "{line}").with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}

pub(super) fn compact_probe_example_text(text: &str, max_chars: usize) -> String {
    let mut compact = text.replace('\r', "\\r").replace('\n', "\\n");
    let char_count = compact.chars().count();
    if char_count <= max_chars {
        return compact;
    }
    let keep = max_chars.saturating_sub(3);
    compact = compact.chars().take(keep).collect();
    compact.push_str("...");
    compact
}

pub(super) fn canonicalize_ruliad_completion_close_marker(
    mut completion: String,
    expected_close_marker: &str,
) -> String {
    if expected_close_marker == "[/R3]" {
        completion = completion.replacen("[/R2]", "[/R3]", 1);
    } else if expected_close_marker == "[/R2]" {
        completion = completion.replacen("[/R3]", "[/R2]", 1);
    }
    completion
}

pub(super) fn emit_ruliad_correctness_metrics(
    run_name: &str,
    epoch: usize,
    absolute_step: usize,
    report: &burn_dragon_universality::RuliadEvalReport,
    bus: &TrainingEventBus,
) {
    emit_ruliad_correctness_metrics_with_labels(RuliadCorrectnessMetrics {
        identity: RuliadProbeIdentity {
            run_name,
            epoch,
            absolute_step,
            probe_name: "ruliad_correctness",
        },
        report,
        bus,
        metric_prefix: None,
        output_degeneracy: None,
        examples: &[],
        schema_alignment: RuliadAnswerSchemaAlignmentSummary::default(),
        completion_degeneracy: None,
        generation_budget: None,
    });
}

pub(super) struct RuliadCorrectnessMetrics<'a> {
    pub(super) identity: RuliadProbeIdentity<'a>,
    pub(super) report: &'a burn_dragon_universality::RuliadEvalReport,
    pub(super) bus: &'a TrainingEventBus,
    pub(super) metric_prefix: Option<&'a str>,
    pub(super) output_degeneracy: Option<&'a crate::train::steps::OutputDegeneracyStats>,
    pub(super) examples: &'a [CapabilityProbeExample],
    pub(super) schema_alignment: RuliadAnswerSchemaAlignmentSummary,
    pub(super) completion_degeneracy: Option<RuliadCompletionDegeneracySummary>,
    pub(super) generation_budget: Option<RuliadProbeGenerationBudgetSummary>,
}

pub(super) fn emit_ruliad_correctness_metrics_with_labels(request: RuliadCorrectnessMetrics<'_>) {
    let RuliadCorrectnessMetrics {
        identity:
            RuliadProbeIdentity {
                run_name,
                epoch,
                absolute_step,
                probe_name,
            },
        report,
        bus,
        metric_prefix,
        output_degeneracy,
        examples,
        schema_alignment,
        completion_degeneracy,
        generation_budget,
    } = request;
    let item_count = report.item_count.max(1) as f64;
    let competence = ruliad_competence_key(report).unwrap_or_default();
    let metrics = [
        ("Ruliad Eval Items", report.item_count as f64),
        ("Ruliad Eval Scored Items", report.scored_count as f64),
        ("Ruliad Competence Score", ruliad_competence_score(report)),
        (
            "Ruliad Competence Verifier PPM",
            competence.verifier_ppm as f64,
        ),
        (
            "Ruliad Competence Semantic PPM",
            competence.semantic_ppm as f64,
        ),
        (
            "Ruliad Competence Partial PPM",
            competence.partial_ppm as f64,
        ),
        (
            "Ruliad Competence Certificate PPM",
            competence.certificate_ppm as f64,
        ),
        (
            "Ruliad Competence Completion Health PPM",
            competence.completion_health_ppm as f64,
        ),
        ("Ruliad Exact Accuracy", f64::from(report.exact_accuracy)),
        (
            "Ruliad Semantic Accuracy",
            f64::from(report.semantic_accuracy),
        ),
        (
            "Ruliad Verifier Accuracy",
            f64::from(report.verifier_accuracy),
        ),
        (
            "Ruliad Partial Credit Rate",
            f64::from(report.partial_credit_rate),
        ),
        (
            "Ruliad Schema Valid Wrong Rate",
            report.schema_valid_wrong_count as f64 / item_count,
        ),
        (
            "Ruliad Malformed Completion Rate",
            report.malformed_completion_count as f64 / item_count,
        ),
        (
            "Ruliad Missing Completion Rate",
            report.missing_completion_count as f64 / item_count,
        ),
        (
            "Ruliad Mean Partial Progress",
            f64::from(report.mean_partial_progress),
        ),
        (
            "Ruliad Answer Field Accuracy",
            f64::from(report.answer_field_accuracy),
        ),
        (
            "Ruliad Answer Field Coverage",
            f64::from(report.answer_field_coverage),
        ),
        (
            "Ruliad Answer Termination Rate",
            f64::from(report.answer_termination_rate),
        ),
        (
            "Ruliad Mean Completion Quality",
            f64::from(report.mean_completion_quality),
        ),
        (
            "Ruliad Expected Answer Distinct Fraction",
            f64::from(report.expected_answer_distinct_fraction),
        ),
        (
            "Ruliad Actual Answer Distinct Fraction",
            f64::from(report.actual_answer_distinct_fraction),
        ),
        (
            "Ruliad Presented Action Rate",
            f64::from(report.presented_action_rate),
        ),
        (
            "Ruliad Presented Action Items",
            report.presented_action_expected_count as f64,
        ),
        (
            "Ruliad Certificate Prefix Coverage",
            f64::from(report.mean_certificate_prefix_coverage),
        ),
        (
            "Ruliad Mean Completion Tokens",
            f64::from(report.mean_completion_tokens),
        ),
        (
            "Ruliad Mean Completion Whitespace Segments",
            f64::from(report.mean_completion_tokens),
        ),
        (
            "Ruliad Answer Key Match Rate",
            schema_alignment.key_match_rate,
        ),
        (
            "Ruliad Answer Key Overlap",
            schema_alignment.mean_key_overlap,
        ),
    ];
    for (name, value) in metrics {
        let metric_name = metric_prefix
            .map(|prefix| format!("{prefix} {name}"))
            .unwrap_or_else(|| name.to_string());
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: run_name.to_string().into(),
            split: TrainingMetricSplit::Valid,
            epoch,
            step_in_epoch: 0,
            absolute_step,
            name: metric_name,
            value,
            running_value: value,
        });
    }
    for group in &report.difficulty_scores {
        let Some(complexity) = group.formal_complexity.as_ref() else {
            continue;
        };
        let label = group
            .label
            .replace(|character: char| !character.is_ascii_alphanumeric(), "_");
        for (coordinate, value) in [
            ("Syntax Nodes", complexity.mean.syntax_nodes as f64),
            ("Axioms", complexity.mean.axiom_count as f64),
            ("Proof Goals", complexity.mean.proof_goal_count as f64),
            ("Proof Steps", complexity.mean.proof_step_count as f64),
            ("Dependency Depth", complexity.mean.dependency_depth as f64),
            ("Dependency Width", complexity.mean.dependency_width as f64),
            ("Variables", complexity.mean.variable_count as f64),
            ("Term Depth", complexity.mean.maximum_term_depth as f64),
            (
                "Distractor Axioms",
                complexity.mean.distractor_axiom_count as f64,
            ),
        ] {
            let name = format!("Ruliad Complexity {label} Mean {coordinate}");
            let metric_name = metric_prefix
                .map(|prefix| format!("{prefix} {name}"))
                .unwrap_or(name);
            let _ = bus.send_metric_sample(TrainingMetricSample {
                run_id: run_name.to_string().into(),
                split: TrainingMetricSplit::Valid,
                epoch,
                step_in_epoch: 0,
                absolute_step,
                name: metric_name,
                value,
                running_value: value,
            });
        }
    }
    if let Some(degeneracy) = completion_degeneracy {
        let mean_model_tokens =
            degeneracy.token_count as f64 / degeneracy.sequence_count.max(1) as f64;
        for (name, value) in [
            ("Ruliad Mean Model Completion Tokens", mean_model_tokens),
            (
                "Ruliad Completion Repetition Fraction",
                degeneracy.repetition_fraction,
            ),
            (
                "Ruliad Completion Distinct-1 Fraction",
                degeneracy.distinct_1_fraction,
            ),
            (
                "Ruliad Completion Distinct-2 Fraction",
                degeneracy.distinct_2_fraction,
            ),
            (
                "Ruliad Completion Max Period 2..16 Fraction",
                degeneracy.max_period_2_to_16_fraction,
            ),
            (
                "Ruliad Completion Max Period 2..64 Fraction",
                degeneracy.max_period_2_to_64_fraction,
            ),
            (
                "Ruliad Completion Dominant Period 2..64",
                degeneracy.dominant_period_2_to_64 as f64,
            ),
        ] {
            let metric_name = metric_prefix
                .map(|prefix| format!("{prefix} {name}"))
                .unwrap_or_else(|| name.to_string());
            let _ = bus.send_metric_sample(TrainingMetricSample {
                run_id: run_name.to_string().into(),
                split: TrainingMetricSplit::Valid,
                epoch,
                step_in_epoch: 0,
                absolute_step,
                name: metric_name,
                value,
                running_value: value,
            });
        }
    }
    if let Some(generation_budget) = generation_budget {
        for (name, value) in [
            (
                "Ruliad Probe Mean Generation Budget",
                generation_budget.mean_max_new_tokens,
            ),
            (
                "Ruliad Probe Mean Minimum Answer Tokens",
                generation_budget.mean_minimum_answer_tokens,
            ),
            (
                "Ruliad Probe Answer Budget Sufficient Rate",
                generation_budget.sufficient_fraction,
            ),
            (
                "Ruliad Probe Generation Hit Budget Rate",
                generation_budget.hit_budget_fraction,
            ),
        ] {
            let metric_name = metric_prefix
                .map(|prefix| format!("{prefix} {name}"))
                .unwrap_or_else(|| name.to_string());
            let _ = bus.send_metric_sample(TrainingMetricSample {
                run_id: run_name.to_string().into(),
                split: TrainingMetricSplit::Valid,
                epoch,
                step_in_epoch: 0,
                absolute_step,
                name: metric_name,
                value,
                running_value: value,
            });
        }
    }
    let _ = bus.send_capability_probe_sample(ruliad_capability_probe_sample(
        RuliadProbeIdentity {
            run_name,
            epoch,
            absolute_step,
            probe_name,
        },
        report,
        competence,
        output_degeneracy,
        examples,
        completion_degeneracy,
    ));
}

pub(super) fn emit_ruliad_capability_gate_metrics(
    run_name: &str,
    report: &burn_dragon_universality::RuliadEvalReport,
    output_degeneracy: Option<&crate::train::steps::OutputDegeneracyStats>,
    gates: &burn_dragon_train::TrainingGatesConfig,
    required_by_deployment_contract: bool,
    event: TrainingEventContext<'_>,
) -> RuliadCapabilityGateStatus {
    let TrainingEventContext {
        epoch,
        absolute_step,
        bus,
    } = event;
    let status = ruliad_capability_gate_status(report, output_degeneracy, gates);
    let (_, _, _, completion_health_rate) = ruliad_capability_rates(report);
    for (name, value) in [
        (
            "Ruliad Capability Gate Passed",
            if status.passed { 1.0 } else { 0.0 },
        ),
        (
            "Ruliad Capability Gate Failure Count",
            status.reasons.len() as f64,
        ),
        (
            "Ruliad Capability Completion Health Rate",
            completion_health_rate,
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
    if gates.enabled && required_by_deployment_contract && !status.passed {
        let _ = bus.send_gate_event(TrainingGateEvent {
            run_id: run_name.to_string().into(),
            gate: "ruliad_capability_gate_failed".to_string(),
            action: TrainingGateAction::Alert,
            severity: TrainingGateSeverity::Warning,
            epoch: Some(epoch),
            absolute_step: Some(absolute_step),
            message: format!(
                "ruliad capability gate failed: {}",
                status.reasons.join(", ")
            ),
        });
    }
    status
}

pub(super) fn ruliad_capability_probe_sample(
    identity: RuliadProbeIdentity<'_>,
    report: &burn_dragon_universality::RuliadEvalReport,
    competence: RuliadCompetenceKey,
    output_degeneracy: Option<&crate::train::steps::OutputDegeneracyStats>,
    examples: &[CapabilityProbeExample],
    completion_degeneracy: Option<RuliadCompletionDegeneracySummary>,
) -> CapabilityProbeSample {
    let RuliadProbeIdentity {
        run_name,
        epoch,
        absolute_step,
        probe_name,
    } = identity;
    let item_count = report.item_count.max(1) as f64;
    let mut group_buckets = Vec::new();
    extend_ruliad_capability_groups(&mut group_buckets, "difficulty", &report.difficulty_scores);
    extend_ruliad_capability_groups(&mut group_buckets, "family", &report.family_scores);
    extend_ruliad_capability_groups(&mut group_buckets, "task", &report.task_scores);
    extend_ruliad_capability_groups(
        &mut group_buckets,
        "contract",
        &report.answer_contract_scores,
    );
    extend_ruliad_capability_groups(&mut group_buckets, "domain", &report.math_domain_scores);
    extend_ruliad_capability_groups(&mut group_buckets, "mode", &report.reasoning_mode_scores);

    CapabilityProbeSample {
        run_id: run_name.to_string().into(),
        split: TrainingMetricSplit::Valid,
        epoch,
        absolute_step,
        probe_name: probe_name.to_string(),
        item_count: report.item_count,
        scored_count: report.scored_count,
        exact_rate: f64::from(report.exact_accuracy),
        semantic_rate: f64::from(report.semantic_accuracy),
        verifier_rate: f64::from(report.verifier_accuracy),
        partial_credit_rate: f64::from(report.partial_credit_rate),
        schema_valid_wrong_rate: report.schema_valid_wrong_count as f64 / item_count,
        malformed_rate: report.malformed_completion_count as f64 / item_count,
        missing_rate: report.missing_completion_count as f64 / item_count,
        certificate_rate: f64::from(competence.certificate_ppm) / 1_000_000.0,
        completion_health_rate: f64::from(competence.completion_health_ppm) / 1_000_000.0,
        mean_partial_progress: f64::from(report.mean_partial_progress),
        answer_field_accuracy: f64::from(report.answer_field_accuracy),
        answer_field_coverage: f64::from(report.answer_field_coverage),
        answer_termination_rate: f64::from(report.answer_termination_rate),
        expected_answer_distinct_fraction: f64::from(report.expected_answer_distinct_fraction),
        actual_answer_distinct_fraction: f64::from(report.actual_answer_distinct_fraction),
        actual_answer_dominant_fraction: Some(f64::from(report.actual_answer_dominant_fraction)),
        field_value_distinct_ratio: Some(f64::from(report.field_value_distinct_ratio)),
        field_value_dominant_fraction: Some(f64::from(report.actual_field_value_dominant_fraction)),
        mean_completion_tokens: f64::from(report.mean_completion_tokens),
        achieved_difficulty_level: ruliad_achieved_verifier_difficulty(report),
        output_entropy_bits: output_degeneracy.map(|stats| stats.entropy_bits),
        output_distinct_2_fraction: output_degeneracy.map(|stats| stats.distinct_2_fraction),
        completion_repetition_fraction: completion_degeneracy
            .map(|stats| stats.repetition_fraction),
        completion_distinct_1_fraction: completion_degeneracy
            .map(|stats| stats.distinct_1_fraction),
        completion_distinct_2_fraction: completion_degeneracy
            .map(|stats| stats.distinct_2_fraction),
        completion_max_period_2_to_16_fraction: completion_degeneracy
            .map(|stats| stats.max_period_2_to_16_fraction),
        completion_max_period_2_to_64_fraction: completion_degeneracy
            .map(|stats| stats.max_period_2_to_64_fraction),
        completion_dominant_period_2_to_64: completion_degeneracy
            .map(|stats| stats.dominant_period_2_to_64),
        group_buckets,
        examples: examples.to_vec(),
    }
}

pub(super) fn extend_ruliad_capability_groups(
    output: &mut Vec<CapabilityProbeGroupMetric>,
    prefix: &str,
    groups: &[burn_dragon_universality::RuliadEvalGroupScore],
) {
    output.extend(groups.iter().map(|group| CapabilityProbeGroupMetric {
        label: format!("{prefix}:{}", group.label),
        item_count: group.count,
        exact_rate: f64::from(group.exact_accuracy),
        semantic_rate: f64::from(group.semantic_accuracy),
        verifier_rate: f64::from(group.verifier_accuracy),
        partial_credit_rate: f64::from(group.partial_credit_rate),
        schema_valid_wrong_rate: ratio_usize(group.schema_valid_wrong_count, group.count),
        malformed_rate: ratio_usize(group.malformed_completion_count, group.count),
        missing_rate: ratio_usize(group.missing_completion_count, group.count),
        mean_partial_progress: f64::from(group.mean_partial_progress),
        answer_field_accuracy: f64::from(group.answer_field_accuracy),
        answer_field_coverage: f64::from(group.answer_field_coverage),
        answer_termination_rate: f64::from(group.answer_termination_rate),
    }));
}

pub(super) fn ratio_usize(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

pub(super) fn ruliad_achieved_verifier_difficulty(
    report: &burn_dragon_universality::RuliadEvalReport,
) -> Option<usize> {
    report
        .difficulty_scores
        .iter()
        .filter(|group| group.verifier_accuracy > 0.0)
        .filter_map(|group| group.label.strip_prefix('d')?.parse::<usize>().ok())
        .max()
}

pub(super) fn run_source_weighted_validation_forward_only<B>(
    env: &ForwardEggrollTrainEnvironment<'_, B>,
    valid_model: &LanguageTrainModel<B>,
    epoch: usize,
    steps_per_epoch: usize,
    bus: &TrainingEventBus,
) -> Result<Option<f64>>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    let requested_batches = env.training.events.source_weighted_validation_batches;
    if requested_batches == 0 {
        return Ok(None);
    }
    let Some(dataset) = env.source_selection_dataset.as_ref() else {
        return Ok(None);
    };
    if !dataset.uses_live_source_selection() {
        return Ok(None);
    }

    let base_absolute_step = epoch.saturating_sub(1).saturating_mul(steps_per_epoch);
    let mut total = 0.0;
    let mut count = 0usize;
    for batch_index in 0..requested_batches {
        let absolute_step = base_absolute_step.saturating_add(batch_index);
        let Some(batch) = dataset.sample_source_weighted_validation_batch::<B>(
            epoch,
            absolute_step,
            env.training.batch_size,
            env.summary_event_token_ids.as_deref(),
            env.device,
        ) else {
            break;
        };
        let output = ValidStep::step(valid_model, batch);
        let loss_value: LossValue<B> = output.adapt();
        let loss = mean_scalar_from_loss(loss_value.value());
        count += 1;
        total += loss;
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: env.run_name.to_string().into(),
            split: TrainingMetricSplit::Valid,
            epoch,
            step_in_epoch: count,
            absolute_step,
            name: "Source Weighted Loss".to_string(),
            value: loss,
            running_value: total / count as f64,
        });
    }

    Ok((count > 0).then_some(total / count as f64))
}
