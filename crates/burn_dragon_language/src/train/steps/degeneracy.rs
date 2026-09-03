//! Output entropy, repetition, periodicity, and marginal-coverage diagnostics.

use super::*;

pub(super) fn output_degeneracy_from_logits<B: BackendTrait>(
    logits: Tensor<B, 3>,
    eos_id: Option<i64>,
) -> OutputDegeneracyStats {
    let [batch, time, vocab] = logits.shape().dims::<3>();
    if batch == 0 || time == 0 || vocab == 0 {
        return OutputDegeneracyStats::default();
    }
    let values = logits
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("validation degeneracy logits vec");
    let mut accumulator = OutputDegeneracyAccumulator::new(eos_id);

    for b in 0..batch {
        for t in 0..time {
            let start = (b * time + t) * vocab;
            if let Some(step) = output_degeneracy_step_from_row(&values[start..start + vocab]) {
                accumulator.record(step);
            }
        }
    }

    accumulator.finish()
}

pub(super) fn validation_degeneracy_prompt_start(
    prompt_index: usize,
    prompt_count: usize,
    available: usize,
) -> usize {
    if available == 0 || prompt_index == 0 || prompt_count <= 1 {
        return 0;
    }
    let min_start = available.min(64);
    let interior = available.saturating_sub(min_start);
    let interior_index = prompt_index.saturating_sub(1);
    let interior_count = prompt_count.saturating_sub(1).max(1);
    min_start + (interior_index.saturating_mul(interior + 1) / interior_count).min(interior)
}

pub(super) fn rollout_prompt_start(
    step_index: usize,
    every_steps: usize,
    block_size: usize,
    prompt_tokens: usize,
) -> usize {
    let available = block_size.saturating_sub(prompt_tokens);
    if available == 0 {
        return 0;
    }
    let min_start = available.min(prompt_tokens.max(1));
    let span = available.saturating_sub(min_start);
    if span == 0 {
        return min_start;
    }
    let rollout_index = step_index / every_steps.max(1);
    min_start + (rollout_index.saturating_mul(prompt_tokens.max(1)) % (span + 1))
}

pub(super) fn lagged_prediction_tensors<B: BackendTrait>(
    log_probs: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    clean_inputs: Tensor<B, 2, Int>,
    lag: usize,
    batch_size: usize,
    time: usize,
    vocab: usize,
) -> Option<LaggedPredictionTensors<B>> {
    if lag == 0 || time == 0 || lag > time {
        return None;
    }
    let start = lag.saturating_sub(1);
    let valid_time = time.saturating_sub(start);
    if valid_time == 0 {
        return None;
    }
    Some((
        log_probs.slice([0..batch_size, start..time, 0..vocab]),
        targets.slice([0..batch_size, start..time]),
        clean_inputs.slice([0..batch_size, 0..valid_time]),
    ))
}

pub(super) fn unlikelihood_from_log_probs<B: BackendTrait>(
    log_probs: Tensor<B, 3>,
    tokens: Tensor<B, 2, Int>,
    epsilon: f32,
) -> Tensor<B, 2> {
    selected_token_log_probs(log_probs, tokens)
        .exp()
        .clamp_min(0.0)
        .clamp_max(1.0 - epsilon)
        .mul_scalar(-1.0)
        .add_scalar(1.0)
        .clamp_min(epsilon)
        .log()
        .mul_scalar(-1.0)
}

pub(super) fn cycle_repeat_mask<B: BackendTrait>(
    next: &Tensor<B, 2, Int>,
    history: &[Tensor<B, 2, Int>],
    min_lag: usize,
    max_lag: usize,
) -> Option<Tensor<B, 2, burn::tensor::Bool>> {
    if history.is_empty() || min_lag == 0 || max_lag < min_lag {
        return None;
    }
    let mut mask: Option<Tensor<B, 2, burn::tensor::Bool>> = None;
    for lag in min_lag..=max_lag {
        let Some(previous) = history.get(lag.saturating_sub(1)) else {
            continue;
        };
        let lag_mask = next.clone().equal(previous.clone());
        mask = Some(match mask {
            Some(accumulated) => accumulated.bool_or(lag_mask),
            None => lag_mask,
        });
    }
    mask
}

#[derive(Clone, Copy, Debug)]
pub(super) struct OutputDegeneracyStep {
    pub(super) argmax: usize,
    pub(super) entropy_bits: f64,
    pub(super) max_probability: f64,
}

#[derive(Debug)]
pub(super) struct OutputDegeneracyAccumulator {
    eos_id: Option<i64>,
    token_count: usize,
    entropy_sum: f64,
    max_probability_sum: f64,
    eos_count: usize,
    repetition_count: usize,
    repetition_denominator: usize,
    previous: Option<usize>,
    unique: HashSet<usize>,
    steps: Vec<OutputDegeneracyStep>,
    prompt_tokens: Vec<i64>,
    generated_tokens: Vec<i64>,
}

pub(super) struct OutputDegeneracySummary {
    entropy_bits: f64,
    mean_max_probability: f64,
    argmax_unique_fraction: f64,
    repetition_fraction: f64,
}

impl OutputDegeneracyAccumulator {
    const MIN_PAYLOAD_TOKENS_BEFORE_EOS_PADDING: usize = 16;

    pub(super) fn new(eos_id: Option<i64>) -> Self {
        Self {
            eos_id,
            token_count: 0,
            entropy_sum: 0.0,
            max_probability_sum: 0.0,
            eos_count: 0,
            repetition_count: 0,
            repetition_denominator: 0,
            previous: None,
            unique: HashSet::new(),
            steps: Vec::new(),
            prompt_tokens: Vec::new(),
            generated_tokens: Vec::new(),
        }
    }

    pub(super) fn record(&mut self, step: OutputDegeneracyStep) {
        self.entropy_sum += step.entropy_bits;
        self.max_probability_sum += step.max_probability;
        if self
            .eos_id
            .is_some_and(|id| id >= 0 && step.argmax == id as usize)
        {
            self.eos_count = self.eos_count.saturating_add(1);
        }
        if let Some(previous) = self.previous {
            self.repetition_denominator = self.repetition_denominator.saturating_add(1);
            if previous == step.argmax {
                self.repetition_count = self.repetition_count.saturating_add(1);
            }
        }
        self.previous = Some(step.argmax);
        self.unique.insert(step.argmax);
        self.steps.push(step);
        self.token_count = self.token_count.saturating_add(1);
    }

    pub(super) fn record_generated_token(&mut self, token: i64) {
        self.generated_tokens.push(token);
    }

    pub(super) fn record_prompt_tokens(&mut self, tokens: impl IntoIterator<Item = i64>) {
        self.prompt_tokens.extend(tokens);
    }

    pub(super) fn finish(self) -> OutputDegeneracyStats {
        if self.token_count == 0 {
            return OutputDegeneracyStats::default();
        }
        let first_eos_index = self.eos_id.and_then(|eos_id| {
            self.generated_tokens
                .iter()
                .position(|token| *token == eos_id)
        });
        let scored_len = first_eos_index
            .filter(|index| *index >= Self::MIN_PAYLOAD_TOKENS_BEFORE_EOS_PADDING)
            .unwrap_or(self.generated_tokens.len())
            .min(self.steps.len());
        let scored_steps = &self.steps[..scored_len];
        let scored_generated_tokens = &self.generated_tokens[..scored_len];
        let scored = Self::summarize_steps(scored_steps).unwrap_or_else(|| {
            Self::summarize_steps(&self.steps).expect("non-empty output degeneracy accumulator")
        });
        let eos_fraction = if scored_len < self.generated_tokens.len() {
            0.0
        } else {
            self.eos_count as f64 / self.token_count as f64
        };
        let distinct_1_fraction = distinct_n_fraction(scored_generated_tokens, 1);
        let distinct_2_fraction = distinct_n_fraction(scored_generated_tokens, 2);
        let period_2_fraction = period_fraction(scored_generated_tokens, 2);
        let period_3_fraction = period_fraction(scored_generated_tokens, 3);
        let max_period_2_to_16_fraction = max_period_fraction(scored_generated_tokens, 2..=16);
        let (dominant_period_2_to_64, max_period_2_to_64_fraction) =
            dominant_period_fraction(scored_generated_tokens, 2..=64);
        let (prompt_dominant_period_2_to_64, prompt_max_period_2_to_64_fraction) =
            dominant_period_fraction(&self.prompt_tokens, 2..=64);
        OutputDegeneracyStats {
            token_count: self.token_count,
            entropy_bits: scored.entropy_bits,
            mean_max_probability: scored.mean_max_probability,
            argmax_unique_fraction: scored.argmax_unique_fraction,
            eos_fraction,
            repetition_fraction: scored.repetition_fraction,
            distinct_1_fraction,
            distinct_2_fraction,
            period_2_fraction,
            period_3_fraction,
            max_period_2_to_16_fraction,
            max_period_2_to_64_fraction,
            dominant_period_2_to_64,
            prompt_max_period_2_to_64_fraction,
            prompt_dominant_period_2_to_64,
            prompt_tokens: self.prompt_tokens,
            generated_tokens: self.generated_tokens,
        }
    }

    pub(super) fn summarize_steps(
        steps: &[OutputDegeneracyStep],
    ) -> Option<OutputDegeneracySummary> {
        if steps.is_empty() {
            return None;
        }
        let mut entropy_sum = 0.0;
        let mut max_probability_sum = 0.0;
        let mut unique = HashSet::new();
        let mut previous = None;
        let mut repetition_count = 0usize;
        let mut repetition_denominator = 0usize;
        for step in steps {
            entropy_sum += step.entropy_bits;
            max_probability_sum += step.max_probability;
            unique.insert(step.argmax);
            if let Some(previous) = previous {
                repetition_denominator = repetition_denominator.saturating_add(1);
                if previous == step.argmax {
                    repetition_count = repetition_count.saturating_add(1);
                }
            }
            previous = Some(step.argmax);
        }
        Some(OutputDegeneracySummary {
            entropy_bits: entropy_sum / steps.len() as f64,
            mean_max_probability: max_probability_sum / steps.len() as f64,
            argmax_unique_fraction: unique.len() as f64 / steps.len() as f64,
            repetition_fraction: if repetition_denominator == 0 {
                0.0
            } else {
                repetition_count as f64 / repetition_denominator as f64
            },
        })
    }
}

pub(super) fn distinct_n_fraction(tokens: &[i64], n: usize) -> f64 {
    if n == 0 || tokens.len() < n {
        return 0.0;
    }
    let total = tokens.len() + 1 - n;
    let distinct = tokens
        .windows(n)
        .map(|window| window.to_vec())
        .collect::<HashSet<_>>()
        .len();
    distinct as f64 / total as f64
}

pub(super) fn period_fraction(tokens: &[i64], period: usize) -> f64 {
    if period == 0 || tokens.len() < period.saturating_mul(2) {
        return 0.0;
    }
    let matches = (period..tokens.len())
        .filter(|idx| tokens[*idx] == tokens[*idx - period])
        .count();
    matches as f64 / (tokens.len() - period) as f64
}

pub(super) fn max_period_fraction(tokens: &[i64], periods: impl IntoIterator<Item = usize>) -> f64 {
    dominant_period_fraction(tokens, periods).1
}

pub(super) fn dominant_period_fraction(
    tokens: &[i64],
    periods: impl IntoIterator<Item = usize>,
) -> (usize, f64) {
    periods
        .into_iter()
        .map(|period| (period, period_fraction(tokens, period)))
        .max_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or((0, 0.0))
}

pub(super) fn selected_token_logits<B: BackendTrait>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
) -> Tensor<B, 2> {
    let [batch, time, _vocab] = logits.shape().dims();
    logits
        .gather(2, targets.reshape([batch, time, 1]))
        .reshape([batch, time])
}

pub(super) fn answer_prefix_input_mask<B: BackendTrait>(
    loss_mask: Tensor<B, 2, Int>,
) -> Tensor<B, 2, Int> {
    let [batch, time] = loss_mask.shape().dims();
    let device = loss_mask.device();
    if time == 0 {
        return Tensor::<B, 2, Int>::zeros([batch, 0], &device);
    }
    let head = Tensor::<B, 2, Int>::zeros([batch, 1], &device);
    if time == 1 {
        return head;
    }
    let previous_targets = loss_mask.slice([0..batch, 0..(time - 1)]);
    Tensor::cat(vec![head, previous_targets], 1)
}

pub(super) fn entropy_floor_loss_from_logits<B: BackendTrait>(
    logits: Tensor<B, 3>,
    target_entropy_bits: f32,
) -> Option<Tensor<B, 1>> {
    entropy_floor_loss_from_log_probs(log_probs_from_logits(logits), target_entropy_bits)
}

pub(super) fn entropy_floor_loss_from_log_probs<B: BackendTrait>(
    log_probs: Tensor<B, 3>,
    target_entropy_bits: f32,
) -> Option<Tensor<B, 1>> {
    let [batch, time, vocab] = log_probs.shape().dims();
    if batch == 0 || time == 0 || vocab == 0 || target_entropy_bits <= f32::EPSILON {
        return None;
    }
    let flat_log_probs = log_probs.reshape([batch * time, vocab]);
    let flat_probs = flat_log_probs.clone().exp();
    entropy_floor_loss_from_flat_log_probs(flat_log_probs, flat_probs, target_entropy_bits)
}

pub(super) fn entropy_floor_loss_from_flat_log_probs<B: BackendTrait>(
    flat_log_probs: Tensor<B, 2>,
    flat_probs: Tensor<B, 2>,
    target_entropy_bits: f32,
) -> Option<Tensor<B, 1>> {
    let [token_count, vocab] = flat_log_probs.shape().dims();
    if token_count == 0 || vocab == 0 || target_entropy_bits <= f32::EPSILON {
        return None;
    }
    let entropy = (flat_probs * flat_log_probs)
        .sum_dim(1)
        .mul_scalar(-1.0)
        .mean()
        .reshape([1]);
    let target_nats = target_entropy_bits * std::f32::consts::LN_2;
    Some(
        entropy
            .mul_scalar(-1.0)
            .add_scalar(target_nats)
            .clamp_min(0.0),
    )
}

pub(super) fn predicted_marginal_from_logits<B: BackendTrait>(
    logits: Tensor<B, 3>,
) -> Option<Tensor<B, 2>> {
    predicted_marginal_from_log_probs(log_probs_from_logits(logits))
}

pub(super) fn predicted_marginal_from_log_probs<B: BackendTrait>(
    log_probs: Tensor<B, 3>,
) -> Option<Tensor<B, 2>> {
    let [batch, time, vocab] = log_probs.shape().dims();
    if batch == 0 || time == 0 || vocab == 0 {
        return None;
    }
    Some(log_probs.reshape([batch * time, vocab]).exp().mean_dim(0))
}

pub(super) fn marginal_entropy_floor_loss_from_logits<B: BackendTrait>(
    logits: Tensor<B, 3>,
    target_entropy_bits: f32,
) -> Option<Tensor<B, 1>> {
    marginal_entropy_floor_loss_from_marginal(
        predicted_marginal_from_logits(logits)?,
        target_entropy_bits,
    )
}

pub(super) fn marginal_entropy_floor_loss_from_marginal<B: BackendTrait>(
    marginal: Tensor<B, 2>,
    target_entropy_bits: f32,
) -> Option<Tensor<B, 1>> {
    if target_entropy_bits <= f32::EPSILON {
        return None;
    }
    let entropy = (marginal.clone() * marginal.clamp_min(1.0e-12).log())
        .sum_dim(1)
        .mul_scalar(-1.0)
        .reshape([1]);
    let target_nats = target_entropy_bits * std::f32::consts::LN_2;
    Some(
        entropy
            .mul_scalar(-1.0)
            .add_scalar(target_nats)
            .clamp_min(0.0),
    )
}

pub(super) fn target_marginal_coverage_loss_from_logits<B: BackendTrait>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    epsilon: f32,
) -> Option<Tensor<B, 1>> {
    target_marginal_coverage_loss_from_marginal(
        predicted_marginal_from_logits(logits)?,
        targets,
        epsilon,
    )
}

pub(super) fn target_marginal_coverage_loss_from_marginal<B: BackendTrait>(
    marginal: Tensor<B, 2>,
    targets: Tensor<B, 2, Int>,
    epsilon: f32,
) -> Option<Tensor<B, 1>> {
    let [_marginal_batch, vocab] = marginal.shape().dims();
    if vocab == 0 || epsilon <= 0.0 || epsilon >= 1.0 {
        return None;
    }
    let [batch, time] = targets.shape().dims();
    let token_count = batch * time;
    if token_count == 0 {
        return None;
    }
    let log_marginal = marginal.clamp_min(epsilon).log().repeat_dim(0, token_count);
    Some(
        log_marginal
            .gather(1, targets.reshape([token_count, 1]))
            .mean()
            .reshape([1])
            .mul_scalar(-1.0),
    )
}

pub(super) fn output_degeneracy_step_from_logits<B: BackendTrait>(
    logits: Tensor<B, 1>,
) -> Option<OutputDegeneracyStep> {
    let values = logits
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("validation free-running degeneracy logits vec");
    output_degeneracy_step_from_row(&values)
}

pub(super) fn output_degeneracy_step_from_row(row: &[f32]) -> Option<OutputDegeneracyStep> {
    let (argmax, max_logit) = row
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })?;
    let mut exp_sum = 0.0f64;
    let mut weighted_logit_sum = 0.0f64;
    for value in row.iter().copied().filter(|value| value.is_finite()) {
        let weight = (value as f64 - max_logit as f64).exp();
        exp_sum += weight;
        weighted_logit_sum += weight * value as f64;
    }
    if exp_sum <= 0.0 || !exp_sum.is_finite() {
        return None;
    }
    let logsumexp = max_logit as f64 + exp_sum.ln();
    let entropy_nats = logsumexp - weighted_logit_sum / exp_sum;
    Some(OutputDegeneracyStep {
        argmax,
        entropy_bits: entropy_nats.max(0.0) / std::f64::consts::LN_2,
        max_probability: 1.0 / exp_sum,
    })
}
