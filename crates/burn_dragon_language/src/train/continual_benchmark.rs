use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};

const PAYLOAD_TOKEN_OFFSET: usize = 1;
const STREAM_DESCRIPTOR_COEFFICIENT_LIMIT: usize = 8;
const MAX_CONTEXT_OBSERVATION_TOKENS: usize = 16;

/// Context-identifiable task in the controlled Dragon continual-learning
/// benchmark. A support prefix makes the active recurrence identifiable before
/// supervised tokens begin, so forgetting cannot be excused as label conflict.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContextRecurrenceTask {
    A,
    B,
    C,
    D,
}

impl ContextRecurrenceTask {
    pub const ALL: [Self; 4] = [Self::A, Self::B, Self::C, Self::D];

    pub const fn index(self) -> usize {
        match self {
            Self::A => 0,
            Self::B => 1,
            Self::C => 2,
            Self::D => 3,
        }
    }

    const fn coefficients(self) -> (usize, usize, usize) {
        match self {
            Self::A => (1, 1, 1),
            Self::B => (3, 1, 2),
            Self::C => (1, 3, 3),
            Self::D => (3, 3, 1),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub struct ContextRecurrenceSpec {
    pub batch_size: usize,
    pub block_size: usize,
    pub payload_modulus: usize,
}

impl ContextRecurrenceSpec {
    pub fn validate(self) -> Result<(), String> {
        if self.batch_size == 0 {
            return Err("context recurrence batch_size must be > 0".to_string());
        }
        if self.block_size < 4 {
            return Err("context recurrence block_size must be >= 4".to_string());
        }
        if self.payload_modulus < 4 {
            return Err("context recurrence payload_modulus must be >= 4".to_string());
        }
        if self.payload_modulus > usize::MAX / 16
            || self.payload_modulus > (i64::MAX as usize).saturating_sub(PAYLOAD_TOKEN_OFFSET)
        {
            return Err(
                "context recurrence payload_modulus exceeds safe token arithmetic".to_string(),
            );
        }
        if self.batch_size.checked_mul(self.block_size).is_none() {
            return Err("context recurrence batch geometry overflows usize".to_string());
        }
        Ok(())
    }

    pub const fn required_vocab_size(self) -> usize {
        PAYLOAD_TOKEN_OFFSET + self.payload_modulus
    }

    pub const fn supervised_tokens_per_batch(self) -> usize {
        self.batch_size
            * (self.block_size - context_recurrence_observation_tokens(self.block_size) + 1)
    }
}

impl Default for ContextRecurrenceSpec {
    fn default() -> Self {
        Self {
            batch_size: 32,
            block_size: 32,
            payload_modulus: 16,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ContextRecurrenceBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
    pub loss_mask: Tensor<B, 2, Int>,
    /// Unit-normalized descriptor computed only from the observed input prefix.
    /// It is intentionally independent of the hidden benchmark task enum.
    pub stream_descriptor: Vec<f32>,
}

fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    state = (state ^ (state >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    state = (state ^ (state >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    state ^ (state >> 31)
}

/// Allocate a deterministic sparse context mask while minimizing reuse of
/// channels selected by prior contexts. Contexts remain disjoint while unused
/// capacity exists; once overlap is necessary, reuse is balanced before any
/// channel receives another assignment.
pub fn balanced_context_mask(
    seed: u64,
    context_index: usize,
    width: usize,
    active_fraction: f32,
    prior_masks: &[Vec<f32>],
) -> Result<Vec<f32>, String> {
    if width == 0 {
        return Err("balanced context mask width must be > 0".to_string());
    }
    if !(0.0..=1.0).contains(&active_fraction)
        || active_fraction == 0.0
        || !active_fraction.is_finite()
    {
        return Err("balanced context mask active_fraction must be in (0, 1]".to_string());
    }
    if context_index != prior_masks.len() || prior_masks.iter().any(|mask| mask.len() != width) {
        return Err(
            "balanced context masks must be allocated sequentially at a stable width".to_string(),
        );
    }
    let active = ((width as f32 * active_fraction).round() as usize).clamp(1, width);
    let mut usage = vec![0usize; width];
    for mask in prior_masks {
        for (index, value) in mask.iter().enumerate() {
            usage[index] += usize::from(*value > 0.0);
        }
    }
    let mut ranked = (0..width)
        .map(|index| {
            let tie_break = splitmix64(
                seed ^ (context_index as u64).wrapping_mul(0xd6e8_feb8_6659_fd93)
                    ^ (index as u64).wrapping_mul(0xa076_1d64_78bd_642f),
            );
            (usage[index], tie_break, index)
        })
        .collect::<Vec<_>>();
    ranked.sort_unstable();
    let mut mask = vec![0.0; width];
    for (_, _, index) in ranked.into_iter().take(active) {
        mask[index] = 1.0;
    }
    Ok(mask)
}

fn sequence(
    task: ContextRecurrenceTask,
    split_seed: u64,
    batch_index: u64,
    row: usize,
    block_size: usize,
    modulus: usize,
) -> Vec<i64> {
    let row_key = split_seed
        ^ batch_index.wrapping_mul(0xd6e8_feb8_6659_fd93)
        ^ (row as u64).wrapping_mul(0xa076_1d64_78bd_642f);
    let first = splitmix64(row_key) as usize % modulus;
    let second = splitmix64(row_key ^ 0x5899_65cc_7537_4cc3) as usize % modulus;
    let (left, right, bias) = task.coefficients();
    let mut payload = Vec::with_capacity(block_size + 1);
    payload.push(first);
    payload.push(second);
    while payload.len() <= block_size {
        let len = payload.len();
        payload.push((left * payload[len - 1] + right * payload[len - 2] + bias) % modulus);
    }

    payload
        .into_iter()
        .map(|value| (PAYLOAD_TOKEN_OFFSET + value) as i64)
        .collect()
}

/// Generate a deterministic train or holdout batch for one task. Different
/// split seeds create disjoint initial-condition streams without changing the
/// recurrence law.
pub fn context_recurrence_batch<B: Backend>(
    task: ContextRecurrenceTask,
    split_seed: u64,
    batch_index: u64,
    spec: ContextRecurrenceSpec,
    device: &B::Device,
) -> Result<ContextRecurrenceBatch<B>, String> {
    spec.validate()?;
    let elements = spec.batch_size * spec.block_size;
    let mut inputs = Vec::with_capacity(elements);
    let mut targets = Vec::with_capacity(elements);
    let mut loss_mask = Vec::with_capacity(elements);
    let supervised_start = context_recurrence_observation_tokens(spec.block_size) - 1;
    for row in 0..spec.batch_size {
        let tokens = sequence(
            task,
            split_seed,
            batch_index,
            row,
            spec.block_size,
            spec.payload_modulus,
        );
        inputs.extend_from_slice(&tokens[..spec.block_size]);
        targets.extend_from_slice(&tokens[1..]);
        loss_mask
            .extend((0..spec.block_size).map(|position| i64::from(position >= supervised_start)));
    }
    let stream_descriptor = observed_stream_descriptor(
        &inputs,
        spec.batch_size,
        spec.block_size,
        spec.payload_modulus,
    );
    Ok(ContextRecurrenceBatch {
        inputs: Tensor::from_data(
            TensorData::new(inputs, [spec.batch_size, spec.block_size]),
            device,
        ),
        targets: Tensor::from_data(
            TensorData::new(targets, [spec.batch_size, spec.block_size]),
            device,
        ),
        loss_mask: Tensor::from_data(
            TensorData::new(loss_mask, [spec.batch_size, spec.block_size]),
            device,
        ),
        stream_descriptor,
    })
}

fn observed_stream_descriptor(
    inputs: &[i64],
    batch_size: usize,
    block_size: usize,
    payload_modulus: usize,
) -> Vec<f32> {
    let coefficient_span = payload_modulus.min(STREAM_DESCRIPTOR_COEFFICIENT_LIMIT);
    let observed = context_recurrence_observation_tokens(block_size);
    let transition_count = batch_size * (observed - 2);
    let chance_match = transition_count as f32 / payload_modulus as f32;
    let mut descriptor = vec![-chance_match; coefficient_span.pow(3)];
    for row in 0..batch_size {
        let offset = row * block_size;
        for triple in inputs[offset..offset + observed].windows(3) {
            let previous_previous = triple[0] as usize - PAYLOAD_TOKEN_OFFSET;
            let previous = triple[1] as usize - PAYLOAD_TOKEN_OFFSET;
            let next = triple[2] as usize - PAYLOAD_TOKEN_OFFSET;
            for left in 0..coefficient_span {
                for right in 0..coefficient_span {
                    let linear = (left * previous + right * previous_previous) % payload_modulus;
                    let bias = (next + payload_modulus - linear) % payload_modulus;
                    if bias < coefficient_span {
                        let index = (left * coefficient_span + right) * coefficient_span + bias;
                        descriptor[index] += 1.0;
                    }
                }
            }
        }
    }
    normalize_descriptor(descriptor)
}

pub const fn context_recurrence_observation_tokens(block_size: usize) -> usize {
    let half_block = block_size.div_ceil(2);
    if half_block < 3 {
        3
    } else if half_block < MAX_CONTEXT_OBSERVATION_TOKENS {
        half_block
    } else {
        MAX_CONTEXT_OBSERVATION_TOKENS
    }
}

fn normalize_descriptor(mut descriptor: Vec<f32>) -> Vec<f32> {
    let norm = descriptor
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt() as f32;
    if norm > 0.0 {
        for value in &mut descriptor {
            *value /= norm;
        }
    }
    descriptor
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct StreamingContextSelectorConfig {
    /// Create a context when every normalized centroid has lower cosine
    /// similarity than this threshold.
    pub novelty_cosine_threshold: f32,
    /// Exponential update rate for the selected context centroid.
    pub centroid_update_rate: f32,
}

impl Default for StreamingContextSelectorConfig {
    fn default() -> Self {
        Self {
            novelty_cosine_threshold: 0.8,
            centroid_update_rate: 0.1,
        }
    }
}

impl StreamingContextSelectorConfig {
    pub fn validate(self) -> Result<(), String> {
        if !(-1.0..=1.0).contains(&self.novelty_cosine_threshold)
            || !self.novelty_cosine_threshold.is_finite()
        {
            return Err(
                "streaming context novelty_cosine_threshold must be finite and in [-1, 1]"
                    .to_string(),
            );
        }
        if !(0.0..=1.0).contains(&self.centroid_update_rate)
            || !self.centroid_update_rate.is_finite()
        {
            return Err(
                "streaming context centroid_update_rate must be finite and in [0, 1]".to_string(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub struct StreamingContextSelection {
    pub context_index: usize,
    pub created: bool,
    pub cosine_similarity: f32,
}

/// Small task-ID-free context memory used by controlled continual-learning
/// experiments. Selection is a normalized dot-product scan over a bounded
/// sketch of observed support-prefix transitions and therefore
/// scales as `O(contexts * descriptor_width)` without model synchronization.
#[derive(Debug, Clone)]
pub struct StreamingContextSelector {
    config: StreamingContextSelectorConfig,
    centroids: Vec<Vec<f32>>,
}

impl StreamingContextSelector {
    pub fn new(config: StreamingContextSelectorConfig) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            config,
            centroids: Vec::new(),
        })
    }

    pub fn known_contexts(&self) -> usize {
        self.centroids.len()
    }

    pub fn select(
        &mut self,
        descriptor: &[f32],
        allow_create: bool,
    ) -> Result<StreamingContextSelection, String> {
        if descriptor.is_empty() || descriptor.iter().any(|value| !value.is_finite()) {
            return Err("streaming context descriptor must be non-empty and finite".to_string());
        }
        let norm = descriptor
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        if (norm - 1.0).abs() > 1.0e-3 {
            return Err("streaming context descriptor must be unit normalized".to_string());
        }
        if self
            .centroids
            .first()
            .is_some_and(|centroid| centroid.len() != descriptor.len())
        {
            return Err("streaming context descriptor width changed".to_string());
        }
        let best = self
            .centroids
            .iter()
            .enumerate()
            .map(|(index, centroid)| {
                let similarity = centroid
                    .iter()
                    .zip(descriptor)
                    .map(|(left, right)| left * right)
                    .sum::<f32>();
                (index, similarity)
            })
            .max_by(|left, right| left.1.total_cmp(&right.1));
        let novel =
            best.is_none_or(|(_, similarity)| similarity < self.config.novelty_cosine_threshold);
        if novel && allow_create {
            let context_index = self.centroids.len();
            self.centroids
                .push(normalize_descriptor(descriptor.to_vec()));
            return Ok(StreamingContextSelection {
                context_index,
                created: true,
                cosine_similarity: best.map_or(0.0, |(_, similarity)| similarity),
            });
        }
        let (context_index, cosine_similarity) = best.ok_or_else(|| {
            "streaming context selector has no context and creation is disabled".to_string()
        })?;
        if allow_create && self.config.centroid_update_rate > 0.0 {
            let update_rate = self.config.centroid_update_rate;
            let centroid = &self.centroids[context_index];
            self.centroids[context_index] = normalize_descriptor(
                centroid
                    .iter()
                    .zip(descriptor)
                    .map(|(old, new)| old * (1.0 - update_rate) + new * update_rate)
                    .collect(),
            );
        }
        Ok(StreamingContextSelection {
            context_index,
            created: false,
            cosine_similarity,
        })
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq)]
pub struct ContinualTaskEvaluation {
    pub loss: f64,
    pub accuracy: f64,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq)]
pub struct ContinualTaskAcquisition {
    pub loss_reduction: f64,
    pub accuracy_gain: f64,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub struct ContinualAcquisitionThresholds {
    pub loss_floor_ratio: f64,
    pub accuracy_tolerance: f64,
    pub minimum_baseline_loss_reduction: f64,
    pub minimum_baseline_accuracy_gain: f64,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct ContinualAcquisitionGateMetrics {
    pub matched: bool,
    pub baseline_acquired: Vec<bool>,
    pub loss_ratios: Vec<f64>,
    pub accuracy_deltas: Vec<f64>,
}

pub fn evaluate_continual_acquisition_gate(
    baseline: &[ContinualTaskAcquisition],
    candidate: &[ContinualTaskAcquisition],
    thresholds: ContinualAcquisitionThresholds,
) -> Result<ContinualAcquisitionGateMetrics, String> {
    if baseline.is_empty() || baseline.len() != candidate.len() {
        return Err(
            "continual acquisition vectors must be non-empty and equally sized".to_string(),
        );
    }
    if !(0.0..=1.0).contains(&thresholds.loss_floor_ratio)
        || thresholds.accuracy_tolerance < 0.0
        || thresholds.minimum_baseline_loss_reduction < 0.0
        || thresholds.minimum_baseline_accuracy_gain < 0.0
        || !thresholds.loss_floor_ratio.is_finite()
        || !thresholds.accuracy_tolerance.is_finite()
        || !thresholds.minimum_baseline_loss_reduction.is_finite()
        || !thresholds.minimum_baseline_accuracy_gain.is_finite()
    {
        return Err("continual acquisition thresholds must be finite and non-negative, with loss_floor_ratio in [0, 1]".to_string());
    }

    let mut matched = true;
    let mut baseline_acquired = Vec::with_capacity(baseline.len());
    let mut loss_ratios = Vec::with_capacity(baseline.len());
    let mut accuracy_deltas = Vec::with_capacity(baseline.len());
    for (&baseline, &candidate) in baseline.iter().zip(candidate) {
        if [
            baseline.loss_reduction,
            baseline.accuracy_gain,
            candidate.loss_reduction,
            candidate.accuracy_gain,
        ]
        .into_iter()
        .any(|value| !value.is_finite())
        {
            return Err("continual acquisition values must be finite".to_string());
        }
        let acquired = baseline.loss_reduction >= thresholds.minimum_baseline_loss_reduction
            && baseline.accuracy_gain >= thresholds.minimum_baseline_accuracy_gain;
        let loss_ratio = if baseline.loss_reduction.abs() > 1.0e-12 {
            candidate.loss_reduction / baseline.loss_reduction
        } else {
            0.0
        };
        let accuracy_delta = candidate.accuracy_gain - baseline.accuracy_gain;
        matched &= acquired
            && candidate.loss_reduction >= baseline.loss_reduction * thresholds.loss_floor_ratio
            && candidate.accuracy_gain + thresholds.accuracy_tolerance >= baseline.accuracy_gain;
        baseline_acquired.push(acquired);
        loss_ratios.push(loss_ratio);
        accuracy_deltas.push(accuracy_delta);
    }
    Ok(ContinualAcquisitionGateMetrics {
        matched,
        baseline_acquired,
        loss_ratios,
        accuracy_deltas,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn values<const D: usize>(tensor: Tensor<TestBackend, D, Int>) -> Vec<i64> {
        tensor
            .to_data()
            .into_vec::<i64>()
            .expect("integer tensor data")
    }

    #[test]
    fn recurrence_batches_are_context_identifiable_and_verifiable() {
        let device = Default::default();
        let spec = ContextRecurrenceSpec {
            batch_size: 2,
            block_size: 8,
            payload_modulus: 11,
        };
        for task in ContextRecurrenceTask::ALL {
            let batch = context_recurrence_batch::<TestBackend>(task, 7, 3, spec, &device)
                .expect("valid recurrence batch");
            let inputs = values(batch.inputs);
            let targets = values(batch.targets);
            let mask = values(batch.loss_mask);
            let (left, right, bias) = task.coefficients();
            let supervised_start = context_recurrence_observation_tokens(spec.block_size) - 1;
            for row in 0..spec.batch_size {
                let offset = row * spec.block_size;
                assert!(
                    mask[offset..offset + supervised_start]
                        .iter()
                        .all(|value| *value == 0)
                );
                assert!(
                    mask[offset + supervised_start..offset + spec.block_size]
                        .iter()
                        .all(|value| *value == 1)
                );
                for position in 1..spec.block_size {
                    let previous = (inputs[offset + position] as usize) - PAYLOAD_TOKEN_OFFSET;
                    let previous_previous =
                        (inputs[offset + position - 1] as usize) - PAYLOAD_TOKEN_OFFSET;
                    let expected = PAYLOAD_TOKEN_OFFSET
                        + (left * previous + right * previous_previous + bias)
                            % spec.payload_modulus;
                    assert_eq!(targets[offset + position] as usize, expected);
                }
            }
        }

        let initial_pairs = ContextRecurrenceTask::ALL.map(|task| {
            let batch = context_recurrence_batch::<TestBackend>(task, 7, 3, spec, &device)
                .expect("valid recurrence batch");
            values(batch.inputs)[..2].to_vec()
        });
        assert!(initial_pairs.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[test]
    fn train_and_holdout_streams_are_distinct_and_reproducible() {
        let device = Default::default();
        let spec = ContextRecurrenceSpec::default();
        let first =
            context_recurrence_batch::<TestBackend>(ContextRecurrenceTask::A, 11, 0, spec, &device)
                .expect("first batch");
        let repeated =
            context_recurrence_batch::<TestBackend>(ContextRecurrenceTask::A, 11, 0, spec, &device)
                .expect("repeated batch");
        let holdout =
            context_recurrence_batch::<TestBackend>(ContextRecurrenceTask::A, 12, 0, spec, &device)
                .expect("holdout batch");
        assert_eq!(values(first.inputs), values(repeated.inputs));
        assert_ne!(values(repeated.targets), values(holdout.targets));
    }

    #[test]
    fn streaming_selector_discovers_and_recognizes_contexts_without_task_ids() {
        let device = Default::default();
        let spec = ContextRecurrenceSpec {
            batch_size: 8,
            block_size: 12,
            payload_modulus: 8,
        };
        let mut selector = StreamingContextSelector::new(StreamingContextSelectorConfig::default())
            .expect("valid selector");
        let mut learned = Vec::new();
        let mut training_descriptors = Vec::new();
        for task in ContextRecurrenceTask::ALL {
            let batch = context_recurrence_batch::<TestBackend>(task, 11, 0, spec, &device)
                .expect("training batch");
            training_descriptors.push(batch.stream_descriptor.clone());
            let selection = selector
                .select(&batch.stream_descriptor, true)
                .expect("learn context");
            assert!(selection.created, "task {task:?} should create a context");
            learned.push(selection.context_index);
        }
        assert_eq!(selector.known_contexts(), ContextRecurrenceTask::ALL.len());
        for (task, expected) in ContextRecurrenceTask::ALL.into_iter().zip(learned) {
            let holdout = context_recurrence_batch::<TestBackend>(task, 29, 7, spec, &device)
                .expect("holdout batch");
            let similarities = training_descriptors
                .iter()
                .map(|descriptor| {
                    descriptor
                        .iter()
                        .zip(&holdout.stream_descriptor)
                        .map(|(left, right)| left * right)
                        .sum::<f32>()
                })
                .collect::<Vec<_>>();
            let same_task = similarities[task.index()];
            let strongest_other = similarities
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != task.index())
                .map(|(_, similarity)| *similarity)
                .fold(f32::NEG_INFINITY, f32::max);
            assert!(
                same_task > 0.3 && same_task > strongest_other + 0.1,
                "task {task:?} lacks descriptor margin: {similarities:?}"
            );
            let selection = selector
                .select(&holdout.stream_descriptor, false)
                .expect("recognize context");
            assert_eq!(selection.context_index, expected);
            assert!(!selection.created);
            assert!(
                selection.cosine_similarity > 0.3,
                "task {task:?} holdout cosine {}",
                selection.cosine_similarity
            );
        }
    }

    #[test]
    fn streaming_selector_rejects_non_normalized_descriptors() {
        let mut selector = StreamingContextSelector::new(StreamingContextSelectorConfig::default())
            .expect("valid selector");
        assert_eq!(
            selector.select(&[2.0, 0.0], true),
            Err("streaming context descriptor must be unit normalized".to_string())
        );
    }

    #[test]
    fn balanced_context_masks_avoid_overlap_until_capacity_is_exhausted() {
        let mut masks = Vec::new();
        for context in 0..4 {
            masks.push(
                balanced_context_mask(7, context, 16, 0.25, &masks).expect("balanced context mask"),
            );
        }
        for left in 0..masks.len() {
            assert_eq!(masks[left].iter().filter(|value| **value > 0.0).count(), 4);
            for right in left + 1..masks.len() {
                assert!(
                    masks[left]
                        .iter()
                        .zip(&masks[right])
                        .all(|(left, right)| *left == 0.0 || *right == 0.0)
                );
            }
        }
    }

    #[test]
    fn acquisition_gate_rejects_an_underlearned_baseline() {
        let metrics = evaluate_continual_acquisition_gate(
            &[ContinualTaskAcquisition {
                loss_reduction: 0.1,
                accuracy_gain: 0.02,
            }],
            &[ContinualTaskAcquisition {
                loss_reduction: 0.1,
                accuracy_gain: 0.02,
            }],
            ContinualAcquisitionThresholds {
                loss_floor_ratio: 0.9,
                accuracy_tolerance: 0.05,
                minimum_baseline_loss_reduction: 0.5,
                minimum_baseline_accuracy_gain: 0.25,
            },
        )
        .expect("valid acquisition gate");
        assert!(!metrics.matched);
        assert_eq!(metrics.baseline_acquired, vec![false]);
        assert_eq!(metrics.loss_ratios, vec![1.0]);
    }

    #[test]
    fn acquisition_gate_requires_candidate_parity_after_baseline_acquisition() {
        let baseline = [ContinualTaskAcquisition {
            loss_reduction: 2.0,
            accuracy_gain: 0.8,
        }];
        let thresholds = ContinualAcquisitionThresholds {
            loss_floor_ratio: 0.9,
            accuracy_tolerance: 0.05,
            minimum_baseline_loss_reduction: 0.5,
            minimum_baseline_accuracy_gain: 0.25,
        };
        let matched = evaluate_continual_acquisition_gate(
            &baseline,
            &[ContinualTaskAcquisition {
                loss_reduction: 1.9,
                accuracy_gain: 0.76,
            }],
            thresholds,
        )
        .expect("valid matched gate");
        assert!(matched.matched);
        assert_eq!(matched.baseline_acquired, vec![true]);

        let underlearned = evaluate_continual_acquisition_gate(
            &baseline,
            &[ContinualTaskAcquisition {
                loss_reduction: 1.0,
                accuracy_gain: 0.4,
            }],
            thresholds,
        )
        .expect("valid underlearned gate");
        assert!(!underlearned.matched);
    }
}
