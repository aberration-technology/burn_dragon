use burn_dragon_time::Instant;
use std::collections::BTreeMap;
use std::mem::size_of;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use burn::data::dataloader::{DataLoader, DataLoaderIterator, Progress};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use rand::prelude::*;
#[cfg(feature = "train")]
use rayon::prelude::*;

use crate::summary_events::summary_event_mask_tensor;
use crate::tokenizer::SharedTokenizer;

use super::DatasetSplit;

#[derive(Clone, Debug)]
pub struct RuliadPolicySample {
    pub item: burn_dragon_universality::RuliadEvalItem,
    pub prompt_tokens: Vec<i64>,
}

#[derive(Clone, Debug)]
pub struct RuliadPolicyBatch {
    pub samples: Vec<RuliadPolicySample>,
    pub tokenization: burn_dragon_universality::RuliadTokenizationConfig,
    pub stop_token_id: Option<i64>,
}

/// Concrete source-selected windows and supervision aligned to their shifted targets.
#[derive(Clone, Debug)]
pub struct SourceSelectedBatch {
    pub windows: Vec<Vec<u32>>,
    pub loss_masks: Option<Vec<Vec<i64>>>,
}

/// One source-selected TBPTT chunk and its logical-document boundary state.
#[derive(Clone, Debug)]
pub struct SourceSelectedStreamBatch {
    pub windows: Vec<Vec<u32>>,
    pub loss_masks: Option<Vec<Vec<i64>>>,
    pub document_complete: bool,
}

/// Abstraction over text corpora that can be converted into DragonModel-compatible batches.
pub trait TokenSequenceDataset: Send + Sync {
    /// Return a shared tokenizer handle (cloned per call).
    fn tokenizer(&self) -> SharedTokenizer;

    /// Return the full number of token ids representing the corpus.
    fn token_count(&self) -> usize;

    /// Copy a contiguous token range into `dst`.
    fn copy_token_range(&self, start: usize, dst: &mut [u32]);

    /// Copy a contiguous token range into `dst`, with epoch context when the dataset wants to
    /// expose deterministic fresh data each epoch. By default, datasets ignore the epoch.
    fn copy_token_range_with_epoch(
        &self,
        split: DatasetSplit,
        epoch_index: usize,
        start: usize,
        dst: &mut [u32],
    ) {
        let _ = (split, epoch_index);
        self.copy_token_range(start, dst);
    }

    /// Ensure a specific epoch slice is ready for consumption before the GPU requests it.
    /// Datasets without epoch-aware generation can ignore this.
    fn prepare_epoch(&self, _split: DatasetSplit, _epoch_index: usize) {}

    /// Opportunistically begin preparing a future epoch in the background.
    /// Datasets without epoch-aware generation can ignore this.
    fn prefetch_epoch(&self, _split: DatasetSplit, _epoch_index: usize) {}

    /// Whether this dataset uses live source selection and should avoid preparing unbounded
    /// future train batches before loss telemetry arrives.
    fn uses_live_source_selection(&self) -> bool {
        false
    }

    /// Return document indices for a source-homogeneous batch, if the dataset supports live
    /// source selection for this split/epoch/step.
    fn source_selected_document_indices(
        &self,
        _split: DatasetSplit,
        _epoch_index: usize,
        _absolute_step: usize,
        _batch_size: usize,
    ) -> Option<Vec<usize>> {
        None
    }

    /// Return concrete source-selected token windows for a batch, if the dataset can generate
    /// them without mapping through global document indices.
    fn source_selected_token_windows(
        &self,
        _split: DatasetSplit,
        _epoch_index: usize,
        _absolute_step: usize,
        _batch_size: usize,
        _block_size: usize,
    ) -> Option<Vec<Vec<u32>>> {
        None
    }

    /// Return concrete source-selected token windows and optional precomputed loss masks for a
    /// batch. Datasets with step-dependent supervision can override this to avoid deriving masks
    /// from a static dataset-level mode.
    fn source_selected_token_windows_with_loss_masks(
        &self,
        split: DatasetSplit,
        epoch_index: usize,
        absolute_step: usize,
        batch_size: usize,
        block_size: usize,
    ) -> Option<SourceSelectedBatch> {
        let windows = self.source_selected_token_windows(
            split,
            epoch_index,
            absolute_step,
            batch_size,
            block_size,
        )?;
        let masks = self.uses_target_loss_mask().then(|| {
            windows
                .iter()
                .map(|window| {
                    let mut mask = vec![0; block_size];
                    self.target_loss_mask_for_window(window, &mut mask);
                    mask
                })
                .collect::<Vec<_>>()
        });
        Some(SourceSelectedBatch {
            windows,
            loss_masks: masks,
        })
    }

    /// Return ruliad prompt/eval metadata aligned to a live source-selected step, if available.
    /// The random data loader requests this only when a verifier-reward policy auxiliary is
    /// explicitly enabled, keeping ordinary batch construction on the token-only hot path.
    fn source_selected_ruliad_policy_batch(
        &self,
        _split: DatasetSplit,
        _epoch_index: usize,
        _absolute_step: usize,
        _batch_size: usize,
        _stratified_difficulty_levels: usize,
    ) -> Option<RuliadPolicyBatch> {
        None
    }

    /// Return concrete source-selected token windows for a streaming TBPTT chunk, if the dataset
    /// can generate them without first materializing a full epoch. `chunk_index_in_document`
    /// identifies the chunk offset inside the logical document currently being streamed, while
    /// `absolute_step` is the feedback key that will later receive the chunk loss.
    fn source_selected_stream_token_windows(
        &self,
        _split: DatasetSplit,
        _epoch_index: usize,
        _absolute_step: usize,
        _chunk_index_in_document: usize,
        _batch_size: usize,
        _block_size: usize,
    ) -> Option<Vec<Vec<u32>>> {
        None
    }

    /// Return concrete source-selected token windows and optional precomputed loss masks for a
    /// streaming TBPTT chunk. Datasets with document-level target structure can override this to
    /// avoid deriving masks from an isolated chunk window.
    fn source_selected_stream_token_windows_with_loss_masks(
        &self,
        split: DatasetSplit,
        epoch_index: usize,
        absolute_step: usize,
        chunk_index_in_document: usize,
        batch_size: usize,
        block_size: usize,
    ) -> Option<SourceSelectedStreamBatch> {
        let windows = self.source_selected_stream_token_windows(
            split,
            epoch_index,
            absolute_step,
            chunk_index_in_document,
            batch_size,
            block_size,
        )?;
        let masks = self.uses_target_loss_mask().then(|| {
            windows
                .iter()
                .map(|window| {
                    let mut mask = vec![0; block_size];
                    self.target_loss_mask_for_window(window, &mut mask);
                    mask
                })
                .collect::<Vec<_>>()
        });
        Some(SourceSelectedStreamBatch {
            windows,
            loss_masks: masks,
            document_complete: false,
        })
    }

    /// Feed aggregate loss telemetry for a previously selected source bucket.
    fn record_source_selection_loss(
        &self,
        _absolute_step: usize,
        _loss: f32,
    ) -> Option<burn_dragon_universality::RuliadMetricSnapshot> {
        None
    }

    fn source_selection_snapshot(&self) -> Option<burn_dragon_universality::RuliadMetricSnapshot> {
        None
    }

    /// Whether sampled batches should include a per-target loss mask.
    fn uses_target_loss_mask(&self) -> bool {
        false
    }

    /// Fill a per-target loss mask for a sampled token window. `window` has length
    /// `block_size + 1`, while `mask` has length `block_size` and aligns with shifted targets.
    fn target_loss_mask_for_window(&self, _window: &[u32], _mask: &mut [i64]) -> bool {
        false
    }

    /// Number of tokens reserved for the training split from the start of the corpus.
    fn train_len(&self) -> usize;

    /// Maximum sequence length per sample.
    fn block_size(&self) -> usize;

    /// Number of sequences per batch.
    fn batch_size(&self) -> usize;

    /// Ratio used when determining train/validation split boundaries.
    fn train_split_ratio(&self) -> f32;

    /// Preferred logical document length, excluding the next-token target, when the dataset has
    /// hard semantic document boundaries that should inform TBPTT streaming and random window
    /// sampling.
    fn preferred_logical_document_tokens(&self, _split: DatasetSplit) -> Option<usize> {
        None
    }

    /// Provide the offset and span of the requested split.
    fn split_offset_and_span(&self, split: DatasetSplit) -> (usize, usize) {
        match split {
            DatasetSplit::Train => (0, self.train_len()),
            DatasetSplit::Val => {
                let tokens = self.token_count();
                let train_len = self.train_len();
                let remaining = tokens.saturating_sub(train_len);
                if remaining <= self.block_size() + 1 {
                    (0, train_len)
                } else {
                    (train_len, remaining)
                }
            }
        }
    }

    /// Number of steps per epoch for a given split (defaults derived from token counts).
    fn steps_per_epoch(&self, split: DatasetSplit) -> usize {
        let (_offset, span) = self.split_offset_and_span(split);
        let tokens_per_step = self.block_size() * self.batch_size();
        if tokens_per_step == 0 {
            return 1;
        }
        let steps = span.div_ceil(tokens_per_step);
        steps.max(1)
    }

    /// Decode token ids back into text.
    fn decode(&self, tokens: &[i64]) -> String {
        let ids: Vec<u32> = tokens
            .iter()
            .filter_map(|&tok| (tok >= 0).then_some(tok as u32))
            .collect();
        self.tokenizer().decode(&ids)
    }
}

/// Sample a random batch from any dataset implementing [`TokenSequenceDataset`].
pub fn sample_batch<B: Backend, T: TokenSequenceDataset + ?Sized>(
    dataset: &T,
    split: DatasetSplit,
    device: &B::Device,
) -> SequenceBatch<B> {
    sample_batch_with_shape::<B, T>(
        dataset,
        split,
        dataset.batch_size(),
        dataset.block_size(),
        None,
        0,
        device,
    )
}

#[allow(clippy::too_many_arguments)]
fn fill_sampled_logical_document_row<T: TokenSequenceDataset + ?Sized>(
    dataset: &T,
    split: DatasetSplit,
    epoch_index: usize,
    absolute_step: usize,
    seed: u64,
    offset: usize,
    document_span: usize,
    num_documents: usize,
    max_start_in_document: usize,
    source_selected_documents: Option<&Vec<usize>>,
    batch_idx: usize,
    block_size: usize,
    input_row: &mut [i64],
    target_row: &mut [i64],
    mask_row: Option<&mut [i64]>,
) {
    let mut rng = deterministic_row_rng(seed, split, epoch_index, absolute_step, batch_idx);
    let doc_index = source_selected_documents
        .and_then(|indices| indices.get(batch_idx))
        .copied()
        .unwrap_or_else(|| {
            if num_documents <= 1 {
                0
            } else {
                rng.gen_range(0..num_documents)
            }
        });
    let start_in_document = if max_start_in_document == 0 {
        0
    } else {
        rng.gen_range(0..=max_start_in_document)
    };
    let start = offset + doc_index.saturating_mul(document_span) + start_in_document;
    let mut sample = vec![0u32; block_size + 1];
    dataset.copy_token_range_with_epoch(split, epoch_index, start, &mut sample);
    fill_rows_from_window(
        dataset, &sample, block_size, input_row, target_row, mask_row,
    );
}

#[allow(clippy::too_many_arguments)]
fn fill_sampled_flat_row<T: TokenSequenceDataset + ?Sized>(
    dataset: &T,
    split: DatasetSplit,
    epoch_index: usize,
    absolute_step: usize,
    seed: u64,
    batch_idx: usize,
    offset: usize,
    span: usize,
    block_size: usize,
    input_row: &mut [i64],
    target_row: &mut [i64],
    mask_row: Option<&mut [i64]>,
) {
    let mut rng = deterministic_row_rng(seed, split, epoch_index, absolute_step, batch_idx);
    let max_start = span.saturating_sub(block_size + 1);
    let start_offset = if max_start == 0 {
        0
    } else {
        rng.gen_range(0..=max_start)
    };
    let start = offset + start_offset;
    let mut sample = vec![0u32; block_size + 1];
    dataset.copy_token_range_with_epoch(split, epoch_index, start, &mut sample);
    fill_rows_from_window(
        dataset, &sample, block_size, input_row, target_row, mask_row,
    );
}

fn fill_rows_from_window<T: TokenSequenceDataset + ?Sized>(
    dataset: &T,
    window: &[u32],
    block_size: usize,
    input_row: &mut [i64],
    target_row: &mut [i64],
    mask_row: Option<&mut [i64]>,
) {
    for t in 0..block_size {
        input_row[t] = window[t] as i64;
        target_row[t] = window[t + 1] as i64;
    }
    if let Some(mask_row) = mask_row {
        dataset.target_loss_mask_for_window(window, mask_row);
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_stream_row_from_document<T: TokenSequenceDataset + ?Sized>(
    dataset: &T,
    split: DatasetSplit,
    epoch_index: usize,
    document_start: usize,
    logical_document_tokens: usize,
    chunk_index_in_document: usize,
    block_size: usize,
    input_row: &mut [i64],
    target_row: &mut [i64],
    mask_row: Option<&mut [i64]>,
) {
    let chunk_offset = chunk_index_in_document.saturating_mul(block_size);
    let valid_pairs = logical_document_tokens
        .saturating_sub(chunk_offset)
        .min(block_size);
    assert!(
        valid_pairs > 0,
        "stream chunk must contain at least one next-token pair"
    );

    let copied_tokens = valid_pairs + 1;
    let mut sample = vec![0u32; block_size + 1];
    dataset.copy_token_range_with_epoch(
        split,
        epoch_index,
        document_start + chunk_offset,
        &mut sample[..copied_tokens],
    );
    let terminal = sample[copied_tokens - 1];
    sample[copied_tokens..].fill(terminal);

    match mask_row {
        Some(mask_row) => {
            for t in 0..block_size {
                input_row[t] = sample[t] as i64;
                target_row[t] = sample[t + 1] as i64;
            }
            if !dataset.target_loss_mask_for_window(&sample, mask_row)
                && !dataset.uses_target_loss_mask()
            {
                mask_row[..valid_pairs].fill(1);
            }
            mask_row[valid_pairs..].fill(0);
        }
        None => fill_rows_from_window(dataset, &sample, block_size, input_row, target_row, None),
    }
}

fn deterministic_row_rng(
    seed: u64,
    split: DatasetSplit,
    epoch_index: usize,
    absolute_step: usize,
    batch_idx: usize,
) -> StdRng {
    let split_tag = match split {
        DatasetSplit::Train => 0xA076_1D64_78BD_642F,
        DatasetSplit::Val => 0xE703_7ED1_A0B4_28DB,
    };
    let mut mixed = seed ^ split_tag;
    for value in [epoch_index, absolute_step, batch_idx] {
        mixed = mixed
            .wrapping_add(value as u64)
            .wrapping_add(0x9E37_79B9_7F4A_7C15);
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        mixed ^= mixed >> 31;
    }
    StdRng::seed_from_u64(mixed)
}

/// Sample a random batch with an explicit batch/block shape from any dataset implementing
/// [`TokenSequenceDataset`].
#[derive(Clone, Copy, Debug)]
struct HostBatchRequest {
    split: DatasetSplit,
    batch_size: usize,
    block_size: usize,
    epoch_index: usize,
    absolute_step: usize,
    seed: u64,
    source_selection_enabled: bool,
    include_ruliad_policy_batch: bool,
    ruliad_policy_stratified_difficulty_levels: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct RuliadPolicyBatchSchedule {
    always: bool,
    cadences: crate::config::RuliadPolicyBatchCadences,
}

impl RuliadPolicyBatchSchedule {
    fn always() -> Self {
        Self {
            always: true,
            ..Self::default()
        }
    }

    fn training(supervision: crate::config::RuliadSupervisionConfig) -> Self {
        Self {
            always: false,
            cadences: supervision.policy_batch_cadences(),
        }
    }

    fn includes(self, absolute_step: usize) -> bool {
        self.always || self.cadences.includes(absolute_step)
    }
}

fn sample_host_batch_with_shape<T>(dataset: &T, request: HostBatchRequest) -> HostSequenceBatch
where
    T: TokenSequenceDataset + ?Sized,
{
    let HostBatchRequest {
        split,
        batch_size,
        block_size,
        epoch_index,
        absolute_step,
        seed,
        source_selection_enabled,
        include_ruliad_policy_batch,
        ruliad_policy_stratified_difficulty_levels,
    } = request;
    let prof_enabled = crate::train::profile::enabled();
    let cpu_start = prof_enabled.then(Instant::now);
    let (offset, span) = dataset.split_offset_and_span(split);

    let mut inputs = vec![0i64; batch_size * block_size];
    let mut targets = vec![0i64; batch_size * block_size];
    let mut loss_mask = dataset
        .uses_target_loss_mask()
        .then(|| vec![0i64; batch_size * block_size]);
    let mut ruliad_policy_batch = None;

    if source_selection_enabled
        && let Some(SourceSelectedBatch {
            windows: source_windows,
            loss_masks: source_loss_masks,
        }) = dataset.source_selected_token_windows_with_loss_masks(
            split,
            epoch_index,
            absolute_step,
            batch_size,
            block_size,
        )
    {
        assert_eq!(
            source_windows.len(),
            batch_size,
            "source-selected token windows must match batch size"
        );
        if let Some(source_loss_masks) = source_loss_masks.as_ref() {
            assert_eq!(
                source_loss_masks.len(),
                batch_size,
                "source-selected token masks must match batch size"
            );
        }
        for (batch_idx, window) in source_windows.iter().take(batch_size).enumerate() {
            assert!(
                window.len() > block_size,
                "source-selected token window must include block_size + 1 tokens"
            );
            for t in 0..block_size {
                inputs[batch_idx * block_size + t] = window[t] as i64;
                targets[batch_idx * block_size + t] = window[t + 1] as i64;
            }
            if let Some(mask) = loss_mask.as_mut() {
                let mask_row = &mut mask[batch_idx * block_size..(batch_idx + 1) * block_size];
                if let Some(source_loss_masks) = source_loss_masks.as_ref() {
                    mask_row.copy_from_slice(
                        source_loss_masks
                            .get(batch_idx)
                            .expect("source-selected token mask row must exist"),
                    );
                } else {
                    dataset.target_loss_mask_for_window(window, mask_row);
                }
            }
        }
    } else if let Some(logical_document_tokens) = dataset.preferred_logical_document_tokens(split) {
        let document_span = logical_document_tokens.saturating_add(1);
        let num_documents = (span / document_span).max(1);
        let source_selected_documents = source_selection_enabled
            .then(|| {
                dataset.source_selected_document_indices(
                    split,
                    epoch_index,
                    absolute_step,
                    batch_size,
                )
            })
            .flatten();
        let max_start_in_document = logical_document_tokens
            .saturating_sub(block_size)
            .min(document_span.saturating_sub(block_size + 1));
        #[cfg(feature = "train")]
        {
            if let Some(mask) = loss_mask.as_mut() {
                inputs
                    .par_chunks_mut(block_size)
                    .zip(targets.par_chunks_mut(block_size))
                    .zip(mask.par_chunks_mut(block_size))
                    .enumerate()
                    .for_each(|(batch_idx, ((input_row, target_row), mask_row))| {
                        fill_sampled_logical_document_row(
                            dataset,
                            split,
                            epoch_index,
                            absolute_step,
                            seed,
                            offset,
                            document_span,
                            num_documents,
                            max_start_in_document,
                            source_selected_documents.as_ref(),
                            batch_idx,
                            block_size,
                            input_row,
                            target_row,
                            Some(mask_row),
                        );
                    });
            } else {
                inputs
                    .par_chunks_mut(block_size)
                    .zip(targets.par_chunks_mut(block_size))
                    .enumerate()
                    .for_each(|(batch_idx, (input_row, target_row))| {
                        fill_sampled_logical_document_row(
                            dataset,
                            split,
                            epoch_index,
                            absolute_step,
                            seed,
                            offset,
                            document_span,
                            num_documents,
                            max_start_in_document,
                            source_selected_documents.as_ref(),
                            batch_idx,
                            block_size,
                            input_row,
                            target_row,
                            None,
                        );
                    });
            }
        }
        #[cfg(not(feature = "train"))]
        for batch_idx in 0..batch_size {
            let mut rng = deterministic_row_rng(seed, split, epoch_index, absolute_step, batch_idx);
            let mut sample = vec![0u32; block_size + 1];
            let doc_index = source_selected_documents
                .as_ref()
                .and_then(|indices| indices.get(batch_idx))
                .copied()
                .unwrap_or_else(|| {
                    if num_documents <= 1 {
                        0
                    } else {
                        rng.gen_range(0..num_documents)
                    }
                });
            let start_in_document = if max_start_in_document == 0 {
                0
            } else {
                rng.gen_range(0..=max_start_in_document)
            };
            let start = offset + doc_index.saturating_mul(document_span) + start_in_document;
            dataset.copy_token_range_with_epoch(split, epoch_index, start, &mut sample);
            for t in 0..block_size {
                inputs[batch_idx * block_size + t] = sample[t] as i64;
                targets[batch_idx * block_size + t] = sample[t + 1] as i64;
            }
            if let Some(mask) = loss_mask.as_mut() {
                dataset.target_loss_mask_for_window(
                    &sample,
                    &mut mask[batch_idx * block_size..(batch_idx + 1) * block_size],
                );
            }
        }
    } else {
        #[cfg(feature = "train")]
        {
            if let Some(mask) = loss_mask.as_mut() {
                inputs
                    .par_chunks_mut(block_size)
                    .zip(targets.par_chunks_mut(block_size))
                    .zip(mask.par_chunks_mut(block_size))
                    .enumerate()
                    .for_each(|(batch_idx, ((input_row, target_row), mask_row))| {
                        fill_sampled_flat_row(
                            dataset,
                            split,
                            epoch_index,
                            absolute_step,
                            seed,
                            batch_idx,
                            offset,
                            span,
                            block_size,
                            input_row,
                            target_row,
                            Some(mask_row),
                        );
                    });
            } else {
                inputs
                    .par_chunks_mut(block_size)
                    .zip(targets.par_chunks_mut(block_size))
                    .enumerate()
                    .for_each(|(batch_idx, (input_row, target_row))| {
                        fill_sampled_flat_row(
                            dataset,
                            split,
                            epoch_index,
                            absolute_step,
                            seed,
                            batch_idx,
                            offset,
                            span,
                            block_size,
                            input_row,
                            target_row,
                            None,
                        );
                    });
            }
        }
        #[cfg(not(feature = "train"))]
        for batch_idx in 0..batch_size {
            let mut rng = deterministic_row_rng(seed, split, epoch_index, absolute_step, batch_idx);
            let mut sample = vec![0u32; block_size + 1];
            let max_start = span.saturating_sub(block_size + 1);
            let start_offset = if max_start == 0 {
                0
            } else {
                rng.gen_range(0..=max_start)
            };
            let start = offset + start_offset;
            dataset.copy_token_range_with_epoch(split, epoch_index, start, &mut sample);
            for t in 0..block_size {
                inputs[batch_idx * block_size + t] = sample[t] as i64;
                targets[batch_idx * block_size + t] = sample[t + 1] as i64;
            }
            if let Some(mask) = loss_mask.as_mut() {
                dataset.target_loss_mask_for_window(
                    &sample,
                    &mut mask[batch_idx * block_size..(batch_idx + 1) * block_size],
                );
            }
        }
    }

    if source_selection_enabled && include_ruliad_policy_batch && ruliad_policy_batch.is_none() {
        ruliad_policy_batch = dataset
            .source_selected_ruliad_policy_batch(
                split,
                epoch_index,
                absolute_step,
                batch_size,
                ruliad_policy_stratified_difficulty_levels,
            )
            .map(Arc::new);
    }

    HostSequenceBatch {
        inputs,
        targets,
        loss_mask,
        ruliad_policy_batch,
        dataloader_cpu_ns: cpu_start
            .map(|start| start.elapsed().as_nanos())
            .unwrap_or_default(),
        reset_stream_state: false,
    }
}

pub fn sample_batch_with_shape<B: Backend, T: TokenSequenceDataset + ?Sized>(
    dataset: &T,
    split: DatasetSplit,
    batch_size: usize,
    block_size: usize,
    summary_event_token_ids: Option<&[u32]>,
    epoch_index: usize,
    device: &B::Device,
) -> SequenceBatch<B> {
    // One-off callers retain the public random-sampling contract. Training and
    // validation loaders pass their configured seeds through their iterators.
    let seed = thread_rng().next_u64();
    let host = sample_host_batch_with_shape(
        dataset,
        HostBatchRequest {
            split,
            batch_size,
            block_size,
            epoch_index,
            absolute_step: 0,
            seed,
            source_selection_enabled: true,
            include_ruliad_policy_batch: false,
            ruliad_policy_stratified_difficulty_levels: 0,
        },
    );
    if crate::train::profile::enabled() {
        crate::train::profile::record_dataloader_foreground_wait(host.dataloader_cpu_ns);
    }
    finalize_host_batch_on_device::<B>(
        host,
        batch_size,
        block_size,
        summary_event_token_ids,
        device,
    )
}

/// Batched token inputs and targets for language modeling.
#[derive(Clone)]
pub struct SequenceBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
    pub loss_mask: Option<Tensor<B, 2, Int>>,
    pub summary_event_mask: Option<Tensor<B, 2, Int>>,
    pub ruliad_policy_batch: Option<Arc<RuliadPolicyBatch>>,
    pub reset_stream_state: bool,
}

struct HostSequenceBatch {
    inputs: Vec<i64>,
    targets: Vec<i64>,
    loss_mask: Option<Vec<i64>>,
    ruliad_policy_batch: Option<Arc<RuliadPolicyBatch>>,
    dataloader_cpu_ns: u128,
    reset_stream_state: bool,
}

struct RandomPrefetch {
    receiver: Option<Receiver<(usize, HostSequenceBatch)>>,
    workers: Vec<JoinHandle<()>>,
    pending: BTreeMap<usize, HostSequenceBatch>,
    next_index: usize,
}

impl RandomPrefetch {
    #[allow(clippy::too_many_arguments)]
    fn spawn(
        dataset: Arc<dyn TokenSequenceDataset>,
        split: DatasetSplit,
        batch_size: usize,
        block_size: usize,
        steps_per_epoch: usize,
        absolute_step_start: usize,
        total_steps: Option<usize>,
        depth: usize,
        workers: usize,
        seed: u64,
        source_selection_enabled: bool,
        ruliad_policy_batch_schedule: RuliadPolicyBatchSchedule,
        ruliad_policy_stratified_difficulty_levels: usize,
    ) -> Self {
        let worker_count = workers.max(1);
        let current_epoch = absolute_step_start / steps_per_epoch.max(1);
        let uses_live_source_selection =
            source_selection_enabled && dataset.uses_live_source_selection();
        if !uses_live_source_selection {
            dataset.prepare_epoch(split, current_epoch);
            dataset.prefetch_epoch(split, current_epoch.saturating_add(1));
            dataset.prefetch_epoch(split, current_epoch.saturating_add(2));
        }
        let (sender, receiver) =
            sync_channel::<(usize, HostSequenceBatch)>(depth.max(worker_count));
        let next_task = Arc::new(AtomicUsize::new(absolute_step_start));
        let mut handles = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let sender = sender.clone();
            let dataset = Arc::clone(&dataset);
            let next_task = Arc::clone(&next_task);
            handles.push(thread::spawn(move || {
                loop {
                    let task_index = next_task.fetch_add(1, Ordering::Relaxed);
                    if let Some(limit) = total_steps
                        && task_index >= limit
                    {
                        break;
                    }
                    let epoch_index = task_index / steps_per_epoch.max(1);
                    if !uses_live_source_selection {
                        dataset.prefetch_epoch(split, epoch_index.saturating_add(1));
                    }
                    let batch = sample_host_batch_with_shape(
                        dataset.as_ref(),
                        HostBatchRequest {
                            split,
                            batch_size,
                            block_size,
                            epoch_index,
                            absolute_step: task_index,
                            seed,
                            source_selection_enabled,
                            include_ruliad_policy_batch: ruliad_policy_batch_schedule
                                .includes(task_index),
                            ruliad_policy_stratified_difficulty_levels,
                        },
                    );
                    if sender.send((task_index, batch)).is_err() {
                        return;
                    }
                }
            }));
        }
        drop(sender);
        let mut prefetch = Self {
            receiver: Some(receiver),
            workers: handles,
            pending: BTreeMap::new(),
            next_index: absolute_step_start,
        };
        prefetch.prime(worker_count.min(depth.max(1)));
        prefetch
    }

    fn seek_to(&mut self, absolute_step: usize) {
        self.next_index = absolute_step;
        self.pending.retain(|index, _| *index >= absolute_step);
    }

    fn recv(&mut self) -> Option<HostSequenceBatch> {
        if let Some(batch) = self.pending.remove(&self.next_index) {
            self.next_index = self.next_index.saturating_add(1);
            return Some(batch);
        }
        loop {
            let (index, batch) = self.receiver.as_ref()?.recv().ok()?;
            if index == self.next_index {
                self.next_index = self.next_index.saturating_add(1);
                return Some(batch);
            }
            self.pending.insert(index, batch);
        }
    }

    fn prime(&mut self, target_ready: usize) {
        if target_ready == 0 {
            return;
        }
        while self.contiguous_ready() < target_ready {
            let Some((index, batch)) = self
                .receiver
                .as_ref()
                .and_then(|receiver| receiver.recv().ok())
            else {
                break;
            };
            self.pending.insert(index, batch);
        }
    }

    fn contiguous_ready(&self) -> usize {
        let mut count = 0usize;
        let mut index = self.next_index;
        while self.pending.contains_key(&index) {
            count = count.saturating_add(1);
            index = index.saturating_add(1);
        }
        count
    }
}

impl Drop for RandomPrefetch {
    fn drop(&mut self) {
        let _ = self.receiver.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

impl<B: Backend> SequenceBatch<B> {
    pub fn new(
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> Self {
        Self {
            inputs,
            targets,
            loss_mask: None,
            summary_event_mask,
            ruliad_policy_batch: None,
            reset_stream_state: false,
        }
    }

    pub fn with_loss_mask(mut self, loss_mask: Option<Tensor<B, 2, Int>>) -> Self {
        self.loss_mask = loss_mask;
        self
    }

    pub fn with_ruliad_policy_batch(
        mut self,
        ruliad_policy_batch: Option<Arc<RuliadPolicyBatch>>,
    ) -> Self {
        self.ruliad_policy_batch = ruliad_policy_batch;
        self
    }

    pub fn with_reset_stream_state(mut self, reset_stream_state: bool) -> Self {
        self.reset_stream_state = reset_stream_state;
        self
    }
}

fn dataset_prefetch_depth() -> usize {
    std::env::var("DragonModel_DATASET_PREFETCH_DEPTH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8)
}

fn dataset_prefetch_workers() -> usize {
    std::env::var("DragonModel_DATASET_PREFETCH_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| {
            let cpus = std::thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(4);
            if cpus >= 24 {
                8
            } else if cpus >= 12 {
                4
            } else {
                2
            }
        })
}

fn live_source_selection_prefetch_depth() -> usize {
    if let Some(configured) = std::env::var("DragonModel_RULIAD_SOURCE_SELECTION_PREFETCH_DEPTH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    {
        return configured;
    }
    dataset_prefetch_depth().max(16)
}

fn finalize_host_batch_on_device<B: Backend>(
    host: HostSequenceBatch,
    batch_size: usize,
    block_size: usize,
    summary_event_token_ids: Option<&[u32]>,
    device: &B::Device,
) -> SequenceBatch<B> {
    let HostSequenceBatch {
        inputs,
        targets,
        loss_mask,
        ruliad_policy_batch,
        dataloader_cpu_ns,
        reset_stream_state,
    } = host;
    let prof_enabled = crate::train::profile::enabled();
    let tensor_copy_start = prof_enabled.then(Instant::now);
    let summary_event_mask = summary_event_mask_tensor::<B>(
        &inputs,
        batch_size,
        block_size,
        summary_event_token_ids,
        device,
    );
    let inputs_tensor =
        Tensor::<B, 2, Int>::from_data(TensorData::new(inputs, [batch_size, block_size]), device);
    let targets_tensor =
        Tensor::<B, 2, Int>::from_data(TensorData::new(targets, [batch_size, block_size]), device);
    let loss_mask_tensor = loss_mask.map(|mask| {
        Tensor::<B, 2, Int>::from_data(TensorData::new(mask, [batch_size, block_size]), device)
    });
    let tensor_copy_ns = tensor_copy_start
        .map(|start| start.elapsed().as_nanos())
        .unwrap_or_default();

    if prof_enabled {
        let values = batch_size.saturating_mul(block_size);
        let tensor_count = 2 + usize::from(loss_mask_tensor.is_some());
        let copy_bytes = (values
            .saturating_mul(tensor_count)
            .saturating_mul(size_of::<i64>())) as u128;
        crate::train::profile::record_dataloader(dataloader_cpu_ns, tensor_copy_ns, copy_bytes, 0);
    }

    SequenceBatch::new(inputs_tensor, targets_tensor, summary_event_mask)
        .with_loss_mask(loss_mask_tensor)
        .with_ruliad_policy_batch(ruliad_policy_batch)
        .with_reset_stream_state(reset_stream_state)
}

/// Data loader that produces random sequences from any `TokenSequenceDataset`.
pub struct RandomDataLoader<B: Backend> {
    dataset: Arc<dyn TokenSequenceDataset>,
    split: DatasetSplit,
    device: B::Device,
    batch_size: usize,
    block_size: usize,
    steps_per_epoch: usize,
    total_steps: Option<usize>,
    consumed_steps: Option<Arc<AtomicUsize>>,
    summary_event_token_ids: Option<Vec<u32>>,
    ruliad_policy_batch_schedule: RuliadPolicyBatchSchedule,
    ruliad_policy_stratified_difficulty_levels: usize,
    prefetch: Arc<Mutex<Option<RandomPrefetch>>>,
    seed: u64,
    source_selection_enabled: bool,
}

pub struct StreamingDataLoader<B: Backend> {
    dataset: Arc<dyn TokenSequenceDataset>,
    split: DatasetSplit,
    device: B::Device,
    batch_size: usize,
    block_size: usize,
    steps_per_epoch: usize,
    total_steps: Option<usize>,
    consumed_steps: Option<Arc<AtomicUsize>>,
    summary_event_token_ids: Option<Vec<u32>>,
    ruliad_policy_batch_schedule: RuliadPolicyBatchSchedule,
    ruliad_policy_stratified_difficulty_levels: usize,
    logical_document_tokens: usize,
    seed: u64,
    source_selection_enabled: bool,
}

impl<B: Backend> Clone for RandomDataLoader<B> {
    fn clone(&self) -> Self {
        Self {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: self.device.clone(),
            batch_size: self.batch_size,
            block_size: self.block_size,
            steps_per_epoch: self.steps_per_epoch,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
            summary_event_token_ids: self.summary_event_token_ids.clone(),
            ruliad_policy_batch_schedule: self.ruliad_policy_batch_schedule,
            ruliad_policy_stratified_difficulty_levels: self
                .ruliad_policy_stratified_difficulty_levels,
            prefetch: Arc::clone(&self.prefetch),
            seed: self.seed,
            source_selection_enabled: self.source_selection_enabled,
        }
    }
}

impl<B: Backend> Clone for StreamingDataLoader<B> {
    fn clone(&self) -> Self {
        Self {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: self.device.clone(),
            batch_size: self.batch_size,
            block_size: self.block_size,
            steps_per_epoch: self.steps_per_epoch,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
            summary_event_token_ids: self.summary_event_token_ids.clone(),
            ruliad_policy_batch_schedule: self.ruliad_policy_batch_schedule,
            ruliad_policy_stratified_difficulty_levels: self
                .ruliad_policy_stratified_difficulty_levels,
            logical_document_tokens: self.logical_document_tokens,
            seed: self.seed,
            source_selection_enabled: self.source_selection_enabled,
        }
    }
}

impl<B: Backend> RandomDataLoader<B> {
    pub fn new<T>(
        dataset: Arc<T>,
        split: DatasetSplit,
        device: &B::Device,
        steps_per_epoch: usize,
        total_steps: Option<usize>,
    ) -> Self
    where
        T: TokenSequenceDataset + 'static,
    {
        let dataset: Arc<dyn TokenSequenceDataset> = dataset;
        let steps_per_epoch = steps_per_epoch.max(1);
        let total_steps = total_steps.filter(|value| *value > 0);
        let consumed_steps = total_steps.as_ref().map(|_| Arc::new(AtomicUsize::new(0)));
        let batch_size = dataset.batch_size().max(1);
        let block_size = dataset.block_size().max(1);

        Self {
            dataset,
            split,
            device: device.clone(),
            batch_size,
            block_size,
            steps_per_epoch,
            total_steps,
            consumed_steps,
            summary_event_token_ids: None,
            ruliad_policy_batch_schedule: RuliadPolicyBatchSchedule::default(),
            ruliad_policy_stratified_difficulty_levels: 0,
            prefetch: Arc::new(Mutex::new(None)),
            seed: 0,
            source_selection_enabled: true,
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.prefetch = Arc::new(Mutex::new(None));
        self
    }

    pub fn with_source_selection_enabled(mut self, enabled: bool) -> Self {
        self.source_selection_enabled = enabled;
        self.prefetch = Arc::new(Mutex::new(None));
        self
    }

    pub fn with_summary_event_token_ids(
        mut self,
        summary_event_token_ids: Option<Vec<u32>>,
    ) -> Self {
        self.summary_event_token_ids = summary_event_token_ids;
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self.prefetch = Arc::new(Mutex::new(None));
        self
    }

    pub fn with_ruliad_policy_batch(mut self, enabled: bool) -> Self {
        self.ruliad_policy_batch_schedule = if enabled {
            RuliadPolicyBatchSchedule::always()
        } else {
            RuliadPolicyBatchSchedule::default()
        };
        self.prefetch = Arc::new(Mutex::new(None));
        self
    }

    pub fn with_ruliad_policy_supervision(
        mut self,
        supervision: crate::config::RuliadSupervisionConfig,
    ) -> Self {
        self.ruliad_policy_batch_schedule = if supervision.needs_ruliad_policy_batch() {
            RuliadPolicyBatchSchedule::training(supervision)
        } else {
            RuliadPolicyBatchSchedule::default()
        };
        self.prefetch = Arc::new(Mutex::new(None));
        self
    }

    pub fn with_ruliad_policy_stratified_difficulty_levels(mut self, levels: usize) -> Self {
        self.ruliad_policy_stratified_difficulty_levels = levels;
        self.prefetch = Arc::new(Mutex::new(None));
        self
    }

    pub fn with_initial_consumed_steps(self, initial_steps: usize) -> Self {
        if let (Some(limit), Some(consumed_steps)) =
            (self.total_steps, self.consumed_steps.as_ref())
        {
            consumed_steps.store(initial_steps.min(limit), Ordering::Relaxed);
        }
        self
    }
}

fn resolve_stream_logical_document_tokens(
    dataset: &dyn TokenSequenceDataset,
    split: DatasetSplit,
    requested_min_logical_block_size: Option<usize>,
) -> usize {
    let block_size = dataset.block_size().max(1);
    if let Some(document_tokens) = dataset.preferred_logical_document_tokens(split) {
        return document_tokens.max(block_size);
    }
    let (_, span) = dataset.split_offset_and_span(split);
    let max_inputs = span.saturating_sub(1);
    let desired = requested_min_logical_block_size
        .unwrap_or(block_size)
        .max(block_size);
    let rounded_up = desired.div_ceil(block_size).saturating_mul(block_size);
    let max_multiple = (max_inputs / block_size).max(1).saturating_mul(block_size);
    rounded_up.min(max_multiple).max(block_size)
}

fn gcd_usize(mut lhs: usize, mut rhs: usize) -> usize {
    while rhs != 0 {
        let remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    lhs
}

fn resolve_stream_document_permutation(
    seed: u64,
    epoch_index: usize,
    num_documents: usize,
) -> (usize, usize) {
    if num_documents <= 1 {
        return (0, 1);
    }
    let mixed_seed = seed
        ^ (epoch_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (num_documents as u64).rotate_left(17);
    let mut rng = StdRng::seed_from_u64(mixed_seed);
    let document_start = rng.gen_range(0..num_documents);
    let document_stride = loop {
        let candidate = rng.gen_range(1..num_documents);
        if gcd_usize(candidate, num_documents) == 1 {
            break candidate;
        }
    };
    (document_start, document_stride)
}

impl<B: Backend> StreamingDataLoader<B> {
    pub fn new<T>(
        dataset: Arc<T>,
        split: DatasetSplit,
        device: &B::Device,
        steps_per_epoch: usize,
        total_steps: Option<usize>,
        min_logical_block_size: Option<usize>,
        seed: u64,
    ) -> Self
    where
        T: TokenSequenceDataset + 'static,
    {
        let dataset: Arc<dyn TokenSequenceDataset> = dataset;
        let steps_per_epoch = steps_per_epoch.max(1);
        let total_steps = total_steps.filter(|value| *value > 0);
        let consumed_steps = total_steps.as_ref().map(|_| Arc::new(AtomicUsize::new(0)));
        let logical_document_tokens =
            resolve_stream_logical_document_tokens(dataset.as_ref(), split, min_logical_block_size);
        let batch_size = dataset.batch_size().max(1);
        let block_size = dataset.block_size().max(1);

        Self {
            dataset,
            split,
            device: device.clone(),
            batch_size,
            block_size,
            steps_per_epoch,
            total_steps,
            consumed_steps,
            summary_event_token_ids: None,
            ruliad_policy_batch_schedule: RuliadPolicyBatchSchedule::default(),
            ruliad_policy_stratified_difficulty_levels: 0,
            logical_document_tokens,
            seed,
            source_selection_enabled: true,
        }
    }

    pub fn with_source_selection_enabled(mut self, enabled: bool) -> Self {
        self.source_selection_enabled = enabled;
        self
    }

    pub fn with_summary_event_token_ids(
        mut self,
        summary_event_token_ids: Option<Vec<u32>>,
    ) -> Self {
        self.summary_event_token_ids = summary_event_token_ids;
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    pub fn with_ruliad_policy_batch(mut self, enabled: bool) -> Self {
        self.ruliad_policy_batch_schedule = if enabled {
            RuliadPolicyBatchSchedule::always()
        } else {
            RuliadPolicyBatchSchedule::default()
        };
        self
    }

    pub fn with_ruliad_policy_supervision(
        mut self,
        supervision: crate::config::RuliadSupervisionConfig,
    ) -> Self {
        self.ruliad_policy_batch_schedule = if supervision.needs_ruliad_policy_batch() {
            RuliadPolicyBatchSchedule::training(supervision)
        } else {
            RuliadPolicyBatchSchedule::default()
        };
        self
    }

    pub fn with_ruliad_policy_stratified_difficulty_levels(mut self, levels: usize) -> Self {
        self.ruliad_policy_stratified_difficulty_levels = levels;
        self
    }

    pub fn with_initial_consumed_steps(self, initial_steps: usize) -> Self {
        if let (Some(limit), Some(consumed_steps)) =
            (self.total_steps, self.consumed_steps.as_ref())
        {
            consumed_steps.store(initial_steps.min(limit), Ordering::Relaxed);
        }
        self
    }
}

impl<B> DataLoader<B, SequenceBatch<B>> for RandomDataLoader<B>
where
    B: Backend + 'static,
    B::Device: Clone,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<SequenceBatch<B>> + 'a> {
        let steps_total =
            if let (Some(limit), Some(consumed)) = (self.total_steps, &self.consumed_steps) {
                let used = consumed.load(Ordering::Relaxed);
                if used >= limit {
                    0
                } else {
                    (limit - used).min(self.steps_per_epoch)
                }
            } else {
                self.steps_per_epoch
            };
        let absolute_step_start = self
            .consumed_steps
            .as_ref()
            .map(|counter| counter.load(Ordering::Relaxed))
            .unwrap_or_default();
        let uses_live_source_selection =
            self.source_selection_enabled && self.dataset.uses_live_source_selection();
        let prefetch_depth = if uses_live_source_selection {
            live_source_selection_prefetch_depth()
        } else {
            dataset_prefetch_depth()
        };
        let prefetch_workers = if uses_live_source_selection {
            dataset_prefetch_workers().min(prefetch_depth.max(1))
        } else {
            dataset_prefetch_workers()
        };
        let use_persistent_prefetch =
            prefetch_depth > 0 && steps_total > 1 && self.split == DatasetSplit::Train;
        if use_persistent_prefetch {
            let mut slot = self.prefetch.lock().expect("random prefetch lock");
            if slot.is_none() {
                *slot = Some(RandomPrefetch::spawn(
                    Arc::clone(&self.dataset),
                    self.split,
                    self.batch_size,
                    self.block_size,
                    self.steps_per_epoch,
                    absolute_step_start,
                    self.total_steps,
                    prefetch_depth,
                    prefetch_workers,
                    self.seed,
                    self.source_selection_enabled,
                    self.ruliad_policy_batch_schedule,
                    self.ruliad_policy_stratified_difficulty_levels,
                ));
            } else if let Some(prefetch) = slot.as_mut() {
                prefetch.seek_to(absolute_step_start);
            }
        }

        Box::new(RandomIterator {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: self.device.clone(),
            batch_size: self.batch_size,
            block_size: self.block_size,
            steps_total,
            step: 0,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.clone(),
            summary_event_token_ids: self.summary_event_token_ids.clone(),
            ruliad_policy_batch_schedule: self.ruliad_policy_batch_schedule,
            ruliad_policy_stratified_difficulty_levels: self
                .ruliad_policy_stratified_difficulty_levels,
            seed: self.seed,
            source_selection_enabled: self.source_selection_enabled,
            epoch_index: self
                .consumed_steps
                .as_ref()
                .map(|counter| counter.load(Ordering::Relaxed) / self.steps_per_epoch.max(1))
                .unwrap_or_default(),
            prefetch: use_persistent_prefetch.then(|| Arc::clone(&self.prefetch)),
        })
    }

    fn num_items(&self) -> usize {
        self.steps_per_epoch
    }

    fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, SequenceBatch<B>>> {
        Arc::new(Self {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: device.clone(),
            batch_size: self.batch_size,
            block_size: self.block_size,
            steps_per_epoch: self.steps_per_epoch,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
            summary_event_token_ids: self.summary_event_token_ids.clone(),
            ruliad_policy_batch_schedule: self.ruliad_policy_batch_schedule,
            ruliad_policy_stratified_difficulty_levels: self
                .ruliad_policy_stratified_difficulty_levels,
            prefetch: Arc::clone(&self.prefetch),
            seed: self.seed,
            source_selection_enabled: self.source_selection_enabled,
        })
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, SequenceBatch<B>>> {
        let end = end.min(self.steps_per_epoch);
        let start = start.min(end);
        let steps = (end - start).max(1);
        let total_steps = self.total_steps.map(|limit| limit.min(steps));
        let consumed_steps = total_steps.as_ref().map(|_| Arc::new(AtomicUsize::new(0)));

        Arc::new(Self {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: self.device.clone(),
            batch_size: self.batch_size,
            block_size: self.block_size,
            steps_per_epoch: steps,
            total_steps,
            consumed_steps,
            summary_event_token_ids: self.summary_event_token_ids.clone(),
            ruliad_policy_batch_schedule: self.ruliad_policy_batch_schedule,
            ruliad_policy_stratified_difficulty_levels: self
                .ruliad_policy_stratified_difficulty_levels,
            prefetch: Arc::new(Mutex::new(None)),
            seed: self.seed,
            source_selection_enabled: self.source_selection_enabled,
        })
    }
}

struct StreamingIterator<B: Backend> {
    dataset: Arc<dyn TokenSequenceDataset>,
    split: DatasetSplit,
    device: B::Device,
    batch_size: usize,
    block_size: usize,
    steps_total: usize,
    step: usize,
    total_steps: Option<usize>,
    consumed_steps: Option<Arc<AtomicUsize>>,
    summary_event_token_ids: Option<Vec<u32>>,
    ruliad_policy_batch_schedule: RuliadPolicyBatchSchedule,
    ruliad_policy_stratified_difficulty_levels: usize,
    steps_per_epoch: usize,
    logical_document_tokens: usize,
    chunks_per_document: usize,
    next_document_group: usize,
    chunk_index_in_document: usize,
    num_documents: usize,
    document_start: usize,
    document_stride: usize,
    epoch_index: usize,
    source_selection_enabled: bool,
}

impl<B: Backend> Iterator for StreamingIterator<B> {
    type Item = SequenceBatch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.step >= self.steps_total {
            return None;
        }
        let step_in_loader = self.step;
        let absolute_step = if let Some(counter) = &self.consumed_steps {
            let previous = counter.fetch_add(1, Ordering::Relaxed);
            if let Some(limit) = self.total_steps
                && previous >= limit
            {
                return None;
            }
            previous
        } else {
            self.epoch_index
                .saturating_mul(self.steps_per_epoch.max(1))
                .saturating_add(step_in_loader)
        };
        self.step += 1;

        let prof_enabled = crate::train::profile::enabled();
        let cpu_start = prof_enabled.then(Instant::now);
        let (offset, _span) = self.dataset.split_offset_and_span(self.split);
        let batch_size = self.batch_size;
        let block_size = self.block_size;
        let mut inputs = vec![0i64; batch_size * block_size];
        let mut targets = vec![0i64; batch_size * block_size];
        let dataset_uses_target_loss_mask = self.dataset.uses_target_loss_mask();
        let chunk_offset = self.chunk_index_in_document.saturating_mul(block_size);
        let partial_document_chunk = self
            .logical_document_tokens
            .saturating_sub(chunk_offset)
            .min(block_size)
            < block_size;
        let mut loss_mask = (dataset_uses_target_loss_mask || partial_document_chunk).then(|| {
            let initial = i64::from(!dataset_uses_target_loss_mask);
            vec![initial; batch_size * block_size]
        });
        let document_span = self.logical_document_tokens + 1;
        let reset_stream_state = self.chunk_index_in_document == 0;

        let mut source_document_complete = None;
        if self.source_selection_enabled
            && let Some(SourceSelectedStreamBatch {
                windows: source_windows,
                loss_masks: source_loss_masks,
                document_complete,
            }) = self
                .dataset
                .source_selected_stream_token_windows_with_loss_masks(
                    self.split,
                    self.epoch_index,
                    absolute_step,
                    self.chunk_index_in_document,
                    batch_size,
                    block_size,
                )
        {
            source_document_complete = Some(document_complete);
            assert_eq!(
                source_windows.len(),
                batch_size,
                "source-selected stream windows must match batch size"
            );
            if let Some(source_loss_masks) = source_loss_masks.as_ref() {
                assert_eq!(
                    source_loss_masks.len(),
                    batch_size,
                    "source-selected stream masks must match batch size"
                );
            }
            for (batch_idx, window) in source_windows.iter().enumerate() {
                assert!(
                    window.len() > block_size,
                    "source-selected stream window must include block_size + 1 tokens"
                );
                let input_row = &mut inputs[batch_idx * block_size..(batch_idx + 1) * block_size];
                let target_row = &mut targets[batch_idx * block_size..(batch_idx + 1) * block_size];
                let mask_row = loss_mask
                    .as_mut()
                    .map(|mask| &mut mask[batch_idx * block_size..(batch_idx + 1) * block_size]);
                if let Some(mask_row) = mask_row {
                    for t in 0..block_size {
                        input_row[t] = window[t] as i64;
                        target_row[t] = window[t + 1] as i64;
                    }
                    if let Some(source_loss_masks) = source_loss_masks.as_ref() {
                        mask_row.copy_from_slice(
                            source_loss_masks
                                .get(batch_idx)
                                .expect("source-selected stream mask missing row"),
                        );
                    } else {
                        self.dataset.target_loss_mask_for_window(window, mask_row);
                    }
                } else {
                    fill_rows_from_window(
                        self.dataset.as_ref(),
                        window,
                        block_size,
                        input_row,
                        target_row,
                        None,
                    );
                }
            }
        } else {
            #[cfg(feature = "train")]
            {
                let next_document_group = self.next_document_group;
                let document_start = self.document_start;
                let document_stride = self.document_stride;
                let num_documents = self.num_documents.max(1);
                let chunk_index_in_document = self.chunk_index_in_document;
                if let Some(mask) = loss_mask.as_mut() {
                    inputs
                        .par_chunks_mut(block_size)
                        .zip(targets.par_chunks_mut(block_size))
                        .zip(mask.par_chunks_mut(block_size))
                        .enumerate()
                        .for_each(|(batch_idx, ((input_row, target_row), mask_row))| {
                            let doc_rank = (next_document_group + batch_idx) % num_documents;
                            let doc_idx = (document_start
                                .wrapping_add(doc_rank.wrapping_mul(document_stride)))
                                % num_documents;
                            let doc_start = offset + doc_idx.saturating_mul(document_span);
                            fill_stream_row_from_document(
                                self.dataset.as_ref(),
                                self.split,
                                self.epoch_index,
                                doc_start,
                                self.logical_document_tokens,
                                chunk_index_in_document,
                                block_size,
                                input_row,
                                target_row,
                                Some(mask_row),
                            );
                        });
                } else {
                    inputs
                        .par_chunks_mut(block_size)
                        .zip(targets.par_chunks_mut(block_size))
                        .enumerate()
                        .for_each(|(batch_idx, (input_row, target_row))| {
                            let doc_rank = (next_document_group + batch_idx) % num_documents;
                            let doc_idx = (document_start
                                .wrapping_add(doc_rank.wrapping_mul(document_stride)))
                                % num_documents;
                            let doc_start = offset + doc_idx.saturating_mul(document_span);
                            fill_stream_row_from_document(
                                self.dataset.as_ref(),
                                self.split,
                                self.epoch_index,
                                doc_start,
                                self.logical_document_tokens,
                                chunk_index_in_document,
                                block_size,
                                input_row,
                                target_row,
                                None,
                            );
                        });
                }
            }
            #[cfg(not(feature = "train"))]
            for batch_idx in 0..batch_size {
                let doc_rank = (self.next_document_group + batch_idx) % self.num_documents.max(1);
                let doc_idx = (self
                    .document_start
                    .wrapping_add(doc_rank.wrapping_mul(self.document_stride)))
                    % self.num_documents.max(1);
                let doc_start = offset + doc_idx.saturating_mul(document_span);
                let input_row = &mut inputs[batch_idx * block_size..(batch_idx + 1) * block_size];
                let target_row = &mut targets[batch_idx * block_size..(batch_idx + 1) * block_size];
                let mask_row = loss_mask
                    .as_mut()
                    .map(|mask| &mut mask[batch_idx * block_size..(batch_idx + 1) * block_size]);
                fill_stream_row_from_document(
                    self.dataset.as_ref(),
                    self.split,
                    self.epoch_index,
                    doc_start,
                    self.logical_document_tokens,
                    self.chunk_index_in_document,
                    block_size,
                    input_row,
                    target_row,
                    mask_row,
                );
            }
        }

        let cpu_ns = cpu_start
            .map(|start| start.elapsed().as_nanos())
            .unwrap_or_default();
        if prof_enabled {
            crate::train::profile::record_dataloader_foreground_wait(cpu_ns);
        }
        let ruliad_policy_batch = if self.source_selection_enabled
            && self.ruliad_policy_batch_schedule.includes(absolute_step)
        {
            let selection_step =
                absolute_step.saturating_sub(self.chunk_index_in_document.min(absolute_step));
            self.dataset
                .source_selected_ruliad_policy_batch(
                    self.split,
                    self.epoch_index,
                    selection_step,
                    batch_size,
                    self.ruliad_policy_stratified_difficulty_levels,
                )
                .map(Arc::new)
        } else {
            None
        };

        let tensor_copy_start = prof_enabled.then(Instant::now);
        let summary_event_mask = summary_event_mask_tensor::<B>(
            &inputs,
            batch_size,
            block_size,
            self.summary_event_token_ids.as_deref(),
            &self.device,
        );
        let inputs_tensor = Tensor::<B, 2, Int>::from_data(
            TensorData::new(inputs, [batch_size, block_size]),
            &self.device,
        );
        let targets_tensor = Tensor::<B, 2, Int>::from_data(
            TensorData::new(targets, [batch_size, block_size]),
            &self.device,
        );
        let loss_mask_tensor = loss_mask.map(|mask| {
            Tensor::<B, 2, Int>::from_data(
                TensorData::new(mask, [batch_size, block_size]),
                &self.device,
            )
        });
        let tensor_copy_ns = tensor_copy_start
            .map(|start| start.elapsed().as_nanos())
            .unwrap_or_default();

        if prof_enabled {
            let values = batch_size.saturating_mul(block_size);
            let tensor_count = 2 + usize::from(loss_mask_tensor.is_some());
            let copy_bytes = (values
                .saturating_mul(tensor_count)
                .saturating_mul(size_of::<i64>())) as u128;
            crate::train::profile::record_dataloader(cpu_ns, tensor_copy_ns, copy_bytes, 0);
        }

        self.chunk_index_in_document += 1;
        if source_document_complete
            .unwrap_or(self.chunk_index_in_document >= self.chunks_per_document)
        {
            self.chunk_index_in_document = 0;
            self.next_document_group =
                (self.next_document_group + batch_size) % self.num_documents.max(1);
        }

        Some(
            SequenceBatch::new(inputs_tensor, targets_tensor, summary_event_mask)
                .with_loss_mask(loss_mask_tensor)
                .with_ruliad_policy_batch(ruliad_policy_batch)
                .with_reset_stream_state(reset_stream_state),
        )
    }
}

impl<B: Backend> DataLoaderIterator<SequenceBatch<B>> for StreamingIterator<B> {
    fn progress(&self) -> Progress {
        Progress::new(self.step, self.steps_total)
    }
}

impl<B> DataLoader<B, SequenceBatch<B>> for StreamingDataLoader<B>
where
    B: Backend + 'static,
    B::Device: Clone,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<SequenceBatch<B>> + 'a> {
        let steps_total =
            if let (Some(limit), Some(consumed)) = (self.total_steps, &self.consumed_steps) {
                let used = consumed.load(Ordering::Relaxed);
                if used >= limit {
                    0
                } else {
                    (limit - used).min(self.steps_per_epoch)
                }
            } else {
                self.steps_per_epoch
            };

        let (offset, span) = self.dataset.split_offset_and_span(self.split);
        let _ = offset;
        let block_size = self.block_size.max(1);
        let logical_document_tokens = self.logical_document_tokens.max(block_size);
        let chunks_per_document = logical_document_tokens.div_ceil(block_size).max(1);
        let document_span = logical_document_tokens + 1;
        let num_documents = (span / document_span).max(1);
        let consumed = self
            .consumed_steps
            .as_ref()
            .map(|counter| counter.load(Ordering::Relaxed))
            .unwrap_or_default();
        let epoch_index = consumed / self.steps_per_epoch.max(1);
        let step_in_epoch = consumed % self.steps_per_epoch.max(1);
        let (document_start, document_stride) =
            resolve_stream_document_permutation(self.seed, epoch_index, num_documents);
        let chunk_index_in_document =
            if self.source_selection_enabled && self.dataset.uses_live_source_selection() {
                0
            } else {
                step_in_epoch % chunks_per_document
            };
        let next_document_group = step_in_epoch
            .checked_div(chunks_per_document)
            .unwrap_or_default()
            .saturating_mul(self.batch_size)
            % num_documents;

        Box::new(StreamingIterator {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: self.device.clone(),
            batch_size: self.batch_size,
            block_size,
            steps_total,
            step: 0,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.clone(),
            summary_event_token_ids: self.summary_event_token_ids.clone(),
            ruliad_policy_batch_schedule: self.ruliad_policy_batch_schedule,
            ruliad_policy_stratified_difficulty_levels: self
                .ruliad_policy_stratified_difficulty_levels,
            steps_per_epoch: self.steps_per_epoch,
            logical_document_tokens,
            chunks_per_document,
            next_document_group,
            chunk_index_in_document,
            num_documents,
            document_start,
            document_stride,
            epoch_index,
            source_selection_enabled: self.source_selection_enabled,
        })
    }

    fn num_items(&self) -> usize {
        self.steps_per_epoch
    }

    fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, SequenceBatch<B>>> {
        Arc::new(Self {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: device.clone(),
            batch_size: self.batch_size,
            block_size: self.block_size,
            steps_per_epoch: self.steps_per_epoch,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
            summary_event_token_ids: self.summary_event_token_ids.clone(),
            ruliad_policy_batch_schedule: self.ruliad_policy_batch_schedule,
            ruliad_policy_stratified_difficulty_levels: self
                .ruliad_policy_stratified_difficulty_levels,
            logical_document_tokens: self.logical_document_tokens,
            seed: self.seed,
            source_selection_enabled: self.source_selection_enabled,
        })
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, SequenceBatch<B>>> {
        let end = end.min(self.steps_per_epoch);
        let start = start.min(end);
        let steps = (end - start).max(1);
        let total_steps = self.total_steps.map(|limit| limit.min(steps));
        let consumed_steps = total_steps.as_ref().map(|_| Arc::new(AtomicUsize::new(0)));

        Arc::new(Self {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: self.device.clone(),
            batch_size: self.batch_size,
            block_size: self.block_size,
            steps_per_epoch: steps,
            total_steps,
            consumed_steps,
            summary_event_token_ids: self.summary_event_token_ids.clone(),
            ruliad_policy_batch_schedule: self.ruliad_policy_batch_schedule,
            ruliad_policy_stratified_difficulty_levels: self
                .ruliad_policy_stratified_difficulty_levels,
            logical_document_tokens: self.logical_document_tokens,
            seed: self.seed,
            source_selection_enabled: self.source_selection_enabled,
        })
    }
}

#[cfg(test)]
mod streaming_tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[derive(Clone)]
    struct TinyDataset {
        tokens: Arc<Vec<u32>>,
        train_len: usize,
        block_size: usize,
        batch_size: usize,
        tokenizer: SharedTokenizer,
        preferred_logical_document_tokens: Option<usize>,
        mask_even_targets: bool,
    }

    impl TokenSequenceDataset for TinyDataset {
        fn tokenizer(&self) -> SharedTokenizer {
            self.tokenizer.clone()
        }

        fn token_count(&self) -> usize {
            self.tokens.len()
        }

        fn copy_token_range(&self, start: usize, dst: &mut [u32]) {
            dst.copy_from_slice(&self.tokens[start..start + dst.len()]);
        }

        fn train_len(&self) -> usize {
            self.train_len
        }

        fn block_size(&self) -> usize {
            self.block_size
        }

        fn batch_size(&self) -> usize {
            self.batch_size
        }

        fn train_split_ratio(&self) -> f32 {
            1.0
        }

        fn preferred_logical_document_tokens(&self, _split: DatasetSplit) -> Option<usize> {
            self.preferred_logical_document_tokens
        }

        fn uses_target_loss_mask(&self) -> bool {
            self.mask_even_targets
        }

        fn target_loss_mask_for_window(&self, window: &[u32], mask: &mut [i64]) -> bool {
            mask.fill(0);
            if !self.mask_even_targets || window.len() < mask.len().saturating_add(1) {
                return false;
            }
            for t in 0..mask.len() {
                mask[t] = i64::from(window[t + 1].is_multiple_of(2));
            }
            mask.contains(&1)
        }
    }

    fn tiny_pretokenized_tokenizer() -> SharedTokenizer {
        use crate::tokenizer::{PretokenizedTokenizerConfig, TokenizerConfig, TokenizerKind};
        TokenizerConfig {
            vocab_path: None,
            kind: TokenizerKind::Pretokenized(PretokenizedTokenizerConfig {
                vocab_size: 256,
                bos_id: None,
                eos_id: Some(255),
                pad_id: None,
                unk_id: None,
            }),
        }
        .fit(std::iter::empty())
        .expect("tokenizer")
    }

    #[test]
    fn streaming_loader_resets_only_on_new_logical_document() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let dataset = Arc::new(TinyDataset {
            tokens: Arc::new((0u32..65).collect()),
            train_len: 65,
            block_size: 4,
            batch_size: 2,
            tokenizer: tiny_pretokenized_tokenizer(),
            preferred_logical_document_tokens: None,
            mask_even_targets: false,
        });
        let loader = StreamingDataLoader::<TestBackend>::new(
            Arc::clone(&dataset),
            DatasetSplit::Train,
            &device,
            4,
            Some(4),
            Some(8),
            1337,
        );
        let mut iter = loader.iter();
        let first = iter.next().expect("first");
        let second = iter.next().expect("second");
        let third = iter.next().expect("third");
        assert!(first.reset_stream_state);
        assert!(!second.reset_stream_state);
        assert!(third.reset_stream_state);
    }

    #[test]
    fn streaming_loader_masks_partial_final_chunk_without_crossing_split_boundary() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let dataset = Arc::new(TinyDataset {
            tokens: Arc::new(vec![
                10, 11, 12, 13, 14, 15, 16, 255, 20, 21, 22, 23, 24, 25, 26, 255,
            ]),
            train_len: 8,
            block_size: 4,
            batch_size: 1,
            tokenizer: tiny_pretokenized_tokenizer(),
            preferred_logical_document_tokens: Some(7),
            mask_even_targets: false,
        });
        let loader = StreamingDataLoader::<TestBackend>::new(
            Arc::clone(&dataset),
            DatasetSplit::Val,
            &device,
            2,
            Some(2),
            None,
            1337,
        );
        let mut iter = loader.iter();
        let first = iter.next().expect("first validation chunk");
        let second = iter.next().expect("partial validation chunk");

        assert!(first.reset_stream_state);
        assert!(!second.reset_stream_state);
        assert!(first.loss_mask.is_none());
        assert_eq!(
            second
                .inputs
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .expect("inputs"),
            vec![24, 25, 26, 255]
        );
        assert_eq!(
            second
                .targets
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .expect("targets"),
            vec![25, 26, 255, 255]
        );
        assert_eq!(
            second
                .loss_mask
                .expect("partial chunk mask")
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .expect("loss mask"),
            vec![1, 1, 1, 0]
        );
    }

    #[test]
    fn streaming_loader_preserves_empty_dataset_mask_on_partial_final_chunk() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let dataset = Arc::new(TinyDataset {
            tokens: Arc::new(vec![
                10, 11, 13, 15, 17, 19, 21, 255, 20, 21, 23, 25, 27, 29, 31, 255,
            ]),
            train_len: 8,
            block_size: 4,
            batch_size: 1,
            tokenizer: tiny_pretokenized_tokenizer(),
            preferred_logical_document_tokens: Some(7),
            mask_even_targets: true,
        });
        let loader = StreamingDataLoader::<TestBackend>::new(
            Arc::clone(&dataset),
            DatasetSplit::Val,
            &device,
            2,
            Some(2),
            None,
            1337,
        );
        let second = loader.iter().nth(1).expect("partial validation chunk");

        assert_eq!(
            second
                .loss_mask
                .expect("dataset-owned partial chunk mask")
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .expect("loss mask"),
            vec![0, 0, 0, 0]
        );
    }

    #[test]
    fn random_sampling_respects_preferred_logical_document_boundaries() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let dataset = Arc::new(TinyDataset {
            tokens: Arc::new(vec![
                100, 101, 102, 103, 104, 105, 106, 107, 255, 200, 201, 202, 203, 204, 205, 206,
                207, 255, 300, 301, 302, 303, 304, 305, 306, 307, 255,
            ]),
            train_len: 27,
            block_size: 4,
            batch_size: 8,
            tokenizer: tiny_pretokenized_tokenizer(),
            preferred_logical_document_tokens: Some(8),
            mask_even_targets: false,
        });

        for _ in 0..32 {
            let batch = sample_batch_with_shape::<TestBackend, _>(
                dataset.as_ref(),
                DatasetSplit::Train,
                dataset.batch_size,
                dataset.block_size,
                None,
                0,
                &device,
            );
            let inputs = batch
                .inputs
                .into_data()
                .to_vec::<i64>()
                .expect("batch inputs");
            let targets = batch
                .targets
                .into_data()
                .to_vec::<i64>()
                .expect("batch targets");
            for row in 0..dataset.batch_size {
                let input_row = &inputs[row * dataset.block_size..(row + 1) * dataset.block_size];
                let target_row = &targets[row * dataset.block_size..(row + 1) * dataset.block_size];
                let bucket = input_row[0] / 100;
                assert!((1..=3).contains(&bucket));
                assert!(input_row.iter().all(|value| *value / 100 == bucket));
                assert!(
                    target_row
                        .iter()
                        .all(|value| { *value == 255 || *value / 100 == bucket })
                );
            }
        }
    }

    #[test]
    fn random_loader_batch_override_controls_emitted_shape() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let dataset = Arc::new(TinyDataset {
            tokens: Arc::new((0u32..128).collect()),
            train_len: 128,
            block_size: 4,
            batch_size: 2,
            tokenizer: tiny_pretokenized_tokenizer(),
            preferred_logical_document_tokens: None,
            mask_even_targets: false,
        });
        let batch = RandomDataLoader::<TestBackend>::new(
            Arc::clone(&dataset),
            DatasetSplit::Train,
            &device,
            1,
            Some(1),
        )
        .with_batch_size(5)
        .iter()
        .next()
        .expect("batch");
        assert_eq!(batch.inputs.shape().dims::<2>(), [5, 4]);
    }

    #[test]
    fn streaming_loader_batch_override_controls_emitted_shape() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let dataset = Arc::new(TinyDataset {
            tokens: Arc::new((0u32..129).collect()),
            train_len: 129,
            block_size: 4,
            batch_size: 2,
            tokenizer: tiny_pretokenized_tokenizer(),
            preferred_logical_document_tokens: None,
            mask_even_targets: false,
        });
        let batch = StreamingDataLoader::<TestBackend>::new(
            Arc::clone(&dataset),
            DatasetSplit::Train,
            &device,
            1,
            Some(1),
            Some(8),
            1337,
        )
        .with_batch_size(6)
        .iter()
        .next()
        .expect("batch");
        assert_eq!(batch.inputs.shape().dims::<2>(), [6, 4]);
    }

    #[test]
    fn random_sampling_uses_full_document_when_block_matches_logical_length() {
        let dataset = TinyDataset {
            tokens: Arc::new(vec![
                10, 11, 12, 13, 14, 15, 16, 17, 255, 20, 21, 22, 23, 24, 25, 26, 27, 255,
            ]),
            train_len: 18,
            block_size: 8,
            batch_size: 4,
            tokenizer: tiny_pretokenized_tokenizer(),
            preferred_logical_document_tokens: Some(8),
            mask_even_targets: false,
        };

        for absolute_step in 0..16 {
            let host = sample_host_batch_with_shape(
                &dataset,
                HostBatchRequest {
                    split: DatasetSplit::Train,
                    batch_size: dataset.batch_size,
                    block_size: dataset.block_size,
                    epoch_index: 0,
                    absolute_step,
                    seed: 1337,
                    source_selection_enabled: true,
                    include_ruliad_policy_batch: false,
                    ruliad_policy_stratified_difficulty_levels: 0,
                },
            );
            for row in 0..dataset.batch_size {
                let input_row =
                    &host.inputs[row * dataset.block_size..(row + 1) * dataset.block_size];
                let target_row =
                    &host.targets[row * dataset.block_size..(row + 1) * dataset.block_size];
                let base = input_row[0];
                assert!(
                    base == 10 || base == 20,
                    "full-document sample should start at document boundary, got {base}"
                );
                assert_eq!(
                    input_row,
                    &[
                        base,
                        base + 1,
                        base + 2,
                        base + 3,
                        base + 4,
                        base + 5,
                        base + 6,
                        base + 7
                    ]
                );
                assert_eq!(
                    target_row,
                    &[
                        base + 1,
                        base + 2,
                        base + 3,
                        base + 4,
                        base + 5,
                        base + 6,
                        base + 7,
                        255
                    ]
                );
            }
        }
    }

    #[test]
    fn streaming_loader_seed_is_stable_but_changes_document_order() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let dataset = Arc::new(TinyDataset {
            tokens: Arc::new((0u32..257).collect()),
            train_len: 257,
            block_size: 4,
            batch_size: 2,
            tokenizer: tiny_pretokenized_tokenizer(),
            preferred_logical_document_tokens: None,
            mask_even_targets: false,
        });
        let batch_inputs = |seed| {
            let loader = StreamingDataLoader::<TestBackend>::new(
                Arc::clone(&dataset),
                DatasetSplit::Train,
                &device,
                4,
                Some(4),
                Some(8),
                seed,
            );
            let batch = loader.iter().next().expect("streaming batch");
            batch
                .inputs
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .expect("batch tokens")
        };

        let first = batch_inputs(1337);
        let repeated = batch_inputs(1337);
        let different = batch_inputs(7331);

        assert_eq!(first, repeated);
        assert_ne!(first, different);
    }

    #[test]
    fn streaming_loader_propagates_target_loss_mask() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let dataset = Arc::new(TinyDataset {
            tokens: Arc::new((0u32..65).collect()),
            train_len: 65,
            block_size: 4,
            batch_size: 2,
            tokenizer: tiny_pretokenized_tokenizer(),
            preferred_logical_document_tokens: None,
            mask_even_targets: true,
        });
        let batch = StreamingDataLoader::<TestBackend>::new(
            Arc::clone(&dataset),
            DatasetSplit::Train,
            &device,
            2,
            Some(2),
            Some(8),
            1337,
        )
        .iter()
        .next()
        .expect("streaming batch");

        let targets = batch
            .targets
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .expect("targets");
        let mask = batch
            .loss_mask
            .expect("streaming loss mask")
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .expect("loss mask");
        let expected = targets
            .iter()
            .map(|target| i64::from(target % 2 == 0))
            .collect::<Vec<_>>();
        assert_eq!(mask, expected);
    }
}

struct RandomIterator<B: Backend> {
    dataset: Arc<dyn TokenSequenceDataset>,
    split: DatasetSplit,
    device: B::Device,
    batch_size: usize,
    block_size: usize,
    steps_total: usize,
    step: usize,
    total_steps: Option<usize>,
    consumed_steps: Option<Arc<AtomicUsize>>,
    summary_event_token_ids: Option<Vec<u32>>,
    ruliad_policy_batch_schedule: RuliadPolicyBatchSchedule,
    ruliad_policy_stratified_difficulty_levels: usize,
    seed: u64,
    source_selection_enabled: bool,
    epoch_index: usize,
    prefetch: Option<Arc<Mutex<Option<RandomPrefetch>>>>,
}

impl<B: Backend> Iterator for RandomIterator<B> {
    type Item = SequenceBatch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.step >= self.steps_total {
            return None;
        }

        let prof_enabled = crate::train::profile::enabled();
        let host = if let Some(prefetch) = self.prefetch.as_ref() {
            let wait_start = prof_enabled.then(Instant::now);
            let mut slot = prefetch.lock().expect("random prefetch lock");
            let host = slot.as_mut()?.recv()?;
            if let Some(start) = wait_start {
                crate::train::profile::record_dataloader_foreground_wait(
                    start.elapsed().as_nanos(),
                );
            }
            host
        } else {
            let absolute_step = self
                .consumed_steps
                .as_ref()
                .map(|counter| counter.load(Ordering::Relaxed))
                .unwrap_or(self.step);
            let host = sample_host_batch_with_shape(
                &*self.dataset,
                HostBatchRequest {
                    split: self.split,
                    batch_size: self.batch_size,
                    block_size: self.block_size,
                    epoch_index: self.epoch_index,
                    absolute_step,
                    seed: self.seed,
                    source_selection_enabled: self.source_selection_enabled,
                    include_ruliad_policy_batch: self
                        .ruliad_policy_batch_schedule
                        .includes(absolute_step),
                    ruliad_policy_stratified_difficulty_levels: self
                        .ruliad_policy_stratified_difficulty_levels,
                },
            );
            if prof_enabled {
                crate::train::profile::record_dataloader_foreground_wait(host.dataloader_cpu_ns);
            }
            host
        };

        if let Some(counter) = &self.consumed_steps {
            if let Some(limit) = self.total_steps {
                let previous = counter.fetch_add(1, Ordering::Relaxed);
                if previous >= limit {
                    return None;
                }
            } else {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }

        self.step += 1;

        Some(finalize_host_batch_on_device::<B>(
            host,
            self.batch_size,
            self.block_size,
            self.summary_event_token_ids.as_deref(),
            &self.device,
        ))
    }
}

impl<B: Backend> DataLoaderIterator<SequenceBatch<B>> for RandomIterator<B> {
    fn progress(&self) -> Progress {
        Progress::new(self.step, self.steps_total)
    }
}

#[cfg(test)]
mod random_loader_tests {
    use super::*;
    use burn_ndarray::NdArray;

    use crate::tokenizer::{PretokenizedTokenizerConfig, TokenizerConfig, TokenizerKind};

    type TestBackend = NdArray<f32>;

    #[derive(Clone)]
    struct EpochAwareDataset {
        block_size: usize,
        batch_size: usize,
        tokenizer: SharedTokenizer,
    }

    #[derive(Clone)]
    struct LivePrefetchDataset {
        block_size: usize,
        batch_size: usize,
        tokenizer: SharedTokenizer,
        selected_steps: Arc<Mutex<Vec<usize>>>,
        policy_steps: Option<Arc<Mutex<Vec<usize>>>>,
    }

    impl TokenSequenceDataset for EpochAwareDataset {
        fn tokenizer(&self) -> SharedTokenizer {
            self.tokenizer.clone()
        }

        fn token_count(&self) -> usize {
            64
        }

        fn copy_token_range(&self, start: usize, dst: &mut [u32]) {
            self.copy_token_range_with_epoch(DatasetSplit::Train, 0, start, dst);
        }

        fn copy_token_range_with_epoch(
            &self,
            _split: DatasetSplit,
            epoch_index: usize,
            _start: usize,
            dst: &mut [u32],
        ) {
            let base = (epoch_index as u32).saturating_mul(100);
            for (idx, value) in dst.iter_mut().enumerate() {
                *value = base.saturating_add(idx as u32);
            }
        }

        fn train_len(&self) -> usize {
            64
        }

        fn block_size(&self) -> usize {
            self.block_size
        }

        fn batch_size(&self) -> usize {
            self.batch_size
        }

        fn train_split_ratio(&self) -> f32 {
            1.0
        }
    }

    impl TokenSequenceDataset for LivePrefetchDataset {
        fn tokenizer(&self) -> SharedTokenizer {
            self.tokenizer.clone()
        }

        fn token_count(&self) -> usize {
            64
        }

        fn copy_token_range(&self, start: usize, dst: &mut [u32]) {
            self.copy_token_range_with_epoch(DatasetSplit::Train, 0, start, dst);
        }

        fn copy_token_range_with_epoch(
            &self,
            _split: DatasetSplit,
            _epoch_index: usize,
            start: usize,
            dst: &mut [u32],
        ) {
            for (idx, value) in dst.iter_mut().enumerate() {
                *value = (start + idx) as u32;
            }
        }

        fn uses_live_source_selection(&self) -> bool {
            true
        }

        fn source_selected_document_indices(
            &self,
            _split: DatasetSplit,
            _epoch_index: usize,
            absolute_step: usize,
            batch_size: usize,
        ) -> Option<Vec<usize>> {
            self.selected_steps
                .lock()
                .expect("selected steps lock")
                .push(absolute_step);
            Some(vec![0; batch_size])
        }

        fn source_selected_stream_token_windows(
            &self,
            _split: DatasetSplit,
            _epoch_index: usize,
            absolute_step: usize,
            _chunk_index_in_document: usize,
            batch_size: usize,
            block_size: usize,
        ) -> Option<Vec<Vec<u32>>> {
            self.selected_steps
                .lock()
                .expect("selected steps lock")
                .push(absolute_step);
            Some(
                (0..batch_size)
                    .map(|row| {
                        (0..=block_size)
                            .map(|column| {
                                300u32
                                    .saturating_add(absolute_step as u32)
                                    .saturating_add((row * (block_size + 1) + column) as u32)
                            })
                            .collect()
                    })
                    .collect(),
            )
        }

        fn source_selected_ruliad_policy_batch(
            &self,
            _split: DatasetSplit,
            _epoch_index: usize,
            absolute_step: usize,
            batch_size: usize,
            _stratified_difficulty_levels: usize,
        ) -> Option<RuliadPolicyBatch> {
            let policy_steps = self.policy_steps.as_ref()?;
            policy_steps
                .lock()
                .expect("policy steps lock")
                .push(absolute_step);
            let samples = (0..batch_size)
                .map(|idx| RuliadPolicySample {
                    item: burn_dragon_universality::RuliadEvalItem {
                        oracle_hash: format!("h{absolute_step}-{idx}"),
                        sample_index: absolute_step.saturating_add(idx),
                        split: burn_dragon_universality::SampleSplit::Train,
                        family: "policy".to_string(),
                        task_kind: "logical_document".to_string(),
                        math_domains: vec!["test".to_string()],
                        reasoning_modes: vec!["test".to_string()],
                        prompt: "?:policy\n!:".to_string(),
                        expected_answer: "ok=1".to_string(),
                        difficulty_level: Some(0),
                        spec: None,
                    },
                    prompt_tokens: vec![1, 2, 3],
                })
                .collect();
            Some(RuliadPolicyBatch {
                samples,
                tokenization:
                    burn_dragon_universality::RuliadTokenizationConfig::Gpt2ByteCompatible {
                        vocab_size: 512,
                        eos_id: Some(511),
                    },
                stop_token_id: Some(511),
            })
        }

        fn train_len(&self) -> usize {
            64
        }

        fn block_size(&self) -> usize {
            self.block_size
        }

        fn batch_size(&self) -> usize {
            self.batch_size
        }

        fn train_split_ratio(&self) -> f32 {
            1.0
        }

        fn preferred_logical_document_tokens(&self, _split: DatasetSplit) -> Option<usize> {
            Some(self.block_size.saturating_mul(2))
        }
    }

    fn tiny_pretokenized_tokenizer() -> SharedTokenizer {
        TokenizerConfig {
            vocab_path: None,
            kind: TokenizerKind::Pretokenized(PretokenizedTokenizerConfig {
                vocab_size: 512,
                bos_id: None,
                eos_id: Some(511),
                pad_id: None,
                unk_id: None,
            }),
        }
        .fit(std::iter::empty())
        .expect("tokenizer")
    }

    #[test]
    fn random_loader_resume_offset_advances_epoch_aware_samples() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let dataset = Arc::new(EpochAwareDataset {
            block_size: 4,
            batch_size: 1,
            tokenizer: tiny_pretokenized_tokenizer(),
        });

        let first_epoch_batch = RandomDataLoader::<TestBackend>::new(
            Arc::clone(&dataset),
            DatasetSplit::Train,
            &device,
            4,
            Some(8),
        )
        .iter()
        .next()
        .expect("first epoch batch")
        .inputs
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .expect("first epoch tokens");

        let resumed_batch = RandomDataLoader::<TestBackend>::new(
            Arc::clone(&dataset),
            DatasetSplit::Train,
            &device,
            4,
            Some(8),
        )
        .with_initial_consumed_steps(4)
        .iter()
        .next()
        .expect("resumed batch")
        .inputs
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .expect("resumed tokens");

        assert_eq!(first_epoch_batch, vec![0, 1, 2, 3]);
        assert_eq!(resumed_batch, vec![100, 101, 102, 103]);
    }

    #[test]
    fn fixed_holdout_sampling_is_seeded_and_bypasses_live_source_selection() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let selected_steps = Arc::new(Mutex::new(Vec::new()));
        let dataset = Arc::new(LivePrefetchDataset {
            block_size: 4,
            batch_size: 4,
            tokenizer: tiny_pretokenized_tokenizer(),
            selected_steps: Arc::clone(&selected_steps),
            policy_steps: None,
        });

        let sample = |seed| {
            RandomDataLoader::<TestBackend>::new(
                Arc::clone(&dataset),
                DatasetSplit::Val,
                &device,
                2,
                None,
            )
            .with_seed(seed)
            .with_source_selection_enabled(false)
            .iter()
            .flat_map(|batch| {
                batch
                    .inputs
                    .to_data()
                    .convert::<i64>()
                    .into_vec::<i64>()
                    .expect("fixed holdout tokens")
            })
            .collect::<Vec<_>>()
        };

        let first = sample(41);
        let repeated = sample(41);
        let different_seed = sample(42);
        assert_eq!(first, repeated);
        assert_ne!(first, different_seed);
        assert!(
            selected_steps
                .lock()
                .expect("selected steps lock")
                .is_empty(),
            "fixed holdout must not consult the live source policy"
        );
    }

    #[test]
    fn live_validation_sampling_remains_explicitly_available() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let selected_steps = Arc::new(Mutex::new(Vec::new()));
        let dataset = Arc::new(LivePrefetchDataset {
            block_size: 4,
            batch_size: 1,
            tokenizer: tiny_pretokenized_tokenizer(),
            selected_steps: Arc::clone(&selected_steps),
            policy_steps: None,
        });

        let _ = RandomDataLoader::<TestBackend>::new(dataset, DatasetSplit::Val, &device, 1, None)
            .with_source_selection_enabled(true)
            .iter()
            .next()
            .expect("live validation batch");

        assert_eq!(
            selected_steps
                .lock()
                .expect("selected steps lock")
                .as_slice(),
            &[0]
        );
    }

    #[test]
    fn fixed_stream_holdout_bypasses_source_and_policy_selection() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let selected_steps = Arc::new(Mutex::new(Vec::new()));
        let policy_steps = Arc::new(Mutex::new(Vec::new()));
        let dataset = Arc::new(LivePrefetchDataset {
            block_size: 4,
            batch_size: 2,
            tokenizer: tiny_pretokenized_tokenizer(),
            selected_steps: Arc::clone(&selected_steps),
            policy_steps: Some(Arc::clone(&policy_steps)),
        });

        let sample = || {
            StreamingDataLoader::<TestBackend>::new(
                Arc::clone(&dataset),
                DatasetSplit::Val,
                &device,
                2,
                None,
                Some(8),
                41,
            )
            .with_source_selection_enabled(false)
            .with_ruliad_policy_batch(true)
            .iter()
            .flat_map(|batch| {
                assert!(batch.ruliad_policy_batch.is_none());
                batch
                    .inputs
                    .to_data()
                    .convert::<i64>()
                    .into_vec::<i64>()
                    .expect("fixed stream tokens")
            })
            .collect::<Vec<_>>()
        };

        assert_eq!(sample(), sample());
        assert!(
            selected_steps
                .lock()
                .expect("selected steps lock")
                .is_empty()
        );
        assert!(policy_steps.lock().expect("policy steps lock").is_empty());
    }

    #[test]
    fn random_loader_prefetches_bounded_live_source_selection_steps() {
        if live_source_selection_prefetch_depth() == 0 {
            return;
        }

        let device = burn::tensor::Device::<TestBackend>::default();
        let selected_steps = Arc::new(Mutex::new(Vec::new()));
        let dataset = Arc::new(LivePrefetchDataset {
            block_size: 4,
            batch_size: 1,
            tokenizer: tiny_pretokenized_tokenizer(),
            selected_steps: Arc::clone(&selected_steps),
            policy_steps: None,
        });

        let loader = RandomDataLoader::<TestBackend>::new(
            Arc::clone(&dataset),
            DatasetSplit::Train,
            &device,
            8,
            Some(8),
        );
        let mut iter = loader.iter();
        let steps_after_prime = selected_steps.lock().expect("selected steps lock").clone();
        assert!(
            steps_after_prime.contains(&0),
            "live prefetch should prepare the current absolute step"
        );
        assert!(
            steps_after_prime.iter().any(|step| *step > 0),
            "live prefetch should prepare at least one bounded future absolute step"
        );

        let _ = iter.next().expect("prefetched live batch");
    }

    #[test]
    fn random_loader_attaches_ruliad_policy_batch_for_logical_document_source_selection() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let selected_steps = Arc::new(Mutex::new(Vec::new()));
        let policy_steps = Arc::new(Mutex::new(Vec::new()));
        let dataset = Arc::new(LivePrefetchDataset {
            block_size: 4,
            batch_size: 1,
            tokenizer: tiny_pretokenized_tokenizer(),
            selected_steps: Arc::clone(&selected_steps),
            policy_steps: Some(Arc::clone(&policy_steps)),
        });

        let batch = RandomDataLoader::<TestBackend>::new(
            Arc::clone(&dataset),
            DatasetSplit::Train,
            &device,
            1,
            Some(1),
        )
        .with_ruliad_policy_batch(true)
        .iter()
        .next()
        .expect("batch");

        let policy_batch = batch
            .ruliad_policy_batch
            .as_ref()
            .expect("policy batch should be attached");
        assert_eq!(policy_batch.samples.len(), 1);
        assert_eq!(policy_batch.samples[0].item.expected_answer, "ok=1");
        assert!(
            policy_steps.lock().expect("policy steps lock").contains(&0),
            "policy batch should be requested for the current absolute step"
        );
    }

    #[test]
    fn loaders_materialize_policy_metadata_only_on_scheduled_training_steps() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let mut supervision = crate::config::RuliadSupervisionConfig::default();
        supervision.proof_policy.enabled = true;
        supervision.proof_policy.weight = 0.25;
        supervision.proof_policy.start_after_steps = 2;
        supervision.proof_policy.every_steps = 2;

        let random_policy_steps = Arc::new(Mutex::new(Vec::new()));
        let random_dataset = Arc::new(LivePrefetchDataset {
            block_size: 4,
            batch_size: 1,
            tokenizer: tiny_pretokenized_tokenizer(),
            selected_steps: Arc::new(Mutex::new(Vec::new())),
            policy_steps: Some(Arc::clone(&random_policy_steps)),
        });
        let random_attached = RandomDataLoader::<TestBackend>::new(
            random_dataset,
            DatasetSplit::Train,
            &device,
            6,
            Some(6),
        )
        .with_ruliad_policy_supervision(supervision)
        .iter()
        .map(|batch| batch.ruliad_policy_batch.is_some())
        .collect::<Vec<_>>();
        assert_eq!(
            random_attached,
            vec![false, false, true, false, true, false]
        );
        let mut random_policy_steps = random_policy_steps
            .lock()
            .expect("random policy steps lock")
            .clone();
        random_policy_steps.sort_unstable();
        assert_eq!(random_policy_steps, vec![2, 4]);

        let streaming_policy_steps = Arc::new(Mutex::new(Vec::new()));
        let streaming_dataset = Arc::new(LivePrefetchDataset {
            block_size: 4,
            batch_size: 1,
            tokenizer: tiny_pretokenized_tokenizer(),
            selected_steps: Arc::new(Mutex::new(Vec::new())),
            policy_steps: Some(Arc::clone(&streaming_policy_steps)),
        });
        let streaming_attached = StreamingDataLoader::<TestBackend>::new(
            streaming_dataset,
            DatasetSplit::Train,
            &device,
            6,
            Some(6),
            Some(4),
            1337,
        )
        .with_ruliad_policy_supervision(supervision)
        .iter()
        .map(|batch| batch.ruliad_policy_batch.is_some())
        .collect::<Vec<_>>();
        assert_eq!(streaming_attached, random_attached);
        assert_eq!(
            streaming_policy_steps
                .lock()
                .expect("streaming policy steps lock")
                .len(),
            2
        );
    }

    #[test]
    fn streaming_loader_attaches_ruliad_policy_batch_for_live_source_selection() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let selected_steps = Arc::new(Mutex::new(Vec::new()));
        let policy_steps = Arc::new(Mutex::new(Vec::new()));
        let dataset = Arc::new(LivePrefetchDataset {
            block_size: 4,
            batch_size: 1,
            tokenizer: tiny_pretokenized_tokenizer(),
            selected_steps,
            policy_steps: Some(Arc::clone(&policy_steps)),
        });

        let batch = StreamingDataLoader::<TestBackend>::new(
            Arc::clone(&dataset),
            DatasetSplit::Train,
            &device,
            1,
            Some(1),
            Some(4),
            1337,
        )
        .with_ruliad_policy_batch(true)
        .iter()
        .next()
        .expect("streaming batch");

        let policy_batch = batch
            .ruliad_policy_batch
            .as_ref()
            .expect("streaming policy batch should be attached");
        assert_eq!(policy_batch.samples.len(), 1);
        assert_eq!(policy_batch.samples[0].item.expected_answer, "ok=1");
        assert!(
            policy_steps.lock().expect("policy steps lock").contains(&0),
            "streaming policy batch should be requested for the current absolute step"
        );
    }

    #[test]
    fn streaming_loader_reuses_document_selection_step_for_tbptt_policy_batch() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let policy_steps = Arc::new(Mutex::new(Vec::new()));
        let dataset = Arc::new(LivePrefetchDataset {
            block_size: 4,
            batch_size: 1,
            tokenizer: tiny_pretokenized_tokenizer(),
            selected_steps: Arc::new(Mutex::new(Vec::new())),
            policy_steps: Some(Arc::clone(&policy_steps)),
        });

        let loader = StreamingDataLoader::<TestBackend>::new(
            Arc::clone(&dataset),
            DatasetSplit::Train,
            &device,
            2,
            Some(2),
            None,
            1337,
        )
        .with_ruliad_policy_batch(true);
        let mut iterator = loader.iter();
        let _ = iterator.next().expect("first stream chunk");
        let _ = iterator.next().expect("second stream chunk");

        assert_eq!(
            *policy_steps.lock().expect("policy steps lock"),
            vec![0, 0],
            "all chunks from one streamed logical document should use the same source-selection policy step"
        );
    }
}
