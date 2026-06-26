mod factory;
mod huggingface;
mod prepared_chunks;
pub mod scheduler;
mod universality;

use crate::tokenizer::SharedTokenizer;
use burn::tensor::backend::Backend;

pub use factory::build_dataset;
pub use huggingface::HuggingFaceDataset;
pub use scheduler::{
    RandomDataLoader, RuliadPolicyBatch, RuliadPolicySample, SequenceBatch, StreamingDataLoader,
    TokenSequenceDataset, sample_batch_with_shape,
};
pub use universality::{
    RuliadSourceSelectionStateSnapshot, RuliadValidationProbeItem, UniversalityDataset,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DatasetSplit {
    Train,
    Val,
}

#[derive(Clone)]
pub enum Dataset {
    HuggingFace(HuggingFaceDataset),
    Universality(UniversalityDataset),
}

impl Dataset {
    pub fn from_huggingface(dataset: HuggingFaceDataset) -> Self {
        Self::HuggingFace(dataset)
    }

    pub fn from_universality(dataset: UniversalityDataset) -> Self {
        Self::Universality(dataset)
    }

    pub fn tokenizer(&self) -> SharedTokenizer {
        TokenSequenceDataset::tokenizer(self)
    }

    pub fn train_split_ratio(&self) -> f32 {
        TokenSequenceDataset::train_split_ratio(self)
    }

    pub fn batch_size(&self) -> usize {
        TokenSequenceDataset::batch_size(self)
    }

    pub fn steps_per_epoch(&self, split: DatasetSplit) -> usize {
        TokenSequenceDataset::steps_per_epoch(self, split)
    }

    pub fn uses_live_source_selection(&self) -> bool {
        TokenSequenceDataset::uses_live_source_selection(self)
    }

    pub fn record_source_selection_loss(
        &self,
        absolute_step: usize,
        loss: f32,
    ) -> Option<burn_dragon_universality::RuliadMetricSnapshot> {
        TokenSequenceDataset::record_source_selection_loss(self, absolute_step, loss)
    }

    pub fn source_selection_snapshot(
        &self,
    ) -> Option<burn_dragon_universality::RuliadMetricSnapshot> {
        TokenSequenceDataset::source_selection_snapshot(self)
    }

    pub fn record_ruliad_capability_feedback(
        &self,
        report: &burn_dragon_universality::RuliadEvalReport,
    ) -> Option<burn_dragon_universality::RuliadMetricSnapshot> {
        match self {
            Dataset::HuggingFace(_) => None,
            Dataset::Universality(dataset) => dataset.record_ruliad_capability_feedback(report),
        }
    }

    pub fn write_source_selection_state(
        &self,
        path: &std::path::Path,
        absolute_step_offset: usize,
    ) -> std::io::Result<Option<RuliadSourceSelectionStateSnapshot>> {
        match self {
            Dataset::HuggingFace(_) => Ok(None),
            Dataset::Universality(dataset) => {
                dataset.write_source_selection_state(path, absolute_step_offset)
            }
        }
    }

    pub fn apply_source_selection_dynamics_control(
        &self,
        difficulty_pressure: f32,
        hash_noise_max_probability: f32,
    ) {
        if let Self::Universality(dataset) = self {
            dataset.apply_source_selection_dynamics_control(
                difficulty_pressure,
                hash_noise_max_probability,
            );
        }
    }

    pub fn sample_source_weighted_validation_batch<B: Backend>(
        &self,
        epoch_index: usize,
        absolute_step: usize,
        batch_size: usize,
        summary_event_token_ids: Option<&[u32]>,
        device: &B::Device,
    ) -> Option<SequenceBatch<B>> {
        match self {
            Dataset::HuggingFace(_) => None,
            Dataset::Universality(dataset) => dataset.sample_source_weighted_validation_batch(
                epoch_index,
                absolute_step,
                batch_size,
                summary_event_token_ids,
                device,
            ),
        }
    }

    pub fn sample_ruliad_validation_probe_items(
        &self,
        epoch_index: usize,
        absolute_step: usize,
        max_items: usize,
    ) -> Vec<RuliadValidationProbeItem> {
        match self {
            Dataset::HuggingFace(_) => Vec::new(),
            Dataset::Universality(dataset) => {
                dataset.sample_ruliad_validation_probe_items(epoch_index, absolute_step, max_items)
            }
        }
    }

    pub fn decode_ruliad_payload_tokens(
        &self,
        tokens: &[i64],
        stop_at_eos: bool,
    ) -> Option<String> {
        match self {
            Dataset::HuggingFace(_) => None,
            Dataset::Universality(dataset) => {
                dataset.decode_ruliad_payload_tokens(tokens, stop_at_eos)
            }
        }
    }

    pub fn ruliad_document_end_token_id(&self) -> Option<u32> {
        match self {
            Dataset::HuggingFace(_) => None,
            Dataset::Universality(dataset) => dataset.ruliad_document_end_token_id(),
        }
    }
}

impl TokenSequenceDataset for Dataset {
    fn tokenizer(&self) -> SharedTokenizer {
        match self {
            Dataset::HuggingFace(dataset) => dataset.tokenizer(),
            Dataset::Universality(dataset) => dataset.tokenizer(),
        }
    }

    fn token_count(&self) -> usize {
        match self {
            Dataset::HuggingFace(dataset) => dataset.token_count(),
            Dataset::Universality(dataset) => dataset.token_count(),
        }
    }

    fn copy_token_range(&self, start: usize, dst: &mut [u32]) {
        match self {
            Dataset::HuggingFace(dataset) => dataset.copy_token_range(start, dst),
            Dataset::Universality(dataset) => dataset.copy_token_range(start, dst),
        }
    }

    fn train_len(&self) -> usize {
        match self {
            Dataset::HuggingFace(dataset) => dataset.train_len(),
            Dataset::Universality(dataset) => dataset.train_len(),
        }
    }

    fn block_size(&self) -> usize {
        match self {
            Dataset::HuggingFace(dataset) => dataset.block_size(),
            Dataset::Universality(dataset) => dataset.block_size(),
        }
    }

    fn batch_size(&self) -> usize {
        match self {
            Dataset::HuggingFace(dataset) => dataset.batch_size(),
            Dataset::Universality(dataset) => dataset.batch_size(),
        }
    }

    fn train_split_ratio(&self) -> f32 {
        match self {
            Dataset::HuggingFace(dataset) => dataset.train_split_ratio(),
            Dataset::Universality(dataset) => dataset.train_split_ratio(),
        }
    }

    fn preferred_logical_document_tokens(&self, split: DatasetSplit) -> Option<usize> {
        match self {
            Dataset::HuggingFace(dataset) => dataset.preferred_logical_document_tokens(split),
            Dataset::Universality(dataset) => dataset.preferred_logical_document_tokens(split),
        }
    }

    fn uses_live_source_selection(&self) -> bool {
        match self {
            Dataset::HuggingFace(dataset) => dataset.uses_live_source_selection(),
            Dataset::Universality(dataset) => dataset.uses_live_source_selection(),
        }
    }

    fn source_selected_document_indices(
        &self,
        split: DatasetSplit,
        epoch_index: usize,
        absolute_step: usize,
        batch_size: usize,
    ) -> Option<Vec<usize>> {
        match self {
            Dataset::HuggingFace(dataset) => dataset.source_selected_document_indices(
                split,
                epoch_index,
                absolute_step,
                batch_size,
            ),
            Dataset::Universality(dataset) => dataset.source_selected_document_indices(
                split,
                epoch_index,
                absolute_step,
                batch_size,
            ),
        }
    }

    fn source_selected_token_windows(
        &self,
        split: DatasetSplit,
        epoch_index: usize,
        absolute_step: usize,
        batch_size: usize,
        block_size: usize,
    ) -> Option<Vec<Vec<u32>>> {
        match self {
            Dataset::HuggingFace(dataset) => dataset.source_selected_token_windows(
                split,
                epoch_index,
                absolute_step,
                batch_size,
                block_size,
            ),
            Dataset::Universality(dataset) => dataset.source_selected_token_windows(
                split,
                epoch_index,
                absolute_step,
                batch_size,
                block_size,
            ),
        }
    }

    fn source_selected_ruliad_policy_batch(
        &self,
        split: DatasetSplit,
        epoch_index: usize,
        absolute_step: usize,
        batch_size: usize,
    ) -> Option<RuliadPolicyBatch> {
        match self {
            Dataset::HuggingFace(dataset) => dataset.source_selected_ruliad_policy_batch(
                split,
                epoch_index,
                absolute_step,
                batch_size,
            ),
            Dataset::Universality(dataset) => dataset.source_selected_ruliad_policy_batch(
                split,
                epoch_index,
                absolute_step,
                batch_size,
            ),
        }
    }

    fn source_selected_stream_token_windows(
        &self,
        split: DatasetSplit,
        epoch_index: usize,
        absolute_step: usize,
        chunk_index_in_document: usize,
        batch_size: usize,
        block_size: usize,
    ) -> Option<Vec<Vec<u32>>> {
        match self {
            Dataset::HuggingFace(dataset) => dataset.source_selected_stream_token_windows(
                split,
                epoch_index,
                absolute_step,
                chunk_index_in_document,
                batch_size,
                block_size,
            ),
            Dataset::Universality(dataset) => dataset.source_selected_stream_token_windows(
                split,
                epoch_index,
                absolute_step,
                chunk_index_in_document,
                batch_size,
                block_size,
            ),
        }
    }

    fn source_selected_stream_token_windows_with_loss_masks(
        &self,
        split: DatasetSplit,
        epoch_index: usize,
        absolute_step: usize,
        chunk_index_in_document: usize,
        batch_size: usize,
        block_size: usize,
    ) -> Option<(Vec<Vec<u32>>, Option<Vec<Vec<i64>>>)> {
        match self {
            Dataset::HuggingFace(dataset) => dataset
                .source_selected_stream_token_windows_with_loss_masks(
                    split,
                    epoch_index,
                    absolute_step,
                    chunk_index_in_document,
                    batch_size,
                    block_size,
                ),
            Dataset::Universality(dataset) => dataset
                .source_selected_stream_token_windows_with_loss_masks(
                    split,
                    epoch_index,
                    absolute_step,
                    chunk_index_in_document,
                    batch_size,
                    block_size,
                ),
        }
    }

    fn record_source_selection_loss(
        &self,
        absolute_step: usize,
        loss: f32,
    ) -> Option<burn_dragon_universality::RuliadMetricSnapshot> {
        match self {
            Dataset::HuggingFace(dataset) => {
                dataset.record_source_selection_loss(absolute_step, loss)
            }
            Dataset::Universality(dataset) => {
                dataset.record_source_selection_loss(absolute_step, loss)
            }
        }
    }

    fn source_selection_snapshot(&self) -> Option<burn_dragon_universality::RuliadMetricSnapshot> {
        match self {
            Dataset::HuggingFace(dataset) => dataset.source_selection_snapshot(),
            Dataset::Universality(dataset) => dataset.source_selection_snapshot(),
        }
    }

    fn uses_target_loss_mask(&self) -> bool {
        match self {
            Dataset::HuggingFace(dataset) => TokenSequenceDataset::uses_target_loss_mask(dataset),
            Dataset::Universality(dataset) => TokenSequenceDataset::uses_target_loss_mask(dataset),
        }
    }

    fn target_loss_mask_for_window(&self, window: &[u32], mask: &mut [i64]) -> bool {
        match self {
            Dataset::HuggingFace(dataset) => {
                TokenSequenceDataset::target_loss_mask_for_window(dataset, window, mask)
            }
            Dataset::Universality(dataset) => {
                TokenSequenceDataset::target_loss_mask_for_window(dataset, window, mask)
            }
        }
    }

    fn split_offset_and_span(&self, split: DatasetSplit) -> (usize, usize) {
        match self {
            Dataset::HuggingFace(dataset) => {
                TokenSequenceDataset::split_offset_and_span(dataset, split)
            }
            Dataset::Universality(dataset) => {
                TokenSequenceDataset::split_offset_and_span(dataset, split)
            }
        }
    }

    fn steps_per_epoch(&self, split: DatasetSplit) -> usize {
        match self {
            Dataset::HuggingFace(dataset) => TokenSequenceDataset::steps_per_epoch(dataset, split),
            Dataset::Universality(dataset) => TokenSequenceDataset::steps_per_epoch(dataset, split),
        }
    }

    fn decode(&self, tokens: &[i64]) -> String {
        match self {
            Dataset::HuggingFace(dataset) => TokenSequenceDataset::decode(dataset, tokens),
            Dataset::Universality(dataset) => TokenSequenceDataset::decode(dataset, tokens),
        }
    }
}
