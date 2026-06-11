mod factory;
mod huggingface;
mod prepared_chunks;
pub mod scheduler;
mod universality;

use crate::tokenizer::SharedTokenizer;

pub use factory::build_dataset;
pub use huggingface::HuggingFaceDataset;
pub use scheduler::{
    RandomDataLoader, SequenceBatch, StreamingDataLoader, TokenSequenceDataset,
    sample_batch_with_shape,
};
pub use universality::UniversalityDataset;

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
