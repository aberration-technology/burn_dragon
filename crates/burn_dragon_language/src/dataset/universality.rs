use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io;
use std::mem::size_of;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::sync_channel;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_dragon_time::Instant;
use memmap2::Mmap;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::DatasetSplit;
use super::prepared_chunks::{ChunkRuntimeCache, load_cached_chunk_from_mutex, mmap_as_u32_slice};
use super::scheduler::{
    RuliadPolicyBatch, RuliadPolicySample, SequenceBatch, SourceSelectedBatch,
    SourceSelectedStreamBatch, TokenSequenceDataset,
};
use crate::config::{RuliadSupervisionConfig, RuliadSupervisionMode};
use crate::summary_events::summary_event_mask_tensor;
use crate::tokenizer::{SharedTokenizer, TokenizerConfig, TokenizerKind};

const DEFAULT_RUNTIME_CHUNK_CACHE_LIMIT: usize = 8;
const DEFAULT_RUNTIME_DOCUMENT_CACHE_LIMIT: usize = 64;
const DEFAULT_RUNTIME_GENERATION_WORKER_LIMIT: usize = 32;
const DEFAULT_LIVE_SOURCE_SELECTION_DOCUMENTS_PER_STEP: usize = 4;
const DEFAULT_LIVE_SOURCE_BATCH_CACHE_LIMIT: usize = 32;
const DEFAULT_LIVE_SOURCE_BATCH_CACHE_BYTES: usize = 512 * 1024 * 1024;
const DEFAULT_SOURCE_SELECTED_EOS_WINDOW_PROBABILITY: f64 = 0.05;
const SOURCE_WEIGHTED_VALIDATION_SPLIT_TAG: u8 = 2;
const RULIAD_VALIDATION_PROBE_PANEL_EPOCH: usize = 0;
const RULIAD_SYMBOLIC_DATA_TOKEN: u32 = 261;
const RULIAD_SYMBOLIC_QUERY_TOKEN: u32 = 262;
const RULIAD_SYMBOLIC_PROOF_STEP_TOKEN: u32 = 263;
const RULIAD_SYMBOLIC_ANSWER_TOKEN: u32 = 264;
const RULIAD_SYMBOLIC_DOCUMENT_END_TOKEN: u32 = 265;
const RULIAD_SOURCE_SELECTION_STATE_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug)]
struct RuliadWindowRequest {
    split: burn_dragon_universality::SampleSplit,
    epoch_index: usize,
    absolute_step: usize,
    batch_size: usize,
    block_size: usize,
    prefer_answer_window: bool,
}

#[derive(Clone, Copy, Debug)]
struct RuliadStreamWindowRequest {
    window: RuliadWindowRequest,
    chunk_index_in_document: usize,
}

#[derive(Clone, Copy, Debug)]
struct RuliadSupervisedStreamRequest {
    stream: RuliadStreamWindowRequest,
    supervision: RuliadSupervisionConfig,
    emit_loss_masks: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct RuliadValidationProbeItem {
    pub item: burn_dragon_universality::RuliadEvalItem,
    pub prompt_tokens: Vec<i64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RuliadValidationPromptMode {
    #[default]
    CanonicalTransfer,
    TrainingSerialization,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct RuliadSourceSelectionStateSnapshot {
    pub version: u32,
    pub absolute_step_offset: usize,
    pub frontier_extension_count: usize,
    #[serde(default)]
    pub released_max_difficulty_level: usize,
    pub control: RuliadSourceSelectionControlSnapshot,
    pub sampler: burn_dragon_universality::RuliadFrontierSamplerState,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
pub struct RuliadSourceSelectionControlSnapshot {
    pub difficulty_pressure: f32,
    pub hash_noise_max_probability: f32,
}

#[derive(Clone)]
enum UniversalityStorage {
    Manifest(ManifestStorage),
    OnTheFly(OnTheFlyStorage),
}

#[derive(Clone)]
struct ManifestStorage {
    tokens: Arc<ChunkedTokens>,
    manifest_path: PathBuf,
    preferred_logical_document_tokens: Option<usize>,
}

#[derive(Clone)]
struct OnTheFlyStorage {
    corpus: Arc<dyn OnlineUniversalityCorpus>,
    config_path: PathBuf,
    source_kind_label: &'static str,
    cache_limit: usize,
    cache: Arc<EpochRuntimeCacheState>,
    live_batch_cache: Arc<LiveDocumentBatchCacheState>,
    source_selection: Option<Arc<LiveSourceSelectionState>>,
    train_probe_summary: burn_dragon_universality::RuntimeCorpusSummary,
    validation_probe_summary: burn_dragon_universality::RuntimeCorpusSummary,
}

#[derive(Clone)]
struct ChunkedTokens {
    chunks: Arc<Vec<ChunkedTokenFile>>,
    cache_limit: usize,
    cache: Arc<Mutex<ChunkRuntimeCache>>,
}

#[derive(Clone)]
struct ChunkedTokenFile {
    path: PathBuf,
    token_offset: usize,
    token_count: usize,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct RuntimeEpochKey {
    split_tag: u8,
    epoch_index: usize,
}

#[derive(Default)]
struct EpochRuntimeCache {
    tick: u64,
    total_cached_documents: usize,
    entries: HashMap<RuntimeEpochKey, CachedEpochDocuments>,
    building: HashSet<RuntimeEpochKey>,
}

struct CachedEpochDocuments {
    documents: Arc<GeneratedEpochDocuments>,
    last_used_tick: u64,
}

struct GeneratedEpochDocuments {
    documents: Vec<Arc<Vec<u32>>>,
    documents_by_bucket: HashMap<String, Vec<usize>>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct LiveDocumentBatchKey {
    split_tag: u8,
    epoch_index: usize,
    selection_step: usize,
    bucket_label: String,
    document_count: usize,
}

#[derive(Default)]
struct LiveDocumentBatchCache {
    tick: u64,
    total_bytes: usize,
    entries: HashMap<LiveDocumentBatchKey, CachedLiveDocumentBatch>,
    building: HashSet<LiveDocumentBatchKey>,
}

struct CachedLiveDocumentBatch {
    documents: Vec<Arc<Vec<u32>>>,
    bytes: usize,
    last_used_tick: u64,
}

struct LiveDocumentBatchCacheState {
    inner: Mutex<LiveDocumentBatchCache>,
    ready: Condvar,
    entry_limit: usize,
    byte_limit: usize,
}

impl LiveDocumentBatchCacheState {
    fn new() -> Self {
        Self {
            inner: Mutex::new(LiveDocumentBatchCache::default()),
            ready: Condvar::new(),
            entry_limit: live_source_batch_cache_limit(),
            byte_limit: live_source_batch_cache_bytes(),
        }
    }
}

impl GeneratedEpochDocuments {
    fn len(&self) -> usize {
        self.documents.len()
    }
}

struct LiveSourceSelectionState {
    sampler: Mutex<burn_dragon_universality::RuliadFrontierSampler>,
    fixed_validation_bucket_labels: Vec<String>,
    corpus_config: burn_dragon_universality::RuliadCorpusConfig,
    frontier_extension: burn_dragon_universality::RuliadFrontierExtensionConfig,
    cold_start: burn_dragon_universality::RuliadSourceSelectionColdStartConfig,
    feedback_updates_enabled: AtomicBool,
    frontier_extension_count: AtomicUsize,
    released_max_difficulty_level: AtomicUsize,
    absolute_step_offset: AtomicUsize,
    pending: Mutex<HashMap<usize, String>>,
    pending_limit: usize,
    control: Mutex<LiveSourceSelectionControl>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RuliadSourceSelectionOverrides {
    pub cold_start_enabled: Option<bool>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
struct LiveSourceSelectionControl {
    difficulty_pressure: f32,
    hash_noise_max_probability: f32,
}

impl Default for LiveSourceSelectionControl {
    fn default() -> Self {
        Self {
            difficulty_pressure: 1.0,
            hash_noise_max_probability: 1.0,
        }
    }
}

impl From<LiveSourceSelectionControl> for RuliadSourceSelectionControlSnapshot {
    fn from(value: LiveSourceSelectionControl) -> Self {
        Self {
            difficulty_pressure: value.difficulty_pressure,
            hash_noise_max_probability: value.hash_noise_max_probability,
        }
    }
}

impl From<RuliadSourceSelectionControlSnapshot> for LiveSourceSelectionControl {
    fn from(value: RuliadSourceSelectionControlSnapshot) -> Self {
        Self {
            difficulty_pressure: value.difficulty_pressure,
            hash_noise_max_probability: value.hash_noise_max_probability,
        }
    }
}

trait OnlineUniversalityCorpus: Send + Sync {
    fn train_samples(&self) -> usize;
    fn validation_samples(&self) -> usize;
    fn document_token_count(&self) -> usize;
    fn eos_id(&self) -> Option<u32> {
        None
    }
    fn generate_document_tokens_for_epoch(
        &self,
        split: burn_dragon_universality::SampleSplit,
        epoch_index: usize,
        sample_index: usize,
    ) -> anyhow::Result<Vec<u32>>;

    fn source_selection_seed(&self) -> u64 {
        0
    }

    fn source_buckets(&self) -> Vec<burn_dragon_universality::RuliadSourceBucket> {
        Vec::new()
    }

    fn ruliad_config(&self) -> Option<&burn_dragon_universality::RuliadCorpusConfig> {
        None
    }

    fn generate_document_tokens_for_source_bucket(
        &self,
        split: burn_dragon_universality::SampleSplit,
        epoch_index: usize,
        sample_index: usize,
        _bucket_label: &str,
    ) -> anyhow::Result<Vec<u32>> {
        self.generate_document_tokens_for_epoch(split, epoch_index, sample_index)
    }

    fn generate_compact_document_tokens_for_source_bucket(
        &self,
        split: burn_dragon_universality::SampleSplit,
        epoch_index: usize,
        sample_index: usize,
        bucket_label: &str,
    ) -> anyhow::Result<Vec<u32>> {
        self.generate_document_tokens_for_source_bucket(
            split,
            epoch_index,
            sample_index,
            bucket_label,
        )
    }

    fn generate_ruliad_eval_item_for_epoch(
        &self,
        _split: burn_dragon_universality::SampleSplit,
        _epoch_index: usize,
        _sample_index: usize,
    ) -> anyhow::Result<Option<burn_dragon_universality::RuliadEvalItem>> {
        Ok(None)
    }

    fn generate_ruliad_eval_item_for_source_bucket(
        &self,
        _split: burn_dragon_universality::SampleSplit,
        _epoch_index: usize,
        _sample_index: usize,
        _bucket_label: &str,
    ) -> anyhow::Result<Option<burn_dragon_universality::RuliadEvalItem>> {
        Ok(None)
    }

    fn generate_ruliad_training_serialization_eval_item_for_epoch(
        &self,
        _split: burn_dragon_universality::SampleSplit,
        _epoch_index: usize,
        _sample_index: usize,
    ) -> anyhow::Result<Option<burn_dragon_universality::RuliadEvalItem>> {
        Ok(None)
    }

    fn generate_ruliad_training_serialization_eval_item_for_source_bucket(
        &self,
        _split: burn_dragon_universality::SampleSplit,
        _epoch_index: usize,
        _sample_index: usize,
        _bucket_label: &str,
    ) -> anyhow::Result<Option<burn_dragon_universality::RuliadEvalItem>> {
        Ok(None)
    }

    fn encode_ruliad_payload_tokens(&self, _text: &str) -> Option<Vec<u32>> {
        None
    }

    fn decode_ruliad_payload_tokens(&self, _tokens: &[u32], _stop_at_eos: bool) -> Option<String> {
        None
    }

    fn ruliad_document_end_token_id(&self) -> Option<u32> {
        // R2 and R3 terminators intentionally share one structural token.
        let tokens = self.encode_ruliad_payload_tokens(
            burn_dragon_universality::ruliad::RULIAD_V2_DOCUMENT_CLOSE_MARKER,
        )?;
        match tokens.as_slice() {
            [token] => Some(*token),
            _ => None,
        }
    }
}

#[derive(Default)]
struct EpochRuntimeCacheState {
    inner: Mutex<EpochRuntimeCache>,
    ready: Condvar,
}

#[derive(Clone)]
pub struct UniversalityDataset {
    storage: UniversalityStorage,
    train_len: usize,
    token_count: usize,
    block_size: usize,
    batch_size: usize,
    train_split_ratio: f32,
    tokenizer: SharedTokenizer,
    dataset_name: String,
    ruliad_supervision: RuliadSupervisionConfig,
}

impl OnlineUniversalityCorpus for burn_dragon_universality::OnlineNcaCorpus {
    fn train_samples(&self) -> usize {
        self.train_samples()
    }

    fn validation_samples(&self) -> usize {
        self.validation_samples()
    }

    fn document_token_count(&self) -> usize {
        self.document_token_count()
    }

    fn eos_id(&self) -> Option<u32> {
        self.tokenizer_manifest().eos_id
    }

    fn generate_document_tokens_for_epoch(
        &self,
        split: burn_dragon_universality::SampleSplit,
        epoch_index: usize,
        sample_index: usize,
    ) -> anyhow::Result<Vec<u32>> {
        self.generate_document_tokens_for_epoch(split, epoch_index, sample_index)
    }
}

impl OnlineUniversalityCorpus for burn_dragon_universality::OnlineRuliadCorpus {
    fn train_samples(&self) -> usize {
        self.train_samples()
    }

    fn validation_samples(&self) -> usize {
        self.validation_samples()
    }

    fn document_token_count(&self) -> usize {
        self.document_token_count()
    }

    fn eos_id(&self) -> Option<u32> {
        self.tokenizer_manifest().eos_id
    }

    fn generate_document_tokens_for_epoch(
        &self,
        split: burn_dragon_universality::SampleSplit,
        epoch_index: usize,
        sample_index: usize,
    ) -> anyhow::Result<Vec<u32>> {
        self.generate_document_tokens_for_epoch(split, epoch_index, sample_index)
    }

    fn source_selection_seed(&self) -> u64 {
        self.config().seed
    }

    fn source_buckets(&self) -> Vec<burn_dragon_universality::RuliadSourceBucket> {
        burn_dragon_universality::OnlineRuliadCorpus::source_buckets(self).to_vec()
    }

    fn ruliad_config(&self) -> Option<&burn_dragon_universality::RuliadCorpusConfig> {
        Some(self.config())
    }

    fn generate_document_tokens_for_source_bucket(
        &self,
        split: burn_dragon_universality::SampleSplit,
        epoch_index: usize,
        sample_index: usize,
        bucket_label: &str,
    ) -> anyhow::Result<Vec<u32>> {
        Ok(self
            .generate_document_for_source_bucket(split, epoch_index, sample_index, bucket_label)?
            .tokens)
    }

    fn generate_compact_document_tokens_for_source_bucket(
        &self,
        split: burn_dragon_universality::SampleSplit,
        epoch_index: usize,
        sample_index: usize,
        bucket_label: &str,
    ) -> anyhow::Result<Vec<u32>> {
        self.generate_compact_document_tokens_for_source_bucket(
            split,
            epoch_index,
            sample_index,
            bucket_label,
        )
    }

    fn generate_ruliad_eval_item_for_epoch(
        &self,
        split: burn_dragon_universality::SampleSplit,
        epoch_index: usize,
        sample_index: usize,
    ) -> anyhow::Result<Option<burn_dragon_universality::RuliadEvalItem>> {
        self.generate_eval_item_for_epoch(split, epoch_index, sample_index)
            .map(Some)
    }

    fn generate_ruliad_eval_item_for_source_bucket(
        &self,
        split: burn_dragon_universality::SampleSplit,
        epoch_index: usize,
        sample_index: usize,
        bucket_label: &str,
    ) -> anyhow::Result<Option<burn_dragon_universality::RuliadEvalItem>> {
        self.generate_eval_item_for_source_bucket(split, epoch_index, sample_index, bucket_label)
            .map(Some)
    }

    fn generate_ruliad_training_serialization_eval_item_for_epoch(
        &self,
        split: burn_dragon_universality::SampleSplit,
        epoch_index: usize,
        sample_index: usize,
    ) -> anyhow::Result<Option<burn_dragon_universality::RuliadEvalItem>> {
        self.generate_training_serialization_eval_item_for_epoch(split, epoch_index, sample_index)
            .map(Some)
    }

    fn generate_ruliad_training_serialization_eval_item_for_source_bucket(
        &self,
        split: burn_dragon_universality::SampleSplit,
        epoch_index: usize,
        sample_index: usize,
        bucket_label: &str,
    ) -> anyhow::Result<Option<burn_dragon_universality::RuliadEvalItem>> {
        self.generate_training_serialization_eval_item_for_source_bucket(
            split,
            epoch_index,
            sample_index,
            bucket_label,
        )
        .map(Some)
    }

    fn encode_ruliad_payload_tokens(&self, text: &str) -> Option<Vec<u32>> {
        Some(self.encode_payload_tokens(text))
    }

    fn decode_ruliad_payload_tokens(&self, tokens: &[u32], stop_at_eos: bool) -> Option<String> {
        Some(self.decode_payload_tokens(tokens, stop_at_eos))
    }
}

mod dataset;
mod manifest;
mod sampling;
mod source_selection;
mod storage;

use manifest::*;
use sampling::*;

#[cfg(test)]
use source_selection::*;

pub(crate) use source_selection::ruliad_capability_feedback_from_report;

#[cfg(test)]
mod tests;
