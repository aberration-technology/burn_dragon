use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::sync_channel;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use memmap2::Mmap;
use rand::prelude::*;

use super::DatasetSplit;
use super::prepared_chunks::{ChunkRuntimeCache, load_cached_chunk_from_mutex, mmap_as_u32_slice};
use super::scheduler::{SequenceBatch, TokenSequenceDataset};
use crate::summary_events::summary_event_mask_tensor;
use crate::tokenizer::{SharedTokenizer, TokenizerConfig, TokenizerKind};

const DEFAULT_RUNTIME_CHUNK_CACHE_LIMIT: usize = 8;
const DEFAULT_RUNTIME_DOCUMENT_CACHE_LIMIT: usize = 64;
const DEFAULT_RUNTIME_GENERATION_WORKER_LIMIT: usize = 32;
const SOURCE_WEIGHTED_VALIDATION_SPLIT_TAG: u8 = 2;

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

impl GeneratedEpochDocuments {
    fn len(&self) -> usize {
        self.documents.len()
    }
}

struct LiveSourceSelectionState {
    sampler: Mutex<burn_dragon_universality::RuliadFrontierSampler>,
    bucket_labels: Vec<String>,
    pending: Mutex<HashMap<usize, String>>,
    pending_limit: usize,
}

trait OnlineUniversalityCorpus: Send + Sync {
    fn train_samples(&self) -> usize;
    fn validation_samples(&self) -> usize;
    fn document_token_count(&self) -> usize;
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

    fn generate_document_tokens_for_source_bucket(
        &self,
        split: burn_dragon_universality::SampleSplit,
        epoch_index: usize,
        sample_index: usize,
        _bucket_label: &str,
    ) -> anyhow::Result<Vec<u32>> {
        self.generate_document_tokens_for_epoch(split, epoch_index, sample_index)
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
}

impl LiveSourceSelectionState {
    fn new(
        config: burn_dragon_universality::RuliadSamplerConfig,
        candidates: Vec<burn_dragon_universality::RuliadSamplerCandidate>,
    ) -> Option<Self> {
        if candidates.is_empty() {
            return None;
        }
        let bucket_labels = candidates
            .iter()
            .map(|candidate| candidate.oracle_hash.clone())
            .collect::<Vec<_>>();
        Some(Self {
            sampler: Mutex::new(burn_dragon_universality::RuliadFrontierSampler::new(
                config, candidates,
            )),
            bucket_labels,
            pending: Mutex::new(HashMap::new()),
            pending_limit: live_source_selection_pending_limit(),
        })
    }

    fn probabilities(&self) -> Vec<f32> {
        self.sampler
            .lock()
            .expect("ruliad source sampler lock poisoned")
            .probabilities()
    }

    fn choose_bucket_for_step(
        &self,
        available: &HashMap<String, Vec<usize>>,
        epoch_index: usize,
        absolute_step: usize,
    ) -> Option<String> {
        self.choose_bucket_for_step_inner(available, epoch_index, absolute_step, true)
    }

    fn choose_bucket_for_validation_step(
        &self,
        available: &HashMap<String, Vec<usize>>,
        epoch_index: usize,
        absolute_step: usize,
    ) -> Option<String> {
        self.choose_bucket_for_step_inner(available, epoch_index, absolute_step, false)
    }

    fn choose_bucket_for_step_inner(
        &self,
        available: &HashMap<String, Vec<usize>>,
        epoch_index: usize,
        absolute_step: usize,
        record_pending: bool,
    ) -> Option<String> {
        let probs = self.probabilities();
        let mut filtered = Vec::new();
        for (index, label) in self.bucket_labels.iter().enumerate() {
            if available
                .get(label)
                .is_some_and(|documents| !documents.is_empty())
            {
                filtered.push((
                    label.clone(),
                    probs
                        .get(index)
                        .copied()
                        .filter(|value| value.is_finite() && *value > 0.0)
                        .unwrap_or(1e-9),
                ));
            }
        }
        if filtered.is_empty() {
            return None;
        }
        let total = filtered.iter().map(|(_, weight)| *weight).sum::<f32>();
        let mut rng = StdRng::seed_from_u64(source_selection_step_seed(
            epoch_index,
            absolute_step,
            filtered.len(),
        ));
        let ticket = rng.r#gen::<f32>() * total.max(1e-12);
        let mut cumulative = 0.0;
        for (label, weight) in filtered {
            cumulative += weight;
            if ticket <= cumulative {
                if record_pending {
                    self.record_pending(absolute_step, &label);
                }
                return Some(label);
            }
        }
        None
    }

    fn record_pending(&self, absolute_step: usize, bucket_label: &str) {
        let mut pending = self
            .pending
            .lock()
            .expect("ruliad source pending lock poisoned");
        pending.insert(absolute_step, bucket_label.to_string());
        if pending.len() > self.pending_limit {
            let remove_count = pending.len().saturating_sub(self.pending_limit);
            let mut keys = pending.keys().copied().collect::<Vec<_>>();
            keys.sort_unstable();
            for key in keys.into_iter().take(remove_count) {
                pending.remove(&key);
            }
        }
    }

    fn record_loss(
        &self,
        absolute_step: usize,
        loss: f32,
    ) -> Option<burn_dragon_universality::RuliadMetricSnapshot> {
        let bucket_label = self
            .pending
            .lock()
            .expect("ruliad source pending lock poisoned")
            .remove(&absolute_step)?;
        let mut sampler = self
            .sampler
            .lock()
            .expect("ruliad source sampler lock poisoned");
        sampler.record_telemetry(&burn_dragon_universality::RuliadSampleTelemetry {
            oracle_hash: bucket_label,
            family: String::new(),
            task_kind: String::new(),
            loss,
            previous_loss: None,
            gradient_alignment: None,
            verification_cost: 1.0,
            accepted: true,
        });
        Some(sampler.snapshot())
    }

    fn snapshot(&self) -> burn_dragon_universality::RuliadMetricSnapshot {
        self.sampler
            .lock()
            .expect("ruliad source sampler lock poisoned")
            .snapshot()
    }
}

impl UniversalityDataset {
    pub fn new(
        manifest_path: impl AsRef<Path>,
        block_size: usize,
        batch_size: usize,
        train_split_ratio: f32,
        tokenizer_cfg: &TokenizerConfig,
    ) -> io::Result<Self> {
        let tokenizer = validate_pretokenized_tokenizer(tokenizer_cfg)?;
        let manifest_path = manifest_path.as_ref().to_path_buf();
        let manifest =
            burn_dragon_universality::load_manifest(&manifest_path).map_err(io::Error::other)?;
        validate_tokenizer_against_manifest(tokenizer.as_ref(), &manifest.tokenizer)?;
        let preferred_logical_document_tokens =
            fixed_manifest_logical_document_tokens(&manifest).map_err(io::Error::other)?;
        if matches!(
            manifest.corpus_kind,
            burn_dragon_universality::CorpusKind::Nca
        ) && matches!(
            preferred_logical_document_tokens,
            Some(logical_document_tokens) if block_size > logical_document_tokens
        ) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "training.block_size={} exceeds prepared NCA logical document length {}; regenerate the manifest with longer single-rule rollouts",
                    block_size,
                    preferred_logical_document_tokens.unwrap_or_default()
                ),
            ));
        }

        let manifest_dir = manifest_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let chunk_root = manifest_dir.join(&manifest.chunk_dir);
        let mut chunks = Vec::with_capacity(manifest.chunks.len());
        for chunk in &manifest.chunks {
            let path = chunk_root.join(&chunk.file_name);
            if !path.is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("universality chunk missing: {}", path.display()),
                ));
            }
            let byte_len = fs::metadata(&path)?.len() as usize;
            let expected_bytes = chunk.token_count.saturating_mul(4);
            if byte_len != expected_bytes {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "universality chunk {} size mismatch (expected={} actual={})",
                        path.display(),
                        expected_bytes,
                        byte_len
                    ),
                ));
            }
            chunks.push(ChunkedTokenFile {
                path,
                token_offset: chunk.token_offset,
                token_count: chunk.token_count,
            });
        }

        Ok(Self {
            storage: UniversalityStorage::Manifest(ManifestStorage {
                tokens: Arc::new(ChunkedTokens {
                    chunks: Arc::new(chunks),
                    cache_limit: runtime_chunk_cache_limit(),
                    cache: Arc::new(Mutex::new(ChunkRuntimeCache::default())),
                }),
                manifest_path,
                preferred_logical_document_tokens,
            }),
            train_len: manifest.train_token_count,
            token_count: manifest.token_count,
            block_size,
            batch_size,
            train_split_ratio,
            tokenizer,
            dataset_name: manifest.dataset_name,
        })
    }

    pub fn new_on_the_fly(
        config_path: impl AsRef<Path>,
        block_size: usize,
        batch_size: usize,
        min_logical_document_tokens: Option<usize>,
        tokenizer_cfg: &TokenizerConfig,
    ) -> io::Result<Self> {
        let tokenizer = validate_pretokenized_tokenizer(tokenizer_cfg)?;
        let config_path = config_path.as_ref().to_path_buf();
        let target_logical_document_tokens = min_logical_document_tokens
            .unwrap_or(block_size)
            .max(block_size);
        let corpus =
            burn_dragon_universality::OnlineNcaCorpus::load_with_min_logical_document_tokens(
                &config_path,
                Some(target_logical_document_tokens),
            )
            .map_err(io::Error::other)?;
        validate_tokenizer_against_manifest(tokenizer.as_ref(), corpus.tokenizer_manifest())?;
        let document_token_count = corpus.document_token_count();
        let logical_document_tokens = document_token_count.saturating_sub(1);
        if logical_document_tokens == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "on-the-fly NCA corpus must yield at least one input token per document",
            ));
        }
        if block_size > logical_document_tokens {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "training.block_size={} exceeds adapted on-the-fly NCA logical document length {}",
                    block_size, logical_document_tokens
                ),
            ));
        }

        let train_probe_summary = corpus
            .default_probe_summary(burn_dragon_universality::SampleSplit::Train)
            .map_err(io::Error::other)?;
        let validation_probe_summary = corpus
            .default_probe_summary(burn_dragon_universality::SampleSplit::Validation)
            .map_err(io::Error::other)?;

        let train_len = corpus.train_token_count();
        let token_count = corpus.total_token_count();
        let train_split_ratio = if token_count == 0 {
            1.0
        } else {
            train_len as f32 / token_count as f32
        };
        let dataset_name = config_file_display_name(
            config_path
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("nca"),
        )
        .to_string();

        Ok(Self {
            storage: UniversalityStorage::OnTheFly(OnTheFlyStorage {
                corpus: Arc::new(corpus),
                config_path,
                source_kind_label: "on-the-fly universality NCA",
                cache_limit: runtime_document_cache_limit(
                    batch_size,
                    train_probe_summary.sample_count,
                    validation_probe_summary.sample_count,
                ),
                cache: Arc::new(EpochRuntimeCacheState::default()),
                source_selection: None,
                train_probe_summary,
                validation_probe_summary,
            }),
            train_len,
            token_count,
            block_size,
            batch_size,
            train_split_ratio,
            tokenizer,
            dataset_name,
        })
    }

    pub fn new_ruliad_on_the_fly(
        config_path: impl AsRef<Path>,
        block_size: usize,
        batch_size: usize,
        tokenizer_cfg: &TokenizerConfig,
    ) -> io::Result<Self> {
        let tokenizer = validate_pretokenized_tokenizer(tokenizer_cfg)?;
        let config_path = config_path.as_ref().to_path_buf();
        let corpus = burn_dragon_universality::OnlineRuliadCorpus::load(&config_path)
            .map_err(io::Error::other)?;
        validate_tokenizer_against_manifest(tokenizer.as_ref(), corpus.tokenizer_manifest())?;
        let document_token_count = corpus.document_token_count();
        let logical_document_tokens = document_token_count.saturating_sub(1);
        if logical_document_tokens == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "on-the-fly ruliad corpus must yield at least one input token per document",
            ));
        }
        if block_size > logical_document_tokens {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "training.block_size={} exceeds on-the-fly ruliad logical document length {}",
                    block_size, logical_document_tokens
                ),
            ));
        }

        let train_probe_summary = corpus
            .default_probe_summary(burn_dragon_universality::SampleSplit::Train)
            .map_err(io::Error::other)?;
        let validation_probe_summary = corpus
            .default_probe_summary(burn_dragon_universality::SampleSplit::Validation)
            .map_err(io::Error::other)?;
        let train_len = corpus.train_token_count();
        let token_count = corpus.total_token_count();
        let train_split_ratio = if token_count == 0 {
            1.0
        } else {
            train_len as f32 / token_count as f32
        };
        let dataset_name = config_file_display_name(
            config_path
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("ruliad"),
        )
        .to_string();
        let source_selection = corpus
            .source_selection_enabled()
            .then(|| {
                LiveSourceSelectionState::new(
                    corpus.config().source_selection.sampler,
                    corpus.sampler_candidates(),
                )
            })
            .flatten()
            .map(Arc::new);

        Ok(Self {
            storage: UniversalityStorage::OnTheFly(OnTheFlyStorage {
                corpus: Arc::new(corpus),
                config_path,
                source_kind_label: "on-the-fly universality ruliad",
                cache_limit: runtime_document_cache_limit(
                    batch_size,
                    train_probe_summary.sample_count,
                    validation_probe_summary.sample_count,
                ),
                cache: Arc::new(EpochRuntimeCacheState::default()),
                source_selection,
                train_probe_summary,
                validation_probe_summary,
            }),
            train_len,
            token_count,
            block_size,
            batch_size,
            train_split_ratio,
            tokenizer,
            dataset_name,
        })
    }

    pub fn dataset_name(&self) -> &str {
        &self.dataset_name
    }

    pub fn source_path(&self) -> &Path {
        match &self.storage {
            UniversalityStorage::Manifest(storage) => &storage.manifest_path,
            UniversalityStorage::OnTheFly(storage) => &storage.config_path,
        }
    }

    pub fn source_kind_label(&self) -> &'static str {
        match &self.storage {
            UniversalityStorage::Manifest(_) => "universality manifest",
            UniversalityStorage::OnTheFly(storage) => storage.source_kind_label,
        }
    }

    pub fn tokenizer(&self) -> SharedTokenizer {
        self.tokenizer.clone()
    }

    pub fn train_split_ratio(&self) -> f32 {
        self.train_split_ratio
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn copy_token_range(&self, start: usize, dst: &mut [u32]) {
        match &self.storage {
            UniversalityStorage::Manifest(storage) => storage.tokens.copy_into(start, dst),
            UniversalityStorage::OnTheFly(storage) => storage.copy_into(start, self.train_len, dst),
        }
    }

    pub fn train_len(&self) -> usize {
        self.train_len
    }

    pub fn steps_per_epoch(&self, split: DatasetSplit) -> usize {
        TokenSequenceDataset::steps_per_epoch(self, split)
    }

    pub fn sample_batch<B: Backend>(
        &self,
        split: DatasetSplit,
        device: &B::Device,
    ) -> SequenceBatch<B> {
        super::scheduler::sample_batch(self, split, device)
    }

    pub fn sample_source_weighted_validation_batch<B: Backend>(
        &self,
        epoch_index: usize,
        absolute_step: usize,
        batch_size: usize,
        summary_event_token_ids: Option<&[u32]>,
        device: &B::Device,
    ) -> Option<SequenceBatch<B>> {
        let storage = match &self.storage {
            UniversalityStorage::Manifest(_) => return None,
            UniversalityStorage::OnTheFly(storage) => storage,
        };
        let documents = storage.source_weighted_validation_documents(
            epoch_index,
            absolute_step,
            batch_size.max(1),
        )?;
        let document_token_count = documents.first()?.len();
        let logical_document_tokens = document_token_count.checked_sub(1)?;
        if self.block_size > logical_document_tokens {
            return None;
        }

        let max_start_in_document = logical_document_tokens
            .saturating_sub(self.block_size)
            .min(document_token_count.saturating_sub(self.block_size + 1));
        let batch_size = batch_size.max(1);
        let mut inputs = vec![0i64; batch_size * self.block_size];
        let mut targets = vec![0i64; batch_size * self.block_size];
        for (batch_idx, document) in documents.iter().enumerate() {
            if document.len() <= self.block_size {
                return None;
            }
            let mut rng = StdRng::seed_from_u64(source_selection_step_seed(
                epoch_index,
                absolute_step,
                SOURCE_WEIGHTED_VALIDATION_SPLIT_TAG as usize ^ batch_idx,
            ));
            let start = if max_start_in_document == 0 {
                0
            } else {
                rng.gen_range(0..=max_start_in_document)
            };
            for token_index in 0..self.block_size {
                let offset = batch_idx * self.block_size + token_index;
                inputs[offset] = document[start + token_index] as i64;
                targets[offset] = document[start + token_index + 1] as i64;
            }
        }

        let summary_event_mask = summary_event_mask_tensor::<B>(
            &inputs,
            batch_size,
            self.block_size,
            summary_event_token_ids,
            device,
        );
        let inputs_tensor = Tensor::<B, 2, Int>::from_data(
            TensorData::new(inputs, [batch_size, self.block_size]),
            device,
        );
        let targets_tensor = Tensor::<B, 2, Int>::from_data(
            TensorData::new(targets, [batch_size, self.block_size]),
            device,
        );
        Some(SequenceBatch::new(
            inputs_tensor,
            targets_tensor,
            summary_event_mask,
        ))
    }

    pub fn train_probe_summary(&self) -> Option<&burn_dragon_universality::RuntimeCorpusSummary> {
        match &self.storage {
            UniversalityStorage::Manifest(_) => None,
            UniversalityStorage::OnTheFly(storage) => Some(&storage.train_probe_summary),
        }
    }

    pub fn validation_probe_summary(
        &self,
    ) -> Option<&burn_dragon_universality::RuntimeCorpusSummary> {
        match &self.storage {
            UniversalityStorage::Manifest(_) => None,
            UniversalityStorage::OnTheFly(storage) => Some(&storage.validation_probe_summary),
        }
    }

    pub fn runtime_document_cache_limit(&self) -> Option<usize> {
        match &self.storage {
            UniversalityStorage::Manifest(_) => None,
            UniversalityStorage::OnTheFly(storage) => Some(storage.cache_limit),
        }
    }

    pub fn uses_live_source_selection(&self) -> bool {
        match &self.storage {
            UniversalityStorage::Manifest(_) => false,
            UniversalityStorage::OnTheFly(storage) => storage.source_selection.is_some(),
        }
    }

    pub fn record_source_selection_loss(
        &self,
        absolute_step: usize,
        loss: f32,
    ) -> Option<burn_dragon_universality::RuliadMetricSnapshot> {
        match &self.storage {
            UniversalityStorage::Manifest(_) => None,
            UniversalityStorage::OnTheFly(storage) => storage
                .source_selection
                .as_ref()
                .and_then(|source_selection| source_selection.record_loss(absolute_step, loss)),
        }
    }

    pub fn source_selection_snapshot(
        &self,
    ) -> Option<burn_dragon_universality::RuliadMetricSnapshot> {
        match &self.storage {
            UniversalityStorage::Manifest(_) => None,
            UniversalityStorage::OnTheFly(storage) => storage
                .source_selection
                .as_ref()
                .map(|source_selection| source_selection.snapshot()),
        }
    }
}

impl TokenSequenceDataset for UniversalityDataset {
    fn tokenizer(&self) -> SharedTokenizer {
        self.tokenizer.clone()
    }

    fn token_count(&self) -> usize {
        self.token_count
    }

    fn copy_token_range(&self, start: usize, dst: &mut [u32]) {
        self.copy_token_range(start, dst);
    }

    fn copy_token_range_with_epoch(
        &self,
        split: DatasetSplit,
        epoch_index: usize,
        start: usize,
        dst: &mut [u32],
    ) {
        match &self.storage {
            UniversalityStorage::Manifest(storage) => storage.tokens.copy_into(start, dst),
            UniversalityStorage::OnTheFly(storage) => {
                storage.copy_into_with_epoch(split, epoch_index, start, self.train_len, dst)
            }
        }
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
        self.train_split_ratio
    }

    fn prepare_epoch(&self, split: DatasetSplit, epoch_index: usize) {
        if let (DatasetSplit::Train, UniversalityStorage::OnTheFly(storage)) =
            (split, &self.storage)
        {
            storage.prepare_epoch(burn_dragon_universality::SampleSplit::Train, epoch_index);
        }
    }

    fn prefetch_epoch(&self, split: DatasetSplit, epoch_index: usize) {
        if let (DatasetSplit::Train, UniversalityStorage::OnTheFly(storage)) =
            (split, &self.storage)
        {
            storage.prefetch_epoch(burn_dragon_universality::SampleSplit::Train, epoch_index);
        }
    }

    fn uses_live_source_selection(&self) -> bool {
        self.uses_live_source_selection()
    }

    fn source_selected_document_indices(
        &self,
        split: DatasetSplit,
        epoch_index: usize,
        absolute_step: usize,
        batch_size: usize,
    ) -> Option<Vec<usize>> {
        match (split, &self.storage) {
            (DatasetSplit::Train, UniversalityStorage::OnTheFly(storage)) => storage
                .source_selected_document_indices(
                    burn_dragon_universality::SampleSplit::Train,
                    epoch_index,
                    absolute_step,
                    batch_size,
                ),
            _ => None,
        }
    }

    fn record_source_selection_loss(
        &self,
        absolute_step: usize,
        loss: f32,
    ) -> Option<burn_dragon_universality::RuliadMetricSnapshot> {
        self.record_source_selection_loss(absolute_step, loss)
    }

    fn source_selection_snapshot(&self) -> Option<burn_dragon_universality::RuliadMetricSnapshot> {
        self.source_selection_snapshot()
    }

    fn preferred_logical_document_tokens(&self, _split: DatasetSplit) -> Option<usize> {
        match &self.storage {
            UniversalityStorage::Manifest(storage) => storage.preferred_logical_document_tokens,
            UniversalityStorage::OnTheFly(storage) => {
                Some(storage.corpus.document_token_count().saturating_sub(1))
            }
        }
    }
}

impl OnTheFlyStorage {
    fn copy_into(&self, start: usize, train_len: usize, dst: &mut [u32]) {
        self.copy_into_with_epoch(DatasetSplit::Train, 0, start, train_len, dst);
    }

    fn copy_into_with_epoch(
        &self,
        requested_split: DatasetSplit,
        epoch_index: usize,
        start: usize,
        train_len: usize,
        dst: &mut [u32],
    ) {
        let mut remaining = dst.len();
        let mut written = 0usize;
        let mut cursor = start;
        let document_token_count = self.corpus.document_token_count();

        while remaining > 0 {
            let (split, split_offset, split_sample_count) = if cursor < train_len {
                (
                    burn_dragon_universality::SampleSplit::Train,
                    cursor,
                    self.corpus.train_samples(),
                )
            } else {
                (
                    burn_dragon_universality::SampleSplit::Validation,
                    cursor.saturating_sub(train_len),
                    self.corpus.validation_samples(),
                )
            };

            let sample_index = split_offset / document_token_count;
            if sample_index >= split_sample_count {
                panic!(
                    "on-the-fly universality token request out of range: split={split:?} sample_index={sample_index} sample_count={split_sample_count} start={start} len={}",
                    dst.len()
                );
            }
            let token_index = split_offset % document_token_count;
            let copy_len = document_token_count
                .saturating_sub(token_index)
                .min(remaining);
            let effective_epoch_index = match split {
                burn_dragon_universality::SampleSplit::Train
                    if matches!(requested_split, DatasetSplit::Train) =>
                {
                    epoch_index
                }
                _ => 0,
            };
            let document_tokens = self.document_tokens(split, sample_index, effective_epoch_index);
            dst[written..written + copy_len]
                .copy_from_slice(&document_tokens[token_index..token_index + copy_len]);
            written += copy_len;
            remaining -= copy_len;
            cursor += copy_len;
        }
    }

    fn document_tokens(
        &self,
        split: burn_dragon_universality::SampleSplit,
        sample_index: usize,
        epoch_index: usize,
    ) -> Arc<Vec<u32>> {
        let epoch = self.epoch_documents(split, epoch_index);
        Arc::clone(
            epoch.documents.get(sample_index).unwrap_or_else(|| {
                panic!(
                    "on-the-fly universality epoch cache out of range: split={split:?} epoch_index={epoch_index} sample_index={sample_index} sample_count={}",
                    epoch.len()
                )
            }),
        )
    }

    fn source_selected_document_indices(
        &self,
        split: burn_dragon_universality::SampleSplit,
        epoch_index: usize,
        absolute_step: usize,
        batch_size: usize,
    ) -> Option<Vec<usize>> {
        if split != burn_dragon_universality::SampleSplit::Train {
            return None;
        }
        let source_selection = self.source_selection.as_ref()?;
        let epoch = self.epoch_documents(split, epoch_index);
        if epoch.documents_by_bucket.is_empty() {
            return None;
        }
        let bucket_label = source_selection.choose_bucket_for_step(
            &epoch.documents_by_bucket,
            epoch_index,
            absolute_step,
        )?;
        let documents = epoch.documents_by_bucket.get(&bucket_label)?;
        if documents.is_empty() {
            return None;
        }
        let mut rng = StdRng::seed_from_u64(source_selection_step_seed(
            epoch_index,
            absolute_step,
            source_label_seed(&bucket_label) as usize,
        ));
        Some(
            (0..batch_size)
                .map(|_| documents[rng.gen_range(0..documents.len())])
                .collect(),
        )
    }

    fn source_weighted_validation_documents(
        &self,
        epoch_index: usize,
        absolute_step: usize,
        batch_size: usize,
    ) -> Option<Vec<Arc<Vec<u32>>>> {
        let source_selection = self.source_selection.as_ref()?;
        let epoch = self.source_weighted_validation_epoch_documents(epoch_index);
        if epoch.documents_by_bucket.is_empty() {
            return None;
        }
        let bucket_label = source_selection.choose_bucket_for_validation_step(
            &epoch.documents_by_bucket,
            epoch_index,
            absolute_step,
        )?;
        let documents = epoch.documents_by_bucket.get(&bucket_label)?;
        if documents.is_empty() {
            return None;
        }
        let mut rng = StdRng::seed_from_u64(source_selection_step_seed(
            epoch_index,
            absolute_step,
            source_label_seed(&bucket_label) as usize
                ^ SOURCE_WEIGHTED_VALIDATION_SPLIT_TAG as usize,
        ));
        Some(
            (0..batch_size)
                .filter_map(|_| {
                    let sample_index = documents[rng.gen_range(0..documents.len())];
                    epoch.documents.get(sample_index).map(Arc::clone)
                })
                .collect(),
        )
        .filter(|selected: &Vec<Arc<Vec<u32>>>| selected.len() == batch_size)
    }

    fn prepare_epoch(&self, split: burn_dragon_universality::SampleSplit, epoch_index: usize) {
        let _ = self.epoch_documents(split, epoch_index);
    }

    fn prefetch_epoch(&self, split: burn_dragon_universality::SampleSplit, epoch_index: usize) {
        let key = RuntimeEpochKey {
            split_tag: split_tag(split),
            epoch_index,
        };
        let should_spawn = {
            let mut cache = self
                .cache
                .inner
                .lock()
                .expect("universality runtime cache poisoned");
            if cache.entries.contains_key(&key) || cache.building.contains(&key) {
                false
            } else {
                cache.building.insert(key);
                true
            }
        };
        if !should_spawn {
            return;
        }
        let storage = self.clone();
        if let Err(error) = thread::Builder::new()
            .name(format!("universality-epoch-prefetch-{epoch_index}"))
            .spawn(move || {
                let _ = storage.build_and_store_epoch(key, split, epoch_index, false);
            })
        {
            self.clear_building_epoch(key);
            panic!("failed to spawn NCA epoch prefetch thread: {error}");
        }
    }

    fn epoch_documents(
        &self,
        split: burn_dragon_universality::SampleSplit,
        epoch_index: usize,
    ) -> Arc<GeneratedEpochDocuments> {
        let key = RuntimeEpochKey {
            split_tag: split_tag(split),
            epoch_index,
        };
        loop {
            let mut cache = self
                .cache
                .inner
                .lock()
                .expect("universality runtime cache poisoned");
            cache.tick = cache.tick.wrapping_add(1);
            let tick = cache.tick;
            if let Some(entry) = cache.entries.get_mut(&key) {
                entry.last_used_tick = tick;
                return Arc::clone(&entry.documents);
            }
            if cache.building.insert(key) {
                drop(cache);
                return self.build_and_store_epoch(key, split, epoch_index, false);
            }
            let _unused = self
                .cache
                .ready
                .wait(cache)
                .expect("universality runtime cache poisoned");
        }
    }

    fn source_weighted_validation_epoch_documents(
        &self,
        epoch_index: usize,
    ) -> Arc<GeneratedEpochDocuments> {
        let key = RuntimeEpochKey {
            split_tag: SOURCE_WEIGHTED_VALIDATION_SPLIT_TAG,
            epoch_index,
        };
        loop {
            let mut cache = self
                .cache
                .inner
                .lock()
                .expect("universality runtime cache poisoned");
            cache.tick = cache.tick.wrapping_add(1);
            let tick = cache.tick;
            if let Some(entry) = cache.entries.get_mut(&key) {
                entry.last_used_tick = tick;
                return Arc::clone(&entry.documents);
            }
            if cache.building.insert(key) {
                drop(cache);
                return self.build_and_store_epoch(
                    key,
                    burn_dragon_universality::SampleSplit::Validation,
                    epoch_index,
                    true,
                );
            }
            let _unused = self
                .cache
                .ready
                .wait(cache)
                .expect("universality runtime cache poisoned");
        }
    }

    fn build_and_store_epoch(
        &self,
        key: RuntimeEpochKey,
        split: burn_dragon_universality::SampleSplit,
        epoch_index: usize,
        source_weighted: bool,
    ) -> Arc<GeneratedEpochDocuments> {
        let result = catch_unwind(AssertUnwindSafe(|| {
            Arc::new(self.generate_epoch_documents(split, epoch_index, source_weighted))
        }));
        match result {
            Ok(generated_documents) => {
                self.store_generated_epoch(key, Arc::clone(&generated_documents));
                generated_documents
            }
            Err(panic_payload) => {
                self.clear_building_epoch(key);
                resume_unwind(panic_payload);
            }
        }
    }

    fn store_generated_epoch(
        &self,
        key: RuntimeEpochKey,
        generated_documents: Arc<GeneratedEpochDocuments>,
    ) {
        let mut cache = self
            .cache
            .inner
            .lock()
            .expect("universality runtime cache poisoned");
        cache.tick = cache.tick.wrapping_add(1);
        let tick = cache.tick;
        cache.building.remove(&key);
        cache.entries.insert(
            key,
            CachedEpochDocuments {
                documents: Arc::clone(&generated_documents),
                last_used_tick: tick,
            },
        );
        cache.total_cached_documents = cache
            .entries
            .values()
            .map(|entry| entry.documents.len())
            .sum();
        while cache.total_cached_documents > self.cache_limit {
            let evict_key = cache
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used_tick)
                .map(|(key, _)| *key)
                .expect("universality runtime cache should not be empty");
            if let Some(removed) = cache.entries.remove(&evict_key) {
                cache.total_cached_documents = cache
                    .total_cached_documents
                    .saturating_sub(removed.documents.len());
            }
        }
        self.cache.ready.notify_all();
    }

    fn clear_building_epoch(&self, key: RuntimeEpochKey) {
        let mut cache = self
            .cache
            .inner
            .lock()
            .expect("universality runtime cache poisoned");
        cache.building.remove(&key);
        self.cache.ready.notify_all();
    }

    fn generate_epoch_documents(
        &self,
        split: burn_dragon_universality::SampleSplit,
        epoch_index: usize,
        source_weighted: bool,
    ) -> GeneratedEpochDocuments {
        let sample_count = match split {
            burn_dragon_universality::SampleSplit::Train => self.corpus.train_samples(),
            burn_dragon_universality::SampleSplit::Validation => self.corpus.validation_samples(),
        };
        if sample_count == 0 {
            return GeneratedEpochDocuments {
                documents: Vec::new(),
                documents_by_bucket: HashMap::new(),
            };
        }

        let source_plan =
            if split == burn_dragon_universality::SampleSplit::Train || source_weighted {
                self.source_selection.as_ref().and_then(|source_selection| {
                    let buckets = self.corpus.source_buckets();
                    (!buckets.is_empty()).then(|| {
                        burn_dragon_universality::plan_epoch_source_buckets(
                            &buckets,
                            &source_selection.probabilities(),
                            sample_count,
                            self.corpus.source_selection_seed(),
                            u64::from(if source_weighted {
                                SOURCE_WEIGHTED_VALIDATION_SPLIT_TAG
                            } else {
                                split_tag(split)
                            }),
                            epoch_index,
                        )
                    })
                })
            } else {
                None
            };
        let source_bucket_plan = source_plan.as_ref().map(|plan| plan.bucket_ids.clone());

        let worker_count = runtime_generation_worker_count(sample_count);
        let (sender, receiver) = sync_channel::<(usize, Arc<Vec<u32>>, Option<String>)>(
            worker_count.saturating_mul(2).max(1),
        );
        let next_index = Arc::new(AtomicUsize::new(0));
        let source_bucket_plan = Arc::new(source_bucket_plan);
        let mut workers = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let sender = sender.clone();
            let next_index = Arc::clone(&next_index);
            let corpus = Arc::clone(&self.corpus);
            let source_bucket_plan = Arc::clone(&source_bucket_plan);
            workers.push(thread::spawn(move || {
                loop {
                    let sample_index = next_index.fetch_add(1, Ordering::Relaxed);
                    if sample_index >= sample_count {
                        break;
                    }
                    let bucket_label = source_bucket_plan
                        .as_ref()
                        .as_ref()
                        .and_then(|plan| plan.get(sample_index).cloned());
                    let tokens = Arc::new(
                        match bucket_label.as_deref() {
                            Some(bucket_label) => corpus
                                .generate_document_tokens_for_source_bucket(
                                    split,
                                    epoch_index,
                                    sample_index,
                                    bucket_label,
                                )
                                .unwrap_or_else(|error| {
                                    panic!(
                                        "failed to generate source-selected universality sample split={split:?} epoch_index={epoch_index} sample_index={sample_index} bucket={bucket_label}: {error:#}"
                                    )
                                }),
                            None => corpus
                                .generate_document_tokens_for_epoch(
                                    split,
                                    epoch_index,
                                    sample_index,
                                )
                                .unwrap_or_else(|error| {
                                    panic!(
                                        "failed to generate on-the-fly universality sample split={split:?} epoch_index={epoch_index} sample_index={sample_index}: {error:#}"
                                    )
                                }),
                        },
                    );
                    if sender.send((sample_index, tokens, bucket_label)).is_err() {
                        return;
                    }
                }
            }));
        }
        drop(sender);

        let mut documents = vec![None; sample_count];
        let mut documents_by_bucket = HashMap::<String, Vec<usize>>::new();
        for _ in 0..sample_count {
            let (sample_index, tokens, bucket_label) = receiver
                .recv()
                .expect("on-the-fly universality epoch generation channel closed early");
            documents[sample_index] = Some(tokens);
            if let Some(bucket_label) = bucket_label {
                documents_by_bucket
                    .entry(bucket_label)
                    .or_default()
                    .push(sample_index);
            }
        }
        for worker in workers {
            let _ = worker.join();
        }
        GeneratedEpochDocuments {
            documents: documents
                .into_iter()
                .map(|entry| {
                    entry.expect("on-the-fly universality epoch generation missing sample")
                })
                .collect(),
            documents_by_bucket,
        }
    }
}

fn split_tag(split: burn_dragon_universality::SampleSplit) -> u8 {
    match split {
        burn_dragon_universality::SampleSplit::Train => 0,
        burn_dragon_universality::SampleSplit::Validation => 1,
    }
}

fn live_source_selection_pending_limit() -> usize {
    std::env::var("DragonModel_RULIAD_SOURCE_SELECTION_PENDING_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(4096)
}

fn source_selection_step_seed(epoch_index: usize, absolute_step: usize, salt: usize) -> u64 {
    0x8B8B_4D1A_51E5_E1ECu64
        ^ (epoch_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (absolute_step as u64).rotate_left(17)
        ^ (salt as u64).rotate_left(31)
}

fn source_label_seed(label: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in label.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

impl ChunkedTokens {
    fn copy_into(&self, start: usize, dst: &mut [u32]) {
        let mut remaining = dst.len();
        let mut written = 0usize;
        let mut cursor = start;

        while remaining > 0 {
            let chunk_idx = self
                .chunks
                .partition_point(|chunk| chunk.token_offset + chunk.token_count <= cursor)
                .min(self.chunks.len().saturating_sub(1));
            let chunk = &self.chunks[chunk_idx];
            let chunk_data = self.chunk_data(chunk_idx);
            let chunk_tokens = mmap_as_u32_slice(&chunk_data, chunk.token_count);
            let chunk_start = cursor.saturating_sub(chunk.token_offset);
            let copy_len = chunk.token_count.saturating_sub(chunk_start).min(remaining);
            dst[written..written + copy_len]
                .copy_from_slice(&chunk_tokens[chunk_start..chunk_start + copy_len]);
            cursor += copy_len;
            written += copy_len;
            remaining -= copy_len;
        }
    }

    fn chunk_data(&self, chunk_idx: usize) -> Arc<Mmap> {
        let chunk = &self.chunks[chunk_idx];
        load_cached_chunk_from_mutex(
            &self.cache,
            self.cache_limit,
            chunk_idx,
            &chunk.path,
            chunk.token_count,
            "universality",
        )
    }
}

fn validate_pretokenized_tokenizer(tokenizer_cfg: &TokenizerConfig) -> io::Result<SharedTokenizer> {
    if !matches!(tokenizer_cfg.kind, TokenizerKind::Pretokenized(_)) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "universality datasets require tokenizer.type = `pretokenized`",
        ));
    }
    tokenizer_cfg
        .fit(std::iter::empty())
        .map_err(io::Error::other)
}

fn validate_tokenizer_against_manifest(
    tokenizer: &dyn crate::tokenizer::Tokenizer,
    manifest: &burn_dragon_universality::UniversalityTokenizerManifest,
) -> io::Result<()> {
    if tokenizer.len() != manifest.vocab_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "universality dataset tokenizer vocab mismatch (config={} manifest={})",
                tokenizer.len(),
                manifest.vocab_size
            ),
        ));
    }
    if tokenizer.bos_id() != manifest.bos_id
        || tokenizer.eos_id() != manifest.eos_id
        || tokenizer.pad_id() != manifest.pad_id
        || tokenizer.unk_id() != manifest.unk_id
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "universality dataset tokenizer special ids do not match manifest",
        ));
    }
    Ok(())
}

fn config_file_display_name(file_stem: &str) -> &str {
    file_stem
}

fn runtime_chunk_cache_limit() -> usize {
    std::env::var("DragonModel_PREPARED_TOKEN_CACHE_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_RUNTIME_CHUNK_CACHE_LIMIT)
}

fn runtime_document_cache_limit(
    batch_size: usize,
    train_samples: usize,
    validation_samples: usize,
) -> usize {
    std::env::var("DragonModel_UNIVERSALITY_RUNTIME_DOCUMENT_CACHE_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| {
            DEFAULT_RUNTIME_DOCUMENT_CACHE_LIMIT
                .max(batch_size.saturating_mul(8))
                .max(
                    train_samples
                        .saturating_mul(2)
                        .saturating_add(validation_samples),
                )
        })
}

fn runtime_generation_worker_count(sample_count: usize) -> usize {
    let available = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(4);
    let configured = std::env::var("DragonModel_UNIVERSALITY_GENERATION_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| available.min(DEFAULT_RUNTIME_GENERATION_WORKER_LIMIT));
    configured.min(sample_count).max(1)
}

fn fixed_manifest_logical_document_tokens(
    manifest: &burn_dragon_universality::UniversalityCorpusManifest,
) -> io::Result<Option<usize>> {
    let train_samples = manifest.stats.train_samples;
    let val_samples = manifest.stats.validation_samples;
    let train_doc_tokens = manifest
        .train_token_count
        .checked_div(train_samples)
        .and_then(|per_doc| {
            manifest
                .train_token_count
                .is_multiple_of(train_samples)
                .then_some(per_doc)
        });
    let val_doc_tokens = manifest
        .val_token_count
        .checked_div(val_samples)
        .and_then(|per_doc| {
            manifest
                .val_token_count
                .is_multiple_of(val_samples)
                .then_some(per_doc)
        });
    let document_token_count = match (train_doc_tokens, val_doc_tokens) {
        (Some(train), Some(val)) if train == val => Some(train),
        (Some(train), None) => Some(train),
        (None, Some(val)) => Some(val),
        (None, None) => None,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "universality manifest has inconsistent prepared document lengths across splits",
            ));
        }
    };

    match document_token_count {
        Some(0) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "universality manifest document token count must be > 0",
        )),
        Some(count) => Ok(Some(count.saturating_sub(1))),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{PretokenizedTokenizerConfig, TokenizerConfig};
    use burn_dragon_universality::config::NcaCorpusConfig;
    use burn_dragon_universality::{
        NcaSerializationConfig, NcaTokenizationConfig, RuliadCorpusConfig, RuliadDocumentMode,
        RuliadFamilyConfig, RuliadFamilyKind, RuliadSerializationConfig, RuliadTokenizationConfig,
        generate_nca_corpus,
    };
    use burn_ndarray::NdArray;
    use tempfile::tempdir;

    fn pretokenized_tokenizer() -> TokenizerConfig {
        TokenizerConfig {
            vocab_path: None,
            kind: TokenizerKind::Pretokenized(PretokenizedTokenizerConfig {
                vocab_size: 50_257,
                bos_id: None,
                eos_id: Some(50_256),
                pad_id: None,
                unk_id: None,
            }),
        }
    }

    fn fixed_runtime_config() -> NcaCorpusConfig {
        let mut config = NcaCorpusConfig {
            output_dir: "ignored".into(),
            seed: 1337,
            name: "runtime".to_string(),
            train_samples: 8,
            validation_samples: 4,
            chunk_token_capacity: 1024,
            serialization: NcaSerializationConfig::default(),
            tokenization: NcaTokenizationConfig::default(),
            families: burn_dragon_universality::config::default_families(),
        };
        for family in &mut config.families {
            family.grid_size =
                Some(burn_dragon_universality::UsizeRangeConfig { min: 12, max: 12 });
            family.steps = Some(burn_dragon_universality::UsizeRangeConfig { min: 10, max: 10 });
            family.state_count =
                Some(burn_dragon_universality::UsizeRangeConfig { min: 10, max: 10 });
            family.step_stride =
                Some(burn_dragon_universality::UsizeRangeConfig { min: 2, max: 2 });
            family.start_step = Some(burn_dragon_universality::UsizeRangeConfig { min: 0, max: 0 });
            family.identity_bias =
                Some(burn_dragon_universality::FloatRangeConfig { min: 0.0, max: 0.0 });
            family.temperature =
                Some(burn_dragon_universality::FloatRangeConfig { min: 0.0, max: 0.0 });
        }
        config
    }

    fn fixed_ruliad_runtime_config() -> RuliadCorpusConfig {
        RuliadCorpusConfig {
            output_dir: "ignored".into(),
            seed: 1337,
            name: "ruliad-runtime".to_string(),
            train_samples: 8,
            validation_samples: 4,
            chunk_token_capacity: 1024,
            serialization: RuliadSerializationConfig {
                document_tokens: 513,
                preview_samples: 2,
                ..RuliadSerializationConfig::default()
            },
            tokenization: RuliadTokenizationConfig::default(),
            source_selection: burn_dragon_universality::RuliadSourceSelectionConfig::default(),
            families: vec![
                RuliadFamilyConfig {
                    kind: RuliadFamilyKind::Eca,
                    weight: 2,
                    width: Some(burn_dragon_universality::UsizeRangeConfig { min: 12, max: 12 }),
                    steps: Some(burn_dragon_universality::UsizeRangeConfig { min: 4, max: 4 }),
                },
                RuliadFamilyConfig {
                    kind: RuliadFamilyKind::Simulation,
                    weight: 1,
                    width: Some(burn_dragon_universality::UsizeRangeConfig { min: 12, max: 12 }),
                    steps: Some(burn_dragon_universality::UsizeRangeConfig { min: 4, max: 4 }),
                },
            ],
            proof_tasks: None,
            lean_task_limit: None,
        }
    }

    fn live_ruliad_runtime_config() -> RuliadCorpusConfig {
        let mut config = fixed_ruliad_runtime_config();
        config.source_selection.enabled = true;
        config
    }

    #[test]
    fn universality_dataset_loads_generated_manifest() {
        let dir = tempdir().expect("tempdir");
        let corpus_dir = dir.path().join("corpus");
        let mut config = fixed_runtime_config();
        config.output_dir = corpus_dir.clone();
        config.train_samples = 4;
        config.validation_samples = 2;
        config.chunk_token_capacity = 128;
        config.name = "dataset".to_string();
        let report = generate_nca_corpus(&config).expect("generate corpus");
        let dataset =
            UniversalityDataset::new(&report.manifest_path, 16, 2, 0.9, &pretokenized_tokenizer())
                .expect("load universality dataset");
        assert_eq!(
            dataset.token_count(),
            report.train_token_count + report.val_token_count
        );
        assert_eq!(
            dataset.preferred_logical_document_tokens(DatasetSplit::Train),
            Some(380)
        );
        let mut buffer = vec![0u32; 17];
        dataset.copy_token_range(0, &mut buffer);
        assert!(buffer.iter().any(|value| *value != 0));
    }

    #[test]
    fn nca_manifest_rejects_block_sizes_longer_than_prepared_document() {
        let dir = tempdir().expect("tempdir");
        let corpus_dir = dir.path().join("corpus");
        let mut config = fixed_runtime_config();
        config.output_dir = corpus_dir.clone();
        config.train_samples = 4;
        config.validation_samples = 2;
        config.chunk_token_capacity = 128;
        config.name = "dataset".to_string();
        let report = generate_nca_corpus(&config).expect("generate corpus");
        let error = match UniversalityDataset::new(
            &report.manifest_path,
            512,
            2,
            0.9,
            &pretokenized_tokenizer(),
        ) {
            Ok(_) => panic!("manifest should reject overlong block size"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("exceeds prepared NCA logical document length"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn on_the_fly_universality_dataset_is_deterministic() {
        let dir = tempdir().expect("tempdir");
        let config_path = dir.path().join("nca.toml");
        let config = fixed_runtime_config();
        fs::write(&config_path, toml::to_string_pretty(&config).expect("toml"))
            .expect("write config");
        let dataset = UniversalityDataset::new_on_the_fly(
            &config_path,
            32,
            2,
            None,
            &pretokenized_tokenizer(),
        )
        .expect("load on-the-fly dataset");
        assert_eq!(
            dataset.preferred_logical_document_tokens(DatasetSplit::Train),
            Some(380)
        );

        let mut first = vec![0u32; 32];
        let mut second = vec![0u32; 32];
        dataset.copy_token_range(0, &mut first);
        dataset.copy_token_range(0, &mut second);
        assert_eq!(first, second);
    }

    #[test]
    fn on_the_fly_universality_dataset_epoch_stream_is_deterministic_across_instances() {
        let dir = tempdir().expect("tempdir");
        let config_path = dir.path().join("nca.toml");
        let config = fixed_runtime_config();
        fs::write(&config_path, toml::to_string_pretty(&config).expect("toml"))
            .expect("write config");

        let dataset_a = UniversalityDataset::new_on_the_fly(
            &config_path,
            32,
            2,
            None,
            &pretokenized_tokenizer(),
        )
        .expect("load on-the-fly dataset a");
        let dataset_b = UniversalityDataset::new_on_the_fly(
            &config_path,
            32,
            2,
            None,
            &pretokenized_tokenizer(),
        )
        .expect("load on-the-fly dataset b");

        dataset_a.prefetch_epoch(DatasetSplit::Train, 4);
        dataset_a.prepare_epoch(DatasetSplit::Train, 4);
        dataset_b.prepare_epoch(DatasetSplit::Train, 4);

        let mut epoch4_a = vec![0u32; 64];
        let mut epoch4_b = vec![0u32; 64];
        dataset_a.copy_token_range_with_epoch(DatasetSplit::Train, 4, 0, &mut epoch4_a);
        dataset_b.copy_token_range_with_epoch(DatasetSplit::Train, 4, 0, &mut epoch4_b);
        assert_eq!(epoch4_a, epoch4_b);

        let mut epoch5_a = vec![0u32; 64];
        let mut epoch5_b = vec![0u32; 64];
        dataset_a.copy_token_range_with_epoch(DatasetSplit::Train, 5, 0, &mut epoch5_a);
        dataset_b.copy_token_range_with_epoch(DatasetSplit::Train, 5, 0, &mut epoch5_b);
        assert_eq!(epoch5_a, epoch5_b);
        assert_ne!(epoch4_a, epoch5_a);
    }

    #[test]
    fn on_the_fly_universality_dataset_spans_documents_without_materializing_corpus() {
        let dir = tempdir().expect("tempdir");
        let config_path = dir.path().join("nca.toml");
        let config = fixed_runtime_config();
        let document_token_count =
            burn_dragon_universality::fixed_document_token_count(&config).expect("doc tokens");
        fs::write(&config_path, toml::to_string_pretty(&config).expect("toml"))
            .expect("write config");
        let dataset = UniversalityDataset::new_on_the_fly(
            &config_path,
            32,
            2,
            None,
            &pretokenized_tokenizer(),
        )
        .expect("load on-the-fly dataset");
        let mut buffer = vec![0u32; 48];
        dataset.copy_token_range(document_token_count.saturating_sub(24), &mut buffer);
        assert!(buffer.iter().any(|value| *value != 0));
        assert_eq!(
            dataset.train_len(),
            config.train_samples * document_token_count
        );
    }

    #[test]
    fn on_the_fly_universality_dataset_adapts_document_length_for_large_block_size() {
        let dir = tempdir().expect("tempdir");
        let config_path = dir.path().join("nca.toml");
        let config = fixed_runtime_config();
        fs::write(&config_path, toml::to_string_pretty(&config).expect("toml"))
            .expect("write config");

        let dataset = UniversalityDataset::new_on_the_fly(
            &config_path,
            4096,
            16,
            Some(4096),
            &pretokenized_tokenizer(),
        )
        .expect("load adapted on-the-fly dataset");

        assert!(dataset.block_size() == 4096);
        assert_eq!(
            dataset.preferred_logical_document_tokens(DatasetSplit::Train),
            Some(4104)
        );
        let mut buffer = vec![0u32; 4097];
        dataset.copy_token_range(0, &mut buffer);
        assert!(buffer.iter().any(|value| *value != 0));
    }

    #[test]
    fn on_the_fly_ruliad_dataset_is_deterministic() {
        let dir = tempdir().expect("tempdir");
        let config_path = dir.path().join("ruliad.toml");
        let config = fixed_ruliad_runtime_config();
        fs::write(&config_path, toml::to_string_pretty(&config).expect("toml"))
            .expect("write config");

        let dataset = UniversalityDataset::new_ruliad_on_the_fly(
            &config_path,
            32,
            2,
            &pretokenized_tokenizer(),
        )
        .expect("load ruliad dataset");
        assert_eq!(
            dataset.source_kind_label(),
            "on-the-fly universality ruliad"
        );
        assert_eq!(
            dataset.preferred_logical_document_tokens(DatasetSplit::Train),
            Some(512)
        );

        let mut first = vec![0u32; 64];
        let mut second = vec![0u32; 64];
        dataset.copy_token_range_with_epoch(DatasetSplit::Train, 2, 0, &mut first);
        dataset.copy_token_range_with_epoch(DatasetSplit::Train, 2, 0, &mut second);
        assert_eq!(first, second);

        let mut next_epoch = vec![0u32; 64];
        dataset.copy_token_range_with_epoch(DatasetSplit::Train, 3, 0, &mut next_epoch);
        assert_ne!(first, next_epoch);
    }

    #[test]
    fn on_the_fly_ruliad_dataset_exposes_multi_chunk_documents() {
        let dir = tempdir().expect("tempdir");
        let config_path = dir.path().join("ruliad-multichunk.toml");
        let mut config = fixed_ruliad_runtime_config();
        config.serialization.document_mode = RuliadDocumentMode::MultiChunkProofTree;
        config.serialization.document_chunks =
            burn_dragon_universality::UsizeRangeConfig { min: 3, max: 3 };
        fs::write(&config_path, toml::to_string_pretty(&config).expect("toml"))
            .expect("write config");

        let dataset = UniversalityDataset::new_ruliad_on_the_fly(
            &config_path,
            512,
            2,
            &pretokenized_tokenizer(),
        )
        .expect("load ruliad dataset");
        assert_eq!(
            dataset.preferred_logical_document_tokens(DatasetSplit::Train),
            Some(1538)
        );

        let mut prefix = vec![0u32; 128];
        let mut later = vec![0u32; 128];
        dataset.copy_token_range_with_epoch(DatasetSplit::Train, 0, 0, &mut prefix);
        dataset.copy_token_range_with_epoch(DatasetSplit::Train, 0, 700, &mut later);
        assert_ne!(prefix, later);
        assert!(later.iter().any(|token| *token != 0));
    }

    #[test]
    fn live_ruliad_source_selection_records_batch_loss_feedback() {
        let dir = tempdir().expect("tempdir");
        let config_path = dir.path().join("ruliad-live.toml");
        let config = live_ruliad_runtime_config();
        fs::write(&config_path, toml::to_string_pretty(&config).expect("toml"))
            .expect("write config");

        let dataset = UniversalityDataset::new_ruliad_on_the_fly(
            &config_path,
            32,
            2,
            &pretokenized_tokenizer(),
        )
        .expect("load ruliad dataset");
        assert!(dataset.uses_live_source_selection());
        let before = dataset.source_selection_snapshot().expect("snapshot");

        let storage = match &dataset.storage {
            UniversalityStorage::OnTheFly(storage) => storage,
            UniversalityStorage::Manifest(_) => panic!("expected on-the-fly storage"),
        };
        let indices = storage
            .source_selected_document_indices(burn_dragon_universality::SampleSplit::Train, 0, 0, 2)
            .expect("source-selected indices");
        assert_eq!(indices.len(), 2);
        let epoch = storage.epoch_documents(burn_dragon_universality::SampleSplit::Train, 0);
        assert!(
            epoch
                .documents_by_bucket
                .values()
                .any(|bucket_indices| indices.iter().all(|index| bucket_indices.contains(index))),
            "batch document indices should come from one source bucket"
        );
        assert!(
            storage
                .source_selected_document_indices(
                    burn_dragon_universality::SampleSplit::Validation,
                    0,
                    1,
                    2,
                )
                .is_none()
        );

        let after = dataset
            .record_source_selection_loss(0, 0.5)
            .expect("loss feedback");
        assert_ne!(before.mean_loss, after.mean_loss);
    }

    #[test]
    fn live_ruliad_source_weighted_validation_samples_without_feedback() {
        type TestBackend = NdArray<f32>;

        let dir = tempdir().expect("tempdir");
        let config_path = dir.path().join("ruliad-live.toml");
        let config = live_ruliad_runtime_config();
        fs::write(&config_path, toml::to_string_pretty(&config).expect("toml"))
            .expect("write config");

        let dataset = UniversalityDataset::new_ruliad_on_the_fly(
            &config_path,
            32,
            2,
            &pretokenized_tokenizer(),
        )
        .expect("load ruliad dataset");
        let before = dataset.source_selection_snapshot().expect("snapshot");
        let device = burn::tensor::Device::<TestBackend>::default();

        let first = dataset
            .sample_source_weighted_validation_batch::<TestBackend>(1, 41, 2, None, &device)
            .expect("source-weighted validation batch");
        let second = dataset
            .sample_source_weighted_validation_batch::<TestBackend>(1, 41, 2, None, &device)
            .expect("repeated source-weighted validation batch");
        assert_eq!(first.inputs.shape().dims::<2>(), [2, 32]);
        assert_eq!(
            first
                .inputs
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .expect("first tokens"),
            second
                .inputs
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .expect("second tokens")
        );

        let storage = match &dataset.storage {
            UniversalityStorage::OnTheFly(storage) => storage,
            UniversalityStorage::Manifest(_) => panic!("expected on-the-fly storage"),
        };
        assert!(
            storage
                .source_selected_document_indices(
                    burn_dragon_universality::SampleSplit::Validation,
                    1,
                    41,
                    2,
                )
                .is_none()
        );
        assert!(
            dataset.record_source_selection_loss(41, 0.25).is_none(),
            "mirror validation must not create pending source-selection feedback"
        );
        let after = dataset.source_selection_snapshot().expect("snapshot");
        assert_eq!(before.mean_loss, after.mean_loss);
    }
}
