use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::mem::size_of;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::sync_channel;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_dragon_time::Instant;
use memmap2::Mmap;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

use super::DatasetSplit;
use super::prepared_chunks::{ChunkRuntimeCache, load_cached_chunk_from_mutex, mmap_as_u32_slice};
use super::scheduler::{
    RuliadPolicyBatch, RuliadPolicySample, SequenceBatch, TokenSequenceDataset,
};
use crate::config::{RuliadSupervisionConfig, RuliadSupervisionMode};
use crate::summary_events::summary_event_mask_tensor;
use crate::tokenizer::{SharedTokenizer, TokenizerConfig, TokenizerKind};

const DEFAULT_RUNTIME_CHUNK_CACHE_LIMIT: usize = 8;
const DEFAULT_RUNTIME_DOCUMENT_CACHE_LIMIT: usize = 64;
const DEFAULT_RUNTIME_GENERATION_WORKER_LIMIT: usize = 32;
const DEFAULT_LIVE_SOURCE_SELECTION_DOCUMENTS_PER_STEP: usize = 4;
const DEFAULT_SOURCE_SELECTED_EOS_WINDOW_PROBABILITY: f64 = 0.05;
const SOURCE_WEIGHTED_VALIDATION_SPLIT_TAG: u8 = 2;
const RULIAD_SYMBOLIC_DATA_TOKEN: u32 = 261;
const RULIAD_SYMBOLIC_QUERY_TOKEN: u32 = 262;
const RULIAD_SYMBOLIC_PROOF_STEP_TOKEN: u32 = 263;
const RULIAD_SYMBOLIC_ANSWER_TOKEN: u32 = 264;
const RULIAD_SYMBOLIC_DOCUMENT_END_TOKEN: u32 = 265;
const RULIAD_SOURCE_SELECTION_STATE_VERSION: u32 = 1;

#[derive(Clone, Debug)]
pub struct RuliadValidationProbeItem {
    pub item: burn_dragon_universality::RuliadEvalItem,
    pub prompt_tokens: Vec<i64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct RuliadSourceSelectionStateSnapshot {
    pub version: u32,
    pub absolute_step_offset: usize,
    pub frontier_extension_count: usize,
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
    corpus_config: burn_dragon_universality::RuliadCorpusConfig,
    frontier_extension: burn_dragon_universality::RuliadFrontierExtensionConfig,
    cold_start: burn_dragon_universality::RuliadSourceSelectionColdStartConfig,
    frontier_extension_count: AtomicUsize,
    absolute_step_offset: AtomicUsize,
    pending: Mutex<HashMap<usize, String>>,
    pending_limit: usize,
    control: Mutex<LiveSourceSelectionControl>,
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

    fn encode_ruliad_payload_tokens(&self, _text: &str) -> Option<Vec<u32>> {
        None
    }

    fn decode_ruliad_payload_tokens(&self, _tokens: &[u32], _stop_at_eos: bool) -> Option<String> {
        None
    }

    fn ruliad_document_end_token_id(&self) -> Option<u32> {
        let tokens = self.encode_ruliad_payload_tokens("[/R2]")?;
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

    fn encode_ruliad_payload_tokens(&self, text: &str) -> Option<Vec<u32>> {
        Some(self.encode_payload_tokens(text))
    }

    fn decode_ruliad_payload_tokens(&self, tokens: &[u32], stop_at_eos: bool) -> Option<String> {
        Some(self.decode_payload_tokens(tokens, stop_at_eos))
    }
}

impl LiveSourceSelectionState {
    fn new(
        source_selection: burn_dragon_universality::RuliadSourceSelectionConfig,
        corpus_config: burn_dragon_universality::RuliadCorpusConfig,
        candidates: Vec<burn_dragon_universality::RuliadSamplerCandidate>,
    ) -> Option<Self> {
        if candidates.is_empty() {
            return None;
        }
        Some(Self {
            sampler: Mutex::new(burn_dragon_universality::RuliadFrontierSampler::new(
                source_selection.sampler,
                candidates,
            )),
            corpus_config,
            frontier_extension: source_selection.frontier_extension,
            cold_start: source_selection.cold_start,
            frontier_extension_count: AtomicUsize::new(0),
            absolute_step_offset: AtomicUsize::new(0),
            pending: Mutex::new(HashMap::new()),
            pending_limit: live_source_selection_pending_limit(),
            control: Mutex::new(LiveSourceSelectionControl::default()),
        })
    }

    fn from_snapshot(
        source_selection: burn_dragon_universality::RuliadSourceSelectionConfig,
        corpus_config: burn_dragon_universality::RuliadCorpusConfig,
        configured_candidates: Vec<burn_dragon_universality::RuliadSamplerCandidate>,
        snapshot: RuliadSourceSelectionStateSnapshot,
    ) -> Option<Self> {
        if snapshot.version != RULIAD_SOURCE_SELECTION_STATE_VERSION {
            return None;
        }
        let mut sampler = burn_dragon_universality::RuliadFrontierSampler::from_state(
            source_selection.sampler,
            snapshot.sampler,
        );
        sampler.add_candidates(configured_candidates);
        if sampler.candidates().is_empty() {
            return None;
        }
        Some(Self {
            sampler: Mutex::new(sampler),
            corpus_config,
            frontier_extension: source_selection.frontier_extension,
            cold_start: source_selection.cold_start,
            frontier_extension_count: AtomicUsize::new(snapshot.frontier_extension_count),
            absolute_step_offset: AtomicUsize::new(snapshot.absolute_step_offset),
            pending: Mutex::new(HashMap::new()),
            pending_limit: live_source_selection_pending_limit(),
            control: Mutex::new(snapshot.control.into()),
        })
    }

    fn export_state(&self, absolute_step_offset: usize) -> RuliadSourceSelectionStateSnapshot {
        self.absolute_step_offset
            .store(absolute_step_offset, Ordering::Relaxed);
        let sampler = self
            .sampler
            .lock()
            .expect("ruliad source sampler lock poisoned");
        let control = *self
            .control
            .lock()
            .expect("ruliad source control lock poisoned");
        RuliadSourceSelectionStateSnapshot {
            version: RULIAD_SOURCE_SELECTION_STATE_VERSION,
            absolute_step_offset,
            frontier_extension_count: self.frontier_extension_count.load(Ordering::Relaxed),
            control: control.into(),
            sampler: sampler.export_state(),
        }
    }

    fn effective_absolute_step(&self, absolute_step: Option<usize>) -> Option<usize> {
        absolute_step
            .map(|step| step.saturating_add(self.absolute_step_offset.load(Ordering::Relaxed)))
    }

    fn probabilities(&self) -> Vec<f32> {
        self.probabilities_for_step(None)
    }

    fn probabilities_for_step(&self, absolute_step: Option<usize>) -> Vec<f32> {
        let mut sampler = self
            .sampler
            .lock()
            .expect("ruliad source sampler lock poisoned");
        self.maybe_extend_frontier_locked(&mut sampler);
        let mut probabilities = sampler.probabilities();
        let control = *self
            .control
            .lock()
            .expect("ruliad source control lock poisoned");
        let effective_step = self.effective_absolute_step(absolute_step);
        apply_source_selection_control(&mut probabilities, sampler.candidates(), control);
        apply_source_selection_cold_start(
            &mut probabilities,
            sampler.candidates(),
            &self.cold_start,
            effective_step,
        );
        probabilities
    }

    fn weighted_bucket_labels(&self, absolute_step: Option<usize>) -> Vec<(String, f32)> {
        let mut sampler = self
            .sampler
            .lock()
            .expect("ruliad source sampler lock poisoned");
        self.maybe_extend_frontier_locked(&mut sampler);
        let mut probabilities = sampler.probabilities();
        let control = *self
            .control
            .lock()
            .expect("ruliad source control lock poisoned");
        let effective_step = self.effective_absolute_step(absolute_step);
        apply_source_selection_control(&mut probabilities, sampler.candidates(), control);
        apply_source_selection_cold_start(
            &mut probabilities,
            sampler.candidates(),
            &self.cold_start,
            effective_step,
        );
        sampler
            .candidates()
            .iter()
            .zip(probabilities)
            .map(|(candidate, weight)| {
                (
                    candidate.oracle_hash.clone(),
                    weight
                        .is_finite()
                        .then_some(weight)
                        .filter(|value| *value > 0.0)
                        .unwrap_or(1e-9),
                )
            })
            .collect()
    }

    fn apply_dynamics_control(&self, difficulty_pressure: f32, hash_noise_max_probability: f32) {
        let mut control = self
            .control
            .lock()
            .expect("ruliad source control lock poisoned");
        control.difficulty_pressure = difficulty_pressure.max(0.0);
        control.hash_noise_max_probability = hash_noise_max_probability.clamp(0.0, 1.0);
    }

    fn choose_bucket_for_step(
        &self,
        available: &HashMap<String, Vec<usize>>,
        epoch_index: usize,
        absolute_step: usize,
    ) -> Option<String> {
        self.choose_bucket_for_step_inner(available, epoch_index, absolute_step, true)
    }

    fn choose_bucket_label_for_step(
        &self,
        epoch_index: usize,
        absolute_step: usize,
    ) -> Option<String> {
        self.choose_bucket_label_for_step_inner(epoch_index, absolute_step, true)
    }

    fn choose_bucket_label_for_validation_step(
        &self,
        epoch_index: usize,
        absolute_step: usize,
    ) -> Option<String> {
        self.choose_bucket_label_for_step_inner(epoch_index, absolute_step, false)
    }

    fn choose_bucket_label_for_stream_step(
        &self,
        epoch_index: usize,
        selection_step: usize,
        feedback_step: usize,
    ) -> Option<String> {
        let label = self.choose_bucket_label_for_step_inner(epoch_index, selection_step, false)?;
        self.record_pending(feedback_step, &label);
        Some(label)
    }

    fn choose_bucket_label_for_step_inner(
        &self,
        epoch_index: usize,
        absolute_step: usize,
        record_pending: bool,
    ) -> Option<String> {
        let weighted = self.weighted_bucket_labels(Some(absolute_step));
        if weighted.is_empty() {
            return None;
        }
        let total = weighted.iter().map(|(_, weight)| *weight).sum::<f32>();
        let effective_step = self
            .effective_absolute_step(Some(absolute_step))
            .unwrap_or(absolute_step);
        let mut rng = StdRng::seed_from_u64(source_selection_step_seed(
            epoch_index,
            effective_step,
            weighted.len(),
        ));
        let ticket = rng.r#gen::<f32>() * total.max(1e-12);
        let mut cumulative = 0.0;
        for (label, weight) in weighted {
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

    fn choose_bucket_for_step_inner(
        &self,
        available: &HashMap<String, Vec<usize>>,
        epoch_index: usize,
        absolute_step: usize,
        record_pending: bool,
    ) -> Option<String> {
        let mut filtered = Vec::new();
        for (label, weight) in self.weighted_bucket_labels(Some(absolute_step)) {
            if available
                .get(&label)
                .is_some_and(|documents| !documents.is_empty())
            {
                filtered.push((label, weight));
            }
        }
        if filtered.is_empty() {
            return None;
        }
        let total = filtered.iter().map(|(_, weight)| *weight).sum::<f32>();
        let effective_step = self
            .effective_absolute_step(Some(absolute_step))
            .unwrap_or(absolute_step);
        let mut rng = StdRng::seed_from_u64(source_selection_step_seed(
            epoch_index,
            effective_step,
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
        self.maybe_extend_frontier_locked(&mut sampler);
        Some(self.snapshot_locked_for_step(&sampler, Some(absolute_step)))
    }

    fn record_capability_feedback(
        &self,
        report: &burn_dragon_universality::RuliadEvalReport,
    ) -> Option<burn_dragon_universality::RuliadMetricSnapshot> {
        let mut sampler = self
            .sampler
            .lock()
            .expect("ruliad source sampler lock poisoned");
        for feedback in ruliad_capability_feedback_from_report(report) {
            sampler.record_capability_feedback(&feedback);
        }
        self.maybe_extend_frontier_locked(&mut sampler);
        Some(self.snapshot_locked_for_step(&sampler, None))
    }

    fn snapshot(&self) -> burn_dragon_universality::RuliadMetricSnapshot {
        let mut sampler = self
            .sampler
            .lock()
            .expect("ruliad source sampler lock poisoned");
        self.maybe_extend_frontier_locked(&mut sampler);
        self.snapshot_locked_for_step(&sampler, None)
    }

    fn snapshot_locked(
        &self,
        sampler: &burn_dragon_universality::RuliadFrontierSampler,
    ) -> burn_dragon_universality::RuliadMetricSnapshot {
        self.snapshot_locked_for_step(sampler, None)
    }

    fn snapshot_locked_for_step(
        &self,
        sampler: &burn_dragon_universality::RuliadFrontierSampler,
        absolute_step: Option<usize>,
    ) -> burn_dragon_universality::RuliadMetricSnapshot {
        let mut probabilities = sampler.probabilities();
        let control = *self
            .control
            .lock()
            .expect("ruliad source control lock poisoned");
        let effective_step = self.effective_absolute_step(absolute_step);
        apply_source_selection_control(&mut probabilities, sampler.candidates(), control);
        apply_source_selection_cold_start(
            &mut probabilities,
            sampler.candidates(),
            &self.cold_start,
            effective_step,
        );
        let mut snapshot = sampler.snapshot_with_probabilities(&probabilities);
        snapshot.frontier_extension_count = self.frontier_extension_count.load(Ordering::Relaxed);
        snapshot.frontier_saturated = self.frontier_saturated(&snapshot);
        snapshot.frontier_unbounded =
            self.frontier_extension.enabled && self.frontier_extension.max_materialized_levels == 0;
        snapshot
    }

    fn maybe_extend_frontier_locked(
        &self,
        sampler: &mut burn_dragon_universality::RuliadFrontierSampler,
    ) {
        if !self.frontier_extension.enabled {
            return;
        }
        let snapshot = self.snapshot_locked(sampler);
        if !self.frontier_extension_pressure(&snapshot) || self.frontier_saturated(&snapshot) {
            return;
        }

        let next_level = snapshot.max_difficulty_level.saturating_add(1);
        let configured_min = self.corpus_config.source_selection.difficulty_levels.min;
        let current_materialized_levels = snapshot
            .max_difficulty_level
            .saturating_sub(configured_min)
            .saturating_add(1);
        let requested = self.frontier_extension.levels_per_extension.max(1);
        let allowed = if self.frontier_extension.max_materialized_levels == 0 {
            requested
        } else {
            self.frontier_extension
                .max_materialized_levels
                .saturating_sub(current_materialized_levels)
                .min(requested)
        };
        if allowed == 0 {
            return;
        }

        let mut new_candidates = Vec::new();
        for level in next_level..next_level.saturating_add(allowed) {
            new_candidates.extend(
                burn_dragon_universality::ruliad_sampler_candidates_for_difficulty(
                    &self.corpus_config,
                    level,
                ),
            );
        }
        if new_candidates.is_empty() {
            return;
        }
        let before = sampler.candidates().len();
        sampler.add_candidates(new_candidates);
        if sampler.candidates().len() > before {
            self.frontier_extension_count
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    fn frontier_extension_pressure(
        &self,
        snapshot: &burn_dragon_universality::RuliadMetricSnapshot,
    ) -> bool {
        self.frontier_extension.enabled && self.frontier_pressure_at_configured_edge(snapshot)
    }

    fn frontier_pressure_at_configured_edge(
        &self,
        snapshot: &burn_dragon_universality::RuliadMetricSnapshot,
    ) -> bool {
        let max_probability_ready = snapshot.max_difficulty_probability
            >= self
                .frontier_extension
                .extend_when_max_difficulty_probability_at_least;
        let normalized_ready = snapshot.normalized_difficulty_score
            >= self
                .frontier_extension
                .extend_when_normalized_difficulty_at_least;
        let mastered_ready = snapshot.mastered_probability
            >= self
                .corpus_config
                .source_selection
                .sampler
                .mastery_escape_threshold
                .clamp(0.0, 1.0);
        let frontier_easy = snapshot.target_loss.is_finite()
            && snapshot.target_loss > 0.0
            && snapshot.frontier_loss.is_finite()
            && snapshot.frontier_loss <= snapshot.target_loss;
        max_probability_ready && (normalized_ready || mastered_ready || frontier_easy)
    }

    fn frontier_saturated(
        &self,
        snapshot: &burn_dragon_universality::RuliadMetricSnapshot,
    ) -> bool {
        if !self.frontier_pressure_at_configured_edge(snapshot) {
            return false;
        }
        if !self.frontier_extension.enabled {
            return true;
        }
        if self.frontier_extension.max_materialized_levels == 0 {
            return false;
        }
        let configured_min = self.corpus_config.source_selection.difficulty_levels.min;
        let current_materialized_levels = snapshot
            .max_difficulty_level
            .saturating_sub(configured_min)
            .saturating_add(1);
        current_materialized_levels >= self.frontier_extension.max_materialized_levels
    }
}

fn ruliad_capability_feedback_from_report(
    report: &burn_dragon_universality::RuliadEvalReport,
) -> Vec<burn_dragon_universality::RuliadCapabilityFeedback> {
    let mut feedback = Vec::new();
    extend_ruliad_capability_feedback(&mut feedback, "difficulty", &report.difficulty_scores);
    extend_ruliad_capability_feedback(&mut feedback, "family", &report.family_scores);
    extend_ruliad_capability_feedback(&mut feedback, "task", &report.task_scores);
    feedback
}

fn extend_ruliad_capability_feedback(
    output: &mut Vec<burn_dragon_universality::RuliadCapabilityFeedback>,
    prefix: &str,
    groups: &[burn_dragon_universality::RuliadEvalGroupScore],
) {
    output.extend(groups.iter().map(|group| {
        let count = group.count.max(1);
        let unhealthy = group
            .schema_valid_wrong_count
            .saturating_add(group.malformed_completion_count)
            .saturating_add(group.missing_completion_count)
            .min(count);
        burn_dragon_universality::RuliadCapabilityFeedback {
            group_label: format!("{prefix}:{}", group.label),
            item_count: group.count,
            verifier_rate: group.verifier_accuracy,
            partial_credit_rate: group.partial_credit_rate,
            schema_valid_wrong_rate: group.schema_valid_wrong_count as f32 / count as f32,
            malformed_rate: group.malformed_completion_count as f32 / count as f32,
            missing_rate: group.missing_completion_count as f32 / count as f32,
            completion_health_rate: count.saturating_sub(unhealthy) as f32 / count as f32,
        }
    }));
}

fn apply_source_selection_control(
    probabilities: &mut [f32],
    candidates: &[burn_dragon_universality::RuliadSamplerCandidate],
    control: LiveSourceSelectionControl,
) {
    if probabilities.is_empty() || probabilities.len() != candidates.len() {
        return;
    }
    let max_difficulty = candidates
        .iter()
        .map(|candidate| candidate.difficulty_level)
        .max()
        .unwrap_or(0)
        .max(1);
    let pressure = control.difficulty_pressure.max(0.0);
    if (pressure - 1.0).abs() > f32::EPSILON {
        for (probability, candidate) in probabilities.iter_mut().zip(candidates) {
            if candidate.is_hash_noise {
                continue;
            }
            let normalized = candidate.difficulty_level as f32 / max_difficulty as f32;
            let boost = 1.0 + (pressure - 1.0) * normalized;
            *probability *= boost.max(0.0);
        }
    }
    let hash_probability = probabilities
        .iter()
        .zip(candidates)
        .filter_map(|(probability, candidate)| candidate.is_hash_noise.then_some(*probability))
        .sum::<f32>();
    let hash_max = control.hash_noise_max_probability.clamp(0.0, 1.0);
    if hash_probability > hash_max && hash_probability > f32::EPSILON {
        let scale = hash_max / hash_probability;
        for (probability, candidate) in probabilities.iter_mut().zip(candidates) {
            if candidate.is_hash_noise {
                *probability *= scale;
            }
        }
    }
    normalize_source_probabilities(probabilities);
}

fn apply_source_selection_cold_start(
    probabilities: &mut [f32],
    candidates: &[burn_dragon_universality::RuliadSamplerCandidate],
    cold_start: &burn_dragon_universality::RuliadSourceSelectionColdStartConfig,
    absolute_step: Option<usize>,
) {
    if probabilities.is_empty() || probabilities.len() != candidates.len() || !cold_start.enabled {
        return;
    }
    let Some(max_allowed_difficulty) =
        current_cold_start_max_difficulty(candidates, cold_start, absolute_step)
    else {
        return;
    };
    let mut changed = false;
    for (probability, candidate) in probabilities.iter_mut().zip(candidates) {
        if candidate.difficulty_level > max_allowed_difficulty {
            *probability = 0.0;
            changed = true;
        }
    }
    if changed {
        normalize_source_probabilities(probabilities);
    }
}

fn current_cold_start_max_difficulty(
    candidates: &[burn_dragon_universality::RuliadSamplerCandidate],
    cold_start: &burn_dragon_universality::RuliadSourceSelectionColdStartConfig,
    absolute_step: Option<usize>,
) -> Option<usize> {
    let absolute_step = absolute_step?;
    let min_difficulty = candidates
        .iter()
        .map(|candidate| candidate.difficulty_level)
        .min()
        .unwrap_or(0);
    let max_difficulty = candidates
        .iter()
        .map(|candidate| candidate.difficulty_level)
        .max()
        .unwrap_or(min_difficulty);
    let start_cap = cold_start
        .max_difficulty_level
        .max(min_difficulty)
        .min(max_difficulty);
    if start_cap >= max_difficulty {
        return None;
    }
    if absolute_step <= cold_start.hold_steps {
        return Some(start_cap);
    }
    let ramp_steps = cold_start.ramp_steps.max(1);
    let ramp_step = absolute_step.saturating_sub(cold_start.hold_steps);
    if ramp_step >= ramp_steps {
        return None;
    }
    let span = max_difficulty.saturating_sub(start_cap);
    let increment = span.saturating_mul(ramp_step) / ramp_steps;
    Some(start_cap.saturating_add(increment).min(max_difficulty))
}

fn normalize_source_probabilities(probabilities: &mut [f32]) {
    let sum = probabilities
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .sum::<f32>();
    if sum <= f32::EPSILON {
        let uniform = if probabilities.is_empty() {
            0.0
        } else {
            1.0 / probabilities.len() as f32
        };
        for probability in probabilities {
            *probability = uniform;
        }
        return;
    }
    for probability in probabilities {
        *probability = if probability.is_finite() && *probability > 0.0 {
            *probability / sum
        } else {
            0.0
        };
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
            ruliad_supervision: RuliadSupervisionConfig::default(),
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
            ruliad_supervision: RuliadSupervisionConfig::default(),
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
                    corpus.config().source_selection.clone(),
                    corpus.config().clone(),
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
            ruliad_supervision: RuliadSupervisionConfig::default(),
        })
    }

    pub fn with_ruliad_supervision(mut self, supervision: RuliadSupervisionConfig) -> Self {
        self.ruliad_supervision = supervision;
        self
    }

    fn ruliad_answer_completion_active(
        &self,
        split: DatasetSplit,
        epoch_index: usize,
        absolute_step: usize,
    ) -> bool {
        self.ruliad_supervision.prefer_answer_window(
            matches!(split, DatasetSplit::Val),
            epoch_index,
            absolute_step,
        )
    }

    pub fn with_source_selection_state_path(mut self, path: Option<&Path>) -> io::Result<Self> {
        if let Some(path) = path {
            self.load_source_selection_state(path)?;
        }
        Ok(self)
    }

    pub fn load_source_selection_state(&mut self, path: &Path) -> io::Result<()> {
        let contents = fs::read_to_string(path)?;
        let snapshot: RuliadSourceSelectionStateSnapshot =
            serde_json::from_str(&contents).map_err(io::Error::other)?;
        if snapshot.version != RULIAD_SOURCE_SELECTION_STATE_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported ruliad source-selection state version {} in {}; expected {}",
                    snapshot.version,
                    path.display(),
                    RULIAD_SOURCE_SELECTION_STATE_VERSION
                ),
            ));
        }
        let UniversalityStorage::OnTheFly(storage) = &mut self.storage else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "source-selection state can only be loaded into on-the-fly ruliad datasets",
            ));
        };
        let Some(configured_candidates) = storage.source_selection.as_ref().map(|_| {
            let sampler_config = storage
                .corpus
                .ruliad_config()
                .map(|config| config.source_selection.sampler)
                .unwrap_or_default();
            let candidates = storage
                .corpus
                .source_buckets()
                .into_iter()
                .map(|bucket| bucket.to_sampler_candidate(sampler_config));
            candidates.collect::<Vec<_>>()
        }) else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "source-selection state requires a live source-selection ruliad dataset",
            ));
        };
        let corpus_config = storage
            .corpus
            .ruliad_config()
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "source-selection state requires a ruliad corpus config",
                )
            })?
            .clone();
        let source_config = corpus_config.source_selection.clone();
        storage.source_selection = LiveSourceSelectionState::from_snapshot(
            source_config,
            corpus_config,
            configured_candidates,
            snapshot,
        )
        .map(Arc::new);
        if storage.source_selection.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid ruliad source-selection state in {}",
                    path.display()
                ),
            ));
        }
        Ok(())
    }

    pub fn write_source_selection_state(
        &self,
        path: &Path,
        absolute_step_offset: usize,
    ) -> io::Result<Option<RuliadSourceSelectionStateSnapshot>> {
        let snapshot = match &self.storage {
            UniversalityStorage::Manifest(_) => None,
            UniversalityStorage::OnTheFly(storage) => storage
                .source_selection
                .as_ref()
                .map(|source_selection| source_selection.export_state(absolute_step_offset)),
        };
        if let Some(snapshot) = &snapshot {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            let payload = serde_json::to_string_pretty(snapshot).map_err(io::Error::other)?;
            fs::write(path, payload)?;
        }
        Ok(snapshot)
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
        let prof_enabled = crate::train::profile::enabled();
        let cpu_start = prof_enabled.then(Instant::now);
        let storage = match &self.storage {
            UniversalityStorage::Manifest(_) => return None,
            UniversalityStorage::OnTheFly(storage) => storage,
        };
        let documents = storage.source_weighted_validation_documents(
            epoch_index,
            absolute_step,
            batch_size.max(1),
        )?;
        let eos_id = self.tokenizer.eos_id();
        let usable_lengths = documents
            .iter()
            .map(|document| valid_document_token_count(document, eos_id))
            .collect::<Vec<_>>();
        if usable_lengths
            .iter()
            .all(|usable_len| *usable_len <= self.block_size)
        {
            return None;
        }

        let batch_size = batch_size.max(1);
        let mut inputs = vec![0i64; batch_size * self.block_size];
        let mut targets = vec![0i64; batch_size * self.block_size];
        let answer_completion =
            self.ruliad_answer_completion_active(DatasetSplit::Val, epoch_index, absolute_step);
        let mut loss_mask = self
            .ruliad_supervision
            .uses_target_loss_mask()
            .then(|| vec![0i64; batch_size * self.block_size]);
        for (batch_idx, document) in documents.iter().enumerate() {
            let usable_len = usable_lengths
                .get(batch_idx)
                .copied()
                .unwrap_or_else(|| valid_document_token_count(document, eos_id));
            if usable_len <= self.block_size {
                return None;
            }
            let max_start_in_document = usable_len.saturating_sub(self.block_size + 1);
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
            let start = selected_window_start(
                document,
                usable_len,
                self.block_size,
                start,
                &mut rng,
                answer_completion,
            );
            for token_index in 0..self.block_size {
                let offset = batch_idx * self.block_size + token_index;
                inputs[offset] = document[start + token_index] as i64;
                targets[offset] = document[start + token_index + 1] as i64;
            }
            if let Some(mask) = loss_mask.as_mut() {
                ruliad_target_loss_mask(
                    &document[start..start + self.block_size + 1],
                    &mut mask[batch_idx * self.block_size..(batch_idx + 1) * self.block_size],
                    self.ruliad_supervision,
                );
            }
        }
        let cpu_ns = cpu_start
            .map(|start| start.elapsed().as_nanos())
            .unwrap_or_default();
        if prof_enabled {
            crate::train::profile::record_dataloader_foreground_wait(cpu_ns);
        }

        let tensor_copy_start = prof_enabled.then(Instant::now);
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
        let tensor_copy_ns = tensor_copy_start
            .map(|start| start.elapsed().as_nanos())
            .unwrap_or_default();
        if prof_enabled {
            let values = batch_size.saturating_mul(self.block_size);
            let copy_bytes = (values.saturating_mul(2).saturating_mul(size_of::<i64>())) as u128;
            crate::train::profile::record_dataloader(cpu_ns, tensor_copy_ns, copy_bytes, 0);
        }
        Some(
            SequenceBatch::new(inputs_tensor, targets_tensor, summary_event_mask).with_loss_mask(
                loss_mask.map(|mask| {
                    Tensor::<B, 2, Int>::from_data(
                        TensorData::new(mask, [batch_size, self.block_size]),
                        device,
                    )
                }),
            ),
        )
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

    pub fn record_ruliad_capability_feedback(
        &self,
        report: &burn_dragon_universality::RuliadEvalReport,
    ) -> Option<burn_dragon_universality::RuliadMetricSnapshot> {
        match &self.storage {
            UniversalityStorage::Manifest(_) => None,
            UniversalityStorage::OnTheFly(storage) => storage
                .source_selection
                .as_ref()
                .and_then(|source_selection| source_selection.record_capability_feedback(report)),
        }
    }

    pub fn apply_source_selection_dynamics_control(
        &self,
        difficulty_pressure: f32,
        hash_noise_max_probability: f32,
    ) {
        if let UniversalityStorage::OnTheFly(storage) = &self.storage
            && let Some(source_selection) = &storage.source_selection
        {
            source_selection
                .apply_dynamics_control(difficulty_pressure, hash_noise_max_probability);
        }
    }

    pub fn sample_ruliad_validation_probe_items(
        &self,
        epoch_index: usize,
        absolute_step: usize,
        max_items: usize,
    ) -> Vec<RuliadValidationProbeItem> {
        let UniversalityStorage::OnTheFly(storage) = &self.storage else {
            return Vec::new();
        };
        if max_items == 0 || storage.corpus.ruliad_config().is_none() {
            return Vec::new();
        }
        let sample_count = storage.corpus.validation_samples().max(1);
        let mut items = Vec::with_capacity(max_items);
        for item_rank in 0..max_items {
            let item_step = absolute_step.saturating_add(item_rank);
            let bucket_label = storage
                .source_selection
                .as_ref()
                .and_then(|source_selection| {
                    source_selection.choose_bucket_label_for_validation_step(epoch_index, item_step)
                });
            let sample_index = match bucket_label.as_deref() {
                Some(label) => live_source_selection_sample_index(
                    sample_count,
                    burn_dragon_universality::SampleSplit::Validation,
                    epoch_index,
                    item_step,
                    label,
                    item_rank,
                ),
                None => deterministic_validation_probe_sample_index(
                    sample_count,
                    epoch_index,
                    absolute_step,
                    item_rank,
                ),
            };
            let item_result = match bucket_label.as_deref() {
                Some(label) => storage.corpus.generate_ruliad_eval_item_for_source_bucket(
                    burn_dragon_universality::SampleSplit::Validation,
                    epoch_index,
                    sample_index,
                    label,
                ),
                None => storage.corpus.generate_ruliad_eval_item_for_epoch(
                    burn_dragon_universality::SampleSplit::Validation,
                    epoch_index,
                    sample_index,
                ),
            };
            let Ok(Some(item)) = item_result else {
                continue;
            };
            let Some(prompt_tokens) = storage.corpus.encode_ruliad_payload_tokens(&item.prompt)
            else {
                continue;
            };
            if prompt_tokens.is_empty() {
                continue;
            }
            items.push(RuliadValidationProbeItem {
                item,
                prompt_tokens: prompt_tokens.into_iter().map(i64::from).collect(),
            });
        }
        items
    }

    pub fn decode_ruliad_payload_tokens(
        &self,
        tokens: &[i64],
        stop_at_eos: bool,
    ) -> Option<String> {
        let UniversalityStorage::OnTheFly(storage) = &self.storage else {
            return None;
        };
        let tokens = tokens
            .iter()
            .filter_map(|token| (*token >= 0).then_some(*token as u32))
            .collect::<Vec<_>>();
        storage
            .corpus
            .decode_ruliad_payload_tokens(&tokens, stop_at_eos)
    }

    pub fn ruliad_document_end_token_id(&self) -> Option<u32> {
        let UniversalityStorage::OnTheFly(storage) = &self.storage else {
            return None;
        };
        storage.corpus.ruliad_document_end_token_id()
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

    fn source_selected_token_windows(
        &self,
        split: DatasetSplit,
        epoch_index: usize,
        absolute_step: usize,
        batch_size: usize,
        block_size: usize,
    ) -> Option<Vec<Vec<u32>>> {
        match (split, &self.storage) {
            (DatasetSplit::Train, UniversalityStorage::OnTheFly(storage)) => storage
                .source_selected_token_windows(
                    burn_dragon_universality::SampleSplit::Train,
                    epoch_index,
                    absolute_step,
                    batch_size,
                    block_size,
                    self.ruliad_answer_completion_active(split, epoch_index, absolute_step),
                ),
            (DatasetSplit::Val, UniversalityStorage::OnTheFly(storage)) => storage
                .source_selected_token_windows(
                    burn_dragon_universality::SampleSplit::Validation,
                    epoch_index,
                    absolute_step,
                    batch_size,
                    block_size,
                    self.ruliad_answer_completion_active(split, epoch_index, absolute_step),
                ),
            _ => None,
        }
    }

    fn source_selected_ruliad_policy_batch(
        &self,
        split: DatasetSplit,
        epoch_index: usize,
        absolute_step: usize,
        batch_size: usize,
    ) -> Option<RuliadPolicyBatch> {
        match (split, &self.storage) {
            (DatasetSplit::Train, UniversalityStorage::OnTheFly(storage)) => storage
                .source_selected_ruliad_policy_batch(
                    burn_dragon_universality::SampleSplit::Train,
                    epoch_index,
                    absolute_step,
                    batch_size,
                ),
            (DatasetSplit::Val, UniversalityStorage::OnTheFly(storage)) => storage
                .source_selected_ruliad_policy_batch(
                    burn_dragon_universality::SampleSplit::Validation,
                    epoch_index,
                    absolute_step,
                    batch_size,
                ),
            _ => None,
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
        match (split, &self.storage) {
            (DatasetSplit::Train, UniversalityStorage::OnTheFly(storage)) => storage
                .source_selected_stream_token_windows(
                    burn_dragon_universality::SampleSplit::Train,
                    epoch_index,
                    absolute_step,
                    chunk_index_in_document,
                    batch_size,
                    block_size,
                    matches!(
                        self.ruliad_supervision.mode,
                        RuliadSupervisionMode::AnswerWindow
                    ),
                ),
            (DatasetSplit::Val, UniversalityStorage::OnTheFly(storage)) => storage
                .source_selected_stream_token_windows(
                    burn_dragon_universality::SampleSplit::Validation,
                    epoch_index,
                    absolute_step,
                    chunk_index_in_document,
                    batch_size,
                    block_size,
                    matches!(
                        self.ruliad_supervision.mode,
                        RuliadSupervisionMode::AnswerWindow
                    ),
                ),
            _ => None,
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
        match (split, &self.storage) {
            (DatasetSplit::Train, UniversalityStorage::OnTheFly(storage)) => storage
                .source_selected_stream_token_windows_with_loss_masks(
                    burn_dragon_universality::SampleSplit::Train,
                    epoch_index,
                    absolute_step,
                    chunk_index_in_document,
                    batch_size,
                    block_size,
                    matches!(
                        self.ruliad_supervision.mode,
                        RuliadSupervisionMode::AnswerWindow
                    ),
                    self.ruliad_supervision,
                ),
            (DatasetSplit::Val, UniversalityStorage::OnTheFly(storage)) => storage
                .source_selected_stream_token_windows_with_loss_masks(
                    burn_dragon_universality::SampleSplit::Validation,
                    epoch_index,
                    absolute_step,
                    chunk_index_in_document,
                    batch_size,
                    block_size,
                    matches!(
                        self.ruliad_supervision.mode,
                        RuliadSupervisionMode::AnswerWindow
                    ),
                    self.ruliad_supervision,
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

    fn uses_target_loss_mask(&self) -> bool {
        self.ruliad_supervision.uses_target_loss_mask()
    }

    fn target_loss_mask_for_window(&self, window: &[u32], mask: &mut [i64]) -> bool {
        if !self.ruliad_supervision.uses_target_loss_mask() {
            return false;
        }
        ruliad_target_loss_mask(window, mask, self.ruliad_supervision)
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
        if self.source_selection.is_none() {
            return None;
        }
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

    fn source_selected_token_windows(
        &self,
        split: burn_dragon_universality::SampleSplit,
        epoch_index: usize,
        absolute_step: usize,
        batch_size: usize,
        block_size: usize,
        prefer_answer_window: bool,
    ) -> Option<Vec<Vec<u32>>> {
        let source_selection = self.source_selection.as_ref()?;
        let bucket_label = match split {
            burn_dragon_universality::SampleSplit::Train => {
                source_selection.choose_bucket_label_for_step(epoch_index, absolute_step)?
            }
            burn_dragon_universality::SampleSplit::Validation => source_selection
                .choose_bucket_label_for_validation_step(epoch_index, absolute_step)?,
        };
        let document_count = live_source_selection_documents_per_step(batch_size);
        let documents = self.generate_source_bucket_documents(
            split,
            epoch_index,
            absolute_step,
            &bucket_label,
            document_count,
        );
        Some(source_selected_windows_from_documents(
            &documents,
            self.corpus.eos_id(),
            epoch_index,
            absolute_step,
            &bucket_label,
            batch_size,
            block_size,
            prefer_answer_window,
        ))
    }

    fn source_selected_ruliad_policy_batch(
        &self,
        split: burn_dragon_universality::SampleSplit,
        epoch_index: usize,
        absolute_step: usize,
        batch_size: usize,
    ) -> Option<RuliadPolicyBatch> {
        let source_selection = self.source_selection.as_ref()?;
        let tokenization = self.corpus.ruliad_config()?.tokenization.clone();
        let bucket_label = match split {
            burn_dragon_universality::SampleSplit::Train => {
                source_selection.choose_bucket_label_for_step(epoch_index, absolute_step)?
            }
            burn_dragon_universality::SampleSplit::Validation => source_selection
                .choose_bucket_label_for_validation_step(epoch_index, absolute_step)?,
        };
        let sample_count = match split {
            burn_dragon_universality::SampleSplit::Train => self.corpus.train_samples(),
            burn_dragon_universality::SampleSplit::Validation => self.corpus.validation_samples(),
        }
        .max(1);
        let mut samples = Vec::with_capacity(batch_size.max(1));
        for sample_rank in 0..batch_size.max(1) {
            let sample_index = live_source_selection_sample_index(
                sample_count,
                split,
                epoch_index,
                absolute_step,
                &bucket_label,
                sample_rank,
            );
            let item = self
                .corpus
                .generate_ruliad_eval_item_for_source_bucket(
                    split,
                    epoch_index,
                    sample_index,
                    &bucket_label,
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "failed to generate live source-selected ruliad policy item split={split:?} epoch_index={epoch_index} absolute_step={absolute_step} sample_index={sample_index} bucket={bucket_label}: {error:#}"
                    )
                })?;
            let prompt_tokens = self
                .corpus
                .encode_ruliad_payload_tokens(&item.prompt)?
                .into_iter()
                .map(i64::from)
                .collect();
            samples.push(RuliadPolicySample {
                item,
                prompt_tokens,
            });
        }
        Some(RuliadPolicyBatch {
            samples,
            tokenization,
            stop_token_id: self.corpus.ruliad_document_end_token_id().map(i64::from),
        })
    }

    fn source_selected_stream_token_windows(
        &self,
        split: burn_dragon_universality::SampleSplit,
        epoch_index: usize,
        absolute_step: usize,
        chunk_index_in_document: usize,
        batch_size: usize,
        block_size: usize,
        prefer_answer_window: bool,
    ) -> Option<Vec<Vec<u32>>> {
        let source_selection = self.source_selection.as_ref()?;
        let selection_step =
            absolute_step.saturating_sub(chunk_index_in_document.min(absolute_step));
        let bucket_label = match split {
            burn_dragon_universality::SampleSplit::Train => source_selection
                .choose_bucket_label_for_stream_step(epoch_index, selection_step, absolute_step)?,
            burn_dragon_universality::SampleSplit::Validation => source_selection
                .choose_bucket_label_for_validation_step(epoch_index, selection_step)?,
        };
        let documents = self.generate_source_bucket_documents(
            split,
            epoch_index,
            selection_step,
            &bucket_label,
            batch_size.max(1),
        );
        Some(source_selected_stream_windows_from_documents(
            &documents,
            self.corpus.eos_id(),
            epoch_index,
            absolute_step,
            batch_size,
            block_size,
            chunk_index_in_document,
            prefer_answer_window,
        ))
    }

    fn source_selected_stream_token_windows_with_loss_masks(
        &self,
        split: burn_dragon_universality::SampleSplit,
        epoch_index: usize,
        absolute_step: usize,
        chunk_index_in_document: usize,
        batch_size: usize,
        block_size: usize,
        prefer_answer_window: bool,
        supervision: RuliadSupervisionConfig,
    ) -> Option<(Vec<Vec<u32>>, Option<Vec<Vec<i64>>>)> {
        let source_selection = self.source_selection.as_ref()?;
        let selection_step =
            absolute_step.saturating_sub(chunk_index_in_document.min(absolute_step));
        let bucket_label = match split {
            burn_dragon_universality::SampleSplit::Train => source_selection
                .choose_bucket_label_for_stream_step(epoch_index, selection_step, absolute_step)?,
            burn_dragon_universality::SampleSplit::Validation => source_selection
                .choose_bucket_label_for_validation_step(epoch_index, selection_step)?,
        };
        let documents = self.generate_source_bucket_documents(
            split,
            epoch_index,
            selection_step,
            &bucket_label,
            batch_size.max(1),
        );
        let windows = source_selected_stream_windows_from_documents(
            &documents,
            self.corpus.eos_id(),
            epoch_index,
            absolute_step,
            batch_size,
            block_size,
            chunk_index_in_document,
            prefer_answer_window,
        );
        let masks = supervision.uses_target_loss_mask().then(|| {
            if prefer_answer_window {
                windows
                    .iter()
                    .map(|window| {
                        let mut mask = vec![0; block_size];
                        ruliad_target_loss_mask(window, &mut mask, supervision);
                        mask
                    })
                    .collect()
            } else {
                source_selected_stream_loss_masks_from_documents(
                    &documents,
                    self.corpus.eos_id(),
                    batch_size,
                    block_size,
                    chunk_index_in_document,
                    supervision,
                )
            }
        });
        Some((windows, masks))
    }

    fn source_weighted_validation_documents(
        &self,
        epoch_index: usize,
        absolute_step: usize,
        batch_size: usize,
    ) -> Option<Vec<Arc<Vec<u32>>>> {
        let source_selection = self.source_selection.as_ref()?;
        let bucket_label =
            source_selection.choose_bucket_label_for_validation_step(epoch_index, absolute_step)?;
        Some(self.generate_source_bucket_documents(
            burn_dragon_universality::SampleSplit::Validation,
            epoch_index,
            absolute_step,
            &bucket_label,
            batch_size.max(1),
        ))
    }

    fn prepare_epoch(&self, split: burn_dragon_universality::SampleSplit, epoch_index: usize) {
        if self.source_selection.is_some() {
            return;
        }
        let _ = self.epoch_documents(split, epoch_index);
    }

    fn prefetch_epoch(&self, split: burn_dragon_universality::SampleSplit, epoch_index: usize) {
        if self.source_selection.is_some() {
            return;
        }
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

    fn generate_source_bucket_documents(
        &self,
        split: burn_dragon_universality::SampleSplit,
        epoch_index: usize,
        absolute_step: usize,
        bucket_label: &str,
        document_count: usize,
    ) -> Vec<Arc<Vec<u32>>> {
        let sample_count = match split {
            burn_dragon_universality::SampleSplit::Train => self.corpus.train_samples(),
            burn_dragon_universality::SampleSplit::Validation => self.corpus.validation_samples(),
        }
        .max(1);
        (0..document_count.max(1))
            .map(|document_rank| {
                let sample_index = live_source_selection_sample_index(
                    sample_count,
                    split,
                    epoch_index,
                    absolute_step,
                    bucket_label,
                    document_rank,
                );
                Arc::new(
                    self.corpus
                        .generate_document_tokens_for_source_bucket(
                            split,
                            epoch_index,
                            sample_index,
                            bucket_label,
                        )
                        .unwrap_or_else(|error| {
                            panic!(
                                "failed to generate live source-selected universality sample split={split:?} epoch_index={epoch_index} absolute_step={absolute_step} sample_index={sample_index} bucket={bucket_label}: {error:#}"
                            )
                        }),
                )
            })
            .collect()
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

fn live_source_selection_documents_per_step(batch_size: usize) -> usize {
    let configured = std::env::var("DragonModel_RULIAD_SOURCE_SELECTION_DOCUMENTS_PER_STEP")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0);
    bounded_live_source_selection_documents_per_step(batch_size, configured)
}

fn bounded_live_source_selection_documents_per_step(
    batch_size: usize,
    configured: Option<usize>,
) -> usize {
    configured
        .unwrap_or(DEFAULT_LIVE_SOURCE_SELECTION_DOCUMENTS_PER_STEP)
        .min(batch_size.max(1))
        .max(1)
}

fn live_source_selection_eos_window_probability() -> f64 {
    std::env::var("DragonModel_RULIAD_SOURCE_SELECTION_EOS_WINDOW_PROBABILITY")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .map(|value| value.clamp(0.0, 1.0))
        .unwrap_or(DEFAULT_SOURCE_SELECTED_EOS_WINDOW_PROBABILITY)
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

fn live_source_selection_sample_index(
    sample_count: usize,
    split: burn_dragon_universality::SampleSplit,
    epoch_index: usize,
    absolute_step: usize,
    bucket_label: &str,
    document_rank: usize,
) -> usize {
    let sample_count = sample_count.max(1);
    let split_salt = match split {
        burn_dragon_universality::SampleSplit::Train => split_tag(split) as usize,
        burn_dragon_universality::SampleSplit::Validation => {
            SOURCE_WEIGHTED_VALIDATION_SPLIT_TAG as usize
        }
    };
    let seed = source_selection_step_seed(
        epoch_index,
        absolute_step,
        source_label_seed(bucket_label) as usize
            ^ document_rank.rotate_left(7)
            ^ split_salt.rotate_left(17),
    );
    let mut rng = StdRng::seed_from_u64(seed);
    rng.gen_range(0..sample_count)
}

fn deterministic_validation_probe_sample_index(
    sample_count: usize,
    epoch_index: usize,
    absolute_step: usize,
    item_rank: usize,
) -> usize {
    let sample_count = sample_count.max(1);
    let seed = source_selection_step_seed(
        epoch_index,
        absolute_step,
        item_rank.rotate_left(11) ^ SOURCE_WEIGHTED_VALIDATION_SPLIT_TAG as usize,
    );
    let mut rng = StdRng::seed_from_u64(seed);
    rng.gen_range(0..sample_count)
}

fn source_selected_windows_from_documents(
    documents: &[Arc<Vec<u32>>],
    eos_id: Option<u32>,
    epoch_index: usize,
    absolute_step: usize,
    bucket_label: &str,
    batch_size: usize,
    block_size: usize,
    prefer_answer_window: bool,
) -> Vec<Vec<u32>> {
    if documents.is_empty() {
        return Vec::new();
    }
    let document_count = documents.len();
    let pad_token = eos_id.unwrap_or(0);
    (0..batch_size)
        .map(|batch_index| {
            let mut rng = StdRng::seed_from_u64(source_selection_step_seed(
                epoch_index,
                absolute_step,
                source_label_seed(bucket_label) as usize ^ batch_index.rotate_left(11),
            ));
            if prefer_answer_window {
                for attempt in 0..document_count {
                    let document = documents
                        .get((batch_index + attempt) % document_count)
                        .expect("source-selected document set must be non-empty");
                    let usable_len = valid_document_token_count(document, eos_id);
                    if let Some(window) = answer_window_from_document(
                        document, usable_len, block_size, pad_token, &mut rng,
                    ) {
                        return window;
                    }
                }
            }
            let document = documents
                .get(batch_index % document_count)
                .expect("source-selected document set must be non-empty");
            let usable_len = valid_document_token_count(document, eos_id);
            if usable_len <= block_size {
                return packed_valid_window_from_documents(
                    documents,
                    eos_id,
                    batch_index,
                    block_size,
                );
            }
            let max_start = usable_len.saturating_sub(block_size + 1);
            let start = if max_start == 0 {
                0
            } else {
                rng.gen_range(0..=max_start)
            };
            let start = selected_window_start(
                document,
                usable_len,
                block_size,
                start,
                &mut rng,
                prefer_answer_window,
            );
            document[start..start + block_size + 1].to_vec()
        })
        .collect()
}

fn source_selected_stream_windows_from_documents(
    documents: &[Arc<Vec<u32>>],
    eos_id: Option<u32>,
    epoch_index: usize,
    absolute_step: usize,
    batch_size: usize,
    block_size: usize,
    chunk_index_in_document: usize,
    prefer_answer_window: bool,
) -> Vec<Vec<u32>> {
    if documents.is_empty() {
        return Vec::new();
    }
    let pad_token = eos_id.unwrap_or(0);
    let start = chunk_index_in_document.saturating_mul(block_size);
    (0..batch_size)
        .map(|batch_index| {
            if prefer_answer_window {
                let mut rng = StdRng::seed_from_u64(source_selection_step_seed(
                    epoch_index,
                    absolute_step,
                    batch_index.rotate_left(13) ^ chunk_index_in_document.rotate_left(23),
                ));
                for attempt in 0..documents.len() {
                    let document = documents
                        .get((batch_index + attempt) % documents.len())
                        .expect("source-selected stream document set must be non-empty");
                    let usable_len = valid_document_token_count(document, eos_id);
                    if let Some(window) = answer_window_from_document(
                        document, usable_len, block_size, pad_token, &mut rng,
                    ) {
                        return window;
                    }
                }
            }
            let document = documents
                .get(batch_index % documents.len())
                .expect("source-selected stream document set must be non-empty");
            let usable_len = valid_document_token_count(document, eos_id);
            let mut window = Vec::with_capacity(block_size + 1);
            for offset in 0..=block_size {
                let index = start.saturating_add(offset);
                window.push(
                    document
                        .get(index)
                        .copied()
                        .filter(|_| index < usable_len)
                        .unwrap_or(pad_token),
                );
            }
            window
        })
        .collect()
}

fn source_selected_stream_loss_masks_from_documents(
    documents: &[Arc<Vec<u32>>],
    eos_id: Option<u32>,
    batch_size: usize,
    block_size: usize,
    chunk_index_in_document: usize,
    supervision: RuliadSupervisionConfig,
) -> Vec<Vec<i64>> {
    if documents.is_empty() {
        return Vec::new();
    }
    let start = chunk_index_in_document.saturating_mul(block_size);
    (0..batch_size)
        .map(|batch_index| {
            let document = documents
                .get(batch_index % documents.len())
                .expect("source-selected stream document set must be non-empty");
            let usable_len = valid_document_token_count(document, eos_id);
            let mut mask = vec![0; block_size];
            ruliad_target_loss_mask_for_document_range(
                document,
                usable_len,
                start,
                block_size,
                &mut mask,
                supervision,
            );
            mask
        })
        .collect()
}

fn ruliad_target_loss_mask_for_document_range(
    document: &[u32],
    usable_len: usize,
    start: usize,
    block_size: usize,
    mask: &mut [i64],
    supervision: RuliadSupervisionConfig,
) -> bool {
    mask.fill(0);
    let usable_len = usable_len.min(document.len());
    if usable_len < 2 || block_size == 0 {
        return false;
    }
    let mut document_mask = vec![0; usable_len - 1];
    if !ruliad_target_loss_mask(&document[..usable_len], &mut document_mask, supervision) {
        return false;
    }
    for (offset, slot) in mask.iter_mut().take(block_size).enumerate() {
        if let Some(value) = document_mask.get(start.saturating_add(offset)) {
            *slot = *value;
        }
    }
    mask.iter().any(|value| *value != 0)
}

fn valid_document_token_count(document: &[u32], eos_id: Option<u32>) -> usize {
    eos_id
        .and_then(|eos_id| {
            document
                .iter()
                .position(|token| *token == eos_id)
                .map(|index| index.saturating_add(1))
        })
        .unwrap_or(document.len())
        .min(document.len())
}

fn selected_window_start<R: Rng + ?Sized>(
    document: &[u32],
    usable_len: usize,
    block_size: usize,
    fallback_start: usize,
    rng: &mut R,
    prefer_answer_window: bool,
) -> usize {
    let max_start = usable_len.saturating_sub(block_size + 1);
    if max_start == 0 {
        return 0;
    }
    if prefer_answer_window {
        let candidates = answer_window_start_candidates(document, usable_len, block_size);
        if !candidates.is_empty() {
            return candidates[rng.gen_range(0..candidates.len())].min(max_start);
        }
    }
    if rng.gen_bool(live_source_selection_eos_window_probability()) {
        return max_start;
    }
    let candidates = semantic_window_start_candidates(document, usable_len, block_size);
    if candidates.is_empty() || !rng.gen_bool(0.85) {
        return fallback_start.min(max_start);
    }
    candidates[rng.gen_range(0..candidates.len())].min(max_start)
}

fn answer_window_from_document<R: Rng + ?Sized>(
    document: &[u32],
    usable_len: usize,
    block_size: usize,
    pad_token: u32,
    rng: &mut R,
) -> Option<Vec<u32>> {
    if usable_len <= block_size + 1 {
        let mut window = vec![pad_token; block_size + 1];
        let copy_len = usable_len.min(document.len()).min(block_size + 1);
        window[..copy_len].copy_from_slice(&document[..copy_len]);
        let mut mask = vec![0; block_size];
        return ruliad_answer_target_loss_mask(&window, &mut mask).then_some(window);
    }
    let candidates = answer_window_start_candidates(document, usable_len, block_size);
    if candidates.is_empty() {
        return None;
    }
    let start = candidates[rng.gen_range(0..candidates.len())]
        .min(usable_len.saturating_sub(block_size + 1));
    Some(document[start..start + block_size + 1].to_vec())
}

fn answer_window_start_candidates(
    document: &[u32],
    usable_len: usize,
    block_size: usize,
) -> Vec<usize> {
    if usable_len <= block_size + 1 {
        return Vec::new();
    }
    let max_start = usable_len.saturating_sub(block_size + 1);
    let lead = (block_size / 8).max(1);
    let mut starts = document
        .iter()
        .take(usable_len)
        .enumerate()
        .filter_map(|(index, _token)| {
            if !is_ruliad_answer_marker_at(document, usable_len, index) {
                return None;
            }
            let start = index.saturating_sub(lead).min(max_start);
            let end = start.saturating_add(block_size + 1).min(document.len());
            let mut mask = vec![0; end.saturating_sub(start).saturating_sub(1)];
            (index >= start
                && index < start.saturating_add(block_size)
                && ruliad_answer_target_loss_mask(&document[start..end], &mut mask))
            .then_some(start)
        })
        .collect::<Vec<_>>();
    starts.sort_unstable();
    starts.dedup();
    starts
}

fn semantic_window_start_candidates(
    document: &[u32],
    usable_len: usize,
    block_size: usize,
) -> Vec<usize> {
    if usable_len <= block_size + 1 {
        return Vec::new();
    }
    let max_start = usable_len.saturating_sub(block_size + 1);
    let lead = (block_size / 4).max(1);
    let mut starts = document
        .iter()
        .take(usable_len)
        .enumerate()
        .filter_map(|(index, token)| {
            if !is_semantic_window_anchor(document, index, *token) {
                return None;
            }
            Some(index.saturating_sub(lead).min(max_start))
        })
        .collect::<Vec<_>>();
    starts.sort_unstable();
    starts.dedup();
    starts
}

fn is_semantic_window_anchor(document: &[u32], index: usize, token: u32) -> bool {
    if matches!(
        token,
        RULIAD_SYMBOLIC_DATA_TOKEN
            | RULIAD_SYMBOLIC_QUERY_TOKEN
            | RULIAD_SYMBOLIC_PROOF_STEP_TOKEN
            | RULIAD_SYMBOLIC_ANSWER_TOKEN
            | RULIAD_SYMBOLIC_DOCUMENT_END_TOKEN
    ) {
        return true;
    }
    let marker = matches!(
        token,
        token if token == u32::from(b'?')
            || token == u32::from(b'>')
            || token == u32::from(b'!')
            || token == u32::from(b'G')
    );
    marker && (index == 0 || document.get(index - 1) == Some(&u32::from(b'\n')))
}

fn ruliad_answer_target_loss_mask(window: &[u32], mask: &mut [i64]) -> bool {
    ruliad_answer_target_loss_mask_with_close_stride(window, mask, 1)
}

fn ruliad_answer_target_loss_mask_with_close_stride(
    window: &[u32],
    mask: &mut [i64],
    close_marker_stride: usize,
) -> bool {
    mask.fill(0);
    if window.len() < mask.len().saturating_add(1) {
        return false;
    }
    let mut in_answer = false;
    let mut in_close_marker = false;
    let mut answer_hash = 0u64;
    let mut any = false;
    for t in 0..mask.len() {
        let input = window[t];
        let target = window[t + 1];
        if input == RULIAD_SYMBOLIC_ANSWER_TOKEN
            || (input == u32::from(b':') && t > 0 && window.get(t - 1) == Some(&u32::from(b'!')))
        {
            in_answer = true;
        }
        let close_marker_start = target == RULIAD_SYMBOLIC_DOCUMENT_END_TOKEN
            || (target == u32::from(b'[') && window.get(t + 2) == Some(&u32::from(b'/')));
        let close_marker_end = target == RULIAD_SYMBOLIC_DOCUMENT_END_TOKEN
            || (in_close_marker && target == u32::from(b']'));
        let supervise_close_marker = close_marker_stride > 0
            && (close_marker_stride == 1 || answer_hash % close_marker_stride as u64 == 0);
        if in_answer && !matches!(target, RULIAD_SYMBOLIC_ANSWER_TOKEN) {
            if close_marker_start && !supervise_close_marker {
                in_answer = false;
                in_close_marker = false;
            } else {
                mask[t] = 1;
                any = true;
                if close_marker_start {
                    in_close_marker = true;
                } else if !in_close_marker {
                    answer_hash = answer_hash
                        .wrapping_mul(1_099_511_628_211)
                        .wrapping_add(u64::from(target).wrapping_add(1));
                }
            }
        }
        if close_marker_end {
            in_answer = false;
            in_close_marker = false;
        }
        if target == RULIAD_SYMBOLIC_ANSWER_TOKEN {
            in_answer = true;
            in_close_marker = false;
            answer_hash = 0;
        }
    }
    any
}

fn ruliad_target_loss_mask(
    window: &[u32],
    mask: &mut [i64],
    supervision: RuliadSupervisionConfig,
) -> bool {
    if window.len() < mask.len().saturating_add(1) {
        mask.fill(0);
        return false;
    }
    let mut any = if supervision.uses_answer_target_mask() {
        ruliad_answer_target_loss_mask_with_close_stride(
            window,
            mask,
            supervision.answer_close_marker_stride,
        )
    } else {
        mask.fill(1);
        !mask.is_empty()
    };
    if supervision.mask_high_entropy_spans {
        mask_high_entropy_ruliad_targets(window, mask);
        any = mask.iter().any(|value| *value != 0);
    }
    any
}

fn mask_high_entropy_ruliad_targets(window: &[u32], mask: &mut [i64]) {
    let mut index = 0usize;
    while index < window.len() {
        let explicit_hash = token_byte(window[index]) == Some(b'h')
            && index > 0
            && token_byte(window[index - 1]) == Some(b':')
            && window
                .get(index + 1)
                .and_then(|token| token_byte(*token))
                .is_some_and(is_hex_byte);
        if !explicit_hash {
            index = index.saturating_add(1);
            continue;
        }
        let start = index.saturating_add(1);
        let mut end = start;
        while end < window.len() && token_byte(window[end]).is_some_and(is_hex_byte) {
            end = end.saturating_add(1);
        }
        if end.saturating_sub(start) >= 8 {
            mask_target_token_indices(mask, start..end);
        }
        index = end.max(index.saturating_add(1));
    }
}

fn mask_target_token_indices(mask: &mut [i64], token_indices: std::ops::Range<usize>) {
    for token_index in token_indices {
        let Some(mask_index) = token_index.checked_sub(1) else {
            continue;
        };
        if let Some(slot) = mask.get_mut(mask_index) {
            *slot = 0;
        }
    }
}

fn token_byte(token: u32) -> Option<u8> {
    (token <= u8::MAX as u32).then_some(token as u8)
}

fn is_hex_byte(byte: u8) -> bool {
    byte.is_ascii_hexdigit()
}

fn is_ruliad_answer_marker_at(document: &[u32], usable_len: usize, index: usize) -> bool {
    document
        .get(index)
        .is_some_and(|token| *token == RULIAD_SYMBOLIC_ANSWER_TOKEN)
        || (index + 1 < usable_len
            && document.get(index) == Some(&u32::from(b'!'))
            && document.get(index + 1) == Some(&u32::from(b':')))
}

fn packed_valid_window_from_documents(
    documents: &[Arc<Vec<u32>>],
    eos_id: Option<u32>,
    first_document: usize,
    block_size: usize,
) -> Vec<u32> {
    let target_len = block_size.saturating_add(1);
    let mut window = Vec::with_capacity(target_len);
    if documents.is_empty() {
        window.resize(target_len, eos_id.unwrap_or(0));
        return window;
    }
    for offset in 0..documents.len().saturating_mul(2) {
        let document = documents
            .get((first_document + offset) % documents.len())
            .expect("source-selected document set must be non-empty");
        let usable_len = valid_document_token_count(document, eos_id);
        if usable_len == 0 {
            continue;
        }
        window.extend(document.iter().take(usable_len).copied());
        if window.len() >= target_len {
            break;
        }
    }
    let fill = eos_id.unwrap_or(0);
    while window.len() < target_len {
        window.push(fill);
    }
    window.truncate(target_len);
    window
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
    use crate::config::RuliadSupervisionMode;
    use crate::tokenizer::{PretokenizedTokenizerConfig, TokenizerConfig};
    use burn::data::dataloader::DataLoader;
    use burn_dragon_universality::config::NcaCorpusConfig;
    use burn_dragon_universality::{
        NcaSerializationConfig, NcaTokenizationConfig, RuliadCorpusConfig, RuliadDocumentMode,
        RuliadFamilyConfig, RuliadFamilyKind, RuliadSerializationConfig, RuliadTokenizationConfig,
        generate_nca_corpus, ruliad_sampler_candidates,
    };
    use burn_ndarray::NdArray;
    use std::sync::Arc;
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

    fn masked_ascii_targets(targets: &[i64], mask: &[i64]) -> String {
        targets
            .iter()
            .zip(mask.iter())
            .filter_map(|(target, mask)| {
                (*mask == 1 && (0..=255).contains(target)).then_some(*target as u8 as char)
            })
            .collect()
    }

    fn masked_ruliad_target_text(
        dataset: &crate::dataset::Dataset,
        targets: &[i64],
        mask: &[i64],
    ) -> String {
        let masked_tokens = targets
            .iter()
            .zip(mask.iter())
            .filter_map(|(target, mask)| (*mask == 1).then_some(*target))
            .collect::<Vec<_>>();
        dataset
            .decode_ruliad_payload_tokens(&masked_tokens, true)
            .unwrap_or_else(|| masked_ascii_targets(targets, mask))
    }

    #[test]
    fn ruliad_answer_target_loss_mask_marks_answer_payload_and_close() {
        let window = vec![
            RULIAD_SYMBOLIC_QUERY_TOKEN,
            11,
            RULIAD_SYMBOLIC_ANSWER_TOKEN,
            21,
            22,
            RULIAD_SYMBOLIC_DOCUMENT_END_TOKEN,
        ];
        let mut mask = vec![0; window.len() - 1];
        assert!(ruliad_answer_target_loss_mask(&window, &mut mask));
        assert_eq!(mask, vec![0, 0, 1, 1, 1]);
    }

    #[test]
    fn ruliad_answer_target_loss_mask_supports_byte_markers() {
        let window = b"?:q\n!:ok=1\n[/R2]"
            .iter()
            .map(|byte| u32::from(*byte))
            .collect::<Vec<_>>();
        let mut mask = vec![0; window.len() - 1];
        assert!(ruliad_answer_target_loss_mask(&window, &mut mask));
        let targets = window
            .iter()
            .skip(1)
            .zip(mask.iter())
            .filter_map(|(token, mask)| (*mask == 1).then_some(*token as u8 as char))
            .collect::<String>();
        assert_eq!(targets, "ok=1\n[/R2]");
    }

    #[test]
    fn ruliad_answer_target_loss_mask_can_thin_close_markers() {
        let window = b"?:q\n!:ok=1\n[/R2]"
            .iter()
            .map(|byte| u32::from(*byte))
            .collect::<Vec<_>>();
        let mut mask = vec![0; window.len() - 1];
        assert!(ruliad_target_loss_mask(
            &window,
            &mut mask,
            RuliadSupervisionConfig {
                mode: RuliadSupervisionMode::AnswerCompletion,
                answer_close_marker_stride: 0,
                ..Default::default()
            },
        ));
        let targets = window
            .iter()
            .skip(1)
            .zip(mask.iter())
            .filter_map(|(token, mask)| (*mask == 1).then_some(*token as u8 as char))
            .collect::<String>();
        assert_eq!(targets, "ok=1\n");
    }

    #[test]
    fn ruliad_target_loss_mask_suppresses_hash_payload_in_answers() {
        let window = b"?:q\n!:x:h0123456789abcdef;ok=1\n[/R2]"
            .iter()
            .map(|byte| u32::from(*byte))
            .collect::<Vec<_>>();
        let mut mask = vec![0; window.len() - 1];
        assert!(ruliad_target_loss_mask(
            &window,
            &mut mask,
            RuliadSupervisionConfig {
                mode: RuliadSupervisionMode::AnswerCompletion,
                mask_high_entropy_spans: true,
                ..Default::default()
            },
        ));
        let supervised = window
            .iter()
            .skip(1)
            .zip(mask.iter())
            .filter_map(|(token, mask)| (*mask == 1).then_some(*token as u8 as char))
            .collect::<String>();
        assert!(
            supervised.contains("x:h;ok=1"),
            "hash payload should be removed while answer structure remains: {supervised:?}"
        );
        assert!(
            !supervised.contains("0123456789abcdef"),
            "hash payload should not be supervised: {supervised:?}"
        );
    }

    #[test]
    fn ruliad_target_loss_mask_suppresses_hash_payload_in_full_windows() {
        let window = b"G:x:h0123456789abcdef;sum=12\n[/R2]"
            .iter()
            .map(|byte| u32::from(*byte))
            .collect::<Vec<_>>();
        let mut mask = vec![0; window.len() - 1];
        assert!(ruliad_target_loss_mask(
            &window,
            &mut mask,
            RuliadSupervisionConfig {
                mode: RuliadSupervisionMode::FullDocument,
                mask_high_entropy_spans: true,
                ..Default::default()
            },
        ));
        let supervised = window
            .iter()
            .skip(1)
            .zip(mask.iter())
            .filter_map(|(token, mask)| (*mask == 1).then_some(*token as u8 as char))
            .collect::<String>();
        assert!(supervised.contains(":x:h;sum=12"));
        assert!(!supervised.contains("0123456789abcdef"));
    }

    #[test]
    fn ruliad_answer_target_loss_mask_leaves_prefix_only_windows_empty() {
        let window = vec![RULIAD_SYMBOLIC_QUERY_TOKEN, 11, 12, 13];
        let mut mask = vec![1; window.len() - 1];
        assert!(!ruliad_answer_target_loss_mask(&window, &mut mask));
        assert_eq!(mask, vec![0, 0, 0]);
    }

    fn source_selection_candidate(
        difficulty_level: usize,
    ) -> burn_dragon_universality::RuliadSamplerCandidate {
        burn_dragon_universality::RuliadSamplerCandidate {
            oracle_hash: format!("candidate-{difficulty_level}"),
            family: "test".to_string(),
            task_kind: "test".to_string(),
            difficulty_level,
            params_hash: format!("{difficulty_level:016x}"),
            prior: 1.0,
            cost: 1.0,
            loss_ema: 0.0,
            previous_loss_ema: 0.0,
            gradient_alignment: 0.0,
            is_hash_noise: false,
        }
    }

    fn live_source_selection_state(dataset: &UniversalityDataset) -> Arc<LiveSourceSelectionState> {
        match &dataset.storage {
            UniversalityStorage::OnTheFly(storage) => storage
                .source_selection
                .as_ref()
                .expect("live source-selection state")
                .clone(),
            UniversalityStorage::Manifest(_) => panic!("expected on-the-fly ruliad dataset"),
        }
    }

    fn capability_group(
        label: &str,
        count: usize,
        verifier_accuracy: f32,
        partial_credit_rate: f32,
        schema_valid_wrong_count: usize,
        malformed_completion_count: usize,
        missing_completion_count: usize,
    ) -> burn_dragon_universality::RuliadEvalGroupScore {
        burn_dragon_universality::RuliadEvalGroupScore {
            label: label.to_string(),
            count,
            exact_match_count: 0,
            semantic_match_count: 0,
            verifier_match_count: (verifier_accuracy.clamp(0.0, 1.0) * count as f32).round()
                as usize,
            partial_credit_count: (partial_credit_rate.clamp(0.0, 1.0) * count as f32).round()
                as usize,
            schema_valid_wrong_count,
            malformed_completion_count,
            missing_completion_count,
            exact_accuracy: 0.0,
            semantic_accuracy: verifier_accuracy,
            verifier_accuracy,
            partial_credit_rate,
            mean_partial_progress: partial_credit_rate,
            answer_field_correct_count: (partial_credit_rate.clamp(0.0, 1.0) * count as f32).round()
                as usize,
            answer_field_expected_count: count,
            answer_field_accuracy: partial_credit_rate,
            answer_terminated_count: count.saturating_sub(malformed_completion_count),
            answer_termination_rate: count.saturating_sub(malformed_completion_count) as f32
                / count.max(1) as f32,
            mean_completion_quality: 1.0,
            expected_answer_distinct_fraction: 1.0,
            actual_answer_distinct_fraction: 1.0,
        }
    }

    fn capability_feedback_report(
        family: burn_dragon_universality::RuliadEvalGroupScore,
    ) -> burn_dragon_universality::RuliadEvalReport {
        let count = family.count;
        burn_dragon_universality::RuliadEvalReport {
            version: burn_dragon_universality::RULIAD_EVAL_REPORT_VERSION,
            reasoning_score_version: burn_dragon_universality::RULIAD_REASONING_SCORE_VERSION,
            dataset_name: "test".to_string(),
            item_count: count,
            scored_count: count,
            exact_match_count: 0,
            semantic_match_count: family.verifier_match_count,
            verifier_match_count: family.verifier_match_count,
            partial_credit_count: family.partial_credit_count,
            schema_valid_wrong_count: family.schema_valid_wrong_count,
            malformed_completion_count: family.malformed_completion_count,
            missing_completion_count: family.missing_completion_count,
            unexpected_completion_count: 0,
            exact_accuracy: 0.0,
            semantic_accuracy: family.semantic_accuracy,
            verifier_accuracy: family.verifier_accuracy,
            partial_credit_rate: family.partial_credit_rate,
            mean_partial_progress: family.mean_partial_progress,
            answer_field_correct_count: family.answer_field_correct_count,
            answer_field_expected_count: family.answer_field_expected_count,
            answer_field_accuracy: family.answer_field_accuracy,
            answer_terminated_count: family.answer_terminated_count,
            answer_termination_rate: family.answer_termination_rate,
            mean_completion_quality: family.mean_completion_quality,
            expected_answer_distinct_fraction: family.expected_answer_distinct_fraction,
            actual_answer_distinct_fraction: family.actual_answer_distinct_fraction,
            mean_certificate_prefix_coverage: 0.0,
            mean_completion_tokens: 8.0,
            canary_count: 0,
            canary_semantic_match_count: 0,
            family_scores: vec![family],
            task_scores: Vec::new(),
            difficulty_scores: Vec::new(),
            math_domain_scores: Vec::new(),
            reasoning_mode_scores: Vec::new(),
            failures: Vec::new(),
        }
    }

    fn high_difficulty_probability(
        state: &LiveSourceSelectionState,
        absolute_step: usize,
        min_difficulty_level: usize,
    ) -> f32 {
        let weighted = state.weighted_bucket_labels(Some(absolute_step));
        let difficulty_by_label = state
            .sampler
            .lock()
            .expect("ruliad sampler lock")
            .candidates()
            .iter()
            .map(|candidate| (candidate.oracle_hash.clone(), candidate.difficulty_level))
            .collect::<HashMap<_, _>>();
        weighted
            .iter()
            .filter_map(|(label, probability)| {
                difficulty_by_label
                    .get(label)
                    .is_some_and(|difficulty| *difficulty >= min_difficulty_level)
                    .then_some(*probability)
            })
            .sum()
    }

    #[test]
    fn live_ruliad_source_selection_cold_start_caps_and_releases_difficulty() {
        let candidates = (0..=4).map(source_selection_candidate).collect::<Vec<_>>();
        let cold_start = burn_dragon_universality::RuliadSourceSelectionColdStartConfig {
            enabled: true,
            max_difficulty_level: 2,
            hold_steps: 10,
            ramp_steps: 10,
        };

        let mut held = vec![0.2; candidates.len()];
        apply_source_selection_cold_start(&mut held, &candidates, &cold_start, Some(0));
        assert!(held[0] > 0.0);
        assert!(held[1] > 0.0);
        assert!(held[2] > 0.0);
        assert_eq!(held[3], 0.0);
        assert_eq!(held[4], 0.0);
        assert!((held.iter().sum::<f32>() - 1.0).abs() < 1e-6);

        let mut ramped = vec![0.2; candidates.len()];
        apply_source_selection_cold_start(&mut ramped, &candidates, &cold_start, Some(15));
        assert!(ramped[3] > 0.0);
        assert_eq!(ramped[4], 0.0);
        assert!((ramped.iter().sum::<f32>() - 1.0).abs() < 1e-6);

        let mut released = vec![0.2; candidates.len()];
        apply_source_selection_cold_start(&mut released, &candidates, &cold_start, Some(20));
        assert_eq!(released, vec![0.2; candidates.len()]);
    }

    #[test]
    fn live_ruliad_source_selection_state_handoff_continues_curriculum() {
        let dir = tempdir().expect("tempdir");
        let config_path = dir.path().join("ruliad-live.toml");
        let state_path = dir.path().join("source-selection-state.json");
        let mut config = live_ruliad_runtime_config();
        config.source_selection.difficulty_levels =
            burn_dragon_universality::UsizeRangeConfig { min: 0, max: 4 };
        config.source_selection.cold_start =
            burn_dragon_universality::RuliadSourceSelectionColdStartConfig {
                enabled: true,
                max_difficulty_level: 0,
                hold_steps: 100,
                ramp_steps: 100,
            };
        fs::write(&config_path, toml::to_string_pretty(&config).expect("toml"))
            .expect("write config");

        let dataset = UniversalityDataset::new_ruliad_on_the_fly(
            &config_path,
            32,
            2,
            &pretokenized_tokenizer(),
        )
        .expect("load ruliad dataset");
        let state = live_source_selection_state(&dataset);
        assert!(
            high_difficulty_probability(&state, 0, 1) < 1e-6,
            "fresh curriculum should still be held at the cold-start difficulty cap"
        );

        let chosen = state
            .choose_bucket_label_for_step(0, 7)
            .expect("source bucket");
        dataset
            .record_source_selection_loss(7, 0.73)
            .expect("loss feedback");
        dataset.apply_source_selection_dynamics_control(2.5, 0.25);
        let snapshot = dataset
            .write_source_selection_state(&state_path, 256)
            .expect("write source-selection state")
            .expect("source-selection snapshot");
        assert_eq!(snapshot.absolute_step_offset, 256);
        assert_eq!(snapshot.control.difficulty_pressure, 2.5);
        assert_eq!(snapshot.control.hash_noise_max_probability, 0.25);
        assert!(
            snapshot
                .sampler
                .candidates
                .iter()
                .any(|candidate| candidate.oracle_hash == chosen && candidate.loss_ema > 0.0),
            "exported sampler state should include AdamW feedback before EGGROLL handoff"
        );

        let restored = UniversalityDataset::new_ruliad_on_the_fly(
            &config_path,
            32,
            2,
            &pretokenized_tokenizer(),
        )
        .expect("load fresh ruliad dataset")
        .with_source_selection_state_path(Some(&state_path))
        .expect("restore source-selection state");
        let restored_state = live_source_selection_state(&restored);
        assert!(
            high_difficulty_probability(&restored_state, 0, 1) > 1e-4,
            "restored EGGROLL phase must apply the AdamW step offset instead of restarting cold-start"
        );

        let restored_snapshot = restored
            .write_source_selection_state(&dir.path().join("restored-state.json"), 256)
            .expect("write restored state")
            .expect("restored snapshot");
        assert_eq!(restored_snapshot.control, snapshot.control);
        assert!(
            restored_snapshot
                .sampler
                .candidates
                .iter()
                .any(|candidate| candidate.oracle_hash == chosen && candidate.loss_ema > 0.0),
            "restored sampler should preserve AdamW source-selection feedback"
        );
    }

    #[test]
    fn source_selected_window_sampler_includes_document_end_windows() {
        let mut document = vec![777u32; 512];
        document[511] = 50_256;
        let usable_len = valid_document_token_count(&document, Some(50_256));
        let block_size = 64;
        let max_start = usable_len.saturating_sub(block_size + 1);
        let mut rng = StdRng::seed_from_u64(1337);
        let mut end_count = 0usize;
        for _ in 0..128 {
            let start =
                selected_window_start(&document, usable_len, block_size, 0, &mut rng, false);
            end_count += usize::from(start == max_start);
        }
        assert!(
            end_count > 0,
            "source-selected windows should include document-end/EOS training targets"
        );
        assert!(
            end_count < 80,
            "EOS end-window sampling should remain mixed with interior windows: {end_count}"
        );
    }

    #[test]
    fn source_selected_window_sampler_uses_symbolic_ruliad_markers() {
        let mut document = vec![777u32; 512];
        document[40] = RULIAD_SYMBOLIC_DATA_TOKEN;
        document[128] = RULIAD_SYMBOLIC_QUERY_TOKEN;
        document[192] = RULIAD_SYMBOLIC_PROOF_STEP_TOKEN;
        document[256] = RULIAD_SYMBOLIC_ANSWER_TOKEN;
        document[320] = RULIAD_SYMBOLIC_DOCUMENT_END_TOKEN;
        document[360] = 4096;
        let usable_len = valid_document_token_count(&document, Some(4096));
        let starts = semantic_window_start_candidates(&document, usable_len, 64);
        assert!(
            starts.len() >= 5,
            "symbolic ruliad structural tokens should anchor semantic windows: {starts:?}"
        );
        assert!(starts.iter().any(|start| (24..=40).contains(start)));
        assert!(starts.iter().any(|start| (240..=256).contains(start)));
        let max_start = usable_len.saturating_sub(64 + 1);
        assert!(starts.contains(&max_start));
    }

    #[test]
    fn answer_target_window_sampler_prefers_nonempty_answer_masks() {
        let mut document = vec![777u32; 512];
        document[96] = RULIAD_SYMBOLIC_QUERY_TOKEN;
        document[256] = RULIAD_SYMBOLIC_ANSWER_TOKEN;
        document[257] = 31;
        document[258] = 32;
        document[259] = 33;
        document[260] = RULIAD_SYMBOLIC_DOCUMENT_END_TOKEN;
        document[360] = 4096;
        let usable_len = valid_document_token_count(&document, Some(4096));
        let mut rng = StdRng::seed_from_u64(1337);
        for _ in 0..64 {
            let start = selected_window_start(&document, usable_len, 64, 0, &mut rng, true);
            let window = &document[start..start + 65];
            let mut mask = vec![0; 64];
            assert!(
                ruliad_answer_target_loss_mask(window, &mut mask),
                "answer-target sampling must yield trainable answer targets"
            );
            assert!(
                mask.iter().any(|value| *value == 1),
                "answer-target window mask should not be empty"
            );
        }
    }

    #[test]
    fn live_source_selection_documents_per_step_is_bounded_by_default() {
        assert_eq!(
            bounded_live_source_selection_documents_per_step(32, None),
            DEFAULT_LIVE_SOURCE_SELECTION_DOCUMENTS_PER_STEP
        );
        assert_eq!(bounded_live_source_selection_documents_per_step(2, None), 2);
        assert_eq!(
            bounded_live_source_selection_documents_per_step(32, Some(8)),
            8
        );
        assert_eq!(
            bounded_live_source_selection_documents_per_step(2, Some(8)),
            2
        );
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
    fn on_the_fly_ruliad_validation_probe_items_are_verifiable() {
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
        let items = dataset.sample_ruliad_validation_probe_items(1, 0, 2);
        assert_eq!(items.len(), 2);
        for probe in items {
            assert!(!probe.prompt_tokens.is_empty());
            let decoded = dataset
                .decode_ruliad_payload_tokens(&probe.prompt_tokens, true)
                .expect("ruliad decode");
            assert!(decoded.contains("!:"));
            let completion = format!("!:{}", probe.item.expected_answer);
            let score = burn_dragon_universality::ruliad::score_ruliad_item_completion(
                &probe.item,
                Some(&completion),
            );
            assert!(
                score.verifier_match(),
                "oracle completion should verify for {}",
                probe.item.oracle_hash
            );
        }
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
        let wrapped = crate::dataset::Dataset::from_universality(dataset.clone());

        let storage = match &dataset.storage {
            UniversalityStorage::OnTheFly(storage) => storage,
            UniversalityStorage::Manifest(_) => panic!("expected on-the-fly storage"),
        };
        dataset.prepare_epoch(DatasetSplit::Train, 0);
        dataset.prefetch_epoch(DatasetSplit::Train, 1);
        let windows = crate::dataset::TokenSequenceDataset::source_selected_token_windows(
            &wrapped,
            DatasetSplit::Train,
            0,
            0,
            2,
            32,
        )
        .expect("source-selected token windows");
        assert_eq!(windows.len(), 2);
        assert!(windows.iter().all(|window| window.len() == 33));
        assert!(
            windows.iter().flatten().any(|token| *token != 0),
            "source-selected windows should contain generated content"
        );
        assert!(
            windows
                .iter()
                .all(|window| !contains_period_filler_pattern(window)),
            "source-selected training windows must not expose ruliad padding filler"
        );
        let policy_batch =
            crate::dataset::TokenSequenceDataset::source_selected_ruliad_policy_batch(
                &wrapped,
                DatasetSplit::Train,
                0,
                0,
                2,
            )
            .expect("source-selected ruliad policy batch");
        assert_eq!(policy_batch.samples.len(), 2);
        for sample in policy_batch.samples.iter() {
            assert!(!sample.prompt_tokens.is_empty());
            let prompt = dataset
                .decode_ruliad_payload_tokens(&sample.prompt_tokens, true)
                .expect("decode ruliad prompt");
            assert!(prompt.contains("!:"));
            let oracle_completion = format!("!:{}\n[/R2]", sample.item.expected_answer);
            let score = burn_dragon_universality::ruliad::score_ruliad_item_completion(
                &sample.item,
                Some(&oracle_completion),
            );
            assert!(score.verifier_match(), "oracle answer should verify");
        }
        let validation_windows =
            crate::dataset::TokenSequenceDataset::source_selected_token_windows(
                &wrapped,
                DatasetSplit::Val,
                0,
                3,
                2,
                32,
            )
            .expect("source-selected validation token windows");
        assert_eq!(validation_windows.len(), 2);
        assert!(
            validation_windows
                .iter()
                .all(|window| !contains_period_filler_pattern(window)),
            "source-selected validation windows must not expose ruliad padding filler"
        );
        {
            let cache = storage.cache.inner.lock().expect("runtime cache lock");
            assert!(
                cache.entries.is_empty(),
                "live source-selected training must not materialize full epoch caches"
            );
            assert!(
                cache.building.is_empty(),
                "live source-selected training must not leave background epoch builds"
            );
        }
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

        let after =
            crate::dataset::TokenSequenceDataset::record_source_selection_loss(&wrapped, 0, 0.5)
                .expect("loss feedback");
        assert_ne!(before.mean_loss, after.mean_loss);
    }

    #[test]
    fn live_ruliad_source_selection_records_capability_feedback() {
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
        let mut report =
            capability_feedback_report(capability_group("simulation", 16, 1.0, 0.5, 0, 0, 0));
        report.difficulty_scores = vec![capability_group("d0", 16, 1.0, 0.5, 0, 0, 0)];

        let after = dataset
            .record_ruliad_capability_feedback(&report)
            .expect("capability feedback snapshot");
        let difficulty = after
            .difficulty_buckets
            .iter()
            .find(|bucket| bucket.label == "d0")
            .expect("d0 difficulty bucket");

        assert!(after.mean_loss < before.mean_loss);
        assert!(
            difficulty.learning_progress > 0.0,
            "capability feedback should register progress for verified difficulty: {difficulty:?}"
        );
    }

    #[test]
    fn live_ruliad_streaming_records_chunk_loss_feedback_without_epoch_cache() {
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
        assert!(dataset.uses_live_source_selection());
        let wrapped = Arc::new(crate::dataset::Dataset::from_universality(dataset.clone()));
        let device = burn::tensor::Device::<TestBackend>::default();
        let loader = crate::dataset::StreamingDataLoader::<TestBackend>::new(
            Arc::clone(&wrapped),
            DatasetSplit::Train,
            &device,
            4,
            Some(4),
            Some(64),
            1337,
        );
        let mut iter = loader.iter();
        let first = iter.next().expect("first stream batch");
        let second = iter.next().expect("second stream batch");
        assert_eq!(first.inputs.shape().dims::<2>(), [2, 32]);
        assert_eq!(second.inputs.shape().dims::<2>(), [2, 32]);
        assert!(first.reset_stream_state);
        assert!(!second.reset_stream_state);
        assert!(
            first
                .inputs
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .expect("first tokens")
                .iter()
                .any(|token| *token != 0),
            "streaming source-selected batches should contain generated content"
        );
        assert!(
            wrapped.record_source_selection_loss(0, 0.5).is_some(),
            "first stream chunk should register pending source-selection feedback"
        );
        assert!(
            wrapped.record_source_selection_loss(1, 0.4).is_some(),
            "second stream chunk should register pending source-selection feedback"
        );
        assert!(
            wrapped.record_source_selection_loss(2, 0.3).is_none(),
            "unseen stream chunks should not fabricate source-selection feedback"
        );

        let storage = match &dataset.storage {
            UniversalityStorage::OnTheFly(storage) => storage,
            UniversalityStorage::Manifest(_) => panic!("expected on-the-fly storage"),
        };
        let cache = storage.cache.inner.lock().expect("runtime cache lock");
        assert!(
            cache.entries.is_empty(),
            "streaming live source-selection must not materialize full epoch caches"
        );
        assert!(
            cache.building.is_empty(),
            "streaming live source-selection must not leave background epoch builds"
        );
    }

    #[test]
    fn live_ruliad_answer_completion_streaming_preserves_context_state_and_masks_answers() {
        type TestBackend = NdArray<f32>;

        let dir = tempdir().expect("tempdir");
        let config_path = dir.path().join("ruliad-live.toml");
        let mut config = live_ruliad_runtime_config();
        config.families = vec![RuliadFamilyConfig {
            kind: RuliadFamilyKind::Simulation,
            weight: 1,
            width: Some(burn_dragon_universality::UsizeRangeConfig { min: 12, max: 12 }),
            steps: Some(burn_dragon_universality::UsizeRangeConfig { min: 4, max: 4 }),
        }];
        fs::write(&config_path, toml::to_string_pretty(&config).expect("toml"))
            .expect("write config");

        let dataset = UniversalityDataset::new_ruliad_on_the_fly(
            &config_path,
            32,
            2,
            &pretokenized_tokenizer(),
        )
        .expect("load ruliad dataset")
        .with_ruliad_supervision(RuliadSupervisionConfig {
            mode: RuliadSupervisionMode::AnswerCompletion,
            ..Default::default()
        });
        assert!(dataset.uses_live_source_selection());
        let wrapped = Arc::new(crate::dataset::Dataset::from_universality(dataset));
        let device = burn::tensor::Device::<TestBackend>::default();
        let loader = crate::dataset::StreamingDataLoader::<TestBackend>::new(
            Arc::clone(&wrapped),
            DatasetSplit::Train,
            &device,
            16,
            Some(16),
            Some(512),
            1337,
        );
        let mut iter = loader.iter();
        let first = iter.next().expect("first stream batch");
        let second = iter.next().expect("second stream batch");

        assert_eq!(first.inputs.shape().dims::<2>(), [2, 32]);
        assert_eq!(second.inputs.shape().dims::<2>(), [2, 32]);
        assert!(first.reset_stream_state);
        assert!(
            !second.reset_stream_state,
            "answer-completion masks should not force recurrent state resets"
        );
        let mut batches = vec![("first".to_string(), first), ("second".to_string(), second)];
        batches.extend(
            iter.take(14)
                .enumerate()
                .map(|(index, batch)| (format!("later-{index}"), batch)),
        );
        let mut saw_context_only_chunk = false;
        let mut answer_mask_rows = 0usize;
        let mut supervised_examples = Vec::new();
        for (_label, batch) in batches {
            let targets = batch
                .targets
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .expect("targets");
            let mask = batch
                .loss_mask
                .expect("answer-completion stream loss mask")
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .expect("loss mask");
            let supervised = masked_ruliad_target_text(wrapped.as_ref(), &targets, &mask);
            if mask.iter().all(|value| *value == 0) {
                saw_context_only_chunk = true;
            }
            if mask.iter().any(|value| *value == 1) {
                answer_mask_rows = answer_mask_rows.saturating_add(1);
                if supervised_examples.len() < 4 {
                    supervised_examples.push(supervised.clone());
                }
            }
        }
        assert!(
            saw_context_only_chunk,
            "streaming answer-completion should preserve prompt/proof context chunks before the answer"
        );
        assert!(
            answer_mask_rows > 0,
            "streaming answer-completion should eventually supervise natural answer targets; supervised_examples={supervised_examples:?}"
        );
    }

    #[test]
    fn ruliad_document_range_loss_mask_preserves_answer_schema_across_chunk_boundaries() {
        let document = b"?:q\n!:ok=1\n[/R2]\n"
            .iter()
            .map(|byte| u32::from(*byte))
            .collect::<Vec<_>>();
        let supervision = RuliadSupervisionConfig {
            mode: RuliadSupervisionMode::AnswerCompletion,
            ..Default::default()
        };
        let mut supervised = String::new();
        for start in (0..document.len()).step_by(3) {
            let mut mask = vec![0; 3];
            ruliad_target_loss_mask_for_document_range(
                &document,
                document.len(),
                start,
                3,
                &mut mask,
                supervision,
            );
            let targets = (0..3)
                .filter_map(|offset| document.get(start + offset + 1).copied())
                .collect::<Vec<_>>();
            supervised.push_str(
                &targets
                    .iter()
                    .zip(mask.iter())
                    .filter_map(|(target, mask)| (*mask == 1).then_some(*target as u8 as char))
                    .collect::<String>(),
            );
        }
        assert_eq!(supervised, "ok=1\n[/R2]");
    }

    #[test]
    fn live_ruliad_answer_completion_profile_sized_streaming_masks_natural_answer_chunks() {
        type TestBackend = NdArray<f32>;

        let dir = tempdir().expect("tempdir");
        let config_path = dir.path().join("ruliad-live.toml");
        let mut config = live_ruliad_runtime_config();
        config.serialization.document_tokens = 512;
        config.tokenization = RuliadTokenizationConfig::StructuredSymbolic {
            vocab_size: 272,
            eos_id: Some(271),
        };
        fs::write(&config_path, toml::to_string_pretty(&config).expect("toml"))
            .expect("write config");

        let dataset = UniversalityDataset::new_ruliad_on_the_fly(
            &config_path,
            128,
            32,
            &TokenizerConfig {
                vocab_path: None,
                kind: TokenizerKind::Pretokenized(PretokenizedTokenizerConfig {
                    vocab_size: 272,
                    bos_id: None,
                    eos_id: Some(271),
                    pad_id: None,
                    unk_id: None,
                }),
            },
        )
        .expect("load structured ruliad dataset")
        .with_ruliad_supervision(RuliadSupervisionConfig {
            mode: RuliadSupervisionMode::AnswerCompletion,
            ..Default::default()
        });
        let wrapped = Arc::new(crate::dataset::Dataset::from_universality(dataset));
        let device = burn::tensor::Device::<TestBackend>::default();
        let loader = crate::dataset::StreamingDataLoader::<TestBackend>::new(
            Arc::clone(&wrapped),
            DatasetSplit::Train,
            &device,
            20,
            Some(20),
            Some(512),
            1337,
        );
        let mut context_only_rows = 0usize;
        let mut answer_rows = 0usize;
        for (step, batch) in loader.iter().take(20).enumerate() {
            assert_eq!(
                batch.reset_stream_state,
                step % 4 == 0,
                "step {step} should reset only at logical document boundaries"
            );
            let inputs = batch
                .inputs
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .expect("inputs");
            let targets = batch
                .targets
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .expect("targets");
            let mask = batch
                .loss_mask
                .expect("answer-completion stream loss mask")
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .expect("loss mask");
            assert_eq!(mask.len(), 32 * 128);
            for (row, mask_row) in mask.chunks(128).enumerate() {
                let input_row = &inputs[row * 128..(row + 1) * 128];
                let target_row = &targets[row * 128..(row + 1) * 128];
                let mut window = input_row.to_vec();
                window.push(target_row[127]);
                if mask_row.iter().any(|value| *value == 1) {
                    answer_rows = answer_rows.saturating_add(1);
                    let window_u32 = window.iter().map(|token| *token as u32).collect::<Vec<_>>();
                    let mut expected_mask = vec![0; 128];
                    assert!(
                        ruliad_answer_target_loss_mask(&window_u32, &mut expected_mask),
                        "step {step} row {row} answer mask should match the natural streamed window: {window:?}"
                    );
                } else {
                    context_only_rows = context_only_rows.saturating_add(1);
                }
            }
        }
        assert!(
            context_only_rows > 0,
            "profile-sized answer-completion stream should include unmasked context rows"
        );
        assert!(
            answer_rows > 0,
            "profile-sized answer-completion stream should include natural answer rows"
        );
    }

    #[test]
    fn on_the_fly_ruliad_dataset_exposes_structured_document_end_token_id() {
        let dir = tempdir().expect("tempdir");
        let config_path = dir.path().join("ruliad-structured.toml");
        let mut config = fixed_ruliad_runtime_config();
        config.tokenization = RuliadTokenizationConfig::StructuredSymbolic {
            vocab_size: 272,
            eos_id: Some(271),
        };
        fs::write(&config_path, toml::to_string_pretty(&config).expect("toml"))
            .expect("write config");

        let dataset = UniversalityDataset::new_ruliad_on_the_fly(
            &config_path,
            64,
            2,
            &TokenizerConfig {
                vocab_path: None,
                kind: TokenizerKind::Pretokenized(PretokenizedTokenizerConfig {
                    vocab_size: 272,
                    bos_id: None,
                    eos_id: Some(271),
                    pad_id: None,
                    unk_id: None,
                }),
            },
        )
        .expect("dataset");

        assert_eq!(
            dataset.ruliad_document_end_token_id(),
            Some(RULIAD_SYMBOLIC_DOCUMENT_END_TOKEN)
        );
    }

    #[test]
    fn live_ruliad_source_selection_extends_saturated_frontier() {
        let mut config = live_ruliad_runtime_config();
        config.source_selection.difficulty_levels =
            burn_dragon_universality::UsizeRangeConfig { min: 0, max: 0 };
        config.source_selection.frontier_extension.enabled = true;
        config
            .source_selection
            .frontier_extension
            .levels_per_extension = 2;
        config
            .source_selection
            .frontier_extension
            .extend_when_normalized_difficulty_at_least = 0.0;
        config
            .source_selection
            .frontier_extension
            .extend_when_max_difficulty_probability_at_least = 0.0;
        config
            .source_selection
            .frontier_extension
            .max_materialized_levels = 5;

        let state = LiveSourceSelectionState::new(
            config.source_selection.clone(),
            config.clone(),
            ruliad_sampler_candidates(&config),
        )
        .expect("live source-selection state");

        let snapshot = state.snapshot();
        assert_eq!(snapshot.max_difficulty_level, 2);
        assert_eq!(snapshot.frontier_extension_count, 1);
        assert!(!snapshot.frontier_saturated);

        let saturated = state.snapshot();
        assert_eq!(saturated.max_difficulty_level, 4);
        assert_eq!(saturated.frontier_extension_count, 2);
        assert!(saturated.frontier_saturated);
    }

    #[test]
    fn live_ruliad_source_selection_unbounded_frontier_never_saturates() {
        let mut config = live_ruliad_runtime_config();
        config.source_selection.difficulty_levels =
            burn_dragon_universality::UsizeRangeConfig { min: 0, max: 0 };
        config.source_selection.frontier_extension.enabled = true;
        config
            .source_selection
            .frontier_extension
            .levels_per_extension = 2;
        config
            .source_selection
            .frontier_extension
            .extend_when_normalized_difficulty_at_least = 0.0;
        config
            .source_selection
            .frontier_extension
            .extend_when_max_difficulty_probability_at_least = 0.0;
        config
            .source_selection
            .frontier_extension
            .max_materialized_levels = 0;

        let state = LiveSourceSelectionState::new(
            config.source_selection.clone(),
            config.clone(),
            ruliad_sampler_candidates(&config),
        )
        .expect("live source-selection state");

        let mut last_edge = 0usize;
        for _ in 0..8 {
            let snapshot = state.snapshot();
            assert!(
                snapshot.max_difficulty_level > last_edge,
                "unbounded frontier should keep materializing harder levels"
            );
            assert!(
                !snapshot.frontier_saturated,
                "unbounded frontier must not report saturation"
            );
            last_edge = snapshot.max_difficulty_level;
        }
    }

    #[test]
    fn live_ruliad_source_selection_extends_mastered_frontier_below_normalized_threshold() {
        let mut config = live_ruliad_runtime_config();
        config.source_selection.difficulty_levels =
            burn_dragon_universality::UsizeRangeConfig { min: 0, max: 12 };
        config.source_selection.sampler.mastery_escape_threshold = 0.70;
        config
            .source_selection
            .sampler
            .mastery_min_normalized_difficulty = 0.80;
        config
            .source_selection
            .sampler
            .mastery_min_max_difficulty_probability = 0.35;
        config.source_selection.frontier_extension.enabled = true;
        config
            .source_selection
            .frontier_extension
            .levels_per_extension = 8;
        config
            .source_selection
            .frontier_extension
            .extend_when_normalized_difficulty_at_least = 0.88;
        config
            .source_selection
            .frontier_extension
            .extend_when_max_difficulty_probability_at_least = 0.25;
        config
            .source_selection
            .frontier_extension
            .max_materialized_levels = 0;
        let mut candidates = ruliad_sampler_candidates(&config);
        for candidate in &mut candidates {
            candidate.loss_ema = 0.1;
            candidate.previous_loss_ema = 0.2;
        }
        let pre_extension_snapshot = burn_dragon_universality::RuliadFrontierSampler::new(
            config.source_selection.sampler,
            candidates.clone(),
        )
        .snapshot();
        assert!(
            pre_extension_snapshot.normalized_difficulty_score < 0.88,
            "fixture should exercise mastered-frontier extension below normalized threshold: {}",
            pre_extension_snapshot.normalized_difficulty_score
        );
        assert!(
            pre_extension_snapshot.mastered_probability
                >= config.source_selection.sampler.mastery_escape_threshold
        );
        assert!(
            pre_extension_snapshot.max_difficulty_probability
                >= config
                    .source_selection
                    .frontier_extension
                    .extend_when_max_difficulty_probability_at_least
        );

        let state = LiveSourceSelectionState::new(
            config.source_selection.clone(),
            config.clone(),
            candidates,
        )
        .expect("live source-selection state");

        let snapshot = state.snapshot();
        assert!(
            snapshot.frontier_extension_count > 0,
            "mastered frontier should extend even before normalized pressure crosses threshold"
        );
        assert!(
            snapshot.max_difficulty_level > 12,
            "frontier should materialize harder levels"
        );
        assert!(!snapshot.frontier_saturated);
    }

    #[test]
    fn live_ruliad_source_selection_dynamics_control_caps_hash_noise_and_raises_difficulty() {
        let dir = tempdir().expect("tempdir");
        let config_path = dir.path().join("ruliad-live.toml");
        let mut config = live_ruliad_runtime_config();
        config.families.push(RuliadFamilyConfig {
            kind: RuliadFamilyKind::HashNoise,
            weight: 4,
            width: Some(burn_dragon_universality::UsizeRangeConfig { min: 12, max: 12 }),
            steps: Some(burn_dragon_universality::UsizeRangeConfig { min: 4, max: 4 }),
        });
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

        dataset.apply_source_selection_dynamics_control(3.0, 0.05);
        let after = dataset
            .source_selection_snapshot()
            .expect("controlled snapshot");

        assert!(
            after.hash_noise_probability <= 0.0501,
            "hash-noise probability should respect dynamics cap: {}",
            after.hash_noise_probability
        );
        assert!(
            after.mean_difficulty_level >= before.mean_difficulty_level,
            "difficulty pressure should not lower mean difficulty: before={} after={}",
            before.mean_difficulty_level,
            after.mean_difficulty_level
        );
        assert!(
            after.sampler_entropy_bits.is_finite() && after.sampler_entropy_bits >= 0.0,
            "controlled source probabilities should remain a valid sampler distribution"
        );
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
        {
            let cache = storage.cache.inner.lock().expect("runtime cache lock");
            assert!(
                cache.entries.is_empty(),
                "source-weighted validation must not materialize full epoch caches"
            );
        }
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

    fn contains_period_filler_pattern(tokens: &[u32]) -> bool {
        tokens
            .windows(3)
            .any(|window| window == [u32::from(b'\n'), u32::from(b'.'), u32::from(b'\n')])
    }
}
