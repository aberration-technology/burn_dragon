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

use super::DatasetSplit;
use super::prepared_chunks::{ChunkRuntimeCache, load_cached_chunk_from_mutex, mmap_as_u32_slice};
use super::scheduler::{SequenceBatch, TokenSequenceDataset};
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
    pending: Mutex<HashMap<usize, String>>,
    pending_limit: usize,
    control: Mutex<LiveSourceSelectionControl>,
}

#[derive(Clone, Copy, Debug)]
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
            pending: Mutex::new(HashMap::new()),
            pending_limit: live_source_selection_pending_limit(),
            control: Mutex::new(LiveSourceSelectionControl::default()),
        })
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
        apply_source_selection_control(&mut probabilities, sampler.candidates(), control);
        apply_source_selection_cold_start(
            &mut probabilities,
            sampler.candidates(),
            &self.cold_start,
            absolute_step,
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
        apply_source_selection_control(&mut probabilities, sampler.candidates(), control);
        apply_source_selection_cold_start(
            &mut probabilities,
            sampler.candidates(),
            &self.cold_start,
            absolute_step,
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
        let mut rng = StdRng::seed_from_u64(source_selection_step_seed(
            epoch_index,
            absolute_step,
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
        self.maybe_extend_frontier_locked(&mut sampler);
        Some(self.snapshot_locked_for_step(&sampler, Some(absolute_step)))
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
        let mut snapshot = sampler.snapshot();
        let mut probabilities = sampler.probabilities();
        let control = *self
            .control
            .lock()
            .expect("ruliad source control lock poisoned");
        apply_source_selection_control(&mut probabilities, sampler.candidates(), control);
        apply_source_selection_cold_start(
            &mut probabilities,
            sampler.candidates(),
            &self.cold_start,
            absolute_step,
        );
        snapshot.sampler_entropy_bits = probabilities
            .iter()
            .filter(|probability| **probability > 0.0)
            .map(|probability| -probability * probability.log2())
            .sum();
        snapshot.hash_noise_probability = probabilities
            .iter()
            .zip(sampler.candidates())
            .filter_map(|(probability, candidate)| candidate.is_hash_noise.then_some(*probability))
            .sum();
        let max_difficulty = sampler
            .candidates()
            .iter()
            .map(|candidate| candidate.difficulty_level)
            .max()
            .unwrap_or(0);
        snapshot.mean_difficulty_level = probabilities
            .iter()
            .zip(sampler.candidates())
            .map(|(probability, candidate)| *probability * candidate.difficulty_level as f32)
            .sum();
        snapshot.normalized_difficulty_score = if max_difficulty == 0 {
            0.0
        } else {
            snapshot.mean_difficulty_level / max_difficulty as f32
        };
        snapshot.max_difficulty_probability = probabilities
            .iter()
            .zip(sampler.candidates())
            .filter_map(|(probability, candidate)| {
                (candidate.difficulty_level == max_difficulty).then_some(*probability)
            })
            .sum();
        snapshot.mastered_probability = probabilities
            .iter()
            .zip(sampler.candidates())
            .filter_map(|(probability, candidate)| {
                (candidate.loss_ema <= snapshot.target_loss).then_some(*probability)
            })
            .sum();
        snapshot.max_difficulty_level = max_difficulty;
        snapshot.frontier_extension_count = self.frontier_extension_count.load(Ordering::Relaxed);
        snapshot.frontier_saturated = self.frontier_saturated(&snapshot);
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
        if snapshot.normalized_difficulty_score
            < self
                .frontier_extension
                .extend_when_normalized_difficulty_at_least
            || snapshot.max_difficulty_probability
                < self
                    .frontier_extension
                    .extend_when_max_difficulty_probability_at_least
            || self.frontier_saturated(&snapshot)
        {
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

    fn frontier_saturated(
        &self,
        snapshot: &burn_dragon_universality::RuliadMetricSnapshot,
    ) -> bool {
        let pressure_at_ceiling = snapshot.normalized_difficulty_score
            >= self
                .frontier_extension
                .extend_when_normalized_difficulty_at_least
            && snapshot.max_difficulty_probability
                >= self
                    .frontier_extension
                    .extend_when_max_difficulty_probability_at_least;
        if !pressure_at_ceiling {
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
            let start =
                semantic_window_start(document, usable_len, self.block_size, start, &mut rng);
            for token_index in 0..self.block_size {
                let offset = batch_idx * self.block_size + token_index;
                inputs[offset] = document[start + token_index] as i64;
                targets[offset] = document[start + token_index + 1] as i64;
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
                ),
            (DatasetSplit::Val, UniversalityStorage::OnTheFly(storage)) => storage
                .source_selected_token_windows(
                    burn_dragon_universality::SampleSplit::Validation,
                    epoch_index,
                    absolute_step,
                    batch_size,
                    block_size,
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
        ))
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

fn source_selected_windows_from_documents(
    documents: &[Arc<Vec<u32>>],
    eos_id: Option<u32>,
    epoch_index: usize,
    absolute_step: usize,
    bucket_label: &str,
    batch_size: usize,
    block_size: usize,
) -> Vec<Vec<u32>> {
    if documents.is_empty() {
        return Vec::new();
    }
    let document_count = documents.len();
    (0..batch_size)
        .map(|batch_index| {
            let document = documents
                .get(batch_index % document_count)
                .expect("source-selected document set must be non-empty");
            let mut rng = StdRng::seed_from_u64(source_selection_step_seed(
                epoch_index,
                absolute_step,
                source_label_seed(bucket_label) as usize ^ batch_index.rotate_left(11),
            ));
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
            let start = semantic_window_start(document, usable_len, block_size, start, &mut rng);
            document[start..start + block_size + 1].to_vec()
        })
        .collect()
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

fn semantic_window_start<R: Rng + ?Sized>(
    document: &[u32],
    usable_len: usize,
    block_size: usize,
    fallback_start: usize,
    rng: &mut R,
) -> usize {
    let max_start = usable_len.saturating_sub(block_size + 1);
    if max_start == 0 {
        return 0;
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
    use crate::tokenizer::{PretokenizedTokenizerConfig, TokenizerConfig};
    use burn_dragon_universality::config::NcaCorpusConfig;
    use burn_dragon_universality::{
        NcaSerializationConfig, NcaTokenizationConfig, RuliadCorpusConfig, RuliadDocumentMode,
        RuliadFamilyConfig, RuliadFamilyKind, RuliadSerializationConfig, RuliadTokenizationConfig,
        generate_nca_corpus, ruliad_sampler_candidates,
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
    fn source_selected_window_sampler_includes_document_end_windows() {
        let mut document = vec![777u32; 512];
        document[511] = 50_256;
        let usable_len = valid_document_token_count(&document, Some(50_256));
        let block_size = 64;
        let max_start = usable_len.saturating_sub(block_size + 1);
        let mut rng = StdRng::seed_from_u64(1337);
        let mut end_count = 0usize;
        for _ in 0..128 {
            let start = semantic_window_start(&document, usable_len, block_size, 0, &mut rng);
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
