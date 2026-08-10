//! Dataset construction, sampling APIs, and token-sequence integration.

use super::*;

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
                live_batch_cache: Arc::new(LiveDocumentBatchCacheState::new()),
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
        Self::new_ruliad_on_the_fly_with_overrides(
            config_path,
            block_size,
            batch_size,
            tokenizer_cfg,
            RuliadSourceSelectionOverrides::default(),
        )
    }

    pub(crate) fn new_ruliad_on_the_fly_with_overrides(
        config_path: impl AsRef<Path>,
        block_size: usize,
        batch_size: usize,
        tokenizer_cfg: &TokenizerConfig,
        overrides: RuliadSourceSelectionOverrides,
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
                let mut source_selection = corpus.config().source_selection.clone();
                if let Some(enabled) = overrides.cold_start_enabled {
                    source_selection.cold_start.enabled = enabled;
                }
                LiveSourceSelectionState::new(
                    source_selection,
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
                live_batch_cache: Arc::new(LiveDocumentBatchCacheState::new()),
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

    pub fn with_source_selection_feedback_updates_enabled(self, enabled: Option<bool>) -> Self {
        if let Some(enabled) = enabled
            && let UniversalityStorage::OnTheFly(storage) = &self.storage
            && let Some(source_selection) = &storage.source_selection
        {
            source_selection
                .feedback_updates_enabled
                .store(enabled, Ordering::Relaxed);
        }
        self
    }

    pub fn source_selection_feedback_updates_enabled(&self) -> Option<bool> {
        match &self.storage {
            UniversalityStorage::Manifest(_) => None,
            UniversalityStorage::OnTheFly(storage) => {
                storage.source_selection.as_ref().map(|source_selection| {
                    source_selection
                        .feedback_updates_enabled
                        .load(Ordering::Relaxed)
                })
            }
        }
    }

    pub fn source_selection_cold_start_enabled(&self) -> Option<bool> {
        match &self.storage {
            UniversalityStorage::Manifest(_) => None,
            UniversalityStorage::OnTheFly(storage) => storage
                .source_selection
                .as_ref()
                .map(|source_selection| source_selection.cold_start.enabled),
        }
    }

    pub(super) fn emits_target_loss_mask(&self) -> bool {
        self.ruliad_supervision.uses_target_loss_mask() || self.tokenizer.eos_id().is_some()
    }

    pub(super) fn fill_target_loss_mask(
        &self,
        window: &[u32],
        mask: &mut [i64],
        supervision: RuliadSupervisionConfig,
    ) -> bool {
        let valid = if supervision.uses_target_loss_mask() {
            ruliad_target_loss_mask(window, mask, supervision)
        } else if window.len() >= mask.len().saturating_add(1) {
            mask.fill(1);
            !mask.is_empty()
        } else {
            mask.fill(0);
            false
        };
        if !valid {
            return false;
        }
        mask_fixed_document_eos_padding(window, mask, self.tokenizer.eos_id())
    }

    pub(super) fn effective_ruliad_supervision(
        &self,
        split: DatasetSplit,
        epoch_index: usize,
        absolute_step: usize,
    ) -> RuliadSupervisionConfig {
        let mut supervision = self.ruliad_supervision;
        if matches!(supervision.mode, RuliadSupervisionMode::Mixed) {
            supervision.mode = if self.ruliad_supervision.prefer_answer_window(
                matches!(split, DatasetSplit::Val),
                epoch_index,
                absolute_step,
            ) {
                RuliadSupervisionMode::AnswerCompletion
            } else {
                RuliadSupervisionMode::FullDocument
            };
        }
        supervision
    }

    pub(super) fn ruliad_answer_completion_active(
        &self,
        split: DatasetSplit,
        epoch_index: usize,
        absolute_step: usize,
    ) -> bool {
        self.effective_ruliad_supervision(split, epoch_index, absolute_step)
            .prefer_answer_window(
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
        crate::dataset::scheduler::sample_batch(self, split, device)
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
        let effective_supervision =
            self.effective_ruliad_supervision(DatasetSplit::Val, epoch_index, absolute_step);
        let answer_completion = effective_supervision.uses_answer_target_mask();
        let mut loss_mask = effective_supervision
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
                    effective_supervision,
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

    pub fn source_selection_snapshot_at_step(
        &self,
        absolute_step: usize,
    ) -> Option<burn_dragon_universality::RuliadMetricSnapshot> {
        match &self.storage {
            UniversalityStorage::Manifest(_) => None,
            UniversalityStorage::OnTheFly(storage) => storage
                .source_selection
                .as_ref()
                .map(|source_selection| source_selection.snapshot_at_step(absolute_step)),
        }
    }

    pub fn record_ruliad_capability_feedback(
        &self,
        report: &burn_dragon_universality::RuliadEvalReport,
    ) -> Option<burn_dragon_universality::RuliadMetricSnapshot> {
        self.record_ruliad_capability_feedback_at_step(report, None)
    }

    pub fn record_ruliad_capability_feedback_at_step(
        &self,
        report: &burn_dragon_universality::RuliadEvalReport,
        absolute_step: Option<usize>,
    ) -> Option<burn_dragon_universality::RuliadMetricSnapshot> {
        match &self.storage {
            UniversalityStorage::Manifest(_) => None,
            UniversalityStorage::OnTheFly(storage) => {
                storage
                    .source_selection
                    .as_ref()
                    .and_then(|source_selection| {
                        source_selection.record_capability_feedback(report, absolute_step)
                    })
            }
        }
    }

    pub(crate) fn record_ruliad_capability_feedback_batch_at_step(
        &self,
        feedback: &[burn_dragon_universality::RuliadCapabilityFeedback],
        absolute_step: Option<usize>,
    ) -> Option<burn_dragon_universality::RuliadMetricSnapshot> {
        match &self.storage {
            UniversalityStorage::Manifest(_) => None,
            UniversalityStorage::OnTheFly(storage) => {
                storage
                    .source_selection
                    .as_ref()
                    .and_then(|source_selection| {
                        source_selection.record_capability_feedback_batch(feedback, absolute_step)
                    })
            }
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
        let mut used_samples = HashSet::<(String, usize)>::with_capacity(max_items);
        let mut bucket_ranks = HashMap::<String, usize>::new();
        for item_rank in 0..max_items {
            let item_step = absolute_step.saturating_add(item_rank);
            let bucket_label = storage
                .source_selection
                .as_ref()
                .and_then(|source_selection| {
                    source_selection.choose_bucket_label_for_validation_step(epoch_index, item_step)
                });
            let bucket_key = bucket_label.clone().unwrap_or_default();
            let bucket_rank = bucket_ranks.entry(bucket_key.clone()).or_default();
            let initial_sample_index = fixed_validation_probe_sample_index(
                sample_count,
                bucket_label.as_deref(),
                *bucket_rank,
            );
            *bucket_rank = bucket_rank.saturating_add(1);
            let sample_index = (0..sample_count)
                .map(|offset| initial_sample_index.wrapping_add(offset) % sample_count)
                .find(|sample_index| used_samples.insert((bucket_key.clone(), *sample_index)));
            let Some(sample_index) = sample_index else {
                continue;
            };
            if let Some(item) = ruliad_validation_probe_item(
                storage,
                RULIAD_VALIDATION_PROBE_PANEL_EPOCH,
                sample_index,
                bucket_label.as_deref(),
                RuliadValidationPromptMode::CanonicalTransfer,
            ) {
                items.push(item);
            }
        }
        items
    }

    pub fn sample_ruliad_training_serialization_probe_items(
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
        let mut used_samples = HashSet::<(String, usize)>::with_capacity(max_items);
        let mut bucket_ranks = HashMap::<String, usize>::new();
        for item_rank in 0..max_items {
            let item_step = absolute_step.saturating_add(item_rank);
            let bucket_label = storage
                .source_selection
                .as_ref()
                .and_then(|source_selection| {
                    source_selection.choose_bucket_label_for_validation_step(epoch_index, item_step)
                });
            let bucket_key = bucket_label.clone().unwrap_or_default();
            let bucket_rank = bucket_ranks.entry(bucket_key.clone()).or_default();
            let initial_sample_index = fixed_validation_probe_sample_index(
                sample_count,
                bucket_label.as_deref(),
                *bucket_rank,
            );
            *bucket_rank = bucket_rank.saturating_add(1);
            let sample_index = (0..sample_count)
                .map(|offset| initial_sample_index.wrapping_add(offset) % sample_count)
                .find(|sample_index| used_samples.insert((bucket_key.clone(), *sample_index)));
            let Some(sample_index) = sample_index else {
                continue;
            };
            if let Some(item) = ruliad_validation_probe_item(
                storage,
                RULIAD_VALIDATION_PROBE_PANEL_EPOCH,
                sample_index,
                bucket_label.as_deref(),
                RuliadValidationPromptMode::TrainingSerialization,
            ) {
                items.push(item);
            }
        }
        items
    }

    /// Sample an immutable validation panel without consulting live curriculum state.
    pub fn sample_ruliad_validation_probe_items_fixed(
        &self,
        panel_seed: u64,
        max_items: usize,
        prompt_mode: RuliadValidationPromptMode,
    ) -> Vec<RuliadValidationProbeItem> {
        let UniversalityStorage::OnTheFly(storage) = &self.storage else {
            return Vec::new();
        };
        if max_items == 0 || storage.corpus.ruliad_config().is_none() {
            return Vec::new();
        }
        let sample_count = storage.corpus.validation_samples().max(1);
        let mut items = Vec::with_capacity(max_items);
        let mut used_samples = HashSet::<usize>::with_capacity(max_items);
        for item_rank in 0..sample_count {
            if items.len() >= max_items {
                break;
            }
            let initial_sample_index =
                fixed_seeded_validation_probe_sample_index(sample_count, panel_seed, item_rank);
            let sample_index = (0..sample_count)
                .map(|offset| initial_sample_index.wrapping_add(offset) % sample_count)
                .find(|sample_index| used_samples.insert(*sample_index));
            let Some(sample_index) = sample_index else {
                continue;
            };
            if let Some(item) = ruliad_validation_probe_item(
                storage,
                RULIAD_VALIDATION_PROBE_PANEL_EPOCH,
                sample_index,
                None,
                prompt_mode,
            ) {
                items.push(item);
            }
        }
        items
    }

    pub fn sample_ruliad_validation_probe_items_stratified(
        &self,
        _epoch_index: usize,
        _absolute_step: usize,
        max_items: usize,
        task_kind: &str,
        difficulty_levels: usize,
    ) -> Vec<RuliadValidationProbeItem> {
        let UniversalityStorage::OnTheFly(storage) = &self.storage else {
            return Vec::new();
        };
        stratified_ruliad_validation_probe_items(
            storage,
            0xB134_4A11_DA7A_5EED,
            max_items,
            Some(task_kind),
            difficulty_levels,
            RuliadValidationPromptMode::CanonicalTransfer,
        )
    }

    /// Seed-stable correctness panel balanced across the lowest materialized
    /// difficulty strata and interleaved across family/task source contracts.
    pub fn sample_ruliad_validation_probe_items_stratified_fixed(
        &self,
        panel_seed: u64,
        max_items: usize,
        difficulty_levels: usize,
        prompt_mode: RuliadValidationPromptMode,
    ) -> Vec<RuliadValidationProbeItem> {
        let UniversalityStorage::OnTheFly(storage) = &self.storage else {
            return Vec::new();
        };
        stratified_ruliad_validation_probe_items(
            storage,
            panel_seed,
            max_items,
            None,
            difficulty_levels,
            prompt_mode,
        )
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

    pub fn encode_ruliad_payload_tokens(&self, text: &str) -> Option<Vec<u32>> {
        let UniversalityStorage::OnTheFly(storage) = &self.storage else {
            return None;
        };
        storage.corpus.encode_ruliad_payload_tokens(text)
    }

    pub fn ruliad_document_end_token_id(&self) -> Option<u32> {
        let UniversalityStorage::OnTheFly(storage) = &self.storage else {
            return None;
        };
        storage.corpus.ruliad_document_end_token_id()
    }
}

fn stratified_ruliad_validation_probe_items(
    storage: &OnTheFlyStorage,
    panel_seed: u64,
    max_items: usize,
    task_kind: Option<&str>,
    difficulty_levels: usize,
    prompt_mode: RuliadValidationPromptMode,
) -> Vec<RuliadValidationProbeItem> {
    if max_items == 0 || difficulty_levels == 0 || storage.corpus.ruliad_config().is_none() {
        return Vec::new();
    }
    let Some(source_selection) = &storage.source_selection else {
        return Vec::new();
    };
    let mut grouped = BTreeMap::<usize, BTreeMap<(String, String), Vec<String>>>::new();
    for candidate in source_selection
        .sampler
        .lock()
        .expect("ruliad source sampler lock poisoned")
        .candidates()
    {
        if task_kind.is_some_and(|task_kind| candidate.task_kind != task_kind) {
            continue;
        }
        grouped
            .entry(candidate.difficulty_level)
            .or_default()
            .entry((candidate.family.clone(), candidate.task_kind.clone()))
            .or_default()
            .push(candidate.oracle_hash.clone());
    }

    let strata = grouped
        .into_iter()
        .take(difficulty_levels)
        .filter_map(|(difficulty_level, groups)| {
            let mut groups = groups.into_values().collect::<Vec<_>>();
            for labels in &mut groups {
                labels.sort();
                labels.dedup();
            }
            let max_group_len = groups.iter().map(Vec::len).max().unwrap_or(0);
            let mut labels = Vec::new();
            for rank in 0..max_group_len {
                labels.extend(groups.iter().filter_map(|group| group.get(rank).cloned()));
            }
            (!labels.is_empty()).then_some((difficulty_level, labels))
        })
        .collect::<Vec<_>>();
    if strata.is_empty() {
        return Vec::new();
    }

    let sample_count = storage.corpus.validation_samples().max(1);
    let mut items = Vec::with_capacity(max_items);
    let mut used_samples = HashSet::<(String, usize)>::with_capacity(max_items);
    for item_rank in 0..max_items {
        let (difficulty_level, labels) = &strata[item_rank % strata.len()];
        let rank_in_stratum = item_rank / strata.len();
        let label_offset = fixed_seeded_validation_probe_sample_index(
            labels.len(),
            panel_seed ^ (*difficulty_level as u64).rotate_left(29),
            0,
        );
        let bucket_label = &labels[(label_offset + rank_in_stratum) % labels.len()];
        let bucket_rank = rank_in_stratum / labels.len();
        let sample_seed = panel_seed ^ source_label_seed(bucket_label).rotate_left(17);
        let initial_sample_index =
            fixed_seeded_validation_probe_sample_index(sample_count, sample_seed, bucket_rank);
        let sample_index = (0..sample_count)
            .map(|offset| initial_sample_index.wrapping_add(offset) % sample_count)
            .find(|sample_index| used_samples.insert((bucket_label.clone(), *sample_index)));
        let Some(sample_index) = sample_index else {
            continue;
        };
        if let Some(item) = ruliad_validation_probe_item(
            storage,
            RULIAD_VALIDATION_PROBE_PANEL_EPOCH,
            sample_index,
            Some(bucket_label),
            prompt_mode,
        ) {
            items.push(item);
        }
    }
    items
}

pub(super) fn ruliad_validation_probe_item(
    storage: &OnTheFlyStorage,
    epoch_index: usize,
    sample_index: usize,
    bucket_label: Option<&str>,
    prompt_mode: RuliadValidationPromptMode,
) -> Option<RuliadValidationProbeItem> {
    let split = burn_dragon_universality::SampleSplit::Validation;
    let item_result = match (prompt_mode, bucket_label) {
        (RuliadValidationPromptMode::CanonicalTransfer, Some(label)) => storage
            .corpus
            .generate_ruliad_eval_item_for_source_bucket(split, epoch_index, sample_index, label),
        (RuliadValidationPromptMode::CanonicalTransfer, None) => storage
            .corpus
            .generate_ruliad_eval_item_for_epoch(split, epoch_index, sample_index),
        (RuliadValidationPromptMode::TrainingSerialization, Some(label)) => storage
            .corpus
            .generate_ruliad_training_serialization_eval_item_for_source_bucket(
                split,
                epoch_index,
                sample_index,
                label,
            ),
        (RuliadValidationPromptMode::TrainingSerialization, None) => storage
            .corpus
            .generate_ruliad_training_serialization_eval_item_for_epoch(
                split,
                epoch_index,
                sample_index,
            ),
    };
    let Ok(Some(item)) = item_result else {
        return None;
    };
    let prompt_tokens = storage.corpus.encode_ruliad_payload_tokens(&item.prompt)?;
    if prompt_tokens.is_empty() {
        return None;
    }
    Some(RuliadValidationProbeItem {
        item,
        prompt_tokens: prompt_tokens.into_iter().map(i64::from).collect(),
    })
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
                .source_selected_token_windows(RuliadWindowRequest {
                    split: burn_dragon_universality::SampleSplit::Train,
                    epoch_index,
                    absolute_step,
                    batch_size,
                    block_size,
                    prefer_answer_window: self.ruliad_answer_completion_active(
                        split,
                        epoch_index,
                        absolute_step,
                    ),
                }),
            (DatasetSplit::Val, UniversalityStorage::OnTheFly(storage)) => storage
                .source_selected_token_windows(RuliadWindowRequest {
                    split: burn_dragon_universality::SampleSplit::Validation,
                    epoch_index,
                    absolute_step,
                    batch_size,
                    block_size,
                    prefer_answer_window: self.ruliad_answer_completion_active(
                        split,
                        epoch_index,
                        absolute_step,
                    ),
                }),
            _ => None,
        }
    }

    fn source_selected_token_windows_with_loss_masks(
        &self,
        split: DatasetSplit,
        epoch_index: usize,
        absolute_step: usize,
        batch_size: usize,
        block_size: usize,
    ) -> Option<SourceSelectedBatch> {
        let supervision = self.effective_ruliad_supervision(split, epoch_index, absolute_step);
        let windows = self.source_selected_token_windows(
            split,
            epoch_index,
            absolute_step,
            batch_size,
            block_size,
        )?;
        let emit_masks = self.emits_target_loss_mask() || supervision.uses_target_loss_mask();
        let masks = emit_masks.then(|| {
            windows
                .iter()
                .map(|window| {
                    let mut mask = vec![0; block_size];
                    self.fill_target_loss_mask(window, &mut mask, supervision);
                    mask
                })
                .collect::<Vec<_>>()
        });
        Some(SourceSelectedBatch {
            windows,
            loss_masks: masks,
        })
    }

    fn fixed_holdout_token_windows_with_loss_masks(
        &self,
        _epoch_index: usize,
        absolute_step: usize,
        batch_size: usize,
        block_size: usize,
    ) -> Option<SourceSelectedBatch> {
        let supervision = self.effective_ruliad_supervision(
            DatasetSplit::Val,
            RULIAD_VALIDATION_PROBE_PANEL_EPOCH,
            absolute_step,
        );
        let UniversalityStorage::OnTheFly(storage) = &self.storage else {
            return None;
        };
        let windows = storage.fixed_validation_token_windows(RuliadWindowRequest {
            split: burn_dragon_universality::SampleSplit::Validation,
            epoch_index: RULIAD_VALIDATION_PROBE_PANEL_EPOCH,
            absolute_step,
            batch_size,
            block_size,
            prefer_answer_window: self.ruliad_answer_completion_active(
                DatasetSplit::Val,
                RULIAD_VALIDATION_PROBE_PANEL_EPOCH,
                absolute_step,
            ),
        })?;
        let emit_masks = self.emits_target_loss_mask() || supervision.uses_target_loss_mask();
        let loss_masks = emit_masks.then(|| {
            windows
                .iter()
                .map(|window| {
                    let mut mask = vec![0; block_size];
                    self.fill_target_loss_mask(window, &mut mask, supervision);
                    mask
                })
                .collect()
        });
        Some(SourceSelectedBatch {
            windows,
            loss_masks,
        })
    }

    fn source_selected_ruliad_policy_batch(
        &self,
        split: DatasetSplit,
        epoch_index: usize,
        absolute_step: usize,
        batch_size: usize,
        stratified_difficulty_levels: usize,
    ) -> Option<RuliadPolicyBatch> {
        match (split, &self.storage) {
            (DatasetSplit::Train, UniversalityStorage::OnTheFly(storage)) => storage
                .source_selected_ruliad_policy_batch(
                    burn_dragon_universality::SampleSplit::Train,
                    epoch_index,
                    absolute_step,
                    batch_size,
                    stratified_difficulty_levels,
                ),
            (DatasetSplit::Val, UniversalityStorage::OnTheFly(storage)) => storage
                .source_selected_ruliad_policy_batch(
                    burn_dragon_universality::SampleSplit::Validation,
                    epoch_index,
                    absolute_step,
                    batch_size,
                    stratified_difficulty_levels,
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
                .source_selected_stream_token_windows(RuliadStreamWindowRequest {
                    window: RuliadWindowRequest {
                        split: burn_dragon_universality::SampleSplit::Train,
                        epoch_index,
                        absolute_step,
                        batch_size,
                        block_size,
                        prefer_answer_window: matches!(
                            self.ruliad_supervision.mode,
                            RuliadSupervisionMode::AnswerWindow
                        ),
                    },
                    chunk_index_in_document,
                }),
            (DatasetSplit::Val, UniversalityStorage::OnTheFly(storage)) => storage
                .source_selected_stream_token_windows(RuliadStreamWindowRequest {
                    window: RuliadWindowRequest {
                        split: burn_dragon_universality::SampleSplit::Validation,
                        epoch_index,
                        absolute_step,
                        batch_size,
                        block_size,
                        prefer_answer_window: matches!(
                            self.ruliad_supervision.mode,
                            RuliadSupervisionMode::AnswerWindow
                        ),
                    },
                    chunk_index_in_document,
                }),
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
    ) -> Option<SourceSelectedStreamBatch> {
        let supervision = self.effective_ruliad_supervision(split, epoch_index, absolute_step);
        let prefer_answer_window = matches!(supervision.mode, RuliadSupervisionMode::AnswerWindow);
        let emit_loss_masks = self.emits_target_loss_mask() || supervision.uses_target_loss_mask();
        match (split, &self.storage) {
            (DatasetSplit::Train, UniversalityStorage::OnTheFly(storage)) => storage
                .source_selected_stream_token_windows_with_loss_masks(
                    RuliadSupervisedStreamRequest {
                        stream: RuliadStreamWindowRequest {
                            window: RuliadWindowRequest {
                                split: burn_dragon_universality::SampleSplit::Train,
                                epoch_index,
                                absolute_step,
                                batch_size,
                                block_size,
                                prefer_answer_window,
                            },
                            chunk_index_in_document,
                        },
                        supervision,
                        emit_loss_masks,
                    },
                ),
            (DatasetSplit::Val, UniversalityStorage::OnTheFly(storage)) => storage
                .source_selected_stream_token_windows_with_loss_masks(
                    RuliadSupervisedStreamRequest {
                        stream: RuliadStreamWindowRequest {
                            window: RuliadWindowRequest {
                                split: burn_dragon_universality::SampleSplit::Validation,
                                epoch_index,
                                absolute_step,
                                batch_size,
                                block_size,
                                prefer_answer_window,
                            },
                            chunk_index_in_document,
                        },
                        supervision,
                        emit_loss_masks,
                    },
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

    fn source_selection_snapshot_at_step(
        &self,
        absolute_step: usize,
    ) -> Option<burn_dragon_universality::RuliadMetricSnapshot> {
        self.source_selection_snapshot_at_step(absolute_step)
    }

    fn uses_target_loss_mask(&self) -> bool {
        self.emits_target_loss_mask()
    }

    fn target_loss_mask_for_window(&self, window: &[u32], mask: &mut [i64]) -> bool {
        self.fill_target_loss_mask(window, mask, self.ruliad_supervision)
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
