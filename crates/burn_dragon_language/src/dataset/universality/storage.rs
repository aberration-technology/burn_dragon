//! On-the-fly corpus generation and bounded runtime caches.

use super::*;

impl OnTheFlyStorage {
    pub(super) fn copy_into(&self, start: usize, train_len: usize, dst: &mut [u32]) {
        self.copy_into_with_epoch(DatasetSplit::Train, 0, start, train_len, dst);
    }

    pub(super) fn copy_into_with_epoch(
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

    pub(super) fn document_tokens(
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

    pub(super) fn source_selected_document_indices(
        &self,
        split: burn_dragon_universality::SampleSplit,
        epoch_index: usize,
        absolute_step: usize,
        batch_size: usize,
    ) -> Option<Vec<usize>> {
        self.source_selection.as_ref()?;
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

    pub(super) fn source_selected_token_windows(
        &self,
        request: RuliadWindowRequest,
    ) -> Option<Vec<Vec<u32>>> {
        let RuliadWindowRequest {
            split,
            epoch_index,
            absolute_step,
            batch_size,
            ..
        } = request;
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
            &bucket_label,
            request,
        ))
    }

    pub(super) fn fixed_validation_token_windows(
        &self,
        mut request: RuliadWindowRequest,
    ) -> Option<Vec<Vec<u32>>> {
        let source_selection = self.source_selection.as_ref()?;
        let bucket_label = source_selection
            .choose_bucket_label_for_fixed_validation_step(request.absolute_step)?;
        request.split = burn_dragon_universality::SampleSplit::Validation;
        request.epoch_index = 0;
        let document_count = live_source_selection_documents_per_step(request.batch_size);
        let documents = self.generate_source_bucket_documents(
            burn_dragon_universality::SampleSplit::Validation,
            0,
            request.absolute_step,
            &bucket_label,
            document_count,
        );
        Some(source_selected_windows_from_documents(
            &documents,
            self.corpus.eos_id(),
            &bucket_label,
            request,
        ))
    }

    pub(super) fn source_selected_ruliad_policy_batch(
        &self,
        split: burn_dragon_universality::SampleSplit,
        epoch_index: usize,
        absolute_step: usize,
        batch_size: usize,
        stratified_difficulty_levels: usize,
    ) -> Option<RuliadPolicyBatch> {
        let source_selection = self.source_selection.as_ref()?;
        let tokenization = self.corpus.ruliad_config()?.tokenization.clone();
        let selected_bucket_label = match split {
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
        let stratified_bucket_labels = (stratified_difficulty_levels > 0)
            .then(|| {
                let mut labels = source_selection
                    .sampler
                    .lock()
                    .expect("ruliad source sampler lock poisoned")
                    .candidates()
                    .iter()
                    .filter(|candidate| {
                        candidate.task_kind == "select_proof_action"
                            && candidate.difficulty_level < stratified_difficulty_levels
                    })
                    .map(|candidate| (candidate.difficulty_level, candidate.oracle_hash.clone()))
                    .collect::<Vec<_>>();
                labels.sort();
                labels.dedup();
                labels
                    .into_iter()
                    .map(|(_, label)| label)
                    .collect::<Vec<_>>()
            })
            .filter(|labels| !labels.is_empty())
            .unwrap_or_default();
        let mut samples = Vec::with_capacity(batch_size.max(1));
        for sample_rank in 0..batch_size.max(1) {
            let bucket_label = stratified_bucket_labels
                .get(sample_rank % stratified_bucket_labels.len().max(1))
                .unwrap_or(&selected_bucket_label);
            let sample_index = live_source_selection_sample_index(
                sample_count,
                split,
                epoch_index,
                absolute_step,
                bucket_label,
                sample_rank,
            );
            let item = self
                .corpus
                .generate_ruliad_eval_item_for_source_bucket(
                    split,
                    epoch_index,
                    sample_index,
                    bucket_label,
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

    pub(super) fn source_selected_stream_token_windows(
        &self,
        request: RuliadStreamWindowRequest,
    ) -> Option<Vec<Vec<u32>>> {
        let RuliadStreamWindowRequest {
            window:
                RuliadWindowRequest {
                    split,
                    epoch_index,
                    absolute_step,
                    batch_size,
                    ..
                },
            chunk_index_in_document,
        } = request;
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
            request,
        ))
    }

    pub(super) fn source_selected_stream_token_windows_with_loss_masks(
        &self,
        request: RuliadSupervisedStreamRequest,
    ) -> Option<SourceSelectedStreamBatch> {
        let RuliadSupervisedStreamRequest {
            stream,
            supervision,
            emit_loss_masks,
        } = request;
        let RuliadStreamWindowRequest {
            window:
                RuliadWindowRequest {
                    split,
                    epoch_index,
                    absolute_step,
                    batch_size,
                    block_size,
                    prefer_answer_window,
                },
            chunk_index_in_document,
        } = stream;
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
        let windows =
            source_selected_stream_windows_from_documents(&documents, self.corpus.eos_id(), stream);
        let masks = emit_loss_masks.then(|| {
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
        let document_complete = prefer_answer_window
            || source_selected_stream_document_complete(
                &documents,
                self.corpus.eos_id(),
                block_size,
                chunk_index_in_document,
            );
        Some(SourceSelectedStreamBatch {
            windows,
            loss_masks: masks,
            document_complete,
        })
    }

    pub(super) fn source_weighted_validation_documents(
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

    pub(super) fn prepare_epoch(
        &self,
        split: burn_dragon_universality::SampleSplit,
        epoch_index: usize,
    ) {
        if self.source_selection.is_some() {
            return;
        }
        let _ = self.epoch_documents(split, epoch_index);
    }

    pub(super) fn prefetch_epoch(
        &self,
        split: burn_dragon_universality::SampleSplit,
        epoch_index: usize,
    ) {
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

    pub(super) fn epoch_documents(
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

    pub(super) fn generate_source_bucket_documents(
        &self,
        split: burn_dragon_universality::SampleSplit,
        epoch_index: usize,
        absolute_step: usize,
        bucket_label: &str,
        document_count: usize,
    ) -> Vec<Arc<Vec<u32>>> {
        let document_count = document_count.max(1);
        let key = LiveDocumentBatchKey {
            split_tag: split_tag(split),
            epoch_index,
            selection_step: absolute_step,
            bucket_label: bucket_label.to_string(),
            document_count,
        };
        loop {
            let mut cache = self
                .live_batch_cache
                .inner
                .lock()
                .expect("live ruliad document cache poisoned");
            cache.tick = cache.tick.wrapping_add(1);
            let tick = cache.tick;
            if let Some(entry) = cache.entries.get_mut(&key) {
                entry.last_used_tick = tick;
                return entry.documents.clone();
            }
            if cache.building.insert(key.clone()) {
                break;
            }
            let _unused = self
                .live_batch_cache
                .ready
                .wait(cache)
                .expect("live ruliad document cache poisoned");
        }

        let generated = catch_unwind(AssertUnwindSafe(|| {
            self.build_source_bucket_documents(
                split,
                epoch_index,
                absolute_step,
                bucket_label,
                document_count,
            )
        }));
        match generated {
            Ok(documents) => {
                let bytes = documents.iter().fold(0usize, |total, document| {
                    total.saturating_add(document.len().saturating_mul(size_of::<u32>()))
                });
                let mut cache = self
                    .live_batch_cache
                    .inner
                    .lock()
                    .expect("live ruliad document cache poisoned");
                cache.tick = cache.tick.wrapping_add(1);
                let tick = cache.tick;
                cache.building.remove(&key);
                if let Some(previous) = cache.entries.insert(
                    key,
                    CachedLiveDocumentBatch {
                        documents: documents.clone(),
                        bytes,
                        last_used_tick: tick,
                    },
                ) {
                    cache.total_bytes = cache.total_bytes.saturating_sub(previous.bytes);
                }
                cache.total_bytes = cache.total_bytes.saturating_add(bytes);
                while cache.entries.len() > self.live_batch_cache.entry_limit
                    || (cache.total_bytes > self.live_batch_cache.byte_limit
                        && cache.entries.len() > 1)
                {
                    let evict_key = cache
                        .entries
                        .iter()
                        .min_by_key(|(_, entry)| entry.last_used_tick)
                        .map(|(key, _)| key.clone())
                        .expect("live ruliad document cache should not be empty");
                    if let Some(removed) = cache.entries.remove(&evict_key) {
                        cache.total_bytes = cache.total_bytes.saturating_sub(removed.bytes);
                    }
                }
                self.live_batch_cache.ready.notify_all();
                documents
            }
            Err(payload) => {
                let mut cache = self
                    .live_batch_cache
                    .inner
                    .lock()
                    .expect("live ruliad document cache poisoned");
                cache.building.remove(&key);
                self.live_batch_cache.ready.notify_all();
                drop(cache);
                resume_unwind(payload);
            }
        }
    }

    pub(super) fn build_source_bucket_documents(
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
        (0..document_count)
            .into_par_iter()
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
                        .generate_compact_document_tokens_for_source_bucket(
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

    pub(super) fn build_and_store_epoch(
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

    pub(super) fn store_generated_epoch(
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

    pub(super) fn clear_building_epoch(&self, key: RuntimeEpochKey) {
        let mut cache = self
            .cache
            .inner
            .lock()
            .expect("universality runtime cache poisoned");
        cache.building.remove(&key);
        self.cache.ready.notify_all();
    }

    pub(super) fn generate_epoch_documents(
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
                    let tokens = match bucket_label.as_deref() {
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
                        };
                    assert_eq!(
                        tokens.len(),
                        corpus.document_token_count(),
                        "fixed-envelope universality document length drifted: split={split:?} epoch_index={epoch_index} sample_index={sample_index} bucket={bucket_label:?}"
                    );
                    let tokens = Arc::new(tokens);
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
