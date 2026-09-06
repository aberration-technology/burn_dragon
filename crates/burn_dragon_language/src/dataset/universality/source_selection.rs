//! Live source policy state, capability feedback, and cold-start control.

use super::*;

fn fixed_validation_bucket_labels(
    candidates: &[burn_dragon_universality::RuliadSamplerCandidate],
) -> Vec<String> {
    let mut labels = candidates
        .iter()
        .map(|candidate| candidate.oracle_hash.clone())
        .collect::<Vec<_>>();
    labels.sort();
    labels.dedup();
    labels
}

impl LiveSourceSelectionState {
    pub(super) fn new(
        source_selection: burn_dragon_universality::RuliadSourceSelectionConfig,
        corpus_config: burn_dragon_universality::RuliadCorpusConfig,
        candidates: Vec<burn_dragon_universality::RuliadSamplerCandidate>,
    ) -> Option<Self> {
        if candidates.is_empty() {
            return None;
        }
        let fixed_validation_bucket_labels = fixed_validation_bucket_labels(&candidates);
        let released_max_difficulty_level =
            initial_released_max_difficulty_level(&source_selection.cold_start, &candidates);
        Some(Self {
            sampler: Mutex::new(burn_dragon_universality::RuliadFrontierSampler::new(
                source_selection.sampler,
                candidates,
            )),
            fixed_validation_bucket_labels,
            proof_policy_strata: Mutex::new(BTreeMap::new()),
            corpus_config,
            frontier_extension: source_selection.frontier_extension,
            cold_start: source_selection.cold_start,
            feedback_updates_enabled: AtomicBool::new(source_selection.feedback_updates_enabled),
            frontier_extension_count: AtomicUsize::new(0),
            released_max_difficulty_level: AtomicUsize::new(released_max_difficulty_level),
            run_step_origin: 0,
            pending: Mutex::new(HashMap::new()),
            pending_limit: live_source_selection_pending_limit(),
            consolidation_bucket_catalog: Mutex::new(BTreeMap::new()),
            control: Mutex::new(LiveSourceSelectionControl::default()),
        })
    }

    pub(super) fn from_snapshot(
        source_selection: burn_dragon_universality::RuliadSourceSelectionConfig,
        corpus_config: burn_dragon_universality::RuliadCorpusConfig,
        configured_candidates: Vec<burn_dragon_universality::RuliadSamplerCandidate>,
        snapshot: RuliadSourceSelectionStateSnapshot,
        restore: RuliadSourceSelectionRestore,
    ) -> Option<Self> {
        if snapshot.version != RULIAD_SOURCE_SELECTION_STATE_VERSION {
            return None;
        }
        let fixed_validation_bucket_labels = fixed_validation_bucket_labels(&configured_candidates);
        let mut sampler = burn_dragon_universality::RuliadFrontierSampler::from_state(
            source_selection.sampler,
            snapshot.sampler,
        );
        let restored_max_difficulty = sampler.max_difficulty_level();
        let mut current_candidates = configured_candidates;
        let configured_min_difficulty = corpus_config.source_selection.difficulty_levels.min;
        if restored_max_difficulty >= configured_min_difficulty {
            for difficulty_level in configured_min_difficulty..=restored_max_difficulty {
                current_candidates.extend(
                    burn_dragon_universality::ruliad_sampler_candidates_for_difficulty(
                        &corpus_config,
                        difficulty_level,
                    ),
                );
            }
        }
        sampler.add_candidates(current_candidates);
        if sampler.candidates().is_empty() {
            return None;
        }
        let inferred_released_max = current_cold_start_max_difficulty(
            sampler.candidates(),
            &source_selection.cold_start,
            Some(snapshot.clock.next_global_step().saturating_sub(1)),
        )
        .unwrap_or_else(|| sampler.max_difficulty_level());
        let released_max_difficulty_level = snapshot
            .released_max_difficulty_level
            .max(inferred_released_max)
            .max(initial_released_max_difficulty_level(
                &source_selection.cold_start,
                sampler.candidates(),
            ));
        Some(Self {
            sampler: Mutex::new(sampler),
            fixed_validation_bucket_labels,
            proof_policy_strata: Mutex::new(BTreeMap::new()),
            corpus_config,
            frontier_extension: source_selection.frontier_extension,
            cold_start: source_selection.cold_start,
            feedback_updates_enabled: AtomicBool::new(source_selection.feedback_updates_enabled),
            frontier_extension_count: AtomicUsize::new(snapshot.frontier_extension_count),
            released_max_difficulty_level: AtomicUsize::new(released_max_difficulty_level),
            run_step_origin: match restore {
                RuliadSourceSelectionRestore::ResumeRun => snapshot.clock.run_step_origin,
                RuliadSourceSelectionRestore::StartNewRun => snapshot.clock.next_global_step(),
            },
            pending: Mutex::new(HashMap::new()),
            pending_limit: live_source_selection_pending_limit(),
            consolidation_bucket_catalog: Mutex::new(snapshot.consolidation_bucket_catalog),
            control: Mutex::new(snapshot.control.into()),
        })
    }

    pub(super) fn export_state(
        &self,
        completed_run_steps: usize,
    ) -> RuliadSourceSelectionStateSnapshot {
        let sampler = self
            .sampler
            .lock()
            .expect("ruliad source sampler lock poisoned");
        let control = *self
            .control
            .lock()
            .expect("ruliad source control lock poisoned");
        let consolidation_bucket_catalog = self
            .consolidation_bucket_catalog
            .lock()
            .expect("ruliad consolidation bucket catalog lock poisoned")
            .clone();
        RuliadSourceSelectionStateSnapshot {
            version: RULIAD_SOURCE_SELECTION_STATE_VERSION,
            clock: RuliadSourceSelectionClock {
                run_step_origin: self.run_step_origin,
                completed_run_steps,
            },
            frontier_extension_count: self.frontier_extension_count.load(Ordering::Relaxed),
            released_max_difficulty_level: self
                .released_max_difficulty_level
                .load(Ordering::Relaxed),
            control: control.into(),
            sampler: sampler.export_state(),
            consolidation_bucket_catalog,
        }
    }

    pub(super) fn effective_absolute_step(&self, absolute_step: Option<usize>) -> Option<usize> {
        absolute_step.map(|step| step.saturating_add(self.run_step_origin))
    }

    pub(super) fn probabilities(&self) -> Vec<f32> {
        self.probabilities_for_step(None)
    }

    /// Return the deterministic select-proof-action labels for the lowest
    /// requested difficulty strata. This catalog is immutable for a corpus
    /// config, so cache it outside the per-step batch-construction path.
    pub(super) fn proof_policy_stratified_bucket_labels(
        &self,
        difficulty_levels: usize,
    ) -> Arc<Vec<Vec<String>>> {
        let mut cache = self
            .proof_policy_strata
            .lock()
            .expect("ruliad proof-policy stratum cache lock poisoned");
        cache
            .entry(difficulty_levels)
            .or_insert_with(|| {
                let first_difficulty = self.corpus_config.source_selection.difficulty_levels.min;
                Arc::new(
                    (first_difficulty..first_difficulty.saturating_add(difficulty_levels))
                        .filter_map(|difficulty_level| {
                            let mut labels =
                                burn_dragon_universality::ruliad_sampler_candidates_for_difficulty(
                                    &self.corpus_config,
                                    difficulty_level,
                                )
                                .into_iter()
                                .filter(|candidate| candidate.task_kind == "select_proof_action")
                                .map(|candidate| candidate.oracle_hash)
                                .collect::<Vec<_>>();
                            labels.sort();
                            labels.dedup();
                            (!labels.is_empty()).then_some(labels)
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .clone()
    }

    pub(super) fn probabilities_for_step(&self, absolute_step: Option<usize>) -> Vec<f32> {
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
        let released_max_difficulty =
            self.released_cold_start_max_difficulty(sampler.candidates(), effective_step);
        apply_source_selection_cold_start_with_max(
            &mut probabilities,
            sampler.candidates(),
            &self.cold_start,
            released_max_difficulty,
        );
        sampler.apply_probability_constraints(&mut probabilities);
        apply_source_selection_control(&mut probabilities, sampler.candidates(), control);
        probabilities
    }

    pub(super) fn weighted_bucket_labels(
        &self,
        absolute_step: Option<usize>,
    ) -> Vec<(String, f32)> {
        self.weighted_bucket_labels_at_global_step(self.effective_absolute_step(absolute_step))
    }

    fn weighted_bucket_labels_at_global_step(
        &self,
        global_step: Option<usize>,
    ) -> Vec<(String, f32)> {
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
        let released_max_difficulty =
            self.released_cold_start_max_difficulty(sampler.candidates(), global_step);
        apply_source_selection_cold_start_with_max(
            &mut probabilities,
            sampler.candidates(),
            &self.cold_start,
            released_max_difficulty,
        );
        sampler.apply_probability_constraints(&mut probabilities);
        apply_source_selection_control(&mut probabilities, sampler.candidates(), control);
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
                        .unwrap_or(0.0),
                )
            })
            .collect()
    }

    pub(super) fn apply_dynamics_control(
        &self,
        difficulty_pressure: f32,
        hash_noise_max_probability: f32,
    ) {
        if !self.feedback_updates_enabled.load(Ordering::Relaxed) {
            return;
        }
        let mut control = self
            .control
            .lock()
            .expect("ruliad source control lock poisoned");
        control.difficulty_pressure = difficulty_pressure.max(0.0);
        control.hash_noise_max_probability = hash_noise_max_probability.clamp(0.0, 1.0);
    }

    fn released_cold_start_max_difficulty(
        &self,
        candidates: &[burn_dragon_universality::RuliadSamplerCandidate],
        absolute_step: Option<usize>,
    ) -> Option<usize> {
        if !self.cold_start.enabled || absolute_step.is_none() {
            return None;
        }
        let observed =
            current_cold_start_max_difficulty(candidates, &self.cold_start, absolute_step)
                .unwrap_or_else(|| {
                    candidates
                        .iter()
                        .map(|candidate| candidate.difficulty_level)
                        .max()
                        .unwrap_or(0)
                });
        if !self.cold_start.monotonic_mastery_release {
            return Some(observed);
        }
        let previous = self
            .released_max_difficulty_level
            .fetch_max(observed, Ordering::Relaxed);
        Some(previous.max(observed))
    }

    pub(super) fn choose_bucket_for_step(
        &self,
        available: &HashMap<String, Vec<usize>>,
        epoch_index: usize,
        absolute_step: usize,
    ) -> Option<String> {
        self.choose_bucket_for_step_inner(available, epoch_index, absolute_step, true)
    }

    pub(super) fn choose_bucket_label_for_step(
        &self,
        epoch_index: usize,
        absolute_step: usize,
    ) -> Option<String> {
        self.choose_bucket_label_for_step_inner(epoch_index, absolute_step, true)
    }

    pub(super) fn choose_bucket_label_for_validation_step(
        &self,
        epoch_index: usize,
        absolute_step: usize,
    ) -> Option<String> {
        self.choose_bucket_label_for_step_inner(epoch_index, absolute_step, false)
    }

    pub(super) fn choose_bucket_label_for_fixed_validation_step(
        &self,
        absolute_step: usize,
    ) -> Option<String> {
        let labels = &self.fixed_validation_bucket_labels;
        labels.get(absolute_step % labels.len().max(1)).cloned()
    }

    pub(super) fn choose_bucket_label_for_stream_step(
        &self,
        epoch_index: usize,
        selection_step: usize,
        feedback_step: usize,
    ) -> Option<String> {
        let label = self.choose_bucket_label_for_step_inner(epoch_index, selection_step, false)?;
        self.record_pending(feedback_step, &label);
        Some(label)
    }

    /// Selects a source once for a released generation coordinate and reuses
    /// that assignment on every later replay. The catalog stores labels only;
    /// documents remain generated on demand and bounded by the normal cache.
    pub(super) fn choose_consolidated_bucket_label(
        &self,
        epoch_index: usize,
        generation_step: usize,
        feedback_step: usize,
    ) -> Option<String> {
        let mut catalog = self
            .consolidation_bucket_catalog
            .lock()
            .expect("ruliad consolidation bucket catalog lock poisoned");
        let label = if let Some(label) = catalog.get(&generation_step) {
            label.clone()
        } else {
            let curriculum_step = self.effective_absolute_step(Some(feedback_step))?;
            let label = self.choose_bucket_label_at_global_coordinate(
                epoch_index,
                generation_step,
                curriculum_step,
            )?;
            catalog.insert(generation_step, label.clone());
            label
        };
        drop(catalog);
        self.record_pending(feedback_step, &label);
        Some(label)
    }

    pub(super) fn choose_bucket_label_for_step_inner(
        &self,
        epoch_index: usize,
        absolute_step: usize,
        record_pending: bool,
    ) -> Option<String> {
        let global_step = self.effective_absolute_step(Some(absolute_step))?;
        let label =
            self.choose_bucket_label_at_global_coordinate(epoch_index, global_step, global_step)?;
        if record_pending {
            self.record_pending(absolute_step, &label);
        }
        Some(label)
    }

    fn choose_bucket_label_at_global_coordinate(
        &self,
        epoch_index: usize,
        generation_step: usize,
        curriculum_step: usize,
    ) -> Option<String> {
        Self::sample_weighted_label(
            self.weighted_bucket_labels_at_global_step(Some(curriculum_step)),
            epoch_index,
            generation_step,
        )
    }

    fn sample_weighted_label(
        mut weighted: Vec<(String, f32)>,
        epoch_index: usize,
        global_generation_step: usize,
    ) -> Option<String> {
        use rand::distributions::{Distribution, WeightedIndex};

        let distribution = WeightedIndex::new(weighted.iter().map(|(_, weight)| *weight)).ok()?;
        let mut rng = StdRng::seed_from_u64(source_selection_step_seed(
            epoch_index,
            global_generation_step,
            weighted.len() as u64,
        ));
        Some(weighted.swap_remove(distribution.sample(&mut rng)).0)
    }

    pub(super) fn choose_bucket_for_step_inner(
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
        let effective_step = self
            .effective_absolute_step(Some(absolute_step))
            .unwrap_or(absolute_step);
        let label = Self::sample_weighted_label(filtered, epoch_index, effective_step)?;
        if record_pending {
            self.record_pending(absolute_step, &label);
        }
        Some(label)
    }

    pub(super) fn record_pending(&self, absolute_step: usize, bucket_label: &str) {
        if !self.feedback_updates_enabled.load(Ordering::Relaxed) {
            return;
        }
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

    pub(super) fn record_loss(
        &self,
        absolute_step: usize,
        loss: f32,
    ) -> Option<burn_dragon_universality::RuliadMetricSnapshot> {
        if !self.feedback_updates_enabled.load(Ordering::Relaxed) {
            return Some(self.snapshot_at_step(absolute_step));
        }
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

    pub(super) fn record_capability_feedback(
        &self,
        report: &burn_dragon_universality::RuliadEvalReport,
        absolute_step: Option<usize>,
    ) -> Option<burn_dragon_universality::RuliadMetricSnapshot> {
        self.record_capability_feedback_batch(
            &ruliad_capability_feedback_from_report(report),
            absolute_step,
        )
    }

    pub(super) fn record_capability_feedback_batch(
        &self,
        feedback: &[burn_dragon_universality::RuliadCapabilityFeedback],
        absolute_step: Option<usize>,
    ) -> Option<burn_dragon_universality::RuliadMetricSnapshot> {
        if !self.feedback_updates_enabled.load(Ordering::Relaxed) {
            return Some(self.snapshot_for_step(absolute_step));
        }
        let mut sampler = self
            .sampler
            .lock()
            .expect("ruliad source sampler lock poisoned");
        for feedback in feedback {
            sampler.record_capability_feedback(feedback);
        }
        self.maybe_extend_frontier_locked(&mut sampler);
        Some(self.snapshot_locked_for_step(&sampler, absolute_step))
    }

    pub(super) fn snapshot(&self) -> burn_dragon_universality::RuliadMetricSnapshot {
        self.snapshot_for_step(None)
    }

    pub(super) fn snapshot_at_step(
        &self,
        absolute_step: usize,
    ) -> burn_dragon_universality::RuliadMetricSnapshot {
        self.snapshot_for_step(Some(absolute_step))
    }

    fn snapshot_for_step(
        &self,
        absolute_step: Option<usize>,
    ) -> burn_dragon_universality::RuliadMetricSnapshot {
        let mut sampler = self
            .sampler
            .lock()
            .expect("ruliad source sampler lock poisoned");
        self.maybe_extend_frontier_locked(&mut sampler);
        self.snapshot_locked_for_step(&sampler, absolute_step)
    }

    pub(super) fn snapshot_locked(
        &self,
        sampler: &burn_dragon_universality::RuliadFrontierSampler,
    ) -> burn_dragon_universality::RuliadMetricSnapshot {
        self.snapshot_locked_for_step(sampler, None)
    }

    pub(super) fn snapshot_locked_for_step(
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
        let released_max_difficulty =
            self.released_cold_start_max_difficulty(sampler.candidates(), effective_step);
        apply_source_selection_cold_start_with_max(
            &mut probabilities,
            sampler.candidates(),
            &self.cold_start,
            released_max_difficulty,
        );
        sampler.apply_probability_constraints(&mut probabilities);
        apply_source_selection_control(&mut probabilities, sampler.candidates(), control);
        let mut snapshot = sampler.snapshot_with_probabilities(&probabilities);
        snapshot.curriculum_released_max_difficulty_level =
            self.released_max_difficulty_level.load(Ordering::Relaxed);
        snapshot.frontier_extension_count = self.frontier_extension_count.load(Ordering::Relaxed);
        snapshot.frontier_saturated = self.frontier_saturated(&snapshot);
        snapshot.frontier_unbounded =
            self.frontier_extension.enabled && self.frontier_extension.max_materialized_levels == 0;
        snapshot
    }

    pub(super) fn maybe_extend_frontier_locked(
        &self,
        sampler: &mut burn_dragon_universality::RuliadFrontierSampler,
    ) {
        if !self.feedback_updates_enabled.load(Ordering::Relaxed)
            || !self.frontier_extension.enabled
        {
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

    pub(super) fn frontier_extension_pressure(
        &self,
        snapshot: &burn_dragon_universality::RuliadMetricSnapshot,
    ) -> bool {
        self.frontier_extension.enabled && self.frontier_pressure_at_configured_edge(snapshot)
    }

    pub(super) fn frontier_pressure_at_configured_edge(
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

    pub(super) fn frontier_saturated(
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

pub(crate) fn ruliad_capability_feedback_from_report(
    report: &burn_dragon_universality::RuliadEvalReport,
) -> Vec<burn_dragon_universality::RuliadCapabilityFeedback> {
    if !report.source_scores.is_empty() {
        return report
            .source_scores
            .iter()
            .map(|group| ruliad_capability_feedback_from_group(group.label.clone(), group))
            .collect();
    }

    // Reports before v11 did not carry joint source keys. Preserve the marginal
    // fallback for checkpoint/report compatibility, but new reports must take the
    // source-key path above to avoid duplicate and cross-difficulty updates.
    let mut feedback = Vec::new();
    extend_ruliad_capability_feedback(&mut feedback, "difficulty", &report.difficulty_scores);
    extend_ruliad_capability_feedback(&mut feedback, "family", &report.family_scores);
    extend_ruliad_capability_feedback(&mut feedback, "task", &report.task_scores);
    extend_ruliad_capability_feedback(&mut feedback, "contract", &report.answer_contract_scores);
    extend_ruliad_capability_feedback(&mut feedback, "domain", &report.math_domain_scores);
    extend_ruliad_capability_feedback(&mut feedback, "mode", &report.reasoning_mode_scores);
    feedback
}

pub(super) fn extend_ruliad_capability_feedback(
    output: &mut Vec<burn_dragon_universality::RuliadCapabilityFeedback>,
    prefix: &str,
    groups: &[burn_dragon_universality::RuliadEvalGroupScore],
) {
    output.extend(groups.iter().map(|group| {
        ruliad_capability_feedback_from_group(format!("{prefix}:{}", group.label), group)
    }));
}

pub(super) fn ruliad_capability_feedback_from_group(
    group_label: String,
    group: &burn_dragon_universality::RuliadEvalGroupScore,
) -> burn_dragon_universality::RuliadCapabilityFeedback {
    let count = group.count.max(1);
    let unhealthy = group
        .schema_valid_wrong_count
        .saturating_add(group.malformed_completion_count)
        .saturating_add(group.missing_completion_count)
        .min(count);
    let raw_schema_wrong_rate = group.schema_valid_wrong_count as f32 / count as f32;
    let binding_error = ruliad_capability_group_binding_error(group);
    let raw_completion_health = count.saturating_sub(unhealthy) as f32 / count as f32;
    burn_dragon_universality::RuliadCapabilityFeedback {
        group_label,
        item_count: group.count,
        verifier_rate: group.verifier_accuracy,
        partial_credit_rate: group.partial_credit_rate,
        schema_valid_wrong_rate: raw_schema_wrong_rate.max(binding_error * 0.50),
        malformed_rate: group.malformed_completion_count as f32 / count as f32,
        missing_rate: group.missing_completion_count as f32 / count as f32,
        completion_health_rate: ruliad_capability_group_completion_health(
            group,
            raw_completion_health,
            binding_error,
        ),
    }
}

pub(super) fn ruliad_capability_group_binding_error(
    group: &burn_dragon_universality::RuliadEvalGroupScore,
) -> f32 {
    let answer_error = if group.expected_answer_distinct_fraction >= 0.20 {
        let expected = group.expected_answer_distinct_fraction.max(1.0e-6);
        (1.0 - group.actual_answer_distinct_fraction / expected).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let field_distinct_error = if group.expected_field_value_distinct_fraction > 0.0 {
        (1.0 - group.field_value_distinct_ratio).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let field_dominance_error = if group.count >= 4 {
        ((group.actual_field_value_dominant_fraction - 0.85) / 0.15).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let field_coverage_error = if group.answer_field_expected_count > 0 {
        (1.0 - group.answer_field_coverage).clamp(0.0, 1.0)
    } else {
        0.0
    };
    answer_error
        .max(field_distinct_error)
        .max(field_dominance_error)
        .max(field_coverage_error)
}

pub(super) fn ruliad_capability_group_completion_health(
    group: &burn_dragon_universality::RuliadEvalGroupScore,
    raw_completion_health: f32,
    binding_error: f32,
) -> f32 {
    raw_completion_health.clamp(0.0, 1.0)
        * group.mean_completion_quality.clamp(0.0, 1.0)
        * group.answer_field_coverage.clamp(0.0, 1.0)
        * group.answer_termination_rate.clamp(0.0, 1.0)
        * (1.0 - binding_error * 0.50).clamp(0.0, 1.0)
}

pub(super) fn apply_source_selection_control(
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
        let non_hash_probability = probabilities
            .iter()
            .zip(candidates)
            .filter_map(|(probability, candidate)| {
                (!candidate.is_hash_noise).then_some(*probability)
            })
            .sum::<f32>();
        // Solve scale * H / (scale * H + N) = cap so the cap still
        // holds after normalization. Scaling H directly to `cap` is
        // insufficient because normalization raises its final mass.
        let scale = if hash_max <= f32::EPSILON {
            0.0
        } else if hash_max >= 1.0 - f32::EPSILON || non_hash_probability <= f32::EPSILON {
            1.0
        } else {
            (hash_max * non_hash_probability) / (hash_probability * (1.0 - hash_max))
        };
        for (probability, candidate) in probabilities.iter_mut().zip(candidates) {
            if candidate.is_hash_noise {
                *probability *= scale;
            }
        }
    }
    normalize_source_probabilities(probabilities);
}

#[cfg(test)]
pub(super) fn apply_source_selection_cold_start(
    probabilities: &mut [f32],
    candidates: &[burn_dragon_universality::RuliadSamplerCandidate],
    cold_start: &burn_dragon_universality::RuliadSourceSelectionColdStartConfig,
    absolute_step: Option<usize>,
) {
    if probabilities.is_empty() || probabilities.len() != candidates.len() || !cold_start.enabled {
        return;
    }
    let max_allowed_difficulty =
        current_cold_start_max_difficulty(candidates, cold_start, absolute_step);
    apply_source_selection_cold_start_with_max(
        probabilities,
        candidates,
        cold_start,
        max_allowed_difficulty,
    );
}

fn apply_source_selection_cold_start_with_max(
    probabilities: &mut [f32],
    candidates: &[burn_dragon_universality::RuliadSamplerCandidate],
    cold_start: &burn_dragon_universality::RuliadSourceSelectionColdStartConfig,
    max_allowed_difficulty: Option<usize>,
) {
    if probabilities.is_empty() || probabilities.len() != candidates.len() || !cold_start.enabled {
        return;
    }
    let Some(max_allowed_difficulty) = max_allowed_difficulty else {
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

fn initial_released_max_difficulty_level(
    cold_start: &burn_dragon_universality::RuliadSourceSelectionColdStartConfig,
    candidates: &[burn_dragon_universality::RuliadSamplerCandidate],
) -> usize {
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
    if cold_start.enabled {
        cold_start
            .max_difficulty_level
            .max(min_difficulty)
            .min(max_difficulty)
    } else {
        max_difficulty
    }
}

pub(super) fn current_cold_start_max_difficulty(
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
    let time_cap =
        timed_cold_start_max_difficulty(start_cap, max_difficulty, cold_start, absolute_step);
    let max_allowed_difficulty = if cold_start.release_requires_mastery {
        time_cap.min(mastery_gated_cold_start_max_difficulty(
            candidates,
            cold_start,
            start_cap,
            max_difficulty,
        ))
    } else {
        time_cap
    };
    if max_allowed_difficulty >= max_difficulty {
        None
    } else {
        Some(max_allowed_difficulty)
    }
}

pub(super) fn timed_cold_start_max_difficulty(
    start_cap: usize,
    max_difficulty: usize,
    cold_start: &burn_dragon_universality::RuliadSourceSelectionColdStartConfig,
    absolute_step: usize,
) -> usize {
    if absolute_step <= cold_start.hold_steps {
        return start_cap;
    }
    let ramp_steps = cold_start.ramp_steps.max(1);
    let ramp_step = absolute_step.saturating_sub(cold_start.hold_steps);
    if ramp_step >= ramp_steps {
        return max_difficulty;
    }
    let span = max_difficulty.saturating_sub(start_cap);
    let increment = span.saturating_mul(ramp_step) / ramp_steps;
    start_cap.saturating_add(increment).min(max_difficulty)
}

pub(super) fn mastery_gated_cold_start_max_difficulty(
    candidates: &[burn_dragon_universality::RuliadSamplerCandidate],
    cold_start: &burn_dragon_universality::RuliadSourceSelectionColdStartConfig,
    start_cap: usize,
    max_difficulty: usize,
) -> usize {
    let mut allowed = start_cap;
    while allowed < max_difficulty
        && cold_start_difficulty_mastered(candidates, cold_start, allowed)
    {
        allowed = allowed.saturating_add(1);
    }
    allowed
}

pub(super) fn cold_start_difficulty_mastered(
    candidates: &[burn_dragon_universality::RuliadSamplerCandidate],
    cold_start: &burn_dragon_universality::RuliadSourceSelectionColdStartConfig,
    difficulty_level: usize,
) -> bool {
    let mut saw_candidate = false;
    for candidate in candidates
        .iter()
        .filter(|candidate| candidate.difficulty_level == difficulty_level)
    {
        saw_candidate = true;
        if candidate.capability_feedback_count < cold_start.mastery_min_feedback_count
            || candidate.capability_verifier_ema < cold_start.mastery_verifier_min
            || candidate.capability_completion_health_ema < cold_start.mastery_completion_health_min
            || candidate.capability_schema_wrong_ema > cold_start.mastery_schema_wrong_max
            || candidate.capability_malformed_ema > cold_start.mastery_malformed_max
            || candidate.capability_missing_ema > cold_start.mastery_missing_max
        {
            return false;
        }
    }
    saw_candidate
}

pub(super) fn normalize_source_probabilities(probabilities: &mut [f32]) {
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
