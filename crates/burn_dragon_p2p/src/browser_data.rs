use anyhow::{Result, bail};
use burn_dragon_universality::{OnlineNcaCorpus, OnlineRuliadCorpus, SampleSplit};
use burn_p2p::WorkloadTrainingLease;
use burn_p2p_core::codec::multihash_sha256;
use std::collections::BTreeMap;

use crate::config::{DragonBrowserDatasetSplit, TokenWindowRecord};

pub(crate) fn deterministic_sample_indices(
    sample_count: usize,
    max_samples: Option<usize>,
    selection_key: Option<&str>,
    training_lease: Option<&WorkloadTrainingLease>,
) -> Vec<usize> {
    let limit = max_samples.unwrap_or(sample_count).min(sample_count);
    let mut indices = (0..sample_count).collect::<Vec<_>>();
    let Some(material) = sample_selection_material(selection_key, training_lease) else {
        indices.truncate(limit);
        return indices;
    };

    indices.sort_by_key(|sample_index| {
        (
            sample_selection_rank(&material, *sample_index),
            *sample_index,
        )
    });
    indices.truncate(limit);
    indices
}

pub(crate) struct GeneratedRecordSelection<'a> {
    pub max_documents: Option<usize>,
    pub record_limit: Option<usize>,
    pub selection_key: Option<&'a str>,
    pub training_lease: Option<&'a WorkloadTrainingLease>,
}

const BROWSER_GENERATED_NCA_CACHE_MAX_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BrowserGeneratedNcaCacheStats {
    pub hits: usize,
    pub misses: usize,
    pub bytes: usize,
}

#[derive(Default)]
pub(crate) struct BrowserGeneratedNcaCache {
    source_key: Option<String>,
    runtime: Option<OnlineNcaCorpus>,
    documents: BTreeMap<(u8, usize), Vec<u32>>,
    stats: BrowserGeneratedNcaCacheStats,
}

impl BrowserGeneratedNcaCache {
    pub(crate) fn stats(&self) -> BrowserGeneratedNcaCacheStats {
        self.stats
    }

    fn bind(
        &mut self,
        corpus: &burn_dragon_universality::NcaCorpusConfig,
        block_size: usize,
    ) -> Result<()> {
        let source_key = format!("{block_size}\0{}", serde_json::to_string(corpus)?);
        if self.source_key.as_deref() == Some(source_key.as_str()) {
            return Ok(());
        }
        self.source_key = Some(source_key);
        self.runtime = Some(OnlineNcaCorpus::new_with_min_logical_document_tokens(
            corpus.clone(),
            Some(block_size.saturating_add(1)),
        )?);
        self.documents.clear();
        self.stats = BrowserGeneratedNcaCacheStats::default();
        Ok(())
    }

    fn document_tokens(
        &mut self,
        split: DragonBrowserDatasetSplit,
        sample_index: usize,
    ) -> Result<Vec<u32>> {
        let key = (dataset_split_key(split), sample_index);
        if let Some(tokens) = self.documents.get(&key) {
            self.stats.hits = self.stats.hits.saturating_add(1);
            return Ok(tokens.clone());
        }
        let tokens = self
            .runtime
            .as_ref()
            .expect("generated NCA cache must be bound before use")
            .generate_document_tokens(dataset_split(split), sample_index)?;
        self.stats.misses = self.stats.misses.saturating_add(1);
        let token_bytes = tokens.len().saturating_mul(std::mem::size_of::<u32>());
        if self.stats.bytes.saturating_add(token_bytes) <= BROWSER_GENERATED_NCA_CACHE_MAX_BYTES {
            self.stats.bytes = self.stats.bytes.saturating_add(token_bytes);
            self.documents.insert(key, tokens.clone());
        }
        Ok(tokens)
    }
}

pub(crate) fn generated_nca_records_cached(
    cache: &mut BrowserGeneratedNcaCache,
    corpus: &burn_dragon_universality::NcaCorpusConfig,
    split: DragonBrowserDatasetSplit,
    block_size: usize,
    selection: GeneratedRecordSelection<'_>,
) -> Result<Vec<TokenWindowRecord>> {
    cache.bind(corpus, block_size)?;
    let sample_count = cache
        .runtime
        .as_ref()
        .expect("generated NCA cache must be bound before use")
        .sample_count(dataset_split(split));
    generated_records(sample_count, block_size, selection, |sample_index| {
        cache.document_tokens(split, sample_index)
    })
}

pub(crate) fn generated_ruliad_records(
    corpus: &burn_dragon_universality::RuliadCorpusConfig,
    split: DragonBrowserDatasetSplit,
    block_size: usize,
    supervision: burn_dragon_universality::ruliad::RuliadTokenSupervisionConfig,
    stream_aligned: bool,
    selection: GeneratedRecordSelection<'_>,
) -> Result<Vec<TokenWindowRecord>> {
    let runtime = OnlineRuliadCorpus::new(corpus.clone())?;
    if runtime.document_token_count() <= block_size {
        bail!(
            "generated Ruliad document length {} must exceed browser block_size {}",
            runtime.document_token_count(),
            block_size
        );
    }
    let progress_index = selection
        .training_lease
        .map(|lease| lease.window_id.0 as usize)
        .unwrap_or_default();
    let supervision = supervision.effective_for(
        matches!(split, DragonBrowserDatasetSplit::Validation),
        progress_index,
    );
    generated_records_with_windowizer(
        runtime.sample_count(dataset_split(split)),
        block_size,
        selection,
        |sample_index| {
            runtime.generate_compact_document_tokens_for_epoch(
                dataset_split(split),
                0,
                sample_index,
            )
        },
        |tokens, block_size| {
            ruliad_token_windows_from_tokens(tokens, block_size, supervision, stream_aligned)
        },
    )
}

fn generated_records(
    sample_count: usize,
    block_size: usize,
    selection: GeneratedRecordSelection<'_>,
    generate: impl FnMut(usize) -> Result<Vec<u32>>,
) -> Result<Vec<TokenWindowRecord>> {
    generated_records_with_windowizer(
        sample_count,
        block_size,
        selection,
        generate,
        token_windows_from_tokens,
    )
}

fn generated_records_with_windowizer(
    sample_count: usize,
    block_size: usize,
    selection: GeneratedRecordSelection<'_>,
    mut generate: impl FnMut(usize) -> Result<Vec<u32>>,
    mut windowize: impl FnMut(&[u32], usize) -> Vec<TokenWindowRecord>,
) -> Result<Vec<TokenWindowRecord>> {
    let sample_indices = deterministic_sample_indices(
        sample_count,
        selection.max_documents,
        selection.selection_key,
        selection.training_lease,
    );
    let record_limit = selection.record_limit.unwrap_or(usize::MAX);
    let mut records = Vec::new();
    for sample_index in sample_indices {
        let tokens = generate(sample_index)?;
        records.extend(windowize(&tokens, block_size));
        if records.len() >= record_limit {
            records.truncate(record_limit);
            break;
        }
    }
    Ok(records)
}

fn ruliad_token_windows_from_tokens(
    tokens: &[u32],
    block_size: usize,
    supervision: burn_dragon_universality::ruliad::RuliadTokenSupervisionConfig,
    stream_aligned: bool,
) -> Vec<TokenWindowRecord> {
    use burn_dragon_universality::ruliad::{
        RuliadTokenSupervisionConfig, RuliadTokenSupervisionMode, ruliad_token_loss_mask,
    };

    let answer_window = supervision.mode == RuliadTokenSupervisionMode::AnswerWindow;
    let answer_completion = supervision.mode == RuliadTokenSupervisionMode::AnswerCompletion;
    let mut records = if stream_aligned {
        stream_aligned_token_windows_from_tokens(tokens, block_size)
    } else {
        token_windows_from_tokens(tokens, block_size)
    };
    records.retain_mut(|record| {
        let window = record_window(record);
        if answer_window {
            let mut answer_mask = vec![0; record.targets.len()];
            let contains_answer = ruliad_token_loss_mask(
                &window,
                &mut answer_mask,
                RuliadTokenSupervisionConfig {
                    mode: RuliadTokenSupervisionMode::AnswerCompletion,
                    ..supervision
                },
            );
            intersect_record_loss_mask(record, &mut answer_mask);
            record.loss_mask = Some(answer_mask);
            record.reset_stream_state = true;
            return contains_answer;
        }
        if supervision.uses_target_loss_mask() {
            let mut mask = vec![0; record.targets.len()];
            let any = ruliad_token_loss_mask(&window, &mut mask, supervision);
            intersect_record_loss_mask(record, &mut mask);
            record.loss_mask = Some(mask);
            if answer_completion {
                record.reset_stream_state = true;
                return any;
            }
        }
        true
    });
    records
}

fn intersect_record_loss_mask(record: &TokenWindowRecord, mask: &mut [i64]) {
    if let Some(validity_mask) = record.loss_mask.as_ref() {
        for (weight, valid) in mask.iter_mut().zip(validity_mask) {
            if *valid == 0 {
                *weight = 0;
            }
        }
    }
}

fn record_window(record: &TokenWindowRecord) -> Vec<u32> {
    let mut window = Vec::with_capacity(record.inputs.len().saturating_add(1));
    if let Some(first) = record.inputs.first() {
        window.push(u32::try_from(*first).expect("generated Ruliad token must fit u32"));
    }
    window.extend(
        record
            .targets
            .iter()
            .map(|token| u32::try_from(*token).expect("generated Ruliad token must fit u32")),
    );
    window
}

fn dataset_split(split: DragonBrowserDatasetSplit) -> SampleSplit {
    match split {
        DragonBrowserDatasetSplit::Train => SampleSplit::Train,
        DragonBrowserDatasetSplit::Validation => SampleSplit::Validation,
    }
}

fn dataset_split_key(split: DragonBrowserDatasetSplit) -> u8 {
    match split {
        DragonBrowserDatasetSplit::Train => 0,
        DragonBrowserDatasetSplit::Validation => 1,
    }
}

fn token_windows_from_tokens(tokens: &[u32], block_size: usize) -> Vec<TokenWindowRecord> {
    if tokens.len() <= block_size {
        return Vec::new();
    }
    let max_start = tokens.len() - (block_size + 1);
    let mut records = Vec::new();
    let mut start = 0usize;
    loop {
        let window = &tokens[start..start + block_size + 1];
        records.push(TokenWindowRecord {
            inputs: window[..block_size]
                .iter()
                .map(|token| i64::from(*token))
                .collect(),
            targets: window[1..].iter().map(|token| i64::from(*token)).collect(),
            reset_stream_state: start == 0,
            ..TokenWindowRecord::default()
        });
        if start >= max_start {
            break;
        }
        start = start.saturating_add(block_size).min(max_start);
    }
    records
}

fn stream_aligned_token_windows_from_tokens(
    tokens: &[u32],
    block_size: usize,
) -> Vec<TokenWindowRecord> {
    if tokens.len() <= 1 || block_size == 0 {
        return Vec::new();
    }
    let target_count = tokens.len() - 1;
    let pad_token = tokens.last().copied().unwrap_or_default();
    let mut records = Vec::with_capacity(target_count.div_ceil(block_size));
    for start in (0..target_count).step_by(block_size) {
        let valid_targets = (target_count - start).min(block_size);
        let mut window = vec![pad_token; block_size + 1];
        window[..=valid_targets].copy_from_slice(&tokens[start..=start + valid_targets]);
        let loss_mask = if valid_targets < block_size {
            let mut mask = vec![0; block_size];
            mask[..valid_targets].fill(1);
            Some(mask)
        } else {
            None
        };
        records.push(TokenWindowRecord {
            inputs: window[..block_size]
                .iter()
                .copied()
                .map(i64::from)
                .collect(),
            targets: window[1..].iter().copied().map(i64::from).collect(),
            loss_mask,
            reset_stream_state: start == 0,
            ..TokenWindowRecord::default()
        });
    }
    records
}

fn sample_selection_material(
    selection_key: Option<&str>,
    training_lease: Option<&WorkloadTrainingLease>,
) -> Option<String> {
    let has_selection_key = selection_key.is_some_and(|key| !key.trim().is_empty());
    if !has_selection_key && training_lease.is_none() {
        return None;
    }

    let mut material = selection_key
        .unwrap_or("browser-training")
        .trim()
        .to_owned();
    if let Some(lease) = training_lease {
        material.push_str("|lease=");
        material.push_str(lease.lease_id.as_str());
        material.push_str("|window=");
        material.push_str(&lease.window_id.0.to_string());
        material.push_str("|view=");
        material.push_str(lease.dataset_view_id.as_str());
        material.push_str("|assign=");
        material.push_str(lease.assignment_hash.as_str());
        material.push_str("|micro=");
        for microshard_id in &lease.microshards {
            material.push_str(microshard_id.as_str());
            material.push(',');
        }
    }
    Some(material)
}

fn sample_selection_rank(material: &str, sample_index: usize) -> u64 {
    let digest = multihash_sha256(format!("{material}\0{sample_index}").as_bytes());
    let bytes = digest.get(2..10).unwrap_or(&digest[..digest.len().min(8)]);
    let mut rank = [0_u8; 8];
    for (index, byte) in bytes.iter().enumerate() {
        rank[index] = *byte;
    }
    u64::from_be_bytes(rank)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_dragon_universality::{
        NcaCorpusConfig, NcaSerializationConfig, NcaTokenizationConfig, RuliadCorpusConfig,
        RuliadFormalTaskMixConfig, RuliadSerializationConfig, RuliadSourceSelectionConfig,
        RuliadTokenizationConfig, UsizeRangeConfig,
    };
    use burn_p2p::{ContentId, DatasetViewId, LeaseId, MicroShardId, WindowId};

    fn sample_lease(window_id: u64) -> WorkloadTrainingLease {
        WorkloadTrainingLease {
            lease_id: LeaseId::new(format!("lease-{window_id}")),
            window_id: WindowId(window_id),
            dataset_view_id: DatasetViewId::new("view"),
            assignment_hash: ContentId::new(format!("assignment-{window_id}")),
            microshards: vec![MicroShardId::new("micro-a"), MicroShardId::new("micro-b")],
        }
    }

    fn ruliad_config() -> RuliadCorpusConfig {
        RuliadCorpusConfig {
            output_dir: "ignored".into(),
            seed: 77,
            name: "browser-native-ruliad-parity".into(),
            train_samples: 4,
            validation_samples: 2,
            chunk_token_capacity: 16_384,
            serialization: RuliadSerializationConfig {
                document_tokens: 8192,
                preview_samples: 1,
                ..RuliadSerializationConfig::default()
            },
            tokenization: RuliadTokenizationConfig::StructuredSymbolic {
                vocab_size: 272,
                eos_id: Some(271),
            },
            formal_generalization: Default::default(),
            source_selection: RuliadSourceSelectionConfig {
                difficulty_levels: UsizeRangeConfig { min: 0, max: 1 },
                formal_task_mix: RuliadFormalTaskMixConfig {
                    advance_proof_weight: 1,
                    select_proof_action_weight: 0,
                    construct_proof_weight: 0,
                    check_proof_weight: 0,
                    proof_action_answer_contract: Default::default(),
                },
                ..RuliadSourceSelectionConfig::default()
            },
            families: burn_dragon_universality::ruliad::formal_ruliad_families(),
            proof_tasks: None,
            lean_task_limit: None,
        }
    }

    #[test]
    fn sample_selection_defaults_to_sequential_prefix_without_runtime_identity() {
        assert_eq!(
            deterministic_sample_indices(8, Some(3), None, None),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn sample_selection_uses_active_lease_to_rotate_browser_windows() {
        let first =
            deterministic_sample_indices(32, Some(8), Some("peer-a"), Some(&sample_lease(1)));
        let second =
            deterministic_sample_indices(32, Some(8), Some("peer-a"), Some(&sample_lease(2)));

        assert_eq!(first.len(), 8);
        assert_eq!(second.len(), 8);
        assert_ne!(first, second);
        assert!(first.iter().all(|index| *index < 32));
        assert!(second.iter().all(|index| *index < 32));
    }

    #[test]
    fn generated_ruliad_windows_match_native_runtime_exactly() {
        let config = ruliad_config();
        let block_size = 64;
        let browser = generated_ruliad_records(
            &config,
            DragonBrowserDatasetSplit::Train,
            block_size,
            burn_dragon_universality::ruliad::RuliadTokenSupervisionConfig::default(),
            true,
            GeneratedRecordSelection {
                max_documents: Some(1),
                record_limit: Some(2),
                selection_key: None,
                training_lease: None,
            },
        )
        .expect("browser records");
        let native = OnlineRuliadCorpus::new(config)
            .expect("native corpus")
            .generate_document_tokens(SampleSplit::Train, 0)
            .expect("native tokens");

        assert_eq!(browser.len(), 2);
        assert!(browser[0].reset_stream_state);
        assert!(!browser[1].reset_stream_state);
        assert_eq!(
            browser[0].inputs,
            native[..block_size]
                .iter()
                .copied()
                .map(i64::from)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            browser[0].targets,
            native[1..=block_size]
                .iter()
                .copied()
                .map(i64::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn stream_aligned_tail_is_padded_without_replaying_prior_tokens() {
        let records = stream_aligned_token_windows_from_tokens(&(0..10).collect::<Vec<_>>(), 4);
        assert_eq!(records.len(), 3);
        assert!(records[0].reset_stream_state);
        assert!(!records[1].reset_stream_state);
        assert!(!records[2].reset_stream_state);
        assert_eq!(records[0].inputs, vec![0, 1, 2, 3]);
        assert_eq!(records[1].inputs, vec![4, 5, 6, 7]);
        assert_eq!(records[2].inputs, vec![8, 9, 9, 9]);
        assert_eq!(records[2].targets, vec![9, 9, 9, 9]);
        assert_eq!(records[2].loss_mask, Some(vec![1, 0, 0, 0]));
    }

    #[test]
    fn generated_ruliad_answer_supervision_is_portable_and_stream_safe() {
        use burn_dragon_universality::ruliad::{
            RuliadTokenSupervisionConfig, RuliadTokenSupervisionMode, ruliad_token_loss_mask,
        };

        let supervision = RuliadTokenSupervisionConfig {
            mode: RuliadTokenSupervisionMode::AnswerCompletion,
            mask_high_entropy_spans: true,
            answer_close_marker_weight: 3,
            answer_schema_token_weight: 2,
            answer_schema_start_token_weight: 4,
            answer_value_token_weight: 5,
            ..RuliadTokenSupervisionConfig::default()
        };
        let records = generated_ruliad_records(
            &ruliad_config(),
            DragonBrowserDatasetSplit::Train,
            64,
            supervision,
            false,
            GeneratedRecordSelection {
                max_documents: Some(1),
                record_limit: None,
                selection_key: None,
                training_lease: Some(&sample_lease(2)),
            },
        )
        .expect("answer-supervised browser records");

        assert!(!records.is_empty());
        for record in &records {
            assert!(
                record.reset_stream_state,
                "filtered windows are independent"
            );
            let actual = record.loss_mask.as_ref().expect("portable loss mask");
            let window = record_window(record);
            let mut expected = vec![0; record.targets.len()];
            assert!(ruliad_token_loss_mask(&window, &mut expected, supervision));
            assert_eq!(actual, &expected);
            assert!(actual.iter().any(|weight| *weight > 0));
        }
        assert!(records.iter().any(|record| {
            record
                .targets
                .iter()
                .zip(record.loss_mask.as_ref().expect("mask"))
                .any(|(token, weight)| *token == 265 && *weight == 3)
        }));
    }

    #[test]
    fn generated_ruliad_mixed_supervision_tracks_lease_window() {
        use burn_dragon_universality::ruliad::{
            RuliadTokenSupervisionConfig, RuliadTokenSupervisionMode,
        };

        let supervision = RuliadTokenSupervisionConfig {
            mode: RuliadTokenSupervisionMode::Mixed,
            ..RuliadTokenSupervisionConfig::default()
        };
        let even_lease = sample_lease(2);
        let odd_lease = sample_lease(3);
        let even = generated_ruliad_records(
            &ruliad_config(),
            DragonBrowserDatasetSplit::Train,
            64,
            supervision,
            false,
            GeneratedRecordSelection {
                max_documents: Some(1),
                record_limit: Some(2),
                selection_key: Some("same-peer"),
                training_lease: Some(&even_lease),
            },
        )
        .expect("even-window records");
        let odd = generated_ruliad_records(
            &ruliad_config(),
            DragonBrowserDatasetSplit::Train,
            64,
            supervision,
            false,
            GeneratedRecordSelection {
                max_documents: Some(1),
                record_limit: Some(2),
                selection_key: Some("same-peer"),
                training_lease: Some(&odd_lease),
            },
        )
        .expect("odd-window records");

        assert!(even.iter().all(|record| record.loss_mask.is_some()));
        assert!(odd.iter().all(|record| record.loss_mask.is_none()));
    }

    #[test]
    fn generated_nca_windowing_remains_available_through_shared_path() {
        let config = NcaCorpusConfig {
            output_dir: "ignored".into(),
            seed: 91,
            name: "browser-nca-shared-path".into(),
            train_samples: 2,
            validation_samples: 1,
            chunk_token_capacity: 4096,
            serialization: NcaSerializationConfig::default(),
            tokenization: NcaTokenizationConfig::default(),
            families: burn_dragon_universality::config::default_families(),
        };
        let records = generated_nca_records_cached(
            &mut BrowserGeneratedNcaCache::default(),
            &config,
            DragonBrowserDatasetSplit::Train,
            32,
            GeneratedRecordSelection {
                max_documents: Some(1),
                record_limit: Some(2),
                selection_key: None,
                training_lease: None,
            },
        )
        .expect("NCA records");
        assert!(!records.is_empty());
        assert!(
            records
                .iter()
                .all(|record| record.inputs.len() == 32 && record.targets.len() == 32)
        );
    }

    #[test]
    fn generated_nca_cache_reuses_documents_without_changing_records() {
        let config = NcaCorpusConfig {
            output_dir: "ignored".into(),
            seed: 92,
            name: "browser-nca-cache".into(),
            train_samples: 2,
            validation_samples: 1,
            chunk_token_capacity: 4096,
            serialization: NcaSerializationConfig::default(),
            tokenization: NcaTokenizationConfig::default(),
            families: burn_dragon_universality::config::default_families(),
        };
        let selection = || GeneratedRecordSelection {
            max_documents: Some(1),
            record_limit: Some(1),
            selection_key: None,
            training_lease: None,
        };
        let mut cache = BrowserGeneratedNcaCache::default();
        let first = generated_nca_records_cached(
            &mut cache,
            &config,
            DragonBrowserDatasetSplit::Train,
            32,
            selection(),
        )
        .expect("first cached NCA records");
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);
        assert!(cache.stats().bytes > 0);

        let second = generated_nca_records_cached(
            &mut cache,
            &config,
            DragonBrowserDatasetSplit::Train,
            32,
            selection(),
        )
        .expect("reused cached NCA records");
        assert_eq!(second, first);
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 1);
    }
}
