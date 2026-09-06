use super::*;
use crate::config::RuliadSupervisionMode;
use crate::tokenizer::{PretokenizedTokenizerConfig, TokenizerConfig};
use burn::data::dataloader::DataLoader;
use burn_dragon_universality::config::NcaCorpusConfig;
use burn_dragon_universality::{
    NcaSerializationConfig, NcaTokenizationConfig, RuliadCorpusConfig, RuliadDocumentMode,
    RuliadFamilyConfig, RuliadFamilyKind, RuliadSerializationConfig, RuliadTaskKind,
    RuliadTokenizationConfig, generate_nca_corpus, ruliad_sampler_candidates,
};
use burn_ndarray::NdArray;
use std::collections::{BTreeMap, HashSet};
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
        family.grid_size = Some(burn_dragon_universality::UsizeRangeConfig { min: 12, max: 12 });
        family.steps = Some(burn_dragon_universality::UsizeRangeConfig { min: 10, max: 10 });
        family.state_count = Some(burn_dragon_universality::UsizeRangeConfig { min: 10, max: 10 });
        family.step_stride = Some(burn_dragon_universality::UsizeRangeConfig { min: 2, max: 2 });
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
        formal_generalization: Default::default(),
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
fn fixed_document_padding_mask_keeps_first_eos_and_suppresses_fill() {
    let mut mask = vec![1, 3, 2, 4];
    assert!(mask_fixed_document_eos_padding(
        &[10, 11, 271, 271, 271],
        &mut mask,
        Some(271),
    ));
    assert_eq!(mask, vec![1, 3, 0, 0]);

    let mut padding_only = vec![1; 4];
    assert!(!mask_fixed_document_eos_padding(
        &[271; 5],
        &mut padding_only,
        Some(271),
    ));
    assert_eq!(padding_only, vec![0; 4]);
}

#[test]
fn full_document_universality_emits_eos_padding_loss_masks() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("ruliad.toml");
    fs::write(
        &config_path,
        toml::to_string_pretty(&fixed_ruliad_runtime_config()).expect("toml"),
    )
    .expect("write config");
    let dataset =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 32, 2, &pretokenized_tokenizer())
            .expect("load ruliad dataset");

    assert!(TokenSequenceDataset::uses_target_loss_mask(&dataset));
    let mut mask = vec![0; 4];
    assert!(TokenSequenceDataset::target_loss_mask_for_window(
        &dataset,
        &[10, 11, 50_256, 50_256, 50_256],
        &mut mask,
    ));
    assert_eq!(mask, vec![1, 1, 0, 0]);
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
fn ruliad_answer_target_loss_mask_can_emphasize_answer_values() {
    let window = b"?:q\n!:n=20;alpha=ABC;ok=1\n[/R2]"
        .iter()
        .map(|byte| u32::from(*byte))
        .collect::<Vec<_>>();
    let mut mask = vec![0; window.len() - 1];
    assert!(ruliad_target_loss_mask(
        &window,
        &mut mask,
        RuliadSupervisionConfig {
            mode: RuliadSupervisionMode::AnswerCompletion,
            answer_value_token_weight: 3,
            ..Default::default()
        },
    ));
    let weighted = window
        .iter()
        .skip(1)
        .zip(mask.iter())
        .filter_map(|(token, mask)| (*mask == 3).then_some(*token as u8 as char))
        .collect::<String>();
    let baseline = window
        .iter()
        .skip(1)
        .zip(mask.iter())
        .filter_map(|(token, mask)| (*mask == 1).then_some(*token as u8 as char))
        .collect::<String>();
    assert_eq!(weighted, "20ABC1");
    assert!(
        baseline.contains("n=;alpha=;ok=\n[/R2]"),
        "field names and syntax should remain baseline-supervised: {baseline:?}"
    );
}

#[test]
fn ruliad_answer_target_loss_mask_can_emphasize_answer_schema() {
    let window = b"?:q\n!:n=20;alpha=ABC;ok=1\n[/R2]"
        .iter()
        .map(|byte| u32::from(*byte))
        .collect::<Vec<_>>();
    let mut mask = vec![0; window.len() - 1];
    assert!(ruliad_target_loss_mask(
        &window,
        &mut mask,
        RuliadSupervisionConfig {
            mode: RuliadSupervisionMode::AnswerCompletion,
            answer_schema_token_weight: 3,
            answer_value_token_weight: 1,
            ..Default::default()
        },
    ));
    let weighted = window
        .iter()
        .skip(1)
        .zip(mask.iter())
        .filter_map(|(token, mask)| (*mask == 3).then_some(*token as u8 as char))
        .collect::<String>();
    let baseline = window
        .iter()
        .skip(1)
        .zip(mask.iter())
        .filter_map(|(token, mask)| (*mask == 1).then_some(*token as u8 as char))
        .collect::<String>();
    assert_eq!(weighted, "n=;alpha=;ok=");
    assert!(
        baseline.contains("20ABC1\n[/R2]"),
        "answer values and close marker should remain baseline-supervised: {baseline:?}"
    );
}

#[test]
fn ruliad_answer_target_loss_mask_can_emphasize_schema_starts() {
    let window = b"?:q\n!:xlen=12;xalpha=01;xcounts=8,4;xedge=10\n[/R2]"
        .iter()
        .map(|byte| u32::from(*byte))
        .collect::<Vec<_>>();
    let mut mask = vec![0; window.len() - 1];
    assert!(ruliad_target_loss_mask(
        &window,
        &mut mask,
        RuliadSupervisionConfig {
            mode: RuliadSupervisionMode::AnswerCompletion,
            answer_schema_token_weight: 2,
            answer_schema_start_token_weight: 7,
            answer_value_token_weight: 1,
            ..Default::default()
        },
    ));
    let schema_starts = window
        .iter()
        .skip(1)
        .zip(mask.iter())
        .filter_map(|(token, mask)| (*mask == 7).then_some(*token as u8 as char))
        .collect::<String>();
    let ordinary_schema = window
        .iter()
        .skip(1)
        .zip(mask.iter())
        .filter_map(|(token, mask)| (*mask == 2).then_some(*token as u8 as char))
        .collect::<String>();
    assert_eq!(schema_starts, "xxxx");
    assert!(
        ordinary_schema.contains("len=;alpha=;counts=,;edge="),
        "only the first key byte should be schema-start weighted: {ordinary_schema:?}"
    );
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
fn ruliad_answer_target_loss_mask_can_emphasize_close_markers() {
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
            answer_close_marker_weight: 4,
            ..Default::default()
        },
    ));
    let close = window
        .iter()
        .skip(1)
        .zip(mask.iter())
        .filter_map(|(token, mask)| (*mask == 4).then_some(*token as u8 as char))
        .collect::<String>();
    assert_eq!(close, "[/R2]");
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
fn ruliad_trace_answer_target_loss_mask_supervises_trace_and_weights_answer() {
    let window = b"G:x:h0123456789abcdef;sum=12\n>sum=12\n!:n=20;ok=1\n[/R2]"
        .iter()
        .map(|byte| u32::from(*byte))
        .collect::<Vec<_>>();
    let mut mask = vec![0; window.len() - 1];
    assert!(ruliad_target_loss_mask(
        &window,
        &mut mask,
        RuliadSupervisionConfig {
            mode: RuliadSupervisionMode::TraceAndAnswer,
            mask_high_entropy_spans: true,
            answer_close_marker_weight: 3,
            answer_schema_token_weight: 4,
            answer_value_token_weight: 2,
            ..Default::default()
        },
    ));
    let baseline = window
        .iter()
        .skip(1)
        .zip(mask.iter())
        .filter_map(|(token, mask)| (*mask == 1).then_some(*token as u8 as char))
        .collect::<String>();
    let values = window
        .iter()
        .skip(1)
        .zip(mask.iter())
        .filter_map(|(token, mask)| (*mask == 2).then_some(*token as u8 as char))
        .collect::<String>();
    let close = window
        .iter()
        .skip(1)
        .zip(mask.iter())
        .filter_map(|(token, mask)| (*mask == 3).then_some(*token as u8 as char))
        .collect::<String>();
    let schema = window
        .iter()
        .skip(1)
        .zip(mask.iter())
        .filter_map(|(token, mask)| (*mask == 4).then_some(*token as u8 as char))
        .collect::<String>();

    assert!(baseline.contains(":x:h;sum=12\n>sum=12\n!:"));
    assert!(
        !baseline.contains("0123456789abcdef"),
        "hash payload should be masked even in trace-answer mode: {baseline:?}"
    );
    assert_eq!(schema, "n=;ok=");
    assert_eq!(values, "201");
    assert_eq!(close, "[/R2]");
}

#[test]
fn streamed_ruliad_ranges_preserve_full_document_trace_answer_balance() {
    let text = format!(
        "[R3 x]\nP:{}\n?:root=1\n!:ok=1\n[/R3]",
        "trace;".repeat(256)
    );
    let document = text.bytes().map(u32::from).collect::<Vec<_>>();
    let supervision = RuliadSupervisionConfig {
        mode: RuliadSupervisionMode::TraceAndAnswer,
        balance_trace_answer_mass: true,
        ..Default::default()
    };
    let mut full_mask = vec![0; document.len() - 1];
    assert!(ruliad_target_loss_mask(
        &document,
        &mut full_mask,
        supervision,
    ));

    let block_size = 128;
    let mut stitched = Vec::with_capacity(full_mask.len());
    for start in (0..full_mask.len()).step_by(block_size) {
        let mut chunk_mask = vec![0; block_size];
        assert!(ruliad_target_loss_mask_for_document_range(
            &document,
            document.len(),
            start,
            block_size,
            &mut chunk_mask,
            supervision,
        ));
        let remaining = full_mask.len().saturating_sub(start).min(block_size);
        stitched.extend_from_slice(&chunk_mask[..remaining]);
    }
    assert_eq!(stitched, full_mask);

    let mut answer_targets = vec![0; document.len() - 1];
    assert!(ruliad_answer_target_loss_mask(
        &document,
        &mut answer_targets,
    ));
    let trace_mass = full_mask
        .iter()
        .zip(&answer_targets)
        .filter(|(_, answer)| **answer == 0)
        .map(|(weight, _)| *weight)
        .sum::<i64>();
    let answer_mass = full_mask
        .iter()
        .zip(&answer_targets)
        .filter(|(_, answer)| **answer > 0)
        .map(|(weight, _)| *weight)
        .sum::<i64>();
    let rounding_bound = answer_targets.iter().filter(|weight| **weight > 0).count() as i64;
    assert!((trace_mass - answer_mass).abs() <= rounding_bound);
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
        answer_contract: String::new(),
        difficulty_level,
        params_hash: format!("{difficulty_level:016x}"),
        prior: 1.0,
        cost: 1.0,
        loss_ema: 0.0,
        previous_loss_ema: 0.0,
        gradient_alignment: 0.0,
        is_hash_noise: false,
        capability_feedback_count: 0,
        capability_verifier_ema: 0.0,
        capability_partial_ema: 0.0,
        capability_completion_health_ema: 0.0,
        capability_schema_wrong_ema: 0.0,
        capability_malformed_ema: 0.0,
        capability_missing_ema: 0.0,
    }
}

fn mark_source_selection_candidate_mastered(
    candidate: &mut burn_dragon_universality::RuliadSamplerCandidate,
) {
    candidate.capability_feedback_count = 1;
    candidate.capability_verifier_ema = 0.90;
    candidate.capability_completion_health_ema = 0.95;
    candidate.capability_schema_wrong_ema = 0.05;
    candidate.capability_malformed_ema = 0.0;
    candidate.capability_missing_ema = 0.0;
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
        verifier_match_count: (verifier_accuracy.clamp(0.0, 1.0) * count as f32).round() as usize,
        partial_credit_count: (partial_credit_rate.clamp(0.0, 1.0) * count as f32).round() as usize,
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
        answer_field_observed_count: count,
        answer_field_coverage: 1.0,
        answer_terminated_count: count.saturating_sub(malformed_completion_count),
        answer_termination_rate: count.saturating_sub(malformed_completion_count) as f32
            / count.max(1) as f32,
        mean_completion_quality: 1.0,
        expected_answer_distinct_fraction: 1.0,
        actual_answer_distinct_fraction: 1.0,
        actual_answer_dominant_fraction: 1.0 / count.max(1) as f32,
        expected_field_value_distinct_fraction: 1.0,
        actual_field_value_distinct_fraction: 1.0,
        field_value_distinct_ratio: 1.0,
        actual_field_value_dominant_fraction: 1.0 / count.max(1) as f32,
        presented_action_expected_count: 0,
        presented_action_match_count: 0,
        presented_action_rate: 0.0,
        formal_complexity: None,
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
        answer_field_observed_count: family.answer_field_observed_count,
        answer_field_coverage: family.answer_field_coverage,
        answer_terminated_count: family.answer_terminated_count,
        answer_termination_rate: family.answer_termination_rate,
        mean_completion_quality: family.mean_completion_quality,
        expected_answer_distinct_fraction: family.expected_answer_distinct_fraction,
        actual_answer_distinct_fraction: family.actual_answer_distinct_fraction,
        actual_answer_dominant_fraction: family.actual_answer_dominant_fraction,
        expected_field_value_distinct_fraction: family.expected_field_value_distinct_fraction,
        actual_field_value_distinct_fraction: family.actual_field_value_distinct_fraction,
        field_value_distinct_ratio: family.field_value_distinct_ratio,
        actual_field_value_dominant_fraction: family.actual_field_value_dominant_fraction,
        presented_action_expected_count: family.presented_action_expected_count,
        presented_action_match_count: family.presented_action_match_count,
        presented_action_rate: family.presented_action_rate,
        mean_certificate_prefix_coverage: 0.0,
        mean_completion_tokens: 8.0,
        canary_count: 0,
        canary_semantic_match_count: 0,
        family_scores: vec![family],
        task_scores: Vec::new(),
        difficulty_scores: Vec::new(),
        answer_contract_scores: Vec::new(),
        source_scores: Vec::new(),
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

fn family_probability(
    snapshot: &burn_dragon_universality::RuliadMetricSnapshot,
    family: &str,
) -> f32 {
    snapshot
        .family_buckets
        .iter()
        .find(|bucket| bucket.label == family)
        .map(|bucket| bucket.probability)
        .unwrap_or(0.0)
}

#[test]
fn live_ruliad_source_selection_cold_start_caps_and_releases_difficulty() {
    let candidates = (0..=4).map(source_selection_candidate).collect::<Vec<_>>();
    let cold_start = burn_dragon_universality::RuliadSourceSelectionColdStartConfig {
        enabled: true,
        max_difficulty_level: 2,
        hold_steps: 10,
        ramp_steps: 10,
        ..Default::default()
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
fn live_ruliad_source_selection_cold_start_mastery_gate_blocks_blind_release() {
    let mut candidates = (0..=4).map(source_selection_candidate).collect::<Vec<_>>();
    let cold_start = burn_dragon_universality::RuliadSourceSelectionColdStartConfig {
        enabled: true,
        max_difficulty_level: 0,
        hold_steps: 0,
        ramp_steps: 1,
        release_requires_mastery: true,
        mastery_min_feedback_count: 1,
        ..Default::default()
    };

    let mut unmastered = vec![0.2; candidates.len()];
    apply_source_selection_cold_start(&mut unmastered, &candidates, &cold_start, Some(10));
    assert!(unmastered[0] > 0.0);
    assert_eq!(unmastered[1], 0.0);
    assert_eq!(unmastered[2], 0.0);
    assert_eq!(unmastered[3], 0.0);
    assert_eq!(unmastered[4], 0.0);

    mark_source_selection_candidate_mastered(&mut candidates[0]);
    let mut d0_mastered = vec![0.2; candidates.len()];
    apply_source_selection_cold_start(&mut d0_mastered, &candidates, &cold_start, Some(10));
    assert!(d0_mastered[0] > 0.0);
    assert!(d0_mastered[1] > 0.0);
    assert_eq!(d0_mastered[2], 0.0);
    assert_eq!(d0_mastered[3], 0.0);
    assert_eq!(d0_mastered[4], 0.0);

    mark_source_selection_candidate_mastered(&mut candidates[1]);
    let mut d1_mastered = vec![0.2; candidates.len()];
    apply_source_selection_cold_start(&mut d1_mastered, &candidates, &cold_start, Some(10));
    assert!(d1_mastered[0] > 0.0);
    assert!(d1_mastered[1] > 0.0);
    assert!(d1_mastered[2] > 0.0);
    assert_eq!(d1_mastered[3], 0.0);
    assert_eq!(d1_mastered[4], 0.0);
}

#[test]
fn live_ruliad_mastery_release_is_monotonic_and_checkpointed() {
    let mut candidates = (0..=2).map(source_selection_candidate).collect::<Vec<_>>();
    mark_source_selection_candidate_mastered(&mut candidates[0]);
    let source_selection = burn_dragon_universality::RuliadSourceSelectionConfig {
        enabled: true,
        difficulty_levels: burn_dragon_universality::UsizeRangeConfig { min: 0, max: 2 },
        cold_start: burn_dragon_universality::RuliadSourceSelectionColdStartConfig {
            enabled: true,
            max_difficulty_level: 0,
            hold_steps: 0,
            ramp_steps: 1,
            release_requires_mastery: true,
            mastery_min_feedback_count: 1,
            monotonic_mastery_release: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut corpus = live_ruliad_runtime_config();
    corpus.source_selection = source_selection.clone();
    let state =
        LiveSourceSelectionState::new(source_selection.clone(), corpus.clone(), candidates.clone())
            .expect("live source state");

    assert!(high_difficulty_probability(&state, 10, 1) > 0.0);
    let mut snapshot = state.export_state(10);
    assert_eq!(snapshot.released_max_difficulty_level, 1);
    let d0 = snapshot
        .sampler
        .candidates
        .iter_mut()
        .find(|candidate| candidate.difficulty_level == 0)
        .expect("difficulty zero candidate");
    d0.capability_missing_ema = 1.0;

    let restored = LiveSourceSelectionState::from_snapshot(
        source_selection,
        corpus,
        candidates,
        snapshot,
        RuliadSourceSelectionRestore::StartNewRun,
    )
    .expect("restored source state");
    assert!(
        high_difficulty_probability(&restored, 0, 1) > 0.0,
        "forgetting should shift adaptive probability without revoking an already released level"
    );
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
            ..Default::default()
        };
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");

    let dataset =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 32, 2, &pretokenized_tokenizer())
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
    assert_eq!(snapshot.clock.next_global_step(), 256);
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

    let restored =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 32, 2, &pretokenized_tokenizer())
            .expect("load fresh ruliad dataset")
            .with_source_selection_state_path(
                Some(&state_path),
                RuliadSourceSelectionRestore::StartNewRun,
            )
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

fn clock_regression_state() -> (
    LiveSourceSelectionState,
    burn_dragon_universality::RuliadCorpusConfig,
) {
    let mut corpus = live_ruliad_runtime_config();
    corpus.source_selection.difficulty_levels =
        burn_dragon_universality::UsizeRangeConfig { min: 0, max: 1 };
    corpus.source_selection.sampler.exploration_floor = 1.0;
    corpus.source_selection.cold_start =
        burn_dragon_universality::RuliadSourceSelectionColdStartConfig {
            enabled: true,
            max_difficulty_level: 0,
            hold_steps: 47,
            ramp_steps: 1,
            ..Default::default()
        };
    let candidates = ruliad_sampler_candidates(&corpus);
    let state =
        LiveSourceSelectionState::new(corpus.source_selection.clone(), corpus.clone(), candidates)
            .expect("clock regression source state");
    (state, corpus)
}

#[test]
fn source_selection_checkpoint_export_does_not_advance_live_clock() {
    let (state, _) = clock_regression_state();
    let weights = state.weighted_bucket_labels(Some(0));
    for candidate in state.sampler.lock().expect("sampler").candidates() {
        if candidate.difficulty_level > 0 {
            assert_eq!(
                weights
                    .iter()
                    .find(|(label, _)| label == &candidate.oracle_hash)
                    .expect("bucket")
                    .1,
                0.0,
                "hard eligibility masks must not become tiny positive weights"
            );
        }
    }
    for step in 0..64 {
        let before = state
            .choose_bucket_label_for_step(1, step)
            .expect("source before checkpoint");
        if (step + 1) % 16 == 0 {
            let snapshot = state.export_state(step + 1);
            assert_eq!(snapshot.clock.run_step_origin, 0);
            assert_eq!(snapshot.clock.completed_run_steps, step + 1);
            assert_eq!(snapshot, state.export_state(step + 1));
        }
        assert_eq!(state.effective_absolute_step(Some(step)), Some(step));
        assert_eq!(
            state.choose_bucket_label_for_step(1, step).as_deref(),
            Some(before.as_str()),
            "checkpoint frequency must not change source selection"
        );
        let harder = high_difficulty_probability(&state, step, 1);
        if step < 48 {
            assert!(harder < 1e-6, "early release at step {step}");
        } else {
            assert!(harder > 0.0, "missing release at step {step}");
        }
    }
}

#[test]
fn source_selection_clock_distinguishes_exact_resume_and_new_phase() {
    let (state, corpus) = clock_regression_state();
    let restore = |snapshot, mode| {
        LiveSourceSelectionState::from_snapshot(
            corpus.source_selection.clone(),
            corpus.clone(),
            ruliad_sampler_candidates(&corpus),
            snapshot,
            mode,
        )
        .expect("restore clock")
    };
    let snapshot = state.export_state(16);
    let resumed = restore(snapshot.clone(), RuliadSourceSelectionRestore::ResumeRun);
    let phase = restore(snapshot, RuliadSourceSelectionRestore::StartNewRun);
    for global_step in 16..64 {
        assert_eq!(
            resumed.effective_absolute_step(Some(global_step)),
            Some(global_step)
        );
        assert_eq!(
            phase.effective_absolute_step(Some(global_step - 16)),
            Some(global_step)
        );
        assert_eq!(
            state.choose_bucket_label_for_step(1, global_step),
            resumed.choose_bucket_label_for_step(1, global_step),
            "exact resume changed the sampler trajectory"
        );
        assert_eq!(
            state.choose_bucket_label_for_step(1, global_step),
            phase.choose_bucket_label_for_step(1, global_step - 16),
            "new phase must apply its offset exactly once"
        );
    }
    let next = phase.export_state(64);
    assert_eq!(next.clock.run_step_origin, 16);
    assert_eq!(next.clock.next_global_step(), 80);
    let resumed_phase = restore(next.clone(), RuliadSourceSelectionRestore::ResumeRun);
    let third_phase = restore(next, RuliadSourceSelectionRestore::StartNewRun);
    assert_eq!(resumed_phase.effective_absolute_step(Some(64)), Some(80));
    assert_eq!(third_phase.effective_absolute_step(Some(0)), Some(80));
    assert_eq!(phase.effective_absolute_step(Some(0)), Some(16));
}

#[test]
fn source_selection_restore_rejects_mutating_clock_contract() {
    let (state, corpus) = clock_regression_state();
    let mut old = state.export_state(16);
    old.version = 1;
    assert!(
        LiveSourceSelectionState::from_snapshot(
            corpus.source_selection.clone(),
            corpus.clone(),
            ruliad_sampler_candidates(&corpus),
            old,
            RuliadSourceSelectionRestore::ResumeRun,
        )
        .is_none()
    );
}

#[test]
fn source_selection_phase_handoff_preserves_document_coordinates() {
    for consolidation in [false, true] {
        source_selection_phase_document_handoff(consolidation);
    }
}

fn source_selection_phase_document_handoff(consolidation: bool) {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("source.toml");
    let state_path = dir.path().join("source-state.json");
    let corpus = live_ruliad_runtime_config();
    let mut supervision = RuliadSupervisionConfig::default();
    supervision.consolidation = RuliadConsolidationConfig {
        enabled: consolidation,
        initial_unique_steps: 2,
        hold_steps: 10,
        novelty_interval_steps: 4,
        seed: 23,
    };
    let request = |absolute_step| RuliadWindowRequest {
        split: burn_dragon_universality::SampleSplit::Train,
        epoch_index: 1,
        absolute_step,
        batch_size: 2,
        block_size: 32,
        prefer_answer_window: false,
    };
    fs::write(&config_path, toml::to_string(&corpus).expect("corpus TOML")).expect("write corpus");
    let dataset =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 32, 2, &pretokenized_tokenizer())
            .expect("source dataset")
            .with_ruliad_supervision(supervision);
    let UniversalityStorage::OnTheFly(original) = &dataset.storage else {
        panic!("on-the-fly storage required");
    };
    for step in 0..16 {
        original
            .source_selected_token_windows(request(step))
            .expect("prefix data");
    }
    dataset
        .write_source_selection_state(&state_path, 16)
        .expect("source checkpoint");
    let restored =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 32, 2, &pretokenized_tokenizer())
            .expect("new dataset")
            .with_source_selection_state_path(
                Some(&state_path),
                RuliadSourceSelectionRestore::StartNewRun,
            )
            .expect("new phase")
            .with_ruliad_supervision(supervision);
    let (UniversalityStorage::OnTheFly(original), UniversalityStorage::OnTheFly(phase)) =
        (&dataset.storage, &restored.storage)
    else {
        panic!("on-the-fly storage required");
    };
    let bucket = live_source_selection_state(&dataset)
        .choose_bucket_label_for_step(1, 16)
        .expect("source bucket");
    for local_step in 0..32 {
        assert_eq!(
            original
                .source_selected_token_windows(request(local_step + 16))
                .expect("continuous data"),
            phase
                .source_selected_token_windows(request(local_step))
                .expect("phase data"),
            "document identity changed at local step {local_step}, consolidation={consolidation}"
        );
    }
    assert_eq!(
        original.build_source_bucket_documents(
            burn_dragon_universality::SampleSplit::Validation,
            1,
            0,
            &bucket,
            2,
        ),
        phase.build_source_bucket_documents(
            burn_dragon_universality::SampleSplit::Validation,
            1,
            0,
            &bucket,
            2,
        ),
        "a training phase must not move the validation panel"
    );
}

#[test]
fn source_selection_restore_rehydrates_dynamic_semantic_contract_metadata() {
    let mut config = live_ruliad_runtime_config();
    config.families = burn_dragon_universality::ruliad::formal_ruliad_families();
    config.source_selection.difficulty_levels =
        burn_dragon_universality::UsizeRangeConfig { min: 0, max: 0 };
    config.source_selection.formal_task_mix = burn_dragon_universality::RuliadFormalTaskMixConfig {
        advance_proof_weight: 0,
        select_proof_action_weight: 1,
        construct_proof_weight: 0,
        check_proof_weight: 0,
        proof_action_answer_contract:
            burn_dragon_universality::RuliadProofActionAnswerContract::SemanticStep,
    };
    let configured_candidates = ruliad_sampler_candidates(&config);
    let cached_bucket_label = configured_candidates
        .first()
        .expect("configured candidate")
        .oracle_hash
        .clone();
    let mut dynamic_candidate =
        burn_dragon_universality::ruliad_sampler_candidates_for_difficulty(&config, 3)
            .into_iter()
            .next()
            .expect("dynamic candidate");
    dynamic_candidate.answer_contract.clear();
    dynamic_candidate.loss_ema = 1.25;
    dynamic_candidate.capability_feedback_count = 4;
    let snapshot = RuliadSourceSelectionStateSnapshot {
        version: RULIAD_SOURCE_SELECTION_STATE_VERSION,
        clock: RuliadSourceSelectionClock {
            run_step_origin: 0,
            completed_run_steps: 100,
        },
        frontier_extension_count: 3,
        released_max_difficulty_level: 2,
        control: RuliadSourceSelectionControlSnapshot {
            difficulty_pressure: 1.0,
            hash_noise_max_probability: 1.0,
        },
        sampler: burn_dragon_universality::RuliadFrontierSamplerState {
            candidates: vec![dynamic_candidate],
            capability_posteriors: Default::default(),
            verifier_failures: 0,
        },
        consolidation_bucket_catalog: BTreeMap::from([(7, cached_bucket_label.clone())]),
    };

    let restored = LiveSourceSelectionState::from_snapshot(
        config.source_selection.clone(),
        config,
        configured_candidates,
        snapshot,
        RuliadSourceSelectionRestore::ResumeRun,
    )
    .expect("restored source selection");
    let sampler = restored.sampler.lock().expect("sampler");
    let dynamic = sampler
        .candidates()
        .iter()
        .find(|candidate| candidate.difficulty_level == 3)
        .expect("restored dynamic candidate");

    assert_eq!(dynamic.answer_contract, "proof_action_step");
    assert_eq!(dynamic.loss_ema, 1.25);
    assert_eq!(dynamic.capability_feedback_count, 4);
    drop(sampler);
    assert_eq!(
        restored
            .export_state(100)
            .consolidation_bucket_catalog
            .get(&7),
        Some(&cached_bucket_label)
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
        let start = selected_window_start(&document, usable_len, block_size, 0, &mut rng, false);
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
            mask.contains(&1),
            "answer-target window mask should not be empty"
        );
    }
}

#[test]
fn live_source_selection_documents_per_step_defaults_to_the_full_batch() {
    assert_eq!(
        bounded_live_source_selection_documents_per_step(32, None),
        32
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
fn live_training_coordinates_are_unique_within_each_materialized_page() {
    let coordinates = (0..64)
        .map(|document_rank| {
            live_source_selection_sample_coordinate(
                256,
                burn_dragon_universality::SampleSplit::Train,
                7,
                19,
                "select_proof_action:difficulty=3",
                document_rank,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        coordinates.iter().copied().collect::<HashSet<_>>().len(),
        coordinates.len(),
        "a live batch must not duplicate generated training coordinates"
    );
    assert!(
        coordinates
            .iter()
            .all(|coordinate| coordinate.epoch_index <= u32::MAX as usize),
        "virtual epochs must use a native/wasm-stable fixed width"
    );

    let repeated = (0..64)
        .map(|document_rank| {
            live_source_selection_sample_coordinate(
                256,
                burn_dragon_universality::SampleSplit::Train,
                7,
                19,
                "select_proof_action:difficulty=3",
                document_rank,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(coordinates, repeated, "coordinates must be reproducible");

    let next_step = (0..64)
        .map(|document_rank| {
            live_source_selection_sample_coordinate(
                256,
                burn_dragon_universality::SampleSplit::Train,
                7,
                20,
                "select_proof_action:difficulty=3",
                document_rank,
            )
        })
        .collect::<Vec<_>>();
    assert_ne!(
        coordinates, next_step,
        "successive live steps must address fresh generated documents"
    );
}

#[test]
fn live_validation_coordinates_preserve_the_fixed_panel_epoch() {
    let coordinate = live_source_selection_sample_coordinate(
        32,
        burn_dragon_universality::SampleSplit::Validation,
        11,
        23,
        "select_proof_action:difficulty=2",
        5,
    );
    assert_eq!(coordinate.epoch_index, 11);
    assert!(coordinate.sample_index < 32);
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
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");
    let dataset =
        UniversalityDataset::new_on_the_fly(&config_path, 32, 2, None, &pretokenized_tokenizer())
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
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");

    let dataset_a =
        UniversalityDataset::new_on_the_fly(&config_path, 32, 2, None, &pretokenized_tokenizer())
            .expect("load on-the-fly dataset a");
    let dataset_b =
        UniversalityDataset::new_on_the_fly(&config_path, 32, 2, None, &pretokenized_tokenizer())
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
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");
    let dataset =
        UniversalityDataset::new_on_the_fly(&config_path, 32, 2, None, &pretokenized_tokenizer())
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
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");

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
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");

    let dataset =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 32, 2, &pretokenized_tokenizer())
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
fn source_selected_ruliad_flat_stream_keeps_fixed_document_envelopes() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("ruliad-live.toml");
    let config = live_ruliad_runtime_config();
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");

    let dataset =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 64, 4, &pretokenized_tokenizer())
            .expect("load source-selected ruliad dataset");
    let document_token_count = match &dataset.storage {
        UniversalityStorage::OnTheFly(storage) => {
            let epoch = storage.epoch_documents(burn_dragon_universality::SampleSplit::Train, 2);
            assert!(
                epoch
                    .documents
                    .iter()
                    .all(|document| document.len() == storage.corpus.document_token_count()),
                "flat-stream epoch cache must contain fixed-envelope documents"
            );
            storage.corpus.document_token_count()
        }
        UniversalityStorage::Manifest(_) => panic!("expected on-the-fly ruliad dataset"),
    };

    let mut across_boundary = vec![0u32; 96];
    dataset.copy_token_range_with_epoch(
        DatasetSplit::Train,
        2,
        document_token_count - 32,
        &mut across_boundary,
    );
    assert!(
        across_boundary.iter().any(|token| *token != 0),
        "cross-document stream read should contain generated tokens"
    );
}

#[test]
fn on_the_fly_ruliad_validation_probe_items_are_verifiable() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("ruliad.toml");
    let config = fixed_ruliad_runtime_config();
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");

    let dataset =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 32, 2, &pretokenized_tokenizer())
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
fn fixed_validation_panel_is_seeded_and_independent_of_live_selection() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("ruliad-live.toml");
    let config = live_ruliad_runtime_config();
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");
    let dataset =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 32, 2, &pretokenized_tokenizer())
            .expect("load live Ruliad dataset");

    let first = dataset.sample_ruliad_validation_probe_items_fixed(
        71,
        4,
        RuliadValidationPromptMode::CanonicalTransfer,
    );
    dataset.apply_source_selection_dynamics_control(4.0, 0.0);
    let after_control = dataset.sample_ruliad_validation_probe_items_fixed(
        71,
        4,
        RuliadValidationPromptMode::CanonicalTransfer,
    );
    let other_seed = dataset.sample_ruliad_validation_probe_items_fixed(
        72,
        4,
        RuliadValidationPromptMode::CanonicalTransfer,
    );
    assert_eq!(first.len(), 4);
    assert_eq!(first, after_control);
    assert_ne!(first, other_seed);

    let training_serialization = dataset.sample_ruliad_validation_probe_items_fixed(
        71,
        4,
        RuliadValidationPromptMode::TrainingSerialization,
    );
    assert_eq!(training_serialization.len(), first.len());
    for (canonical, training) in first.iter().zip(&training_serialization) {
        assert_eq!(canonical.item.oracle_hash, training.item.oracle_hash);
        assert_eq!(canonical.item.sample_index, training.item.sample_index);
    }
}

#[test]
fn stratified_fixed_validation_panel_balances_difficulty_and_preserves_coordinates() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("ruliad-stratified.toml");
    let mut config = live_ruliad_runtime_config();
    config.source_selection.difficulty_levels =
        burn_dragon_universality::UsizeRangeConfig { min: 0, max: 1 };
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");
    let dataset =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 32, 2, &pretokenized_tokenizer())
            .expect("load live Ruliad dataset");

    let canonical = dataset.sample_ruliad_validation_probe_items_stratified_fixed(
        91,
        16,
        4,
        RuliadValidationPromptMode::CanonicalTransfer,
    );
    let repeated = dataset.sample_ruliad_validation_probe_items_stratified_fixed(
        91,
        16,
        4,
        RuliadValidationPromptMode::CanonicalTransfer,
    );
    let training = dataset.sample_ruliad_validation_probe_items_stratified_fixed(
        91,
        16,
        4,
        RuliadValidationPromptMode::TrainingSerialization,
    );
    assert_eq!(canonical, repeated);
    assert_eq!(canonical.len(), 16);
    assert_eq!(training.len(), canonical.len());

    let mut difficulty_counts = BTreeMap::<usize, usize>::new();
    let mut families = HashSet::new();
    let mut tasks = HashSet::new();
    for (canonical, training) in canonical.iter().zip(&training) {
        let difficulty = canonical
            .item
            .difficulty_level
            .expect("stratified panel item difficulty");
        *difficulty_counts.entry(difficulty).or_default() += 1;
        families.insert(canonical.item.family.clone());
        tasks.insert(canonical.item.task_kind.clone());
        assert_eq!(canonical.item.oracle_hash, training.item.oracle_hash);
        assert_eq!(canonical.item.sample_index, training.item.sample_index);
        assert_eq!(
            canonical.item.difficulty_level,
            training.item.difficulty_level
        );
    }
    assert_eq!(
        difficulty_counts,
        BTreeMap::from([(0, 4), (1, 4), (2, 4), (3, 4)])
    );
    assert!(
        families.len() >= 2,
        "panel should span families: {families:?}"
    );
    assert!(tasks.len() >= 2, "panel should span tasks: {tasks:?}");
}

#[test]
fn fixed_task_panel_uses_the_requested_seed_and_corpus_identity() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("corpus.toml");
    let mut config = live_ruliad_runtime_config();
    config.serialization.document_tokens = 65_537;
    config.source_selection.formal_task_mix.advance_proof_weight = 0;
    config
        .source_selection
        .formal_task_mix
        .construct_proof_weight = 0;
    config.source_selection.formal_task_mix.check_proof_weight = 0;
    config
        .source_selection
        .formal_task_mix
        .select_proof_action_weight = 1;
    config.families = vec![burn_dragon_universality::ruliad::RuliadFamilyConfig {
        kind: burn_dragon_universality::ruliad::RuliadFamilyKind::FormalProof,
        ..config.families[0].clone()
    }];
    fs::write(&path, toml::to_string(&config).unwrap()).unwrap();
    let dataset =
        UniversalityDataset::new_ruliad_on_the_fly(&path, 32, 2, &pretokenized_tokenizer())
            .unwrap();
    let sample =
        |seed| dataset.sample_ruliad_task_probe_items_fixed(seed, 16, "select_proof_action", 4);
    let first = sample(91);
    assert_eq!(first.len(), 16);
    assert_eq!(first, sample(91));
    assert_ne!(first, sample(92));
    assert!(
        first
            .iter()
            .all(|probe| probe.item.task_kind == "select_proof_action")
    );
    let identity = dataset.ruliad_semantic_fingerprint().unwrap();
    config.seed += 1;
    fs::write(&path, toml::to_string(&config).unwrap()).unwrap();
    let changed =
        UniversalityDataset::new_ruliad_on_the_fly(&path, 32, 2, &pretokenized_tokenizer())
            .unwrap();
    assert_ne!(identity, changed.ruliad_semantic_fingerprint().unwrap());
    assert_eq!(identity, dataset.ruliad_semantic_fingerprint().unwrap());
}

#[test]
fn multi_chunk_validation_exposes_matched_and_transfer_prompt_panels() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("ruliad-multi-chunk.toml");
    let mut config = fixed_ruliad_runtime_config();
    config.serialization.document_tokens = 1539;
    config.serialization.document_mode =
        burn_dragon_universality::RuliadDocumentMode::MultiChunkProofTree;
    config.serialization.document_chunks =
        burn_dragon_universality::UsizeRangeConfig { min: 3, max: 3 };
    config.families = vec![RuliadFamilyConfig {
        kind: RuliadFamilyKind::FormalProof,
        weight: 1,
        width: Some(burn_dragon_universality::UsizeRangeConfig { min: 2, max: 2 }),
        steps: Some(burn_dragon_universality::UsizeRangeConfig { min: 2, max: 2 }),
    }];
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");
    let dataset =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 128, 2, &pretokenized_tokenizer())
            .expect("load ruliad dataset");

    let canonical = dataset.sample_ruliad_validation_probe_items(0, 0, 1);
    let matched = dataset.sample_ruliad_training_serialization_probe_items(0, 0, 1);
    assert_eq!(canonical.len(), 1);
    assert_eq!(matched.len(), 1);
    assert!(canonical[0].item.prompt.trim_start().starts_with("[R3"));
    assert!(matched[0].item.prompt.trim_start().starts_with("[R2"));
    assert!(matched[0].item.prompt.ends_with("\n!:"));
    assert_eq!(matched[0].item.document_close_marker(), "[/R2]");
    assert_eq!(
        matched[0].item.expected_answer,
        canonical[0].item.expected_answer
    );
    assert_eq!(matched[0].item.oracle_hash, canonical[0].item.oracle_hash);
}

#[test]
fn ruliad_validation_probe_panel_is_stable_across_epochs() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("ruliad.toml");
    let config = fixed_ruliad_runtime_config();
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");
    let dataset =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 32, 2, &pretokenized_tokenizer())
            .expect("load ruliad dataset");

    let first = dataset.sample_ruliad_validation_probe_items(1, 31, 4);
    let later = dataset.sample_ruliad_validation_probe_items(19, 91_337, 4);
    let signature = |items: &[RuliadValidationProbeItem]| {
        items
            .iter()
            .map(|probe| {
                (
                    probe.item.oracle_hash.clone(),
                    probe.item.prompt.clone(),
                    probe.item.expected_answer.clone(),
                    probe.prompt_tokens.clone(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(signature(&first), signature(&later));
}

#[test]
fn ruliad_validation_probe_deduplicates_and_stops_at_holdout_capacity() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("ruliad.toml");
    let config = fixed_ruliad_runtime_config();
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");
    let dataset =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 32, 2, &pretokenized_tokenizer())
            .expect("load ruliad dataset");

    let items = dataset.sample_ruliad_validation_probe_items(3, 91, 8);
    assert_eq!(items.len(), config.validation_samples);
    let unique = items
        .iter()
        .map(|probe| probe.item.oracle_hash.as_str())
        .collect::<HashSet<_>>();
    assert_eq!(unique.len(), items.len());
}

#[test]
fn ruliad_policy_probe_materializes_requested_difficulty_buckets() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("ruliad-action.toml");
    let mut config = fixed_ruliad_runtime_config();
    config.validation_samples = 8;
    config.serialization.document_tokens = 8_193;
    config.source_selection.enabled = true;
    config.source_selection.difficulty_levels =
        burn_dragon_universality::UsizeRangeConfig { min: 0, max: 1 };
    config.source_selection.formal_task_mix.advance_proof_weight = 0;
    config
        .source_selection
        .formal_task_mix
        .select_proof_action_weight = 1;
    config
        .source_selection
        .formal_task_mix
        .construct_proof_weight = 0;
    config.source_selection.formal_task_mix.check_proof_weight = 0;
    config.families = vec![RuliadFamilyConfig {
        kind: RuliadFamilyKind::FormalProof,
        weight: 1,
        width: Some(burn_dragon_universality::UsizeRangeConfig { min: 2, max: 3 }),
        steps: Some(burn_dragon_universality::UsizeRangeConfig { min: 2, max: 3 }),
    }];
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");
    let dataset =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 512, 2, &pretokenized_tokenizer())
            .expect("load ruliad dataset");

    let items = dataset.sample_ruliad_validation_probe_items_stratified(
        1,
        10,
        12,
        RuliadTaskKind::SelectProofAction.label(),
        3,
    );
    let later_items = dataset.sample_ruliad_validation_probe_items_stratified(
        17,
        91_337,
        12,
        RuliadTaskKind::SelectProofAction.label(),
        3,
    );

    assert_eq!(items.len(), 12);
    assert_eq!(
        items
            .iter()
            .map(|probe| (
                probe.item.oracle_hash.as_str(),
                probe.item.difficulty_level,
                probe.prompt_tokens.as_slice(),
            ))
            .collect::<Vec<_>>(),
        later_items
            .iter()
            .map(|probe| (
                probe.item.oracle_hash.as_str(),
                probe.item.difficulty_level,
                probe.prompt_tokens.as_slice(),
            ))
            .collect::<Vec<_>>()
    );
    let mut counts = BTreeMap::<usize, usize>::new();
    for probe in items {
        let difficulty_level = probe.item.difficulty_level.expect("difficulty level");
        let Some(burn_dragon_universality::RuliadSampleSpec::FormalProof { task, .. }) =
            probe.item.spec
        else {
            panic!("expected formal proof policy item");
        };
        assert_eq!(task, RuliadTaskKind::SelectProofAction);
        *counts.entry(difficulty_level).or_default() += 1;
    }
    assert_eq!(counts, BTreeMap::from([(0, 4), (1, 4), (2, 4)]));

    let wrapped = crate::dataset::Dataset::from_universality(dataset);
    let training_batch = TokenSequenceDataset::source_selected_ruliad_policy_batch(
        &wrapped,
        DatasetSplit::Train,
        1,
        10,
        12,
        3,
    )
    .expect("stratified training policy batch");
    let training_counts = training_batch.samples.iter().fold(
        BTreeMap::<usize, usize>::new(),
        |mut counts, sample| {
            *counts
                .entry(sample.item.difficulty_level.expect("difficulty level"))
                .or_default() += 1;
            counts
        },
    );
    assert_eq!(training_counts, BTreeMap::from([(0, 4), (1, 4), (2, 4)]));
}

#[test]
fn on_the_fly_ruliad_dataset_exposes_multi_chunk_documents() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("ruliad-multichunk.toml");
    let mut config = fixed_ruliad_runtime_config();
    config.serialization.document_mode = RuliadDocumentMode::MultiChunkProofTree;
    config.serialization.document_chunks =
        burn_dragon_universality::UsizeRangeConfig { min: 3, max: 3 };
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");

    let dataset =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 512, 2, &pretokenized_tokenizer())
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
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");

    let dataset =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 32, 2, &pretokenized_tokenizer())
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
    let policy_batch = crate::dataset::TokenSequenceDataset::source_selected_ruliad_policy_batch(
        &wrapped,
        DatasetSplit::Train,
        0,
        0,
        2,
        0,
    )
    .expect("source-selected ruliad policy batch");
    assert_eq!(policy_batch.samples.len(), 2);
    for sample in policy_batch.samples.iter() {
        assert!(!sample.prompt_tokens.is_empty());
        let prompt = dataset
            .decode_ruliad_payload_tokens(&sample.prompt_tokens, true)
            .expect("decode ruliad prompt");
        assert!(prompt.contains("!:"));
        let oracle_completion = format!(
            "!:{}\n{}",
            sample.item.expected_answer,
            sample.item.document_close_marker()
        );
        let score = burn_dragon_universality::ruliad::score_ruliad_item_completion(
            &sample.item,
            Some(&oracle_completion),
        );
        assert!(score.verifier_match(), "oracle answer should verify");
    }
    let validation_windows = crate::dataset::TokenSequenceDataset::source_selected_token_windows(
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
fn static_live_ruliad_source_selection_ignores_training_feedback() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("ruliad-static-live.toml");
    let mut config = live_ruliad_runtime_config();
    config.source_selection.difficulty_levels =
        burn_dragon_universality::UsizeRangeConfig { min: 0, max: 4 };
    config.source_selection.cold_start =
        burn_dragon_universality::RuliadSourceSelectionColdStartConfig {
            enabled: true,
            max_difficulty_level: 1,
            hold_steps: 100,
            ramp_steps: 100,
            ..Default::default()
        };
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");

    let make_dataset = || {
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 32, 2, &pretokenized_tokenizer())
            .expect("load static live ruliad dataset")
            .with_source_selection_feedback_updates_enabled(Some(false))
    };
    let dataset = make_dataset();
    let comparison = make_dataset();
    assert_eq!(
        dataset.source_selection_feedback_updates_enabled(),
        Some(false)
    );
    let wrapped = crate::dataset::Dataset::from_universality(dataset.clone());
    let comparison_wrapped = crate::dataset::Dataset::from_universality(comparison.clone());
    let mut fingerprints = HashSet::new();
    for absolute_step in 0..32 {
        let left =
            crate::dataset::TokenSequenceDataset::source_selected_token_windows_with_loss_masks(
                &wrapped,
                DatasetSplit::Train,
                0,
                absolute_step,
                2,
                32,
            )
            .expect("left source-selected batch");
        let right =
            crate::dataset::TokenSequenceDataset::source_selected_token_windows_with_loss_masks(
                &comparison_wrapped,
                DatasetSplit::Train,
                0,
                absolute_step,
                2,
                32,
            )
            .expect("right source-selected batch");
        let left_fingerprint = left.fingerprint();
        let right_fingerprint = right.fingerprint();
        assert_eq!(
            left_fingerprint, right_fingerprint,
            "open-loop batches diverged at absolute step {absolute_step}"
        );
        fingerprints.insert(left_fingerprint);

        let left_snapshot = crate::dataset::TokenSequenceDataset::record_source_selection_loss(
            &wrapped,
            absolute_step,
            0.01 + absolute_step as f32,
        )
        .expect("left static snapshot");
        let right_snapshot = crate::dataset::TokenSequenceDataset::record_source_selection_loss(
            &comparison_wrapped,
            absolute_step,
            1000.0 - absolute_step as f32,
        )
        .expect("right static snapshot");
        assert_eq!(left_snapshot, right_snapshot);
        assert_eq!(left_snapshot.max_difficulty_level, 4);
        assert!(left_snapshot.active_max_difficulty_level <= 1);
        assert!(
            left_snapshot
                .difficulty_buckets
                .iter()
                .filter(|bucket| bucket.mean_difficulty_level > 1.0)
                .all(|bucket| bucket.probability <= 1e-6),
            "static telemetry must reflect the effective cold-start distribution at step {absolute_step}: {:?}",
            left_snapshot.difficulty_buckets
        );
    }
    assert!(
        fingerprints.len() >= 24,
        "open-loop stream should remain diverse across steps: unique={}",
        fingerprints.len()
    );
}

#[test]
fn ruliad_cold_start_override_exposes_the_materialized_open_loop_frontier() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("ruliad-cold-start-override.toml");
    let mut config = live_ruliad_runtime_config();
    config.source_selection.difficulty_levels =
        burn_dragon_universality::UsizeRangeConfig { min: 0, max: 4 };
    config.source_selection.cold_start =
        burn_dragon_universality::RuliadSourceSelectionColdStartConfig {
            enabled: true,
            max_difficulty_level: 1,
            hold_steps: 1_000_000,
            ramp_steps: 1_000_000,
            ..Default::default()
        };
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");

    let capped =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 32, 2, &pretokenized_tokenizer())
            .expect("load capped ruliad dataset");
    let aligned = UniversalityDataset::new_ruliad_on_the_fly_with_overrides(
        &config_path,
        32,
        2,
        &pretokenized_tokenizer(),
        RuliadSourceSelectionOverrides {
            cold_start_enabled: Some(false),
            documents_per_step: None,
        },
    )
    .expect("load aligned ruliad dataset");

    assert_eq!(capped.source_selection_cold_start_enabled(), Some(true));
    assert_eq!(aligned.source_selection_cold_start_enabled(), Some(false));
    assert_eq!(
        capped
            .source_selection_snapshot_at_step(0)
            .expect("capped source snapshot")
            .active_max_difficulty_level,
        1
    );
    assert_eq!(
        aligned
            .source_selection_snapshot_at_step(0)
            .expect("aligned source snapshot")
            .active_max_difficulty_level,
        4
    );
}

#[test]
fn live_ruliad_source_batches_are_shared_across_tbptt_chunks() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("ruliad-live-cache.toml");
    let config = live_ruliad_runtime_config();
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");
    let dataset =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 32, 2, &pretokenized_tokenizer())
            .expect("load ruliad dataset");
    let storage = match &dataset.storage {
        UniversalityStorage::OnTheFly(storage) => storage.clone(),
        UniversalityStorage::Manifest(_) => panic!("expected on-the-fly storage"),
    };
    let bucket = storage
        .corpus
        .source_buckets()
        .into_iter()
        .next()
        .expect("source bucket")
        .label();

    let first = storage.generate_source_bucket_documents(
        burn_dragon_universality::SampleSplit::Train,
        3,
        11,
        &bucket,
        8,
    );
    let second = storage.generate_source_bucket_documents(
        burn_dragon_universality::SampleSplit::Train,
        3,
        11,
        &bucket,
        8,
    );
    assert_eq!(first.len(), second.len());
    assert!(
        first
            .iter()
            .zip(&second)
            .all(|(left, right)| Arc::ptr_eq(left, right)),
        "the same source decision should reuse its generated documents"
    );
    assert_eq!(
        first
            .iter()
            .map(|document| document.as_slice())
            .collect::<HashSet<_>>()
            .len(),
        first.len(),
        "one live source batch should contain distinct generated documents"
    );

    let next = storage.generate_source_bucket_documents(
        burn_dragon_universality::SampleSplit::Train,
        3,
        12,
        &bucket,
        8,
    );
    assert!(
        first
            .iter()
            .zip(&next)
            .any(|(left, right)| !Arc::ptr_eq(left, right)),
        "a new source decision must not alias an older batch"
    );
    assert_ne!(
        first
            .iter()
            .map(|document| document.as_slice())
            .collect::<Vec<_>>(),
        next.iter()
            .map(|document| document.as_slice())
            .collect::<Vec<_>>(),
        "successive source decisions should generate fresh document content"
    );
    let cache = storage.live_batch_cache.inner.lock().expect("live cache");
    assert_eq!(cache.entries.len(), 2);
    assert!(cache.total_bytes > 0);
}

#[test]
fn live_ruliad_source_selection_records_capability_feedback() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("ruliad-live.toml");
    let config = live_ruliad_runtime_config();
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");

    let dataset =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 32, 2, &pretokenized_tokenizer())
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
fn versioned_source_feedback_replaces_overlapping_marginals() {
    let mut report =
        capability_feedback_report(capability_group("formal_proof", 8, 0.0, 0.25, 6, 0, 0));
    report.difficulty_scores = vec![capability_group("d0", 8, 0.0, 0.25, 6, 0, 0)];
    report.task_scores = vec![capability_group(
        "select_proof_action",
        8,
        0.0,
        0.25,
        6,
        0,
        0,
    )];
    let source_label = burn_dragon_universality::ruliad_source_capability_label(
        "formal_proof",
        "select_proof_action",
        0,
        "proof_action_step",
    );
    report.source_scores = vec![capability_group(&source_label, 8, 0.75, 0.8, 2, 0, 0)];

    let feedback = ruliad_capability_feedback_from_report(&report);

    assert_eq!(feedback.len(), 1);
    assert_eq!(feedback[0].group_label, source_label);
    assert_eq!(feedback[0].item_count, 8);
    assert!((feedback[0].verifier_rate - 0.75).abs() < 1.0e-6);
}

#[test]
fn live_ruliad_capability_feedback_snapshot_honors_cold_start_step() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("ruliad-live.toml");
    let mut config = live_ruliad_runtime_config();
    config.source_selection.difficulty_levels =
        burn_dragon_universality::UsizeRangeConfig { min: 0, max: 4 };
    config.source_selection.cold_start =
        burn_dragon_universality::RuliadSourceSelectionColdStartConfig {
            enabled: true,
            max_difficulty_level: 1,
            hold_steps: 100,
            ramp_steps: 100,
            ..Default::default()
        };
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");

    let dataset =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 32, 2, &pretokenized_tokenizer())
            .expect("load ruliad dataset");
    let mut report =
        capability_feedback_report(capability_group("simulation", 16, 1.0, 0.5, 0, 0, 0));
    report.difficulty_scores = vec![capability_group("d0", 8, 1.0, 0.5, 0, 0, 0)];

    let unconstrained = dataset
        .record_ruliad_capability_feedback(&report)
        .expect("unconstrained capability snapshot");
    let held = dataset
        .record_ruliad_capability_feedback_at_step(&report, Some(0))
        .expect("step-aware capability snapshot");

    assert!(
        unconstrained.mean_difficulty_level > held.mean_difficulty_level,
        "legacy snapshot should expose the full sampler while step-aware telemetry should reflect the held curriculum"
    );
    assert_eq!(held.max_difficulty_level, 4);
    assert_eq!(held.active_max_difficulty_level, 1);
    assert!(held.active_max_difficulty_probability > 0.0);
    assert!(
        held.difficulty_buckets
            .iter()
            .filter(|bucket| bucket.mean_difficulty_level > 1.0)
            .all(|bucket| bucket.probability <= 1e-6),
        "cold-start telemetry must not report probability mass above the active cap: {:?}",
        held.difficulty_buckets
    );
    assert!(
        held.top_buckets
            .iter()
            .filter(|bucket| bucket.probability > 1e-6)
            .all(|bucket| bucket.difficulty_level <= 1),
        "positive-probability top buckets should match the current cold-start cap: {:?}",
        held.top_buckets
    );
}

#[test]
fn live_ruliad_source_selection_records_domain_and_mode_capability_feedback() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("ruliad-live.toml");
    let mut config = live_ruliad_runtime_config();
    config.families = vec![
        RuliadFamilyConfig {
            kind: RuliadFamilyKind::Category,
            weight: 1,
            width: Some(burn_dragon_universality::UsizeRangeConfig { min: 4, max: 4 }),
            steps: Some(burn_dragon_universality::UsizeRangeConfig { min: 2, max: 2 }),
        },
        RuliadFamilyConfig {
            kind: RuliadFamilyKind::Rewrite,
            weight: 1,
            width: Some(burn_dragon_universality::UsizeRangeConfig { min: 4, max: 4 }),
            steps: Some(burn_dragon_universality::UsizeRangeConfig { min: 2, max: 2 }),
        },
    ];
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");

    let dataset =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 32, 2, &pretokenized_tokenizer())
            .expect("load ruliad dataset");
    let before = dataset.source_selection_snapshot().expect("snapshot");
    let before_category = family_probability(&before, "category");
    let before_rewrite = family_probability(&before, "rewrite");
    let mut report =
        capability_feedback_report(capability_group("unused_family", 16, 0.0, 0.0, 0, 0, 0));
    report.family_scores.clear();
    report.math_domain_scores = vec![capability_group("category_theory", 16, 0.0, 0.25, 12, 0, 0)];

    let after_domain = dataset
        .record_ruliad_capability_feedback(&report)
        .expect("domain capability feedback snapshot");
    let after_domain_rewrite = family_probability(&after_domain, "rewrite");
    assert!(
        family_probability(&after_domain, "category") > before_category,
        "category-theory domain feedback should raise category sampling probability"
    );

    report.math_domain_scores.clear();
    report.reasoning_mode_scores = vec![capability_group("normalization", 16, 0.0, 0.25, 12, 0, 0)];
    let after_mode = dataset
        .record_ruliad_capability_feedback(&report)
        .expect("mode capability feedback snapshot");
    assert!(
        family_probability(&after_mode, "rewrite") > after_domain_rewrite
            && family_probability(&after_mode, "rewrite") > before_rewrite * 0.90,
        "normalization-mode feedback should raise rewrite sampling probability"
    );
}

#[test]
fn live_ruliad_source_selection_records_answer_contract_capability_feedback() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("ruliad-live.toml");
    let mut config = live_ruliad_runtime_config();
    config.families = vec![
        RuliadFamilyConfig {
            kind: RuliadFamilyKind::Category,
            weight: 1,
            width: Some(burn_dragon_universality::UsizeRangeConfig { min: 4, max: 4 }),
            steps: Some(burn_dragon_universality::UsizeRangeConfig { min: 2, max: 2 }),
        },
        RuliadFamilyConfig {
            kind: RuliadFamilyKind::Automaton,
            weight: 1,
            width: Some(burn_dragon_universality::UsizeRangeConfig { min: 4, max: 4 }),
            steps: Some(burn_dragon_universality::UsizeRangeConfig { min: 2, max: 2 }),
        },
    ];
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");

    let dataset =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 32, 2, &pretokenized_tokenizer())
            .expect("load ruliad dataset");
    let before = dataset.source_selection_snapshot().expect("snapshot");
    let before_category = family_probability(&before, "category");
    let before_automaton = family_probability(&before, "automaton");
    let mut report =
        capability_feedback_report(capability_group("unused_family", 16, 0.0, 0.0, 0, 0, 0));
    report.family_scores.clear();
    report.difficulty_scores.clear();
    report.answer_contract_scores = vec![capability_group("ok,l,r", 16, 0.75, 0.85, 1, 0, 0)];

    let after = dataset
        .record_ruliad_capability_feedback(&report)
        .expect("contract capability feedback snapshot");
    assert!(
        family_probability(&after, "category") > before_category,
        "ok/l/r contract feedback should raise category sampling probability"
    );
    assert!(
        family_probability(&after, "automaton") <= before_automaton,
        "ok/l/r contract feedback should not promote automaton acc buckets"
    );
}

#[test]
fn live_ruliad_source_selection_treats_field_collapse_as_contract_remediation() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("ruliad-live.toml");
    let mut config = live_ruliad_runtime_config();
    config.families = vec![
        RuliadFamilyConfig {
            kind: RuliadFamilyKind::Category,
            weight: 1,
            width: Some(burn_dragon_universality::UsizeRangeConfig { min: 4, max: 4 }),
            steps: Some(burn_dragon_universality::UsizeRangeConfig { min: 2, max: 2 }),
        },
        RuliadFamilyConfig {
            kind: RuliadFamilyKind::Automaton,
            weight: 1,
            width: Some(burn_dragon_universality::UsizeRangeConfig { min: 4, max: 4 }),
            steps: Some(burn_dragon_universality::UsizeRangeConfig { min: 2, max: 2 }),
        },
    ];
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");

    let dataset =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 32, 2, &pretokenized_tokenizer())
            .expect("load ruliad dataset");
    let before = dataset.source_selection_snapshot().expect("snapshot");
    let before_category = family_probability(&before, "category");
    let before_automaton = family_probability(&before, "automaton");
    let mut report =
        capability_feedback_report(capability_group("unused_family", 16, 0.0, 0.0, 0, 0, 0));
    report.family_scores.clear();
    report.difficulty_scores.clear();
    let mut collapsed_contract = capability_group("ok,l,r", 16, 0.75, 0.85, 0, 0, 0);
    collapsed_contract.field_value_distinct_ratio = 0.05;
    collapsed_contract.actual_field_value_distinct_fraction = 0.05;
    collapsed_contract.actual_field_value_dominant_fraction = 0.95;
    report.answer_contract_scores = vec![collapsed_contract];

    let after = dataset
        .record_ruliad_capability_feedback(&report)
        .expect("contract capability feedback snapshot");
    let state = live_source_selection_state(&dataset);
    let sampler = state.sampler.lock().expect("ruliad sampler lock");
    let category_feedback = sampler
        .candidates()
        .iter()
        .filter(|candidate| candidate.family == "category")
        .find(|candidate| candidate.capability_feedback_count > 0)
        .expect("category ok/l/r candidate feedback");
    let automaton_feedback = sampler
        .candidates()
        .iter()
        .filter(|candidate| candidate.family == "automaton")
        .all(|candidate| candidate.capability_feedback_count == 0);

    assert!(
        family_probability(&after, "category") > before_category,
        "field collapse in ok/l/r contract should raise category remediation probability"
    );
    assert!(
        family_probability(&after, "automaton") <= before_automaton,
        "field collapse in ok/l/r contract should not target automaton acc buckets"
    );
    assert!(
        category_feedback.capability_schema_wrong_ema >= 0.45,
        "field collapse should become schema/binding remediation pressure: {category_feedback:?}"
    );
    assert!(
        category_feedback.capability_completion_health_ema < 0.80,
        "field collapse should lower effective completion health: {category_feedback:?}"
    );
    assert!(automaton_feedback);
}

#[test]
fn live_ruliad_streaming_records_chunk_loss_feedback_without_epoch_cache() {
    type TestBackend = NdArray<f32>;

    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("ruliad-live.toml");
    let config = live_ruliad_runtime_config();
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");

    let dataset =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 32, 2, &pretokenized_tokenizer())
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
fn production_verifier_stream_materializes_every_scheduled_policy_panel() {
    type TestBackend = NdArray<f32>;

    let tokenizer = TokenizerConfig {
        vocab_path: None,
        kind: TokenizerKind::Pretokenized(PretokenizedTokenizerConfig {
            vocab_size: 272,
            bos_id: None,
            eos_id: Some(271),
            pad_id: None,
            unk_id: None,
        }),
    };
    let mut supervision = crate::config::RuliadSupervisionConfig::default();
    supervision.proof_policy.enabled = true;
    supervision.proof_policy.weight = 1.0;
    supervision.proof_policy.every_steps = 16;
    supervision.proof_policy.start_after_steps = 0;
    supervision.proof_policy.stratified_difficulty_levels = 4;
    assert!(supervision.needs_ruliad_policy_batch());
    assert!(supervision.needs_ruliad_policy_batch_at_step(0));
    assert!(!supervision.needs_ruliad_policy_batch_at_step(1));
    let corpus_config = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../burn_dragon_p2p/deploy/profiles/ruliad-r3.semantic-action.corpus.toml");
    let dataset = UniversalityDataset::new_ruliad_on_the_fly(&corpus_config, 128, 32, &tokenizer)
        .expect("load production verifier corpus")
        .with_ruliad_supervision(supervision);
    let wrapped = Arc::new(crate::dataset::Dataset::from_universality(dataset));
    assert!(wrapped.uses_live_source_selection());
    assert!(
        TokenSequenceDataset::source_selected_ruliad_policy_batch(
            wrapped.as_ref(),
            DatasetSplit::Train,
            0,
            7,
            32,
            4,
        )
        .is_some(),
        "production verifier corpus must expose policy batches"
    );
    let device = burn::tensor::Device::<TestBackend>::default();
    let loader = crate::dataset::StreamingDataLoader::<TestBackend>::new(
        wrapped,
        DatasetSplit::Train,
        &device,
        64,
        Some(64),
        Some(64),
        20260831,
    )
    .with_ruliad_policy_supervision(supervision)
    .with_ruliad_policy_stratified_difficulty_levels(4);

    let batches = loader
        .iter()
        .map(|batch| (batch.absolute_step, batch.ruliad_policy_batch))
        .collect::<Vec<_>>();
    assert_eq!(
        batches.iter().map(|(step, _)| *step).collect::<Vec<_>>(),
        (0..64).map(Some).collect::<Vec<_>>()
    );
    let scheduled = batches
        .into_iter()
        .filter_map(|(step, policy)| policy.map(|policy| (step, policy)))
        .collect::<Vec<_>>();
    assert_eq!(
        scheduled.iter().map(|(step, _)| *step).collect::<Vec<_>>(),
        vec![Some(0), Some(16), Some(32), Some(48)]
    );
    for (_, policy) in scheduled {
        assert_eq!(policy.samples.len(), 32);
        assert_eq!(
            policy
                .samples
                .iter()
                .filter_map(|sample| sample.item.difficulty_level)
                .collect::<HashSet<_>>(),
            HashSet::from([0, 1, 2, 3]),
        );
        assert!(policy.samples.iter().all(|sample| matches!(
            sample.item.spec,
            Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
                task: RuliadTaskKind::SelectProofAction,
                ..
            })
        )));
    }
}

#[test]
fn production_verifier_supervision_audit_exposes_stateless_context_gap() {
    let tokenizer = TokenizerConfig {
        vocab_path: None,
        kind: TokenizerKind::Pretokenized(PretokenizedTokenizerConfig {
            vocab_size: 272,
            bos_id: None,
            eos_id: Some(271),
            pad_id: None,
            unk_id: None,
        }),
    };
    let corpus_config = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../burn_dragon_p2p/deploy/profiles/ruliad-r3.semantic-action.corpus.toml");
    let supervision = RuliadSupervisionConfig {
        mode: RuliadSupervisionMode::AnswerCompletion,
        ..Default::default()
    };
    let audit = |block_size| {
        UniversalityDataset::new_ruliad_on_the_fly(&corpus_config, block_size, 2, &tokenizer)
            .expect("load production verifier corpus")
            .with_ruliad_supervision(supervision)
            .with_ruliad_supervision_audit(2)
            .expect("audit production verifier corpus")
            .ruliad_supervision_audit()
            .expect("Ruliad audit report")
            .clone()
    };

    let short = audit(128);
    assert!(short.total_query_conditioning_samples > 0);
    assert_eq!(short.query_visible_within_block_fraction, 0.0);
    assert!(
        short
            .buckets
            .iter()
            .any(|bucket| bucket.max_query_to_answer_tokens > 128)
    );

    let boundary_sensitive = audit(256);
    assert_eq!(boundary_sensitive.query_visible_within_block_fraction, 0.25);

    let sufficient = audit(4096);
    assert_eq!(sufficient.query_visible_within_block_fraction, 1.0);
}

#[test]
fn fixed_holdout_uses_semantic_windows_instead_of_padded_document_capacity() {
    type TestBackend = NdArray<f32>;

    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("ruliad-fixed-holdout.toml");
    let mut config = live_ruliad_runtime_config();
    config.serialization.document_tokens = 65_537;
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");

    let dataset = Arc::new(crate::dataset::Dataset::from_universality(
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 32, 4, &pretokenized_tokenizer())
            .expect("load ruliad dataset"),
    ));
    let device = burn::tensor::Device::<TestBackend>::default();
    let loader = crate::dataset::RandomDataLoader::<TestBackend>::new(
        Arc::clone(&dataset),
        DatasetSplit::Val,
        &device,
        2,
        None,
    )
    .with_seed(7)
    .with_source_selection_enabled(false);
    let batches = loader.iter().collect::<Vec<_>>();

    assert_eq!(batches.len(), 2);
    for batch in batches {
        let supervised_tokens = batch
            .loss_mask
            .expect("fixed holdout loss mask")
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .expect("loss mask")
            .into_iter()
            .sum::<i64>();
        assert!(
            supervised_tokens > 0,
            "fixed holdout must not sample only EOS padding"
        );
    }
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
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");

    let dataset =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 32, 2, &pretokenized_tokenizer())
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
        let known_supervised_token_count = batch.supervised_token_count;
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
        assert_eq!(
            known_supervised_token_count,
            Some(mask.iter().filter(|value| **value != 0).count()),
            "streaming loader supervision metadata must match its emitted mask"
        );
        let supervised = masked_ruliad_target_text(wrapped.as_ref(), &targets, &mask);
        if mask.iter().all(|value| *value == 0) {
            saw_context_only_chunk = true;
        }
        if mask.contains(&1) {
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
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");

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
    let mut reset_count = 0usize;
    let mut previous_batch_reached_eos = true;
    for (step, batch) in loader.iter().take(20).enumerate() {
        if batch.reset_stream_state {
            assert!(
                step == 0 || previous_batch_reached_eos,
                "step {step} reset before the compact document reached EOS"
            );
            reset_count = reset_count.saturating_add(1);
        }
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
        assert!(
            targets.iter().any(|token| *token != 271),
            "step {step} should not train a padding-only stream chunk"
        );
        previous_batch_reached_eos = targets.contains(&271);
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
            if mask_row.contains(&1) {
                answer_rows = answer_rows.saturating_add(1);
                let window_u32 = window.iter().map(|token| *token as u32).collect::<Vec<_>>();
                let mut expected_mask = vec![0; 128];
                if ruliad_answer_target_loss_mask(&window_u32, &mut expected_mask) {
                    assert_eq!(
                        mask_row, expected_mask,
                        "step {step} row {row} answer mask should match the local streamed window"
                    );
                } else {
                    let has_local_answer_marker = window_u32
                        .windows(2)
                        .any(|pair| pair == [u32::from(b'!'), u32::from(b':')]);
                    assert!(
                        !has_local_answer_marker,
                        "step {step} row {row} masked answer continuation should not miss a local answer marker: {window:?}"
                    );
                }
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
        reset_count > 1,
        "compact source documents should complete within the bounded smoke"
    );
    assert!(
        answer_rows > 0,
        "profile-sized answer-completion stream should include natural answer rows"
    );
}

#[test]
fn live_ruliad_mixed_profile_sized_streaming_alternates_answer_and_full_masks() {
    type TestBackend = NdArray<f32>;

    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("ruliad-live.toml");
    let mut config = live_ruliad_runtime_config();
    config.serialization.document_tokens = 512;
    config.tokenization = RuliadTokenizationConfig::StructuredSymbolic {
        vocab_size: 272,
        eos_id: Some(271),
    };
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");

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
        mode: RuliadSupervisionMode::Mixed,
        mask_high_entropy_spans: false,
        ..Default::default()
    });
    let wrapped = Arc::new(crate::dataset::Dataset::from_universality(dataset));
    let device = burn::tensor::Device::<TestBackend>::default();
    let loader = crate::dataset::StreamingDataLoader::<TestBackend>::new(
        Arc::clone(&wrapped),
        DatasetSplit::Train,
        &device,
        4,
        Some(4),
        Some(512),
        1337,
    );
    let mut iter = loader.iter();
    let answer_batch = iter.next().expect("mixed answer-supervised batch");
    let full_batch = iter.next().expect("mixed full-document batch");

    let answer_mask = answer_batch
        .loss_mask
        .expect("mixed answer batch should expose a mask")
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .expect("answer mask");
    assert_eq!(answer_mask.len(), 32 * 128);
    assert!(
        answer_mask.contains(&0),
        "mixed answer step should not degrade into a full-document all-ones mask"
    );
    assert!(
        answer_mask.contains(&1),
        "mixed answer step should retain answer targets somewhere in the batch"
    );

    let full_inputs = full_batch
        .inputs
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .expect("full inputs");
    let full_mask = full_batch
        .loss_mask
        .expect("mixed full-document batch should expose an explicit mask")
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .expect("full mask");
    assert_eq!(full_mask.len(), 32 * 128);
    let eos_id = 271i64;
    let mut expected_valid_targets = Vec::with_capacity(32 * 128);
    for input_row in full_inputs.chunks(128) {
        let mut reached_padding = false;
        for input in input_row {
            if reached_padding || *input == eos_id {
                reached_padding = true;
                expected_valid_targets.push(0);
            } else {
                expected_valid_targets.push(1);
            }
        }
    }
    assert_eq!(
        full_mask, expected_valid_targets,
        "mixed full-document step should supervise every valid target and only mask padding"
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
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");

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
fn ruliad_consolidation_replays_primary_windows_and_policy_sidecars_together() {
    let tokenizer = TokenizerConfig {
        vocab_path: None,
        kind: TokenizerKind::Pretokenized(PretokenizedTokenizerConfig {
            vocab_size: 272,
            bos_id: None,
            eos_id: Some(271),
            pad_id: None,
            unk_id: None,
        }),
    };
    let mut supervision = crate::config::RuliadSupervisionConfig::default();
    supervision.consolidation = crate::config::RuliadConsolidationConfig {
        enabled: true,
        initial_unique_steps: 2,
        hold_steps: 10,
        novelty_interval_steps: 4,
        seed: 23,
    };
    let corpus_config = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../burn_dragon_p2p/deploy/profiles/ruliad-r3.semantic-action.corpus.toml");
    let dataset = UniversalityDataset::new_ruliad_on_the_fly(&corpus_config, 128, 8, &tokenizer)
        .expect("load production verifier corpus")
        .with_ruliad_supervision(supervision);
    let wrapped = crate::dataset::Dataset::from_universality(dataset);
    let replay_step = 2;
    let source_step = supervision
        .consolidation
        .coordinate(replay_step)
        .generation_step;
    assert!(source_step < supervision.consolidation.initial_unique_steps);

    let original_policy = TokenSequenceDataset::source_selected_ruliad_policy_batch(
        &wrapped,
        DatasetSplit::Train,
        0,
        source_step,
        8,
        2,
    )
    .expect("original policy panel");
    let replayed_policy = TokenSequenceDataset::source_selected_ruliad_policy_batch(
        &wrapped,
        DatasetSplit::Train,
        7,
        replay_step,
        8,
        2,
    )
    .expect("replayed policy panel");
    assert_eq!(original_policy.fingerprint(), replayed_policy.fingerprint());
    assert_eq!(
        replayed_policy.sampling_metadata,
        Some(crate::dataset::RuliadPolicySamplingMetadata {
            logical_epoch_index: 7,
            logical_selection_step: replay_step,
            generation_epoch_index: 0,
            generation_step: source_step,
            released_unique_steps: supervision.consolidation.initial_unique_steps,
            novel: false,
            consolidation_enabled: true,
        })
    );

    let original_windows = TokenSequenceDataset::source_selected_token_windows(
        &wrapped,
        DatasetSplit::Train,
        0,
        source_step,
        8,
        128,
    )
    .expect("original primary windows");
    let replayed_windows = TokenSequenceDataset::source_selected_token_windows(
        &wrapped,
        DatasetSplit::Train,
        7,
        replay_step,
        8,
        128,
    )
    .expect("replayed primary windows");
    assert_eq!(original_windows, replayed_windows);
}

#[test]
fn formal_r3_policy_batch_uses_structural_stop_and_verifiable_completion() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("ruliad-r3.toml");
    let mut config = fixed_ruliad_runtime_config();
    config.serialization.document_tokens = 8192;
    config.chunk_token_capacity = 16_384;
    config.tokenization = RuliadTokenizationConfig::StructuredSymbolic {
        vocab_size: 272,
        eos_id: Some(271),
    };
    config.families = burn_dragon_universality::ruliad::formal_ruliad_families();
    config.source_selection.enabled = true;
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");

    let dataset = UniversalityDataset::new_ruliad_on_the_fly(
        &config_path,
        64,
        1,
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
    let wrapped = crate::dataset::Dataset::from_universality(dataset);
    let batch = TokenSequenceDataset::source_selected_ruliad_policy_batch(
        &wrapped,
        DatasetSplit::Train,
        0,
        0,
        1,
        0,
    )
    .expect("policy batch");

    assert_eq!(
        batch.stop_token_id,
        Some(i64::from(RULIAD_SYMBOLIC_DOCUMENT_END_TOKEN))
    );
    let sample = batch.samples.first().expect("sample");
    assert_eq!(sample.item.document_close_marker(), "[/R3]");
    assert!(sample.item.prompt.starts_with("[R3 "));
    let completion = format!(
        "!:{}\n{}",
        sample.item.expected_answer,
        sample.item.document_close_marker()
    );
    let score = burn_dragon_universality::ruliad::score_ruliad_item_completion(
        &sample.item,
        Some(&completion),
    );
    assert!(
        score.verifier_match(),
        "R3 oracle completion must replay: {score:?}"
    );
    assert!(score.answer_terminated);
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
        candidate.capability_feedback_count = 1;
        candidate.capability_verifier_ema = 1.0;
        candidate.capability_completion_health_ema = 1.0;
        candidate.capability_schema_wrong_ema = 0.0;
        candidate.capability_malformed_ema = 0.0;
        candidate.capability_missing_ema = 0.0;
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

    let state =
        LiveSourceSelectionState::new(config.source_selection.clone(), config.clone(), candidates)
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
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");

    let dataset =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 32, 2, &pretokenized_tokenizer())
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
    fs::write(&config_path, toml::to_string_pretty(&config).expect("toml")).expect("write config");

    let dataset =
        UniversalityDataset::new_ruliad_on_the_fly(&config_path, 32, 2, &pretokenized_tokenizer())
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
