//! Verifier-backed ruliad source for bounded computable artifacts.
//!
//! This module extends the original NCA-focused universality source with a
//! heterogeneous stream of exact finite rule systems, simulations, proof tasks,
//! and canaries. The hot path remains Rust-native; Lean is only an optional
//! external trust anchor during explicit verification.
//!
//! The ruliad profile is a trace-pretraining source: generated documents are
//! compact next-token sequences with verifier-backed question/proof/answer
//! slots. Live source selection is a curriculum policy over source buckets, not
//! a long-rollout reinforcement objective.

pub mod category;
pub mod config;
pub mod eca;
pub mod eval;
pub mod generate;
pub mod metrics;
pub mod oracles;
pub mod rng;
pub mod runtime;
pub mod search;
pub mod source_selection;
pub mod stable_json;
pub mod tokenize;
pub mod verification;

#[cfg(feature = "cli")]
pub mod cli;

pub use category::{RuliadCategoryFunctor, RuliadCategoryMorphism, RuliadNaturalityCheck};
pub use config::{
    LeanMode, RULIAD_REQUIRED_MATH_DOMAINS, RULIAD_REQUIRED_REASONING_MODES, RuliadCorpusConfig,
    RuliadDocumentMode, RuliadFamilyConfig, RuliadFamilyKind, RuliadFrontierExtensionConfig,
    RuliadMathDomain, RuliadReasoningMode, RuliadSerializationConfig,
    RuliadSourceSelectionColdStartConfig, RuliadSourceSelectionConfig, RuliadSourceSemantics,
    RuliadTaskKind, RuliadTokenizationConfig, compact_ruliad_families, default_ruliad_families,
    load_ruliad_config, ruliad_source_semantics,
};
pub use eval::{
    RULIAD_DIAGNOSTIC_REPORT_VERSION, RULIAD_EVAL_REPORT_VERSION, RULIAD_REASONING_SCORE_VERSION,
    RULIAD_VERIFIER_REWARD_VECTOR_DIM, RuliadAnswerKeyAlignment, RuliadAnswerStatus,
    RuliadCompletionRecord, RuliadCountShare, RuliadDiagnosticReport, RuliadDiagnosticThresholds,
    RuliadEvalBaseline, RuliadEvalConfig, RuliadEvalFailure, RuliadEvalGroupScore, RuliadEvalItem,
    RuliadEvalReport, RuliadExtractedCompletion, RuliadReasoningScore, RuliadReasoningScoreKey,
    RuliadSourceBucketDiagnostic, RuliadVerifierRewardVector, RuliadVerifierRewardWeights,
    baseline_completions, build_eval_items_from_manifest, centered_advantages, diagnose_config,
    diagnose_manifest, evaluate_completions, extract_ruliad_answer, extract_ruliad_completion,
    normalized_advantages, read_completion_records, ruliad_answer_key_alignment,
    ruliad_answers_exact_match, ruliad_answers_semantic_match, ruliad_reasoning_rank_fitness,
    ruliad_reasoning_rank_order, ruliad_verifier_reward, ruliad_verifier_reward_vector,
    ruliad_vpo_independent_utilities, score_ruliad_answer, score_ruliad_completion,
    score_ruliad_item_completion, write_completion_records_jsonl, write_eval_items_jsonl,
};
pub use generate::{GeneratedRuliadCorpusReport, generate_ruliad_corpus};
pub use metrics::{
    RuliadBucketMetric, RuliadCapabilityFeedback, RuliadGroupMetric, RuliadMetricSnapshot,
    RuliadSampleTelemetry,
};
pub use oracles::{
    GeneratedRuliadSample, LeanProofTask, RULIAD_VERIFIER_VERSION, RuliadCategoricalPresentation,
    RuliadSampleSpec, load_proof_tasks, ruliad_answer_contract, ruliad_answer_values,
    ruliad_categorical_presentation, ruliad_expected_answer, ruliad_prompt_prefix,
};
pub use runtime::{
    OnlineRuliadCorpus, RuliadRuntimeSampleDocument, fixed_ruliad_document_token_count,
};
pub use search::{
    RuliadFrontierSampler, RuliadFrontierSamplerState, RuliadSamplerCandidate, RuliadSamplerConfig,
};
pub use source_selection::{
    RuliadEpochSourcePlan, RuliadSourceBucket, RuliadSourceBucketId, plan_epoch_source_buckets,
    ruliad_sampler_candidates, ruliad_sampler_candidates_for_difficulty,
    ruliad_source_bucket_by_label, ruliad_source_buckets, ruliad_source_buckets_for_difficulty,
};
pub use verification::{RuliadVerificationReport, verify_manifest, verify_sample};
