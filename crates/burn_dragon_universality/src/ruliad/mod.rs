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
    RuliadFamilyConfig, RuliadFamilyKind, RuliadMathDomain, RuliadReasoningMode,
    RuliadSerializationConfig, RuliadSourceSelectionConfig, RuliadSourceSemantics, RuliadTaskKind,
    RuliadTokenizationConfig, compact_ruliad_families, default_ruliad_families, load_ruliad_config,
    ruliad_source_semantics,
};
pub use eval::{
    RULIAD_DIAGNOSTIC_REPORT_VERSION, RULIAD_EVAL_REPORT_VERSION, RuliadCompletionRecord,
    RuliadCountShare, RuliadDiagnosticReport, RuliadDiagnosticThresholds, RuliadEvalBaseline,
    RuliadEvalConfig, RuliadEvalFailure, RuliadEvalGroupScore, RuliadEvalItem, RuliadEvalReport,
    RuliadSourceBucketDiagnostic, baseline_completions, build_eval_items_from_manifest,
    diagnose_config, diagnose_manifest, evaluate_completions, extract_ruliad_answer,
    read_completion_records, ruliad_answers_exact_match, ruliad_answers_semantic_match,
    write_completion_records_jsonl, write_eval_items_jsonl,
};
pub use generate::{GeneratedRuliadCorpusReport, generate_ruliad_corpus};
pub use metrics::{RuliadMetricSnapshot, RuliadSampleTelemetry};
pub use oracles::{
    GeneratedRuliadSample, LeanProofTask, RULIAD_VERIFIER_VERSION, RuliadCategoricalPresentation,
    RuliadSampleSpec, load_proof_tasks, ruliad_categorical_presentation, ruliad_expected_answer,
    ruliad_prompt_prefix,
};
pub use runtime::{
    OnlineRuliadCorpus, RuliadRuntimeSampleDocument, fixed_ruliad_document_token_count,
};
pub use search::{RuliadFrontierSampler, RuliadSamplerCandidate, RuliadSamplerConfig};
pub use source_selection::{
    RuliadEpochSourcePlan, RuliadSourceBucket, RuliadSourceBucketId, plan_epoch_source_buckets,
    ruliad_sampler_candidates, ruliad_source_buckets,
};
pub use verification::{RuliadVerificationReport, verify_manifest, verify_sample};
