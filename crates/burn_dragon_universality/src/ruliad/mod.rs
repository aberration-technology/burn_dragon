//! Verifier-backed ruliad source for bounded computable artifacts.
//!
//! This module extends the original NCA-focused universality source with a
//! heterogeneous stream of exact finite rule systems, simulations, proof tasks,
//! and canaries. The hot path remains Rust-native; Lean is only an optional
//! external trust anchor during explicit verification.

pub mod config;
pub mod eca;
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

pub use config::{
    LeanMode, RULIAD_REQUIRED_MATH_DOMAINS, RULIAD_REQUIRED_REASONING_MODES, RuliadCorpusConfig,
    RuliadFamilyConfig, RuliadFamilyKind, RuliadMathDomain, RuliadReasoningMode,
    RuliadSerializationConfig, RuliadSourceSelectionConfig, RuliadSourceSemantics, RuliadTaskKind,
    RuliadTokenizationConfig, default_ruliad_families, load_ruliad_config, ruliad_source_semantics,
};
pub use generate::{GeneratedRuliadCorpusReport, generate_ruliad_corpus};
pub use metrics::{RuliadMetricSnapshot, RuliadSampleTelemetry};
pub use oracles::{
    GeneratedRuliadSample, LeanProofTask, RULIAD_VERIFIER_VERSION, RuliadCategoricalPresentation,
    RuliadSampleSpec, load_proof_tasks, ruliad_categorical_presentation,
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
