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
pub mod contract;
pub mod eca;
pub mod eval;
pub mod formal;
pub mod generate;
pub mod ir;
pub mod kernel;
pub mod lean;
pub mod metrics;
pub mod oracles;
pub mod policy;
pub mod rng;
pub mod runtime;
pub mod search;
pub mod source_selection;
pub mod stable_json;
pub mod supervision;
pub mod tokenize;
pub mod verification;
pub mod wire;
pub mod world;

#[cfg(feature = "cli")]
pub mod cli;

pub use category::{RuliadCategoryFunctor, RuliadCategoryMorphism, RuliadNaturalityCheck};
pub use config::{
    LeanMode, RULIAD_REQUIRED_MATH_DOMAINS, RULIAD_REQUIRED_REASONING_MODES, RuliadCorpusConfig,
    RuliadDocumentMode, RuliadFamilyConfig, RuliadFamilyKind, RuliadFormalGeneralizationContract,
    RuliadFormalTaskMixConfig, RuliadFrontierExtensionConfig, RuliadMathDomain,
    RuliadProofActionAnswerContract, RuliadReasoningMode, RuliadSerializationConfig,
    RuliadSourceSelectionColdStartConfig, RuliadSourceSelectionConfig, RuliadSourceSemantics,
    RuliadTaskKind, RuliadTokenizationConfig, compact_ruliad_families, default_ruliad_families,
    formal_ruliad_families, load_ruliad_config, ruliad_source_semantics,
};
pub use contract::{
    RULIAD_GENERATOR_SEMANTICS_ID, RULIAD_KERNEL_SEMANTICS_ID, RULIAD_SEMANTIC_CONTRACT_VERSION,
    RULIAD_SOURCE_SELECTION_SEMANTICS_ID, RULIAD_WIRE_SEMANTICS_ID, RuliadSemanticContract,
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
    ruliad_answers_exact_match, ruliad_answers_semantic_match, ruliad_presented_action_answers,
    ruliad_presented_action_match, ruliad_reasoning_rank_fitness, ruliad_reasoning_rank_order,
    ruliad_verifier_reward, ruliad_verifier_reward_vector, ruliad_vpo_independent_utilities,
    score_ruliad_answer, score_ruliad_completion, score_ruliad_item_completion,
    write_completion_records_jsonl, write_eval_items_jsonl,
};
pub use formal::{RuliadFormalGenerationSplit, RuliadFormalGeneratorConfig};
pub use generate::{GeneratedRuliadCorpusReport, generate_ruliad_corpus};
pub use ir::{
    RULIAD_IR_VERSION, RuliadComplexityVector, RuliadEquality, RuliadFormalDomain,
    RuliadGoalCertificate, RuliadProofBundle, RuliadProofCertificate, RuliadProofGoal,
    RuliadProofProblem, RuliadProofSource, RuliadProofStep, RuliadRewriteAxiom,
    RuliadRewriteDirection, RuliadTerm,
};
pub use kernel::{
    RuliadGoalTransitionKernel, RuliadKernelFailure, RuliadKernelFailureKind, RuliadKernelLimits,
    RuliadReplayReport, complexity_vector, replay_certificate, replay_goal_prefix,
    validate_problem,
};
pub use lean::{
    RULIAD_LEAN_CHECKER_VERSION, RULIAD_LEAN_PANEL_CONTRACT, RuliadLeanPanelReport,
    RuliadLeanVerificationReport, render_lean_verification_module,
};
pub use metrics::{
    RULIAD_SOURCE_CAPABILITY_LABEL_PREFIX, RuliadBucketMetric, RuliadCapabilityCoverageMetric,
    RuliadCapabilityFeedback, RuliadGroupMetric, RuliadMetricSnapshot, RuliadSampleTelemetry,
    ruliad_source_capability_label,
};
pub use oracles::{
    GeneratedRuliadSample, LeanProofTask, RULIAD_V2_DOCUMENT_CLOSE_MARKER,
    RULIAD_V3_DOCUMENT_CLOSE_MARKER, RULIAD_VERIFIER_VERSION, RuliadCategoricalPresentation,
    RuliadSampleSpec, load_proof_tasks, ruliad_answer_contract, ruliad_answer_values,
    ruliad_categorical_presentation, ruliad_document_close_marker, ruliad_expected_answer,
    ruliad_prompt_prefix, ruliad_proof_action_exact_prompt, ruliad_proof_action_local_prompt,
    ruliad_proof_action_prompt, ruliad_proof_action_query, ruliad_sample_math_domains,
    ruliad_sample_reasoning_modes,
};
pub use policy::{
    DEFAULT_PROOF_ACTION_CANDIDATES, RuliadProofActionCandidate, RuliadProofActionSet,
    RuliadProofPolicyState, RuliadProofRolloutReport, counterfactual_proof_action_target,
    oracle_proof_action_set, parse_proof_action_index, proof_action_answer,
    proof_action_answer_for_semantic_index, resolve_proof_action_answer, rollout_proof_policy,
    ruliad_term_distance,
};
pub use runtime::{
    OnlineRuliadCorpus, RULIAD_SUPERVISION_AUDIT_VERSION, RuliadFrontierFeasibilityReport,
    RuliadFrontierFeasibilitySample, RuliadRuntimeSampleDocument, RuliadSupervisionAuditBucket,
    RuliadSupervisionAuditReport, fixed_ruliad_document_token_count,
};
pub use search::{
    RuliadFrontierSampler, RuliadFrontierSamplerState, RuliadSamplerCandidate, RuliadSamplerConfig,
};
pub use source_selection::{
    RuliadEpochSourcePlan, RuliadSourceBucket, RuliadSourceBucketId, plan_epoch_source_buckets,
    ruliad_sampler_candidates, ruliad_sampler_candidates_for_difficulty,
    ruliad_source_bucket_by_label, ruliad_source_buckets, ruliad_source_buckets_for_difficulty,
};
pub use supervision::{
    RuliadTokenSupervisionConfig, RuliadTokenSupervisionMode, ruliad_token_loss_mask,
};
pub use verification::{
    RuliadVerificationReport, verify_formal_panel, verify_manifest, verify_sample,
};
pub use wire::{
    RuliadModelCertificatePrefix, decode_certificate, decode_model_certificate,
    decode_model_certificate_prefix, decode_model_proof_step, decode_problem, encode_certificate,
    encode_model_certificate, encode_model_proof_step, encode_problem,
};
pub use world::{
    BernoulliEvidence, RULIAD_TASK_GRAPH_CONTRACT_VERSION, RuliadCapabilityCoverage,
    RuliadCapabilityMasteryThresholds, RuliadCapabilityPosterior, RuliadDifficultyVector,
    RuliadFormalProofWorld, RuliadTaskGraph, RuliadTransitionCost, RuliadTransitionResult,
    RuliadVerifiedTransition, RuliadWorldDescriptor,
};
