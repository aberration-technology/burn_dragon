#![recursion_limit = "256"]

//! Synthetic corpus generation focused on universality-style pre-pretraining.
//!
//! The initial implementation centers on Neural Cellular Automata-like rollouts
//! serialized into language-model token streams.

pub mod config;
pub mod generate;
pub mod manifest;
pub mod nca;
pub mod ruliad;
pub mod runtime;
pub mod stats;
pub mod tokenize;

pub mod api {
    //! Curated universality surface.

    pub mod config {
        pub use crate::config::{
            FloatRangeConfig, NcaComplexityBand, NcaComplexityMetric, NcaCorpusConfig,
            NcaFamilyConfig, NcaFamilyKind, NcaRuleFilterConfig, NcaSerializationConfig,
            NcaTokenizationConfig, UsizeRangeConfig, default_families,
            default_rule_filter_for_band, load_nca_config,
        };
        pub use crate::ruliad::config::{
            LeanMode, RULIAD_REQUIRED_MATH_DOMAINS, RULIAD_REQUIRED_REASONING_MODES,
            RuliadCorpusConfig, RuliadFamilyConfig, RuliadFamilyKind, RuliadMathDomain,
            RuliadReasoningMode, RuliadSerializationConfig, RuliadSourceSemantics, RuliadTaskKind,
            RuliadTokenizationConfig, default_ruliad_families, load_ruliad_config,
            ruliad_source_semantics,
        };
    }

    pub mod manifest {
        pub use crate::manifest::{
            CorpusKind, SampleSplit, UniversalityChunkManifest, UniversalityCorpusManifest,
            UniversalitySampleRecord, UniversalityTokenizerManifest, load_manifest,
        };
    }

    pub mod generate {
        pub use crate::generate::{GeneratedCorpusReport, generate_nca_corpus};
        pub use crate::ruliad::generate::{GeneratedRuliadCorpusReport, generate_ruliad_corpus};
    }

    pub mod runtime {
        pub use crate::ruliad::runtime::{
            OnlineRuliadCorpus, RuliadRuntimeSampleDocument, fixed_ruliad_document_token_count,
        };
        pub use crate::runtime::{
            OnlineNcaCorpus, RuntimeCorpusSummary, RuntimeSampleDocument,
            fixed_document_token_count,
        };
    }

    pub mod ruliad {
        pub use crate::ruliad::*;
    }

    pub mod stats {
        pub use crate::stats::{
            ComplexityHistogramBin, CorpusStats, SampleStats, build_complexity_histogram,
        };
    }

    pub mod expert {
        pub use crate::{config, generate, manifest, nca, ruliad, stats, tokenize};
    }
}

pub use config::{
    FloatRangeConfig, NcaComplexityBand, NcaComplexityMetric, NcaCorpusConfig, NcaFamilyConfig,
    NcaFamilyKind, NcaRuleFilterConfig, NcaSerializationConfig, NcaTokenizationConfig,
    UsizeRangeConfig, default_families, default_rule_filter_for_band, load_nca_config,
};
pub use generate::{GeneratedCorpusReport, generate_nca_corpus};
pub use manifest::{
    CorpusKind, SampleSplit, UniversalityChunkManifest, UniversalityCorpusManifest,
    UniversalitySampleRecord, UniversalityTokenizerManifest, load_manifest,
};
pub use ruliad::{
    GeneratedRuliadCorpusReport, LeanMode, LeanProofTask, OnlineRuliadCorpus,
    RULIAD_DIAGNOSTIC_REPORT_VERSION, RULIAD_EVAL_REPORT_VERSION, RULIAD_REQUIRED_MATH_DOMAINS,
    RULIAD_REQUIRED_REASONING_MODES, RULIAD_VERIFIER_VERSION, RuliadCategoricalPresentation,
    RuliadCategoryFunctor, RuliadCategoryMorphism, RuliadCompletionRecord, RuliadCorpusConfig,
    RuliadCountShare, RuliadDiagnosticReport, RuliadDiagnosticThresholds, RuliadEvalBaseline,
    RuliadEvalConfig, RuliadEvalFailure, RuliadEvalGroupScore, RuliadEvalItem, RuliadEvalReport,
    RuliadFamilyConfig, RuliadFamilyKind, RuliadFrontierSampler, RuliadMathDomain,
    RuliadMetricSnapshot, RuliadNaturalityCheck, RuliadReasoningMode, RuliadRuntimeSampleDocument,
    RuliadSampleSpec, RuliadSampleTelemetry, RuliadSamplerCandidate, RuliadSamplerConfig,
    RuliadSerializationConfig, RuliadSourceBucket, RuliadSourceBucketDiagnostic,
    RuliadSourceBucketId, RuliadSourceSelectionConfig, RuliadSourceSemantics, RuliadTaskKind,
    RuliadTokenizationConfig, RuliadVerificationReport, baseline_completions,
    build_eval_items_from_manifest, default_ruliad_families, diagnose_config, diagnose_manifest,
    evaluate_completions, extract_ruliad_answer, fixed_ruliad_document_token_count,
    generate_ruliad_corpus, load_proof_tasks, load_ruliad_config, plan_epoch_source_buckets,
    read_completion_records, ruliad_answers_exact_match, ruliad_answers_semantic_match,
    ruliad_categorical_presentation, ruliad_expected_answer, ruliad_prompt_prefix,
    ruliad_sampler_candidates, ruliad_source_buckets, ruliad_source_semantics, verify_manifest,
    verify_sample, write_completion_records_jsonl, write_eval_items_jsonl,
};
pub use runtime::{
    OnlineNcaCorpus, RuntimeCorpusSummary, RuntimeSampleDocument, fixed_document_token_count,
};
pub use stats::{ComplexityHistogramBin, CorpusStats, SampleStats, build_complexity_histogram};
