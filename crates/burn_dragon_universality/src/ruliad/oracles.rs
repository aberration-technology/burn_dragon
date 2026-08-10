use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::manifest::SampleSplit;
use crate::ruliad::category::{
    RuliadCategoryFunctor, RuliadCategoryMorphism, RuliadNaturalityCheck, compose_path,
    generate_category_fields, naturality_commutes, valid_finite_category, valid_functor,
};
use crate::ruliad::config::{
    RuliadCorpusConfig, RuliadFamilyConfig, RuliadFamilyKind, RuliadFormalGeneralizationContract,
    RuliadFormalTaskMixConfig, RuliadMathDomain, RuliadProofActionAnswerContract,
    RuliadReasoningMode, RuliadTaskKind, ruliad_source_semantics,
};
use crate::ruliad::eca;
use crate::ruliad::formal::{
    RuliadFormalGenerationSplit, RuliadFormalGeneratorConfig, corrupt_formal_certificate,
    generate_formal_bundle,
};
use crate::ruliad::ir::{
    RuliadFormalDomain, RuliadProofCertificate, RuliadProofProblem, RuliadProofSource,
    RuliadRewriteDirection, RuliadTerm,
};
use crate::ruliad::kernel::{
    RuliadKernelLimits, complexity_vector, replay_certificate, replay_goal_prefix,
};
use crate::ruliad::rng::{SplitMix64, mix_seed};
use crate::ruliad::source_selection::RuliadSourceBucket;
use crate::ruliad::stable_json::{sha256_hex, stable_json_hash};
use crate::ruliad::wire::{encode_certificate, encode_model_certificate, encode_problem};
use crate::stats::SampleStats;

pub const RULIAD_VERIFIER_VERSION: u32 = 9;

const TRAIN_SPLIT_TAG: u64 = 0xA11C_E5ED_D15C_A11A;
const VAL_SPLIT_TAG: u64 = 0xBADC_0FFE_E5E1_7A1D;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct LeanProofTask {
    pub id: String,
    pub statement: String,
    pub proof: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_hash: Option<String>,
}

impl LeanProofTask {
    pub fn computed_payload_hash(&self) -> String {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(self.id.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(self.statement.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(self.proof.as_bytes());
        sha256_hex(&bytes)
    }

    pub fn validate_hash(&self) -> bool {
        self.payload_hash
            .as_deref()
            .is_none_or(|expected| expected == self.computed_payload_hash())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadRewriteRule {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuliadAlgebraLaw {
    Associativity,
    Commutativity,
}

impl RuliadAlgebraLaw {
    pub fn label(self) -> &'static str {
        match self {
            Self::Associativity => "associativity",
            Self::Commutativity => "commutativity",
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuliadSampleSpec {
    Eca {
        rule: u8,
        width: usize,
        steps: usize,
        initial: String,
        trace: Vec<String>,
        task: RuliadTaskKind,
    },
    Simulation {
        source_rule: u8,
        target_rule: u8,
        width: usize,
        steps: usize,
        source_initial: String,
        target_initial: String,
        source_trace: Vec<String>,
        target_trace: Vec<String>,
        mapped_source_trace: Vec<String>,
        task: RuliadTaskKind,
    },
    Automaton {
        state_count: usize,
        transitions: Vec<Vec<usize>>,
        start_state: usize,
        accept_states: Vec<usize>,
        input: String,
        trace: Vec<usize>,
        accepted: bool,
        task: RuliadTaskKind,
    },
    Rewrite {
        alphabet: String,
        rules: Vec<RuliadRewriteRule>,
        initial: String,
        steps: usize,
        trace: Vec<String>,
        normal_form: String,
        task: RuliadTaskKind,
    },
    Algebra {
        carrier_size: usize,
        operation_table: Vec<Vec<usize>>,
        law: RuliadAlgebraLaw,
        operands: Vec<usize>,
        lhs: usize,
        rhs: usize,
        holds: bool,
        task: RuliadTaskKind,
    },
    Category {
        object_count: usize,
        morphisms: Vec<RuliadCategoryMorphism>,
        identities: Vec<usize>,
        composition: Vec<Vec<Option<usize>>>,
        path: Vec<usize>,
        composed: usize,
        lhs: usize,
        rhs: usize,
        holds: bool,
        proof_steps: Vec<String>,
        functor: Option<RuliadCategoryFunctor>,
        naturality: Option<RuliadNaturalityCheck>,
        task: RuliadTaskKind,
    },
    ProofTree {
        modulus: usize,
        u: [usize; 2],
        v: [usize; 2],
        sum: [usize; 2],
        dot: usize,
        norm_u: usize,
        norm_v: usize,
        norm_sum: usize,
        lhs: usize,
        rhs: usize,
        holds: bool,
        lemmas: Vec<String>,
        proof_steps: Vec<String>,
        task: RuliadTaskKind,
    },
    FormalProof {
        problem: RuliadProofProblem,
        certificate: RuliadProofCertificate,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        candidate: Option<RuliadProofCertificate>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        proof_step_index: Option<usize>,
        /// Cyclic action-menu presentation used by `select_proof_action` documents.
        /// Missing values preserve the version-7 canonical presentation.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        action_presentation_rotation: Option<usize>,
        #[serde(default)]
        action_answer_contract: RuliadProofActionAnswerContract,
        task: RuliadTaskKind,
    },
    LeanTask {
        task_id: String,
        statement: String,
        proof: String,
        payload_hash: String,
        task: RuliadTaskKind,
    },
    HashNoise {
        bytes_hex: String,
        payload_hash: String,
        task: RuliadTaskKind,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadCategoricalPresentation {
    pub abstraction: String,
    pub source_family: String,
    pub task_kind: String,
    pub presentation: String,
    pub objects: Vec<String>,
    pub morphisms: Vec<String>,
    pub functors: Vec<String>,
    pub laws: Vec<String>,
    pub query: String,
    pub answer: String,
    pub categorical_core: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedRuliadSample {
    pub spec: RuliadSampleSpec,
    pub categorical_presentation: RuliadCategoricalPresentation,
    pub family: RuliadFamilyKind,
    pub task_kind: RuliadTaskKind,
    pub verifier_version: u32,
    pub oracle_hash: String,
    pub text: String,
    pub stats: SampleStats,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuliadOracleReport {
    pub ok: bool,
    pub family: RuliadFamilyKind,
    pub task_kind: RuliadTaskKind,
    pub oracle_hash: String,
}

mod classification;
mod document;
mod families;
mod generation;
mod statistics;
mod verifier;

use classification::*;
use document::*;
use families::*;
use statistics::*;

#[cfg(test)]
use generation::*;

pub(crate) use classification::scale_family_for_difficulty;
pub use classification::{ruliad_sample_math_domains, ruliad_sample_reasoning_modes};
pub use document::{
    RULIAD_V2_DOCUMENT_CLOSE_MARKER, RULIAD_V3_DOCUMENT_CLOSE_MARKER, compact_ruliad_label,
    ruliad_answer_contract, ruliad_answer_values, ruliad_document_close_marker,
    ruliad_expected_answer, ruliad_prompt_prefix, ruliad_proof_action_prompt,
    ruliad_proof_action_query, sample_text,
};
pub use generation::{
    default_proof_tasks, generate_sample, generate_sample_for_source_bucket, load_proof_tasks,
    ruliad_categorical_presentation,
};
pub(crate) use statistics::is_degenerate_spec;
pub use verifier::verify_spec;

#[cfg(test)]
mod tests;
