//! Hardware-neutral semantic identity for Ruliad generation and verification.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::ruliad::config::{RuliadCorpusConfig, load_ruliad_config};
use crate::ruliad::ir::RULIAD_IR_VERSION;
use crate::ruliad::kernel::RuliadKernelLimits;
use crate::ruliad::oracles::RULIAD_VERIFIER_VERSION;
use crate::ruliad::stable_json::{sha256_hex, stable_json_hash};

pub const RULIAD_SEMANTIC_CONTRACT_VERSION: u32 = 10;
pub const RULIAD_GENERATOR_SEMANTICS_ID: &str = "burn-dragon-ruliad-task-graph-generator-v9";
pub const RULIAD_KERNEL_SEMANTICS_ID: &str = "burn-dragon-ruliad-rewrite-kernel-v1";
pub const RULIAD_WIRE_SEMANTICS_ID: &str = "burn-dragon-ruliad-symbol-term-dag-wire-v3";
pub const RULIAD_SOURCE_SELECTION_SEMANTICS_ID: &str =
    "burn-dragon-ruliad-confidence-coverage-frontier-v5";

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadSemanticContract {
    pub version: u32,
    pub ir_version: u32,
    pub verifier_version: u32,
    pub generator_semantics_id: String,
    pub kernel_semantics_id: String,
    pub wire_semantics_id: String,
    pub source_selection_semantics_id: String,
    pub kernel_limits: RuliadKernelLimits,
    pub corpus: RuliadCorpusConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proof_tasks_sha256: Option<String>,
}

impl RuliadSemanticContract {
    pub fn from_config_path(path: &Path) -> Result<Self> {
        let config = load_ruliad_config(path)?;
        Self::from_config(&config, path.parent())
    }

    pub fn from_config(config: &RuliadCorpusConfig, base_dir: Option<&Path>) -> Result<Self> {
        config.validate()?;
        let proof_tasks_sha256 = config
            .proof_tasks
            .as_deref()
            .map(|path| resolve_input_path(path, base_dir))
            .map(|path| {
                fs::read(&path)
                    .with_context(|| format!("read Ruliad proof tasks {}", path.display()))
                    .map(|bytes| sha256_hex(&bytes))
            })
            .transpose()?;
        let mut corpus = config.clone();
        corpus.output_dir = PathBuf::new();
        corpus.name.clear();
        corpus.chunk_token_capacity = 0;
        corpus.serialization.preview_samples = 0;
        corpus.proof_tasks = None;
        Ok(Self {
            version: RULIAD_SEMANTIC_CONTRACT_VERSION,
            ir_version: RULIAD_IR_VERSION,
            verifier_version: RULIAD_VERIFIER_VERSION,
            generator_semantics_id: RULIAD_GENERATOR_SEMANTICS_ID.to_string(),
            kernel_semantics_id: RULIAD_KERNEL_SEMANTICS_ID.to_string(),
            wire_semantics_id: RULIAD_WIRE_SEMANTICS_ID.to_string(),
            source_selection_semantics_id: RULIAD_SOURCE_SELECTION_SEMANTICS_ID.to_string(),
            kernel_limits: RuliadKernelLimits::default(),
            corpus,
            proof_tasks_sha256,
        })
    }

    pub fn canonical_hash(&self) -> Result<String> {
        stable_json_hash(self)
    }
}

fn resolve_input_path(path: &Path, base_dir: Option<&Path>) -> PathBuf {
    if path.is_absolute() || path.exists() {
        return path.to_path_buf();
    }
    base_dir
        .map(|base_dir| base_dir.join(path))
        .unwrap_or_else(|| path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ruliad::config::{
        RuliadCorpusConfig, RuliadSerializationConfig, RuliadSourceSelectionConfig,
        RuliadTokenizationConfig, default_ruliad_families,
    };

    fn test_config() -> RuliadCorpusConfig {
        RuliadCorpusConfig {
            output_dir: "target/ruliad-contract-test".into(),
            seed: 1,
            name: "ruliad-contract-test".to_string(),
            train_samples: 8,
            validation_samples: 2,
            chunk_token_capacity: 4096,
            serialization: RuliadSerializationConfig::default(),
            tokenization: RuliadTokenizationConfig::default(),
            formal_generalization: Default::default(),
            source_selection: RuliadSourceSelectionConfig::default(),
            families: default_ruliad_families(),
            proof_tasks: None,
            lean_task_limit: None,
        }
    }

    #[test]
    fn non_semantic_output_settings_do_not_change_contract_identity() {
        let left = test_config();
        let mut right = left.clone();
        right.output_dir = "somewhere-else".into();
        right.name = "operator-label".to_string();
        right.chunk_token_capacity = right.chunk_token_capacity.saturating_mul(2);
        right.serialization.preview_samples = 99;
        let left = RuliadSemanticContract::from_config(&left, None).expect("left contract");
        let right = RuliadSemanticContract::from_config(&right, None).expect("right contract");
        assert_eq!(
            left.canonical_hash().expect("left hash"),
            right.canonical_hash().expect("right hash")
        );
    }

    #[test]
    fn generation_semantics_change_contract_identity() {
        let left = test_config();
        let mut right = left.clone();
        right.seed = right.seed.wrapping_add(1);
        let left = RuliadSemanticContract::from_config(&left, None).expect("left contract");
        let right = RuliadSemanticContract::from_config(&right, None).expect("right contract");
        assert_ne!(
            left.canonical_hash().expect("left hash"),
            right.canonical_hash().expect("right hash")
        );
    }

    #[test]
    fn structural_holdout_is_bound_into_contract_identity() {
        let left = test_config();
        let mut right = left.clone();
        right.formal_generalization =
            crate::ruliad::config::RuliadFormalGeneralizationContract::StructuralHoldoutV1;
        let left = RuliadSemanticContract::from_config(&left, None).expect("left contract");
        let right = RuliadSemanticContract::from_config(&right, None).expect("right contract");
        assert_ne!(
            left.canonical_hash().expect("left hash"),
            right.canonical_hash().expect("right hash")
        );
    }

    #[test]
    fn training_grammar_control_has_a_distinct_semantic_contract() {
        use crate::ruliad::config::RuliadFormalGeneralizationContract;
        let mut config = test_config();
        let mut identities = std::collections::HashSet::new();
        for contract in [
            RuliadFormalGeneralizationContract::SeedDisjointV1,
            RuliadFormalGeneralizationContract::StructuralHoldoutV1,
            RuliadFormalGeneralizationContract::StructuralTrainSeedDisjointV1,
        ] {
            config.formal_generalization = contract;
            identities.insert(
                RuliadSemanticContract::from_config(&config, None)
                    .unwrap()
                    .canonical_hash()
                    .unwrap(),
            );
        }
        assert_eq!(identities.len(), 3);
    }

    #[test]
    fn proof_action_answer_contract_is_bound_into_semantic_identity() {
        let left = test_config();
        let mut right = left.clone();
        right
            .source_selection
            .formal_task_mix
            .proof_action_answer_contract =
            crate::ruliad::config::RuliadProofActionAnswerContract::SemanticStep;
        let left = RuliadSemanticContract::from_config(&left, None).expect("left contract");
        let right = RuliadSemanticContract::from_config(&right, None).expect("right contract");
        assert_ne!(
            left.canonical_hash().expect("left hash"),
            right.canonical_hash().expect("right hash")
        );
    }
}
