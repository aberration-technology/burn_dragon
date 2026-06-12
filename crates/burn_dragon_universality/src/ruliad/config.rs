use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::config::UsizeRangeConfig;
use crate::ruliad::search::RuliadSamplerConfig;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum LeanMode {
    #[default]
    Off,
    Optional,
    Required,
}

impl std::str::FromStr for LeanMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "off" => Ok(Self::Off),
            "optional" => Ok(Self::Optional),
            "required" => Ok(Self::Required),
            other => Err(anyhow!(
                "invalid lean mode `{other}`; expected off, optional, or required"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum RuliadFamilyKind {
    #[default]
    Eca,
    Simulation,
    Automaton,
    Rewrite,
    Algebra,
    Category,
    LeanTask,
    HashNoise,
}

impl RuliadFamilyKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Eca => "eca",
            Self::Simulation => "simulation",
            Self::Automaton => "automaton",
            Self::Rewrite => "rewrite",
            Self::Algebra => "algebra",
            Self::Category => "category",
            Self::LeanTask => "lean_task",
            Self::HashNoise => "hash_noise",
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum RuliadTaskKind {
    #[default]
    NextState,
    MultiStepState,
    VerifySimulation,
    EvaluateAutomaton,
    RewriteNormalForm,
    CheckAlgebraLaw,
    ComposeCategoryPath,
    VerifyCategoryLaw,
    VerifyFunctorPreservation,
    VerifyNaturalitySquare,
    CompleteProof,
    HashCanary,
}

impl RuliadTaskKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::NextState => "next_state",
            Self::MultiStepState => "multi_step_state",
            Self::VerifySimulation => "verify_simulation",
            Self::EvaluateAutomaton => "evaluate_automaton",
            Self::RewriteNormalForm => "rewrite_normal_form",
            Self::CheckAlgebraLaw => "check_algebra_law",
            Self::ComposeCategoryPath => "compose_category_path",
            Self::VerifyCategoryLaw => "verify_category_law",
            Self::VerifyFunctorPreservation => "verify_functor_preservation",
            Self::VerifyNaturalitySquare => "verify_naturality_square",
            Self::CompleteProof => "complete_proof",
            Self::HashCanary => "hash_canary",
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum RuliadMathDomain {
    DiscreteDynamics,
    ComputationTheory,
    SymbolicRewriting,
    UniversalAlgebra,
    CategoryTheory,
    FormalProof,
    InformationTheory,
}

impl RuliadMathDomain {
    pub fn label(self) -> &'static str {
        match self {
            Self::DiscreteDynamics => "discrete_dynamics",
            Self::ComputationTheory => "computation_theory",
            Self::SymbolicRewriting => "symbolic_rewriting",
            Self::UniversalAlgebra => "universal_algebra",
            Self::CategoryTheory => "category_theory",
            Self::FormalProof => "formal_proof",
            Self::InformationTheory => "information_theory",
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum RuliadReasoningMode {
    LocalRuleEvaluation,
    IteratedDynamics,
    StateMachineExecution,
    SimulationEquivalence,
    StructurePreservation,
    Normalization,
    EquationalReasoning,
    CounterexampleEvaluation,
    CompositionalReasoning,
    Associativity,
    FormalDeduction,
    EntropyCanary,
}

impl RuliadReasoningMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::LocalRuleEvaluation => "local_rule_evaluation",
            Self::IteratedDynamics => "iterated_dynamics",
            Self::StateMachineExecution => "state_machine_execution",
            Self::SimulationEquivalence => "simulation_equivalence",
            Self::StructurePreservation => "structure_preservation",
            Self::Normalization => "normalization",
            Self::EquationalReasoning => "equational_reasoning",
            Self::CounterexampleEvaluation => "counterexample_evaluation",
            Self::CompositionalReasoning => "compositional_reasoning",
            Self::Associativity => "associativity",
            Self::FormalDeduction => "formal_deduction",
            Self::EntropyCanary => "entropy_canary",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuliadSourceSemantics {
    pub math_domains: &'static [RuliadMathDomain],
    pub reasoning_modes: &'static [RuliadReasoningMode],
    pub description: &'static str,
}

pub const RULIAD_REQUIRED_MATH_DOMAINS: &[RuliadMathDomain] = &[
    RuliadMathDomain::DiscreteDynamics,
    RuliadMathDomain::ComputationTheory,
    RuliadMathDomain::SymbolicRewriting,
    RuliadMathDomain::UniversalAlgebra,
    RuliadMathDomain::CategoryTheory,
    RuliadMathDomain::FormalProof,
    RuliadMathDomain::InformationTheory,
];

pub const RULIAD_REQUIRED_REASONING_MODES: &[RuliadReasoningMode] = &[
    RuliadReasoningMode::LocalRuleEvaluation,
    RuliadReasoningMode::IteratedDynamics,
    RuliadReasoningMode::StateMachineExecution,
    RuliadReasoningMode::SimulationEquivalence,
    RuliadReasoningMode::StructurePreservation,
    RuliadReasoningMode::Normalization,
    RuliadReasoningMode::EquationalReasoning,
    RuliadReasoningMode::CounterexampleEvaluation,
    RuliadReasoningMode::CompositionalReasoning,
    RuliadReasoningMode::Associativity,
    RuliadReasoningMode::FormalDeduction,
    RuliadReasoningMode::EntropyCanary,
];

pub fn ruliad_source_semantics(
    family: RuliadFamilyKind,
    task_kind: RuliadTaskKind,
) -> RuliadSourceSemantics {
    use RuliadFamilyKind as Family;
    use RuliadMathDomain as Domain;
    use RuliadReasoningMode as Mode;
    use RuliadTaskKind as Task;

    match (family, task_kind) {
        (Family::Eca, Task::NextState) => RuliadSourceSemantics {
            math_domains: &[Domain::DiscreteDynamics, Domain::ComputationTheory],
            reasoning_modes: &[Mode::LocalRuleEvaluation],
            description: "one-step evaluation of a finite local rule",
        },
        (Family::Eca, Task::MultiStepState) => RuliadSourceSemantics {
            math_domains: &[Domain::DiscreteDynamics, Domain::ComputationTheory],
            reasoning_modes: &[Mode::LocalRuleEvaluation, Mode::IteratedDynamics],
            description: "bounded rollout of a finite dynamical system",
        },
        (Family::Simulation, Task::VerifySimulation) => RuliadSourceSemantics {
            math_domains: &[
                Domain::DiscreteDynamics,
                Domain::ComputationTheory,
                Domain::CategoryTheory,
            ],
            reasoning_modes: &[
                Mode::SimulationEquivalence,
                Mode::StructurePreservation,
                Mode::CompositionalReasoning,
            ],
            description: "verification that a map commutes with bounded dynamics",
        },
        (Family::Automaton, Task::EvaluateAutomaton) => RuliadSourceSemantics {
            math_domains: &[Domain::ComputationTheory],
            reasoning_modes: &[Mode::StateMachineExecution, Mode::CounterexampleEvaluation],
            description: "finite automaton execution and acceptance evaluation",
        },
        (Family::Rewrite, Task::RewriteNormalForm) => RuliadSourceSemantics {
            math_domains: &[Domain::SymbolicRewriting, Domain::ComputationTheory],
            reasoning_modes: &[Mode::Normalization, Mode::IteratedDynamics],
            description: "terminating symbolic rewrite search toward a normal form",
        },
        (Family::Algebra, Task::CheckAlgebraLaw) => RuliadSourceSemantics {
            math_domains: &[Domain::UniversalAlgebra],
            reasoning_modes: &[
                Mode::EquationalReasoning,
                Mode::Associativity,
                Mode::CounterexampleEvaluation,
            ],
            description: "finite operation-table evaluation of algebraic laws",
        },
        (Family::Category, Task::ComposeCategoryPath) => RuliadSourceSemantics {
            math_domains: &[Domain::CategoryTheory],
            reasoning_modes: &[
                Mode::CompositionalReasoning,
                Mode::Associativity,
                Mode::StructurePreservation,
            ],
            description: "path composition in a finite category",
        },
        (Family::Category, Task::VerifyCategoryLaw) => RuliadSourceSemantics {
            math_domains: &[Domain::CategoryTheory],
            reasoning_modes: &[
                Mode::Associativity,
                Mode::EquationalReasoning,
                Mode::StructurePreservation,
            ],
            description: "identity or associativity law verification in a finite category",
        },
        (Family::Category, Task::VerifyFunctorPreservation) => RuliadSourceSemantics {
            math_domains: &[Domain::CategoryTheory],
            reasoning_modes: &[
                Mode::StructurePreservation,
                Mode::CompositionalReasoning,
                Mode::EquationalReasoning,
            ],
            description: "verification that a finite functor preserves composition",
        },
        (Family::Category, Task::VerifyNaturalitySquare) => RuliadSourceSemantics {
            math_domains: &[Domain::CategoryTheory],
            reasoning_modes: &[
                Mode::StructurePreservation,
                Mode::CompositionalReasoning,
                Mode::FormalDeduction,
            ],
            description: "verification that a finite naturality square commutes",
        },
        (Family::LeanTask, Task::CompleteProof) => RuliadSourceSemantics {
            math_domains: &[Domain::FormalProof, Domain::CategoryTheory],
            reasoning_modes: &[
                Mode::FormalDeduction,
                Mode::StructurePreservation,
                Mode::CompositionalReasoning,
            ],
            description: "proof-task payload anchored by the Lean seed project",
        },
        (Family::HashNoise, Task::HashCanary) => RuliadSourceSemantics {
            math_domains: &[Domain::InformationTheory],
            reasoning_modes: &[Mode::EntropyCanary, Mode::CounterexampleEvaluation],
            description: "high-entropy canary for source-selection and memorization checks",
        },
        _ => RuliadSourceSemantics {
            math_domains: family_default_domains(family),
            reasoning_modes: task_default_reasoning_modes(task_kind),
            description: "fallback semantics for a ruliad source",
        },
    }
}

fn family_default_domains(family: RuliadFamilyKind) -> &'static [RuliadMathDomain] {
    match family {
        RuliadFamilyKind::Eca | RuliadFamilyKind::Simulation => {
            &[RuliadMathDomain::DiscreteDynamics]
        }
        RuliadFamilyKind::Automaton => &[RuliadMathDomain::ComputationTheory],
        RuliadFamilyKind::Rewrite => &[RuliadMathDomain::SymbolicRewriting],
        RuliadFamilyKind::Algebra => &[RuliadMathDomain::UniversalAlgebra],
        RuliadFamilyKind::Category => &[RuliadMathDomain::CategoryTheory],
        RuliadFamilyKind::LeanTask => &[RuliadMathDomain::FormalProof],
        RuliadFamilyKind::HashNoise => &[RuliadMathDomain::InformationTheory],
    }
}

fn task_default_reasoning_modes(task_kind: RuliadTaskKind) -> &'static [RuliadReasoningMode] {
    match task_kind {
        RuliadTaskKind::NextState => &[RuliadReasoningMode::LocalRuleEvaluation],
        RuliadTaskKind::MultiStepState => &[RuliadReasoningMode::IteratedDynamics],
        RuliadTaskKind::VerifySimulation => &[RuliadReasoningMode::SimulationEquivalence],
        RuliadTaskKind::EvaluateAutomaton => &[RuliadReasoningMode::StateMachineExecution],
        RuliadTaskKind::RewriteNormalForm => &[RuliadReasoningMode::Normalization],
        RuliadTaskKind::CheckAlgebraLaw => &[RuliadReasoningMode::EquationalReasoning],
        RuliadTaskKind::ComposeCategoryPath => &[RuliadReasoningMode::CompositionalReasoning],
        RuliadTaskKind::VerifyCategoryLaw => &[RuliadReasoningMode::Associativity],
        RuliadTaskKind::VerifyFunctorPreservation => &[RuliadReasoningMode::StructurePreservation],
        RuliadTaskKind::VerifyNaturalitySquare => &[RuliadReasoningMode::StructurePreservation],
        RuliadTaskKind::CompleteProof => &[RuliadReasoningMode::FormalDeduction],
        RuliadTaskKind::HashCanary => &[RuliadReasoningMode::EntropyCanary],
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadFamilyConfig {
    pub kind: RuliadFamilyKind,
    #[serde(default = "default_weight")]
    pub weight: usize,
    #[serde(default)]
    pub width: Option<UsizeRangeConfig>,
    #[serde(default)]
    pub steps: Option<UsizeRangeConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadSerializationConfig {
    #[serde(default = "default_document_tokens")]
    pub document_tokens: usize,
    #[serde(default = "default_preview_samples")]
    pub preview_samples: usize,
}

impl Default for RuliadSerializationConfig {
    fn default() -> Self {
        Self {
            document_tokens: default_document_tokens(),
            preview_samples: default_preview_samples(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RuliadTokenizationConfig {
    Gpt2ByteCompatible {
        #[serde(default = "default_gpt2_vocab_size")]
        vocab_size: usize,
        #[serde(default = "default_gpt2_eos_id")]
        eos_id: Option<u32>,
    },
}

impl Default for RuliadTokenizationConfig {
    fn default() -> Self {
        Self::Gpt2ByteCompatible {
            vocab_size: default_gpt2_vocab_size(),
            eos_id: default_gpt2_eos_id(),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct RuliadSourceSelectionConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub sampler: RuliadSamplerConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadCorpusConfig {
    pub output_dir: PathBuf,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_name")]
    pub name: String,
    pub train_samples: usize,
    pub validation_samples: usize,
    #[serde(default = "default_chunk_token_capacity")]
    pub chunk_token_capacity: usize,
    #[serde(default)]
    pub serialization: RuliadSerializationConfig,
    #[serde(default)]
    pub tokenization: RuliadTokenizationConfig,
    #[serde(default)]
    pub source_selection: RuliadSourceSelectionConfig,
    #[serde(default = "default_ruliad_families")]
    pub families: Vec<RuliadFamilyConfig>,
    #[serde(default)]
    pub proof_tasks: Option<PathBuf>,
    #[serde(default)]
    pub lean_task_limit: Option<usize>,
}

impl RuliadCorpusConfig {
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(anyhow!("name must not be empty"));
        }
        if self.train_samples == 0 {
            return Err(anyhow!("train_samples must be > 0"));
        }
        if self.chunk_token_capacity == 0 {
            return Err(anyhow!("chunk_token_capacity must be > 0"));
        }
        if self.serialization.document_tokens <= 1 {
            return Err(anyhow!("serialization.document_tokens must be > 1"));
        }
        if self.serialization.preview_samples == 0 {
            return Err(anyhow!("serialization.preview_samples must be > 0"));
        }
        if self.families.is_empty() {
            return Err(anyhow!("families must not be empty"));
        }
        for (index, family) in self.families.iter().enumerate() {
            if family.weight == 0 {
                return Err(anyhow!("families[{index}].weight must be > 0"));
            }
            if let Some(range) = &family.width {
                range.validate(&format!("families[{index}].width"))?;
                if range.min == 0 {
                    return Err(anyhow!("families[{index}].width bounds must be > 0"));
                }
            }
            if let Some(range) = &family.steps {
                range.validate(&format!("families[{index}].steps"))?;
                if range.min == 0 {
                    return Err(anyhow!("families[{index}].steps bounds must be > 0"));
                }
            }
        }
        match &self.tokenization {
            RuliadTokenizationConfig::Gpt2ByteCompatible { vocab_size, eos_id } => {
                if *vocab_size < 257 {
                    return Err(anyhow!(
                        "tokenization.vocab_size must be >= 257 for gpt2_byte_compatible"
                    ));
                }
                if matches!(eos_id, Some(id) if *id as usize >= *vocab_size) {
                    return Err(anyhow!(
                        "tokenization.eos_id must be < tokenization.vocab_size"
                    ));
                }
            }
        }
        Ok(())
    }
}

pub fn load_ruliad_config(path: &Path) -> Result<RuliadCorpusConfig> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read ruliad config {}", path.display()))?;
    let config: RuliadCorpusConfig =
        toml::from_str(&contents).with_context(|| format!("failed to parse {}", path.display()))?;
    config.validate()?;
    Ok(config)
}

pub fn default_ruliad_families() -> Vec<RuliadFamilyConfig> {
    vec![
        RuliadFamilyConfig {
            kind: RuliadFamilyKind::Eca,
            weight: 4,
            width: Some(UsizeRangeConfig { min: 16, max: 32 }),
            steps: Some(UsizeRangeConfig { min: 4, max: 10 }),
        },
        RuliadFamilyConfig {
            kind: RuliadFamilyKind::Simulation,
            weight: 2,
            width: Some(UsizeRangeConfig { min: 16, max: 32 }),
            steps: Some(UsizeRangeConfig { min: 4, max: 8 }),
        },
        RuliadFamilyConfig {
            kind: RuliadFamilyKind::Automaton,
            weight: 2,
            width: Some(UsizeRangeConfig { min: 3, max: 8 }),
            steps: Some(UsizeRangeConfig { min: 6, max: 20 }),
        },
        RuliadFamilyConfig {
            kind: RuliadFamilyKind::Rewrite,
            weight: 2,
            width: Some(UsizeRangeConfig { min: 8, max: 20 }),
            steps: Some(UsizeRangeConfig { min: 4, max: 12 }),
        },
        RuliadFamilyConfig {
            kind: RuliadFamilyKind::Algebra,
            weight: 2,
            width: Some(UsizeRangeConfig { min: 2, max: 6 }),
            steps: None,
        },
        RuliadFamilyConfig {
            kind: RuliadFamilyKind::Category,
            weight: 1,
            width: Some(UsizeRangeConfig { min: 3, max: 7 }),
            steps: Some(UsizeRangeConfig { min: 3, max: 6 }),
        },
        RuliadFamilyConfig {
            kind: RuliadFamilyKind::LeanTask,
            weight: 1,
            width: None,
            steps: None,
        },
        RuliadFamilyConfig {
            kind: RuliadFamilyKind::HashNoise,
            weight: 1,
            width: None,
            steps: None,
        },
    ]
}

fn default_seed() -> u64 {
    1337
}

fn default_name() -> String {
    "ruliad_universality".to_string()
}

fn default_weight() -> usize {
    1
}

fn default_document_tokens() -> usize {
    513
}

fn default_preview_samples() -> usize {
    4
}

fn default_chunk_token_capacity() -> usize {
    1_048_576
}

fn default_gpt2_vocab_size() -> usize {
    50_257
}

fn default_gpt2_eos_id() -> Option<u32> {
    Some(50_256)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn default_config_validates() {
        let dir = tempdir().expect("tempdir");
        let config = RuliadCorpusConfig {
            output_dir: dir.path().join("out"),
            seed: 1,
            name: "demo".to_string(),
            train_samples: 8,
            validation_samples: 2,
            chunk_token_capacity: 1024,
            serialization: RuliadSerializationConfig::default(),
            tokenization: RuliadTokenizationConfig::default(),
            source_selection: RuliadSourceSelectionConfig::default(),
            families: default_ruliad_families(),
            proof_tasks: None,
            lean_task_limit: None,
        };

        config.validate().expect("valid config");
    }
}
