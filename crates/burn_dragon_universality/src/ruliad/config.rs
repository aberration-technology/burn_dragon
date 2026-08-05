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
    ProofTree,
    FormalProof,
    LeanTask,
    HashNoise,
}

impl RuliadFamilyKind {
    pub const ALL: [Self; 10] = [
        Self::Eca,
        Self::Simulation,
        Self::Automaton,
        Self::Rewrite,
        Self::Algebra,
        Self::Category,
        Self::ProofTree,
        Self::FormalProof,
        Self::LeanTask,
        Self::HashNoise,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Eca => "eca",
            Self::Simulation => "simulation",
            Self::Automaton => "automaton",
            Self::Rewrite => "rewrite",
            Self::Algebra => "algebra",
            Self::Category => "category",
            Self::ProofTree => "proof_tree",
            Self::FormalProof => "formal_proof",
            Self::LeanTask => "lean_task",
            Self::HashNoise => "hash_noise",
        }
    }

    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "eca" => Some(Self::Eca),
            "simulation" => Some(Self::Simulation),
            "automaton" => Some(Self::Automaton),
            "rewrite" => Some(Self::Rewrite),
            "algebra" => Some(Self::Algebra),
            "category" => Some(Self::Category),
            "proof_tree" => Some(Self::ProofTree),
            "formal_proof" => Some(Self::FormalProof),
            "lean_task" => Some(Self::LeanTask),
            "hash_noise" => Some(Self::HashNoise),
            _ => None,
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
    ProveTheorem,
    ConstructProof,
    AdvanceProof,
    SelectProofAction,
    CheckProof,
    CompleteProof,
    HashCanary,
}

impl RuliadTaskKind {
    pub const ALL: [Self; 17] = [
        Self::NextState,
        Self::MultiStepState,
        Self::VerifySimulation,
        Self::EvaluateAutomaton,
        Self::RewriteNormalForm,
        Self::CheckAlgebraLaw,
        Self::ComposeCategoryPath,
        Self::VerifyCategoryLaw,
        Self::VerifyFunctorPreservation,
        Self::VerifyNaturalitySquare,
        Self::ProveTheorem,
        Self::ConstructProof,
        Self::AdvanceProof,
        Self::SelectProofAction,
        Self::CheckProof,
        Self::CompleteProof,
        Self::HashCanary,
    ];

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
            Self::ProveTheorem => "prove_theorem",
            Self::ConstructProof => "construct_proof",
            Self::AdvanceProof => "advance_proof",
            Self::SelectProofAction => "select_proof_action",
            Self::CheckProof => "check_proof",
            Self::CompleteProof => "complete_proof",
            Self::HashCanary => "hash_canary",
        }
    }

    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "next_state" => Some(Self::NextState),
            "multi_step_state" => Some(Self::MultiStepState),
            "verify_simulation" => Some(Self::VerifySimulation),
            "evaluate_automaton" => Some(Self::EvaluateAutomaton),
            "rewrite_normal_form" => Some(Self::RewriteNormalForm),
            "check_algebra_law" => Some(Self::CheckAlgebraLaw),
            "compose_category_path" => Some(Self::ComposeCategoryPath),
            "verify_category_law" => Some(Self::VerifyCategoryLaw),
            "verify_functor_preservation" => Some(Self::VerifyFunctorPreservation),
            "verify_naturality_square" => Some(Self::VerifyNaturalitySquare),
            "prove_theorem" => Some(Self::ProveTheorem),
            "construct_proof" => Some(Self::ConstructProof),
            "advance_proof" => Some(Self::AdvanceProof),
            "select_proof_action" => Some(Self::SelectProofAction),
            "check_proof" => Some(Self::CheckProof),
            "complete_proof" => Some(Self::CompleteProof),
            "hash_canary" => Some(Self::HashCanary),
            _ => None,
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
    Logic,
    ProcessCalculus,
    MetagraphRewriting,
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
            Self::Logic => "logic",
            Self::ProcessCalculus => "process_calculus",
            Self::MetagraphRewriting => "metagraph_rewriting",
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
    ProofConstruction,
    ProofChecking,
    PatternMatching,
    Substitution,
    DependencyClosure,
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
            Self::ProofConstruction => "proof_construction",
            Self::ProofChecking => "proof_checking",
            Self::PatternMatching => "pattern_matching",
            Self::Substitution => "substitution",
            Self::DependencyClosure => "dependency_closure",
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
        (Family::ProofTree, Task::ProveTheorem) => RuliadSourceSemantics {
            math_domains: &[
                Domain::UniversalAlgebra,
                Domain::CategoryTheory,
                Domain::FormalProof,
            ],
            reasoning_modes: &[
                Mode::EquationalReasoning,
                Mode::CompositionalReasoning,
                Mode::FormalDeduction,
                Mode::StructurePreservation,
            ],
            description: "verified synthetic theorem DAG over unnamed algebraic structure",
        },
        (Family::FormalProof, Task::ConstructProof) => RuliadSourceSemantics {
            math_domains: &[
                Domain::SymbolicRewriting,
                Domain::ComputationTheory,
                Domain::UniversalAlgebra,
                Domain::CategoryTheory,
                Domain::FormalProof,
                Domain::Logic,
                Domain::ProcessCalculus,
                Domain::MetagraphRewriting,
            ],
            reasoning_modes: &[
                Mode::ProofConstruction,
                Mode::PatternMatching,
                Mode::Substitution,
                Mode::DependencyClosure,
                Mode::CompositionalReasoning,
            ],
            description: "construct a replayable proof certificate over the shared Ruliad IR",
        },
        (Family::FormalProof, Task::AdvanceProof) => RuliadSourceSemantics {
            math_domains: &[
                Domain::SymbolicRewriting,
                Domain::ComputationTheory,
                Domain::UniversalAlgebra,
                Domain::CategoryTheory,
                Domain::FormalProof,
                Domain::Logic,
                Domain::ProcessCalculus,
                Domain::MetagraphRewriting,
            ],
            reasoning_modes: &[
                Mode::ProofConstruction,
                Mode::LocalRuleEvaluation,
                Mode::PatternMatching,
                Mode::Substitution,
                Mode::DependencyClosure,
            ],
            description: "advance one verifier-backed edge of the shared formal proof DAG",
        },
        (Family::FormalProof, Task::SelectProofAction) => RuliadSourceSemantics {
            math_domains: &[
                Domain::SymbolicRewriting,
                Domain::ComputationTheory,
                Domain::UniversalAlgebra,
                Domain::CategoryTheory,
                Domain::FormalProof,
                Domain::Logic,
                Domain::ProcessCalculus,
                Domain::MetagraphRewriting,
            ],
            reasoning_modes: &[
                Mode::ProofConstruction,
                Mode::LocalRuleEvaluation,
                Mode::PatternMatching,
                Mode::Substitution,
                Mode::CompositionalReasoning,
            ],
            description: "select a verifier-backed action from a formal proof state",
        },
        (Family::FormalProof, Task::CheckProof) => RuliadSourceSemantics {
            math_domains: &[
                Domain::SymbolicRewriting,
                Domain::ComputationTheory,
                Domain::UniversalAlgebra,
                Domain::CategoryTheory,
                Domain::FormalProof,
                Domain::Logic,
                Domain::ProcessCalculus,
                Domain::MetagraphRewriting,
            ],
            reasoning_modes: &[
                Mode::ProofChecking,
                Mode::PatternMatching,
                Mode::DependencyClosure,
                Mode::CounterexampleEvaluation,
            ],
            description: "replay or reject a proposed proof certificate with a localized failure",
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
        RuliadFamilyKind::ProofTree => &[RuliadMathDomain::FormalProof],
        RuliadFamilyKind::FormalProof => &[
            RuliadMathDomain::FormalProof,
            RuliadMathDomain::MetagraphRewriting,
        ],
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
        RuliadTaskKind::ProveTheorem => &[RuliadReasoningMode::FormalDeduction],
        RuliadTaskKind::ConstructProof => &[RuliadReasoningMode::ProofConstruction],
        RuliadTaskKind::AdvanceProof => &[
            RuliadReasoningMode::ProofConstruction,
            RuliadReasoningMode::LocalRuleEvaluation,
        ],
        RuliadTaskKind::SelectProofAction => &[
            RuliadReasoningMode::ProofConstruction,
            RuliadReasoningMode::CompositionalReasoning,
        ],
        RuliadTaskKind::CheckProof => &[RuliadReasoningMode::ProofChecking],
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

/// Defines what is held out by the generated formal-proof validation split.
///
/// This is versioned as part of the Ruliad semantic contract. Older corpora
/// retain seed-disjoint validation, while promotion profiles can require a
/// structural split that cannot be solved by memorizing semantic law names or
/// the training proof-DAG topology.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum RuliadFormalGeneralizationContract {
    #[default]
    SeedDisjointV1,
    StructuralHoldoutV1,
}

impl RuliadFormalGeneralizationContract {
    pub fn label(self) -> &'static str {
        match self {
            Self::SeedDisjointV1 => "seed_disjoint_v1",
            Self::StructuralHoldoutV1 => "structural_holdout_v1",
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum RuliadDocumentMode {
    #[default]
    SingleSample,
    MultiChunkProofTree,
}

impl RuliadDocumentMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::SingleSample => "single_sample",
            Self::MultiChunkProofTree => "multi_chunk_proof_tree",
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadSerializationConfig {
    #[serde(default = "default_document_tokens")]
    pub document_tokens: usize,
    #[serde(default = "default_preview_samples")]
    pub preview_samples: usize,
    #[serde(default)]
    pub document_mode: RuliadDocumentMode,
    #[serde(default = "default_document_chunks")]
    pub document_chunks: UsizeRangeConfig,
}

impl Default for RuliadSerializationConfig {
    fn default() -> Self {
        Self {
            document_tokens: default_document_tokens(),
            preview_samples: default_preview_samples(),
            document_mode: RuliadDocumentMode::default(),
            document_chunks: default_document_chunks(),
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
    Symbolic {
        #[serde(default = "default_ruliad_symbolic_vocab_size")]
        vocab_size: usize,
        #[serde(default = "default_ruliad_symbolic_eos_id")]
        eos_id: Option<u32>,
    },
    StructuredSymbolic {
        #[serde(default = "default_ruliad_structured_symbolic_vocab_size")]
        vocab_size: usize,
        #[serde(default = "default_ruliad_structured_symbolic_eos_id")]
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

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadFrontierExtensionConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_frontier_levels_per_extension")]
    pub levels_per_extension: usize,
    #[serde(default = "default_frontier_extend_normalized_difficulty")]
    pub extend_when_normalized_difficulty_at_least: f32,
    #[serde(default = "default_frontier_extend_max_difficulty_probability")]
    pub extend_when_max_difficulty_probability_at_least: f32,
    #[serde(default = "default_frontier_max_materialized_levels")]
    pub max_materialized_levels: usize,
}

impl Default for RuliadFrontierExtensionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            levels_per_extension: default_frontier_levels_per_extension(),
            extend_when_normalized_difficulty_at_least:
                default_frontier_extend_normalized_difficulty(),
            extend_when_max_difficulty_probability_at_least:
                default_frontier_extend_max_difficulty_probability(),
            max_materialized_levels: default_frontier_max_materialized_levels(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadFormalTaskMixConfig {
    #[serde(default)]
    pub advance_proof_weight: usize,
    #[serde(default)]
    pub select_proof_action_weight: usize,
    #[serde(default = "default_weight")]
    pub construct_proof_weight: usize,
    #[serde(default = "default_weight")]
    pub check_proof_weight: usize,
    #[serde(default)]
    pub proof_action_answer_contract: RuliadProofActionAnswerContract,
}

impl Default for RuliadFormalTaskMixConfig {
    fn default() -> Self {
        Self {
            advance_proof_weight: 0,
            select_proof_action_weight: 0,
            construct_proof_weight: default_weight(),
            check_proof_weight: default_weight(),
            proof_action_answer_contract: RuliadProofActionAnswerContract::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuliadProofActionAnswerContract {
    /// Presentation-relative menu index retained only as an explicit ablation control.
    #[default]
    PresentationIndex,
    /// Executable proof step, invariant to candidate-menu presentation order.
    SemanticStep,
}

impl RuliadProofActionAnswerContract {
    pub const fn label(self) -> &'static str {
        match self {
            Self::PresentationIndex => "presentation_index",
            Self::SemanticStep => "semantic_step",
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadSourceSelectionConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_source_selection_feedback_updates_enabled")]
    pub feedback_updates_enabled: bool,
    #[serde(default)]
    pub sampler: RuliadSamplerConfig,
    #[serde(default = "default_difficulty_levels")]
    pub difficulty_levels: UsizeRangeConfig,
    #[serde(default)]
    pub frontier_extension: RuliadFrontierExtensionConfig,
    #[serde(default)]
    pub cold_start: RuliadSourceSelectionColdStartConfig,
    #[serde(default)]
    pub formal_task_mix: RuliadFormalTaskMixConfig,
}

impl Default for RuliadSourceSelectionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            feedback_updates_enabled: default_source_selection_feedback_updates_enabled(),
            sampler: RuliadSamplerConfig::default(),
            difficulty_levels: default_difficulty_levels(),
            frontier_extension: RuliadFrontierExtensionConfig::default(),
            cold_start: RuliadSourceSelectionColdStartConfig::default(),
            formal_task_mix: RuliadFormalTaskMixConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadSourceSelectionColdStartConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_cold_start_max_difficulty_level")]
    pub max_difficulty_level: usize,
    #[serde(default = "default_cold_start_hold_steps")]
    pub hold_steps: usize,
    #[serde(default = "default_cold_start_ramp_steps")]
    pub ramp_steps: usize,
    #[serde(default)]
    pub release_requires_mastery: bool,
    #[serde(default = "default_cold_start_mastery_min_feedback_count")]
    pub mastery_min_feedback_count: usize,
    #[serde(default = "default_cold_start_mastery_verifier_min")]
    pub mastery_verifier_min: f32,
    #[serde(default = "default_cold_start_mastery_completion_health_min")]
    pub mastery_completion_health_min: f32,
    #[serde(default = "default_cold_start_mastery_schema_wrong_max")]
    pub mastery_schema_wrong_max: f32,
    #[serde(default = "default_cold_start_mastery_malformed_max")]
    pub mastery_malformed_max: f32,
    #[serde(default = "default_cold_start_mastery_missing_max")]
    pub mastery_missing_max: f32,
}

impl Default for RuliadSourceSelectionColdStartConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_difficulty_level: default_cold_start_max_difficulty_level(),
            hold_steps: default_cold_start_hold_steps(),
            ramp_steps: default_cold_start_ramp_steps(),
            release_requires_mastery: false,
            mastery_min_feedback_count: default_cold_start_mastery_min_feedback_count(),
            mastery_verifier_min: default_cold_start_mastery_verifier_min(),
            mastery_completion_health_min: default_cold_start_mastery_completion_health_min(),
            mastery_schema_wrong_max: default_cold_start_mastery_schema_wrong_max(),
            mastery_malformed_max: default_cold_start_mastery_malformed_max(),
            mastery_missing_max: default_cold_start_mastery_missing_max(),
        }
    }
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
    pub formal_generalization: RuliadFormalGeneralizationContract,
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
        self.serialization
            .document_chunks
            .validate("serialization.document_chunks")?;
        if self.serialization.document_chunks.min == 0 {
            return Err(anyhow!("serialization.document_chunks bounds must be > 0"));
        }
        self.source_selection
            .difficulty_levels
            .validate("source_selection.difficulty_levels")?;
        let sampler = &self.source_selection.sampler;
        if !sampler.capability_frontier_min_coverage.is_finite()
            || !(0.0..=1.0).contains(&sampler.capability_frontier_min_coverage)
        {
            return Err(anyhow!(
                "source_selection.sampler.capability_frontier_min_coverage must be finite in [0, 1]"
            ));
        }
        if sampler.capability_mastery.minimum_items == 0 {
            return Err(anyhow!(
                "source_selection.sampler.capability_mastery.minimum_items must be > 0"
            ));
        }
        if !sampler.capability_mastery.confidence_z.is_finite()
            || sampler.capability_mastery.confidence_z < 0.0
        {
            return Err(anyhow!(
                "source_selection.sampler.capability_mastery.confidence_z must be finite and >= 0"
            ));
        }
        for (name, value) in [
            ("verifier_min", sampler.capability_mastery.verifier_min),
            (
                "completion_health_min",
                sampler.capability_mastery.completion_health_min,
            ),
            (
                "schema_wrong_max",
                sampler.capability_mastery.schema_wrong_max,
            ),
            ("malformed_max", sampler.capability_mastery.malformed_max),
            ("missing_max", sampler.capability_mastery.missing_max),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(anyhow!(
                    "source_selection.sampler.capability_mastery.{name} must be finite in [0, 1]"
                ));
            }
        }
        let formal_task_weight = self
            .source_selection
            .formal_task_mix
            .advance_proof_weight
            .saturating_add(
                self.source_selection
                    .formal_task_mix
                    .select_proof_action_weight,
            )
            .saturating_add(self.source_selection.formal_task_mix.construct_proof_weight)
            .saturating_add(self.source_selection.formal_task_mix.check_proof_weight);
        if formal_task_weight == 0 {
            return Err(anyhow!(
                "source_selection.formal_task_mix requires at least one non-zero task weight"
            ));
        }
        if self.source_selection.cold_start.enabled {
            if self.source_selection.cold_start.max_difficulty_level
                < self.source_selection.difficulty_levels.min
            {
                return Err(anyhow!(
                    "source_selection.cold_start.max_difficulty_level must be >= source_selection.difficulty_levels.min"
                ));
            }
            if self.source_selection.cold_start.ramp_steps == 0 {
                return Err(anyhow!(
                    "source_selection.cold_start.ramp_steps must be > 0 when cold_start is enabled"
                ));
            }
            if self.source_selection.cold_start.mastery_min_feedback_count == 0 {
                return Err(anyhow!(
                    "source_selection.cold_start.mastery_min_feedback_count must be > 0 when cold_start is enabled"
                ));
            }
            if !(0.0..=1.0).contains(&self.source_selection.cold_start.mastery_verifier_min)
                || !self
                    .source_selection
                    .cold_start
                    .mastery_verifier_min
                    .is_finite()
            {
                return Err(anyhow!(
                    "source_selection.cold_start.mastery_verifier_min must be finite in [0, 1]"
                ));
            }
            if !(0.0..=1.0).contains(
                &self
                    .source_selection
                    .cold_start
                    .mastery_completion_health_min,
            ) || !self
                .source_selection
                .cold_start
                .mastery_completion_health_min
                .is_finite()
            {
                return Err(anyhow!(
                    "source_selection.cold_start.mastery_completion_health_min must be finite in [0, 1]"
                ));
            }
            if !(0.0..=1.0).contains(&self.source_selection.cold_start.mastery_schema_wrong_max)
                || !self
                    .source_selection
                    .cold_start
                    .mastery_schema_wrong_max
                    .is_finite()
            {
                return Err(anyhow!(
                    "source_selection.cold_start.mastery_schema_wrong_max must be finite in [0, 1]"
                ));
            }
            if !(0.0..=1.0).contains(&self.source_selection.cold_start.mastery_malformed_max)
                || !self
                    .source_selection
                    .cold_start
                    .mastery_malformed_max
                    .is_finite()
            {
                return Err(anyhow!(
                    "source_selection.cold_start.mastery_malformed_max must be finite in [0, 1]"
                ));
            }
            if !(0.0..=1.0).contains(&self.source_selection.cold_start.mastery_missing_max)
                || !self
                    .source_selection
                    .cold_start
                    .mastery_missing_max
                    .is_finite()
            {
                return Err(anyhow!(
                    "source_selection.cold_start.mastery_missing_max must be finite in [0, 1]"
                ));
            }
        }
        if self.source_selection.frontier_extension.enabled {
            if self
                .source_selection
                .frontier_extension
                .levels_per_extension
                == 0
            {
                return Err(anyhow!(
                    "source_selection.frontier_extension.levels_per_extension must be > 0"
                ));
            }
            let normalized_threshold = self
                .source_selection
                .frontier_extension
                .extend_when_normalized_difficulty_at_least;
            if !normalized_threshold.is_finite() || !(0.0..=1.0).contains(&normalized_threshold) {
                return Err(anyhow!(
                    "source_selection.frontier_extension.extend_when_normalized_difficulty_at_least must be in [0, 1]"
                ));
            }
            let max_probability_threshold = self
                .source_selection
                .frontier_extension
                .extend_when_max_difficulty_probability_at_least;
            if !max_probability_threshold.is_finite()
                || !(0.0..=1.0).contains(&max_probability_threshold)
            {
                return Err(anyhow!(
                    "source_selection.frontier_extension.extend_when_max_difficulty_probability_at_least must be in [0, 1]"
                ));
            }
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
            RuliadTokenizationConfig::Symbolic { vocab_size, eos_id } => {
                if *vocab_size < 512 {
                    return Err(anyhow!(
                        "tokenization.vocab_size must be >= 512 for symbolic"
                    ));
                }
                if matches!(eos_id, Some(id) if *id as usize >= *vocab_size) {
                    return Err(anyhow!(
                        "tokenization.eos_id must be < tokenization.vocab_size"
                    ));
                }
            }
            RuliadTokenizationConfig::StructuredSymbolic { vocab_size, eos_id } => {
                if *vocab_size < 272 {
                    return Err(anyhow!(
                        "tokenization.vocab_size must be >= 272 for structured_symbolic"
                    ));
                }
                if matches!(eos_id, Some(id) if *id as usize >= *vocab_size) {
                    return Err(anyhow!(
                        "tokenization.eos_id must be < tokenization.vocab_size"
                    ));
                }
                if matches!(eos_id, Some(id) if *id < 271) {
                    return Err(anyhow!(
                        "tokenization.eos_id must not collide with structured_symbolic byte, structural, or class tokens"
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
            kind: RuliadFamilyKind::ProofTree,
            weight: 2,
            width: Some(UsizeRangeConfig { min: 5, max: 13 }),
            steps: Some(UsizeRangeConfig { min: 4, max: 9 }),
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

pub fn compact_ruliad_families() -> Vec<RuliadFamilyConfig> {
    let mut families = default_ruliad_families();
    for family in &mut families {
        match family.kind {
            RuliadFamilyKind::Eca | RuliadFamilyKind::Simulation => {
                family.width = Some(UsizeRangeConfig { min: 12, max: 16 });
                family.steps = Some(UsizeRangeConfig { min: 4, max: 6 });
            }
            RuliadFamilyKind::Automaton => {
                family.width = Some(UsizeRangeConfig { min: 3, max: 6 });
                family.steps = Some(UsizeRangeConfig { min: 4, max: 8 });
            }
            RuliadFamilyKind::Rewrite => {
                family.width = Some(UsizeRangeConfig { min: 8, max: 12 });
                family.steps = Some(UsizeRangeConfig { min: 4, max: 8 });
            }
            RuliadFamilyKind::Algebra => {
                family.width = Some(UsizeRangeConfig { min: 2, max: 5 });
                family.steps = None;
            }
            RuliadFamilyKind::Category => {
                family.width = Some(UsizeRangeConfig { min: 3, max: 5 });
                family.steps = Some(UsizeRangeConfig { min: 3, max: 5 });
            }
            RuliadFamilyKind::ProofTree => {
                family.width = Some(UsizeRangeConfig { min: 5, max: 8 });
                family.steps = Some(UsizeRangeConfig { min: 4, max: 6 });
            }
            RuliadFamilyKind::FormalProof => {
                family.width = Some(UsizeRangeConfig { min: 2, max: 3 });
                family.steps = Some(UsizeRangeConfig { min: 2, max: 3 });
            }
            RuliadFamilyKind::LeanTask | RuliadFamilyKind::HashNoise => {
                family.width = None;
                family.steps = None;
            }
        }
    }
    families
}

pub fn formal_ruliad_families() -> Vec<RuliadFamilyConfig> {
    vec![RuliadFamilyConfig {
        kind: RuliadFamilyKind::FormalProof,
        weight: 1,
        width: Some(UsizeRangeConfig { min: 2, max: 4 }),
        steps: Some(UsizeRangeConfig { min: 2, max: 4 }),
    }]
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

fn default_document_chunks() -> UsizeRangeConfig {
    UsizeRangeConfig { min: 1, max: 1 }
}

fn default_difficulty_levels() -> UsizeRangeConfig {
    UsizeRangeConfig { min: 0, max: 0 }
}

fn default_source_selection_feedback_updates_enabled() -> bool {
    true
}

fn default_cold_start_max_difficulty_level() -> usize {
    2
}

fn default_cold_start_hold_steps() -> usize {
    1024
}

fn default_cold_start_ramp_steps() -> usize {
    8192
}

fn default_cold_start_mastery_min_feedback_count() -> usize {
    1
}

fn default_cold_start_mastery_verifier_min() -> f32 {
    0.50
}

fn default_cold_start_mastery_completion_health_min() -> f32 {
    0.75
}

fn default_cold_start_mastery_schema_wrong_max() -> f32 {
    0.25
}

fn default_cold_start_mastery_malformed_max() -> f32 {
    0.05
}

fn default_cold_start_mastery_missing_max() -> f32 {
    0.05
}

fn default_frontier_levels_per_extension() -> usize {
    8
}

fn default_frontier_extend_normalized_difficulty() -> f32 {
    0.88
}

fn default_frontier_extend_max_difficulty_probability() -> f32 {
    0.25
}

fn default_frontier_max_materialized_levels() -> usize {
    0
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

fn default_ruliad_symbolic_vocab_size() -> usize {
    4097
}

fn default_ruliad_symbolic_eos_id() -> Option<u32> {
    Some(4096)
}

fn default_ruliad_structured_symbolic_vocab_size() -> usize {
    272
}

fn default_ruliad_structured_symbolic_eos_id() -> Option<u32> {
    Some(271)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn default_config_validates() {
        let dir = tempdir().expect("tempdir");
        let mut config = RuliadCorpusConfig {
            output_dir: dir.path().join("out"),
            seed: 1,
            name: "demo".to_string(),
            train_samples: 8,
            validation_samples: 2,
            chunk_token_capacity: 1024,
            serialization: RuliadSerializationConfig::default(),
            tokenization: RuliadTokenizationConfig::default(),
            formal_generalization: Default::default(),
            source_selection: RuliadSourceSelectionConfig::default(),
            families: default_ruliad_families(),
            proof_tasks: None,
            lean_task_limit: None,
        };

        config.validate().expect("valid config");

        config
            .source_selection
            .formal_task_mix
            .construct_proof_weight = 0;
        config.source_selection.formal_task_mix.check_proof_weight = 0;
        let error = config
            .validate()
            .expect_err("empty formal task mix must fail");
        assert!(error.to_string().contains("formal_task_mix"));
    }

    #[test]
    fn canonical_family_and_task_labels_round_trip() {
        for family in RuliadFamilyKind::ALL {
            assert_eq!(RuliadFamilyKind::from_label(family.label()), Some(family));
        }
        for task in RuliadTaskKind::ALL {
            assert_eq!(RuliadTaskKind::from_label(task.label()), Some(task));
        }
        assert_eq!(RuliadFamilyKind::from_label("unknown"), None);
        assert_eq!(RuliadTaskKind::from_label("unknown"), None);
    }

    #[test]
    fn structural_generalization_contract_round_trips_in_corpus_config() {
        let mut config = RuliadCorpusConfig {
            output_dir: "target/test-ruliad-structural-holdout".into(),
            seed: 1,
            name: "structural-holdout".to_string(),
            train_samples: 8,
            validation_samples: 2,
            chunk_token_capacity: 1024,
            serialization: RuliadSerializationConfig::default(),
            tokenization: RuliadTokenizationConfig::default(),
            formal_generalization: RuliadFormalGeneralizationContract::StructuralHoldoutV1,
            source_selection: RuliadSourceSelectionConfig::default(),
            families: formal_ruliad_families(),
            proof_tasks: None,
            lean_task_limit: None,
        };
        config
            .source_selection
            .formal_task_mix
            .proof_action_answer_contract = RuliadProofActionAnswerContract::SemanticStep;
        let encoded = toml::to_string(&config).expect("serialize config");
        let decoded: RuliadCorpusConfig = toml::from_str(&encoded).expect("deserialize config");
        assert_eq!(
            decoded.formal_generalization,
            RuliadFormalGeneralizationContract::StructuralHoldoutV1
        );
        assert_eq!(
            decoded
                .source_selection
                .formal_task_mix
                .proof_action_answer_contract,
            RuliadProofActionAnswerContract::SemanticStep
        );

        config.formal_generalization = RuliadFormalGeneralizationContract::default();
        assert_eq!(config.formal_generalization.label(), "seed_disjoint_v1");
    }

    #[test]
    fn compact_families_preserve_default_span_with_small_bounds() {
        let families = compact_ruliad_families();
        let default_kinds = default_ruliad_families()
            .into_iter()
            .map(|family| family.kind)
            .collect::<Vec<_>>();
        let compact_kinds = families
            .iter()
            .map(|family| family.kind)
            .collect::<Vec<_>>();
        assert_eq!(compact_kinds, default_kinds);

        let config = RuliadCorpusConfig {
            output_dir: "target/test-ruliad-compact".into(),
            seed: 1,
            name: "compact".to_string(),
            train_samples: 8,
            validation_samples: 2,
            chunk_token_capacity: 1024,
            serialization: RuliadSerializationConfig::default(),
            tokenization: RuliadTokenizationConfig::default(),
            formal_generalization: Default::default(),
            source_selection: RuliadSourceSelectionConfig::default(),
            families,
            proof_tasks: None,
            lean_task_limit: None,
        };
        config.validate().expect("valid compact config");
        for family in &config.families {
            if let Some(width) = family.width {
                assert!(width.max <= 16);
            }
            if let Some(steps) = family.steps {
                assert!(steps.max <= 8);
            }
        }
    }

    #[test]
    fn formal_families_use_only_the_shared_proof_ir() {
        let families = formal_ruliad_families();
        assert_eq!(families.len(), 1);
        assert_eq!(families[0].kind, RuliadFamilyKind::FormalProof);
        assert!(families[0].width.is_some());
        assert!(families[0].steps.is_some());
    }

    #[test]
    fn cold_start_mastery_gate_config_validates_thresholds() {
        let mut config = RuliadCorpusConfig {
            output_dir: "target/test-ruliad-cold-start".into(),
            seed: 1,
            name: "cold-start".to_string(),
            train_samples: 8,
            validation_samples: 2,
            chunk_token_capacity: 1024,
            serialization: RuliadSerializationConfig::default(),
            tokenization: RuliadTokenizationConfig::default(),
            formal_generalization: Default::default(),
            source_selection: RuliadSourceSelectionConfig {
                enabled: true,
                difficulty_levels: UsizeRangeConfig { min: 0, max: 4 },
                cold_start: RuliadSourceSelectionColdStartConfig {
                    enabled: true,
                    max_difficulty_level: 0,
                    release_requires_mastery: true,
                    ..Default::default()
                },
                ..Default::default()
            },
            families: compact_ruliad_families(),
            proof_tasks: None,
            lean_task_limit: None,
        };
        config.validate().expect("valid mastery-gated cold start");

        config
            .source_selection
            .cold_start
            .mastery_min_feedback_count = 0;
        let err = config.validate().expect_err("zero feedback count rejected");
        assert!(err.to_string().contains("mastery_min_feedback_count"));

        config
            .source_selection
            .cold_start
            .mastery_min_feedback_count = 1;
        config.source_selection.cold_start.mastery_verifier_min = 1.5;
        let err = config
            .validate()
            .expect_err("invalid verifier threshold rejected");
        assert!(err.to_string().contains("mastery_verifier_min"));
    }
}
