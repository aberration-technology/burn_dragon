//! Versioned, portable proof objects for the Ruliad corpus.
//!
//! The IR is deliberately smaller than any source language.  Generators and
//! importers compile into equality goals over typed-looking first-order terms;
//! the replay kernel is the sole authority for whether a certificate is valid.

use std::collections::BTreeSet;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::ruliad::stable_json::stable_json_hash;

pub const RULIAD_IR_VERSION: u32 = 3;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum RuliadFormalDomain {
    Equational,
    Category,
    Logic,
    Automata,
    Process,
    Metagraph,
}

impl RuliadFormalDomain {
    pub const ALL: [Self; 6] = [
        Self::Equational,
        Self::Category,
        Self::Logic,
        Self::Automata,
        Self::Process,
        Self::Metagraph,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Equational => "equational",
            Self::Category => "category",
            Self::Logic => "logic",
            Self::Automata => "automata",
            Self::Process => "process",
            Self::Metagraph => "metagraph",
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuliadTerm {
    Variable {
        index: u32,
    },
    Atom {
        symbol: String,
    },
    Apply {
        operator: String,
        arguments: Vec<RuliadTerm>,
    },
}

impl RuliadTerm {
    pub fn variable(index: u32) -> Self {
        Self::Variable { index }
    }

    pub fn atom(symbol: impl Into<String>) -> Self {
        Self::Atom {
            symbol: symbol.into(),
        }
    }

    pub fn apply(operator: impl Into<String>, arguments: Vec<Self>) -> Self {
        Self::Apply {
            operator: operator.into(),
            arguments,
        }
    }

    pub fn node_count(&self) -> usize {
        match self {
            Self::Variable { .. } | Self::Atom { .. } => 1,
            Self::Apply { arguments, .. } => 1usize.saturating_add(
                arguments
                    .iter()
                    .map(Self::node_count)
                    .fold(0usize, usize::saturating_add),
            ),
        }
    }

    pub fn depth(&self) -> usize {
        match self {
            Self::Variable { .. } | Self::Atom { .. } => 1,
            Self::Apply { arguments, .. } => {
                1 + arguments.iter().map(Self::depth).max().unwrap_or(0)
            }
        }
    }

    pub fn at_path(&self, path: &[usize]) -> Option<&Self> {
        let Some((&index, rest)) = path.split_first() else {
            return Some(self);
        };
        let Self::Apply { arguments, .. } = self else {
            return None;
        };
        arguments.get(index)?.at_path(rest)
    }

    pub fn variables(&self) -> BTreeSet<u32> {
        let mut variables = BTreeSet::new();
        self.collect_variables(&mut variables);
        variables
    }

    fn collect_variables(&self, variables: &mut BTreeSet<u32>) {
        match self {
            Self::Variable { index } => {
                variables.insert(*index);
            }
            Self::Atom { .. } => {}
            Self::Apply { arguments, .. } => {
                for argument in arguments {
                    argument.collect_variables(variables);
                }
            }
        }
    }

    pub fn canonical_text(&self) -> String {
        match self {
            Self::Variable { index } => format!("?{index}"),
            Self::Atom { symbol } => symbol.clone(),
            Self::Apply {
                operator,
                arguments,
            } => {
                let arguments = arguments
                    .iter()
                    .map(Self::canonical_text)
                    .collect::<Vec<_>>()
                    .join(",");
                format!("{operator}({arguments})")
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadRewriteAxiom {
    pub id: String,
    pub lhs: RuliadTerm,
    pub rhs: RuliadTerm,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadEquality {
    pub lhs: RuliadTerm,
    pub rhs: RuliadTerm,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadProofGoal {
    pub id: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dependencies: Vec<usize>,
    pub claim: RuliadEquality,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadProofProblem {
    pub version: u32,
    pub domain: RuliadFormalDomain,
    pub theory: String,
    pub axioms: Vec<RuliadRewriteAxiom>,
    pub goals: Vec<RuliadProofGoal>,
    pub root: usize,
}

impl RuliadProofProblem {
    pub fn canonical_hash(&self) -> Result<String> {
        stable_json_hash(self)
    }

    pub fn required_goal_indices(&self) -> Vec<usize> {
        fn visit(problem: &RuliadProofProblem, index: usize, required: &mut BTreeSet<usize>) {
            if !required.insert(index) {
                return;
            }
            if let Some(goal) = problem.goals.get(index) {
                for dependency in &goal.dependencies {
                    visit(problem, *dependency, required);
                }
            }
        }

        let mut required = BTreeSet::new();
        visit(self, self.root, &mut required);
        required.into_iter().collect()
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuliadRewriteDirection {
    Forward,
    Reverse,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuliadProofSource {
    Axiom { id: String },
    Lemma { goal: usize },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadProofStep {
    pub source: RuliadProofSource,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub path: Vec<usize>,
    pub direction: RuliadRewriteDirection,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadGoalCertificate {
    pub goal: usize,
    pub steps: Vec<RuliadProofStep>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadProofCertificate {
    pub version: u32,
    pub problem_hash: String,
    pub goals: Vec<RuliadGoalCertificate>,
}

impl RuliadProofCertificate {
    pub fn step_count(&self) -> usize {
        self.goals.iter().map(|goal| goal.steps.len()).sum()
    }

    pub fn step_at(&self, mut index: usize) -> Option<(usize, &RuliadProofStep)> {
        for goal in &self.goals {
            if index < goal.steps.len() {
                return Some((goal.goal, &goal.steps[index]));
            }
            index = index.saturating_sub(goal.steps.len());
        }
        None
    }

    pub fn prefix_before(&self, mut index: usize) -> Option<Self> {
        let mut goals = Vec::new();
        for goal in &self.goals {
            if index < goal.steps.len() {
                goals.push(RuliadGoalCertificate {
                    goal: goal.goal,
                    steps: goal.steps[..index].to_vec(),
                });
                return Some(Self {
                    version: self.version,
                    problem_hash: self.problem_hash.clone(),
                    goals,
                });
            }
            goals.push(goal.clone());
            index = index.saturating_sub(goal.steps.len());
        }
        None
    }

    pub fn single_step_at(&self, index: usize) -> Option<Self> {
        let (goal, step) = self.step_at(index)?;
        Some(Self {
            version: self.version,
            problem_hash: self.problem_hash.clone(),
            goals: vec![RuliadGoalCertificate {
                goal,
                steps: vec![step.clone()],
            }],
        })
    }

    pub fn with_step_replaced(
        &self,
        mut index: usize,
        replacement: RuliadProofStep,
    ) -> Option<Self> {
        let mut certificate = self.clone();
        for goal in &mut certificate.goals {
            if index < goal.steps.len() {
                goal.steps[index] = replacement;
                return Some(certificate);
            }
            index = index.saturating_sub(goal.steps.len());
        }
        None
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadProofBundle {
    pub problem: RuliadProofProblem,
    pub certificate: RuliadProofCertificate,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadComplexityVector {
    pub syntax_nodes: usize,
    pub axiom_count: usize,
    pub proof_goal_count: usize,
    pub proof_step_count: usize,
    pub dependency_depth: usize,
    pub dependency_width: usize,
    pub variable_count: usize,
    pub maximum_term_depth: usize,
    pub distractor_axiom_count: usize,
    #[serde(default)]
    pub branch_entropy_millibits: usize,
    #[serde(default)]
    pub binder_depth: usize,
    #[serde(default)]
    pub abstraction_depth: usize,
    #[serde(default)]
    pub memory_horizon: usize,
    #[serde(default)]
    pub representation_shift: usize,
    #[serde(default)]
    pub solution_multiplicity: usize,
    #[serde(default)]
    pub search_branching: usize,
    #[serde(default)]
    pub verifier_work: usize,
}

impl RuliadComplexityVector {
    pub fn dominates(&self, other: &Self) -> bool {
        let dimensions = [
            (self.syntax_nodes, other.syntax_nodes),
            (self.axiom_count, other.axiom_count),
            (self.proof_goal_count, other.proof_goal_count),
            (self.proof_step_count, other.proof_step_count),
            (self.dependency_depth, other.dependency_depth),
            (self.dependency_width, other.dependency_width),
            (self.variable_count, other.variable_count),
            (self.maximum_term_depth, other.maximum_term_depth),
            (self.distractor_axiom_count, other.distractor_axiom_count),
            (
                self.branch_entropy_millibits,
                other.branch_entropy_millibits,
            ),
            (self.binder_depth, other.binder_depth),
            (self.abstraction_depth, other.abstraction_depth),
            (self.memory_horizon, other.memory_horizon),
            (self.representation_shift, other.representation_shift),
            (self.solution_multiplicity, other.solution_multiplicity),
            (self.search_branching, other.search_branching),
            (self.verifier_work, other.verifier_work),
        ];
        dimensions.iter().all(|(left, right)| left >= right)
            && dimensions.iter().any(|(left, right)| left > right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn certificate() -> RuliadProofCertificate {
        let step = |id: &str| RuliadProofStep {
            source: RuliadProofSource::Axiom { id: id.into() },
            path: vec![1],
            direction: RuliadRewriteDirection::Forward,
        };
        RuliadProofCertificate {
            version: RULIAD_IR_VERSION,
            problem_hash: "problem".into(),
            goals: vec![
                RuliadGoalCertificate {
                    goal: 0,
                    steps: vec![step("a"), step("b")],
                },
                RuliadGoalCertificate {
                    goal: 1,
                    steps: vec![step("c")],
                },
            ],
        }
    }

    #[test]
    fn certificate_transition_views_preserve_global_step_order() {
        let certificate = certificate();
        assert_eq!(certificate.step_count(), 3);
        assert_eq!(certificate.step_at(2).map(|(goal, _)| goal), Some(1));

        let prefix = certificate.prefix_before(2).expect("prefix");
        assert_eq!(prefix.goals[0].steps.len(), 2);
        assert!(prefix.goals[1].steps.is_empty());

        let next = certificate.single_step_at(2).expect("next step");
        assert_eq!(next.goals[0].goal, 1);
        assert_eq!(next.goals[0].steps.len(), 1);

        let replacement = certificate.step_at(0).expect("replacement").1.clone();
        let replaced = certificate
            .with_step_replaced(2, replacement.clone())
            .expect("replace");
        assert_eq!(replaced.step_at(2).expect("replaced").1, &replacement);
        assert!(certificate.prefix_before(3).is_none());
    }

    #[test]
    fn term_path_focus_is_structural_and_bounds_checked() {
        let term = RuliadTerm::apply(
            "compose",
            vec![
                RuliadTerm::atom("f"),
                RuliadTerm::apply("inverse", vec![RuliadTerm::atom("g")]),
            ],
        );
        assert_eq!(
            term.at_path(&[1, 0]).map(RuliadTerm::canonical_text),
            Some("g".to_string())
        );
        assert_eq!(term.at_path(&[]), Some(&term));
        assert!(term.at_path(&[2]).is_none());
        assert!(term.at_path(&[0, 0]).is_none());
    }
}
