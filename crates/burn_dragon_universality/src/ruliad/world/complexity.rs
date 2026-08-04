use serde::{Deserialize, Serialize};

use crate::ruliad::ir::RuliadComplexityVector;

/// Semantic coordinates used for curriculum decisions.
///
/// Coordinates are deliberately independent. A scalar display score may be
/// derived for dashboards, but it must not be used as the capability contract.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq, Hash)]
pub struct RuliadDifficultyVector {
    pub syntax_nodes: usize,
    pub transition_steps: usize,
    pub dependency_depth: usize,
    pub dependency_width: usize,
    pub branch_entropy_millibits: usize,
    pub binder_depth: usize,
    pub abstraction_depth: usize,
    pub memory_horizon: usize,
    pub distractor_count: usize,
    pub representation_shift: usize,
    pub solution_multiplicity: usize,
    pub search_branching: usize,
    pub verifier_work: usize,
}

impl RuliadDifficultyVector {
    pub const DIMENSIONS: usize = 13;

    pub fn coordinates(&self) -> [usize; Self::DIMENSIONS] {
        [
            self.syntax_nodes,
            self.transition_steps,
            self.dependency_depth,
            self.dependency_width,
            self.branch_entropy_millibits,
            self.binder_depth,
            self.abstraction_depth,
            self.memory_horizon,
            self.distractor_count,
            self.representation_shift,
            self.solution_multiplicity,
            self.search_branching,
            self.verifier_work,
        ]
    }

    /// Pareto dominance, used to identify a genuinely harder frontier.
    pub fn dominates(&self, other: &Self) -> bool {
        let left = self.coordinates();
        let right = other.coordinates();
        left.iter().zip(right).all(|(left, right)| *left >= right)
            && left.iter().zip(right).any(|(left, right)| *left > right)
    }

    pub fn differs_on_semantic_axis(&self, other: &Self) -> bool {
        self.coordinates() != other.coordinates()
    }

    /// Cost estimate for scheduling only. This is not a mastery score.
    pub fn estimated_work(&self) -> usize {
        self.syntax_nodes
            .saturating_add(self.transition_steps)
            .saturating_add(self.verifier_work)
            .saturating_add(self.memory_horizon)
            .max(1)
    }
}

impl From<&RuliadComplexityVector> for RuliadDifficultyVector {
    fn from(value: &RuliadComplexityVector) -> Self {
        Self {
            syntax_nodes: value.syntax_nodes,
            transition_steps: value.proof_step_count,
            dependency_depth: value.dependency_depth,
            dependency_width: value.dependency_width,
            branch_entropy_millibits: value.branch_entropy_millibits,
            binder_depth: value.binder_depth,
            abstraction_depth: value.abstraction_depth,
            memory_horizon: value.memory_horizon,
            distractor_count: value.distractor_axiom_count,
            representation_shift: value.representation_shift,
            solution_multiplicity: value.solution_multiplicity,
            search_branching: value.search_branching,
            verifier_work: value.verifier_work.max(value.proof_step_count),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pareto_dominance_requires_no_regressed_coordinate() {
        let base = RuliadDifficultyVector {
            transition_steps: 8,
            memory_horizon: 16,
            ..Default::default()
        };
        let harder = RuliadDifficultyVector {
            transition_steps: 9,
            memory_horizon: 16,
            ..Default::default()
        };
        let tradeoff = RuliadDifficultyVector {
            transition_steps: 9,
            memory_horizon: 15,
            ..Default::default()
        };
        assert!(harder.dominates(&base));
        assert!(!tradeoff.dominates(&base));
        assert!(tradeoff.differs_on_semantic_axis(&base));
    }
}
