use serde::{Deserialize, Serialize};

use super::RuliadDifficultyVector;

pub const RULIAD_TASK_GRAPH_CONTRACT_VERSION: u32 = 1;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadWorldDescriptor {
    pub contract_version: u32,
    pub world_id: String,
    pub semantics_id: String,
    pub state_schema: String,
    pub action_schema: String,
    pub certificate_schema: String,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadTransitionCost {
    pub model_steps: usize,
    pub verifier_steps: usize,
    pub bytes_read: usize,
    pub bytes_written: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadVerifiedTransition<State, Action, Certificate> {
    pub before: State,
    pub action: Action,
    pub after: State,
    pub certificate: Certificate,
    pub cost: RuliadTransitionCost,
}

pub type RuliadTransitionResult<State, Action, Certificate, Failure> =
    Result<RuliadVerifiedTransition<State, Action, Certificate>, Failure>;

/// Common contract for a verifier-backed task category.
///
/// States are objects and accepted transitions are morphisms. Implementations
/// retain their own semantics and verifier rather than lowering correctness to
/// a shared text format.
pub trait RuliadTaskGraph {
    type Problem;
    type State: Clone + PartialEq;
    type Action: Clone;
    type Certificate: Clone;
    type Failure;

    fn descriptor(&self) -> RuliadWorldDescriptor;

    fn initial_state(&self, problem: &Self::Problem) -> Result<Self::State, Self::Failure>;

    fn transition(
        &self,
        problem: &Self::Problem,
        state: &Self::State,
        action: &Self::Action,
    ) -> RuliadTransitionResult<Self::State, Self::Action, Self::Certificate, Self::Failure>;

    fn verify_transition(
        &self,
        problem: &Self::Problem,
        transition: &RuliadVerifiedTransition<Self::State, Self::Action, Self::Certificate>,
    ) -> Result<(), Self::Failure>;

    fn difficulty(&self, problem: &Self::Problem) -> RuliadDifficultyVector;
}
