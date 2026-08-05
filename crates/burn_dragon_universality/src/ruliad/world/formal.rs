use crate::ruliad::ir::{RuliadProofProblem, RuliadProofStep, RuliadTerm};
use crate::ruliad::kernel::{
    RuliadGoalTransitionKernel, RuliadKernelFailure, RuliadKernelFailureKind, RuliadKernelLimits,
    complexity_vector,
};

use super::{
    RULIAD_TASK_GRAPH_CONTRACT_VERSION, RuliadDifficultyVector, RuliadTaskGraph,
    RuliadTransitionCost, RuliadTransitionResult, RuliadVerifiedTransition, RuliadWorldDescriptor,
};

pub struct RuliadFormalProofWorld {
    goal_index: usize,
    limits: RuliadKernelLimits,
}

impl RuliadFormalProofWorld {
    pub fn new(goal_index: usize, limits: RuliadKernelLimits) -> Self {
        Self { goal_index, limits }
    }

    fn kernel<'a>(
        &self,
        problem: &'a RuliadProofProblem,
    ) -> Result<RuliadGoalTransitionKernel<'a>, RuliadKernelFailure> {
        RuliadGoalTransitionKernel::new(problem, self.goal_index, self.limits)
    }
}

impl RuliadTaskGraph for RuliadFormalProofWorld {
    type Problem = RuliadProofProblem;
    type State = RuliadTerm;
    type Action = RuliadProofStep;
    type Certificate = RuliadProofStep;
    type Failure = RuliadKernelFailure;

    fn descriptor(&self) -> RuliadWorldDescriptor {
        RuliadWorldDescriptor {
            contract_version: RULIAD_TASK_GRAPH_CONTRACT_VERSION,
            world_id: "formal_proof".to_string(),
            semantics_id: "burn-dragon-ruliad-formal-transition-v1".to_string(),
            state_schema: "ruliad_term_v3".to_string(),
            action_schema: "ruliad_proof_step_v3".to_string(),
            certificate_schema: "replayable_proof_step_v3".to_string(),
        }
    }

    fn initial_state(&self, problem: &Self::Problem) -> Result<Self::State, Self::Failure> {
        Ok(self.kernel(problem)?.initial())
    }

    fn transition(
        &self,
        problem: &Self::Problem,
        state: &Self::State,
        action: &Self::Action,
    ) -> RuliadTransitionResult<Self::State, Self::Action, Self::Certificate, Self::Failure> {
        let after = self.kernel(problem)?.apply(state, action)?;
        Ok(RuliadVerifiedTransition {
            before: state.clone(),
            action: action.clone(),
            after,
            certificate: action.clone(),
            cost: RuliadTransitionCost {
                model_steps: 1,
                verifier_steps: 1,
                ..Default::default()
            },
        })
    }

    fn verify_transition(
        &self,
        problem: &Self::Problem,
        transition: &RuliadVerifiedTransition<Self::State, Self::Action, Self::Certificate>,
    ) -> Result<(), Self::Failure> {
        let replayed = self
            .kernel(problem)?
            .apply(&transition.before, &transition.certificate)?;
        if replayed != transition.after || transition.action != transition.certificate {
            return Err(RuliadKernelFailure {
                kind: RuliadKernelFailureKind::GoalMismatch,
                goal: Some(self.goal_index),
                step: None,
                message: "transition envelope does not match deterministic replay".to_string(),
            });
        }
        Ok(())
    }

    fn difficulty(&self, problem: &Self::Problem) -> RuliadDifficultyVector {
        RuliadDifficultyVector::from(&complexity_vector(problem, None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ruliad::formal::{RuliadFormalGeneratorConfig, generate_formal_bundle};

    #[test]
    fn formal_world_transition_replays_through_shared_contract() {
        let bundle = generate_formal_bundle(101, RuliadFormalGeneratorConfig::default())
            .expect("formal bundle");
        let goal = bundle.certificate.goals.first().expect("goal certificate");
        let action = goal.steps.first().expect("proof step");
        let world = RuliadFormalProofWorld::new(goal.goal, RuliadKernelLimits::default());
        let state = world.initial_state(&bundle.problem).expect("initial state");
        let transition = world
            .transition(&bundle.problem, &state, action)
            .expect("verified transition");
        world
            .verify_transition(&bundle.problem, &transition)
            .expect("transition replay");
        assert_ne!(transition.before, transition.after);
    }
}
