//! Versioned contracts for verifier-backed, open-ended task worlds.
//!
//! A world owns its semantics. The shared contract describes how a learner
//! observes states, proposes typed actions, and receives independently
//! verifiable transitions without forcing every domain into one proof kernel.

mod capability;
mod complexity;
mod contract;
mod formal;

pub use capability::{
    BernoulliEvidence, RuliadCapabilityCoverage, RuliadCapabilityMasteryThresholds,
    RuliadCapabilityPosterior,
};
pub use complexity::RuliadDifficultyVector;
pub use contract::{
    RULIAD_TASK_GRAPH_CONTRACT_VERSION, RuliadTaskGraph, RuliadTransitionCost,
    RuliadTransitionResult, RuliadVerifiedTransition, RuliadWorldDescriptor,
};
pub use formal::RuliadFormalProofWorld;
