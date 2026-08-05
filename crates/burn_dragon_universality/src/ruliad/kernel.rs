//! Deterministic replay kernel for Ruliad IR certificates.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::ruliad::ir::{
    RULIAD_IR_VERSION, RuliadComplexityVector, RuliadEquality, RuliadProofCertificate,
    RuliadProofProblem, RuliadProofSource, RuliadProofStep, RuliadRewriteAxiom,
    RuliadRewriteDirection, RuliadTerm,
};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadKernelLimits {
    pub maximum_axioms: usize,
    pub maximum_goals: usize,
    pub maximum_steps: usize,
    pub maximum_term_nodes: usize,
    pub maximum_path_depth: usize,
}

impl Default for RuliadKernelLimits {
    fn default() -> Self {
        Self {
            maximum_axioms: 16_384,
            maximum_goals: 16_384,
            maximum_steps: 1_048_576,
            maximum_term_nodes: 4_194_304,
            maximum_path_depth: 4096,
        }
    }
}

/// Reusable transition kernel for one proof goal.
///
/// Construction validates the problem and indexes axioms once. Policy search
/// can then evaluate a bounded action menu without repeatedly hashing and
/// validating the entire proof DAG.
pub struct RuliadGoalTransitionKernel<'a> {
    problem: &'a RuliadProofProblem,
    goal_index: usize,
    goal: &'a crate::ruliad::ir::RuliadProofGoal,
    axioms: BTreeMap<&'a str, &'a RuliadRewriteAxiom>,
    limits: RuliadKernelLimits,
}

impl<'a> RuliadGoalTransitionKernel<'a> {
    pub fn new(
        problem: &'a RuliadProofProblem,
        goal_index: usize,
        limits: RuliadKernelLimits,
    ) -> Result<Self, RuliadKernelFailure> {
        validate_problem(problem, limits)?;
        if !problem.required_goal_indices().contains(&goal_index) {
            return Err(failure(
                RuliadKernelFailureKind::MalformedCertificate,
                Some(goal_index),
                None,
                "goal transition is outside the root dependency closure".to_string(),
            ));
        }
        let goal = problem.goals.get(goal_index).ok_or_else(|| {
            failure(
                RuliadKernelFailureKind::MalformedCertificate,
                Some(goal_index),
                None,
                "goal transition references a missing goal".to_string(),
            )
        })?;
        let axioms = problem
            .axioms
            .iter()
            .map(|axiom| (axiom.id.as_str(), axiom))
            .collect();
        Ok(Self {
            problem,
            goal_index,
            goal,
            axioms,
            limits,
        })
    }

    pub fn initial(&self) -> RuliadTerm {
        self.goal.claim.lhs.clone()
    }

    pub fn target(&self) -> &RuliadTerm {
        &self.goal.claim.rhs
    }

    pub fn apply(
        &self,
        current: &RuliadTerm,
        step: &RuliadProofStep,
    ) -> Result<RuliadTerm, RuliadKernelFailure> {
        apply_step(
            self.problem,
            self.goal_index,
            self.goal,
            &self.axioms,
            current,
            step,
            self.limits,
        )
    }

    pub fn replay_prefix(
        &self,
        steps: &[RuliadProofStep],
    ) -> Result<RuliadTerm, RuliadKernelFailure> {
        if steps.len() > self.limits.maximum_steps {
            return Err(failure(
                RuliadKernelFailureKind::ResourceLimit,
                Some(self.goal_index),
                None,
                "goal prefix step count exceeds kernel limit".to_string(),
            ));
        }
        let mut current = self.initial();
        for (step_index, step) in steps.iter().enumerate() {
            current = self.apply(&current, step).map_err(|mut failure| {
                failure.goal = Some(self.goal_index);
                failure.step = Some(step_index);
                failure
            })?;
        }
        Ok(current)
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuliadKernelFailureKind {
    Version,
    ResourceLimit,
    MalformedProblem,
    ProblemHash,
    MalformedCertificate,
    MissingDependency,
    UnknownSource,
    InvalidPath,
    PatternMismatch,
    GoalMismatch,
}

impl RuliadKernelFailureKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Version => "version",
            Self::ResourceLimit => "resource_limit",
            Self::MalformedProblem => "malformed_problem",
            Self::ProblemHash => "problem_hash",
            Self::MalformedCertificate => "malformed_certificate",
            Self::MissingDependency => "missing_dependency",
            Self::UnknownSource => "unknown_source",
            Self::InvalidPath => "invalid_path",
            Self::PatternMismatch => "pattern_mismatch",
            Self::GoalMismatch => "goal_mismatch",
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadKernelFailure {
    pub kind: RuliadKernelFailureKind,
    pub goal: Option<usize>,
    pub step: Option<usize>,
    pub message: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadReplayReport {
    pub accepted: bool,
    pub problem_hash: Option<String>,
    pub required_goals: usize,
    pub verified_goals: usize,
    pub submitted_steps: usize,
    pub verified_steps: usize,
    pub root_verified: bool,
    pub complexity: RuliadComplexityVector,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<RuliadKernelFailure>,
}

impl RuliadReplayReport {
    pub fn proof_progress(&self) -> f32 {
        if self.required_goals == 0 {
            return f32::from(self.accepted);
        }
        self.verified_goals as f32 / self.required_goals as f32
    }
}

pub fn replay_certificate(
    problem: &RuliadProofProblem,
    certificate: &RuliadProofCertificate,
    limits: RuliadKernelLimits,
) -> RuliadReplayReport {
    let complexity = complexity_vector(problem, Some(certificate));
    let submitted_steps = certificate.goals.iter().map(|goal| goal.steps.len()).sum();
    let required = problem.required_goal_indices();
    let mut report = RuliadReplayReport {
        accepted: false,
        problem_hash: None,
        required_goals: required.len(),
        verified_goals: 0,
        submitted_steps,
        verified_steps: 0,
        root_verified: false,
        complexity,
        failure: None,
    };

    let problem_hash = match validate_problem(problem, limits) {
        Ok(hash) => hash,
        Err(failure) => {
            report.failure = Some(failure);
            return report;
        }
    };
    report.problem_hash = Some(problem_hash.clone());

    if certificate.version != RULIAD_IR_VERSION {
        report.failure = Some(failure(
            RuliadKernelFailureKind::Version,
            None,
            None,
            format!(
                "certificate version {} does not match kernel version {RULIAD_IR_VERSION}",
                certificate.version
            ),
        ));
        return report;
    }
    if certificate.problem_hash != problem_hash {
        report.failure = Some(failure(
            RuliadKernelFailureKind::ProblemHash,
            None,
            None,
            "certificate problem hash does not match the canonical problem".to_string(),
        ));
        return report;
    }
    if submitted_steps > limits.maximum_steps {
        report.failure = Some(failure(
            RuliadKernelFailureKind::ResourceLimit,
            None,
            None,
            "certificate step count exceeds kernel limit".to_string(),
        ));
        return report;
    }

    let required = required.into_iter().collect::<BTreeSet<_>>();
    let mut certificates = BTreeMap::new();
    let mut previous_goal = None;
    for node in &certificate.goals {
        if previous_goal.is_some_and(|previous| node.goal <= previous) {
            report.failure = Some(failure(
                RuliadKernelFailureKind::MalformedCertificate,
                Some(node.goal),
                None,
                "certificate goals must be unique and strictly increasing".to_string(),
            ));
            return report;
        }
        previous_goal = Some(node.goal);
        if !required.contains(&node.goal) {
            report.failure = Some(failure(
                RuliadKernelFailureKind::MalformedCertificate,
                Some(node.goal),
                None,
                "certificate contains a goal outside the root dependency closure".to_string(),
            ));
            return report;
        }
        certificates.insert(node.goal, node);
    }

    let axioms = problem
        .axioms
        .iter()
        .map(|axiom| (axiom.id.as_str(), axiom))
        .collect::<BTreeMap<_, _>>();
    let mut verified = BTreeSet::new();
    for goal_index in required {
        let Some(node_certificate) = certificates.get(&goal_index) else {
            report.failure = Some(failure(
                RuliadKernelFailureKind::MalformedCertificate,
                Some(goal_index),
                None,
                "certificate is missing a required goal".to_string(),
            ));
            return report;
        };
        let goal = &problem.goals[goal_index];
        if let Some(dependency) = goal
            .dependencies
            .iter()
            .find(|dependency| !verified.contains(*dependency))
        {
            report.failure = Some(failure(
                RuliadKernelFailureKind::MissingDependency,
                Some(goal_index),
                None,
                format!("dependency {dependency} has not been verified"),
            ));
            return report;
        }

        let mut current = goal.claim.lhs.clone();
        for (step_index, step) in node_certificate.steps.iter().enumerate() {
            match apply_step(problem, goal_index, goal, &axioms, &current, step, limits) {
                Ok(next) => {
                    current = next;
                    report.verified_steps += 1;
                }
                Err(mut failure) => {
                    failure.goal = Some(goal_index);
                    failure.step = Some(step_index);
                    report.failure = Some(failure);
                    return report;
                }
            }
        }
        if current != goal.claim.rhs {
            report.failure = Some(failure(
                RuliadKernelFailureKind::GoalMismatch,
                Some(goal_index),
                Some(node_certificate.steps.len()),
                format!(
                    "replay ended at `{}` instead of `{}`",
                    current.canonical_text(),
                    goal.claim.rhs.canonical_text()
                ),
            ));
            return report;
        }
        verified.insert(goal_index);
        report.verified_goals += 1;
    }

    report.root_verified = verified.contains(&problem.root);
    report.accepted = report.root_verified && report.verified_goals == report.required_goals;
    report
}

/// Replays a valid local prefix for one goal and returns the resulting proof
/// state. Full-certificate acceptance remains the authority for completed
/// proofs; this API exists for verifier-backed transition supervision.
pub fn replay_goal_prefix(
    problem: &RuliadProofProblem,
    goal_index: usize,
    steps: &[RuliadProofStep],
    limits: RuliadKernelLimits,
) -> Result<RuliadTerm, RuliadKernelFailure> {
    RuliadGoalTransitionKernel::new(problem, goal_index, limits)?.replay_prefix(steps)
}

pub fn validate_problem(
    problem: &RuliadProofProblem,
    limits: RuliadKernelLimits,
) -> Result<String, RuliadKernelFailure> {
    if problem.version != RULIAD_IR_VERSION {
        return Err(failure(
            RuliadKernelFailureKind::Version,
            None,
            None,
            format!(
                "problem version {} does not match kernel version {RULIAD_IR_VERSION}",
                problem.version
            ),
        ));
    }
    if problem.theory.trim().is_empty()
        || problem.goals.is_empty()
        || problem.root >= problem.goals.len()
    {
        return Err(failure(
            RuliadKernelFailureKind::MalformedProblem,
            None,
            None,
            "problem requires a theory, at least one goal, and a valid root".to_string(),
        ));
    }
    if problem.axioms.len() > limits.maximum_axioms || problem.goals.len() > limits.maximum_goals {
        return Err(failure(
            RuliadKernelFailureKind::ResourceLimit,
            None,
            None,
            "problem exceeds axiom or goal limits".to_string(),
        ));
    }

    let mut axiom_ids = BTreeSet::new();
    let mut term_nodes = 0usize;
    for axiom in &problem.axioms {
        if !valid_identifier(&axiom.id) || !axiom_ids.insert(axiom.id.as_str()) {
            return Err(failure(
                RuliadKernelFailureKind::MalformedProblem,
                None,
                None,
                format!("invalid or duplicate axiom id `{}`", axiom.id),
            ));
        }
        validate_term(&axiom.lhs)?;
        validate_term(&axiom.rhs)?;
        if !axiom.rhs.variables().is_subset(&axiom.lhs.variables()) {
            return Err(failure(
                RuliadKernelFailureKind::MalformedProblem,
                None,
                None,
                format!("axiom `{}` introduces an unbound variable", axiom.id),
            ));
        }
        if matches!(axiom.lhs, RuliadTerm::Variable { .. }) {
            return Err(failure(
                RuliadKernelFailureKind::MalformedProblem,
                None,
                None,
                format!("axiom `{}` cannot rewrite a bare variable", axiom.id),
            ));
        }
        term_nodes = term_nodes
            .saturating_add(axiom.lhs.node_count())
            .saturating_add(axiom.rhs.node_count());
    }

    let mut goal_ids = BTreeSet::new();
    for (index, goal) in problem.goals.iter().enumerate() {
        if !valid_identifier(&goal.id) || !goal_ids.insert(goal.id.as_str()) {
            return Err(failure(
                RuliadKernelFailureKind::MalformedProblem,
                Some(index),
                None,
                format!("invalid or duplicate goal id `{}`", goal.id),
            ));
        }
        validate_term(&goal.claim.lhs)?;
        validate_term(&goal.claim.rhs)?;
        if goal.dependencies.windows(2).any(|pair| pair[0] >= pair[1])
            || goal
                .dependencies
                .iter()
                .any(|dependency| *dependency >= index)
        {
            return Err(failure(
                RuliadKernelFailureKind::MalformedProblem,
                Some(index),
                None,
                "goal dependencies must be unique, sorted, and precede the goal".to_string(),
            ));
        }
        term_nodes = term_nodes
            .saturating_add(goal.claim.lhs.node_count())
            .saturating_add(goal.claim.rhs.node_count());
    }
    if term_nodes > limits.maximum_term_nodes {
        return Err(failure(
            RuliadKernelFailureKind::ResourceLimit,
            None,
            None,
            "problem term nodes exceed kernel limit".to_string(),
        ));
    }

    problem.canonical_hash().map_err(|error| {
        failure(
            RuliadKernelFailureKind::MalformedProblem,
            None,
            None,
            format!("failed to hash problem: {error}"),
        )
    })
}

pub fn complexity_vector(
    problem: &RuliadProofProblem,
    certificate: Option<&RuliadProofCertificate>,
) -> RuliadComplexityVector {
    let mut syntax_nodes = 0usize;
    let mut variables = BTreeSet::new();
    let mut maximum_term_depth = 0usize;
    for axiom in &problem.axioms {
        for term in [&axiom.lhs, &axiom.rhs] {
            syntax_nodes = syntax_nodes.saturating_add(term.node_count());
            variables.extend(term.variables());
            maximum_term_depth = maximum_term_depth.max(term.depth());
        }
    }
    for goal in &problem.goals {
        for term in [&goal.claim.lhs, &goal.claim.rhs] {
            syntax_nodes = syntax_nodes.saturating_add(term.node_count());
            variables.extend(term.variables());
            maximum_term_depth = maximum_term_depth.max(term.depth());
        }
    }

    let mut depths = vec![1usize; problem.goals.len()];
    for (index, goal) in problem.goals.iter().enumerate() {
        depths[index] = 1 + goal
            .dependencies
            .iter()
            .filter_map(|dependency| depths.get(*dependency))
            .copied()
            .max()
            .unwrap_or(0);
    }
    let proof_step_count = certificate
        .map(|certificate| certificate.goals.iter().map(|goal| goal.steps.len()).sum())
        .unwrap_or(0);

    RuliadComplexityVector {
        syntax_nodes,
        axiom_count: problem.axioms.len(),
        proof_goal_count: problem.required_goal_indices().len(),
        proof_step_count,
        dependency_depth: depths.get(problem.root).copied().unwrap_or(0),
        dependency_width: problem
            .goals
            .iter()
            .map(|goal| goal.dependencies.len())
            .max()
            .unwrap_or(0),
        variable_count: variables.len(),
        maximum_term_depth,
        distractor_axiom_count: problem
            .axioms
            .iter()
            .filter(|axiom| axiom.id.starts_with("aux_"))
            .count(),
        branch_entropy_millibits: problem
            .goals
            .iter()
            .map(|goal| goal.dependencies.len())
            .filter(|width| *width > 1)
            .map(|width| (width.ilog2() as usize).saturating_mul(1000))
            .sum(),
        abstraction_depth: depths.get(problem.root).copied().unwrap_or(0),
        memory_horizon: proof_step_count,
        solution_multiplicity: 1,
        search_branching: problem.axioms.len().saturating_mul(2),
        verifier_work: syntax_nodes.saturating_add(proof_step_count),
        ..Default::default()
    }
}

fn apply_step(
    problem: &RuliadProofProblem,
    goal_index: usize,
    goal: &crate::ruliad::ir::RuliadProofGoal,
    axioms: &BTreeMap<&str, &RuliadRewriteAxiom>,
    current: &RuliadTerm,
    step: &RuliadProofStep,
    limits: RuliadKernelLimits,
) -> Result<RuliadTerm, RuliadKernelFailure> {
    if step.path.len() > limits.maximum_path_depth {
        return Err(failure(
            RuliadKernelFailureKind::ResourceLimit,
            Some(goal_index),
            None,
            "rewrite path exceeds kernel limit".to_string(),
        ));
    }
    let equality = match &step.source {
        RuliadProofSource::Axiom { id } => axioms
            .get(id.as_str())
            .map(|axiom| RuliadEquality {
                lhs: axiom.lhs.clone(),
                rhs: axiom.rhs.clone(),
            })
            .ok_or_else(|| {
                failure(
                    RuliadKernelFailureKind::UnknownSource,
                    Some(goal_index),
                    None,
                    format!("unknown axiom `{id}`"),
                )
            })?,
        RuliadProofSource::Lemma { goal: dependency } => {
            if !goal.dependencies.contains(dependency) {
                return Err(failure(
                    RuliadKernelFailureKind::MissingDependency,
                    Some(goal_index),
                    None,
                    format!("goal {dependency} is not a declared dependency"),
                ));
            }
            problem
                .goals
                .get(*dependency)
                .map(|goal| goal.claim.clone())
                .ok_or_else(|| {
                    failure(
                        RuliadKernelFailureKind::UnknownSource,
                        Some(goal_index),
                        None,
                        format!("unknown lemma goal {dependency}"),
                    )
                })?
        }
    };
    let (pattern, replacement) = match step.direction {
        RuliadRewriteDirection::Forward => (&equality.lhs, &equality.rhs),
        RuliadRewriteDirection::Reverse => (&equality.rhs, &equality.lhs),
    };
    rewrite_at_path(current, &step.path, pattern, replacement)
}

fn rewrite_at_path(
    term: &RuliadTerm,
    path: &[usize],
    pattern: &RuliadTerm,
    replacement: &RuliadTerm,
) -> Result<RuliadTerm, RuliadKernelFailure> {
    if let Some((&index, rest)) = path.split_first() {
        let RuliadTerm::Apply {
            operator,
            arguments,
        } = term
        else {
            return Err(failure(
                RuliadKernelFailureKind::InvalidPath,
                None,
                None,
                "rewrite path descends through a non-application term".to_string(),
            ));
        };
        let Some(argument) = arguments.get(index) else {
            return Err(failure(
                RuliadKernelFailureKind::InvalidPath,
                None,
                None,
                format!("rewrite path argument {index} is out of bounds"),
            ));
        };
        let rewritten = rewrite_at_path(argument, rest, pattern, replacement)?;
        let mut next_arguments = arguments.clone();
        next_arguments[index] = rewritten;
        return Ok(RuliadTerm::apply(operator.clone(), next_arguments));
    }

    let mut substitution = BTreeMap::new();
    if !match_term(pattern, term, &mut substitution) {
        return Err(failure(
            RuliadKernelFailureKind::PatternMismatch,
            None,
            None,
            format!(
                "pattern `{}` does not match `{}`",
                pattern.canonical_text(),
                term.canonical_text()
            ),
        ));
    }
    instantiate(replacement, &substitution).ok_or_else(|| {
        failure(
            RuliadKernelFailureKind::MalformedProblem,
            None,
            None,
            "replacement references an unbound variable".to_string(),
        )
    })
}

fn match_term(
    pattern: &RuliadTerm,
    value: &RuliadTerm,
    substitution: &mut BTreeMap<u32, RuliadTerm>,
) -> bool {
    match (pattern, value) {
        (RuliadTerm::Variable { index }, value) => match substitution.get(index) {
            Some(bound) => bound == value,
            None => {
                substitution.insert(*index, value.clone());
                true
            }
        },
        (RuliadTerm::Atom { symbol: left }, RuliadTerm::Atom { symbol: right }) => left == right,
        (
            RuliadTerm::Apply {
                operator: left_operator,
                arguments: left_arguments,
            },
            RuliadTerm::Apply {
                operator: right_operator,
                arguments: right_arguments,
            },
        ) => {
            left_operator == right_operator
                && left_arguments.len() == right_arguments.len()
                && left_arguments
                    .iter()
                    .zip(right_arguments)
                    .all(|(left, right)| match_term(left, right, substitution))
        }
        _ => false,
    }
}

fn instantiate(
    template: &RuliadTerm,
    substitution: &BTreeMap<u32, RuliadTerm>,
) -> Option<RuliadTerm> {
    match template {
        RuliadTerm::Variable { index } => substitution.get(index).cloned(),
        RuliadTerm::Atom { symbol } => Some(RuliadTerm::atom(symbol.clone())),
        RuliadTerm::Apply {
            operator,
            arguments,
        } => Some(RuliadTerm::apply(
            operator.clone(),
            arguments
                .iter()
                .map(|argument| instantiate(argument, substitution))
                .collect::<Option<Vec<_>>>()?,
        )),
    }
}

fn validate_term(term: &RuliadTerm) -> Result<(), RuliadKernelFailure> {
    match term {
        RuliadTerm::Variable { .. } => Ok(()),
        RuliadTerm::Atom { symbol } => validate_symbol(symbol),
        RuliadTerm::Apply {
            operator,
            arguments,
        } => {
            validate_symbol(operator)?;
            if arguments.is_empty() {
                return Err(failure(
                    RuliadKernelFailureKind::MalformedProblem,
                    None,
                    None,
                    format!("application `{operator}` requires at least one argument"),
                ));
            }
            for argument in arguments {
                validate_term(argument)?;
            }
            Ok(())
        }
    }
}

fn validate_symbol(symbol: &str) -> Result<(), RuliadKernelFailure> {
    if valid_identifier(symbol) {
        Ok(())
    } else {
        Err(failure(
            RuliadKernelFailureKind::MalformedProblem,
            None,
            None,
            format!("invalid symbol `{symbol}`"),
        ))
    }
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
}

fn failure(
    kind: RuliadKernelFailureKind,
    goal: Option<usize>,
    step: Option<usize>,
    message: String,
) -> RuliadKernelFailure {
    RuliadKernelFailure {
        kind,
        goal,
        step,
        message,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ruliad::formal::{RuliadFormalGeneratorConfig, generate_formal_bundle};

    #[test]
    fn generated_certificate_replays_and_binds_problem_hash() {
        let bundle = generate_formal_bundle(7, RuliadFormalGeneratorConfig::default())
            .expect("formal bundle");
        let report = replay_certificate(
            &bundle.problem,
            &bundle.certificate,
            RuliadKernelLimits::default(),
        );
        assert!(report.accepted, "{:?}", report.failure);
        assert!(report.root_verified);
        assert_eq!(report.verified_goals, report.required_goals);
        assert!(report.verified_steps > 0);
    }

    #[test]
    fn corrupted_step_reports_semantic_partial_progress() {
        let mut bundle = generate_formal_bundle(11, RuliadFormalGeneratorConfig::default())
            .expect("formal bundle");
        let last = bundle
            .certificate
            .goals
            .last_mut()
            .expect("root certificate");
        last.steps[0].path = vec![99];
        let report = replay_certificate(
            &bundle.problem,
            &bundle.certificate,
            RuliadKernelLimits::default(),
        );
        assert!(!report.accepted);
        assert!(report.verified_goals > 0);
        assert!(report.proof_progress() > 0.0);
        assert_eq!(
            report.failure.as_ref().map(|failure| failure.kind),
            Some(RuliadKernelFailureKind::InvalidPath)
        );
    }

    #[test]
    fn certificate_cannot_use_an_undeclared_lemma() {
        let mut bundle = generate_formal_bundle(13, RuliadFormalGeneratorConfig::default())
            .expect("formal bundle");
        let root = bundle
            .certificate
            .goals
            .last_mut()
            .expect("root certificate");
        root.steps[0].source = RuliadProofSource::Lemma { goal: 0 };
        let report = replay_certificate(
            &bundle.problem,
            &bundle.certificate,
            RuliadKernelLimits::default(),
        );
        assert!(!report.accepted);
        assert_eq!(
            report.failure.as_ref().map(|failure| failure.kind),
            Some(RuliadKernelFailureKind::MissingDependency)
        );
    }
}
