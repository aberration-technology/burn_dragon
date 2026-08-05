//! Compact, versioned wire representation for model-facing Ruliad IR.

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::ruliad::ir::{
    RuliadEquality, RuliadFormalDomain, RuliadGoalCertificate, RuliadProofCertificate,
    RuliadProofGoal, RuliadProofProblem, RuliadProofSource, RuliadProofStep, RuliadRewriteAxiom,
    RuliadRewriteDirection, RuliadTerm,
};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
enum WireNode {
    V(u32),
    A(usize),
    P(usize, Vec<usize>),
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
struct WireAxiom(String, usize, usize);

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
struct WireGoal(String, Vec<usize>, usize, usize);

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
struct WireProblem(
    u32,
    RuliadFormalDomain,
    String,
    Vec<String>,
    Vec<WireNode>,
    Vec<WireAxiom>,
    Vec<WireGoal>,
    usize,
);

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
enum WireSource {
    A(String),
    L(usize),
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
struct WireStep(WireSource, bool, Vec<usize>);

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
struct WireGoalCertificate(usize, Vec<WireStep>);

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
struct WireCertificate(u32, String, Vec<WireGoalCertificate>);

/// Incrementally parseable model-facing proof body. The verifier binds the
/// body to the prompt problem; the model never predicts a cryptographic hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuliadModelCertificatePrefix {
    pub certificate: RuliadProofCertificate,
    pub parsed_steps: usize,
    pub syntax_complete: bool,
}

pub fn encode_problem(problem: &RuliadProofProblem) -> Result<String> {
    serde_json::to_string(&WireProblem::from_problem(problem))
        .context("encode compact ruliad problem")
}

pub fn decode_problem(payload: &str) -> Result<RuliadProofProblem> {
    let wire: WireProblem =
        serde_json::from_str(payload).context("decode compact ruliad problem")?;
    wire.into_problem()
}

pub fn encode_certificate(certificate: &RuliadProofCertificate) -> Result<String> {
    serde_json::to_string(&WireCertificate::from(certificate))
        .context("encode compact ruliad certificate")
}

pub fn decode_certificate(payload: &str) -> Result<RuliadProofCertificate> {
    let wire: WireCertificate =
        serde_json::from_str(payload).context("decode compact ruliad certificate")?;
    Ok(wire.into())
}

pub fn encode_model_certificate(certificate: &RuliadProofCertificate) -> Result<String> {
    let mut lines = Vec::new();
    for goal in &certificate.goals {
        for step in &goal.steps {
            lines.push(encode_model_proof_step(goal.goal, step));
        }
    }
    Ok(lines.join(";"))
}

/// Encode one executable proof action without a menu-position label.
pub fn encode_model_proof_step(goal: usize, step: &RuliadProofStep) -> String {
    let source = match &step.source {
        RuliadProofSource::Axiom { id } => format!("a:{id}"),
        RuliadProofSource::Lemma { goal } => format!("l:{goal}"),
    };
    let direction = match step.direction {
        RuliadRewriteDirection::Forward => "f",
        RuliadRewriteDirection::Reverse => "r",
    };
    let path = if step.path.is_empty() {
        "-".to_string()
    } else {
        step.path
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(".")
    };
    format!("g{goal}|{source}|{direction}|{path}")
}

/// Decode exactly one executable proof action.
pub fn decode_model_proof_step(payload: &str) -> Option<(usize, RuliadProofStep)> {
    let payload = payload.trim();
    (!payload.is_empty() && !payload.contains(';'))
        .then(|| decode_model_certificate_step(payload))
        .flatten()
}

pub fn decode_model_certificate(
    payload: &str,
    problem_version: u32,
    problem_hash: impl Into<String>,
) -> Result<RuliadProofCertificate> {
    let parsed = decode_model_certificate_prefix(payload, problem_version, problem_hash)?;
    if !parsed.syntax_complete {
        return Err(anyhow!("incomplete model-facing ruliad certificate"));
    }
    Ok(parsed.certificate)
}

pub fn decode_model_certificate_prefix(
    payload: &str,
    problem_version: u32,
    problem_hash: impl Into<String>,
) -> Result<RuliadModelCertificatePrefix> {
    let mut goals = Vec::<RuliadGoalCertificate>::new();
    let mut parsed_steps = 0usize;
    let mut syntax_complete = true;
    for line in payload.split(';') {
        if line.is_empty() {
            syntax_complete = false;
            break;
        }
        let Some((goal, step)) = decode_model_certificate_step(line) else {
            syntax_complete = false;
            break;
        };
        if let Some(existing) = goals.iter_mut().find(|candidate| candidate.goal == goal) {
            existing.steps.push(step);
        } else {
            goals.push(RuliadGoalCertificate {
                goal,
                steps: vec![step],
            });
        }
        parsed_steps = parsed_steps.saturating_add(1);
    }
    Ok(RuliadModelCertificatePrefix {
        certificate: RuliadProofCertificate {
            version: problem_version,
            problem_hash: problem_hash.into(),
            goals,
        },
        parsed_steps,
        syntax_complete,
    })
}

fn decode_model_certificate_step(line: &str) -> Option<(usize, RuliadProofStep)> {
    let mut fields = line.split('|');
    let goal = fields.next()?.strip_prefix('g')?.parse::<usize>().ok()?;
    let source = fields.next()?;
    let direction = fields.next()?;
    let path = fields.next()?;
    if fields.next().is_some() {
        return None;
    }
    let source = if let Some(id) = source.strip_prefix("a:") {
        RuliadProofSource::Axiom { id: id.to_string() }
    } else {
        let goal = source.strip_prefix("l:")?;
        RuliadProofSource::Lemma {
            goal: goal.parse().ok()?,
        }
    };
    let direction = match direction {
        "f" => RuliadRewriteDirection::Forward,
        "r" => RuliadRewriteDirection::Reverse,
        _ => return None,
    };
    let path = if path == "-" {
        Vec::new()
    } else {
        path.split('.')
            .map(str::parse::<usize>)
            .collect::<Result<Vec<_>, _>>()
            .ok()?
    };
    Some((
        goal,
        RuliadProofStep {
            source,
            path,
            direction,
        },
    ))
}

#[derive(Default)]
struct WireInterner {
    symbols: Vec<String>,
    symbol_indices: HashMap<String, usize>,
    nodes: Vec<WireNode>,
    node_indices: HashMap<RuliadTerm, usize>,
}

impl WireInterner {
    fn symbol(&mut self, symbol: &str) -> usize {
        if let Some(index) = self.symbol_indices.get(symbol) {
            return *index;
        }
        let index = self.symbols.len();
        self.symbols.push(symbol.to_string());
        self.symbol_indices.insert(symbol.to_string(), index);
        index
    }

    fn term(&mut self, term: &RuliadTerm) -> usize {
        if let Some(index) = self.node_indices.get(term) {
            return *index;
        }
        let node = match term {
            RuliadTerm::Variable { index } => WireNode::V(*index),
            RuliadTerm::Atom { symbol } => WireNode::A(self.symbol(symbol)),
            RuliadTerm::Apply {
                operator,
                arguments,
            } => {
                let operator = self.symbol(operator);
                let arguments = arguments
                    .iter()
                    .map(|argument| self.term(argument))
                    .collect();
                WireNode::P(operator, arguments)
            }
        };
        let index = self.nodes.len();
        self.nodes.push(node);
        self.node_indices.insert(term.clone(), index);
        index
    }
}

impl WireProblem {
    fn from_problem(problem: &RuliadProofProblem) -> Self {
        let mut interner = WireInterner::default();
        let axioms = problem
            .axioms
            .iter()
            .map(|axiom| {
                WireAxiom(
                    axiom.id.clone(),
                    interner.term(&axiom.lhs),
                    interner.term(&axiom.rhs),
                )
            })
            .collect();
        let goals = problem
            .goals
            .iter()
            .map(|goal| {
                WireGoal(
                    goal.id.clone(),
                    goal.dependencies.clone(),
                    interner.term(&goal.claim.lhs),
                    interner.term(&goal.claim.rhs),
                )
            })
            .collect();
        Self(
            problem.version,
            problem.domain,
            problem.theory.clone(),
            interner.symbols,
            interner.nodes,
            axioms,
            goals,
            problem.root,
        )
    }

    fn into_problem(self) -> Result<RuliadProofProblem> {
        let Self(version, domain, theory, symbols, wire_nodes, axioms, goals, root) = self;
        let mut nodes = Vec::<RuliadTerm>::with_capacity(wire_nodes.len());
        for (node_index, node) in wire_nodes.into_iter().enumerate() {
            let term = match node {
                WireNode::V(index) => RuliadTerm::variable(index),
                WireNode::A(symbol) => RuliadTerm::atom(
                    symbols
                        .get(symbol)
                        .ok_or_else(|| {
                            anyhow!("wire node {node_index} has invalid symbol {symbol}")
                        })?
                        .clone(),
                ),
                WireNode::P(operator, arguments) => {
                    let operator = symbols
                        .get(operator)
                        .ok_or_else(|| {
                            anyhow!("wire node {node_index} has invalid operator {operator}")
                        })?
                        .clone();
                    let arguments = arguments
                        .into_iter()
                        .map(|argument| {
                            nodes.get(argument).cloned().ok_or_else(|| {
                                anyhow!(
                                    "wire node {node_index} has non-topological argument {argument}"
                                )
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    RuliadTerm::apply(operator, arguments)
                }
            };
            nodes.push(term);
        }
        let term = |index: usize, location: &str| {
            nodes
                .get(index)
                .cloned()
                .ok_or_else(|| anyhow!("{location} references missing wire term {index}"))
        };
        Ok(RuliadProofProblem {
            version,
            domain,
            theory,
            axioms: axioms
                .into_iter()
                .map(|WireAxiom(id, lhs, rhs)| {
                    Ok(RuliadRewriteAxiom {
                        lhs: term(lhs, &format!("axiom {id} lhs"))?,
                        rhs: term(rhs, &format!("axiom {id} rhs"))?,
                        id,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            goals: goals
                .into_iter()
                .map(|WireGoal(id, dependencies, lhs, rhs)| {
                    Ok(RuliadProofGoal {
                        claim: RuliadEquality {
                            lhs: term(lhs, &format!("goal {id} lhs"))?,
                            rhs: term(rhs, &format!("goal {id} rhs"))?,
                        },
                        id,
                        dependencies,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            root,
        })
    }
}

impl From<&RuliadProofCertificate> for WireCertificate {
    fn from(certificate: &RuliadProofCertificate) -> Self {
        Self(
            certificate.version,
            certificate.problem_hash.clone(),
            certificate
                .goals
                .iter()
                .map(|goal| {
                    WireGoalCertificate(
                        goal.goal,
                        goal.steps
                            .iter()
                            .map(|step| {
                                let source = match &step.source {
                                    RuliadProofSource::Axiom { id } => WireSource::A(id.clone()),
                                    RuliadProofSource::Lemma { goal } => WireSource::L(*goal),
                                };
                                WireStep(
                                    source,
                                    step.direction == RuliadRewriteDirection::Reverse,
                                    step.path.clone(),
                                )
                            })
                            .collect(),
                    )
                })
                .collect(),
        )
    }
}

impl From<WireCertificate> for RuliadProofCertificate {
    fn from(certificate: WireCertificate) -> Self {
        let WireCertificate(version, problem_hash, goals) = certificate;
        Self {
            version,
            problem_hash,
            goals: goals
                .into_iter()
                .map(|WireGoalCertificate(goal, steps)| RuliadGoalCertificate {
                    goal,
                    steps: steps
                        .into_iter()
                        .map(|WireStep(source, reverse, path)| RuliadProofStep {
                            source: match source {
                                WireSource::A(id) => RuliadProofSource::Axiom { id },
                                WireSource::L(goal) => RuliadProofSource::Lemma { goal },
                            },
                            path,
                            direction: if reverse {
                                RuliadRewriteDirection::Reverse
                            } else {
                                RuliadRewriteDirection::Forward
                            },
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ruliad::formal::{RuliadFormalGeneratorConfig, generate_formal_bundle};

    #[test]
    fn compact_wire_round_trips_without_hash_drift() {
        let bundle =
            generate_formal_bundle(41, RuliadFormalGeneratorConfig::default()).expect("bundle");
        let problem_payload = encode_problem(&bundle.problem).expect("encode problem");
        let certificate_payload =
            encode_certificate(&bundle.certificate).expect("encode certificate");
        let problem = decode_problem(&problem_payload).expect("decode problem");
        let certificate = decode_certificate(&certificate_payload).expect("decode certificate");
        assert_eq!(problem, bundle.problem);
        assert_eq!(certificate, bundle.certificate);
        assert_eq!(
            problem.canonical_hash().expect("round trip hash"),
            bundle.problem.canonical_hash().expect("source hash")
        );
        assert!(!problem_payload.contains(char::is_whitespace));
        assert!(!certificate_payload.contains(char::is_whitespace));
    }

    #[test]
    fn compact_problem_wire_shares_repeated_terms() {
        let bundle = generate_formal_bundle(
            41,
            RuliadFormalGeneratorConfig {
                leaf_count: 32,
                ..RuliadFormalGeneratorConfig::default()
            },
        )
        .expect("bundle");
        let compact = encode_problem(&bundle.problem).expect("compact problem");
        let expanded = serde_json::to_string(&bundle.problem).expect("expanded problem");
        assert!(
            compact.len() * 2 < expanded.len(),
            "term DAG should materially compact repeated proof structure: compact={} expanded={}",
            compact.len(),
            expanded.len()
        );
    }

    #[test]
    fn model_certificate_omits_hash_and_binds_to_prompt_problem() {
        let bundle =
            generate_formal_bundle(43, RuliadFormalGeneratorConfig::default()).expect("bundle");
        let payload = encode_model_certificate(&bundle.certificate).expect("model certificate");
        assert!(!payload.contains(&bundle.certificate.problem_hash));
        assert!(payload.starts_with("g0|"), "{payload}");

        let decoded = decode_model_certificate(
            &payload,
            bundle.problem.version,
            bundle.certificate.problem_hash.clone(),
        )
        .expect("bound certificate");
        assert_eq!(decoded, bundle.certificate);
    }

    #[test]
    fn model_proof_step_round_trips_as_one_semantic_action() {
        let bundle =
            generate_formal_bundle(45, RuliadFormalGeneratorConfig::default()).expect("bundle");
        let goal = &bundle.certificate.goals[0];
        let step = &goal.steps[0];
        let payload = encode_model_proof_step(goal.goal, step);
        assert_eq!(
            decode_model_proof_step(&payload),
            Some((goal.goal, step.clone()))
        );
        assert!(decode_model_proof_step(&format!("{payload};{payload}")).is_none());
    }

    #[test]
    fn model_certificate_prefix_preserves_verifiable_steps_before_malformed_tail() {
        let bundle =
            generate_formal_bundle(47, RuliadFormalGeneratorConfig::default()).expect("bundle");
        let payload = encode_model_certificate(&bundle.certificate).expect("model certificate");
        let first_steps = payload.split(';').take(3).collect::<Vec<_>>().join(";");
        let truncated = format!("{first_steps};not-a-proof-step");

        let parsed = decode_model_certificate_prefix(
            &truncated,
            bundle.problem.version,
            bundle.certificate.problem_hash.clone(),
        )
        .expect("prefix");
        assert!(!parsed.syntax_complete);
        assert_eq!(parsed.parsed_steps, 3);
        assert_eq!(parsed.certificate.goals[0].steps.len(), 3);
    }

    #[test]
    fn compact_problem_wire_rejects_forward_term_references() {
        let malformed = serde_json::to_string(&WireProblem(
            3,
            RuliadFormalDomain::Equational,
            "t".to_string(),
            vec!["f".to_string()],
            vec![WireNode::P(0, vec![1])],
            Vec::new(),
            Vec::new(),
            0,
        ))
        .expect("malformed wire payload");
        let error = decode_problem(&malformed).expect_err("forward reference must fail");
        assert!(error.to_string().contains("non-topological argument"));
    }
}
