//! Deterministic proof-DAG generation over the portable Ruliad IR.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, anyhow};

use crate::ruliad::ir::{
    RULIAD_IR_VERSION, RuliadEquality, RuliadFormalDomain, RuliadGoalCertificate,
    RuliadProofBundle, RuliadProofCertificate, RuliadProofGoal, RuliadProofProblem,
    RuliadProofSource, RuliadProofStep, RuliadRewriteAxiom, RuliadRewriteDirection, RuliadTerm,
};
use crate::ruliad::kernel::{RuliadKernelLimits, replay_certificate};
use crate::ruliad::rng::SplitMix64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuliadFormalGeneratorConfig {
    pub domain: Option<RuliadFormalDomain>,
    pub rewrite_depth: usize,
    pub leaf_count: usize,
    pub context_depth: usize,
    pub distractor_axioms: usize,
    pub generation_split: RuliadFormalGenerationSplit,
}

/// Model-facing formal generation partition.
///
/// Structural partitions use local alpha-renamed symbols in both splits. The
/// training partition excludes one domain law and keeps the balanced proof
/// topology; validation uses only that held-out law and a left-fold topology.
/// The axiom itself remains present in the validation problem, so this tests
/// in-context symbolic use rather than inaccessible mathematical knowledge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RuliadFormalGenerationSplit {
    #[default]
    Shared,
    StructuralTrainV1,
    StructuralValidationV1,
}

impl RuliadFormalGenerationSplit {
    pub fn label(self) -> &'static str {
        match self {
            Self::Shared => "seed_disjoint_v1",
            Self::StructuralTrainV1 => "structural_train_v1",
            Self::StructuralValidationV1 => "structural_validation_v1",
        }
    }

    fn alpha_renamed(self) -> bool {
        !matches!(self, Self::Shared)
    }
}

impl Default for RuliadFormalGeneratorConfig {
    fn default() -> Self {
        Self {
            domain: None,
            rewrite_depth: 4,
            leaf_count: 4,
            context_depth: 2,
            distractor_axioms: 3,
            generation_split: RuliadFormalGenerationSplit::default(),
        }
    }
}

impl RuliadFormalGeneratorConfig {
    pub fn for_difficulty(level: usize) -> Self {
        let coordinate = level.saturating_add(1);
        let depth_bits = usize::BITS as usize - coordinate.leading_zeros() as usize;
        let dependency_bits = usize::BITS as usize - (coordinate / 2 + 1).leading_zeros() as usize;
        Self {
            domain: None,
            rewrite_depth: 2usize.saturating_add(depth_bits),
            leaf_count: 1usize
                .checked_shl(dependency_bits.min(12) as u32)
                .unwrap_or(4096)
                .clamp(2, 4096),
            // These coordinates grow logarithmically without cycling. Leaf count is a
            // bounded per-document resource; depth remains the open semantic axis.
            context_depth: 1usize.saturating_add(depth_bits / 2),
            distractor_axioms: depth_bits.saturating_add(dependency_bits / 2),
            generation_split: RuliadFormalGenerationSplit::default(),
        }
    }
}

pub fn generate_formal_bundle(
    seed: u64,
    config: RuliadFormalGeneratorConfig,
) -> Result<RuliadProofBundle> {
    if config.rewrite_depth == 0 || config.leaf_count == 0 {
        return Err(anyhow!(
            "formal generator requires non-zero depth and leaves"
        ));
    }
    let mut rng = SplitMix64::new(seed);
    let domain = config
        .domain
        .unwrap_or_else(|| RuliadFormalDomain::ALL[rng.next_usize(RuliadFormalDomain::ALL.len())]);
    let salt = rng.next_u64();
    let theory = domain_theory(domain);

    let selected_laws = selected_domain_laws(&theory, config.generation_split);
    let mut axioms = selected_laws
        .iter()
        .copied()
        .map(DomainLaw::axiom)
        .collect::<Vec<_>>();
    for index in 0..config.distractor_axioms {
        axioms.push(theory.derived_distractor(index, selected_laws));
    }

    let mut goals = Vec::new();
    let mut certificates = Vec::new();
    let mut frontier = Vec::new();
    for leaf in 0..config.leaf_count {
        let proof_laws = (0..config.rewrite_depth)
            .map(|_| selected_laws[rng.next_usize(selected_laws.len())])
            .collect::<Vec<_>>();
        let rewrite_path = (0..config.context_depth)
            .map(|_| rng.next_usize(2))
            .collect::<Vec<_>>();
        let atom = RuliadTerm::atom(format!("{}{}_{salt:08x}", theory.atom, leaf));
        let reducible = proof_laws
            .iter()
            .rev()
            .fold(atom.clone(), |term, law| law.wrap(term));
        let lhs = wrap_context(theory.context, salt, &rewrite_path, reducible);
        let rhs = wrap_context(theory.context, salt, &rewrite_path, atom);
        let goal_index = goals.len();
        goals.push(RuliadProofGoal {
            id: format!("g{goal_index}"),
            dependencies: Vec::new(),
            claim: RuliadEquality { lhs, rhs },
        });
        certificates.push(RuliadGoalCertificate {
            goal: goal_index,
            steps: (0..config.rewrite_depth)
                .map(|rule| RuliadProofStep {
                    source: RuliadProofSource::Axiom {
                        id: proof_laws[rule].id.to_string(),
                    },
                    path: rewrite_path.clone(),
                    direction: RuliadRewriteDirection::Forward,
                })
                .collect(),
        });
        frontier.push(goal_index);
    }

    match config.generation_split {
        RuliadFormalGenerationSplit::StructuralValidationV1 => {
            let mut root = frontier[0];
            for right in frontier.iter().copied().skip(1) {
                root = append_composition_goal(&theory, &mut goals, &mut certificates, root, right);
            }
            frontier = vec![root];
        }
        RuliadFormalGenerationSplit::Shared | RuliadFormalGenerationSplit::StructuralTrainV1 => {
            while frontier.len() > 1 {
                if matches!(config.generation_split, RuliadFormalGenerationSplit::Shared) {
                    for index in (1..frontier.len()).rev() {
                        let swap = rng.next_usize(index + 1);
                        frontier.swap(index, swap);
                    }
                }
                let mut next = Vec::with_capacity(frontier.len().div_ceil(2));
                for pair in frontier.chunks(2) {
                    if pair.len() == 1 {
                        next.push(pair[0]);
                        continue;
                    }
                    next.push(append_composition_goal(
                        &theory,
                        &mut goals,
                        &mut certificates,
                        pair[0],
                        pair[1],
                    ));
                }
                frontier = next;
            }
        }
    }

    let problem = RuliadProofProblem {
        version: RULIAD_IR_VERSION,
        domain,
        theory: theory.name.to_string(),
        axioms,
        root: frontier[0],
        goals,
    };
    let certificate = RuliadProofCertificate {
        version: RULIAD_IR_VERSION,
        problem_hash: String::new(),
        goals: certificates,
    };
    let mut bundle = RuliadProofBundle {
        problem,
        certificate,
    };
    if config.generation_split.alpha_renamed() {
        alpha_rename_bundle(&mut bundle, rng.next_u64());
    }
    bundle.certificate.problem_hash = bundle.problem.canonical_hash()?;
    let report = replay_certificate(
        &bundle.problem,
        &bundle.certificate,
        RuliadKernelLimits::default(),
    );
    if !report.accepted {
        return Err(anyhow!(
            "generated formal proof failed replay: {:?}",
            report.failure
        ));
    }
    Ok(bundle)
}

fn selected_domain_laws(
    theory: &DomainTheory,
    generation_split: RuliadFormalGenerationSplit,
) -> &[DomainLaw] {
    match generation_split {
        RuliadFormalGenerationSplit::Shared => theory.laws,
        RuliadFormalGenerationSplit::StructuralTrainV1 if theory.laws.len() > 1 => {
            &theory.laws[..theory.laws.len() - 1]
        }
        RuliadFormalGenerationSplit::StructuralValidationV1 if theory.laws.len() > 1 => {
            &theory.laws[theory.laws.len() - 1..]
        }
        RuliadFormalGenerationSplit::StructuralTrainV1
        | RuliadFormalGenerationSplit::StructuralValidationV1 => theory.laws,
    }
}

fn append_composition_goal(
    theory: &DomainTheory,
    goals: &mut Vec<RuliadProofGoal>,
    certificates: &mut Vec<RuliadGoalCertificate>,
    left: usize,
    right: usize,
) -> usize {
    let (left, right) = if left < right {
        (left, right)
    } else {
        (right, left)
    };
    let goal_index = goals.len();
    goals.push(RuliadProofGoal {
        id: format!("g{goal_index}"),
        dependencies: vec![left, right],
        claim: RuliadEquality {
            lhs: RuliadTerm::apply(
                theory.combine,
                vec![
                    goals[left].claim.lhs.clone(),
                    goals[right].claim.lhs.clone(),
                ],
            ),
            rhs: RuliadTerm::apply(
                theory.combine,
                vec![
                    goals[left].claim.rhs.clone(),
                    goals[right].claim.rhs.clone(),
                ],
            ),
        },
    });
    certificates.push(RuliadGoalCertificate {
        goal: goal_index,
        steps: vec![
            RuliadProofStep {
                source: RuliadProofSource::Lemma { goal: left },
                path: vec![0],
                direction: RuliadRewriteDirection::Forward,
            },
            RuliadProofStep {
                source: RuliadProofSource::Lemma { goal: right },
                path: vec![1],
                direction: RuliadRewriteDirection::Forward,
            },
        ],
    });
    goal_index
}

fn alpha_rename_bundle(bundle: &mut RuliadProofBundle, seed: u64) {
    let mut axiom_ids = BTreeSet::new();
    let mut operators = BTreeSet::new();
    let mut atoms = BTreeSet::new();
    let mut goal_ids = BTreeSet::new();
    for axiom in &bundle.problem.axioms {
        axiom_ids.insert(axiom.id.clone());
        collect_term_symbols(&axiom.lhs, &mut operators, &mut atoms);
        collect_term_symbols(&axiom.rhs, &mut operators, &mut atoms);
    }
    for goal in &bundle.problem.goals {
        goal_ids.insert(goal.id.clone());
        collect_term_symbols(&goal.claim.lhs, &mut operators, &mut atoms);
        collect_term_symbols(&goal.claim.rhs, &mut operators, &mut atoms);
    }

    let mut rng = SplitMix64::new(seed);
    let axiom_map = local_symbol_map(axiom_ids, "r", &mut rng);
    let operator_map = local_symbol_map(operators, "f", &mut rng);
    let atom_map = local_symbol_map(atoms, "c", &mut rng);
    let goal_map = local_symbol_map(goal_ids, "g", &mut rng);

    bundle.problem.theory = format!("t{}", rng.next_usize(32));
    for axiom in &mut bundle.problem.axioms {
        axiom.id = axiom_map[&axiom.id].clone();
        rename_term_symbols(&mut axiom.lhs, &operator_map, &atom_map);
        rename_term_symbols(&mut axiom.rhs, &operator_map, &atom_map);
    }
    for goal in &mut bundle.problem.goals {
        goal.id = goal_map[&goal.id].clone();
        rename_term_symbols(&mut goal.claim.lhs, &operator_map, &atom_map);
        rename_term_symbols(&mut goal.claim.rhs, &operator_map, &atom_map);
    }
    for goal in &mut bundle.certificate.goals {
        for step in &mut goal.steps {
            if let RuliadProofSource::Axiom { id } = &mut step.source {
                *id = axiom_map[id].clone();
            }
        }
    }
    shuffle(&mut bundle.problem.axioms, &mut rng);
}

fn collect_term_symbols(
    term: &RuliadTerm,
    operators: &mut BTreeSet<String>,
    atoms: &mut BTreeSet<String>,
) {
    match term {
        RuliadTerm::Variable { .. } => {}
        RuliadTerm::Atom { symbol } => {
            atoms.insert(symbol.clone());
        }
        RuliadTerm::Apply {
            operator,
            arguments,
        } => {
            operators.insert(operator.clone());
            for argument in arguments {
                collect_term_symbols(argument, operators, atoms);
            }
        }
    }
}

fn rename_term_symbols(
    term: &mut RuliadTerm,
    operators: &BTreeMap<String, String>,
    atoms: &BTreeMap<String, String>,
) {
    match term {
        RuliadTerm::Variable { .. } => {}
        RuliadTerm::Atom { symbol } => *symbol = atoms[symbol].clone(),
        RuliadTerm::Apply {
            operator,
            arguments,
        } => {
            *operator = operators[operator].clone();
            for argument in arguments {
                rename_term_symbols(argument, operators, atoms);
            }
        }
    }
}

fn local_symbol_map(
    symbols: BTreeSet<String>,
    prefix: &str,
    rng: &mut SplitMix64,
) -> BTreeMap<String, String> {
    let mut labels = (0..symbols.len())
        .map(|index| format!("{prefix}{index}"))
        .collect::<Vec<_>>();
    shuffle(&mut labels, rng);
    symbols.into_iter().zip(labels).collect()
}

fn shuffle<T>(values: &mut [T], rng: &mut SplitMix64) {
    for index in (1..values.len()).rev() {
        values.swap(index, rng.next_usize(index + 1));
    }
}

pub fn corrupt_formal_certificate(
    certificate: &RuliadProofCertificate,
) -> Result<RuliadProofCertificate> {
    let mut corrupted = certificate.clone();
    let node = corrupted
        .goals
        .last_mut()
        .ok_or_else(|| anyhow!("formal certificate has no goals"))?;
    let step = node
        .steps
        .first_mut()
        .ok_or_else(|| anyhow!("formal root certificate has no steps"))?;
    step.path.insert(0, usize::MAX);
    Ok(corrupted)
}

#[derive(Clone, Copy)]
enum DomainLawShape {
    LeftUnit {
        operator: &'static str,
        unit: &'static str,
    },
    RightUnit {
        operator: &'static str,
        unit: &'static str,
    },
    CancelPair {
        outer: &'static str,
        inner: &'static str,
    },
}

#[derive(Clone, Copy)]
struct DomainLaw {
    id: &'static str,
    shape: DomainLawShape,
}

impl DomainLaw {
    fn wrap(self, term: RuliadTerm) -> RuliadTerm {
        match self.shape {
            DomainLawShape::LeftUnit { operator, unit } => {
                RuliadTerm::apply(operator, vec![RuliadTerm::atom(unit), term])
            }
            DomainLawShape::RightUnit { operator, unit } => {
                RuliadTerm::apply(operator, vec![term, RuliadTerm::atom(unit)])
            }
            DomainLawShape::CancelPair { outer, inner } => {
                RuliadTerm::apply(outer, vec![RuliadTerm::apply(inner, vec![term])])
            }
        }
    }

    fn axiom(self) -> RuliadRewriteAxiom {
        let variable = RuliadTerm::variable(0);
        RuliadRewriteAxiom {
            id: self.id.to_string(),
            lhs: self.wrap(variable.clone()),
            rhs: variable,
        }
    }
}

struct DomainTheory {
    name: &'static str,
    atom: &'static str,
    context: &'static str,
    combine: &'static str,
    laws: &'static [DomainLaw],
}

impl DomainTheory {
    fn derived_distractor(&self, index: usize, selected_laws: &[DomainLaw]) -> RuliadRewriteAxiom {
        if index == 0 {
            let x = RuliadTerm::variable(0);
            let y = RuliadTerm::variable(1);
            let z = RuliadTerm::variable(2);
            return RuliadRewriteAxiom {
                id: "aux_assoc".to_string(),
                lhs: RuliadTerm::apply(
                    self.combine,
                    vec![
                        RuliadTerm::apply(self.combine, vec![x.clone(), y.clone()]),
                        z.clone(),
                    ],
                ),
                rhs: RuliadTerm::apply(
                    self.combine,
                    vec![x, RuliadTerm::apply(self.combine, vec![y, z])],
                ),
            };
        }
        let variable = RuliadTerm::variable(0);
        let lhs = (0..=index).fold(variable.clone(), |term, law_index| {
            selected_laws[law_index % selected_laws.len()].wrap(term)
        });
        RuliadRewriteAxiom {
            id: format!("aux_derived_{index}"),
            lhs,
            rhs: variable,
        }
    }
}

const EQUATIONAL_LAWS: &[DomainLaw] = &[
    DomainLaw {
        id: "add_zero_left",
        shape: DomainLawShape::LeftUnit {
            operator: "add",
            unit: "zero",
        },
    },
    DomainLaw {
        id: "mul_one_right",
        shape: DomainLawShape::RightUnit {
            operator: "mul",
            unit: "one",
        },
    },
    DomainLaw {
        id: "double_negation",
        shape: DomainLawShape::CancelPair {
            outer: "neg",
            inner: "neg",
        },
    },
];

const CATEGORY_LAWS: &[DomainLaw] = &[
    DomainLaw {
        id: "identity_left",
        shape: DomainLawShape::LeftUnit {
            operator: "compose",
            unit: "identity",
        },
    },
    DomainLaw {
        id: "identity_right",
        shape: DomainLawShape::RightUnit {
            operator: "compose",
            unit: "identity",
        },
    },
];

const LOGIC_LAWS: &[DomainLaw] = &[
    DomainLaw {
        id: "and_top",
        shape: DomainLawShape::LeftUnit {
            operator: "and",
            unit: "top",
        },
    },
    DomainLaw {
        id: "or_bottom",
        shape: DomainLawShape::LeftUnit {
            operator: "or",
            unit: "bottom",
        },
    },
    DomainLaw {
        id: "double_negation",
        shape: DomainLawShape::CancelPair {
            outer: "not",
            inner: "not",
        },
    },
];

const AUTOMATA_LAWS: &[DomainLaw] = &[
    DomainLaw {
        id: "epsilon_prefix",
        shape: DomainLawShape::LeftUnit {
            operator: "concat",
            unit: "epsilon",
        },
    },
    DomainLaw {
        id: "empty_union",
        shape: DomainLawShape::LeftUnit {
            operator: "union",
            unit: "empty_language",
        },
    },
    DomainLaw {
        id: "encode_decode",
        shape: DomainLawShape::CancelPair {
            outer: "decode_state",
            inner: "encode_state",
        },
    },
];

const PROCESS_LAWS: &[DomainLaw] = &[
    DomainLaw {
        id: "nil_parallel",
        shape: DomainLawShape::LeftUnit {
            operator: "parallel",
            unit: "nil",
        },
    },
    DomainLaw {
        id: "nil_choice",
        shape: DomainLawShape::LeftUnit {
            operator: "choice",
            unit: "nil",
        },
    },
    DomainLaw {
        id: "quote_eval",
        shape: DomainLawShape::CancelPair {
            outer: "unquote",
            inner: "quote",
        },
    },
];

const METAGRAPH_LAWS: &[DomainLaw] = &[
    DomainLaw {
        id: "empty_merge",
        shape: DomainLawShape::LeftUnit {
            operator: "merge",
            unit: "empty_graph",
        },
    },
    DomainLaw {
        id: "empty_overlay",
        shape: DomainLawShape::LeftUnit {
            operator: "overlay",
            unit: "empty_graph",
        },
    },
    DomainLaw {
        id: "quote_unquote",
        shape: DomainLawShape::CancelPair {
            outer: "unquote_atom",
            inner: "quote_atom",
        },
    },
];

fn domain_theory(domain: RuliadFormalDomain) -> DomainTheory {
    match domain {
        RuliadFormalDomain::Equational => DomainTheory {
            name: "equational_monoid_normalization",
            atom: "a",
            context: "equational_context",
            combine: "conjoin_equalities",
            laws: EQUATIONAL_LAWS,
        },
        RuliadFormalDomain::Category => DomainTheory {
            name: "free_category_path_normalization",
            atom: "morphism",
            context: "functor_map",
            combine: "conjoin_diagrams",
            laws: CATEGORY_LAWS,
        },
        RuliadFormalDomain::Logic => DomainTheory {
            name: "propositional_normalization",
            atom: "prop",
            context: "under_assumption",
            combine: "conjoin",
            laws: LOGIC_LAWS,
        },
        RuliadFormalDomain::Automata => DomainTheory {
            name: "regular_language_normalization",
            atom: "language",
            context: "left_quotient",
            combine: "language_product",
            laws: AUTOMATA_LAWS,
        },
        RuliadFormalDomain::Process => DomainTheory {
            name: "rho_process_structural_congruence",
            atom: "process",
            context: "process_context",
            combine: "parallel",
            laws: PROCESS_LAWS,
        },
        RuliadFormalDomain::Metagraph => DomainTheory {
            name: "metagraph_pattern_rewriting",
            atom: "atom",
            context: "grounded_scope",
            combine: "merge",
            laws: METAGRAPH_LAWS,
        },
    }
}

fn wrap_context(context: &str, salt: u64, path: &[usize], mut term: RuliadTerm) -> RuliadTerm {
    for (index, hole) in path.iter().rev().copied().enumerate() {
        let scope = RuliadTerm::atom(format!("scope{index}_{salt:08x}"));
        let arguments = if hole == 0 {
            vec![term, scope]
        } else {
            vec![scope, term]
        };
        term = RuliadTerm::apply(context, arguments);
    }
    term
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ruliad::kernel::{RuliadKernelFailureKind, complexity_vector};

    #[test]
    fn formal_generation_is_deterministic_and_domain_spanning() {
        let left =
            generate_formal_bundle(29, RuliadFormalGeneratorConfig::default()).expect("left");
        let right =
            generate_formal_bundle(29, RuliadFormalGeneratorConfig::default()).expect("right");
        assert_eq!(left, right);

        let domains = (0..256)
            .map(|seed| {
                generate_formal_bundle(seed, RuliadFormalGeneratorConfig::default())
                    .expect("bundle")
                    .problem
                    .domain
            })
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(domains.len(), RuliadFormalDomain::ALL.len());
    }

    #[test]
    fn proof_dag_dependencies_are_semantically_required() {
        let mut bundle =
            generate_formal_bundle(31, RuliadFormalGeneratorConfig::default()).expect("bundle");
        let root = bundle.problem.root;
        bundle.problem.goals[root].dependencies.remove(0);
        bundle.certificate.problem_hash = bundle.problem.canonical_hash().expect("hash");
        let report = replay_certificate(
            &bundle.problem,
            &bundle.certificate,
            RuliadKernelLimits::default(),
        );
        assert!(!report.accepted);
        assert_eq!(
            report.failure.as_ref().map(|failure| failure.kind),
            Some(RuliadKernelFailureKind::MalformedCertificate)
        );
    }

    #[test]
    fn higher_difficulty_expands_a_complexity_coordinate() {
        let easy = generate_formal_bundle(37, RuliadFormalGeneratorConfig::for_difficulty(0))
            .expect("easy");
        let hard = generate_formal_bundle(37, RuliadFormalGeneratorConfig::for_difficulty(32))
            .expect("hard");
        let easy = complexity_vector(&easy.problem, Some(&easy.certificate));
        let hard = complexity_vector(&hard.problem, Some(&hard.certificate));
        assert!(hard.dominates(&easy), "easy={easy:?} hard={hard:?}");
    }

    #[test]
    fn difficulty_never_wraps_context_depth() {
        let levels = [7usize, 15, 31, 63, 127, 255, 511, 1023];
        let depths =
            levels.map(|level| RuliadFormalGeneratorConfig::for_difficulty(level).context_depth);
        assert!(depths.windows(2).all(|pair| pair[1] >= pair[0]));
        assert!(depths.last() > depths.first(), "depths={depths:?}");
    }

    #[test]
    fn generated_proofs_vary_law_sequences_and_context_paths() {
        let bundle = generate_formal_bundle(
            47,
            RuliadFormalGeneratorConfig {
                rewrite_depth: 8,
                leaf_count: 16,
                context_depth: 6,
                ..RuliadFormalGeneratorConfig::default()
            },
        )
        .expect("varied bundle");
        let signatures = bundle
            .certificate
            .goals
            .iter()
            .take(16)
            .map(|goal| {
                goal.steps
                    .iter()
                    .map(|step| format!("{:?}:{:?}", step.source, step.path))
                    .collect::<Vec<_>>()
            })
            .collect::<BTreeSet<_>>();
        assert!(signatures.len() > 4, "signatures={signatures:?}");
    }

    #[test]
    fn every_domain_frontend_uses_named_semantic_laws() {
        for domain in RuliadFormalDomain::ALL {
            let bundle = generate_formal_bundle(
                43,
                RuliadFormalGeneratorConfig {
                    domain: Some(domain),
                    ..RuliadFormalGeneratorConfig::default()
                },
            )
            .expect("domain bundle");
            assert_eq!(bundle.problem.domain, domain);
            assert!(!bundle.problem.theory.contains("43"));
            assert!(
                bundle
                    .problem
                    .axioms
                    .iter()
                    .any(|axiom| !axiom.id.starts_with("aux_"))
            );
            let report = replay_certificate(
                &bundle.problem,
                &bundle.certificate,
                RuliadKernelLimits::default(),
            );
            assert!(report.accepted, "domain={domain:?}: {report:?}");
        }
    }

    #[test]
    fn structural_validation_holds_out_law_and_dependency_topology() {
        let base = RuliadFormalGeneratorConfig {
            domain: Some(RuliadFormalDomain::Category),
            rewrite_depth: 3,
            leaf_count: 8,
            context_depth: 2,
            distractor_axioms: 0,
            generation_split: RuliadFormalGenerationSplit::StructuralTrainV1,
        };
        let train = generate_formal_bundle(53, base).expect("structural train");
        let validation = generate_formal_bundle(
            53,
            RuliadFormalGeneratorConfig {
                generation_split: RuliadFormalGenerationSplit::StructuralValidationV1,
                ..base
            },
        )
        .expect("structural validation");

        let train_lhs = &train.problem.axioms[0].lhs;
        let validation_lhs = &validation.problem.axioms[0].lhs;
        assert!(matches!(
            train_lhs,
            RuliadTerm::Apply { arguments, .. }
                if matches!(arguments.as_slice(), [RuliadTerm::Atom { .. }, RuliadTerm::Variable { .. }])
        ));
        assert!(matches!(
            validation_lhs,
            RuliadTerm::Apply { arguments, .. }
                if matches!(arguments.as_slice(), [RuliadTerm::Variable { .. }, RuliadTerm::Atom { .. }])
        ));

        let train_complexity = complexity_vector(&train.problem, Some(&train.certificate));
        let validation_complexity =
            complexity_vector(&validation.problem, Some(&validation.certificate));
        assert!(
            validation_complexity.dependency_depth > train_complexity.dependency_depth,
            "train={train_complexity:?} validation={validation_complexity:?}"
        );
        let train_root_dependencies = &train.problem.goals[train.problem.root].dependencies;
        let validation_root_dependencies =
            &validation.problem.goals[validation.problem.root].dependencies;
        assert!(
            train_root_dependencies
                .iter()
                .all(|dependency| *dependency >= 8)
        );
        assert!(
            validation_root_dependencies
                .iter()
                .any(|dependency| *dependency < 8)
        );

        for bundle in [&train, &validation] {
            assert!(bundle.problem.theory.starts_with('t'));
            assert!(
                bundle
                    .problem
                    .axioms
                    .iter()
                    .all(|axiom| axiom.id.starts_with('r'))
            );
            for axiom in &bundle.problem.axioms {
                assert_local_symbols(&axiom.lhs);
                assert_local_symbols(&axiom.rhs);
            }
            for goal in &bundle.problem.goals {
                assert_local_symbols(&goal.claim.lhs);
                assert_local_symbols(&goal.claim.rhs);
            }
            let report = replay_certificate(
                &bundle.problem,
                &bundle.certificate,
                RuliadKernelLimits::default(),
            );
            assert!(report.accepted, "{report:?}");
        }
    }

    fn assert_local_symbols(term: &RuliadTerm) {
        match term {
            RuliadTerm::Variable { .. } => {}
            RuliadTerm::Atom { symbol } => assert!(symbol.starts_with('c'), "{symbol}"),
            RuliadTerm::Apply {
                operator,
                arguments,
            } => {
                assert!(operator.starts_with('f'), "{operator}");
                for argument in arguments {
                    assert_local_symbols(argument);
                }
            }
        }
    }
}
