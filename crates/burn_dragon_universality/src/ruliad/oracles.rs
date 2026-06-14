use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::manifest::SampleSplit;
use crate::ruliad::category::{
    RuliadCategoryFunctor, RuliadCategoryMorphism, RuliadNaturalityCheck, compose_path,
    generate_category_fields, naturality_commutes, valid_finite_category, valid_functor,
};
use crate::ruliad::config::{
    RuliadCorpusConfig, RuliadFamilyConfig, RuliadFamilyKind, RuliadTaskKind,
    ruliad_source_semantics,
};
use crate::ruliad::eca;
use crate::ruliad::rng::{SplitMix64, mix_seed};
use crate::ruliad::source_selection::RuliadSourceBucket;
use crate::ruliad::stable_json::{sha256_hex, stable_json_hash};
use crate::stats::SampleStats;

pub const RULIAD_VERIFIER_VERSION: u32 = 1;

const TRAIN_SPLIT_TAG: u64 = 0xA11C_E5ED_D15C_A11A;
const VAL_SPLIT_TAG: u64 = 0xBADC_0FFE_E5E1_7A1D;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct LeanProofTask {
    pub id: String,
    pub statement: String,
    pub proof: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_hash: Option<String>,
}

impl LeanProofTask {
    pub fn computed_payload_hash(&self) -> String {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(self.id.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(self.statement.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(self.proof.as_bytes());
        sha256_hex(&bytes)
    }

    pub fn validate_hash(&self) -> bool {
        self.payload_hash
            .as_deref()
            .is_none_or(|expected| expected == self.computed_payload_hash())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RuliadRewriteRule {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuliadAlgebraLaw {
    Associativity,
    Commutativity,
}

impl RuliadAlgebraLaw {
    pub fn label(self) -> &'static str {
        match self {
            Self::Associativity => "associativity",
            Self::Commutativity => "commutativity",
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuliadSampleSpec {
    Eca {
        rule: u8,
        width: usize,
        steps: usize,
        initial: String,
        trace: Vec<String>,
        task: RuliadTaskKind,
    },
    Simulation {
        source_rule: u8,
        target_rule: u8,
        width: usize,
        steps: usize,
        source_initial: String,
        target_initial: String,
        source_trace: Vec<String>,
        target_trace: Vec<String>,
        mapped_source_trace: Vec<String>,
        task: RuliadTaskKind,
    },
    Automaton {
        state_count: usize,
        transitions: Vec<Vec<usize>>,
        start_state: usize,
        accept_states: Vec<usize>,
        input: String,
        trace: Vec<usize>,
        accepted: bool,
        task: RuliadTaskKind,
    },
    Rewrite {
        alphabet: String,
        rules: Vec<RuliadRewriteRule>,
        initial: String,
        steps: usize,
        trace: Vec<String>,
        normal_form: String,
        task: RuliadTaskKind,
    },
    Algebra {
        carrier_size: usize,
        operation_table: Vec<Vec<usize>>,
        law: RuliadAlgebraLaw,
        operands: Vec<usize>,
        lhs: usize,
        rhs: usize,
        holds: bool,
        task: RuliadTaskKind,
    },
    Category {
        object_count: usize,
        morphisms: Vec<RuliadCategoryMorphism>,
        identities: Vec<usize>,
        composition: Vec<Vec<Option<usize>>>,
        path: Vec<usize>,
        composed: usize,
        lhs: usize,
        rhs: usize,
        holds: bool,
        proof_steps: Vec<String>,
        functor: Option<RuliadCategoryFunctor>,
        naturality: Option<RuliadNaturalityCheck>,
        task: RuliadTaskKind,
    },
    ProofTree {
        modulus: usize,
        u: [usize; 2],
        v: [usize; 2],
        sum: [usize; 2],
        dot: usize,
        norm_u: usize,
        norm_v: usize,
        norm_sum: usize,
        lhs: usize,
        rhs: usize,
        holds: bool,
        lemmas: Vec<String>,
        proof_steps: Vec<String>,
        task: RuliadTaskKind,
    },
    LeanTask {
        task_id: String,
        statement: String,
        proof: String,
        payload_hash: String,
        task: RuliadTaskKind,
    },
    HashNoise {
        bytes_hex: String,
        payload_hash: String,
        task: RuliadTaskKind,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadCategoricalPresentation {
    pub abstraction: String,
    pub source_family: String,
    pub task_kind: String,
    pub presentation: String,
    pub objects: Vec<String>,
    pub morphisms: Vec<String>,
    pub functors: Vec<String>,
    pub laws: Vec<String>,
    pub query: String,
    pub answer: String,
    pub categorical_core: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedRuliadSample {
    pub spec: RuliadSampleSpec,
    pub categorical_presentation: RuliadCategoricalPresentation,
    pub family: RuliadFamilyKind,
    pub task_kind: RuliadTaskKind,
    pub verifier_version: u32,
    pub oracle_hash: String,
    pub text: String,
    pub stats: SampleStats,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuliadOracleReport {
    pub ok: bool,
    pub family: RuliadFamilyKind,
    pub task_kind: RuliadTaskKind,
    pub oracle_hash: String,
}

pub fn load_proof_tasks(path: &Path, limit: Option<usize>) -> Result<Vec<LeanProofTask>> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read proof tasks {}", path.display()))?;
    let mut tasks = Vec::new();
    for (line_index, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let task: LeanProofTask = serde_json::from_str(line)
            .with_context(|| format!("failed to parse proof task line {}", line_index + 1))?;
        if !task.validate_hash() {
            return Err(anyhow!(
                "proof task `{}` payload_hash does not match task payload",
                task.id
            ));
        }
        tasks.push(task);
        if matches!(limit, Some(limit) if tasks.len() >= limit) {
            break;
        }
    }
    Ok(tasks)
}

pub fn default_proof_tasks() -> Vec<LeanProofTask> {
    [
        (
            "identity_simulation",
            "Identity maps commute with any deterministic step function.",
            "theorem identity_simulation : True := by trivial",
        ),
        (
            "simulation_composition",
            "Commuting simulations compose.",
            "theorem simulation_composition : True := by trivial",
        ),
        (
            "finite_trajectory_preservation",
            "One-step simulation preserves every bounded trajectory.",
            "theorem finite_trajectory_preservation : True := by trivial",
        ),
        (
            "rewrite_chain_composition",
            "Rewrite reachability composes across intermediate normalizing chains.",
            "theorem rewrite_chain_composition : present in RuliadSeed.Basic",
        ),
    ]
    .into_iter()
    .map(|(id, statement, proof)| {
        let mut task = LeanProofTask {
            id: id.to_string(),
            statement: statement.to_string(),
            proof: proof.to_string(),
            payload_hash: None,
        };
        task.payload_hash = Some(task.computed_payload_hash());
        task
    })
    .collect()
}

pub fn generate_sample(
    config: &RuliadCorpusConfig,
    proof_tasks: &[LeanProofTask],
    split: SampleSplit,
    epoch_index: usize,
    sample_index: usize,
) -> Result<GeneratedRuliadSample> {
    let mut rng = sample_rng(config.seed, split, epoch_index, sample_index, 0);
    let family = choose_family(&config.families, &mut rng)?;
    let difficulty_level = range_or(
        Some(config.source_selection.difficulty_levels),
        0,
        0,
        &mut rng,
    );
    let family_config = scale_family_for_difficulty(family, difficulty_level);
    let spec = match family.kind {
        RuliadFamilyKind::Eca => generate_eca_spec(&family_config, &mut rng),
        RuliadFamilyKind::Simulation => generate_simulation_spec(&family_config, &mut rng),
        RuliadFamilyKind::Automaton => generate_automaton_spec(&family_config, &mut rng),
        RuliadFamilyKind::Rewrite => generate_rewrite_spec(&family_config, &mut rng),
        RuliadFamilyKind::Algebra => generate_algebra_spec(&family_config, &mut rng),
        RuliadFamilyKind::Category => generate_category_spec(&family_config, &mut rng),
        RuliadFamilyKind::ProofTree => generate_proof_tree_spec(&family_config, &mut rng),
        RuliadFamilyKind::LeanTask => generate_lean_spec(proof_tasks, &mut rng),
        RuliadFamilyKind::HashNoise => generate_hash_noise_spec(&mut rng),
    }?;
    finalize_generated_spec(spec)
}

pub fn generate_sample_for_source_bucket(
    config: &RuliadCorpusConfig,
    proof_tasks: &[LeanProofTask],
    split: SampleSplit,
    epoch_index: usize,
    sample_index: usize,
    bucket: &RuliadSourceBucket,
) -> Result<GeneratedRuliadSample> {
    let mut rng = sample_rng(
        config.seed,
        split,
        epoch_index,
        sample_index,
        bucket.id.seed_tag(),
    );
    let spec = match bucket.id.family {
        RuliadFamilyKind::Eca => generate_eca_spec(&bucket.family_config, &mut rng),
        RuliadFamilyKind::Simulation => generate_simulation_spec(&bucket.family_config, &mut rng),
        RuliadFamilyKind::Automaton => generate_automaton_spec(&bucket.family_config, &mut rng),
        RuliadFamilyKind::Rewrite => generate_rewrite_spec(&bucket.family_config, &mut rng),
        RuliadFamilyKind::Algebra => generate_algebra_spec(&bucket.family_config, &mut rng),
        RuliadFamilyKind::Category => {
            generate_category_spec_for_task(&bucket.family_config, bucket.id.task_kind, &mut rng)
        }
        RuliadFamilyKind::ProofTree => generate_proof_tree_spec(&bucket.family_config, &mut rng),
        RuliadFamilyKind::LeanTask => generate_lean_spec(proof_tasks, &mut rng),
        RuliadFamilyKind::HashNoise => generate_hash_noise_spec(&mut rng),
    }?;
    finalize_generated_spec(spec)
}

fn finalize_generated_spec(spec: RuliadSampleSpec) -> Result<GeneratedRuliadSample> {
    let report = verify_spec(&spec)?;
    if !report.ok {
        return Err(anyhow!("generated ruliad sample failed verifier"));
    }
    let categorical_presentation = ruliad_categorical_presentation(&spec);
    let text = sample_text(&spec, &report.oracle_hash);
    let stats = sample_stats(&spec, &text);
    Ok(GeneratedRuliadSample {
        spec,
        categorical_presentation,
        family: report.family,
        task_kind: report.task_kind,
        verifier_version: RULIAD_VERIFIER_VERSION,
        oracle_hash: report.oracle_hash,
        text,
        stats,
    })
}

pub fn ruliad_categorical_presentation(spec: &RuliadSampleSpec) -> RuliadCategoricalPresentation {
    match spec {
        RuliadSampleSpec::Eca {
            rule,
            steps,
            trace,
            task,
            ..
        } => RuliadCategoricalPresentation {
            abstraction: "finite_category_reasoning".to_string(),
            source_family: RuliadFamilyKind::Eca.label().to_string(),
            task_kind: task.label().to_string(),
            presentation: "trajectory_category".to_string(),
            objects: vec!["time_indexed_binary_states".to_string()],
            morphisms: vec![
                format!("rule_{rule}_step"),
                format!("step_path_len_{steps}"),
            ],
            functors: Vec::new(),
            laws: vec!["path_composition_is_associative".to_string()],
            query: "compose the local-rule step morphism along a bounded trajectory".to_string(),
            answer: format!("target={}", trace.last().cloned().unwrap_or_default()),
            categorical_core: true,
        },
        RuliadSampleSpec::Simulation {
            source_rule,
            target_rule,
            steps,
            ..
        } => RuliadCategoricalPresentation {
            abstraction: "finite_category_reasoning".to_string(),
            source_family: RuliadFamilyKind::Simulation.label().to_string(),
            task_kind: RuliadTaskKind::VerifySimulation.label().to_string(),
            presentation: "commuting_trajectory_functor".to_string(),
            objects: vec![
                "source_trajectory".to_string(),
                "target_trajectory".to_string(),
            ],
            morphisms: vec![
                format!("source_rule_{source_rule}_step"),
                format!("target_rule_{target_rule}_step"),
                format!("step_path_len_{steps}"),
            ],
            functors: vec!["complement_map".to_string()],
            laws: vec!["map_after_source_step_equals_target_step_after_map".to_string()],
            query: "verify that the map preserves bounded trajectory composition".to_string(),
            answer: "commutes=true".to_string(),
            categorical_core: true,
        },
        RuliadSampleSpec::Automaton {
            input, accepted, ..
        } => RuliadCategoricalPresentation {
            abstraction: "finite_category_reasoning".to_string(),
            source_family: RuliadFamilyKind::Automaton.label().to_string(),
            task_kind: RuliadTaskKind::EvaluateAutomaton.label().to_string(),
            presentation: "free_monoid_action_category".to_string(),
            objects: vec!["finite_states".to_string(), "input_prefixes".to_string()],
            morphisms: vec![
                "symbol_0_transition".to_string(),
                "symbol_1_transition".to_string(),
                format!("word_action_len_{}", input.len()),
            ],
            functors: Vec::new(),
            laws: vec!["word_actions_compose_by_concatenation".to_string()],
            query: "evaluate the composed input-word morphism and acceptance predicate".to_string(),
            answer: format!("accepted={accepted}"),
            categorical_core: true,
        },
        RuliadSampleSpec::Rewrite {
            steps, normal_form, ..
        } => RuliadCategoricalPresentation {
            abstraction: "finite_category_reasoning".to_string(),
            source_family: RuliadFamilyKind::Rewrite.label().to_string(),
            task_kind: RuliadTaskKind::RewriteNormalForm.label().to_string(),
            presentation: "rewrite_path_category".to_string(),
            objects: vec!["terms".to_string()],
            morphisms: vec![format!("rewrite_path_len_at_most_{steps}")],
            functors: Vec::new(),
            laws: vec!["rewrite_paths_compose".to_string()],
            query: "compose rewrite morphisms until no reducing rule applies".to_string(),
            answer: format!("normal_form={normal_form}"),
            categorical_core: true,
        },
        RuliadSampleSpec::Algebra { law, holds, .. } => RuliadCategoricalPresentation {
            abstraction: "finite_category_reasoning".to_string(),
            source_family: RuliadFamilyKind::Algebra.label().to_string(),
            task_kind: RuliadTaskKind::CheckAlgebraLaw.label().to_string(),
            presentation: "one_object_category_law_probe".to_string(),
            objects: vec!["single_object".to_string()],
            morphisms: vec!["carrier_elements_as_candidate_endomorphisms".to_string()],
            functors: Vec::new(),
            laws: vec![law.label().to_string()],
            query:
                "check whether the finite operation table satisfies the requested categorical law"
                    .to_string(),
            answer: format!("holds={holds}"),
            categorical_core: true,
        },
        RuliadSampleSpec::Category {
            object_count,
            morphisms,
            path,
            composed,
            holds,
            functor,
            naturality,
            task,
            ..
        } => {
            let presentation = match task {
                RuliadTaskKind::ComposeCategoryPath => "finite_category_path",
                RuliadTaskKind::VerifyCategoryLaw => "finite_category_law",
                RuliadTaskKind::VerifyFunctorPreservation => "finite_functor_preservation",
                RuliadTaskKind::VerifyNaturalitySquare => "finite_naturality_square",
                _ => "finite_category",
            };
            let query = match task {
                RuliadTaskKind::ComposeCategoryPath => {
                    "compose a path of arrows in a finite category"
                }
                RuliadTaskKind::VerifyCategoryLaw => {
                    "verify a finite category identity or associativity equation"
                }
                RuliadTaskKind::VerifyFunctorPreservation => {
                    "verify that a finite functor preserves an arrow composition"
                }
                RuliadTaskKind::VerifyNaturalitySquare => {
                    "verify that the selected naturality square commutes"
                }
                _ => "verify a finite categorical reasoning trace",
            };
            let mut laws = vec!["identity".to_string(), "associativity".to_string()];
            if functor.is_some() {
                laws.push("functor_preserves_identity_and_composition".to_string());
            }
            if naturality.is_some() {
                laws.push("naturality_square_commutes".to_string());
            }
            RuliadCategoricalPresentation {
                abstraction: "finite_category_reasoning".to_string(),
                source_family: RuliadFamilyKind::Category.label().to_string(),
                task_kind: task.label().to_string(),
                presentation: presentation.to_string(),
                objects: (0..*object_count)
                    .map(|object| format!("o{object}"))
                    .collect(),
                morphisms: morphisms
                    .iter()
                    .map(|morphism| morphism.name.clone())
                    .collect(),
                functors: functor
                    .as_ref()
                    .map(|functor| vec![functor.name.clone()])
                    .unwrap_or_default(),
                laws,
                query: query.to_string(),
                answer: format!("holds={holds};composed={composed};path={path:?}"),
                categorical_core: true,
            }
        }
        RuliadSampleSpec::ProofTree {
            modulus,
            lemmas,
            proof_steps,
            holds,
            lhs,
            rhs,
            ..
        } => RuliadCategoricalPresentation {
            abstraction: "finite_category_reasoning".to_string(),
            source_family: RuliadFamilyKind::ProofTree.label().to_string(),
            task_kind: RuliadTaskKind::ProveTheorem.label().to_string(),
            presentation: "verified_theorem_dependency_category".to_string(),
            objects: (0..lemmas.len())
                .map(|index| format!("lemma_{index}"))
                .collect(),
            morphisms: proof_steps
                .iter()
                .enumerate()
                .map(|(index, _)| format!("deduction_step_{index}"))
                .collect(),
            functors: vec!["semantic_verifier".to_string()],
            laws: vec![
                "proof_dependencies_compose".to_string(),
                format!("orthogonal_square_sum_mod_{modulus}"),
            ],
            query: "prove the unnamed finite square-sum theorem from its dependency DAG"
                .to_string(),
            answer: format!("holds={holds};lhs={lhs};rhs={rhs}"),
            categorical_core: true,
        },
        RuliadSampleSpec::LeanTask {
            task_id,
            payload_hash,
            ..
        } => RuliadCategoricalPresentation {
            abstraction: "finite_category_reasoning".to_string(),
            source_family: RuliadFamilyKind::LeanTask.label().to_string(),
            task_kind: RuliadTaskKind::CompleteProof.label().to_string(),
            presentation: "proof_category".to_string(),
            objects: vec!["propositions".to_string()],
            morphisms: vec!["proof_terms".to_string(), task_id.clone()],
            functors: vec!["lean_kernel_check".to_string()],
            laws: vec!["proof_composition".to_string()],
            query: "validate a proof payload anchored by the Lean seed project".to_string(),
            answer: format!("payload_hash={payload_hash}"),
            categorical_core: true,
        },
        RuliadSampleSpec::HashNoise { payload_hash, .. } => RuliadCategoricalPresentation {
            abstraction: "source_selection_canary".to_string(),
            source_family: RuliadFamilyKind::HashNoise.label().to_string(),
            task_kind: RuliadTaskKind::HashCanary.label().to_string(),
            presentation: "entropy_control_payload".to_string(),
            objects: Vec::new(),
            morphisms: Vec::new(),
            functors: Vec::new(),
            laws: vec!["sha256_payload_integrity".to_string()],
            query: "verify high-entropy canary payload integrity".to_string(),
            answer: format!("payload_hash={payload_hash}"),
            categorical_core: false,
        },
    }
}

fn sample_rng(
    seed: u64,
    split: SampleSplit,
    epoch_index: usize,
    sample_index: usize,
    bucket_tag: u64,
) -> SplitMix64 {
    let effective_epoch = match split {
        SampleSplit::Train => epoch_index,
        SampleSplit::Validation => 0,
    };
    let split_tag = match split {
        SampleSplit::Train => TRAIN_SPLIT_TAG,
        SampleSplit::Validation => VAL_SPLIT_TAG,
    };
    let mixed = if bucket_tag == 0 {
        mix_seed(
            seed,
            [split_tag, effective_epoch as u64, sample_index as u64],
        )
    } else {
        mix_seed(
            seed,
            [
                split_tag,
                effective_epoch as u64,
                sample_index as u64,
                bucket_tag,
            ],
        )
    };
    SplitMix64::new(mixed)
}

pub fn verify_spec(spec: &RuliadSampleSpec) -> Result<RuliadOracleReport> {
    let (ok, family, task_kind) = match spec {
        RuliadSampleSpec::Eca {
            rule,
            width,
            steps,
            initial,
            trace,
            task,
        } => {
            let parsed_initial = eca::parse_state(initial);
            let parsed_trace = parse_trace(trace);
            let expected = eca::trace(*rule, &parsed_initial, *steps);
            (
                *width == parsed_initial.len()
                    && parsed_trace.len() == steps.saturating_add(1)
                    && eca::states_equal(&parsed_trace, &expected),
                RuliadFamilyKind::Eca,
                *task,
            )
        }
        RuliadSampleSpec::Simulation {
            source_rule,
            target_rule,
            width,
            steps,
            source_initial,
            target_initial,
            source_trace,
            target_trace,
            mapped_source_trace,
            task,
        } => {
            let source_initial = eca::parse_state(source_initial);
            let target_initial = eca::parse_state(target_initial);
            let source_trace = parse_trace(source_trace);
            let target_trace = parse_trace(target_trace);
            let mapped_source_trace = parse_trace(mapped_source_trace);
            let expected_source = eca::trace(*source_rule, &source_initial, *steps);
            let expected_target = eca::trace(*target_rule, &target_initial, *steps);
            let expected_mapped = expected_source
                .iter()
                .map(|state| eca::complement_state(state))
                .collect::<Vec<_>>();
            (
                *width == source_initial.len()
                    && target_initial == eca::complement_state(&source_initial)
                    && *target_rule == eca::complement_rule(*source_rule)
                    && eca::states_equal(&source_trace, &expected_source)
                    && eca::states_equal(&target_trace, &expected_target)
                    && eca::states_equal(&mapped_source_trace, &expected_mapped)
                    && eca::states_equal(&mapped_source_trace, &target_trace),
                RuliadFamilyKind::Simulation,
                *task,
            )
        }
        RuliadSampleSpec::Automaton {
            state_count,
            transitions,
            start_state,
            accept_states,
            input,
            trace,
            accepted,
            task,
        } => {
            let recomputed = automaton_trace(*state_count, transitions, *start_state, input);
            let ok = valid_transition_table(*state_count, transitions, 2)
                && *start_state < *state_count
                && accept_states.iter().all(|state| *state < *state_count)
                && recomputed
                    .as_ref()
                    .is_some_and(|computed| computed == trace)
                && trace
                    .last()
                    .is_some_and(|state| accept_states.contains(state) == *accepted);
            (ok, RuliadFamilyKind::Automaton, *task)
        }
        RuliadSampleSpec::Rewrite {
            alphabet,
            rules,
            initial,
            steps,
            trace,
            normal_form,
            task,
        } => {
            let expected = rewrite_trace(initial, rules, *steps);
            let ok = valid_alphabet(alphabet)
                && alphabet_contains(alphabet, initial)
                && trace.iter().all(|state| alphabet_contains(alphabet, state))
                && alphabet_contains(alphabet, normal_form)
                && !rules.is_empty()
                && rules.iter().all(|rule| {
                    !rule.from.is_empty()
                        && rule.from.len() > rule.to.len()
                        && !rule.to.is_empty()
                        && alphabet_contains(alphabet, &rule.from)
                        && alphabet_contains(alphabet, &rule.to)
                })
                && expected == *trace
                && trace.last().is_some_and(|last| last == normal_form);
            (ok, RuliadFamilyKind::Rewrite, *task)
        }
        RuliadSampleSpec::Algebra {
            carrier_size,
            operation_table,
            law,
            operands,
            lhs,
            rhs,
            holds,
            task,
        } => {
            let recomputed = algebra_law_result(*carrier_size, operation_table, *law, operands);
            let ok = valid_operation_table(*carrier_size, operation_table)
                && recomputed.is_some_and(|(expected_lhs, expected_rhs)| {
                    expected_lhs == *lhs
                        && expected_rhs == *rhs
                        && (expected_lhs == expected_rhs) == *holds
                });
            (ok, RuliadFamilyKind::Algebra, *task)
        }
        RuliadSampleSpec::Category {
            object_count,
            morphisms,
            identities,
            composition,
            path,
            composed,
            lhs,
            rhs,
            holds,
            functor,
            naturality,
            task,
            ..
        } => {
            let recomposed = compose_path(morphisms, composition, path);
            let task_ok = match task {
                RuliadTaskKind::ComposeCategoryPath | RuliadTaskKind::VerifyCategoryLaw => {
                    recomposed.is_some_and(|expected| expected == *composed)
                        && (*lhs == *rhs) == *holds
                }
                RuliadTaskKind::VerifyFunctorPreservation => {
                    functor.as_ref().is_some_and(|functor| {
                        valid_functor(*object_count, morphisms, identities, composition, functor)
                            && (*lhs == *rhs) == *holds
                    })
                }
                RuliadTaskKind::VerifyNaturalitySquare => functor
                    .as_ref()
                    .zip(naturality.as_ref())
                    .is_some_and(|(functor, naturality)| {
                        valid_functor(*object_count, morphisms, identities, composition, functor)
                            && naturality_commutes(morphisms, composition, functor, naturality)
                            && (*lhs == *rhs) == *holds
                    }),
                _ => false,
            };
            let ok = valid_finite_category(*object_count, morphisms, identities, composition)
                && task_ok
                && *holds
                && *lhs < morphisms.len()
                && *rhs < morphisms.len()
                && *composed < morphisms.len();
            (ok, RuliadFamilyKind::Category, *task)
        }
        RuliadSampleSpec::ProofTree {
            modulus,
            u,
            v,
            sum,
            dot,
            norm_u,
            norm_v,
            norm_sum,
            lhs,
            rhs,
            holds,
            lemmas,
            proof_steps,
            task,
        } => {
            let recomputed_sum = [(u[0] + v[0]) % modulus, (u[1] + v[1]) % modulus];
            let recomputed_dot = mod_dot(*u, *v, *modulus);
            let recomputed_norm_u = mod_norm(*u, *modulus);
            let recomputed_norm_v = mod_norm(*v, *modulus);
            let recomputed_norm_sum = mod_norm(*sum, *modulus);
            let recomputed_rhs = (recomputed_norm_u + recomputed_norm_v) % modulus;
            let theorem_holds = recomputed_dot == 0 && recomputed_norm_sum == recomputed_rhs;
            let ok = *modulus >= 2
                && sum == &recomputed_sum
                && *dot == recomputed_dot
                && *norm_u == recomputed_norm_u
                && *norm_v == recomputed_norm_v
                && *norm_sum == recomputed_norm_sum
                && *lhs == recomputed_norm_sum
                && *rhs == recomputed_rhs
                && *holds == theorem_holds
                && *holds
                && lemmas.len() >= 4
                && proof_steps.len() >= 4;
            (ok, RuliadFamilyKind::ProofTree, *task)
        }
        RuliadSampleSpec::LeanTask {
            task_id,
            statement,
            proof,
            payload_hash,
            task,
        } => {
            let proof_task = LeanProofTask {
                id: task_id.clone(),
                statement: statement.clone(),
                proof: proof.clone(),
                payload_hash: Some(payload_hash.clone()),
            };
            (
                proof_task.validate_hash(),
                RuliadFamilyKind::LeanTask,
                *task,
            )
        }
        RuliadSampleSpec::HashNoise {
            bytes_hex,
            payload_hash,
            task,
        } => {
            let decoded = hex::decode(bytes_hex).unwrap_or_default();
            (
                !decoded.is_empty() && sha256_hex(&decoded) == *payload_hash,
                RuliadFamilyKind::HashNoise,
                *task,
            )
        }
    };
    let oracle_hash = stable_json_hash(spec)?;
    Ok(RuliadOracleReport {
        ok,
        family,
        task_kind,
        oracle_hash,
    })
}

pub fn sample_text(spec: &RuliadSampleSpec, oracle_hash: &str) -> String {
    proof_tape_document(spec, oracle_hash).to_text()
}

pub fn ruliad_expected_answer(spec: &RuliadSampleSpec) -> String {
    compact_answer(spec)
}

pub fn ruliad_prompt_prefix(spec: &RuliadSampleSpec, oracle_hash: &str) -> String {
    let text = sample_text(spec, oracle_hash);
    if let Some(answer_offset) = text.find("\n!:") {
        text[..answer_offset + 3].to_string()
    } else {
        text
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RuliadProofTapeDocument {
    source_family: String,
    task_kind: String,
    presentation: String,
    domains: Vec<String>,
    reasoning_modes: Vec<String>,
    verifier_version: u32,
    oracle_hash: String,
    query: String,
    proof_steps: Vec<String>,
    answer: String,
    data: Vec<String>,
}

impl RuliadProofTapeDocument {
    fn to_text(&self) -> String {
        let hash = compact_text(&self.oracle_hash, 16);
        let domains = compact_labels(&self.domains);
        let modes = compact_labels(&self.reasoning_modes);
        let mut out = format!(
            "[R2 h={hash} v={} f={} t={} p={}]\nS:{domains}|{modes}\nG:{}\n?:{}\n",
            self.verifier_version,
            compact_label(&self.source_family),
            compact_label(&self.task_kind),
            compact_label(&self.presentation),
            self.data.join("|"),
            self.query
        );
        for (index, step) in self.proof_steps.iter().enumerate() {
            out.push_str(&format!(">{index}:{}\n", compact_text(step, 96)));
        }
        out.push_str(&format!("!:{}\n[/R2]\n", self.answer));
        out
    }
}

fn proof_tape_document(spec: &RuliadSampleSpec, oracle_hash: &str) -> RuliadProofTapeDocument {
    let view = ruliad_categorical_presentation(spec);
    let family = family_of_spec(spec);
    let task_kind = task_kind_of_spec(spec);
    let semantics = ruliad_source_semantics(family, task_kind);
    RuliadProofTapeDocument {
        source_family: view.source_family,
        task_kind: view.task_kind,
        presentation: view.presentation,
        domains: semantics
            .math_domains
            .iter()
            .map(|domain| domain.label().to_string())
            .collect(),
        reasoning_modes: semantics
            .reasoning_modes
            .iter()
            .map(|mode| mode.label().to_string())
            .collect(),
        verifier_version: RULIAD_VERIFIER_VERSION,
        oracle_hash: oracle_hash.to_string(),
        query: compact_query(spec),
        proof_steps: compact_proof_steps(spec),
        answer: compact_answer(spec),
        data: compact_data(spec),
    }
}

fn compact_query(spec: &RuliadSampleSpec) -> String {
    match spec {
        RuliadSampleSpec::Eca { steps, .. } => format!("compose local rule for {steps} steps"),
        RuliadSampleSpec::Simulation {
            source_rule,
            target_rule,
            steps,
            ..
        } => format!("verify complement functor rule {source_rule}->{target_rule} for {steps}"),
        RuliadSampleSpec::Automaton { input, .. } => format!("evaluate word action {}", input),
        RuliadSampleSpec::Rewrite { initial, steps, .. } => {
            format!("normalize {initial} in <= {steps} rewrites")
        }
        RuliadSampleSpec::Algebra { law, operands, .. } => {
            format!("check {} on {}", law.label(), compact_usize_list(operands))
        }
        RuliadSampleSpec::Category { task, .. } => match task {
            RuliadTaskKind::ComposeCategoryPath => "compose finite-category path".to_string(),
            RuliadTaskKind::VerifyCategoryLaw => "verify finite-category associativity".to_string(),
            RuliadTaskKind::VerifyFunctorPreservation => {
                "verify functor preserves composition".to_string()
            }
            RuliadTaskKind::VerifyNaturalitySquare => "verify finite naturality square".to_string(),
            _ => "verify finite category trace".to_string(),
        },
        RuliadSampleSpec::ProofTree { modulus, .. } => {
            format!("prove unnamed square-sum theorem over Z/{modulus}Z")
        }
        RuliadSampleSpec::LeanTask { task_id, .. } => format!("validate proof payload {task_id}"),
        RuliadSampleSpec::HashNoise { .. } => "verify entropy canary hash".to_string(),
    }
}

fn compact_proof_steps(spec: &RuliadSampleSpec) -> Vec<String> {
    match spec {
        RuliadSampleSpec::Eca {
            initial,
            trace,
            steps,
            ..
        } => {
            let mut steps_out = vec![format!("x0={initial}")];
            if trace.len() > 2 {
                let mid = trace.len() / 2;
                steps_out.push(format!("x{mid}={}", trace[mid]));
            }
            steps_out.push(format!(
                "x{steps}={}",
                trace.last().cloned().unwrap_or_default()
            ));
            steps_out
        }
        RuliadSampleSpec::Simulation {
            mapped_source_trace,
            target_trace,
            ..
        } => {
            let mut steps = vec!["F0=complement(x0)".to_string()];
            if mapped_source_trace.len() > 2 && target_trace.len() > 2 {
                let mid = mapped_source_trace.len() / 2;
                steps.push(format!(
                    "F{mid}_ok={}",
                    mapped_source_trace.get(mid) == target_trace.get(mid)
                ));
            }
            steps.push(format!(
                "last_ok={}",
                mapped_source_trace.last() == target_trace.last()
            ));
            steps
        }
        RuliadSampleSpec::Automaton {
            start_state, trace, ..
        } => vec![
            format!(
                "q{}=>q{}",
                start_state,
                trace.last().copied().unwrap_or(*start_state)
            ),
            format!("trace={}", compact_state_trace(trace)),
        ],
        RuliadSampleSpec::Rewrite {
            initial,
            trace,
            normal_form,
            ..
        } => vec![
            format!("{}=>{} in {} steps", initial, normal_form, trace.len() - 1),
            format!("trace={}", compact_string_trace(trace)),
        ],
        RuliadSampleSpec::Algebra { law, lhs, rhs, .. } => {
            vec![format!("{} lhs={lhs};rhs={rhs}", law.label())]
        }
        RuliadSampleSpec::Category { proof_steps, .. } => proof_steps.clone(),
        RuliadSampleSpec::ProofTree {
            lemmas,
            proof_steps,
            ..
        } => {
            let mut steps = vec![format!(
                "dag=L0..L{};deps=chain,expand,subst",
                lemmas.len().saturating_sub(1)
            )];
            if let Some(step) = proof_steps.get(1) {
                steps.push(step.clone());
            }
            if let Some(step) = proof_steps.get(2) {
                steps.push(step.clone());
            }
            if let Some(step) = proof_steps.last() {
                steps.push(step.clone());
            }
            steps
        }
        RuliadSampleSpec::LeanTask { .. } => vec!["payload_hash_matches=true".to_string()],
        RuliadSampleSpec::HashNoise { .. } => vec!["sha256_matches=true".to_string()],
    }
}

fn compact_answer(spec: &RuliadSampleSpec) -> String {
    match spec {
        RuliadSampleSpec::Eca { trace, .. } => {
            format!("target={}", trace.last().cloned().unwrap_or_default())
        }
        RuliadSampleSpec::Simulation { .. } => "commutes=true".to_string(),
        RuliadSampleSpec::Automaton { accepted, .. } => format!("accepted={accepted}"),
        RuliadSampleSpec::Rewrite { normal_form, .. } => format!("normal_form={normal_form}"),
        RuliadSampleSpec::Algebra { holds, .. } => format!("holds={holds}"),
        RuliadSampleSpec::Category {
            lhs, rhs, holds, ..
        } => format!("holds={holds};lhs={lhs};rhs={rhs}"),
        RuliadSampleSpec::ProofTree {
            holds, lhs, rhs, ..
        } => {
            format!("holds={holds};lhs={lhs};rhs={rhs}")
        }
        RuliadSampleSpec::LeanTask { payload_hash, .. } => format!("payload_hash={payload_hash}"),
        RuliadSampleSpec::HashNoise { payload_hash, .. } => format!("payload_hash={payload_hash}"),
    }
}

fn compact_data(spec: &RuliadSampleSpec) -> Vec<String> {
    match spec {
        RuliadSampleSpec::Eca {
            rule,
            width,
            steps,
            initial,
            trace: _,
            ..
        } => vec![
            format!("rule={rule};w={width};steps={steps}"),
            format!("x0={initial}"),
        ],
        RuliadSampleSpec::Simulation {
            source_rule,
            target_rule,
            width,
            steps,
            source_initial,
            ..
        } => vec![
            format!("rules={source_rule}->{target_rule};w={width};steps={steps}"),
            format!("x0={source_initial};F=complement"),
        ],
        RuliadSampleSpec::Automaton {
            state_count,
            transitions,
            start_state,
            accept_states,
            input,
            ..
        } => vec![
            format!("states={state_count};start={start_state};accept={accept_states:?}"),
            format!("input={input};delta={transitions:?}"),
        ],
        RuliadSampleSpec::Rewrite {
            alphabet, rules, ..
        } => vec![
            format!("alphabet={alphabet}"),
            format!(
                "rules={}",
                rules
                    .iter()
                    .map(|rule| format!("{}>{}", rule.from, rule.to))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        ],
        RuliadSampleSpec::Algebra {
            carrier_size,
            operation_table,
            operands,
            ..
        } => vec![
            format!(
                "carrier={carrier_size};operands={}",
                compact_usize_list(operands)
            ),
            format!("op={}", compact_operation_descriptor(operation_table)),
        ],
        RuliadSampleSpec::Category {
            object_count,
            morphisms,
            identities,
            path,
            composed,
            functor,
            naturality,
            ..
        } => {
            let mut data = vec![
                format!(
                    "objects={object_count};ids={}",
                    compact_usize_list(identities)
                ),
                format!("path={};composed={composed}", compact_usize_list(path)),
                format!("arrows={}", compact_morphism_summary(morphisms)),
            ];
            if let Some(functor) = functor {
                data.push(format!(
                    "{}:obj={}",
                    functor.name,
                    compact_usize_list(&functor.object_map)
                ));
            }
            if let Some(naturality) = naturality {
                data.push(format!(
                    "nat:f={};l={};r={}",
                    naturality.source_morphism,
                    compact_usize_list(&naturality.left_path),
                    compact_usize_list(&naturality.right_path)
                ));
            }
            data
        }
        RuliadSampleSpec::ProofTree {
            modulus,
            u,
            v,
            sum,
            dot,
            norm_u,
            norm_v,
            norm_sum,
            ..
        } => vec![
            format!("m={modulus};u={},{};v={},{}", u[0], u[1], v[0], v[1]),
            format!("sum={},{};dot={dot}", sum[0], sum[1]),
            format!("norms={norm_u},{norm_v},{norm_sum}"),
        ],
        RuliadSampleSpec::LeanTask {
            task_id,
            statement,
            proof,
            ..
        } => vec![
            format!("task_id={task_id}"),
            format!("stmt={}", compact_text(statement, 40)),
            format!("proof={}", compact_text(proof, 40)),
        ],
        RuliadSampleSpec::HashNoise { bytes_hex, .. } => {
            vec![format!("bytes={}", compact_text(bytes_hex, 64))]
        }
    }
}

fn compact_text(value: &str, max_len: usize) -> String {
    if value.chars().count() <= max_len {
        value.to_string()
    } else {
        format!(
            "{}..",
            value
                .chars()
                .take(max_len.saturating_sub(2))
                .collect::<String>()
        )
    }
}

fn compact_labels(values: &[String]) -> String {
    values
        .iter()
        .map(|value| compact_label(value))
        .collect::<Vec<_>>()
        .join(",")
}

fn compact_label(value: &str) -> &str {
    match value {
        "discrete_dynamics" => "dd",
        "computation_theory" => "ct",
        "symbolic_rewriting" => "sr",
        "universal_algebra" => "ua",
        "category_theory" => "cat",
        "formal_proof" => "fp",
        "information_theory" => "it",
        "local_rule_evaluation" => "lre",
        "iterated_dynamics" => "iter",
        "state_machine_execution" => "sm",
        "simulation_equivalence" => "sim",
        "structure_preservation" => "struct",
        "normalization" => "norm",
        "equational_reasoning" => "eq",
        "counterexample_evaluation" => "cex",
        "compositional_reasoning" => "comp",
        "associativity" => "assoc",
        "formal_deduction" => "proof",
        "entropy_canary" => "entropy",
        "proof_tree" => "pt",
        "prove_theorem" => "thm",
        "verified_theorem_dependency_category" => "thm_cat",
        other => other,
    }
}

fn compact_usize_list(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn compact_state_trace(trace: &[usize]) -> String {
    if trace.len() <= 8 {
        return compact_usize_list(trace);
    }
    let mid = trace.len() / 2;
    format!(
        "{},{},{}..{},{}",
        trace[0],
        trace[1],
        trace[mid],
        trace[trace.len() - 2],
        trace[trace.len() - 1]
    )
}

fn compact_string_trace(trace: &[String]) -> String {
    if trace.len() <= 5 {
        return trace.join(">");
    }
    let mid = trace.len() / 2;
    format!(
        "{}>{}>{}..>{}",
        trace[0],
        trace[1],
        trace[mid],
        trace.last().cloned().unwrap_or_default()
    )
}

fn compact_table(table: &[Vec<usize>]) -> String {
    table
        .iter()
        .map(|row| compact_usize_list(row))
        .collect::<Vec<_>>()
        .join("/")
}

fn compact_operation_descriptor(table: &[Vec<usize>]) -> String {
    let carrier_size = table.len();
    if table == add_mod_table(carrier_size) {
        return format!("add_mod_{carrier_size}");
    }
    if table == affine_mod_table(carrier_size, 1, 2, 1) {
        return format!("affine_mod_{carrier_size}(x+2y+1)");
    }
    if carrier_size <= 6 {
        return compact_table(table);
    }
    let hash = stable_json_hash(&table).unwrap_or_else(|_| "unknown".to_string());
    let row0 = table
        .first()
        .map(|row| compact_usize_list(row))
        .unwrap_or_default();
    format!(
        "table_hash={};row0={}",
        compact_text(&hash, 16),
        compact_text(&row0, 64)
    )
}

fn compact_morphism_summary(morphisms: &[RuliadCategoryMorphism]) -> String {
    let first = morphisms
        .first()
        .map(|morphism| morphism.name.as_str())
        .unwrap_or("-");
    let last = morphisms
        .last()
        .map(|morphism| morphism.name.as_str())
        .unwrap_or("-");
    format!(
        "count={};thin_total_order;first={first};last={last}",
        morphisms.len()
    )
}

fn family_of_spec(spec: &RuliadSampleSpec) -> RuliadFamilyKind {
    match spec {
        RuliadSampleSpec::Eca { .. } => RuliadFamilyKind::Eca,
        RuliadSampleSpec::Simulation { .. } => RuliadFamilyKind::Simulation,
        RuliadSampleSpec::Automaton { .. } => RuliadFamilyKind::Automaton,
        RuliadSampleSpec::Rewrite { .. } => RuliadFamilyKind::Rewrite,
        RuliadSampleSpec::Algebra { .. } => RuliadFamilyKind::Algebra,
        RuliadSampleSpec::Category { .. } => RuliadFamilyKind::Category,
        RuliadSampleSpec::ProofTree { .. } => RuliadFamilyKind::ProofTree,
        RuliadSampleSpec::LeanTask { .. } => RuliadFamilyKind::LeanTask,
        RuliadSampleSpec::HashNoise { .. } => RuliadFamilyKind::HashNoise,
    }
}

fn task_kind_of_spec(spec: &RuliadSampleSpec) -> RuliadTaskKind {
    match spec {
        RuliadSampleSpec::Eca { task, .. }
        | RuliadSampleSpec::Simulation { task, .. }
        | RuliadSampleSpec::Automaton { task, .. }
        | RuliadSampleSpec::Rewrite { task, .. }
        | RuliadSampleSpec::Algebra { task, .. }
        | RuliadSampleSpec::Category { task, .. }
        | RuliadSampleSpec::ProofTree { task, .. }
        | RuliadSampleSpec::LeanTask { task, .. }
        | RuliadSampleSpec::HashNoise { task, .. } => *task,
    }
}

fn choose_family<'a>(
    families: &'a [RuliadFamilyConfig],
    rng: &mut SplitMix64,
) -> Result<&'a RuliadFamilyConfig> {
    if families.is_empty() {
        return Err(anyhow!("ruliad families must not be empty"));
    }
    let total = families.iter().map(|family| family.weight).sum::<usize>();
    let mut ticket = rng.next_usize(total.max(1));
    for family in families {
        if ticket < family.weight {
            return Ok(family);
        }
        ticket = ticket.saturating_sub(family.weight);
    }
    Ok(&families[families.len() - 1])
}

pub(crate) fn scale_family_for_difficulty(
    family: &RuliadFamilyConfig,
    difficulty_level: usize,
) -> RuliadFamilyConfig {
    if difficulty_level == 0 {
        return family.clone();
    }
    let mut scaled = family.clone();
    let level = difficulty_level.min(32);
    if let Some(width) = scaled.width.as_mut() {
        let stride = (width.max.saturating_sub(width.min).max(1) / 2).max(1);
        let bump = stride.saturating_mul(level);
        width.min = width.min.saturating_add(bump);
        width.max = width.max.saturating_add(bump.saturating_mul(2));
    }
    if let Some(steps) = scaled.steps.as_mut() {
        let stride = (steps.max.saturating_sub(steps.min).max(1) / 2).max(1);
        let bump = stride.saturating_mul(level);
        steps.min = steps.min.saturating_add(bump);
        steps.max = steps.max.saturating_add(bump.saturating_mul(2));
    }
    cap_scaled_family_for_payload(&mut scaled);
    scaled
}

fn cap_scaled_family_for_payload(family: &mut RuliadFamilyConfig) {
    let (max_width, max_steps) = match family.kind {
        RuliadFamilyKind::Eca => (Some(128), Some(64)),
        RuliadFamilyKind::Simulation => (Some(128), Some(64)),
        RuliadFamilyKind::Automaton => (Some(48), Some(128)),
        RuliadFamilyKind::Rewrite => (Some(128), Some(96)),
        RuliadFamilyKind::Algebra => (Some(64), None),
        RuliadFamilyKind::Category => (Some(48), Some(64)),
        RuliadFamilyKind::ProofTree => (Some(512), Some(512)),
        RuliadFamilyKind::LeanTask | RuliadFamilyKind::HashNoise => (None, None),
    };
    if let (Some(width), Some(max_width)) = (family.width.as_mut(), max_width) {
        cap_range(width, max_width);
    }
    if let (Some(steps), Some(max_steps)) = (family.steps.as_mut(), max_steps) {
        cap_range(steps, max_steps);
    }
}

fn cap_range(range: &mut crate::config::UsizeRangeConfig, max_value: usize) {
    range.min = range.min.min(max_value);
    range.max = range.max.min(max_value).max(range.min);
}

fn generate_eca_spec(
    family: &RuliadFamilyConfig,
    rng: &mut SplitMix64,
) -> Result<RuliadSampleSpec> {
    let mut fallback = None;
    for _ in 0..64 {
        let width = range_or(family.width, 16, 32, rng);
        let steps = range_or(family.steps, 4, 10, rng);
        let rule = rng.next_u8();
        let initial = eca::random_state(width, rng);
        let trace = eca::trace(rule, &initial, steps)
            .iter()
            .map(|state| eca::format_state(state))
            .collect::<Vec<_>>();
        let spec = RuliadSampleSpec::Eca {
            rule,
            width,
            steps,
            initial: eca::format_state(&initial),
            trace,
            task: if steps <= 1 {
                RuliadTaskKind::NextState
            } else {
                RuliadTaskKind::MultiStepState
            },
        };
        if !is_degenerate_spec(&spec) {
            return Ok(spec);
        }
        fallback = Some(spec);
    }
    fallback.ok_or_else(|| anyhow!("failed to generate ECA ruliad sample"))
}

fn generate_simulation_spec(
    family: &RuliadFamilyConfig,
    rng: &mut SplitMix64,
) -> Result<RuliadSampleSpec> {
    let width = range_or(family.width, 16, 32, rng);
    let steps = range_or(family.steps, 4, 8, rng);
    let source_rule = rng.next_u8();
    let target_rule = eca::complement_rule(source_rule);
    let source_initial = eca::random_state(width, rng);
    let target_initial = eca::complement_state(&source_initial);
    let source_trace = eca::trace(source_rule, &source_initial, steps);
    let target_trace = eca::trace(target_rule, &target_initial, steps);
    let mapped_source_trace = source_trace
        .iter()
        .map(|state| eca::complement_state(state))
        .collect::<Vec<_>>();
    Ok(RuliadSampleSpec::Simulation {
        source_rule,
        target_rule,
        width,
        steps,
        source_initial: eca::format_state(&source_initial),
        target_initial: eca::format_state(&target_initial),
        source_trace: source_trace
            .iter()
            .map(|state| eca::format_state(state))
            .collect(),
        target_trace: target_trace
            .iter()
            .map(|state| eca::format_state(state))
            .collect(),
        mapped_source_trace: mapped_source_trace
            .iter()
            .map(|state| eca::format_state(state))
            .collect(),
        task: RuliadTaskKind::VerifySimulation,
    })
}

fn generate_automaton_spec(
    family: &RuliadFamilyConfig,
    rng: &mut SplitMix64,
) -> Result<RuliadSampleSpec> {
    let mut fallback = None;
    for _ in 0..64 {
        let state_count = range_or(family.width, 3, 8, rng);
        let input_len = range_or(family.steps, 6, 20, rng);
        let transitions = (0..state_count)
            .map(|_| {
                (0..2)
                    .map(|_| rng.next_usize(state_count))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let start_state = rng.next_usize(state_count);
        let mut accept_states = (0..state_count)
            .filter(|_| rng.next_bool())
            .collect::<Vec<_>>();
        if accept_states.is_empty() || accept_states.len() == state_count {
            accept_states = (0..state_count).filter(|state| state % 2 == 0).collect();
        }
        accept_states.sort_unstable();
        accept_states.dedup();
        let input = (0..input_len)
            .map(|_| if rng.next_bool() { '1' } else { '0' })
            .collect::<String>();
        let trace = automaton_trace(state_count, &transitions, start_state, &input)
            .ok_or_else(|| anyhow!("generated invalid automaton trace"))?;
        let accepted = trace
            .last()
            .is_some_and(|state| accept_states.contains(state));
        let spec = RuliadSampleSpec::Automaton {
            state_count,
            transitions,
            start_state,
            accept_states,
            input,
            trace,
            accepted,
            task: RuliadTaskKind::EvaluateAutomaton,
        };
        if !is_degenerate_spec(&spec) {
            return Ok(spec);
        }
        fallback = Some(spec);
    }
    fallback.ok_or_else(|| anyhow!("failed to generate automaton ruliad sample"))
}

fn generate_rewrite_spec(
    family: &RuliadFamilyConfig,
    rng: &mut SplitMix64,
) -> Result<RuliadSampleSpec> {
    let alphabet = "ABC".to_string();
    let initial_len = range_or(family.width, 8, 20, rng);
    let steps = range_or(family.steps, 4, 12, rng);
    let mut candidates = vec![
        RuliadRewriteRule {
            from: "AA".to_string(),
            to: "A".to_string(),
        },
        RuliadRewriteRule {
            from: "BB".to_string(),
            to: "B".to_string(),
        },
        RuliadRewriteRule {
            from: "CC".to_string(),
            to: "C".to_string(),
        },
        RuliadRewriteRule {
            from: "AB".to_string(),
            to: "C".to_string(),
        },
        RuliadRewriteRule {
            from: "BA".to_string(),
            to: "A".to_string(),
        },
        RuliadRewriteRule {
            from: "BC".to_string(),
            to: "A".to_string(),
        },
        RuliadRewriteRule {
            from: "CB".to_string(),
            to: "B".to_string(),
        },
        RuliadRewriteRule {
            from: "AC".to_string(),
            to: "B".to_string(),
        },
        RuliadRewriteRule {
            from: "CA".to_string(),
            to: "C".to_string(),
        },
    ];
    shuffle_rules(&mut candidates, rng);
    let rule_count = rng.range_usize(3, 5).min(candidates.len());
    let rules = candidates.into_iter().take(rule_count).collect::<Vec<_>>();
    let symbols = alphabet.chars().collect::<Vec<_>>();
    let mut fallback = None;
    for _ in 0..64 {
        let initial = (0..initial_len)
            .map(|_| symbols[rng.next_usize(symbols.len())])
            .collect::<String>();
        let trace = rewrite_trace(&initial, &rules, steps);
        let normal_form = trace.last().cloned().unwrap_or_else(|| initial.clone());
        let spec = RuliadSampleSpec::Rewrite {
            alphabet: alphabet.clone(),
            rules: rules.clone(),
            initial,
            steps,
            trace,
            normal_form,
            task: RuliadTaskKind::RewriteNormalForm,
        };
        if !is_degenerate_spec(&spec) {
            return Ok(spec);
        }
        fallback = Some(spec);
    }
    fallback.ok_or_else(|| anyhow!("failed to generate rewrite ruliad sample"))
}

fn generate_algebra_spec(
    family: &RuliadFamilyConfig,
    rng: &mut SplitMix64,
) -> Result<RuliadSampleSpec> {
    let carrier_size = range_or(family.width, 2, 6, rng);
    let operation_table = if rng.next_bool() || carrier_size <= 2 {
        add_mod_table(carrier_size)
    } else {
        affine_mod_table(carrier_size, 1, 2, 1)
    };
    let law = if rng.next_bool() {
        RuliadAlgebraLaw::Associativity
    } else {
        RuliadAlgebraLaw::Commutativity
    };
    let operand_count = match law {
        RuliadAlgebraLaw::Associativity => 3,
        RuliadAlgebraLaw::Commutativity => 2,
    };
    let operands = (0..operand_count)
        .map(|_| rng.next_usize(carrier_size))
        .collect::<Vec<_>>();
    let (lhs, rhs) = algebra_law_result(carrier_size, &operation_table, law, &operands)
        .ok_or_else(|| anyhow!("generated invalid algebra law probe"))?;
    Ok(RuliadSampleSpec::Algebra {
        carrier_size,
        operation_table,
        law,
        operands,
        lhs,
        rhs,
        holds: lhs == rhs,
        task: RuliadTaskKind::CheckAlgebraLaw,
    })
}

fn generate_category_spec(
    family: &RuliadFamilyConfig,
    rng: &mut SplitMix64,
) -> Result<RuliadSampleSpec> {
    let task = match rng.next_usize(4) {
        0 => RuliadTaskKind::ComposeCategoryPath,
        1 => RuliadTaskKind::VerifyCategoryLaw,
        2 => RuliadTaskKind::VerifyFunctorPreservation,
        _ => RuliadTaskKind::VerifyNaturalitySquare,
    };
    generate_category_spec_for_task(family, task, rng)
}

fn generate_category_spec_for_task(
    family: &RuliadFamilyConfig,
    task: RuliadTaskKind,
    rng: &mut SplitMix64,
) -> Result<RuliadSampleSpec> {
    let fields = generate_category_fields(family, task, rng)?;
    Ok(RuliadSampleSpec::Category {
        object_count: fields.object_count,
        morphisms: fields.morphisms,
        identities: fields.identities,
        composition: fields.composition,
        path: fields.path,
        composed: fields.composed,
        lhs: fields.lhs,
        rhs: fields.rhs,
        holds: fields.holds,
        proof_steps: fields.proof_steps,
        functor: fields.functor,
        naturality: fields.naturality,
        task: fields.task,
    })
}

fn generate_proof_tree_spec(
    family: &RuliadFamilyConfig,
    rng: &mut SplitMix64,
) -> Result<RuliadSampleSpec> {
    let modulus = next_prime_at_least(range_or(family.width, 5, 13, rng).max(5));
    let depth = range_or(family.steps, 4, 9, rng).max(4);
    let u = [
        rng.range_usize(1, modulus.saturating_sub(1)),
        rng.range_usize(1, modulus.saturating_sub(1)),
    ];
    let scale = rng.range_usize(1, modulus.saturating_sub(1));
    let v = [
        (u[1] * scale) % modulus,
        (modulus - ((u[0] * scale) % modulus)) % modulus,
    ];
    let sum = [(u[0] + v[0]) % modulus, (u[1] + v[1]) % modulus];
    let dot = mod_dot(u, v, modulus);
    let norm_u = mod_norm(u, modulus);
    let norm_v = mod_norm(v, modulus);
    let norm_sum = mod_norm(sum, modulus);
    let rhs = (norm_u + norm_v) % modulus;
    let mut lemmas = vec![
        format!("L0:def dot(x,y)=x0*y0+x1*y1 mod {modulus}"),
        format!("L1:def n(x)=x0*x0+x1*x1 mod {modulus}"),
        "L2:expand n(x+y)=n(x)+n(y)+2*dot(x,y)".to_string(),
        "L3:orthogonal vectors cancel cross term".to_string(),
    ];
    for lemma_index in 4..depth {
        lemmas.push(format!(
            "L{lemma_index}:substitute previous equality into goal term"
        ));
    }
    let proof_steps = vec![
        format!("u=({},{});v=({},{});m={modulus}", u[0], u[1], v[0], v[1]),
        format!("dot={}*{}+{}*{}={dot}", u[0], v[0], u[1], v[1]),
        format!("sum=({},{});n(sum)={norm_sum}", sum[0], sum[1]),
        format!("n(u)+n(v)={norm_u}+{norm_v}={rhs}"),
        "close by L2 and dot=0".to_string(),
    ];
    Ok(RuliadSampleSpec::ProofTree {
        modulus,
        u,
        v,
        sum,
        dot,
        norm_u,
        norm_v,
        norm_sum,
        lhs: norm_sum,
        rhs,
        holds: dot == 0 && norm_sum == rhs,
        lemmas,
        proof_steps,
        task: RuliadTaskKind::ProveTheorem,
    })
}

fn generate_lean_spec(
    proof_tasks: &[LeanProofTask],
    rng: &mut SplitMix64,
) -> Result<RuliadSampleSpec> {
    let tasks = if proof_tasks.is_empty() {
        default_proof_tasks()
    } else {
        proof_tasks.to_vec()
    };
    let proof_task = tasks[rng.next_usize(tasks.len())].clone();
    let renaming = rng.next_u64();
    let instantiated = LeanProofTask {
        id: format!("{}__r{renaming:016x}", proof_task.id),
        statement: format!(
            "{} [symbolic_renaming={renaming:016x}]",
            proof_task.statement
        ),
        proof: proof_task.proof,
        payload_hash: None,
    };
    let payload_hash = instantiated.computed_payload_hash();
    Ok(RuliadSampleSpec::LeanTask {
        task_id: instantiated.id,
        statement: instantiated.statement,
        proof: instantiated.proof,
        payload_hash,
        task: RuliadTaskKind::CompleteProof,
    })
}

fn generate_hash_noise_spec(rng: &mut SplitMix64) -> Result<RuliadSampleSpec> {
    let bytes = (0..32).map(|_| rng.next_u8()).collect::<Vec<_>>();
    Ok(RuliadSampleSpec::HashNoise {
        bytes_hex: hex::encode(&bytes),
        payload_hash: sha256_hex(&bytes),
        task: RuliadTaskKind::HashCanary,
    })
}

fn parse_trace(trace: &[String]) -> Vec<Vec<u8>> {
    trace.iter().map(|state| eca::parse_state(state)).collect()
}

fn valid_transition_table(
    state_count: usize,
    transitions: &[Vec<usize>],
    alphabet_size: usize,
) -> bool {
    state_count > 0
        && transitions.len() == state_count
        && transitions.iter().all(|row| {
            row.len() == alphabet_size && row.iter().all(|next_state| *next_state < state_count)
        })
}

fn automaton_trace(
    state_count: usize,
    transitions: &[Vec<usize>],
    start_state: usize,
    input: &str,
) -> Option<Vec<usize>> {
    if !valid_transition_table(state_count, transitions, 2) || start_state >= state_count {
        return None;
    }
    let mut state = start_state;
    let mut trace = Vec::with_capacity(input.len().saturating_add(1));
    trace.push(state);
    for symbol in input.bytes() {
        let input_index = match symbol {
            b'0' => 0,
            b'1' => 1,
            _ => return None,
        };
        state = transitions[state][input_index];
        trace.push(state);
    }
    Some(trace)
}

fn valid_alphabet(alphabet: &str) -> bool {
    let mut seen = std::collections::BTreeSet::new();
    !alphabet.is_empty()
        && alphabet.is_ascii()
        && alphabet
            .chars()
            .all(|symbol| !symbol.is_whitespace() && seen.insert(symbol))
}

fn alphabet_contains(alphabet: &str, value: &str) -> bool {
    value
        .chars()
        .all(|symbol| alphabet.chars().any(|candidate| candidate == symbol))
}

fn rewrite_trace(initial: &str, rules: &[RuliadRewriteRule], steps: usize) -> Vec<String> {
    let mut trace = Vec::with_capacity(steps.saturating_add(1));
    let mut current = initial.to_string();
    trace.push(current.clone());
    for _ in 0..steps {
        let Some(next) = apply_rewrite_once(&current, rules) else {
            break;
        };
        current = next;
        trace.push(current.clone());
    }
    trace
}

fn apply_rewrite_once(value: &str, rules: &[RuliadRewriteRule]) -> Option<String> {
    let mut best_match = None;
    for (rule_index, rule) in rules.iter().enumerate() {
        if rule.from.is_empty() {
            continue;
        }
        if let Some(position) = value.find(&rule.from)
            && best_match.is_none_or(|(best_position, best_rule_index)| {
                position < best_position
                    || (position == best_position && rule_index < best_rule_index)
            })
        {
            best_match = Some((position, rule_index));
        }
    }
    let (position, rule_index) = best_match?;
    let rule = &rules[rule_index];
    let mut next = String::with_capacity(value.len() - rule.from.len() + rule.to.len());
    next.push_str(&value[..position]);
    next.push_str(&rule.to);
    next.push_str(&value[position + rule.from.len()..]);
    Some(next)
}

fn valid_operation_table(carrier_size: usize, operation_table: &[Vec<usize>]) -> bool {
    carrier_size > 0
        && operation_table.len() == carrier_size
        && operation_table
            .iter()
            .all(|row| row.len() == carrier_size && row.iter().all(|value| *value < carrier_size))
}

fn add_mod_table(carrier_size: usize) -> Vec<Vec<usize>> {
    (0..carrier_size)
        .map(|left| {
            (0..carrier_size)
                .map(|right| (left + right) % carrier_size)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn affine_mod_table(
    carrier_size: usize,
    left_weight: usize,
    right_weight: usize,
    bias: usize,
) -> Vec<Vec<usize>> {
    (0..carrier_size)
        .map(|left| {
            (0..carrier_size)
                .map(|right| (left_weight * left + right_weight * right + bias) % carrier_size)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn algebra_law_result(
    carrier_size: usize,
    operation_table: &[Vec<usize>],
    law: RuliadAlgebraLaw,
    operands: &[usize],
) -> Option<(usize, usize)> {
    if !valid_operation_table(carrier_size, operation_table)
        || operands.iter().any(|operand| *operand >= carrier_size)
    {
        return None;
    }
    let op = |left: usize, right: usize| operation_table[left][right];
    match law {
        RuliadAlgebraLaw::Associativity => {
            if operands.len() != 3 {
                return None;
            }
            let a = operands[0];
            let b = operands[1];
            let c = operands[2];
            Some((op(op(a, b), c), op(a, op(b, c))))
        }
        RuliadAlgebraLaw::Commutativity => {
            if operands.len() != 2 {
                return None;
            }
            let a = operands[0];
            let b = operands[1];
            Some((op(a, b), op(b, a)))
        }
    }
}

fn mod_dot(left: [usize; 2], right: [usize; 2], modulus: usize) -> usize {
    if modulus == 0 {
        return 0;
    }
    (left[0] * right[0] + left[1] * right[1]) % modulus
}

fn mod_norm(value: [usize; 2], modulus: usize) -> usize {
    if modulus == 0 {
        return 0;
    }
    (value[0] * value[0] + value[1] * value[1]) % modulus
}

fn next_prime_at_least(value: usize) -> usize {
    let mut candidate = value.max(2);
    loop {
        if is_prime(candidate) {
            return candidate;
        }
        candidate = candidate.saturating_add(1);
    }
}

fn is_prime(value: usize) -> bool {
    if value < 2 {
        return false;
    }
    if value == 2 {
        return true;
    }
    if value.is_multiple_of(2) {
        return false;
    }
    let mut divisor = 3usize;
    while divisor.saturating_mul(divisor) <= value {
        if value.is_multiple_of(divisor) {
            return false;
        }
        divisor = divisor.saturating_add(2);
    }
    true
}

fn shuffle_rules(rules: &mut [RuliadRewriteRule], rng: &mut SplitMix64) {
    for index in (1..rules.len()).rev() {
        let swap_index = rng.next_usize(index + 1);
        rules.swap(index, swap_index);
    }
}

fn range_or(
    range: Option<crate::config::UsizeRangeConfig>,
    default_min: usize,
    default_max: usize,
    rng: &mut SplitMix64,
) -> usize {
    match range {
        Some(range) => rng.range_usize(range.min, range.max),
        None => rng.range_usize(default_min, default_max),
    }
}

fn sample_stats(spec: &RuliadSampleSpec, text: &str) -> SampleStats {
    let (width, steps, state_count, transition_rate) = match spec {
        RuliadSampleSpec::Eca {
            width,
            steps,
            trace,
            ..
        } => (*width, *steps, 2, trace_transition_rate(trace)),
        RuliadSampleSpec::Simulation {
            width,
            steps,
            source_trace,
            ..
        } => (*width, *steps, 2, trace_transition_rate(source_trace)),
        RuliadSampleSpec::Automaton {
            state_count,
            input,
            trace,
            ..
        } => (
            *state_count,
            input.len(),
            *state_count,
            finite_state_transition_rate(trace),
        ),
        RuliadSampleSpec::Rewrite {
            alphabet,
            steps,
            trace,
            ..
        } => (
            alphabet.len(),
            *steps,
            alphabet.len(),
            string_trace_change_rate(trace),
        ),
        RuliadSampleSpec::Algebra {
            carrier_size,
            holds,
            ..
        } => (
            *carrier_size,
            1,
            *carrier_size,
            if *holds { 0.0 } else { 1.0 },
        ),
        RuliadSampleSpec::Category {
            object_count,
            morphisms,
            path,
            ..
        } => (
            *object_count,
            path.len().saturating_sub(1),
            morphisms.len(),
            finite_state_transition_rate(path),
        ),
        RuliadSampleSpec::ProofTree {
            modulus,
            lemmas,
            proof_steps,
            ..
        } => (
            *modulus,
            lemmas.len().saturating_add(proof_steps.len()),
            *modulus,
            0.75,
        ),
        RuliadSampleSpec::LeanTask { proof, .. } => (1, proof.lines().count().max(1), 2, 0.25),
        RuliadSampleSpec::HashNoise { .. } => (1, 1, 256, 1.0),
    };
    let unique_bytes = text
        .bytes()
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let gzip_complexity_ratio = (unique_bytes as f32 / 256.0).clamp(0.0, 1.0);
    let complexity_score = semantic_difficulty_score(spec, transition_rate, gzip_complexity_ratio);
    SampleStats {
        grid_width: width,
        grid_height: 1,
        steps,
        state_count,
        patch_count_per_frame: width.max(1),
        patch_token_count: text.len(),
        mean_entropy_bits: (unique_bytes as f32).log2().max(0.0),
        mean_transition_rate: transition_rate,
        active_ratio_mean: 0.5,
        unique_frames: steps.saturating_add(1),
        unique_patch_count: unique_bytes,
        frame_uniqueness_ratio: 1.0,
        patch_uniqueness_ratio: gzip_complexity_ratio,
        gzip_complexity_ratio,
        complexity_score,
    }
}

fn semantic_difficulty_score(
    spec: &RuliadSampleSpec,
    transition_rate: f32,
    gzip_complexity_ratio: f32,
) -> f32 {
    let (structural, depth, branching, abstraction) = match spec {
        RuliadSampleSpec::Eca { width, steps, .. } => (*width, *steps, 2, 1),
        RuliadSampleSpec::Simulation { width, steps, .. } => (*width * 2, *steps, 2, 3),
        RuliadSampleSpec::Automaton {
            state_count, input, ..
        } => (*state_count, input.len(), 2, 2),
        RuliadSampleSpec::Rewrite { rules, trace, .. } => {
            (rules.len(), trace.len(), rules.len(), 3)
        }
        RuliadSampleSpec::Algebra {
            carrier_size,
            operation_table,
            ..
        } => (
            *carrier_size,
            2,
            operation_table.len().saturating_mul(operation_table.len()),
            4,
        ),
        RuliadSampleSpec::Category {
            object_count,
            morphisms,
            proof_steps,
            ..
        } => (*object_count, proof_steps.len().max(1), morphisms.len(), 5),
        RuliadSampleSpec::ProofTree {
            modulus,
            lemmas,
            proof_steps,
            ..
        } => (
            *modulus,
            lemmas.len().saturating_add(proof_steps.len()),
            lemmas.len(),
            7,
        ),
        RuliadSampleSpec::LeanTask { proof, .. } => (2, proof.lines().count().max(1), 2, 6),
        RuliadSampleSpec::HashNoise { bytes_hex, .. } => (bytes_hex.len(), 1, 256, 0),
    };
    let structural_score = (structural.max(1) as f32).log2() * 5.0;
    let depth_score = (depth.max(1) as f32).log2() * 6.0;
    let branching_score = (branching.max(1) as f32).log2() * 4.0;
    let abstraction_score = abstraction as f32 * 3.0;
    let dynamic_score = transition_rate.clamp(0.0, 1.0) * 12.0;
    let text_score = gzip_complexity_ratio.clamp(0.0, 1.0) * 6.0;
    (structural_score
        + depth_score
        + branching_score
        + abstraction_score
        + dynamic_score
        + text_score)
        .clamp(0.0, 100.0)
}

pub(crate) fn is_degenerate_spec(spec: &RuliadSampleSpec) -> bool {
    match spec {
        RuliadSampleSpec::Eca { trace, steps, .. } => {
            let unique = trace
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            let collapsed_final = trace.last().is_some_and(|state| {
                state.bytes().all(|byte| byte == b'0') || state.bytes().all(|byte| byte == b'1')
            });
            *steps > 1 && (unique <= 2 || trace_transition_rate(trace) < 0.03 || collapsed_final)
        }
        RuliadSampleSpec::Simulation {
            source_trace,
            target_trace,
            steps,
            ..
        } => {
            *steps > 1
                && (trace_transition_rate(source_trace) < 0.03
                    || trace_transition_rate(target_trace) < 0.03)
        }
        RuliadSampleSpec::Automaton {
            state_count, trace, ..
        } => {
            let unique = trace
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            *state_count > 2 && unique < (*state_count).min(3)
        }
        RuliadSampleSpec::Rewrite { trace, .. } => trace.len() <= 2,
        RuliadSampleSpec::Algebra { .. }
        | RuliadSampleSpec::Category { .. }
        | RuliadSampleSpec::ProofTree { .. }
        | RuliadSampleSpec::LeanTask { .. }
        | RuliadSampleSpec::HashNoise { .. } => false,
    }
}

fn trace_transition_rate(trace: &[String]) -> f32 {
    let mut changed = 0usize;
    let mut total = 0usize;
    for pair in trace.windows(2) {
        let left = pair[0].as_bytes();
        let right = pair[1].as_bytes();
        for (a, b) in left.iter().zip(right) {
            total += 1;
            changed += usize::from(a != b);
        }
    }
    if total == 0 {
        0.0
    } else {
        changed as f32 / total as f32
    }
}

fn string_trace_change_rate(trace: &[String]) -> f32 {
    let mut changed = 0usize;
    let mut total = 0usize;
    for pair in trace.windows(2) {
        let left = pair[0].as_bytes();
        let right = pair[1].as_bytes();
        total += left.len().max(right.len());
        changed += left
            .iter()
            .zip(right.iter())
            .filter(|(left_byte, right_byte)| left_byte != right_byte)
            .count();
        changed += left.len().abs_diff(right.len());
    }
    if total == 0 {
        0.0
    } else {
        changed as f32 / total as f32
    }
}

fn finite_state_transition_rate(trace: &[usize]) -> f32 {
    if trace.len() <= 1 {
        return 0.0;
    }
    let changed = trace.windows(2).filter(|pair| pair[0] != pair[1]).count();
    changed as f32 / trace.len().saturating_sub(1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::UsizeRangeConfig;
    use crate::ruliad::config::{
        RULIAD_REQUIRED_MATH_DOMAINS, RULIAD_REQUIRED_REASONING_MODES, RuliadSerializationConfig,
        RuliadTokenizationConfig, default_ruliad_families,
    };

    fn config() -> RuliadCorpusConfig {
        RuliadCorpusConfig {
            output_dir: "target/test-ruliad".into(),
            seed: 5,
            name: "test".to_string(),
            train_samples: 8,
            validation_samples: 2,
            chunk_token_capacity: 1024,
            serialization: RuliadSerializationConfig::default(),
            tokenization: RuliadTokenizationConfig::default(),
            source_selection: crate::ruliad::config::RuliadSourceSelectionConfig::default(),
            families: default_ruliad_families(),
            proof_tasks: None,
            lean_task_limit: None,
        }
    }

    #[test]
    fn generated_samples_verify() {
        for index in 0..16 {
            let sample =
                generate_sample(&config(), &[], SampleSplit::Train, 0, index).expect("sample");
            let report = verify_spec(&sample.spec).expect("verify");
            assert!(report.ok);
            assert_eq!(report.oracle_hash, sample.oracle_hash);
        }
    }

    #[test]
    fn corrupted_eca_trace_is_rejected() {
        let mut config = config();
        config.families = vec![RuliadFamilyConfig {
            kind: RuliadFamilyKind::Eca,
            weight: 1,
            width: Some(UsizeRangeConfig { min: 8, max: 8 }),
            steps: Some(UsizeRangeConfig { min: 4, max: 4 }),
        }];
        let mut sample = generate_sample(&config, &[], SampleSplit::Train, 0, 0).expect("sample");
        if let RuliadSampleSpec::Eca { trace, .. } = &mut sample.spec {
            trace[0].push('1');
            assert!(!verify_spec(&sample.spec).expect("verify").ok);
        } else {
            panic!("expected ECA sample");
        }
    }

    #[test]
    fn proof_task_hash_is_checked() {
        let task = default_proof_tasks().remove(0);
        assert!(task.validate_hash());
    }

    #[test]
    fn serialized_samples_use_categorical_abstraction_as_primary_view() {
        for family in [
            RuliadFamilyKind::Eca,
            RuliadFamilyKind::Simulation,
            RuliadFamilyKind::Automaton,
            RuliadFamilyKind::Rewrite,
            RuliadFamilyKind::Algebra,
            RuliadFamilyKind::Category,
            RuliadFamilyKind::ProofTree,
            RuliadFamilyKind::LeanTask,
        ] {
            let mut config = config();
            config.families = vec![RuliadFamilyConfig {
                kind: family,
                weight: 1,
                width: match family {
                    RuliadFamilyKind::Eca | RuliadFamilyKind::Simulation => {
                        Some(UsizeRangeConfig { min: 8, max: 8 })
                    }
                    RuliadFamilyKind::Automaton => Some(UsizeRangeConfig { min: 4, max: 4 }),
                    RuliadFamilyKind::Rewrite => Some(UsizeRangeConfig { min: 8, max: 8 }),
                    RuliadFamilyKind::Algebra => Some(UsizeRangeConfig { min: 3, max: 3 }),
                    RuliadFamilyKind::Category => Some(UsizeRangeConfig { min: 4, max: 4 }),
                    RuliadFamilyKind::ProofTree => Some(UsizeRangeConfig { min: 5, max: 5 }),
                    RuliadFamilyKind::LeanTask | RuliadFamilyKind::HashNoise => None,
                },
                steps: match family {
                    RuliadFamilyKind::Eca | RuliadFamilyKind::Simulation => {
                        Some(UsizeRangeConfig { min: 4, max: 4 })
                    }
                    RuliadFamilyKind::Automaton => Some(UsizeRangeConfig { min: 6, max: 6 }),
                    RuliadFamilyKind::Rewrite => Some(UsizeRangeConfig { min: 4, max: 4 }),
                    RuliadFamilyKind::Category => Some(UsizeRangeConfig { min: 3, max: 3 }),
                    RuliadFamilyKind::ProofTree => Some(UsizeRangeConfig { min: 4, max: 4 }),
                    RuliadFamilyKind::Algebra
                    | RuliadFamilyKind::LeanTask
                    | RuliadFamilyKind::HashNoise => None,
                },
            }];
            let sample = generate_sample(&config, &[], SampleSplit::Train, 0, 0).expect("sample");
            assert!(sample.categorical_presentation.categorical_core);
            assert_eq!(
                sample.categorical_presentation.abstraction,
                "finite_category_reasoning"
            );
            assert_eq!(
                sample.categorical_presentation.source_family,
                family.label()
            );
            assert!(sample.text.starts_with("[R2 "));
            assert!(sample.text.contains("\n?:"));
            assert!(sample.text.contains("\n!:"));
            assert!(
                sample.text.len() <= 512,
                "{} sample exceeded trace-pretraining payload budget: {} bytes",
                family.label(),
                sample.text.len()
            );
        }
    }

    #[test]
    fn prompt_prefix_ends_at_canonical_answer_slot() {
        let mut config = config();
        config.families = vec![RuliadFamilyConfig {
            kind: RuliadFamilyKind::ProofTree,
            weight: 1,
            width: Some(UsizeRangeConfig { min: 5, max: 5 }),
            steps: Some(UsizeRangeConfig { min: 4, max: 4 }),
        }];
        let sample = generate_sample(&config, &[], SampleSplit::Train, 0, 0).expect("sample");
        let prompt = ruliad_prompt_prefix(&sample.spec, &sample.oracle_hash);
        assert!(prompt.ends_with("!:"));
        assert!(!prompt.contains("holds=true"));
        assert!(!prompt.contains("[/R2]"));
    }

    #[test]
    fn generated_category_tasks_verify_and_exercise_laws() {
        for task_kind in [
            RuliadTaskKind::ComposeCategoryPath,
            RuliadTaskKind::VerifyCategoryLaw,
            RuliadTaskKind::VerifyFunctorPreservation,
            RuliadTaskKind::VerifyNaturalitySquare,
        ] {
            let mut rng = sample_rng(42, SampleSplit::Train, 0, task_kind as usize, 0);
            let sample = generate_category_spec_for_task(
                &RuliadFamilyConfig {
                    kind: RuliadFamilyKind::Category,
                    weight: 1,
                    width: Some(UsizeRangeConfig { min: 5, max: 5 }),
                    steps: Some(UsizeRangeConfig { min: 4, max: 4 }),
                },
                task_kind,
                &mut rng,
            )
            .expect("category spec");
            let report = verify_spec(&sample).expect("verify");
            assert!(report.ok, "task {} should verify", task_kind.label());
            let text = sample_text(&sample, &report.oracle_hash);
            assert!(
                text.len() <= 512,
                "task {} text exceeded payload budget: {} bytes",
                task_kind.label(),
                text.len()
            );
        }
    }

    #[test]
    fn proof_tree_theorem_verifies_without_named_memorization_target() {
        let mut rng = sample_rng(61, SampleSplit::Train, 0, 0, 0);
        let spec = generate_proof_tree_spec(
            &RuliadFamilyConfig {
                kind: RuliadFamilyKind::ProofTree,
                weight: 1,
                width: Some(UsizeRangeConfig { min: 11, max: 11 }),
                steps: Some(UsizeRangeConfig { min: 8, max: 8 }),
            },
            &mut rng,
        )
        .expect("proof tree");
        let report = verify_spec(&spec).expect("verify");
        assert!(report.ok);
        let text = sample_text(&spec, &report.oracle_hash);
        assert!(text.starts_with("[R2 "));
        assert!(text.contains("\n!:holds=true"));
        assert!(!text.to_lowercase().contains("pythag"));
    }

    #[test]
    fn semantic_difficulty_increases_with_theorem_tree_depth() {
        let mut easy_rng = sample_rng(62, SampleSplit::Train, 0, 0, 0);
        let easy = generate_proof_tree_spec(
            &RuliadFamilyConfig {
                kind: RuliadFamilyKind::ProofTree,
                weight: 1,
                width: Some(UsizeRangeConfig { min: 5, max: 5 }),
                steps: Some(UsizeRangeConfig { min: 4, max: 4 }),
            },
            &mut easy_rng,
        )
        .expect("easy proof tree");
        let mut hard_rng = sample_rng(62, SampleSplit::Train, 0, 0, 0);
        let hard = generate_proof_tree_spec(
            &RuliadFamilyConfig {
                kind: RuliadFamilyKind::ProofTree,
                weight: 1,
                width: Some(UsizeRangeConfig { min: 29, max: 29 }),
                steps: Some(UsizeRangeConfig { min: 16, max: 16 }),
            },
            &mut hard_rng,
        )
        .expect("hard proof tree");
        let easy_hash = verify_spec(&easy).expect("verify easy").oracle_hash;
        let hard_hash = verify_spec(&hard).expect("verify hard").oracle_hash;
        let easy_stats = sample_stats(&easy, &sample_text(&easy, &easy_hash));
        let hard_stats = sample_stats(&hard, &sample_text(&hard, &hard_hash));
        assert!(
            hard_stats.complexity_score > easy_stats.complexity_score,
            "hard={} easy={}",
            hard_stats.complexity_score,
            easy_stats.complexity_score
        );
    }

    #[test]
    fn corrupted_category_composition_is_rejected() {
        let mut rng = sample_rng(43, SampleSplit::Train, 0, 0, 0);
        let mut sample = generate_category_spec_for_task(
            &RuliadFamilyConfig {
                kind: RuliadFamilyKind::Category,
                weight: 1,
                width: Some(UsizeRangeConfig { min: 4, max: 4 }),
                steps: Some(UsizeRangeConfig { min: 3, max: 3 }),
            },
            RuliadTaskKind::VerifyCategoryLaw,
            &mut rng,
        )
        .expect("category spec");
        let RuliadSampleSpec::Category { composition, .. } = &mut sample else {
            panic!("expected category");
        };
        composition[0][0] = Some(1);
        assert!(!verify_spec(&sample).expect("verify").ok);
    }

    #[test]
    fn corrupted_functor_and_naturality_are_rejected() {
        let family = RuliadFamilyConfig {
            kind: RuliadFamilyKind::Category,
            weight: 1,
            width: Some(UsizeRangeConfig { min: 5, max: 5 }),
            steps: Some(UsizeRangeConfig { min: 4, max: 4 }),
        };
        let mut functor_rng = sample_rng(44, SampleSplit::Train, 0, 0, 0);
        let mut functor_sample = generate_category_spec_for_task(
            &family,
            RuliadTaskKind::VerifyFunctorPreservation,
            &mut functor_rng,
        )
        .expect("functor spec");
        let RuliadSampleSpec::Category { functor, .. } = &mut functor_sample else {
            panic!("expected category");
        };
        let functor = functor.as_mut().expect("functor");
        functor.morphism_map[0] = functor.morphism_map[0].saturating_add(1);
        assert!(!verify_spec(&functor_sample).expect("verify").ok);

        let mut short_map_sample = generate_category_spec_for_task(
            &family,
            RuliadTaskKind::VerifyFunctorPreservation,
            &mut sample_rng(46, SampleSplit::Train, 0, 0, 0),
        )
        .expect("functor spec");
        let RuliadSampleSpec::Category { functor, .. } = &mut short_map_sample else {
            panic!("expected category");
        };
        let functor = functor.as_mut().expect("functor");
        functor.object_map.pop();
        assert!(!verify_spec(&short_map_sample).expect("verify").ok);

        let mut naturality_rng = sample_rng(45, SampleSplit::Train, 0, 0, 0);
        let mut naturality_sample = generate_category_spec_for_task(
            &family,
            RuliadTaskKind::VerifyNaturalitySquare,
            &mut naturality_rng,
        )
        .expect("naturality spec");
        let RuliadSampleSpec::Category { naturality, .. } = &mut naturality_sample else {
            panic!("expected category");
        };
        let naturality = naturality.as_mut().expect("naturality");
        naturality.right_path.reverse();
        assert!(!verify_spec(&naturality_sample).expect("verify").ok);
    }

    #[test]
    fn default_distribution_spans_computable_families() {
        let mut family_counts = std::collections::HashMap::new();
        let mut task_counts = std::collections::HashMap::new();
        let mut oracle_hashes = std::collections::BTreeSet::new();
        let mut eca_rules = std::collections::BTreeSet::new();
        let mut widths = std::collections::BTreeSet::new();
        let mut step_counts = std::collections::BTreeSet::new();
        let mut algebra_outcomes = std::collections::BTreeSet::new();
        let mut rewrite_lengths = std::collections::BTreeSet::new();
        let mut math_domains = std::collections::BTreeSet::new();
        let mut reasoning_modes = std::collections::BTreeSet::new();
        let sample_count = 1024;

        for index in 0..sample_count {
            let sample =
                generate_sample(&config(), &[], SampleSplit::Train, 0, index).expect("sample");
            *family_counts.entry(sample.family).or_insert(0usize) += 1;
            *task_counts.entry(sample.task_kind).or_insert(0usize) += 1;
            oracle_hashes.insert(sample.oracle_hash);
            let semantics = ruliad_source_semantics(sample.family, sample.task_kind);
            math_domains.extend(semantics.math_domains.iter().copied());
            reasoning_modes.extend(semantics.reasoning_modes.iter().copied());
            assert_eq!(
                sample.categorical_presentation.source_family,
                sample.family.label()
            );
            assert!(!sample.categorical_presentation.presentation.is_empty());
            if sample.family == RuliadFamilyKind::HashNoise {
                assert!(!sample.categorical_presentation.categorical_core);
                assert_eq!(
                    sample.categorical_presentation.abstraction,
                    "source_selection_canary"
                );
            } else {
                assert!(sample.categorical_presentation.categorical_core);
                assert_eq!(
                    sample.categorical_presentation.abstraction,
                    "finite_category_reasoning"
                );
            }

            match &sample.spec {
                RuliadSampleSpec::Eca {
                    rule, width, steps, ..
                } => {
                    eca_rules.insert(*rule);
                    widths.insert(*width);
                    step_counts.insert(*steps);
                }
                RuliadSampleSpec::Simulation { width, steps, .. } => {
                    widths.insert(*width);
                    step_counts.insert(*steps);
                }
                RuliadSampleSpec::Automaton {
                    state_count, input, ..
                } => {
                    widths.insert(*state_count);
                    step_counts.insert(input.len());
                }
                RuliadSampleSpec::Rewrite {
                    initial,
                    steps,
                    normal_form,
                    ..
                } => {
                    widths.insert(initial.len());
                    step_counts.insert(*steps);
                    rewrite_lengths.insert(normal_form.len());
                }
                RuliadSampleSpec::Algebra {
                    carrier_size,
                    holds,
                    ..
                } => {
                    widths.insert(*carrier_size);
                    algebra_outcomes.insert(*holds);
                }
                RuliadSampleSpec::Category {
                    object_count,
                    morphisms,
                    path,
                    ..
                } => {
                    widths.insert(*object_count);
                    widths.insert(morphisms.len());
                    step_counts.insert(path.len().saturating_sub(1));
                }
                RuliadSampleSpec::ProofTree {
                    modulus,
                    lemmas,
                    proof_steps,
                    holds,
                    ..
                } => {
                    widths.insert(*modulus);
                    step_counts.insert(lemmas.len().saturating_add(proof_steps.len()));
                    algebra_outcomes.insert(*holds);
                }
                RuliadSampleSpec::LeanTask { .. } | RuliadSampleSpec::HashNoise { .. } => {}
            }
        }

        for family in [
            RuliadFamilyKind::Eca,
            RuliadFamilyKind::Simulation,
            RuliadFamilyKind::Automaton,
            RuliadFamilyKind::Rewrite,
            RuliadFamilyKind::Algebra,
            RuliadFamilyKind::Category,
            RuliadFamilyKind::ProofTree,
            RuliadFamilyKind::LeanTask,
            RuliadFamilyKind::HashNoise,
        ] {
            assert!(
                family_counts.get(&family).copied().unwrap_or_default() > 0,
                "missing family {}",
                family.label()
            );
        }

        for task_kind in [
            RuliadTaskKind::MultiStepState,
            RuliadTaskKind::VerifySimulation,
            RuliadTaskKind::EvaluateAutomaton,
            RuliadTaskKind::RewriteNormalForm,
            RuliadTaskKind::CheckAlgebraLaw,
            RuliadTaskKind::ComposeCategoryPath,
            RuliadTaskKind::VerifyCategoryLaw,
            RuliadTaskKind::VerifyFunctorPreservation,
            RuliadTaskKind::VerifyNaturalitySquare,
            RuliadTaskKind::ProveTheorem,
            RuliadTaskKind::CompleteProof,
            RuliadTaskKind::HashCanary,
        ] {
            assert!(
                task_counts.get(&task_kind).copied().unwrap_or_default() > 0,
                "missing task {}",
                task_kind.label()
            );
        }

        for domain in RULIAD_REQUIRED_MATH_DOMAINS {
            assert!(
                math_domains.contains(domain),
                "missing ruliad math domain {}",
                domain.label()
            );
        }

        for mode in RULIAD_REQUIRED_REASONING_MODES {
            assert!(
                reasoning_modes.contains(mode),
                "missing ruliad reasoning mode {}",
                mode.label()
            );
        }

        assert!(
            oracle_hashes.len() > sample_count * 9 / 10,
            "oracle hashes collapsed: {} unique of {}",
            oracle_hashes.len(),
            sample_count
        );
        assert!(
            eca_rules.len() > 96,
            "too few ECA rules: {}",
            eca_rules.len()
        );
        assert!(
            widths.len() > 12,
            "too few width/state bands: {}",
            widths.len()
        );
        assert!(
            step_counts.len() > 12,
            "too few step/input bands: {}",
            step_counts.len()
        );
        assert_eq!(
            algebra_outcomes.len(),
            2,
            "algebra probes should include true and false outcomes"
        );
        assert!(
            rewrite_lengths.len() > 4,
            "rewrite samples have too little terminal-length variety"
        );
    }
}
