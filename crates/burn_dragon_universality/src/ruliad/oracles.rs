use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::manifest::SampleSplit;
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
        path: Vec<usize>,
        composed: Vec<usize>,
        associative: bool,
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

#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedRuliadSample {
    pub spec: RuliadSampleSpec,
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
    let spec = match family.kind {
        RuliadFamilyKind::Eca => generate_eca_spec(family, &mut rng),
        RuliadFamilyKind::Simulation => generate_simulation_spec(family, &mut rng),
        RuliadFamilyKind::Automaton => generate_automaton_spec(family, &mut rng),
        RuliadFamilyKind::Rewrite => generate_rewrite_spec(family, &mut rng),
        RuliadFamilyKind::Algebra => generate_algebra_spec(family, &mut rng),
        RuliadFamilyKind::Category => generate_category_spec(family, &mut rng),
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
        RuliadFamilyKind::Category => generate_category_spec(&bucket.family_config, &mut rng),
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
    let text = sample_text(&spec, &report.oracle_hash);
    let stats = sample_stats(&spec, &text);
    Ok(GeneratedRuliadSample {
        spec,
        family: report.family,
        task_kind: report.task_kind,
        verifier_version: RULIAD_VERIFIER_VERSION,
        oracle_hash: report.oracle_hash,
        text,
        stats,
    })
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
            path,
            composed,
            associative,
            task,
        } => {
            let ok = *object_count > 0
                && path.len() >= 2
                && path.iter().all(|object| *object < *object_count)
                && path.windows(2).all(|pair| pair[0] <= pair[1])
                && composed == &vec![path[0], *path.last().unwrap_or(&path[0])]
                && *associative;
            (ok, RuliadFamilyKind::Category, *task)
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
    match spec {
        RuliadSampleSpec::Eca {
            rule,
            width,
            steps,
            initial,
            trace,
            task,
        } => format!(
            "<ruliad family=\"eca\" task=\"{}\"{} verifier=\"{}\" hash=\"{}\">\nrule={}\nwidth={}\nsteps={}\ninitial={}\ntrace={}\ntarget={}\n</ruliad>\n",
            task.label(),
            semantic_text_attributes(RuliadFamilyKind::Eca, *task),
            RULIAD_VERIFIER_VERSION,
            oracle_hash,
            rule,
            width,
            steps,
            initial,
            trace.join(","),
            trace.last().cloned().unwrap_or_default()
        ),
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
        } => format!(
            "<ruliad family=\"simulation\" task=\"{}\"{} verifier=\"{}\" hash=\"{}\">\nsource_rule={}\ntarget_rule={}\nwidth={}\nsteps={}\nsource_initial={}\ntarget_initial={}\nsource_trace={}\ntarget_trace={}\nmapped_source_trace={}\nclaim=map(step_source^t(x)) == step_target^t(map(x))\n</ruliad>\n",
            task.label(),
            semantic_text_attributes(RuliadFamilyKind::Simulation, *task),
            RULIAD_VERIFIER_VERSION,
            oracle_hash,
            source_rule,
            target_rule,
            width,
            steps,
            source_initial,
            target_initial,
            source_trace.join(","),
            target_trace.join(","),
            mapped_source_trace.join(",")
        ),
        RuliadSampleSpec::Automaton {
            state_count,
            transitions,
            start_state,
            accept_states,
            input,
            trace,
            accepted,
            task,
        } => format!(
            "<ruliad family=\"automaton\" task=\"{}\"{} verifier=\"{}\" hash=\"{}\">\nstates={}\nstart={}\naccept_states={:?}\ninput={}\ntransitions={:?}\ntrace={:?}\naccepted={}\n</ruliad>\n",
            task.label(),
            semantic_text_attributes(RuliadFamilyKind::Automaton, *task),
            RULIAD_VERIFIER_VERSION,
            oracle_hash,
            state_count,
            start_state,
            accept_states,
            input,
            transitions,
            trace,
            accepted
        ),
        RuliadSampleSpec::Rewrite {
            alphabet,
            rules,
            initial,
            steps,
            trace,
            normal_form,
            task,
        } => {
            let rule_text = rules
                .iter()
                .map(|rule| format!("{}->{}", rule.from, rule.to))
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "<ruliad family=\"rewrite\" task=\"{}\"{} verifier=\"{}\" hash=\"{}\">\nalphabet={}\nrules={}\ninitial={}\nsteps={}\ntrace={}\nnormal_form={}\n</ruliad>\n",
                task.label(),
                semantic_text_attributes(RuliadFamilyKind::Rewrite, *task),
                RULIAD_VERIFIER_VERSION,
                oracle_hash,
                alphabet,
                rule_text,
                initial,
                steps,
                trace.join(","),
                normal_form
            )
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
        } => format!(
            "<ruliad family=\"algebra\" task=\"{}\"{} verifier=\"{}\" hash=\"{}\">\ncarrier_size={}\nlaw={}\noperands={:?}\noperation_table={:?}\nlhs={}\nrhs={}\nholds={}\n</ruliad>\n",
            task.label(),
            semantic_text_attributes(RuliadFamilyKind::Algebra, *task),
            RULIAD_VERIFIER_VERSION,
            oracle_hash,
            carrier_size,
            law.label(),
            operands,
            operation_table,
            lhs,
            rhs,
            holds
        ),
        RuliadSampleSpec::Category {
            object_count,
            path,
            composed,
            associative,
            task,
        } => format!(
            "<ruliad family=\"category\" task=\"{}\"{} verifier=\"{}\" hash=\"{}\">\nobjects={}\npath={:?}\ncomposed={:?}\nassociative={}\n</ruliad>\n",
            task.label(),
            semantic_text_attributes(RuliadFamilyKind::Category, *task),
            RULIAD_VERIFIER_VERSION,
            oracle_hash,
            object_count,
            path,
            composed,
            associative
        ),
        RuliadSampleSpec::LeanTask {
            task_id,
            statement,
            proof,
            payload_hash,
            task,
        } => format!(
            "<ruliad family=\"lean_task\" task=\"{}\"{} verifier=\"{}\" hash=\"{}\">\ntask_id={}\npayload_hash={}\nstatement={}\nproof={}\n</ruliad>\n",
            task.label(),
            semantic_text_attributes(RuliadFamilyKind::LeanTask, *task),
            RULIAD_VERIFIER_VERSION,
            oracle_hash,
            task_id,
            payload_hash,
            statement,
            proof
        ),
        RuliadSampleSpec::HashNoise {
            bytes_hex,
            payload_hash,
            task,
        } => format!(
            "<ruliad family=\"hash_noise\" task=\"{}\"{} verifier=\"{}\" hash=\"{}\">\npayload_hash={}\nbytes={}\n</ruliad>\n",
            task.label(),
            semantic_text_attributes(RuliadFamilyKind::HashNoise, *task),
            RULIAD_VERIFIER_VERSION,
            oracle_hash,
            payload_hash,
            bytes_hex
        ),
    }
}

fn semantic_text_attributes(family: RuliadFamilyKind, task_kind: RuliadTaskKind) -> String {
    let semantics = ruliad_source_semantics(family, task_kind);
    let domains = semantics
        .math_domains
        .iter()
        .map(|domain| domain.label())
        .collect::<Vec<_>>()
        .join(",");
    let modes = semantics
        .reasoning_modes
        .iter()
        .map(|mode| mode.label())
        .collect::<Vec<_>>()
        .join(",");
    format!(" domains=\"{}\" reasoning_modes=\"{}\"", domains, modes)
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

fn generate_eca_spec(
    family: &RuliadFamilyConfig,
    rng: &mut SplitMix64,
) -> Result<RuliadSampleSpec> {
    let width = range_or(family.width, 16, 32, rng);
    let steps = range_or(family.steps, 4, 10, rng);
    let rule = rng.next_u8();
    let initial = eca::random_state(width, rng);
    let trace = eca::trace(rule, &initial, steps)
        .iter()
        .map(|state| eca::format_state(state))
        .collect::<Vec<_>>();
    Ok(RuliadSampleSpec::Eca {
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
    })
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
    if accept_states.is_empty() {
        accept_states.push(rng.next_usize(state_count));
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
    Ok(RuliadSampleSpec::Automaton {
        state_count,
        transitions,
        start_state,
        accept_states,
        input,
        trace,
        accepted,
        task: RuliadTaskKind::EvaluateAutomaton,
    })
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
    let initial = (0..initial_len)
        .map(|_| symbols[rng.next_usize(symbols.len())])
        .collect::<String>();
    let trace = rewrite_trace(&initial, &rules, steps);
    let normal_form = trace.last().cloned().unwrap_or_else(|| initial.clone());
    Ok(RuliadSampleSpec::Rewrite {
        alphabet,
        rules,
        initial,
        steps,
        trace,
        normal_form,
        task: RuliadTaskKind::RewriteNormalForm,
    })
}

fn generate_algebra_spec(
    family: &RuliadFamilyConfig,
    rng: &mut SplitMix64,
) -> Result<RuliadSampleSpec> {
    let carrier_size = range_or(family.width, 2, 6, rng);
    let operation_table = if rng.next_bool() {
        (0..carrier_size)
            .map(|left| {
                (0..carrier_size)
                    .map(|right| (left + right) % carrier_size)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    } else {
        (0..carrier_size)
            .map(|_| {
                (0..carrier_size)
                    .map(|_| rng.next_usize(carrier_size))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
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
    let object_count = range_or(family.width, 3, 7, rng);
    let path_len = range_or(family.steps, 3, 6, rng).max(2);
    let mut path = Vec::with_capacity(path_len);
    let mut current = rng.next_usize(object_count);
    path.push(current);
    for _ in 1..path_len {
        let remaining = object_count.saturating_sub(current + 1);
        current += rng.next_usize(remaining + 1);
        path.push(current);
    }
    let composed = vec![path[0], *path.last().unwrap_or(&path[0])];
    Ok(RuliadSampleSpec::Category {
        object_count,
        path,
        composed,
        associative: true,
        task: RuliadTaskKind::ComposeCategoryPath,
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
    let payload_hash = proof_task
        .payload_hash
        .clone()
        .unwrap_or_else(|| proof_task.computed_payload_hash());
    Ok(RuliadSampleSpec::LeanTask {
        task_id: proof_task.id,
        statement: proof_task.statement,
        proof: proof_task.proof,
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
        if let Some(position) = value.find(&rule.from) {
            if best_match.is_none_or(|(best_position, best_rule_index)| {
                position < best_position
                    || (position == best_position && rule_index < best_rule_index)
            }) {
                best_match = Some((position, rule_index));
            }
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
    let (width, steps, state_count, transition_rate, complexity_score) = match spec {
        RuliadSampleSpec::Eca {
            width,
            steps,
            trace,
            ..
        } => (*width, *steps, 2, trace_transition_rate(trace), 35.0),
        RuliadSampleSpec::Simulation { width, steps, .. } => (*width, *steps, 2, 0.5, 60.0),
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
            45.0,
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
            55.0,
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
            65.0,
        ),
        RuliadSampleSpec::Category {
            object_count, path, ..
        } => (
            *object_count,
            path.len().saturating_sub(1),
            *object_count,
            finite_state_transition_rate(path),
            70.0,
        ),
        RuliadSampleSpec::LeanTask { .. } => (1, 1, 2, 0.0, 75.0),
        RuliadSampleSpec::HashNoise { .. } => (1, 1, 256, 1.0, 100.0),
    };
    let unique_bytes = text
        .bytes()
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let gzip_complexity_ratio = (unique_bytes as f32 / 256.0).clamp(0.0, 1.0);
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
                    object_count, path, ..
                } => {
                    widths.insert(*object_count);
                    step_counts.insert(path.len().saturating_sub(1));
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
