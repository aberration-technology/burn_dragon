//! Independent Lean replay for generated Ruliad proof certificates.

use std::path::Path;

#[cfg(not(target_arch = "wasm32"))]
use std::{
    env, fs,
    process::Command,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[cfg(not(target_arch = "wasm32"))]
use anyhow::Context;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
#[cfg(not(target_arch = "wasm32"))]
use sha2::{Digest, Sha256};

use crate::ruliad::formal::{
    RuliadFormalGenerationSplit, RuliadFormalGeneratorConfig, generate_formal_bundle,
};
use crate::ruliad::ir::{
    RuliadFormalDomain, RuliadProofBundle, RuliadProofCertificate, RuliadProofProblem,
    RuliadProofSource, RuliadProofStep, RuliadRewriteDirection, RuliadTerm,
};
use crate::ruliad::rng::SplitMix64;

pub const RULIAD_LEAN_CHECKER_VERSION: u32 = 1;
pub const RULIAD_LEAN_PANEL_CONTRACT: &str = "burn-dragon-ruliad-lean-panel-v1";
pub const RULIAD_LEAN_MAX_BUNDLES_PER_MODULE: usize = 4;
pub const RULIAD_LEAN_MAX_HEARTBEATS_PER_COMMAND: u64 = 2_000_000;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadLeanVerificationReport {
    pub checker_version: u32,
    pub formal_samples_checked: usize,
    pub negative_cases_checked: usize,
    pub modules_checked: usize,
    pub max_bundles_per_module: usize,
    pub max_heartbeats_per_command: u64,
    pub source_bytes: usize,
    pub source_sha256: String,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuliadLeanPanelReport {
    pub contract: String,
    pub seed: u64,
    pub difficulty_levels: Vec<usize>,
    pub samples_per_domain: usize,
    pub formal_domains: Vec<String>,
    pub generation_split: String,
    pub verification: RuliadLeanVerificationReport,
}

fn formal_verification_panel(
    seed: u64,
    difficulty_levels: &[usize],
    samples_per_domain: usize,
) -> Result<(Vec<RuliadProofBundle>, Vec<usize>)> {
    if difficulty_levels.is_empty() {
        return Err(anyhow!(
            "Lean verification panel requires a difficulty level"
        ));
    }
    if samples_per_domain == 0 {
        return Err(anyhow!(
            "Lean verification panel samples_per_domain must be greater than zero"
        ));
    }
    let mut levels = difficulty_levels.to_vec();
    levels.sort_unstable();
    levels.dedup();
    let panel_len = levels
        .len()
        .checked_mul(RuliadFormalDomain::ALL.len())
        .and_then(|count| count.checked_mul(samples_per_domain))
        .ok_or_else(|| anyhow!("Lean verification panel size overflow"))?;
    let mut bundles = Vec::with_capacity(panel_len);
    let mut seeds = SplitMix64::new(seed);
    for level in &levels {
        for domain in RuliadFormalDomain::ALL {
            for _ in 0..samples_per_domain {
                let mut config = RuliadFormalGeneratorConfig::for_difficulty(*level);
                config.domain = Some(domain);
                config.generation_split = RuliadFormalGenerationSplit::StructuralValidationV1;
                bundles.push(generate_formal_bundle(seeds.next_u64(), config)?);
            }
        }
    }
    Ok((bundles, levels))
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn verify_formal_panel_with_lean(
    seed: u64,
    difficulty_levels: &[usize],
    samples_per_domain: usize,
    project: &Path,
    lake: &Path,
) -> Result<RuliadLeanPanelReport> {
    let (bundles, difficulty_levels) =
        formal_verification_panel(seed, difficulty_levels, samples_per_domain)?;
    let verification = verify_formal_bundles_with_lean(&bundles, project, lake)?;
    Ok(RuliadLeanPanelReport {
        contract: RULIAD_LEAN_PANEL_CONTRACT.to_string(),
        seed,
        difficulty_levels,
        samples_per_domain,
        formal_domains: RuliadFormalDomain::ALL
            .into_iter()
            .map(|domain| domain.label().to_string())
            .collect(),
        generation_split: RuliadFormalGenerationSplit::StructuralValidationV1
            .label()
            .to_string(),
        verification,
    })
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn verify_formal_panel_with_lean(
    _seed: u64,
    _difficulty_levels: &[usize],
    _samples_per_domain: usize,
    project: &Path,
    _lake: &Path,
) -> Result<RuliadLeanPanelReport> {
    Err(anyhow!(
        "generated Lean panel verification is unavailable for wasm target ({})",
        project.display()
    ))
}

pub fn render_lean_verification_module(bundles: &[RuliadProofBundle]) -> Result<(String, usize)> {
    let mut source = format!(
        "import RuliadSeed.Checker\n\nset_option maxHeartbeats {RULIAD_LEAN_MAX_HEARTBEATS_PER_COMMAND}\n\nnamespace RuliadExternalVerification\nopen RuliadSeed\n\n"
    );
    let mut negative_cases = 0usize;

    for (index, bundle) in bundles.iter().enumerate() {
        source.push_str(&format!(
            "def problem{index} : Problem := {}\n\n",
            render_problem(&bundle.problem)
        ));
        source.push_str(&format!(
            "def certificate{index} : Certificate := {}\n\n",
            render_certificate(&bundle.certificate)
        ));
        source.push_str(&format!(
            "example : checkCertificate problem{index} certificate{index} = true := by\n  native_decide\n\n"
        ));

        for (label, certificate) in adversarial_certificates(bundle) {
            negative_cases = negative_cases.saturating_add(1);
            let name = format!("certificate{index}_{label}");
            source.push_str(&format!(
                "def {name} : Certificate := {}\n\n",
                render_certificate(&certificate)
            ));
            source.push_str(&format!(
                "example : checkCertificate problem{index} {name} = false := by\n  native_decide\n\n"
            ));
        }
    }

    source.push_str("end RuliadExternalVerification\n");
    Ok((source, negative_cases))
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn verify_formal_bundles_with_lean(
    bundles: &[RuliadProofBundle],
    project: &Path,
    lake: &Path,
) -> Result<RuliadLeanVerificationReport> {
    let started = Instant::now();
    let mut source_hash = Sha256::new();
    let mut source_bytes = 0usize;
    let mut negative_cases_checked = 0usize;
    let mut modules_checked = 0usize;
    for (module_index, shard) in bundles
        .chunks(RULIAD_LEAN_MAX_BUNDLES_PER_MODULE)
        .enumerate()
    {
        let (source, negative_cases) = render_lean_verification_module(shard)?;
        let source_len = u64::try_from(source.len()).unwrap_or(u64::MAX);
        source_hash.update(source_len.to_le_bytes());
        source_hash.update(source.as_bytes());
        source_bytes = source_bytes.saturating_add(source.len());
        negative_cases_checked = negative_cases_checked.saturating_add(negative_cases);
        verify_lean_source(&source, module_index, project, lake)?;
        modules_checked = modules_checked.saturating_add(1);
    }

    Ok(RuliadLeanVerificationReport {
        checker_version: RULIAD_LEAN_CHECKER_VERSION,
        formal_samples_checked: bundles.len(),
        negative_cases_checked,
        modules_checked,
        max_bundles_per_module: RULIAD_LEAN_MAX_BUNDLES_PER_MODULE,
        max_heartbeats_per_command: RULIAD_LEAN_MAX_HEARTBEATS_PER_COMMAND,
        source_bytes,
        source_sha256: hex::encode(source_hash.finalize()),
        elapsed_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn verify_lean_source(
    source: &str,
    module_index: usize,
    project: &Path,
    lake: &Path,
) -> Result<()> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let source_path = env::temp_dir().join(format!(
        "burn-dragon-ruliad-lean-{}-{nonce}-{module_index}.lean",
        std::process::id()
    ));
    fs::write(&source_path, source).with_context(|| {
        format!(
            "write generated Lean verifier input {}",
            source_path.display()
        )
    })?;

    let output = Command::new(lake)
        .args(["env", "lean"])
        .arg(&source_path)
        .current_dir(project)
        .output();
    let _ = fs::remove_file(&source_path);
    let output = output.with_context(|| {
        format!(
            "failed to launch generated Lean verification module {module_index} in {}",
            project.display()
        )
    })?;
    if !output.status.success() {
        return Err(anyhow!(
            "generated Lean verification module {module_index} failed with status {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            bounded_output(&output.stdout),
            bounded_output(&output.stderr)
        ));
    }
    Ok(())
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn verify_formal_bundles_with_lean(
    _bundles: &[RuliadProofBundle],
    project: &Path,
    _lake: &Path,
) -> Result<RuliadLeanVerificationReport> {
    Err(anyhow!(
        "generated Lean verification is unavailable for wasm target ({})",
        project.display()
    ))
}

fn adversarial_certificates(
    bundle: &RuliadProofBundle,
) -> Vec<(&'static str, RuliadProofCertificate)> {
    let mut cases = Vec::with_capacity(3);

    let mut missing_goal = bundle.certificate.clone();
    if missing_goal.goals.pop().is_some() {
        cases.push(("missing_goal", missing_goal));
    }

    let mut unknown_source = bundle.certificate.clone();
    let mut missing_id = "__lean_external_missing_axiom__".to_string();
    while bundle
        .problem
        .axioms
        .iter()
        .any(|axiom| axiom.id == missing_id)
    {
        missing_id.push('_');
    }
    if let Some(step) = first_step_mut(&mut unknown_source) {
        step.source = RuliadProofSource::Axiom { id: missing_id };
        cases.push(("unknown_source", unknown_source));
    }

    let mut invalid_path = bundle.certificate.clone();
    let invalid_index = maximum_term_arity_problem(&bundle.problem);
    if let Some(step) = first_step_mut(&mut invalid_path) {
        step.path = vec![invalid_index];
        cases.push(("invalid_path", invalid_path));
    }

    cases
}

fn first_step_mut(certificate: &mut RuliadProofCertificate) -> Option<&mut RuliadProofStep> {
    certificate
        .goals
        .iter_mut()
        .find_map(|goal| goal.steps.first_mut())
}

fn maximum_term_arity_problem(problem: &RuliadProofProblem) -> usize {
    let axiom_max = problem
        .axioms
        .iter()
        .flat_map(|axiom| [&axiom.lhs, &axiom.rhs])
        .map(maximum_term_arity)
        .max()
        .unwrap_or(0);
    let goal_max = problem
        .goals
        .iter()
        .flat_map(|goal| [&goal.claim.lhs, &goal.claim.rhs])
        .map(maximum_term_arity)
        .max()
        .unwrap_or(0);
    axiom_max.max(goal_max)
}

fn maximum_term_arity(term: &RuliadTerm) -> usize {
    match term {
        RuliadTerm::Variable { .. } | RuliadTerm::Atom { .. } => 0,
        RuliadTerm::Apply { arguments, .. } => arguments
            .iter()
            .map(maximum_term_arity)
            .max()
            .unwrap_or(0)
            .max(arguments.len()),
    }
}

fn render_problem(problem: &RuliadProofProblem) -> String {
    let axioms = render_list(&problem.axioms, |axiom| {
        format!(
            "{{ id := {}, lhs := {}, rhs := {} }}",
            lean_string(&axiom.id),
            render_term(&axiom.lhs),
            render_term(&axiom.rhs)
        )
    });
    let goals = render_list(&problem.goals, |goal| {
        format!(
            "{{ dependencies := {}, claim := {{ lhs := {}, rhs := {} }} }}",
            render_nat_list(&goal.dependencies),
            render_term(&goal.claim.lhs),
            render_term(&goal.claim.rhs)
        )
    });
    format!(
        "{{ version := {}, axioms := {axioms}, goals := {goals}, root := {} }}",
        problem.version, problem.root
    )
}

fn render_certificate(certificate: &RuliadProofCertificate) -> String {
    let goals = render_list(&certificate.goals, |goal| {
        format!(
            "{{ goal := {}, steps := {} }}",
            goal.goal,
            render_list(&goal.steps, render_step)
        )
    });
    format!("{{ version := {}, goals := {goals} }}", certificate.version)
}

fn render_step(step: &RuliadProofStep) -> String {
    let source = match &step.source {
        RuliadProofSource::Axiom { id } => format!(".namedAxiom {}", lean_string(id)),
        RuliadProofSource::Lemma { goal } => format!(".priorGoal {goal}"),
    };
    let direction = match step.direction {
        RuliadRewriteDirection::Forward => ".forward",
        RuliadRewriteDirection::Reverse => ".reverse",
    };
    format!(
        "{{ source := {source}, path := {}, direction := {direction} }}",
        render_nat_list(&step.path)
    )
}

fn render_term(term: &RuliadTerm) -> String {
    match term {
        RuliadTerm::Variable { index } => format!(".variable {index}"),
        RuliadTerm::Atom { symbol } => format!(".atom {}", lean_string(symbol)),
        RuliadTerm::Apply {
            operator,
            arguments,
        } => format!(
            ".apply {} {}",
            lean_string(operator),
            render_list(arguments, render_term)
        ),
    }
}

fn render_nat_list(values: &[usize]) -> String {
    render_list(values, usize::to_string)
}

fn render_list<T>(values: &[T], render: impl FnMut(&T) -> String) -> String {
    format!(
        "[{}]",
        values.iter().map(render).collect::<Vec<_>>().join(", ")
    )
}

fn lean_string(value: &str) -> String {
    serde_json::to_string(value).expect("serializing a Rust string cannot fail")
}

#[cfg(not(target_arch = "wasm32"))]
fn bounded_output(bytes: &[u8]) -> String {
    const MAX_BYTES: usize = 16 * 1024;
    String::from_utf8_lossy(&bytes[..bytes.len().min(MAX_BYTES)]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ruliad::formal::{
        RuliadFormalGenerationSplit, RuliadFormalGeneratorConfig, generate_formal_bundle,
    };
    use crate::ruliad::ir::RuliadFormalDomain;

    fn bundles() -> Vec<RuliadProofBundle> {
        RuliadFormalDomain::ALL
            .into_iter()
            .enumerate()
            .map(|(index, domain)| {
                generate_formal_bundle(
                    1_337 + index as u64,
                    RuliadFormalGeneratorConfig {
                        domain: Some(domain),
                        rewrite_depth: 3,
                        leaf_count: 3,
                        context_depth: 2,
                        distractor_axioms: 1,
                        generation_split: RuliadFormalGenerationSplit::StructuralValidationV1,
                    },
                )
                .expect("formal bundle")
            })
            .collect()
    }

    #[test]
    fn exporter_covers_positive_and_adversarial_contracts() {
        let bundles = bundles();
        let (source, negative_cases) = render_lean_verification_module(&bundles).expect("render");
        assert_eq!(negative_cases, bundles.len() * 3);
        assert_eq!(source.matches("= true := by").count(), bundles.len());
        assert_eq!(source.matches("= false := by").count(), negative_cases);
        assert!(source.contains(".priorGoal"));
        assert!(source.contains(".namedAxiom"));
    }

    #[test]
    fn formal_panel_is_deterministic_canonical_and_domain_complete() {
        let (left, levels) = formal_verification_panel(1_337, &[4, 0, 4], 2).expect("panel");
        let (right, right_levels) =
            formal_verification_panel(1_337, &[0, 4], 2).expect("canonical panel");
        assert_eq!(levels, vec![0, 4]);
        assert_eq!(right_levels, levels);
        assert_eq!(left.len(), 2 * RuliadFormalDomain::ALL.len() * 2);
        let (left_source, left_negatives) =
            render_lean_verification_module(&left).expect("left source");
        let (right_source, right_negatives) =
            render_lean_verification_module(&right).expect("right source");
        assert_eq!(left_source, right_source);
        assert_eq!(left_negatives, left.len() * 3);
        assert_eq!(right_negatives, left_negatives);
    }

    #[test]
    fn formal_panel_rejects_empty_or_zero_sized_contracts() {
        assert!(formal_verification_panel(1_337, &[], 1).is_err());
        assert!(formal_verification_panel(1_337, &[0], 0).is_err());
    }

    #[test]
    #[ignore = "requires an installed Lean toolchain"]
    fn generated_certificates_replay_in_independent_lean_checker() {
        let project = Path::new(env!("CARGO_MANIFEST_DIR")).join("lean/ruliad_seed");
        let report = verify_formal_panel_with_lean(1_337, &[0, 4], 1, &project, Path::new("lake"))
            .expect("Lean panel verification");
        assert_eq!(report.difficulty_levels, vec![0, 4]);
        assert_eq!(report.formal_domains.len(), RuliadFormalDomain::ALL.len());
        assert_eq!(
            report.verification.formal_samples_checked,
            2 * RuliadFormalDomain::ALL.len()
        );
        assert_eq!(
            report.verification.negative_cases_checked,
            2 * RuliadFormalDomain::ALL.len() * 3
        );
    }
}
