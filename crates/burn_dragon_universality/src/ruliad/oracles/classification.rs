//! Family classification, reasoning metadata, and difficulty scaling.

use super::*;

pub(super) fn family_of_spec(spec: &RuliadSampleSpec) -> RuliadFamilyKind {
    match spec {
        RuliadSampleSpec::Eca { .. } => RuliadFamilyKind::Eca,
        RuliadSampleSpec::Simulation { .. } => RuliadFamilyKind::Simulation,
        RuliadSampleSpec::Automaton { .. } => RuliadFamilyKind::Automaton,
        RuliadSampleSpec::Rewrite { .. } => RuliadFamilyKind::Rewrite,
        RuliadSampleSpec::Algebra { .. } => RuliadFamilyKind::Algebra,
        RuliadSampleSpec::Category { .. } => RuliadFamilyKind::Category,
        RuliadSampleSpec::ProofTree { .. } => RuliadFamilyKind::ProofTree,
        RuliadSampleSpec::FormalProof { .. } => RuliadFamilyKind::FormalProof,
        RuliadSampleSpec::LeanTask { .. } => RuliadFamilyKind::LeanTask,
        RuliadSampleSpec::HashNoise { .. } => RuliadFamilyKind::HashNoise,
    }
}

pub(super) fn task_kind_of_spec(spec: &RuliadSampleSpec) -> RuliadTaskKind {
    match spec {
        RuliadSampleSpec::Eca { task, .. }
        | RuliadSampleSpec::Simulation { task, .. }
        | RuliadSampleSpec::Automaton { task, .. }
        | RuliadSampleSpec::Rewrite { task, .. }
        | RuliadSampleSpec::Algebra { task, .. }
        | RuliadSampleSpec::Category { task, .. }
        | RuliadSampleSpec::ProofTree { task, .. }
        | RuliadSampleSpec::FormalProof { task, .. }
        | RuliadSampleSpec::LeanTask { task, .. }
        | RuliadSampleSpec::HashNoise { task, .. } => *task,
    }
}

pub fn ruliad_sample_math_domains(spec: &RuliadSampleSpec) -> Vec<RuliadMathDomain> {
    if let RuliadSampleSpec::FormalProof { problem, .. } = spec {
        let domain = match problem.domain {
            RuliadFormalDomain::Equational => vec![
                RuliadMathDomain::SymbolicRewriting,
                RuliadMathDomain::UniversalAlgebra,
            ],
            RuliadFormalDomain::Category => vec![RuliadMathDomain::CategoryTheory],
            RuliadFormalDomain::Logic => vec![RuliadMathDomain::Logic],
            RuliadFormalDomain::Automata => vec![RuliadMathDomain::ComputationTheory],
            RuliadFormalDomain::Process => vec![RuliadMathDomain::ProcessCalculus],
            RuliadFormalDomain::Metagraph => vec![RuliadMathDomain::MetagraphRewriting],
        };
        return domain
            .into_iter()
            .chain(std::iter::once(RuliadMathDomain::FormalProof))
            .collect();
    }
    let semantics = ruliad_source_semantics(family_of_spec(spec), task_kind_of_spec(spec));
    semantics.math_domains.to_vec()
}

pub fn ruliad_sample_reasoning_modes(spec: &RuliadSampleSpec) -> Vec<RuliadReasoningMode> {
    ruliad_source_semantics(family_of_spec(spec), task_kind_of_spec(spec))
        .reasoning_modes
        .to_vec()
}

pub(super) fn choose_family<'a>(
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
    let level = if family.kind == RuliadFamilyKind::FormalProof {
        difficulty_level.saturating_add(1).ilog2().saturating_add(1) as usize
    } else {
        difficulty_level.min(4096)
    };
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

pub(super) fn cap_scaled_family_for_payload(family: &mut RuliadFamilyConfig) {
    let (max_width, max_steps) = match family.kind {
        RuliadFamilyKind::Eca => (Some(128), Some(64)),
        RuliadFamilyKind::Simulation => (Some(128), Some(64)),
        RuliadFamilyKind::Automaton => (Some(48), Some(128)),
        RuliadFamilyKind::Rewrite => (Some(128), Some(96)),
        RuliadFamilyKind::Algebra => (Some(64), None),
        RuliadFamilyKind::Category => (Some(48), Some(64)),
        RuliadFamilyKind::ProofTree => (Some(4096), Some(4096)),
        RuliadFamilyKind::FormalProof => (None, None),
        RuliadFamilyKind::LeanTask | RuliadFamilyKind::HashNoise => (None, None),
    };
    if let (Some(width), Some(max_width)) = (family.width.as_mut(), max_width) {
        cap_range(width, max_width);
    }
    if let (Some(steps), Some(max_steps)) = (family.steps.as_mut(), max_steps) {
        cap_range(steps, max_steps);
    }
}

pub(super) fn cap_range(range: &mut crate::config::UsizeRangeConfig, max_value: usize) {
    range.min = range.min.min(max_value);
    range.max = range.max.min(max_value).max(range.min);
}
