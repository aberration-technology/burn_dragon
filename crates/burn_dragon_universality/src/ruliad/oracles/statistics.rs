//! Sample complexity statistics and degeneracy detection.

use super::*;

pub(super) fn sample_stats(spec: &RuliadSampleSpec, text: &str) -> SampleStats {
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
        RuliadSampleSpec::FormalProof {
            problem,
            certificate,
            ..
        } => {
            let complexity = complexity_vector(problem, Some(certificate));
            (
                complexity.syntax_nodes.max(1),
                complexity.proof_step_count.max(1),
                complexity
                    .proof_goal_count
                    .saturating_add(complexity.axiom_count)
                    .max(1),
                1.0,
            )
        }
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

pub(super) fn semantic_difficulty_score(
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
        RuliadSampleSpec::FormalProof {
            problem,
            certificate,
            ..
        } => {
            let complexity = complexity_vector(problem, Some(certificate));
            (
                complexity.syntax_nodes,
                complexity
                    .proof_step_count
                    .saturating_add(complexity.dependency_depth),
                complexity
                    .axiom_count
                    .saturating_add(complexity.dependency_width),
                10,
            )
        }
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
        | RuliadSampleSpec::FormalProof { .. }
        | RuliadSampleSpec::LeanTask { .. }
        | RuliadSampleSpec::HashNoise { .. } => false,
    }
}

pub(super) fn trace_transition_rate(trace: &[String]) -> f32 {
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

pub(super) fn string_trace_change_rate(trace: &[String]) -> f32 {
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

pub(super) fn finite_state_transition_rate(trace: &[usize]) -> f32 {
    if trace.len() <= 1 {
        return 0.0;
    }
    let changed = trace.windows(2).filter(|pair| pair[0] != pair[1]).count();
    changed as f32 / trace.len().saturating_sub(1) as f32
}
