//! Compact proof documents, prompts, answers, and symbolic rendering.

use super::*;

pub fn sample_text(spec: &RuliadSampleSpec, oracle_hash: &str) -> String {
    match spec {
        RuliadSampleSpec::FormalProof { .. } => formal_proof_document(spec, oracle_hash)
            .unwrap_or_else(|error| format!("[R3 invalid]\nE:{error}\n[/R3]\n")),
        _ => proof_tape_document(spec, oracle_hash).to_text(),
    }
}

pub const RULIAD_V2_DOCUMENT_CLOSE_MARKER: &str = "[/R2]";
pub const RULIAD_V3_DOCUMENT_CLOSE_MARKER: &str = "[/R3]";

/// Return the document terminator belonging to the sample's wire format.
///
/// R3 formal proofs intentionally use a different textual envelope from the
/// compact R2 proof tapes. Keeping this decision next to serialization avoids
/// completion, evaluation, and stop-token paths drifting from the generator.
pub fn ruliad_document_close_marker(spec: &RuliadSampleSpec) -> &'static str {
    if matches!(spec, RuliadSampleSpec::FormalProof { .. }) {
        RULIAD_V3_DOCUMENT_CLOSE_MARKER
    } else {
        RULIAD_V2_DOCUMENT_CLOSE_MARKER
    }
}

pub fn ruliad_expected_answer(spec: &RuliadSampleSpec) -> String {
    compact_answer(spec)
}

pub fn ruliad_answer_contract(spec: &RuliadSampleSpec) -> String {
    match spec {
        RuliadSampleSpec::FormalProof {
            task: RuliadTaskKind::ConstructProof,
            ..
        } => "certificate".to_string(),
        RuliadSampleSpec::FormalProof {
            task: RuliadTaskKind::AdvanceProof,
            ..
        } => "proof_step".to_string(),
        RuliadSampleSpec::FormalProof {
            task: RuliadTaskKind::SelectProofAction,
            action_answer_contract,
            ..
        } => match action_answer_contract {
            RuliadProofActionAnswerContract::PresentationIndex => "action_index".to_string(),
            RuliadProofActionAnswerContract::SemanticStep => "proof_action_step".to_string(),
        },
        _ => compact_answer_keys(&compact_answer(spec)),
    }
}

pub fn ruliad_answer_values(spec: &RuliadSampleSpec) -> String {
    compact_answer_values(&compact_answer(spec))
}

pub fn ruliad_prompt_prefix(spec: &RuliadSampleSpec, oracle_hash: &str) -> String {
    let text = sample_text(spec, oracle_hash);
    if let Some(answer_offset) = text.find("\n!:") {
        text[..answer_offset + 3].to_string()
    } else {
        text
    }
}

pub(super) fn formal_proof_document(spec: &RuliadSampleSpec, oracle_hash: &str) -> Result<String> {
    let RuliadSampleSpec::FormalProof {
        problem,
        certificate,
        candidate,
        proof_step_index,
        action_presentation_rotation,
        action_candidate_count,
        action_answer_contract,
        task,
    } = spec
    else {
        return Err(anyhow!("formal document requires a formal proof spec"));
    };
    let problem_wire = encode_problem(problem)?;
    let (query, answer) = match task {
        RuliadTaskKind::ConstructProof => (
            format!("?:root={}", problem.root),
            encode_model_certificate(certificate)?,
        ),
        RuliadTaskKind::AdvanceProof => formal_advance_query(
            problem,
            certificate,
            proof_step_index
                .ok_or_else(|| anyhow!("advance-proof document requires proof_step_index"))?,
        )?,
        RuliadTaskKind::SelectProofAction => formal_select_action_query(
            problem,
            certificate,
            proof_step_index
                .ok_or_else(|| anyhow!("proof-action document requires proof_step_index"))?,
            *action_presentation_rotation,
            *action_candidate_count,
            *action_answer_contract,
        )?,
        RuliadTaskKind::CheckProof => {
            let candidate = candidate
                .as_ref()
                .ok_or_else(|| anyhow!("check-proof document requires candidate"))?;
            (
                format!("?:root={}", problem.root),
                formal_check_answer(&replay_certificate(
                    problem,
                    candidate,
                    RuliadKernelLimits::default(),
                )),
            )
        }
        _ => return Err(anyhow!("unsupported formal proof task {task:?}")),
    };
    let candidate_line = candidate
        .as_ref()
        .map(encode_certificate)
        .transpose()?
        .map(|payload| format!("C:{payload}\n"))
        .unwrap_or_default();
    Ok(format!(
        "[R3 {} {}/{}]\nP:{}\n{}{}\n!:{}\n[/R3]\n",
        compact_text(oracle_hash, 16),
        problem.domain.label(),
        task.label(),
        problem_wire,
        candidate_line,
        query,
        answer
    ))
}

pub(super) fn formal_advance_query(
    problem: &RuliadProofProblem,
    certificate: &RuliadProofCertificate,
    step_index: usize,
) -> Result<(String, String)> {
    let prefix = certificate
        .prefix_before(step_index)
        .ok_or_else(|| anyhow!("advance-proof step index {step_index} is out of bounds"))?;
    let next = certificate
        .single_step_at(step_index)
        .ok_or_else(|| anyhow!("advance-proof step index {step_index} is out of bounds"))?;
    let next_goal = next
        .goals
        .first()
        .ok_or_else(|| anyhow!("advance-proof transition has no goal"))?;
    let local_prefix = prefix
        .goals
        .iter()
        .find(|candidate| candidate.goal == next_goal.goal)
        .map(|candidate| candidate.steps.as_slice())
        .unwrap_or_default();
    let current = replay_goal_prefix(
        problem,
        next_goal.goal,
        local_prefix,
        RuliadKernelLimits::default(),
    )
    .map_err(|failure| anyhow!("invalid advance-proof prefix: {}", failure.message))?;
    let step = next_goal
        .steps
        .first()
        .ok_or_else(|| anyhow!("advance-proof transition has no step"))?;
    let mut advanced_prefix = local_prefix.to_vec();
    advanced_prefix.push(step.clone());
    let advanced = replay_goal_prefix(
        problem,
        next_goal.goal,
        &advanced_prefix,
        RuliadKernelLimits::default(),
    )
    .map_err(|failure| anyhow!("invalid advance-proof step: {}", failure.message))?;
    current
        .at_path(&step.path)
        .ok_or_else(|| anyhow!("advance-proof rewrite path is absent before the step"))?;
    advanced
        .at_path(&step.path)
        .ok_or_else(|| anyhow!("advance-proof rewrite path is absent after the step"))?;
    let source = match &step.source {
        RuliadProofSource::Axiom { id } => format!("a:{id}"),
        RuliadProofSource::Lemma { goal } => format!("l:{goal}"),
    };
    let (source_lhs, source_rhs) = formal_proof_source_equality(problem, &step.source)?;
    let (before_pattern, after_pattern) = match step.direction {
        RuliadRewriteDirection::Forward => (source_lhs, source_rhs),
        RuliadRewriteDirection::Reverse => (source_rhs, source_lhs),
    };
    let (before_focus, after_focus) = transition_pattern_focus(before_pattern, after_pattern);
    let path = if step.path.is_empty() {
        "-".to_string()
    } else {
        step.path
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(".")
    };
    let query = format!(
        "?:advance;g={};p={};cur={};dst={};src={}",
        next_goal.goal,
        path,
        bounded_transition_pattern(before_focus),
        bounded_transition_pattern(after_focus),
        source,
    );
    Ok((query, encode_model_certificate(&next)?))
}

pub(super) fn formal_select_action_query(
    problem: &RuliadProofProblem,
    certificate: &RuliadProofCertificate,
    step_index: usize,
    presentation_rotation: Option<usize>,
    candidate_count: Option<usize>,
    answer_contract: RuliadProofActionAnswerContract,
) -> Result<(String, String)> {
    let actions = crate::ruliad::policy::oracle_proof_action_set(
        problem,
        certificate,
        step_index,
        candidate_count
            .unwrap_or(crate::ruliad::policy::DEFAULT_PROOF_ACTION_CANDIDATES)
            .max(2),
    )?;
    let actions = actions
        .rotate_left(presentation_rotation.unwrap_or_default() % actions.candidates.len().max(1))?;
    let answer = crate::ruliad::policy::proof_action_answer(
        &actions,
        actions.selected_index,
        answer_contract,
    )?;
    Ok((ruliad_proof_action_query(problem, &actions)?, answer))
}

pub fn ruliad_proof_action_query(
    problem: &RuliadProofProblem,
    actions: &crate::ruliad::policy::RuliadProofActionSet,
) -> Result<String> {
    let candidates = actions
        .candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            let source = match &candidate.step.source {
                RuliadProofSource::Axiom { id } => format!("a:{id}"),
                RuliadProofSource::Lemma { goal } => format!("l:{goal}"),
            };
            let direction = match candidate.step.direction {
                RuliadRewriteDirection::Forward => "f",
                RuliadRewriteDirection::Reverse => "r",
            };
            let path = if candidate.step.path.is_empty() {
                "-".to_string()
            } else {
                candidate
                    .step
                    .path
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(".")
            };
            let (lhs, rhs) = formal_proof_source_equality(problem, &candidate.step.source)?;
            let (before, after) = match candidate.step.direction {
                RuliadRewriteDirection::Forward => (lhs, rhs),
                RuliadRewriteDirection::Reverse => (rhs, lhs),
            };
            let (before, after) = transition_pattern_focus(before, after);
            Ok(format!(
                "c{index}={source}|{direction}|{path}|{}>{}",
                bounded_transition_pattern(before),
                bounded_transition_pattern(after)
            ))
        })
        .collect::<Result<Vec<_>>>()?
        .join(",");
    let (difference_path, current_focus, target_focus) =
        first_state_difference(&actions.current, &actions.target);
    let difference_path = if difference_path.is_empty() {
        "-".to_string()
    } else {
        difference_path
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(".")
    };
    Ok(format!(
        "?:select;g={};cur={};{};dst={};at={}",
        actions.goal,
        bounded_policy_state(current_focus),
        candidates,
        bounded_policy_state(target_focus),
        difference_path
    ))
}

pub fn ruliad_proof_action_prompt(
    problem: &RuliadProofProblem,
    actions: &crate::ruliad::policy::RuliadProofActionSet,
) -> Result<String> {
    let problem_hash = problem.canonical_hash()?;
    Ok(format!(
        "[R3 {} {}/select_proof_action]\nP:{}\n{}\n!:",
        compact_text(&problem_hash, 16),
        problem.domain.label(),
        encode_problem(problem)?,
        ruliad_proof_action_query(problem, actions)?,
    ))
}

/// Minimal verifier-sufficient interface for proof-action policy learning.
///
/// The action menu already contains the local rewrite patterns, current and
/// target focus, goal id, and difference path needed to choose a transition.
/// Omitting the serialized global problem and random content hash prevents a
/// fixed context window from silently retaining an arbitrary suffix of those
/// fields and removes a high-entropy memorization channel from the policy.
pub fn ruliad_proof_action_local_prompt(
    problem: &RuliadProofProblem,
    actions: &crate::ruliad::policy::RuliadProofActionSet,
) -> Result<String> {
    Ok(format!(
        "{}\n!:",
        ruliad_proof_action_query(problem, actions)?
    ))
}

pub(super) fn first_state_difference<'a>(
    current: &'a RuliadTerm,
    target: &'a RuliadTerm,
) -> (Vec<usize>, &'a RuliadTerm, &'a RuliadTerm) {
    fn recurse<'a>(
        current: &'a RuliadTerm,
        target: &'a RuliadTerm,
        path: &mut Vec<usize>,
    ) -> (&'a RuliadTerm, &'a RuliadTerm) {
        let (
            RuliadTerm::Apply {
                operator: current_operator,
                arguments: current_arguments,
            },
            RuliadTerm::Apply {
                operator: target_operator,
                arguments: target_arguments,
            },
        ) = (current, target)
        else {
            return (current, target);
        };
        if current_operator != target_operator || current_arguments.len() != target_arguments.len()
        {
            return (current, target);
        }
        let Some((index, (current, target))) = current_arguments
            .iter()
            .zip(target_arguments)
            .enumerate()
            .find(|(_, (current, target))| current != target)
        else {
            return (current, target);
        };
        path.push(index);
        recurse(current, target, path)
    }

    let mut path = Vec::new();
    let (current, target) = recurse(current, target, &mut path);
    (path, current, target)
}

pub(super) fn bounded_policy_state(term: &RuliadTerm) -> String {
    let canonical = term.canonical_text();
    if canonical.chars().count() <= 96 {
        return canonical;
    }
    let detailed = render_transition_pattern(term, 0, 4, 4, 12);
    if detailed.chars().count() <= 96 {
        detailed
    } else {
        render_transition_pattern(term, 0, 3, 3, 8)
    }
}

pub(super) fn formal_proof_source_equality<'a>(
    problem: &'a RuliadProofProblem,
    source: &RuliadProofSource,
) -> Result<(&'a RuliadTerm, &'a RuliadTerm)> {
    match source {
        RuliadProofSource::Axiom { id } => problem
            .axioms
            .iter()
            .find(|axiom| axiom.id == *id)
            .map(|axiom| (&axiom.lhs, &axiom.rhs)),
        RuliadProofSource::Lemma { goal } => problem
            .goals
            .get(*goal)
            .map(|goal| (&goal.claim.lhs, &goal.claim.rhs)),
    }
    .ok_or_else(|| anyhow!("advance-proof transition references an unknown source"))
}

pub(super) fn bounded_transition_pattern(term: &RuliadTerm) -> String {
    let detailed = render_transition_pattern(term, 0, 2, 2, 12);
    if detailed.chars().count() <= 24 {
        return detailed;
    }
    let shallow = render_transition_pattern(term, 0, 1, 2, 8);
    if shallow.chars().count() <= 24 {
        return shallow;
    }
    match term {
        RuliadTerm::Variable { index } => format!("?{index}"),
        RuliadTerm::Atom { symbol } => bounded_transition_symbol(symbol, 24),
        RuliadTerm::Apply { operator, .. } => {
            format!("{}(_)", bounded_transition_symbol(operator, 20))
        }
    }
}

pub(super) fn transition_pattern_focus<'a>(
    before: &'a RuliadTerm,
    after: &'a RuliadTerm,
) -> (&'a RuliadTerm, &'a RuliadTerm) {
    let (
        RuliadTerm::Apply {
            operator: before_operator,
            arguments: before_arguments,
        },
        RuliadTerm::Apply {
            operator: after_operator,
            arguments: after_arguments,
        },
    ) = (before, after)
    else {
        return (before, after);
    };
    if before_operator != after_operator || before_arguments.len() != after_arguments.len() {
        return (before, after);
    }
    let mut differences = before_arguments
        .iter()
        .zip(after_arguments)
        .filter(|(left, right)| left != right);
    let Some((different_before, different_after)) = differences.next() else {
        return (before, after);
    };
    if differences.next().is_some() {
        return (before, after);
    }
    transition_pattern_focus(different_before, different_after)
}

pub(super) fn render_transition_pattern(
    term: &RuliadTerm,
    depth: usize,
    max_depth: usize,
    max_arguments: usize,
    max_symbol_chars: usize,
) -> String {
    match term {
        RuliadTerm::Variable { index } => format!("?{index}"),
        RuliadTerm::Atom { symbol } => bounded_transition_symbol(symbol, max_symbol_chars),
        RuliadTerm::Apply {
            operator,
            arguments,
        } if depth < max_depth => {
            let mut rendered = arguments
                .iter()
                .take(max_arguments)
                .map(|argument| {
                    render_transition_pattern(
                        argument,
                        depth.saturating_add(1),
                        max_depth,
                        max_arguments,
                        max_symbol_chars,
                    )
                })
                .collect::<Vec<_>>();
            if arguments.len() > rendered.len() {
                rendered.push("_".to_string());
            }
            format!(
                "{}({})",
                bounded_transition_symbol(operator, max_symbol_chars),
                rendered.join(",")
            )
        }
        RuliadTerm::Apply { .. } => "_".to_string(),
    }
}

pub(super) fn bounded_transition_symbol(symbol: &str, max_chars: usize) -> String {
    if symbol.chars().count() <= max_chars {
        symbol.to_string()
    } else {
        let prefix = symbol
            .chars()
            .take(max_chars.saturating_sub(1))
            .collect::<String>();
        format!("{prefix}~")
    }
}

pub(super) fn formal_check_answer(report: &crate::ruliad::kernel::RuliadReplayReport) -> String {
    let kind = report
        .failure
        .as_ref()
        .map(|failure| failure.kind.label())
        .unwrap_or("none");
    let failed_goal = report
        .failure
        .as_ref()
        .and_then(|failure| failure.goal)
        .map(|goal| goal.to_string())
        .unwrap_or_else(|| "none".to_string());
    let failed_step = report
        .failure
        .as_ref()
        .and_then(|failure| failure.step)
        .map(|step| step.to_string())
        .unwrap_or_else(|| "none".to_string());
    format!(
        "ok={};vg={};vs={};g={failed_goal};s={failed_step};k={kind}",
        bit(report.accepted),
        report.verified_goals,
        report.verified_steps
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RuliadProofTapeDocument {
    source_family: String,
    task_kind: String,
    presentation: String,
    domains: Vec<String>,
    reasoning_modes: Vec<String>,
    verifier_version: u32,
    oracle_hash: String,
    query: String,
    answer_contract: String,
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
            "[R2 {hash} v{} {}/{}/{}]\nS:{domains}|{modes}\nG:{}\n?:{}\nA:{}\n",
            self.verifier_version,
            compact_ruliad_label(&self.source_family),
            compact_ruliad_label(&self.task_kind),
            compact_ruliad_label(&self.presentation),
            self.data
                .iter()
                .map(|item| compact_text(item, 96))
                .collect::<Vec<_>>()
                .join("|"),
            self.query,
            self.answer_contract
        );
        for step in compact_proof_step_runs(&self.proof_steps, 32) {
            out.push_str(&format!(">{}\n", compact_text(&step, 96)));
        }
        out.push_str(&format!("!:{}\n[/R2]\n", self.answer));
        out
    }
}

pub(super) fn compact_proof_step_runs(steps: &[String], max_steps: usize) -> Vec<String> {
    let mut compacted = Vec::with_capacity(steps.len().min(max_steps.max(1)));
    let mut index = 0usize;
    while index < steps.len() {
        let step = &steps[index];
        let mut run_len = 1usize;
        while steps
            .get(index + run_len)
            .is_some_and(|candidate| candidate == step)
        {
            run_len = run_len.saturating_add(1);
        }
        if run_len >= 3 {
            compacted.push(format!("{step} *{run_len}"));
        } else {
            for _ in 0..run_len {
                compacted.push(step.clone());
            }
        }
        index = index.saturating_add(run_len);
    }
    if compacted.len() <= max_steps {
        return compacted;
    }
    let head = max_steps.saturating_div(2).max(1);
    let tail = max_steps.saturating_sub(head).saturating_sub(1).max(1);
    let omitted = compacted.len().saturating_sub(head + tail);
    let mut bounded = Vec::with_capacity(max_steps);
    bounded.extend(compacted.iter().take(head).cloned());
    bounded.push(format!("omit={omitted}"));
    bounded.extend(
        compacted
            .iter()
            .skip(compacted.len().saturating_sub(tail))
            .cloned(),
    );
    bounded
}

pub(super) fn proof_tape_document(
    spec: &RuliadSampleSpec,
    oracle_hash: &str,
) -> RuliadProofTapeDocument {
    let view = ruliad_categorical_presentation(spec);
    let family = family_of_spec(spec);
    let task_kind = task_kind_of_spec(spec);
    let semantics = ruliad_source_semantics(family, task_kind);
    let answer = compact_answer(spec);
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
        answer_contract: compact_answer_keys(&answer),
        proof_steps: compact_proof_steps(spec),
        answer,
        data: compact_data(spec),
    }
}

pub(super) fn compact_query(spec: &RuliadSampleSpec) -> String {
    match spec {
        RuliadSampleSpec::Eca { steps, .. } => format!("eca^{steps}"),
        RuliadSampleSpec::Simulation {
            source_rule,
            target_rule,
            steps,
            ..
        } => format!("Fcomp:{source_rule}->{target_rule};n={steps}"),
        RuliadSampleSpec::Automaton { .. } => "act(w)".to_string(),
        RuliadSampleSpec::Rewrite { steps, .. } => format!("nf(x0)<={steps}"),
        RuliadSampleSpec::Algebra { law, operands, .. } => {
            format!(
                "{}({})",
                compact_ruliad_label(law.label()),
                compact_usize_list(operands)
            )
        }
        RuliadSampleSpec::Category { task, .. } => match task {
            RuliadTaskKind::ComposeCategoryPath => "cp".to_string(),
            RuliadTaskKind::VerifyCategoryLaw => "ca".to_string(),
            RuliadTaskKind::VerifyFunctorPreservation => "cf".to_string(),
            RuliadTaskKind::VerifyNaturalitySquare => "cn".to_string(),
            _ => "ct".to_string(),
        },
        RuliadSampleSpec::ProofTree { modulus, .. } => {
            format!("ss:Z/{modulus}Z")
        }
        RuliadSampleSpec::FormalProof { problem, task, .. } => {
            format!(
                "{}:root{}:{}",
                problem.domain.label(),
                problem.root,
                task.label()
            )
        }
        RuliadSampleSpec::LeanTask { task_id, .. } => format!("ln:{task_id}"),
        RuliadSampleSpec::HashNoise { .. } => "sha:canary".to_string(),
    }
}

pub(super) fn compact_proof_steps(spec: &RuliadSampleSpec) -> Vec<String> {
    match spec {
        RuliadSampleSpec::Eca {
            initial,
            trace,
            steps,
            ..
        } => {
            let mut steps_out = vec![format!("x0={}", compact_symbolic_word(initial, 48))];
            if trace.len() > 2 {
                let mid = trace.len() / 2;
                steps_out.push(format!("x{mid}={}", compact_symbolic_word(&trace[mid], 48)));
            }
            steps_out.push(format!(
                "x{steps}={}",
                trace
                    .last()
                    .map(|value| compact_symbolic_word(value, 48))
                    .unwrap_or_default()
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
            format!("tr={}", compact_state_trace(trace)),
        ],
        RuliadSampleSpec::Rewrite {
            initial,
            trace,
            normal_form,
            ..
        } => vec![
            format!(
                "{}=>{};n={}",
                compact_symbolic_word(initial, 48),
                compact_symbolic_word(normal_form, 32),
                trace.len() - 1
            ),
            format!("tr={}", compact_string_trace(trace)),
        ],
        RuliadSampleSpec::Algebra { law, lhs, rhs, .. } => {
            vec![format!(
                "{} l={lhs};r={rhs}",
                compact_ruliad_label(law.label())
            )]
        }
        RuliadSampleSpec::Category { proof_steps, .. } => proof_steps.clone(),
        RuliadSampleSpec::ProofTree {
            lemmas,
            proof_steps,
            ..
        } => {
            let mut steps = vec![format!("D=L0..L{};d=ces", lemmas.len().saturating_sub(1))];
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
        RuliadSampleSpec::FormalProof { .. } => Vec::new(),
        RuliadSampleSpec::LeanTask { .. } => vec!["h=1".to_string()],
        RuliadSampleSpec::HashNoise { .. } => vec!["h=1".to_string()],
    }
}

pub(super) fn compact_answer(spec: &RuliadSampleSpec) -> String {
    match spec {
        RuliadSampleSpec::Eca { trace, .. } => trace
            .last()
            .map(|value| symbolic_word_certificate("x", value, "01"))
            .unwrap_or_else(|| symbolic_word_certificate("x", "", "01")),
        RuliadSampleSpec::Simulation { .. } => "ok=1".to_string(),
        RuliadSampleSpec::Automaton { accepted, .. } => format!("acc={}", bit(*accepted)),
        RuliadSampleSpec::Rewrite {
            alphabet,
            normal_form,
            ..
        } => symbolic_word_certificate("nf", normal_form, alphabet),
        RuliadSampleSpec::Algebra { holds, .. } => format!("ok={}", bit(*holds)),
        RuliadSampleSpec::Category {
            lhs, rhs, holds, ..
        } => format!("ok={};l={lhs};r={rhs}", bit(*holds)),
        RuliadSampleSpec::ProofTree {
            holds, lhs, rhs, ..
        } => {
            format!("ok={};l={lhs};r={rhs}", bit(*holds))
        }
        RuliadSampleSpec::FormalProof {
            problem,
            certificate,
            candidate,
            proof_step_index,
            action_presentation_rotation,
            action_candidate_count,
            action_answer_contract,
            task,
        } => match task {
            RuliadTaskKind::ConstructProof => encode_model_certificate(certificate)
                .unwrap_or_else(|error| format!("invalid_certificate={error}")),
            RuliadTaskKind::AdvanceProof => proof_step_index
                .and_then(|index| certificate.single_step_at(index))
                .and_then(|next| encode_model_certificate(&next).ok())
                .unwrap_or_else(|| "invalid_transition".to_string()),
            RuliadTaskKind::SelectProofAction => proof_step_index
                .and_then(|index| {
                    crate::ruliad::policy::oracle_proof_action_set(
                        problem,
                        certificate,
                        index,
                        action_candidate_count
                            .unwrap_or(crate::ruliad::policy::DEFAULT_PROOF_ACTION_CANDIDATES)
                            .max(2),
                    )
                    .ok()
                })
                .and_then(|actions| {
                    actions
                        .rotate_left(
                            action_presentation_rotation.unwrap_or_default()
                                % actions.candidates.len().max(1),
                        )
                        .ok()
                })
                .and_then(|actions| {
                    crate::ruliad::policy::proof_action_answer(
                        &actions,
                        actions.selected_index,
                        *action_answer_contract,
                    )
                    .ok()
                })
                .unwrap_or_else(|| "invalid_action_set".to_string()),
            RuliadTaskKind::CheckProof => candidate
                .as_ref()
                .map(|candidate| {
                    formal_check_answer(&replay_certificate(
                        problem,
                        candidate,
                        RuliadKernelLimits::default(),
                    ))
                })
                .unwrap_or_else(|| "invalid_candidate".to_string()),
            _ => "invalid_formal_task".to_string(),
        },
        RuliadSampleSpec::LeanTask { payload_hash, .. } => format!("sha={payload_hash}"),
        RuliadSampleSpec::HashNoise { payload_hash, .. } => format!("sha={payload_hash}"),
    }
}

pub(super) fn compact_answer_keys(answer: &str) -> String {
    let keys = answer
        .split(';')
        .filter_map(|part| {
            let (key, _value) = part.split_once('=')?;
            let key = key.trim();
            (!key.is_empty()).then_some(compact_ruliad_label(key))
        })
        .collect::<Vec<_>>();
    if keys.is_empty() {
        "value".to_string()
    } else {
        keys.join(",")
    }
}

pub(super) fn compact_answer_values(answer: &str) -> String {
    let values = answer
        .split(';')
        .filter_map(|part| {
            let (_key, value) = part.split_once('=')?;
            let value = value.trim();
            (!value.is_empty()).then_some(value.to_string())
        })
        .collect::<Vec<_>>();
    if values.is_empty() {
        answer.trim().to_string()
    } else {
        values.join(";")
    }
}

pub(super) fn symbolic_word_certificate(prefix: &str, value: &str, alphabet: &str) -> String {
    let len = value.chars().count();
    let alphabet = if alphabet.is_empty() { "_" } else { alphabet };
    let counts = alphabet
        .chars()
        .map(|symbol| {
            value
                .chars()
                .filter(|candidate| *candidate == symbol)
                .count()
                .to_string()
        })
        .collect::<Vec<_>>()
        .join(",");
    let edge = match (value.chars().next(), value.chars().last()) {
        (Some(first), Some(last)) => format!("{first}{last}"),
        _ => "_".to_string(),
    };
    format!(
        "{prefix}len={len};{prefix}alpha={alphabet};{prefix}counts={counts};{prefix}edge={edge}"
    )
}

pub(super) fn compact_data(spec: &RuliadSampleSpec) -> Vec<String> {
    match spec {
        RuliadSampleSpec::Eca {
            rule,
            width,
            steps,
            initial,
            ..
        } => vec![
            format!("r={rule};w={width};n={steps}"),
            format!("x0={}", compact_symbolic_word(initial, 64)),
        ],
        RuliadSampleSpec::Simulation {
            source_rule,
            target_rule,
            width,
            steps,
            source_initial,
            ..
        } => vec![
            format!("rs={source_rule}->{target_rule};w={width};n={steps}"),
            format!(
                "x0={};F=complement",
                compact_symbolic_word(source_initial, 64)
            ),
        ],
        RuliadSampleSpec::Automaton {
            state_count,
            transitions,
            start_state,
            accept_states,
            input,
            ..
        } => vec![
            format!(
                "q={state_count};s={start_state};a={}",
                compact_usize_list(accept_states)
            ),
            format!(
                "w={};d={}",
                compact_symbolic_word(input, 64),
                compact_transition_table(transitions)
            ),
        ],
        RuliadSampleSpec::Rewrite {
            alphabet, rules, ..
        } => vec![
            format!("A={alphabet}"),
            format!(
                "R={}",
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
            format!("C={carrier_size};xs={}", compact_usize_list(operands)),
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
                format!("O={object_count};I={}", compact_usize_list(identities)),
                format!("P={};C={composed}", compact_usize_list(path)),
                format!("A={}", compact_morphism_summary(morphisms)),
            ];
            if let Some(functor) = functor {
                data.push(format!(
                    "{}:o={}",
                    functor.name,
                    compact_usize_list(&functor.object_map)
                ));
            }
            if let Some(naturality) = naturality {
                data.push(format!(
                    "N:f={};l={};r={}",
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
        RuliadSampleSpec::FormalProof { problem, .. } => {
            vec![encode_problem(problem).unwrap_or_else(|error| format!("invalid_problem={error}"))]
        }
        RuliadSampleSpec::LeanTask {
            task_id,
            statement,
            proof,
            ..
        } => vec![
            format!("id={task_id}"),
            format!(
                "s=len{};h{}",
                statement.len(),
                compact_text(&sha256_hex(statement.as_bytes()), 16)
            ),
            format!(
                "p=len{};h{}",
                proof.len(),
                compact_text(&sha256_hex(proof.as_bytes()), 16)
            ),
        ],
        RuliadSampleSpec::HashNoise { bytes_hex, .. } => {
            vec![format!("bytes={}", compact_text(bytes_hex, 64))]
        }
    }
}

pub(super) fn bit(value: bool) -> u8 {
    u8::from(value)
}

pub(super) fn compact_text(value: &str, max_len: usize) -> String {
    let value = bound_repeated_chars(value, 6);
    if value.chars().count() <= max_len {
        value
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

pub(super) fn compact_symbolic_word(value: &str, max_len: usize) -> String {
    let alphabet = symbolic_alphabet(value);
    let len = value.chars().count();
    if len > 16 && alphabet.len() <= 4 {
        if alphabet.iter().all(|ch| matches!(ch, '0' | '1')) {
            return compact_binary_word(value, max_len);
        }
        return compact_low_alphabet_word(value, &alphabet, max_len);
    }
    let value = bound_repeated_chars(value, 6);
    if value.chars().count() <= max_len {
        return value;
    }
    if alphabet.iter().all(|ch| matches!(ch, '0' | '1')) {
        return compact_binary_word(&value, max_len);
    }
    if alphabet.len() <= 4 {
        return compact_low_alphabet_word(&value, &alphabet, max_len);
    }
    compact_text(&value, max_len)
}

pub(super) fn symbolic_alphabet(value: &str) -> Vec<char> {
    let mut alphabet = Vec::new();
    for ch in value.chars() {
        if !alphabet.contains(&ch) {
            alphabet.push(ch);
        }
        if alphabet.len() > 4 {
            break;
        }
    }
    alphabet
}

pub(super) fn compact_binary_word(value: &str, max_len: usize) -> String {
    let hash = sha256_hex(value.as_bytes())
        .chars()
        .take(16)
        .collect::<String>();
    let ones = value.bytes().filter(|byte| *byte == b'1').count();
    compact_text(&format!("b{}:h{hash}:w{ones}", value.len()), max_len)
}

pub(super) fn compact_low_alphabet_word(value: &str, alphabet: &[char], max_len: usize) -> String {
    let hash = sha256_hex(value.as_bytes())
        .chars()
        .take(16)
        .collect::<String>();
    let alphabet = alphabet.iter().collect::<String>();
    let counts = alphabet
        .chars()
        .map(|symbol| {
            value
                .chars()
                .filter(|candidate| *candidate == symbol)
                .count()
                .to_string()
        })
        .collect::<Vec<_>>()
        .join(",");
    compact_text(
        &format!("s{}:{alphabet}:h{hash}:c{counts}", value.chars().count()),
        max_len,
    )
}

pub(super) fn bound_repeated_chars(value: &str, max_run: usize) -> String {
    if max_run == 0 {
        return String::new();
    }
    let mut out = String::with_capacity(value.len());
    let mut chars = value.chars().peekable();
    while let Some(ch) = chars.next() {
        let mut run_len = 1usize;
        while chars.peek().is_some_and(|next| *next == ch) {
            chars.next();
            run_len = run_len.saturating_add(1);
        }
        if run_len <= max_run {
            for _ in 0..run_len {
                out.push(ch);
            }
        } else {
            for _ in 0..max_run {
                out.push(ch);
            }
            out.push('^');
            out.push_str(&run_len.to_string());
        }
    }
    out
}

pub(super) fn compact_labels(values: &[String]) -> String {
    values
        .iter()
        .map(|value| compact_ruliad_label(value))
        .collect::<Vec<_>>()
        .join(",")
}

pub fn compact_ruliad_label(value: &str) -> &str {
    match value {
        "discrete_dynamics" => "dd",
        "computation_theory" => "ct",
        "symbolic_rewriting" => "sr",
        "universal_algebra" => "ua",
        "category_theory" => "cg",
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
        "associativity" => "as",
        "commutativity" => "cm",
        "formal_deduction" => "fd",
        "entropy_canary" => "ec",
        "eca" => "E",
        "simulation" => "S",
        "automaton" => "M",
        "rewrite" => "R",
        "algebra" => "A",
        "category" => "C",
        "proof_tree" => "P",
        "lean_task" => "L",
        "hash_noise" => "H",
        "next_state" => "ns",
        "multi_step_state" => "ms",
        "verify_simulation" => "vsim",
        "evaluate_automaton" => "aut",
        "rewrite_normal_form" => "rw",
        "check_algebra_law" => "alg",
        "compose_category_path" => "cpath",
        "verify_category_law" => "claw",
        "verify_functor_preservation" => "fun",
        "verify_naturality_square" => "nat",
        "prove_theorem" => "thm",
        "complete_proof" => "lp",
        "hash_canary" => "hash",
        "trajectory_category" => "tc",
        "commuting_trajectory_functor" => "tf",
        "free_monoid_action_category" => "ma",
        "rewrite_path_category" => "rp",
        "verified_theorem_dependency_category" => "td",
        "proof_category" => "pc",
        "one_object_category_law_probe" => "ol",
        "finite_category_law" => "fl",
        "finite_category_path" => "cp",
        "finite_functor_preservation" => "ff",
        "finite_naturality_square" => "fn",
        other => other,
    }
}

pub(super) fn compact_usize_list(values: &[usize]) -> String {
    if values.len() > 16 {
        return compact_long_usize_list(values);
    }
    let mut parts = Vec::new();
    let mut index = 0usize;
    while index < values.len() {
        let value = values[index];
        let mut run_len = 1usize;
        while values
            .get(index + run_len)
            .is_some_and(|next| *next == value)
        {
            run_len = run_len.saturating_add(1);
        }
        if run_len >= 4 {
            parts.push(format!("{value}*{run_len}"));
        } else {
            for _ in 0..run_len {
                parts.push(value.to_string());
            }
        }
        index = index.saturating_add(run_len);
    }
    parts.join(",")
}

pub(super) fn compact_long_usize_list(values: &[usize]) -> String {
    let hash = stable_json_hash(&values)
        .unwrap_or_else(|_| "unknown".to_string())
        .chars()
        .take(16)
        .collect::<String>();
    let first = values.first().copied().unwrap_or_default();
    let last = values.last().copied().unwrap_or_default();
    let max = values.iter().copied().max().unwrap_or_default();
    let checksum = values.iter().fold(0usize, |acc, value| {
        acc.wrapping_mul(131).wrapping_add(*value)
    });
    format!(
        "u{}:h{}:f{}:z{}:m{}:c{:x}",
        values.len(),
        hash,
        first,
        last,
        max,
        checksum & 0xffff
    )
}

pub(super) fn compact_transition_table(transitions: &[Vec<usize>]) -> String {
    let hash = stable_json_hash(&transitions)
        .unwrap_or_else(|_| "unknown".to_string())
        .chars()
        .take(16)
        .collect::<String>();
    let first = transitions.first();
    let last = transitions.last();
    let first_0 = first
        .and_then(|row| row.first())
        .copied()
        .unwrap_or_default();
    let first_1 = first
        .and_then(|row| row.get(1))
        .copied()
        .unwrap_or_default();
    let last_0 = last
        .and_then(|row| row.first())
        .copied()
        .unwrap_or_default();
    let last_1 = last.and_then(|row| row.get(1)).copied().unwrap_or_default();
    format!(
        "t{}:h{}:f{}-{}:z{}-{}",
        transitions.len(),
        hash,
        first_0,
        first_1,
        last_0,
        last_1
    )
}

pub(super) fn compact_state_trace(trace: &[usize]) -> String {
    if trace.len() <= 4 {
        return compact_usize_list(trace);
    }
    let hash = stable_json_hash(&trace)
        .unwrap_or_else(|_| "unknown".to_string())
        .chars()
        .take(16)
        .collect::<String>();
    let mid = trace.len() / 2;
    format!(
        "q{}:h{}:f{}:s{}:m{}:p{}:z{}",
        trace.len(),
        hash,
        trace[0],
        trace[1],
        trace[mid],
        trace[trace.len() - 2],
        trace[trace.len() - 1]
    )
}

pub(super) fn compact_string_trace(trace: &[String]) -> String {
    if trace.len() <= 5 {
        return trace
            .iter()
            .map(|value| compact_symbolic_word(value, 48))
            .collect::<Vec<_>>()
            .join(">");
    }
    let mid = trace.len() / 2;
    format!(
        "{}>{}>{}..>{}",
        compact_symbolic_word(&trace[0], 40),
        compact_symbolic_word(&trace[1], 40),
        compact_symbolic_word(&trace[mid], 40),
        trace
            .last()
            .map(|value| compact_symbolic_word(value, 40))
            .unwrap_or_default()
    )
}

pub(super) fn compact_table(table: &[Vec<usize>]) -> String {
    table
        .iter()
        .map(|row| compact_usize_list(row))
        .collect::<Vec<_>>()
        .join("/")
}

pub(super) fn compact_operation_descriptor(table: &[Vec<usize>]) -> String {
    let carrier_size = table.len();
    if table == add_mod_table(carrier_size) {
        return format!("add{carrier_size}");
    }
    if table == affine_mod_table(carrier_size, 1, 2, 1) {
        return format!("aff{carrier_size}(x+2y+1)");
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

pub(super) fn compact_morphism_summary(morphisms: &[RuliadCategoryMorphism]) -> String {
    let first = morphisms
        .first()
        .map(|morphism| morphism.name.as_str())
        .unwrap_or("-");
    let last = morphisms
        .last()
        .map(|morphism| morphism.name.as_str())
        .unwrap_or("-");
    format!("n={};to;f={first};z={last}", morphisms.len())
}
