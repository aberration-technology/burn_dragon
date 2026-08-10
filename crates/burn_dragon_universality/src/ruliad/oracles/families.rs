//! Family-specific sample generation and algebraic helpers.

use super::*;

pub(super) fn generate_eca_spec(
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

pub(super) fn generate_simulation_spec(
    family: &RuliadFamilyConfig,
    rng: &mut SplitMix64,
) -> Result<RuliadSampleSpec> {
    let mut fallback = None;
    for _ in 0..128 {
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
        let spec = RuliadSampleSpec::Simulation {
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
        };
        if !is_degenerate_spec(&spec) {
            return Ok(spec);
        }
        fallback = Some(spec);
    }
    fallback.ok_or_else(|| anyhow!("failed to generate simulation ruliad sample"))
}

pub(super) fn generate_automaton_spec(
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

pub(super) fn generate_rewrite_spec(
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

pub(super) fn generate_algebra_spec(
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

pub(super) fn generate_category_spec(
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

pub(super) fn generate_category_spec_for_task(
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

pub(super) fn generate_proof_tree_spec(
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
        format!("L0:dot=x0*y0+x1*y1 mod {modulus}"),
        format!("L1:n=x0*x0+x1*x1 mod {modulus}"),
        "L2:n(x+y)=n(x)+n(y)+2dot".to_string(),
        "L3:dot0=>cross0".to_string(),
    ];
    for lemma_index in 4..depth {
        lemmas.push(format!("L{lemma_index}:subst"));
    }
    let proof_steps = vec![
        format!("u=({},{});v=({},{});m={modulus}", u[0], u[1], v[0], v[1]),
        format!("dot={}*{}+{}*{}={dot}", u[0], v[0], u[1], v[1]),
        format!("sum=({},{});n(sum)={norm_sum}", sum[0], sum[1]),
        format!("n(u)+n(v)={norm_u}+{norm_v}={rhs}"),
        "close:L2,dot0".to_string(),
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

pub(super) fn generate_formal_spec(
    family: &RuliadFamilyConfig,
    task: RuliadTaskKind,
    action_answer_contract: RuliadProofActionAnswerContract,
    generation_split: RuliadFormalGenerationSplit,
    rng: &mut SplitMix64,
) -> Result<RuliadSampleSpec> {
    if !matches!(
        task,
        RuliadTaskKind::ConstructProof
            | RuliadTaskKind::AdvanceProof
            | RuliadTaskKind::SelectProofAction
            | RuliadTaskKind::CheckProof
    ) {
        return Err(anyhow!(
            "formal proof family does not support task {task:?}"
        ));
    }
    let leaf_count = range_or(family.width, 2, 4, rng).max(2);
    let rewrite_depth = range_or(family.steps, 2, 4, rng).max(2);
    let context_depth = 1usize.saturating_add(rewrite_depth.ilog2() as usize);
    let bundle = generate_formal_bundle(
        rng.next_u64(),
        RuliadFormalGeneratorConfig {
            domain: None,
            rewrite_depth,
            leaf_count,
            context_depth,
            distractor_axioms: rewrite_depth.div_ceil(2),
            generation_split,
        },
    )?;
    let candidate = if task == RuliadTaskKind::CheckProof {
        Some(if rng.next_bool() {
            bundle.certificate.clone()
        } else {
            corrupt_formal_certificate(&bundle.certificate)?
        })
    } else {
        None
    };
    let proof_step_index = if matches!(
        task,
        RuliadTaskKind::AdvanceProof | RuliadTaskKind::SelectProofAction
    ) {
        let step_count = bundle.certificate.step_count();
        if step_count == 0 {
            return Err(anyhow!("advance-proof oracle certificate has no steps"));
        }
        Some(rng.next_usize(step_count))
    } else {
        None
    };
    let action_presentation_rotation = (task == RuliadTaskKind::SelectProofAction)
        .then(|| rng.next_usize(crate::ruliad::policy::DEFAULT_PROOF_ACTION_CANDIDATES));
    Ok(RuliadSampleSpec::FormalProof {
        problem: bundle.problem,
        certificate: bundle.certificate,
        candidate,
        proof_step_index,
        action_presentation_rotation,
        action_answer_contract,
        task,
    })
}

pub(super) fn generate_lean_spec(
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

pub(super) fn generate_hash_noise_spec(rng: &mut SplitMix64) -> Result<RuliadSampleSpec> {
    let bytes = (0..32).map(|_| rng.next_u8()).collect::<Vec<_>>();
    Ok(RuliadSampleSpec::HashNoise {
        bytes_hex: hex::encode(&bytes),
        payload_hash: sha256_hex(&bytes),
        task: RuliadTaskKind::HashCanary,
    })
}

pub(super) fn parse_trace(trace: &[String]) -> Vec<Vec<u8>> {
    trace.iter().map(|state| eca::parse_state(state)).collect()
}

pub(super) fn valid_transition_table(
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

pub(super) fn automaton_trace(
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

pub(super) fn valid_alphabet(alphabet: &str) -> bool {
    let mut seen = std::collections::BTreeSet::new();
    !alphabet.is_empty()
        && alphabet.is_ascii()
        && alphabet
            .chars()
            .all(|symbol| !symbol.is_whitespace() && seen.insert(symbol))
}

pub(super) fn alphabet_contains(alphabet: &str, value: &str) -> bool {
    value
        .chars()
        .all(|symbol| alphabet.chars().any(|candidate| candidate == symbol))
}

pub(super) fn rewrite_trace(
    initial: &str,
    rules: &[RuliadRewriteRule],
    steps: usize,
) -> Vec<String> {
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

pub(super) fn apply_rewrite_once(value: &str, rules: &[RuliadRewriteRule]) -> Option<String> {
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

pub(super) fn valid_operation_table(carrier_size: usize, operation_table: &[Vec<usize>]) -> bool {
    carrier_size > 0
        && operation_table.len() == carrier_size
        && operation_table
            .iter()
            .all(|row| row.len() == carrier_size && row.iter().all(|value| *value < carrier_size))
}

pub(super) fn add_mod_table(carrier_size: usize) -> Vec<Vec<usize>> {
    (0..carrier_size)
        .map(|left| {
            (0..carrier_size)
                .map(|right| (left + right) % carrier_size)
                .collect::<Vec<_>>()
        })
        .collect()
}

pub(super) fn affine_mod_table(
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

pub(super) fn algebra_law_result(
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

pub(super) fn mod_dot(left: [usize; 2], right: [usize; 2], modulus: usize) -> usize {
    if modulus == 0 {
        return 0;
    }
    (left[0] * right[0] + left[1] * right[1]) % modulus
}

pub(super) fn mod_norm(value: [usize; 2], modulus: usize) -> usize {
    if modulus == 0 {
        return 0;
    }
    (value[0] * value[0] + value[1] * value[1]) % modulus
}

pub(super) fn next_prime_at_least(value: usize) -> usize {
    let mut candidate = value.max(2);
    loop {
        if is_prime(candidate) {
            return candidate;
        }
        candidate = candidate.saturating_add(1);
    }
}

pub(super) fn is_prime(value: usize) -> bool {
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

pub(super) fn shuffle_rules(rules: &mut [RuliadRewriteRule], rng: &mut SplitMix64) {
    for index in (1..rules.len()).rev() {
        let swap_index = rng.next_usize(index + 1);
        rules.swap(index, swap_index);
    }
}

pub(super) fn range_or(
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
