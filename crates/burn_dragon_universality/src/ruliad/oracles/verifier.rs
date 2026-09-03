//! Deterministic oracle verification for every Ruliad sample family.

use super::*;

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
        RuliadSampleSpec::FormalProof {
            problem,
            certificate,
            candidate,
            proof_step_index,
            action_presentation_rotation,
            task,
            ..
        } => {
            let oracle = replay_certificate(problem, certificate, RuliadKernelLimits::default());
            let task_shape_ok = match task {
                RuliadTaskKind::ConstructProof => {
                    candidate.is_none()
                        && proof_step_index.is_none()
                        && action_presentation_rotation.is_none()
                }
                RuliadTaskKind::AdvanceProof => {
                    candidate.is_none()
                        && action_presentation_rotation.is_none()
                        && proof_step_index
                            .is_some_and(|index| certificate.step_at(index).is_some())
                }
                RuliadTaskKind::SelectProofAction => {
                    candidate.is_none()
                        && proof_step_index.is_some_and(|index| {
                            crate::ruliad::policy::oracle_proof_action_set(
                                problem,
                                certificate,
                                index,
                                crate::ruliad::policy::DEFAULT_PROOF_ACTION_CANDIDATES,
                            )
                            .is_ok_and(|actions| {
                                action_presentation_rotation
                                    .is_none_or(|rotation| rotation < actions.candidates.len())
                            })
                        })
                }
                RuliadTaskKind::CheckProof => {
                    candidate.is_some()
                        && proof_step_index.is_none()
                        && action_presentation_rotation.is_none()
                }
                _ => false,
            };
            (
                oracle.accepted && task_shape_ok,
                RuliadFamilyKind::FormalProof,
                *task,
            )
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
