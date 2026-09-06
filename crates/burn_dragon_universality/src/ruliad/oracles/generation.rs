//! Proof-task loading, sample selection, and categorical presentations.

use super::*;

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
        RuliadFamilyKind::FormalProof => {
            let task = choose_formal_task(&config.source_selection.formal_task_mix, &mut rng);
            generate_formal_spec(
                &family_config,
                task,
                config
                    .source_selection
                    .formal_task_mix
                    .proof_action_answer_contract,
                formal_generation_split(config.formal_generalization, split),
                &mut rng,
            )
        }
        RuliadFamilyKind::LeanTask => generate_lean_spec(proof_tasks, &mut rng),
        RuliadFamilyKind::HashNoise => generate_hash_noise_spec(&mut rng),
    }?;
    finalize_generated_spec(spec)
}

pub(super) fn choose_formal_task(
    mix: &RuliadFormalTaskMixConfig,
    rng: &mut SplitMix64,
) -> RuliadTaskKind {
    let weights = [
        (RuliadTaskKind::AdvanceProof, mix.advance_proof_weight),
        (
            RuliadTaskKind::SelectProofAction,
            mix.select_proof_action_weight,
        ),
        (RuliadTaskKind::ConstructProof, mix.construct_proof_weight),
        (RuliadTaskKind::CheckProof, mix.check_proof_weight),
    ];
    let total = weights
        .iter()
        .map(|(_, weight)| *weight)
        .sum::<usize>()
        .max(1);
    let mut draw = rng.next_usize(total);
    for (task, weight) in weights {
        if draw < weight {
            return task;
        }
        draw = draw.saturating_sub(weight);
    }
    RuliadTaskKind::ConstructProof
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
        RuliadFamilyKind::FormalProof => generate_formal_spec(
            &bucket.family_config,
            bucket.id.task_kind,
            config
                .source_selection
                .formal_task_mix
                .proof_action_answer_contract,
            formal_generation_split(config.formal_generalization, split),
            &mut rng,
        ),
        RuliadFamilyKind::LeanTask => generate_lean_spec(proof_tasks, &mut rng),
        RuliadFamilyKind::HashNoise => generate_hash_noise_spec(&mut rng),
    }?;
    finalize_generated_spec(spec)
}

pub(super) fn formal_generation_split(
    contract: RuliadFormalGeneralizationContract,
    split: SampleSplit,
) -> RuliadFormalGenerationSplit {
    match (contract, split) {
        (RuliadFormalGeneralizationContract::SeedDisjointV1, _) => {
            RuliadFormalGenerationSplit::Shared
        }
        (RuliadFormalGeneralizationContract::StructuralHoldoutV1, SampleSplit::Train) => {
            RuliadFormalGenerationSplit::StructuralTrainV1
        }
        (RuliadFormalGeneralizationContract::StructuralTrainSeedDisjointV1, _) => {
            RuliadFormalGenerationSplit::StructuralTrainV1
        }
        (RuliadFormalGeneralizationContract::StructuralHoldoutV1, SampleSplit::Validation) => {
            RuliadFormalGenerationSplit::StructuralValidationV1
        }
    }
}

pub(super) fn finalize_generated_spec(spec: RuliadSampleSpec) -> Result<GeneratedRuliadSample> {
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
            answer: trace
                .last()
                .map(|value| symbolic_word_certificate("target", value, "01"))
                .unwrap_or_else(|| symbolic_word_certificate("target", "", "01")),
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
            alphabet,
            steps,
            normal_form,
            ..
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
            answer: symbolic_word_certificate("normal_form", normal_form, alphabet),
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
                answer: format!(
                    "holds={holds};composed={composed};path={}",
                    compact_usize_list(path)
                ),
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
        RuliadSampleSpec::FormalProof {
            problem,
            certificate,
            candidate,
            proof_step_index,
            action_presentation_rotation,
            task,
            ..
        } => {
            let candidate_report = candidate.as_ref().map(|candidate| {
                replay_certificate(problem, candidate, RuliadKernelLimits::default())
            });
            RuliadCategoricalPresentation {
                abstraction: "verified_derivation_category".to_string(),
                source_family: RuliadFamilyKind::FormalProof.label().to_string(),
                task_kind: task.label().to_string(),
                presentation: format!("{}_proof_dag", problem.domain.label()),
                objects: problem.goals.iter().map(|goal| goal.id.clone()).collect(),
                morphisms: problem
                    .axioms
                    .iter()
                    .map(|axiom| axiom.id.clone())
                    .chain(
                        problem
                            .goals
                            .iter()
                            .enumerate()
                            .map(|(index, _)| format!("lemma_{index}")),
                    )
                    .collect(),
                functors: vec!["portable_replay_kernel".to_string()],
                laws: vec![
                    "substitution_preserves_equality".to_string(),
                    "verified_derivations_compose".to_string(),
                    "dependency_order_is_acyclic".to_string(),
                ],
                query: match task {
                    RuliadTaskKind::ConstructProof => {
                        "construct a certificate whose replay closes the root obligation"
                            .to_string()
                    }
                    RuliadTaskKind::AdvanceProof => format!(
                        "advance verifier-backed proof transition {} toward the root obligation",
                        proof_step_index.unwrap_or_default()
                    ),
                    RuliadTaskKind::SelectProofAction => format!(
                        "select verifier-backed proof action {} toward the root obligation under cyclic presentation {}",
                        proof_step_index.unwrap_or_default(),
                        action_presentation_rotation.unwrap_or_default()
                    ),
                    RuliadTaskKind::CheckProof => {
                        "replay the proposed certificate and localize its first failure".to_string()
                    }
                    _ => "verify a formal Ruliad proof".to_string(),
                },
                answer: if *task == RuliadTaskKind::AdvanceProof {
                    "one replayable proof-DAG edge".to_string()
                } else if *task == RuliadTaskKind::SelectProofAction {
                    "one selected verifier-backed proof action".to_string()
                } else if let Some(report) = candidate_report {
                    formal_check_answer(&report)
                } else {
                    format!(
                        "certificate={};root={}",
                        compact_text(&certificate.problem_hash, 16),
                        problem.root
                    )
                },
                categorical_core: true,
            }
        }
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

pub(super) fn sample_rng(
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
