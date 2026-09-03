use super::*;
use crate::config::UsizeRangeConfig;
use crate::ruliad::config::{
    RULIAD_REQUIRED_MATH_DOMAINS, RULIAD_REQUIRED_REASONING_MODES, RuliadSerializationConfig,
    RuliadTokenizationConfig, default_ruliad_families, formal_ruliad_families,
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
        formal_generalization: Default::default(),
        source_selection: crate::ruliad::config::RuliadSourceSelectionConfig::default(),
        families: default_ruliad_families(),
        proof_tasks: None,
        lean_task_limit: None,
    }
}

fn longest_repeated_char_run(text: &str) -> usize {
    let mut longest = 0usize;
    let mut previous = None;
    let mut current = 0usize;
    for ch in text.chars() {
        if Some(ch) == previous {
            current = current.saturating_add(1);
        } else {
            longest = longest.max(current);
            previous = Some(ch);
            current = 1;
        }
    }
    longest.max(current)
}

fn longest_periodic_motif_run(text: &str, motif_len: usize) -> usize {
    let bytes = text.as_bytes();
    if motif_len == 0 || bytes.len() < motif_len * 2 {
        return 0;
    }
    let mut longest = 0usize;
    for start in 0..=bytes.len().saturating_sub(motif_len * 2) {
        let motif = &bytes[start..start + motif_len];
        let mut run = motif_len;
        let mut offset = start + motif_len;
        while offset + motif_len <= bytes.len() && &bytes[offset..offset + motif_len] == motif {
            run = run.saturating_add(motif_len);
            offset = offset.saturating_add(motif_len);
        }
        longest = longest.max(run);
    }
    longest
}

#[test]
fn generated_samples_verify() {
    for index in 0..16 {
        let sample = generate_sample(&config(), &[], SampleSplit::Train, 0, index).expect("sample");
        let report = verify_spec(&sample.spec).expect("verify");
        assert!(report.ok);
        assert_eq!(report.oracle_hash, sample.oracle_hash);
    }
}

#[test]
fn advance_proof_sample_exposes_local_state_and_one_replayable_edge() {
    let mut config = config();
    config.families = formal_ruliad_families();
    config.source_selection.formal_task_mix = RuliadFormalTaskMixConfig {
        advance_proof_weight: 1,
        select_proof_action_weight: 0,
        construct_proof_weight: 0,
        check_proof_weight: 0,
        proof_action_answer_contract: RuliadProofActionAnswerContract::default(),
    };
    let sample = generate_sample(&config, &[], SampleSplit::Train, 0, 7).expect("advance sample");
    assert_eq!(sample.task_kind, RuliadTaskKind::AdvanceProof);
    assert!(sample.text.contains("?:advance;g="), "{}", sample.text);
    assert!(sample.text.contains(";p="), "{}", sample.text);
    assert!(sample.text.contains(";cur="), "{}", sample.text);
    assert!(sample.text.contains(";dst="), "{}", sample.text);
    let query = sample
        .text
        .lines()
        .find(|line| line.starts_with("?:advance;"))
        .expect("advance query");
    assert!(query.chars().count() <= 120, "{query}");
    assert!(!query.contains(".."), "{query}");
    assert_eq!(query.matches('(').count(), query.matches(')').count());
    assert_eq!(ruliad_answer_contract(&sample.spec), "proof_step");

    let RuliadSampleSpec::FormalProof {
        certificate,
        proof_step_index: Some(step_index),
        ..
    } = &sample.spec
    else {
        panic!("expected advance formal proof spec");
    };
    let expected = certificate
        .single_step_at(*step_index)
        .and_then(|next| encode_model_certificate(&next).ok())
        .expect("single replayable edge");
    assert_eq!(ruliad_expected_answer(&sample.spec), expected);
    assert!(verify_spec(&sample.spec).expect("verify").ok);
}

#[test]
fn proof_action_sample_hides_selection_behind_a_verifier_backed_menu() {
    let mut config = config();
    config.families = formal_ruliad_families();
    config.source_selection.formal_task_mix = RuliadFormalTaskMixConfig {
        advance_proof_weight: 0,
        select_proof_action_weight: 1,
        construct_proof_weight: 0,
        check_proof_weight: 0,
        proof_action_answer_contract: RuliadProofActionAnswerContract::default(),
    };
    let sample =
        generate_sample(&config, &[], SampleSplit::Train, 0, 11).expect("proof action sample");
    assert_eq!(sample.task_kind, RuliadTaskKind::SelectProofAction);
    let query = sample
        .text
        .lines()
        .find(|line| line.starts_with("?:select;"))
        .expect("action query");
    assert!(query.contains(";cur="), "{query}");
    assert!(query.contains(";dst="), "{query}");
    assert!(query.contains(";at="), "{query}");
    assert!(query.contains(";c0="), "{query}");
    assert!(query.contains(",c1="), "{query}");
    let current = query
        .split(";cur=")
        .nth(1)
        .and_then(|suffix| suffix.split(";c0=").next())
        .expect("current focus");
    let target = query
        .split(";dst=")
        .nth(1)
        .and_then(|suffix| suffix.split(";at=").next())
        .expect("target focus");
    assert_ne!(
        current, target,
        "action state must expose a live obligation"
    );
    assert_eq!(ruliad_answer_contract(&sample.spec), "action_index");
    let RuliadSampleSpec::FormalProof {
        problem,
        certificate,
        proof_step_index: Some(step_index),
        action_presentation_rotation: Some(rotation),
        ..
    } = &sample.spec
    else {
        panic!("expected presented proof-action spec");
    };
    let actions = crate::ruliad::policy::oracle_proof_action_set(
        problem,
        certificate,
        *step_index,
        crate::ruliad::policy::DEFAULT_PROOF_ACTION_CANDIDATES,
    )
    .and_then(|actions| actions.rotate_left(*rotation))
    .expect("presented actions");
    assert_eq!(
        ruliad_expected_answer(&sample.spec),
        format!("c={}", actions.selected_index)
    );
    assert!(verify_spec(&sample.spec).expect("verify").ok);
}

#[test]
fn semantic_proof_action_sample_emits_a_rotation_invariant_executable_step() {
    let mut config = config();
    config.families = formal_ruliad_families();
    config.source_selection.formal_task_mix = RuliadFormalTaskMixConfig {
        advance_proof_weight: 0,
        select_proof_action_weight: 1,
        construct_proof_weight: 0,
        check_proof_weight: 0,
        proof_action_answer_contract: RuliadProofActionAnswerContract::SemanticStep,
    };
    let sample = generate_sample(&config, &[], SampleSplit::Train, 0, 11).expect("semantic action");
    assert_eq!(ruliad_answer_contract(&sample.spec), "proof_action_step");
    assert!(sample.text.contains("!:g"), "{}", sample.text);
    assert!(!sample.text.contains("!:c="), "{}", sample.text);
    let RuliadSampleSpec::FormalProof {
        problem,
        certificate,
        proof_step_index: Some(step_index),
        action_presentation_rotation: Some(rotation),
        ..
    } = &sample.spec
    else {
        panic!("expected semantic proof-action spec");
    };
    let actions = crate::ruliad::policy::oracle_proof_action_set(
        problem,
        certificate,
        *step_index,
        crate::ruliad::policy::DEFAULT_PROOF_ACTION_CANDIDATES,
    )
    .and_then(|actions| actions.rotate_left(*rotation))
    .expect("presented semantic actions");
    let expected = crate::ruliad::policy::proof_action_answer(
        &actions,
        actions.selected_index,
        RuliadProofActionAnswerContract::SemanticStep,
    )
    .expect("semantic answer");
    assert_eq!(ruliad_expected_answer(&sample.spec), expected);
    let rerotated = actions.rotate_left(1).expect("rerotated actions");
    assert_eq!(
        crate::ruliad::policy::proof_action_answer(
            &rerotated,
            rerotated.selected_index,
            RuliadProofActionAnswerContract::SemanticStep,
        )
        .expect("rerotated semantic answer"),
        expected
    );
    assert!(verify_spec(&sample.spec).expect("verify").ok);
}

#[test]
fn proof_action_primary_stream_spans_every_cyclic_presentation() {
    let mut config = config();
    config.families = formal_ruliad_families();
    config.source_selection.formal_task_mix = RuliadFormalTaskMixConfig {
        advance_proof_weight: 0,
        select_proof_action_weight: 1,
        construct_proof_weight: 0,
        check_proof_weight: 0,
        proof_action_answer_contract: RuliadProofActionAnswerContract::default(),
    };
    let mut counts = [0usize; crate::ruliad::policy::DEFAULT_PROOF_ACTION_CANDIDATES];
    for sample_index in 0..128 {
        let sample = generate_sample(&config, &[], SampleSplit::Train, 0, sample_index)
            .expect("proof action sample");
        let RuliadSampleSpec::FormalProof {
            action_presentation_rotation: Some(rotation),
            ..
        } = sample.spec
        else {
            panic!("expected presented proof-action spec");
        };
        counts[rotation] = counts[rotation].saturating_add(1);
    }

    assert!(counts.iter().all(|count| *count > 0), "{counts:?}");
    let minimum = counts.iter().copied().min().unwrap_or_default();
    let maximum = counts.iter().copied().max().unwrap_or_default();
    assert!(maximum - minimum <= 24, "{counts:?}");
}

#[test]
fn transition_pattern_focus_removes_only_shared_one_hole_context() {
    let variable = RuliadTerm::variable(0);
    let before = RuliadTerm::apply(
        "context",
        vec![RuliadTerm::apply(
            "context",
            vec![RuliadTerm::apply("identity", vec![variable.clone()])],
        )],
    );
    let after = RuliadTerm::apply(
        "context",
        vec![RuliadTerm::apply("context", vec![variable.clone()])],
    );
    let (before_focus, after_focus) = transition_pattern_focus(&before, &after);
    assert_eq!(before_focus.canonical_text(), "identity(?0)");
    assert_eq!(after_focus, &variable);
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
                    Some(UsizeRangeConfig { min: 32, max: 32 })
                }
                RuliadFamilyKind::Automaton => Some(UsizeRangeConfig { min: 4, max: 4 }),
                RuliadFamilyKind::Rewrite => Some(UsizeRangeConfig { min: 24, max: 24 }),
                RuliadFamilyKind::Algebra => Some(UsizeRangeConfig { min: 3, max: 3 }),
                RuliadFamilyKind::Category => Some(UsizeRangeConfig { min: 4, max: 4 }),
                RuliadFamilyKind::ProofTree => Some(UsizeRangeConfig { min: 5, max: 5 }),
                RuliadFamilyKind::FormalProof => Some(UsizeRangeConfig { min: 2, max: 2 }),
                RuliadFamilyKind::LeanTask | RuliadFamilyKind::HashNoise => None,
            },
            steps: match family {
                RuliadFamilyKind::Eca | RuliadFamilyKind::Simulation => {
                    Some(UsizeRangeConfig { min: 8, max: 8 })
                }
                RuliadFamilyKind::Automaton => Some(UsizeRangeConfig { min: 24, max: 24 }),
                RuliadFamilyKind::Rewrite => Some(UsizeRangeConfig { min: 12, max: 12 }),
                RuliadFamilyKind::Category => Some(UsizeRangeConfig { min: 3, max: 3 }),
                RuliadFamilyKind::ProofTree => Some(UsizeRangeConfig { min: 4, max: 4 }),
                RuliadFamilyKind::FormalProof => Some(UsizeRangeConfig { min: 2, max: 2 }),
                RuliadFamilyKind::Algebra
                | RuliadFamilyKind::LeanTask
                | RuliadFamilyKind::HashNoise => None,
            },
        }];
        let sample = generate_sample(&config, &[], SampleSplit::Train, 0, 0).expect("sample");
        assert!(sample.categorical_presentation.categorical_core);
        let expected_abstraction = if family == RuliadFamilyKind::FormalProof {
            "verified_derivation_category"
        } else {
            "finite_category_reasoning"
        };
        assert_eq!(
            sample.categorical_presentation.abstraction,
            expected_abstraction
        );
        assert_eq!(
            sample.categorical_presentation.source_family,
            family.label()
        );
        let formal = family == RuliadFamilyKind::FormalProof;
        assert!(
            sample
                .text
                .starts_with(if formal { "[R3 " } else { "[R2 " })
        );
        assert!(sample.text.contains("\n?:"));
        assert!(sample.text.contains("\n!:"));
        for forbidden in if formal {
            &[][..]
        } else {
            &[
                "category",
                "finite",
                "normalize",
                "accepted",
                "theorem",
                "proof",
                "omitted",
                "steps",
                "chain",
                "expand",
                "add_mod",
                "affine_mod",
                "commutativity",
                "trace",
            ][..]
        } {
            assert!(
                !sample.text.contains(forbidden),
                "{} sample retained rote prose anchor `{forbidden}`:\n{}",
                family.label(),
                sample.text
            );
        }
        assert!(
            sample.text.len() <= if formal { 8192 } else { 512 },
            "{} sample exceeded trace-pretraining payload budget: {} bytes",
            family.label(),
            sample.text.len()
        );
        assert!(
            longest_repeated_char_run(&sample.text) <= 8,
            "{} sample has a degenerate repeated-character run:\n{}",
            family.label(),
            sample.text
        );
        assert!(
            longest_periodic_motif_run(&sample.text, 2) <= 24,
            "{} sample has a degenerate period-2 run:\n{}",
            family.label(),
            sample.text
        );
        assert!(
            longest_periodic_motif_run(&sample.text, 3) <= 24,
            "{} sample has a degenerate period-3 run:\n{}",
            family.label(),
            sample.text
        );
        assert!(
            sample.text.matches(',').count() <= sample.text.len() / 12,
            "{} sample retained a comma-heavy numeric surface:\n{}",
            family.label(),
            sample.text
        );
    }
}

#[test]
fn compact_usize_list_run_length_encodes_degenerate_runs() {
    assert_eq!(compact_usize_list(&[1, 1, 1]), "1,1,1");
    assert_eq!(compact_usize_list(&[1, 1, 1, 1]), "1*4");
    assert_eq!(
        compact_usize_list(&[2, 2, 2, 2, 3, 4, 4, 4, 4, 4]),
        "2*4,3,4*5"
    );
    let long = compact_usize_list(&(0..48).collect::<Vec<_>>());
    assert!(long.starts_with("u48:h"));
    assert!(!long.contains(','));
}

#[test]
fn compact_text_bounds_repeated_character_runs() {
    let text = compact_text("x=1111111111111111;y=BBBBBBBBBB", 128);
    assert!(text.contains("1^16"));
    assert!(text.contains("B^10"));
    assert!(longest_repeated_char_run(&text) <= 8);
}

#[test]
fn compact_symbolic_word_packs_long_low_alphabet_payloads() {
    let binary = compact_symbolic_word(&"0101".repeat(40), 48);
    assert!(binary.starts_with("b160:"), "{binary}");
    assert!(!binary.contains("01010101010101010101"), "{binary}");
    assert!(!binary.contains("555555"), "{binary}");

    let ternary = compact_symbolic_word(&"ABC".repeat(50), 48);
    assert!(ternary.starts_with("s150:"), "{ternary}");
    assert!(ternary.contains(":h"), "{ternary}");
    assert!(!ternary.contains("ABCABC"), "{ternary}");
    assert!(ternary.len() <= 64, "{ternary}");

    let medium = compact_symbolic_word(&"BC".repeat(12), 64);
    assert!(medium.starts_with("s24:"), "{medium}");
    assert!(!medium.contains("BCBC"), "{medium}");
}

#[test]
fn categorical_presentation_uses_low_entropy_symbolic_certificates() {
    let binary = "0101".repeat(24);
    let eca = RuliadSampleSpec::Eca {
        rule: 30,
        width: binary.len(),
        steps: 2,
        initial: binary.clone(),
        trace: vec![binary.clone(), binary.clone(), binary.clone()],
        task: RuliadTaskKind::MultiStepState,
    };
    let eca_answer = ruliad_categorical_presentation(&eca).answer;
    assert!(
        eca_answer.starts_with("targetlen=96;targetalpha=01;"),
        "{eca_answer}"
    );
    assert!(eca_answer.contains(";targetcounts=48,48;"), "{eca_answer}");
    assert!(!eca_answer.contains("010101010101"), "{eca_answer}");
    assert!(!eca_answer.contains(":h"), "{eca_answer}");

    let normal_form = "AB".repeat(32);
    let rewrite = RuliadSampleSpec::Rewrite {
        alphabet: "AB".to_string(),
        rules: vec![RuliadRewriteRule {
            from: "AA".to_string(),
            to: "A".to_string(),
        }],
        initial: normal_form.clone(),
        steps: 1,
        trace: vec![normal_form.clone()],
        normal_form,
        task: RuliadTaskKind::RewriteNormalForm,
    };
    let rewrite_answer = ruliad_categorical_presentation(&rewrite).answer;
    assert!(
        rewrite_answer.starts_with("normal_formlen=64;normal_formalpha=AB;"),
        "{rewrite_answer}"
    );
    assert!(
        rewrite_answer.contains(";normal_formcounts=32,32;"),
        "{rewrite_answer}"
    );
    assert!(!rewrite_answer.contains("ABABABAB"), "{rewrite_answer}");
    assert!(!rewrite_answer.contains(":h"), "{rewrite_answer}");
}

#[test]
fn compact_proof_step_runs_collapse_repeated_identity_chains() {
    let mut steps = vec!["a*b=c".to_string()];
    steps.extend(std::iter::repeat_n("c*id=c".to_string(), 32));
    steps.push("close".to_string());
    let compacted = compact_proof_step_runs(&steps, 32);
    assert_eq!(
        compacted,
        vec![
            "a*b=c".to_string(),
            "c*id=c *32".to_string(),
            "close".to_string()
        ]
    );
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
    assert!(
        prompt.contains("\nA:ok,l,r\n"),
        "prompt should expose answer schema without values: {prompt}"
    );
    assert!(!prompt.contains("ok=1"));
    assert!(!prompt.contains("[/R2]"));
}

#[test]
fn sample_text_answer_slot_matches_verifier_expected_answer() {
    let mut config = config();
    config.families = vec![RuliadFamilyConfig {
        kind: RuliadFamilyKind::ProofTree,
        weight: 1,
        width: Some(UsizeRangeConfig { min: 5, max: 5 }),
        steps: Some(UsizeRangeConfig { min: 4, max: 4 }),
    }];
    let sample = generate_sample(&config, &[], SampleSplit::Train, 0, 0).expect("sample");
    let text = sample_text(&sample.spec, &sample.oracle_hash);
    let expected_line = format!("!:{}", ruliad_expected_answer(&sample.spec));
    assert!(
        text.contains(&expected_line),
        "canonical ruliad documents must train the full keyed verifier answer: {text}"
    );
    assert!(
        !text.contains("\n!:1;"),
        "canonical ruliad documents must not train compact value-only answers: {text}"
    );
}

#[test]
fn answer_contract_lists_keys_without_values() {
    assert_eq!(compact_answer_keys("ok=1;l=19;r=19"), "ok,l,r");
    assert_eq!(compact_answer_keys("sha=abcdef0123456789"), "sha");
    assert_eq!(compact_answer_keys("bare"), "value");
    assert_eq!(compact_answer_values("ok=1;l=19;r=19"), "1;19;19");
    assert_eq!(
        compact_answer_values("nf=s38:CAB:hc0ee52f513ebc9aa:c14,.."),
        "s38:CAB:hc0ee52f513ebc9aa:c14,.."
    );
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
fn functor_preservation_handles_one_arrow_paths() {
    let family = RuliadFamilyConfig {
        kind: RuliadFamilyKind::Category,
        weight: 1,
        width: Some(UsizeRangeConfig { min: 3, max: 6 }),
        steps: Some(UsizeRangeConfig { min: 2, max: 2 }),
    };
    for sample_index in 0..64 {
        let mut rng = sample_rng(42, SampleSplit::Train, sample_index, 0, 0);
        let sample = generate_category_spec_for_task(
            &family,
            RuliadTaskKind::VerifyFunctorPreservation,
            &mut rng,
        )
        .expect("functor-preservation category spec");
        let report = verify_spec(&sample).expect("verify");
        assert!(
            report.ok,
            "one-arrow functor-preservation sample {sample_index} should verify"
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
    assert!(text.contains("\nA:ok,l,r\n"));
    assert!(text.contains("\n!:ok="));
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
        let sample = generate_sample(&config(), &[], SampleSplit::Train, 0, index).expect("sample");
        assert!(
            !is_degenerate_spec(&sample.spec),
            "degenerate generated sample: family={} task={} index={index}",
            sample.family.label(),
            sample.task_kind.label()
        );
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
        } else if sample.family == RuliadFamilyKind::FormalProof {
            assert!(sample.categorical_presentation.categorical_core);
            assert_eq!(
                sample.categorical_presentation.abstraction,
                "verified_derivation_category"
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
            RuliadSampleSpec::FormalProof {
                problem,
                certificate,
                ..
            } => {
                let complexity = complexity_vector(problem, Some(certificate));
                widths.insert(complexity.syntax_nodes);
                step_counts.insert(complexity.proof_step_count);
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

#[test]
fn far_out_difficulty_continues_scaling_after_legacy_clamp() {
    let family = RuliadFamilyConfig {
        kind: RuliadFamilyKind::ProofTree,
        weight: 1,
        width: Some(UsizeRangeConfig { min: 5, max: 17 }),
        steps: Some(UsizeRangeConfig { min: 4, max: 12 }),
    };
    let near = scale_family_for_difficulty(&family, 32);
    let far = scale_family_for_difficulty(&family, 96);
    assert!(
        far.width.as_ref().expect("far width").max > near.width.as_ref().expect("near width").max,
        "far-out proof-tree modulus range should exceed the legacy d32 clamp"
    );
    assert!(
        far.steps.as_ref().expect("far steps").max > near.steps.as_ref().expect("near steps").max,
        "far-out proof-tree depth range should exceed the legacy d32 clamp"
    );
}

#[test]
fn formal_sample_provenance_reports_the_concrete_ir_domain() {
    for domain in RuliadFormalDomain::ALL {
        let bundle = generate_formal_bundle(
            101,
            RuliadFormalGeneratorConfig {
                domain: Some(domain),
                ..RuliadFormalGeneratorConfig::default()
            },
        )
        .expect("formal bundle");
        let spec = RuliadSampleSpec::FormalProof {
            problem: bundle.problem,
            certificate: bundle.certificate,
            candidate: None,
            proof_step_index: None,
            action_presentation_rotation: None,
            action_answer_contract: RuliadProofActionAnswerContract::default(),
            task: RuliadTaskKind::ConstructProof,
        };
        let domains = ruliad_sample_math_domains(&spec);
        assert!(domains.contains(&RuliadMathDomain::FormalProof));
        assert_eq!(
            domains.len(),
            if domain == RuliadFormalDomain::Equational {
                3
            } else {
                2
            }
        );
        assert_eq!(
            domains
                .iter()
                .filter(|candidate| **candidate == RuliadMathDomain::CategoryTheory)
                .count(),
            usize::from(domain == RuliadFormalDomain::Category)
        );
        assert_eq!(
            domains
                .iter()
                .filter(|candidate| **candidate == RuliadMathDomain::ProcessCalculus)
                .count(),
            usize::from(domain == RuliadFormalDomain::Process)
        );
    }
}
