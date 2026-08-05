use burn_dragon_universality::ruliad::formal::{
    RuliadFormalGeneratorConfig, generate_formal_bundle,
};
use burn_dragon_universality::ruliad::{
    DEFAULT_PROOF_ACTION_CANDIDATES, RuliadKernelLimits, RuliadProofPolicyState, encode_problem,
    oracle_proof_action_set, replay_certificate,
};
use burn_dragon_universality::{
    OnlineRuliadCorpus, RuliadCorpusConfig, RuliadFamilyConfig, RuliadFamilyKind,
    RuliadFrontierSampler, RuliadSamplerCandidate, RuliadSamplerConfig, RuliadSerializationConfig,
    RuliadSourceSelectionConfig, RuliadTokenizationConfig, SampleSplit, UsizeRangeConfig,
    plan_epoch_source_buckets, ruliad_source_buckets, verify_sample,
};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};

fn ruliad_config() -> RuliadCorpusConfig {
    RuliadCorpusConfig {
        output_dir: "target/ruliad-bench".into(),
        seed: 1337,
        name: "ruliad-bench".to_string(),
        train_samples: 128,
        validation_samples: 32,
        chunk_token_capacity: 4096,
        serialization: RuliadSerializationConfig {
            document_tokens: 513,
            preview_samples: 1,
            ..RuliadSerializationConfig::default()
        },
        tokenization: RuliadTokenizationConfig::default(),
        formal_generalization: Default::default(),
        source_selection: RuliadSourceSelectionConfig::default(),
        families: vec![
            RuliadFamilyConfig {
                kind: RuliadFamilyKind::Eca,
                weight: 4,
                width: Some(UsizeRangeConfig { min: 16, max: 32 }),
                steps: Some(UsizeRangeConfig { min: 4, max: 10 }),
            },
            RuliadFamilyConfig {
                kind: RuliadFamilyKind::Simulation,
                weight: 2,
                width: Some(UsizeRangeConfig { min: 16, max: 32 }),
                steps: Some(UsizeRangeConfig { min: 4, max: 8 }),
            },
            RuliadFamilyConfig {
                kind: RuliadFamilyKind::LeanTask,
                weight: 1,
                width: None,
                steps: None,
            },
            RuliadFamilyConfig {
                kind: RuliadFamilyKind::HashNoise,
                weight: 1,
                width: None,
                steps: None,
            },
        ],
        proof_tasks: None,
        lean_task_limit: None,
    }
}

fn sampler_candidates(count: usize) -> Vec<RuliadSamplerCandidate> {
    (0..count)
        .map(|index| RuliadSamplerCandidate {
            oracle_hash: format!("candidate-{index}"),
            family: if index % 17 == 0 {
                "hash_noise".to_string()
            } else {
                "eca".to_string()
            },
            task_kind: if index % 17 == 0 {
                "hash_canary".to_string()
            } else {
                "multi_step_state".to_string()
            },
            answer_contract: String::new(),
            difficulty_level: index % 4,
            params_hash: format!("{index:016x}"),
            prior: 1.0,
            cost: 1.0 + (index % 8) as f32,
            loss_ema: 1.0 + (index % 13) as f32 * 0.25,
            previous_loss_ema: 1.5 + (index % 13) as f32 * 0.25,
            gradient_alignment: if index % 5 == 0 { 0.5 } else { 0.0 },
            is_hash_noise: index % 17 == 0,
            capability_feedback_count: 0,
            capability_verifier_ema: 0.0,
            capability_partial_ema: 0.0,
            capability_completion_health_ema: 0.0,
            capability_schema_wrong_ema: 0.0,
            capability_malformed_ema: 0.0,
            capability_missing_ema: 0.0,
        })
        .collect()
}

fn reconstruct_prefix_through_action_menus(
    problem: &burn_dragon_universality::ruliad::RuliadProofProblem,
    certificate: &burn_dragon_universality::ruliad::RuliadProofCertificate,
    step_index: usize,
) -> RuliadProofPolicyState {
    let mut state = RuliadProofPolicyState::new(problem);
    for index in 0..step_index {
        let actions =
            oracle_proof_action_set(problem, certificate, index, DEFAULT_PROOF_ACTION_CANDIDATES)
                .expect("proof action menu");
        state
            .apply(&actions, actions.selected_index)
            .expect("apply oracle action");
    }
    state
}

fn bench_ruliad(c: &mut Criterion) {
    let corpus = OnlineRuliadCorpus::new(ruliad_config()).expect("corpus");
    let mut sample_index = 0usize;
    c.bench_function("ruliad/generate_raw_sample", |b| {
        b.iter(|| {
            sample_index = sample_index.wrapping_add(1);
            corpus
                .generate_raw_sample(SampleSplit::Train, sample_index / 128, sample_index % 128)
                .expect("sample")
        })
    });

    let sample = corpus
        .generate_raw_sample(SampleSplit::Train, 0, 7)
        .expect("sample");
    c.bench_function("ruliad/verify_sample", |b| {
        b.iter(|| verify_sample(&sample.spec).expect("verify"))
    });

    c.bench_function("ruliad/token_document", |b| {
        b.iter_batched(
            || {
                sample_index = sample_index.wrapping_add(1);
                sample_index
            },
            |index| {
                corpus
                    .generate_document_tokens_for_epoch(
                        SampleSplit::Train,
                        index / 128,
                        index % 128,
                    )
                    .expect("document")
            },
            BatchSize::SmallInput,
        )
    });

    let sampler =
        RuliadFrontierSampler::new(RuliadSamplerConfig::default(), sampler_candidates(10_000));
    c.bench_function("ruliad/sampler_probabilities_10k", |b| {
        b.iter(|| sampler.probabilities())
    });

    let config = ruliad_config();
    let buckets = ruliad_source_buckets(&config);
    let probabilities = vec![1.0 / buckets.len().max(1) as f32; buckets.len()];
    c.bench_function("ruliad/source_plan_1k", |b| {
        b.iter(|| plan_epoch_source_buckets(&buckets, &probabilities, 1_024, 1337, 0, 3))
    });

    for (label, difficulty) in [("d0", 0), ("d32", 32), ("d256", 256)] {
        let config = RuliadFormalGeneratorConfig::for_difficulty(difficulty);
        let mut seed = 1_337u64;
        c.bench_function(&format!("ruliad/r3_generate_{label}"), |b| {
            b.iter(|| {
                seed = seed.wrapping_add(1);
                generate_formal_bundle(seed, config).expect("formal bundle")
            })
        });
        let bundle = generate_formal_bundle(1_337, config).expect("formal bundle");
        c.bench_function(&format!("ruliad/r3_replay_{label}"), |b| {
            b.iter(|| {
                replay_certificate(
                    &bundle.problem,
                    &bundle.certificate,
                    RuliadKernelLimits::default(),
                )
            })
        });
        c.bench_function(&format!("ruliad/r3_encode_problem_{label}"), |b| {
            b.iter(|| encode_problem(&bundle.problem).expect("compact problem"))
        });
        let step_index = bundle.certificate.step_count() / 2;
        c.bench_function(&format!("ruliad/r3_action_menu_{label}"), |b| {
            b.iter(|| {
                oracle_proof_action_set(
                    &bundle.problem,
                    &bundle.certificate,
                    step_index,
                    DEFAULT_PROOF_ACTION_CANDIDATES,
                )
                .expect("proof action menu")
            })
        });
        if difficulty <= 32 {
            c.bench_function(&format!("ruliad/r3_prefix_direct_{label}"), |b| {
                b.iter(|| {
                    RuliadProofPolicyState::from_certificate_prefix(
                        &bundle.problem,
                        &bundle.certificate,
                        step_index,
                    )
                    .expect("direct certificate prefix")
                })
            });
            c.bench_function(&format!("ruliad/r3_prefix_action_menus_{label}"), |b| {
                b.iter(|| {
                    reconstruct_prefix_through_action_menus(
                        &bundle.problem,
                        &bundle.certificate,
                        step_index,
                    )
                })
            });
        }
    }
}

criterion_group!(benches, bench_ruliad);
criterion_main!(benches);
