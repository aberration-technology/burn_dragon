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
        },
        tokenization: RuliadTokenizationConfig::default(),
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
            prior: 1.0,
            cost: 1.0 + (index % 8) as f32,
            loss_ema: 1.0 + (index % 13) as f32 * 0.25,
            previous_loss_ema: 1.5 + (index % 13) as f32 * 0.25,
            gradient_alignment: if index % 5 == 0 { 0.5 } else { 0.0 },
            is_hash_noise: index % 17 == 0,
        })
        .collect()
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
}

criterion_group!(benches, bench_ruliad);
criterion_main!(benches);
