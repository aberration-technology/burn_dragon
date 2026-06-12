use std::hint::black_box;
use std::path::Path;

use burn_dragon_language::dataset::{DatasetSplit, TokenSequenceDataset, UniversalityDataset};
use burn_dragon_language::tokenizer::{
    PretokenizedTokenizerConfig, TokenizerConfig, TokenizerKind,
};
use burn_dragon_universality::{
    FloatRangeConfig, NcaCorpusConfig, NcaSerializationConfig, NcaTokenizationConfig,
    RuliadCorpusConfig, RuliadSerializationConfig, RuliadSourceSelectionConfig,
    RuliadTokenizationConfig, UsizeRangeConfig, compact_ruliad_families, default_families,
};
use burn_ndarray::NdArray;
use criterion::Criterion;
use tempfile::{TempDir, tempdir};

type BenchBackend = NdArray<f32>;

struct BenchDataset {
    _dir: TempDir,
    dataset: UniversalityDataset,
}

fn device() -> burn::tensor::Device<BenchBackend> {
    Default::default()
}

fn pretokenized_tokenizer() -> TokenizerConfig {
    TokenizerConfig {
        vocab_path: None,
        kind: TokenizerKind::Pretokenized(PretokenizedTokenizerConfig {
            vocab_size: 50_257,
            bos_id: None,
            eos_id: Some(50_256),
            pad_id: None,
            unk_id: None,
        }),
    }
}

fn nca_config(output_dir: &Path) -> NcaCorpusConfig {
    let mut config = NcaCorpusConfig {
        output_dir: output_dir.into(),
        seed: 1337,
        name: "universality-loader-nca".to_string(),
        train_samples: 32,
        validation_samples: 8,
        chunk_token_capacity: 4096,
        serialization: NcaSerializationConfig::default(),
        tokenization: NcaTokenizationConfig::default(),
        families: default_families(),
    };
    for family in &mut config.families {
        family.grid_size = Some(UsizeRangeConfig { min: 12, max: 12 });
        family.steps = Some(UsizeRangeConfig { min: 10, max: 10 });
        family.state_count = Some(UsizeRangeConfig { min: 10, max: 10 });
        family.step_stride = Some(UsizeRangeConfig { min: 2, max: 2 });
        family.start_step = Some(UsizeRangeConfig { min: 0, max: 0 });
        family.identity_bias = Some(FloatRangeConfig { min: 0.0, max: 0.0 });
        family.temperature = Some(FloatRangeConfig { min: 0.0, max: 0.0 });
    }
    config
}

fn ruliad_config(output_dir: &Path) -> RuliadCorpusConfig {
    let source_selection = RuliadSourceSelectionConfig {
        enabled: true,
        ..RuliadSourceSelectionConfig::default()
    };
    RuliadCorpusConfig {
        output_dir: output_dir.into(),
        seed: 1337,
        name: "universality-loader-ruliad".to_string(),
        train_samples: 32,
        validation_samples: 8,
        chunk_token_capacity: 4096,
        serialization: RuliadSerializationConfig {
            document_tokens: 513,
            preview_samples: 1,
        },
        tokenization: RuliadTokenizationConfig::default(),
        source_selection,
        families: compact_ruliad_families(),
        proof_tasks: None,
        lean_task_limit: None,
    }
}

fn build_nca_dataset(block_size: usize, batch_size: usize) -> BenchDataset {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("nca.toml");
    let config = nca_config(dir.path());
    std::fs::write(
        &config_path,
        toml::to_string_pretty(&config).expect("nca config toml"),
    )
    .expect("write nca config");
    let dataset = UniversalityDataset::new_on_the_fly(
        &config_path,
        block_size,
        batch_size,
        Some(block_size),
        &pretokenized_tokenizer(),
    )
    .expect("nca dataset");
    BenchDataset { _dir: dir, dataset }
}

fn build_ruliad_dataset(block_size: usize, batch_size: usize) -> BenchDataset {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("ruliad.toml");
    let config = ruliad_config(dir.path());
    std::fs::write(
        &config_path,
        toml::to_string_pretty(&config).expect("ruliad config toml"),
    )
    .expect("write ruliad config");
    let dataset = UniversalityDataset::new_ruliad_on_the_fly(
        &config_path,
        block_size,
        batch_size,
        &pretokenized_tokenizer(),
    )
    .expect("ruliad dataset");
    BenchDataset { _dir: dir, dataset }
}

fn bench_universality_loader(c: &mut Criterion) {
    let block_size = 512;
    let batch_size = 2;
    let device = device();

    let ruliad_cold = build_ruliad_dataset(block_size, batch_size);
    let mut ruliad_epoch = 0usize;
    c.bench_function("universality_loader/ruliad_prepare_cold_epoch", |b| {
        b.iter(|| {
            ruliad_epoch = ruliad_epoch.wrapping_add(1);
            ruliad_cold
                .dataset
                .prepare_epoch(DatasetSplit::Train, ruliad_epoch);
            black_box(ruliad_epoch);
        })
    });

    let nca_cold = build_nca_dataset(block_size, batch_size);
    let mut nca_epoch = 0usize;
    c.bench_function("universality_loader/nca_prepare_cold_epoch", |b| {
        b.iter(|| {
            nca_epoch = nca_epoch.wrapping_add(1);
            nca_cold
                .dataset
                .prepare_epoch(DatasetSplit::Train, nca_epoch);
            black_box(nca_epoch);
        })
    });

    let ruliad_warm = build_ruliad_dataset(block_size, batch_size);
    ruliad_warm.dataset.prepare_epoch(DatasetSplit::Train, 0);
    c.bench_function("universality_loader/ruliad_sample_batch_warm_cache", |b| {
        b.iter(|| {
            black_box(
                ruliad_warm
                    .dataset
                    .sample_batch::<BenchBackend>(DatasetSplit::Train, &device),
            );
        })
    });

    let nca_warm = build_nca_dataset(block_size, batch_size);
    nca_warm.dataset.prepare_epoch(DatasetSplit::Train, 0);
    c.bench_function("universality_loader/nca_sample_batch_warm_cache", |b| {
        b.iter(|| {
            black_box(
                nca_warm
                    .dataset
                    .sample_batch::<BenchBackend>(DatasetSplit::Train, &device),
            );
        })
    });
}

fn cargo_test_invocation() -> bool {
    std::env::args_os().skip(1).any(|arg| {
        arg.to_str()
            .is_some_and(|arg| arg == "--test-threads" || arg.starts_with("--test-threads="))
    })
}

fn main() {
    if cargo_test_invocation() {
        return;
    }

    let mut criterion = Criterion::default().configure_from_args();
    bench_universality_loader(&mut criterion);
    criterion.final_summary();
}
