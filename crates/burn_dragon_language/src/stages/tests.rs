use super::*;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use tempfile::tempdir;

use crate::DatasetSourceConfig;

#[test]
fn bundle_config_validates_stage_references() {
    let dir = tempdir().expect("tempdir");
    let config_path = dir.path().join("bundle.toml");
    fs::write(
        &config_path,
        r#"
name = "demo"
output_dir = "runs/demo"

[[stages]]
name = "gen"
type = "universality_generate"
config = "gen.toml"

[[stages]]
name = "train"
type = "language_train"
config = "train.toml"
backend = "ndarray"
dataset_manifest_from_stage = "gen"
"#,
    )
    .expect("write config");
    let config = load_experiment_bundle_config(&config_path).expect("bundle config");
    assert_eq!(config.stages.len(), 2);
}

#[test]
fn prepare_language_stage_config_injects_manifest_and_checkpoint() {
    let dir = tempdir().expect("tempdir");
    let bundle_path = dir.path().join("bundle.toml");
    fs::write(
        &bundle_path,
        "name = \"demo\"\noutput_dir = \"runs/demo\"\n",
    )
    .expect("write bundle");
    let train_cfg = dir.path().join("train.toml");
    fs::write(
        &train_cfg,
        r#"
[dataset]
cache_dir = "cache"
type = "universality_manifest"
manifest = "placeholder.json"

[dataset.tokenizer]
type = "pretokenized"
vocab_size = 50257
eos_id = 50256

[training]
block_size = 64
batch_size = 2
max_iters = 4
log_frequency = 1

[optimizer]
learning_rate = 0.001
weight_decay = 0.0

[generation]
prompt = "1 2 3"

[model]
n_layer = 1
n_embd = 8
n_head = 1
mlp_internal_dim_multiplier = 1
fused_kernels = true
rotary_embedding = "alibi"
"#,
    )
    .expect("write training config");

    let stage = ExperimentStageConfig {
        name: "train".to_string(),
        depends_on: Vec::new(),
        kind: ExperimentStageKind::LanguageTrain {
            config: PathBuf::from("train.toml"),
            backend: ExperimentBackend::Ndarray,
            dataset_manifest_from_stage: Some("gen".to_string()),
            init_checkpoint_from_stage: Some("pre".to_string()),
            launch_mode: burn_dragon_train::train::pipeline::TrainingLaunchMode::Fresh,
        },
    };
    let mut artifacts = BTreeMap::new();
    artifacts.insert(
        "gen".to_string(),
        ExperimentStageArtifact {
            manifest_path: Some(dir.path().join("gen/manifest.json")),
            corpus_output_dir: Some(dir.path().join("gen")),
            ..ExperimentStageArtifact::default()
        },
    );
    artifacts.insert(
        "pre".to_string(),
        ExperimentStageArtifact {
            latest_checkpoint_dir: Some(dir.path().join("pre/checkpoint")),
            latest_checkpoint_epoch: Some(3),
            ..ExperimentStageArtifact::default()
        },
    );

    let config = prepare_language_stage_config(
        &bundle_path,
        Path::new("train.toml"),
        &dir.path().join("stage"),
        &stage,
        &artifacts,
    )
    .expect("prepare stage config");

    assert!(matches!(
        config.dataset.source,
        DatasetSourceConfig::UniversalityManifest { .. }
    ));
    assert_eq!(
        config.training.init_checkpoint_path,
        Some(dir.path().join("pre/checkpoint"))
    );
    assert_eq!(config.training.init_checkpoint_epoch, Some(3));
    assert_eq!(
        config.run_layout.base_dir,
        Some(dir.path().join("stage").join("runs"))
    );
    assert!(!config.run_layout.mirror_config_path);
}

#[test]
fn prepare_universality_stage_config_accepts_ruliad_corpus() {
    let dir = tempdir().expect("tempdir");
    let bundle_path = dir.path().join("bundle.toml");
    fs::write(
        &bundle_path,
        "name = \"demo\"\noutput_dir = \"runs/demo\"\n",
    )
    .expect("write bundle");
    let corpus_cfg = dir.path().join("ruliad.toml");
    fs::write(
        &corpus_cfg,
        r#"
output_dir = "ignored"
seed = 1337
name = "stage-ruliad"
train_samples = 4
validation_samples = 2
chunk_token_capacity = 256

[serialization]
document_tokens = 96
preview_samples = 1

[tokenization]
type = "gpt2_byte_compatible"
vocab_size = 50257
eos_id = 50256

[[families]]
kind = "eca"
weight = 1
width = { min = 12, max = 12 }
steps = { min = 4, max = 4 }
"#,
    )
    .expect("write corpus config");

    let prepared = prepare_universality_stage_config(
        &bundle_path,
        &dir.path().join("stage"),
        Path::new("ruliad.toml"),
    )
    .expect("prepare ruliad stage");

    match prepared {
        PreparedUniversalityCorpusConfig::Ruliad(config) => {
            assert_eq!(config.output_dir, dir.path().join("stage/output"));
            assert_eq!(config.name, "stage-ruliad");
        }
        PreparedUniversalityCorpusConfig::Nca(_) => panic!("expected ruliad config"),
    }
}

#[test]
fn generate_prepared_universality_stage_corpus_handles_ruliad() {
    let dir = tempdir().expect("tempdir");
    let config =
        PreparedUniversalityCorpusConfig::Ruliad(burn_dragon_universality::RuliadCorpusConfig {
            output_dir: dir.path().join("out"),
            seed: 7,
            name: "stage-ruliad-generate".to_string(),
            train_samples: 2,
            validation_samples: 1,
            chunk_token_capacity: 256,
            serialization: burn_dragon_universality::RuliadSerializationConfig {
                document_tokens: 96,
                preview_samples: 1,
            },
            tokenization: burn_dragon_universality::RuliadTokenizationConfig::default(),
            source_selection: burn_dragon_universality::RuliadSourceSelectionConfig::default(),
            families: vec![burn_dragon_universality::RuliadFamilyConfig {
                kind: burn_dragon_universality::RuliadFamilyKind::Eca,
                weight: 1,
                width: Some(burn_dragon_universality::UsizeRangeConfig { min: 12, max: 12 }),
                steps: Some(burn_dragon_universality::UsizeRangeConfig { min: 4, max: 4 }),
            }],
            proof_tasks: None,
            lean_task_limit: None,
        });
    let report =
        generate_prepared_universality_stage_corpus(&config).expect("generate ruliad corpus");
    assert!(report.manifest_path().is_file());
    assert!(report.sample_records_path().is_file());
}

#[test]
fn relative_bundle_output_dir_resolves_from_cwd() {
    let config = ExperimentBundleConfig {
        name: "demo".to_string(),
        output_dir: PathBuf::from("runs/demo"),
        resume_from_last_completed_stage: true,
        stages: vec![ExperimentStageConfig {
            name: "gen".to_string(),
            depends_on: Vec::new(),
            kind: ExperimentStageKind::UniversalityGenerate {
                config: PathBuf::from("gen.toml"),
            },
        }],
    };
    let expected = std::env::current_dir().expect("cwd").join("runs/demo");
    assert_eq!(resolve_bundle_root(&config), expected);
}

#[test]
fn current_best_large_baseline_bundle_loads() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("config")
        .join("language")
        .join("baselines");
    let bundle_path = root.join("current_best_large.toml");
    let config = load_experiment_bundle_config(&bundle_path).expect("baseline bundle");
    assert_eq!(config.name, "current_best_large_48h");
    assert_eq!(config.stages.len(), 2);
    assert_eq!(config.stages[0].name, "nca_prepretrain");
    assert_eq!(config.stages[1].name, "climbmix_pretrain");
}

#[test]
fn current_best_large_mamba_contender_bundle_loads() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("config")
        .join("language")
        .join("baselines");
    let bundle_path = root.join("current_best_large_mamba_contender.toml");
    let config = load_experiment_bundle_config(&bundle_path).expect("mamba contender bundle");
    assert_eq!(config.name, "current_best_large_mamba_contender_48h");
    assert_eq!(config.stages.len(), 2);
    assert_eq!(config.stages[0].name, "nca_prepretrain");
    assert_eq!(config.stages[1].name, "climbmix_pretrain");
}
