use std::collections::{BTreeMap, BTreeSet};

use burn_dragon_language::{DatasetConfig, DatasetSourceConfig, DragonConfig, TrainingConfig};
use burn_p2p::burn::{BurnArtifactConfig, BurnRecordPrecision, BurnWorkloadConfig};
use burn_p2p::{
    BrowserRolePolicy, BrowserVisibilityPolicy, ChunkingScheme, ClientPlatform,
    ClientReleaseManifest, ContentId, DatasetViewId, DiffusionSteadyStatePolicy,
    ExperimentDirectoryEntry, ExperimentDirectoryPolicyExt, ExperimentId, ExperimentOptInPolicy,
    ExperimentResourceRequirements, ExperimentScope, ExperimentVisibility, HeadPromotionMode,
    HeadPromotionPolicy, LocalOptimizerStatePolicy, MergeStrategy, MergeTopologyPolicy, NetworkId,
    NetworkManifest, ParameterSubsetCatalog, PeerRole, PeerRoleSet, Precision, ProjectFamilyId,
    RecurrentStatePolicy, RevisionId, RevisionManifest, RobustnessPolicy, SchedulerStatePolicy,
    StudyId, SupportedWorkload, TRAINING_CONTRACT_VERSION, TrainingContractManifest,
    TrainingProtocol, UpdateCodec, WindowActivation, WindowId, WorkloadId,
};
use sha2::{Digest, Sha256};

use crate::capability::{DragonCapabilityClass, DragonTrainingFootprint};
use crate::config::{
    DRAGON_LOCAL_PC_PROGRAM_CONTRACT_EXTENSION, DRAGON_RULIAD_SEMANTIC_CONTRACT_EXTENSION,
    DragonExperimentKind, DragonManifestBundle, DragonManifestSeed, DragonPromotionConfig,
    DragonPromotionMode, dragon_model_schema_hash,
};
use crate::profile::{
    DRAGON_BROWSER_EXECUTION_CONTRACT_EXTENSION, DragonBrowserExperimentProfile,
    DragonExperimentProfile, browser_profile_execution_contract_hash,
};

fn stable_content_id<T: serde::Serialize>(label: &str, value: &T) -> ContentId {
    let bytes = serde_json::to_vec(value).expect("stable content id json");
    let mut hasher = Sha256::new();
    hasher.update(label.as_bytes());
    hasher.update([0]);
    hasher.update(bytes);
    ContentId::new(format!("{label}-{:x}", hasher.finalize()))
}

fn dragon_target_artifact_hash(
    target_artifact_id: &str,
    target_platform: ClientPlatform,
    release_train_hash: &ContentId,
) -> ContentId {
    stable_content_id(
        "dragon-target-artifact",
        &serde_json::json!({
            "target_artifact_id": target_artifact_id,
            "target_platform": target_platform,
            "release_train_hash": release_train_hash,
        }),
    )
}

struct DragonTrainingContractInput<'a> {
    experiment_kind: DragonExperimentKind,
    model_config: &'a DragonConfig,
    training_config: Option<&'a TrainingConfig>,
    dataset_view_id: DatasetViewId,
    checkpoint_format_hash: ContentId,
    merge_topology_policy: &'a MergeTopologyPolicy,
    root_ema_update_basis_points: u16,
    browser_profile: Option<&'a DragonBrowserExperimentProfile>,
    random_scaffold_catalog: Option<&'a ParameterSubsetCatalog>,
    random_scaffold_update_encoding: burn_p2p::CompactScalarEncoding,
    training_protocol: &'a TrainingProtocol,
}

struct DragonDatasetContract {
    descriptor: serde_json::Value,
    ruliad_semantic_hash: Option<ContentId>,
}

fn dragon_dataset_contract(
    dataset: Option<&DatasetConfig>,
    dataset_view_id: &DatasetViewId,
) -> anyhow::Result<DragonDatasetContract> {
    let Some(dataset) = dataset else {
        return Ok(DragonDatasetContract {
            descriptor: serde_json::Value::Null,
            ruliad_semantic_hash: None,
        });
    };
    let (source, ruliad_semantic_hash) =
        dragon_dataset_source_contract(&dataset.source, dataset_view_id)?;
    let validation = dataset
        .validation
        .as_ref()
        .map(|validation| {
            dragon_dataset_source_contract(&validation.source, dataset_view_id).map(
                |(source, _)| {
                    serde_json::json!({
                        "train_split_ratio": validation.train_split_ratio,
                        "source": source,
                    })
                },
            )
        })
        .transpose()?;
    Ok(DragonDatasetContract {
        descriptor: serde_json::json!({
            "train_split_ratio": dataset.train_split_ratio,
            "source": source,
            "validation": validation,
        }),
        ruliad_semantic_hash,
    })
}

fn dragon_dataset_source_contract(
    source: &DatasetSourceConfig,
    dataset_view_id: &DatasetViewId,
) -> anyhow::Result<(serde_json::Value, Option<ContentId>)> {
    match source {
        DatasetSourceConfig::UniversalityRuliad { config } => {
            let contract =
                burn_dragon_universality::ruliad::RuliadSemanticContract::from_config_path(config)?;
            let hash = ContentId::new(format!(
                "dragon-ruliad-semantics-{}",
                contract.canonical_hash()?
            ));
            Ok((
                serde_json::json!({
                    "type": "universality_ruliad",
                    "semantic_contract_hash": hash,
                }),
                Some(hash),
            ))
        }
        DatasetSourceConfig::UniversalityNca { config } => {
            let mut config = burn_dragon_universality::load_nca_config(config)?;
            config.output_dir = std::path::PathBuf::new();
            config.name.clear();
            config.chunk_token_capacity = 0;
            config.serialization.preview_samples = 0;
            Ok((
                serde_json::json!({
                    "type": "universality_nca",
                    "semantic_contract_hash": stable_content_id("dragon-nca-corpus", &config),
                }),
                None,
            ))
        }
        DatasetSourceConfig::UniversalityManifest { .. } => Ok((
            serde_json::json!({
                "type": "universality_manifest",
                "dataset_view_id": dataset_view_id,
            }),
            None,
        )),
        DatasetSourceConfig::NemotronClimbMix {
            revision,
            max_records,
        } => Ok((
            serde_json::json!({
                "type": "nemotron_climb_mix",
                "revision": revision,
                "max_records": max_records,
            }),
            None,
        )),
    }
}

fn dragon_training_contract(
    input: DragonTrainingContractInput<'_>,
) -> anyhow::Result<(TrainingContractManifest, ContentId)> {
    let DragonTrainingContractInput {
        experiment_kind,
        model_config,
        training_config,
        dataset_view_id,
        checkpoint_format_hash,
        merge_topology_policy,
        root_ema_update_basis_points,
        browser_profile,
        random_scaffold_catalog,
        random_scaffold_update_encoding,
        training_protocol,
    } = input;
    let dataset_contract = dragon_dataset_contract(
        training_config.map(|config| &config.dataset),
        &dataset_view_id,
    )?;
    let update_codec = match training_config.map(|config| config.optimizer.name) {
        Some(burn_dragon_train::OptimizerKind::Eggroll) => {
            let eggroll = &training_config
                .expect("training config matched above")
                .optimizer
                .eggroll;
            UpdateCodec::SeededFitness {
                population: u32::try_from(eggroll.population.population_size)
                    .map_err(|_| anyhow::anyhow!("EGGROLL population exceeds u32::MAX"))?,
                rank: u32::try_from(eggroll.population.rank)
                    .map_err(|_| anyhow::anyhow!("EGGROLL rank exceeds u32::MAX"))?,
                seed: eggroll.population.seed,
                replay: burn_p2p::SeededFitnessReplayPolicy::default(),
            }
        }
        _ if model_config.random_scaffold.enabled => {
            let catalog = random_scaffold_catalog.ok_or_else(|| {
                anyhow::anyhow!(
                    "random-scaffold training requires a canonical trainable parameter catalog"
                )
            })?;
            catalog.validate()?;
            anyhow::ensure!(
                catalog.model_schema_hash == dragon_model_schema_hash(model_config),
                "random-scaffold parameter catalog model schema mismatch"
            );
            UpdateCodec::MutableSubsetParameters {
                parameter_catalog_hash: catalog.catalog_id()?,
                parameter_count: catalog.parameter_count()?,
                encoding: random_scaffold_update_encoding,
            }
        }
        _ => UpdateCodec::FullModel,
    };
    let forward_only_update = matches!(update_codec, UpdateCodec::SeededFitness { .. });
    let persistent_peer_local_state =
        !forward_only_update && matches!(training_protocol, TrainingProtocol::DiLoCo(_));
    let model_program_hash = stable_content_id(
        "dragon-model-program",
        &serde_json::json!({
            "arch": "dragon_dragon",
            "n_embd": model_config.n_embd,
            "n_head": model_config.n_head,
            "n_layer": model_config.n_layer,
            "latent_total": model_config.latent_total(),
            "latent_per_head": model_config.latent_per_head(),
            "sequence_kernel": model_config.sequence_kernel,
            "vocab_size": model_config.vocab_size,
            "language_head": model_config.language_head,
            "hierarchical_dragon": model_config.hierarchical_dragon,
            "latent_reasoning": model_config.latent_reasoning,
            "next_latent_transition": model_config.next_latent_transition,
            "random_scaffold": model_config.random_scaffold,
        }),
    );
    let tokenizer_hash = stable_content_id(
        "dragon-tokenizer",
        &training_config.map(|config| &config.dataset.tokenizer),
    );
    let preprocessing_hash = stable_content_id(
        "dragon-preprocessing",
        &training_config.map(|config| {
            serde_json::json!({
                "dataset": dataset_contract.descriptor,
                "block_size": config.training.block_size,
                "tbptt_chunk_size": config.training.tbptt_chunk_size,
                "tbptt_credit_window_chunks": config.training.tbptt_credit_window_chunks,
                "tbptt_persist_across_steps": config.training.tbptt_persist_across_steps,
                "sequence_batching": config.training.sequence_batching,
                "context_strategy": config.training.context_strategy,
            })
        }),
    );
    let objective_hash = stable_content_id(
        "dragon-objective",
        &training_config.map(|config| {
            serde_json::json!({
                "algorithm": config.training.algorithm,
                "objective": config.training.objective,
                "input_corruption": config.training.input_corruption,
                "logit_entropy_floor": config.training.logit_entropy_floor,
                "repeat_unlikelihood": config.training.repeat_unlikelihood,
                "greedy_rollout_unlikelihood": config.training.greedy_rollout_unlikelihood,
                "dynamics_anchor": config.training.dynamics_anchor,
                "predictive_coding": config.training.predictive_coding,
                "local_predictive_coding": config.training.local_predictive_coding,
                "predictive_context_routing": config.training.predictive_context_routing,
                "latent_reasoning": config.training.latent_reasoning,
                "ruliad_supervision": config.training.ruliad_supervision,
                "gdpo": config.training.gdpo,
            })
        }),
    );
    let optimizer_hash = stable_content_id(
        "dragon-optimizer",
        &training_config.map(|config| {
            serde_json::json!({
                "optimizer": config.optimizer,
                "module_lr_scales": config.training.module_lr_scales,
                "continual_backprop": config.training.continual_backprop,
                "neuron_scaling": config.training.neuron_scaling,
            })
        }),
    );
    let scheduler_hash = stable_content_id(
        "dragon-scheduler",
        &training_config.map(|config| &config.optimizer),
    );
    let initialization_hash = stable_content_id(
        "dragon-initialization",
        &training_config.map(|config| {
            serde_json::json!({
                "seed": config.training.seed,
                "init_transfer": config.training.init_transfer,
                "init_checkpoint_path": config.training.init_checkpoint_path,
                "init_checkpoint_epoch": config.training.init_checkpoint_epoch,
                "model": model_config,
            })
        }),
    );
    let validation_hash = stable_content_id(
        "dragon-validation",
        &training_config.map(|config| {
            serde_json::json!({
                "validation_dataset": config.dataset.validation,
                "generation": config.generation,
                "gates": config.training.gates,
            })
        }),
    );
    let mut extensions = BTreeMap::from([(
        "experiment_kind".into(),
        stable_content_id("dragon-experiment-kind", &experiment_kind),
    )]);
    if let Some(browser_profile) = browser_profile {
        extensions.insert(
            DRAGON_BROWSER_EXECUTION_CONTRACT_EXTENSION.into(),
            browser_profile_execution_contract_hash(experiment_kind, browser_profile)?,
        );
    }
    if model_config.random_scaffold.enabled {
        extensions.insert(
            "dragon_random_scaffold".into(),
            ContentId::derive(&burn_dragon_core::build_dragon_random_scaffold_manifest(
                model_config,
            ))?,
        );
    }
    if let Some(ruliad_semantic_hash) = dataset_contract.ruliad_semantic_hash {
        extensions.insert(
            DRAGON_RULIAD_SEMANTIC_CONTRACT_EXTENSION.into(),
            ruliad_semantic_hash,
        );
    }
    if let Some(config) = training_config.filter(|config| {
        matches!(
            config.training.algorithm,
            burn_dragon_language::TrainingAlgorithm::PredictiveCoding
        )
    }) {
        let pc_manifest =
            burn_dragon_language::train::dragon_predictive_coding_checkpoint_manifest(
                model_config.n_layer,
                &config.training.local_predictive_coding,
            )?;
        extensions.insert(
            DRAGON_LOCAL_PC_PROGRAM_CONTRACT_EXTENSION.into(),
            stable_content_id("dragon-local-pc-program", &pc_manifest),
        );
    }
    let contract = TrainingContractManifest {
        version: TRAINING_CONTRACT_VERSION,
        workload_id: WorkloadId::new(format!("dragon-{}", experiment_kind.workload_slug())),
        model_program_hash,
        model_schema_hash: dragon_model_schema_hash(model_config),
        checkpoint_format_hash,
        dataset_view_id,
        tokenizer_hash,
        preprocessing_hash,
        objective_hash,
        optimizer_hash,
        scheduler_hash,
        optimizer_state_policy: if forward_only_update {
            LocalOptimizerStatePolicy::StatelessForwardOnly
        } else if persistent_peer_local_state {
            LocalOptimizerStatePolicy::PeerLocalPersistent
        } else {
            LocalOptimizerStatePolicy::ResetPerWindow
        },
        scheduler_state_policy: if forward_only_update {
            SchedulerStatePolicy::CanonicalAcceptedWork
        } else if persistent_peer_local_state {
            SchedulerStatePolicy::PeerLocalPersistent
        } else {
            SchedulerStatePolicy::ResetPerWindow
        },
        recurrent_state_policy: if training_config
            .is_some_and(|config| config.training.tbptt_persist_across_steps)
        {
            RecurrentStatePolicy::LeaseScoped
        } else {
            RecurrentStatePolicy::Ephemeral
        },
        update_codec: update_codec.clone(),
        aggregation_hash: stable_content_id(
            "dragon-aggregation",
            &serde_json::json!({
                "merge_topology": merge_topology_policy,
                "model_merge": {
                    "strategy": "weighted_mean_single_root_ema",
                    "root_ema_update_basis_points": root_ema_update_basis_points,
                },
                "update_codec": update_codec,
            }),
        ),
        validation_hash,
        initialization_hash,
        extensions,
    };
    contract.validate()?;
    let contract_id = contract.contract_id()?;
    Ok((contract, contract_id))
}

fn backend_resource_class(backend_label: &str) -> String {
    if backend_label.eq_ignore_ascii_case("cpu") || backend_label.eq_ignore_ascii_case("ndarray") {
        "cpu".into()
    } else if backend_label.eq_ignore_ascii_case("cuda") {
        "cuda".into()
    } else if backend_label.eq_ignore_ascii_case("rocm") {
        "rocm".into()
    } else {
        "wgpu".into()
    }
}

fn canonical_minimum_system_memory_bytes(footprint: &DragonTrainingFootprint) -> u64 {
    footprint
        .estimated_checkpoint_bytes
        .saturating_add(footprint.estimated_shard_bytes)
        .max(512 * 1024 * 1024)
}

const DRAGON_DIFFUSION_ARTIFACT_SYNC_TIMEOUT_SECS: u32 = 120;

fn dragon_merge_topology(
    experiment_kind: DragonExperimentKind,
    promotion: &DragonPromotionConfig,
) -> anyhow::Result<MergeTopologyPolicy> {
    anyhow::ensure!(
        promotion.validator_quorum > 0,
        "validator promotion quorum must be greater than zero",
    );
    let window_duration_secs = match experiment_kind {
        DragonExperimentKind::NcaPrepretraining => 60,
        DragonExperimentKind::RuliadPretraining => 120,
        DragonExperimentKind::ClimbMixPretraining => 180,
    };

    let (strategy, reducer_replication, upper_fanin, promotion_policy) = match promotion.mode {
        DragonPromotionMode::DiffusionSteadyState => (
            MergeStrategy::KRegularGossip,
            0,
            0,
            HeadPromotionPolicy {
                mode: HeadPromotionMode::DiffusionSteadyState,
                validator_quorum: 1,
                diffusion: Some(DiffusionSteadyStatePolicy {
                    artifact_sync_timeout_secs: DRAGON_DIFFUSION_ARTIFACT_SYNC_TIMEOUT_SECS,
                    ..DiffusionSteadyStatePolicy::default()
                }),
                ..HeadPromotionPolicy::default()
            },
        ),
        DragonPromotionMode::ValidatorQuorum => (
            MergeStrategy::MicrocohortReducePlusValidatorPromotion,
            1,
            2,
            HeadPromotionPolicy {
                mode: HeadPromotionMode::ValidatorQuorum,
                validator_quorum: promotion.validator_quorum,
                diffusion: None,
                ..HeadPromotionPolicy::default()
            },
        ),
    };

    Ok(MergeTopologyPolicy {
        strategy,
        reducer_replication,
        target_leaf_cohort: 3,
        upper_fanin,
        window_duration_secs,
        publish_jitter_ms: 750,
        staleness_windows: 2,
        promotion_policy,
    })
}

#[cfg(test)]
fn dragon_diffusion_merge_topology(experiment_kind: DragonExperimentKind) -> MergeTopologyPolicy {
    dragon_merge_topology(experiment_kind, &DragonPromotionConfig::default())
        .expect("default Dragon diffusion topology is valid")
}

fn dragon_robustness_policy(experiment_kind: DragonExperimentKind) -> RobustnessPolicy {
    let mut policy = RobustnessPolicy::balanced();
    policy.validator_canary_policy.minimum_evaluator_quorum = 1;

    if matches!(
        experiment_kind,
        DragonExperimentKind::NcaPrepretraining | DragonExperimentKind::RuliadPretraining
    ) {
        policy.validator_canary_policy.maximum_regression_delta = 1.0;
    }

    policy
}

fn browser_trainer_wgpu_enabled(
    profile: &DragonExperimentProfile,
    footprint: &DragonTrainingFootprint,
) -> bool {
    profile
        .browser
        .as_ref()
        .filter(|browser| browser.trainer_support.is_supported())
        .and_then(|browser| {
            browser
                .capability_policy
                .memory_budget_bytes(DragonCapabilityClass::BrowserWgpu)
        })
        .is_some_and(|budget| footprint.estimated_training_bytes <= budget)
}

#[allow(clippy::too_many_arguments)]
pub fn build_manifest_bundle(
    seed: &DragonManifestSeed,
    experiment_kind: DragonExperimentKind,
    backend_label: &str,
    model_config: &DragonConfig,
    training_config: Option<&TrainingConfig>,
    random_scaffold_catalog: Option<&ParameterSubsetCatalog>,
    profile: &DragonExperimentProfile,
    dataset_view_id: DatasetViewId,
    footprint: &DragonTrainingFootprint,
    app_semver: semver::Version,
    git_commit: &str,
    enabled_features_label: &str,
) -> anyhow::Result<DragonManifestBundle> {
    seed.training_protocol
        .validate()
        .map_err(anyhow::Error::from)?;
    anyhow::ensure!(
        seed.aggregation.root_ema_update_basis_points
            <= crate::config::DragonAggregationConfig::MAX_BASIS_POINTS,
        "root EMA update weight must be at most {} basis points, got {}",
        crate::config::DragonAggregationConfig::MAX_BASIS_POINTS,
        seed.aggregation.root_ema_update_basis_points,
    );
    let root_ema_update_weight = seed.aggregation.root_ema_update_weight();
    let workload_id = WorkloadId::new(format!("dragon-{}", experiment_kind.workload_slug()));
    let model_program_hash = stable_content_id(
        "dragon-model-program",
        &serde_json::json!({
            "arch": "dragon_dragon",
            "n_embd": model_config.n_embd,
            "n_head": model_config.n_head,
            "n_layer": model_config.n_layer,
            "latent_total": model_config.latent_total(),
            "latent_per_head": model_config.latent_per_head(),
            "sequence_kernel": model_config.sequence_kernel,
            "vocab_size": model_config.vocab_size,
            "random_scaffold": model_config.random_scaffold,
        }),
    );
    let checkpoint_format_hash = stable_content_id(
        "dragon-checkpoint-format",
        &serde_json::json!({
            "format": "named_mpk",
            "precision": "half",
            "chunk_size_bytes": 1024 * 1024,
            "dragon_schema_version": burn_dragon_core::DRAGON_CHECKPOINT_SCHEMA_VERSION,
        }),
    );
    let revision_family_hash = stable_content_id(
        "dragon-revision-family",
        &serde_json::json!({
            "experiment_kind": experiment_kind,
        }),
    );
    let supported_workload = SupportedWorkload {
        workload_id: workload_id.clone(),
        workload_name: format!("burn_dragon {}", experiment_kind.display_name()),
        model_program_hash,
        checkpoint_format_hash: checkpoint_format_hash.clone(),
        supported_revision_family: revision_family_hash,
        resource_class: backend_resource_class(backend_label),
    };
    let release_train_hash = stable_content_id(
        "dragon-release-train",
        &serde_json::json!({
            "project_family_id": seed.project_family_id,
            "experiment_kind": experiment_kind,
            "app_semver": app_semver,
        }),
    );
    let target_artifact_id = if backend_label.eq_ignore_ascii_case("cuda") {
        "native-cuda"
    } else if backend_label.eq_ignore_ascii_case("rocm") {
        "native-rocm"
    } else if backend_label.eq_ignore_ascii_case("wgpu") {
        "native-wgpu"
    } else {
        "native-cpu"
    };
    let target_platform = ClientPlatform::Native;
    let target_artifact_hash = dragon_target_artifact_hash(
        target_artifact_id,
        target_platform.clone(),
        &release_train_hash,
    );
    let release_manifest = ClientReleaseManifest {
        project_family_id: ProjectFamilyId::new(&seed.project_family_id),
        release_train_hash: release_train_hash.clone(),
        target_artifact_id: target_artifact_id.into(),
        target_artifact_hash: target_artifact_hash.clone(),
        target_platform,
        app_semver,
        git_commit: git_commit.into(),
        cargo_lock_hash: stable_content_id("dragon-cargo-lock", &"workspace"),
        burn_version_string: "0.21.0".into(),
        enabled_features_hash: stable_content_id("dragon-features", &enabled_features_label),
        protocol_major: seed.protocol_major,
        supported_workloads: vec![supported_workload.clone()],
        built_at: seed.release_built_at,
    };
    let network_manifest = NetworkManifest {
        network_id: NetworkId::new(&seed.network_id),
        project_family_id: release_manifest.project_family_id.clone(),
        protocol_major: seed.protocol_major,
        minimum_client_version: release_manifest.app_semver.clone(),
        required_release_train_hash: release_train_hash.clone(),
        allowed_target_artifact_hashes: [
            ("native-cpu", ClientPlatform::Native),
            ("native-wgpu", ClientPlatform::Native),
            ("native-cuda", ClientPlatform::Native),
            ("native-rocm", ClientPlatform::Native),
            ("browser-wasm", ClientPlatform::Browser),
        ]
        .into_iter()
        .map(|(target, platform)| {
            dragon_target_artifact_hash(target, platform, &release_train_hash)
        })
        .collect(),
        authority_public_keys: seed.authority_public_keys.clone(),
        bootstrap_addrs: seed.bootstrap_addrs.clone(),
        auth_policy_hash: stable_content_id("dragon-auth-policy", &seed.project_family_id),
        created_at: seed.created_at,
        description: seed.description.clone(),
    };
    let experiment_id = ExperimentId::new(&seed.experiment_id);
    let merge_topology_policy = dragon_merge_topology(experiment_kind, &seed.promotion)?;
    let (training_contract, training_contract_id) =
        dragon_training_contract(DragonTrainingContractInput {
            experiment_kind,
            model_config,
            training_config,
            dataset_view_id: dataset_view_id.clone(),
            checkpoint_format_hash: checkpoint_format_hash.clone(),
            merge_topology_policy: &merge_topology_policy,
            root_ema_update_basis_points: seed.aggregation.root_ema_update_basis_points,
            browser_profile: profile.browser.as_ref(),
            random_scaffold_catalog,
            random_scaffold_update_encoding: seed.random_scaffold_update_encoding,
            training_protocol: &seed.training_protocol,
        })?;
    let resource_requirements = ExperimentResourceRequirements {
        minimum_roles: BTreeSet::new(),
        // Runtime capability advertisements decide whether a concrete peer can
        // train. The authority-signed revision must remain identical across
        // CPU, GPU, and browser release artifacts.
        minimum_device_memory_bytes: None,
        minimum_system_memory_bytes: Some(canonical_minimum_system_memory_bytes(footprint)),
        estimated_download_bytes: footprint
            .estimated_checkpoint_bytes
            .saturating_add(footprint.estimated_shard_bytes),
        estimated_window_seconds: 30,
    };
    let browser_trainer_wgpu = browser_trainer_wgpu_enabled(profile, footprint);
    let mut allowed_role_values = vec![
        PeerRole::TrainerCpu,
        PeerRole::TrainerGpu,
        PeerRole::Validator,
        PeerRole::Evaluator,
        PeerRole::Archive,
        PeerRole::Viewer,
    ];
    if profile.browser.is_some() {
        allowed_role_values.push(PeerRole::BrowserObserver);
        allowed_role_values.push(PeerRole::BrowserVerifier);
    }
    if browser_trainer_wgpu {
        allowed_role_values.push(PeerRole::BrowserTrainerWgpu);
    }
    let allowed_roles = PeerRoleSet::new(allowed_role_values);
    let allowed_scopes = BTreeSet::from([
        ExperimentScope::Connect,
        ExperimentScope::Discover,
        ExperimentScope::Train {
            experiment_id: experiment_id.clone(),
        },
        ExperimentScope::Archive {
            experiment_id: experiment_id.clone(),
        },
        ExperimentScope::Validate {
            experiment_id: experiment_id.clone(),
        },
    ]);
    let mut metadata = BTreeMap::from([
        (
            "experiment_kind".into(),
            experiment_kind.workload_slug().into(),
        ),
        (
            "estimated_training_bytes".into(),
            footprint.estimated_training_bytes.to_string(),
        ),
        (
            "estimated_checkpoint_bytes".into(),
            footprint.estimated_checkpoint_bytes.to_string(),
        ),
        (
            "estimated_shard_bytes".into(),
            footprint.estimated_shard_bytes.to_string(),
        ),
        (
            "estimated_tokens_per_second".into(),
            format!("{:.1}", footprint.estimated_tokens_per_second),
        ),
        (
            "root_ema_update_basis_points".into(),
            seed.aggregation.root_ema_update_basis_points.to_string(),
        ),
        (
            "promotion_mode".into(),
            match seed.promotion.mode {
                DragonPromotionMode::DiffusionSteadyState => "diffusion_steady_state",
                DragonPromotionMode::ValidatorQuorum => "validator_quorum",
            }
            .into(),
        ),
        (
            "validator_quorum".into(),
            merge_topology_policy
                .promotion_policy
                .validator_quorum
                .to_string(),
        ),
        (
            "training_protocol".into(),
            match &seed.training_protocol {
                TrainingProtocol::ArtifactWindows => "artifact_windows",
                TrainingProtocol::DiLoCo(_) => "diloco",
            }
            .into(),
        ),
    ]);
    if let TrainingProtocol::DiLoCo(policy) = &seed.training_protocol {
        metadata.insert(
            "diloco_num_inner_steps".into(),
            policy.num_inner_steps.to_string(),
        );
        metadata.insert(
            "diloco_target_group_size".into(),
            policy.target_group_size.to_string(),
        );
        metadata.insert(
            "diloco_minimum_group_size".into(),
            policy.minimum_group_size.to_string(),
        );
    }
    let mut experiment_directory_entry = ExperimentDirectoryEntry {
        network_id: network_manifest.network_id.clone(),
        study_id: StudyId::new(&seed.study_id),
        experiment_id: experiment_id.clone(),
        workload_id: workload_id.clone(),
        display_name: seed.display_name.clone(),
        model_schema_hash: stable_content_id("dragon-model-schema", &model_config),
        dataset_view_id,
        resource_requirements,
        visibility: ExperimentVisibility::Public,
        opt_in_policy: ExperimentOptInPolicy::Open,
        current_revision_id: RevisionId::new(&seed.revision_id),
        current_head_id: None,
        allowed_roles,
        allowed_scopes,
        training_protocol: seed.training_protocol.clone(),
        metadata,
    };
    profile.attach_to_entry(&mut experiment_directory_entry)?;
    let mut robustness_policy = dragon_robustness_policy(experiment_kind);
    robustness_policy
        .validator_canary_policy
        .minimum_evaluator_quorum = merge_topology_policy.promotion_policy.validator_quorum;
    let revision_manifest = RevisionManifest {
        experiment_id: experiment_id.clone(),
        revision_id: RevisionId::new(&seed.revision_id),
        workload_id: workload_id.clone(),
        required_release_train_hash: release_manifest.release_train_hash.clone(),
        model_schema_hash: experiment_directory_entry.model_schema_hash.clone(),
        checkpoint_format_hash: checkpoint_format_hash.clone(),
        dataset_view_id: experiment_directory_entry.dataset_view_id.clone(),
        training_config_hash: training_contract_id.clone(),
        merge_topology_policy_hash: stable_content_id(
            "dragon-merge-topology",
            &merge_topology_policy,
        ),
        training_protocol: seed.training_protocol.clone(),
        slot_requirements: experiment_directory_entry.resource_requirements.clone(),
        activation_window: WindowActivation {
            activation_window: WindowId(0),
            grace_windows: 0,
        },
        lag_policy: Default::default(),
        merge_window_miss_policy: Default::default(),
        robustness_policy: Some(robustness_policy),
        browser_enabled: profile.browser.is_some(),
        browser_role_policy: BrowserRolePolicy {
            observer: true,
            verifier: profile.browser.is_some(),
            trainer_wgpu: browser_trainer_wgpu,
            fallback: true,
        },
        max_browser_checkpoint_bytes: Some(footprint.estimated_checkpoint_bytes),
        max_browser_window_secs: Some(30),
        max_browser_shard_bytes: Some(footprint.estimated_shard_bytes),
        requires_webgpu: true,
        max_browser_batch_size: Some(8),
        recommended_browser_precision: Some(Precision::Fp16),
        visibility_policy: BrowserVisibilityPolicy::SwarmEligible,
        description: seed.description.clone(),
    };
    experiment_directory_entry.apply_revision_policy(&revision_manifest);
    experiment_directory_entry.metadata.insert(
        "burn_p2p.revision.merge_topology.policy_json".into(),
        serde_json::to_string(&merge_topology_policy)
            .expect("dragon diffusion merge topology should serialize"),
    );
    let experiment_directory = vec![experiment_directory_entry];
    let workload_config = BurnWorkloadConfig::new(
        supported_workload.clone(),
        BurnArtifactConfig::named_mpk(BurnRecordPrecision::Half, ChunkingScheme::new(1024 * 1024)?),
    )
    .with_model_schema_hash(training_contract.model_schema_hash.clone())
    .with_diloco_parameter_subset(random_scaffold_catalog.cloned())
    .with_root_ema(root_ema_update_weight);
    Ok(DragonManifestBundle {
        release_manifest,
        network_manifest,
        revision_manifest,
        supported_workload,
        experiment_directory,
        workload_config,
        training_contract,
        training_contract_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_p2p::burn::BurnMergeConfig;
    use burn_p2p::{DiLoCoPolicy, GradientCodec, OuterOptimizerPolicy};
    use semver::Version;

    fn seed() -> DragonManifestSeed {
        DragonManifestSeed {
            project_family_id: "dragon-family".into(),
            network_id: "dragon-net".into(),
            study_id: "dragon-study".into(),
            experiment_id: "dragon-exp".into(),
            revision_id: "r1".into(),
            display_name: "dragon".into(),
            description: "dragon".into(),
            protocol_major: 0,
            authority_public_keys: Vec::new(),
            bootstrap_addrs: Vec::new(),
            ..DragonManifestSeed::default()
        }
    }

    fn local_training_contract(training_config: &TrainingConfig) -> TrainingContractManifest {
        let model_config = DragonConfig::default();
        let merge_topology =
            dragon_diffusion_merge_topology(DragonExperimentKind::NcaPrepretraining);
        dragon_training_contract(DragonTrainingContractInput {
            experiment_kind: DragonExperimentKind::NcaPrepretraining,
            model_config: &model_config,
            training_config: Some(training_config),
            dataset_view_id: DatasetViewId::new("dataset"),
            checkpoint_format_hash: ContentId::new("checkpoint"),
            merge_topology_policy: &merge_topology,
            root_ema_update_basis_points: 10_000,
            browser_profile: None,
            random_scaffold_catalog: None,
            random_scaffold_update_encoding: burn_p2p::CompactScalarEncoding::Fp32,
            training_protocol: &TrainingProtocol::ArtifactWindows,
        })
        .expect("training contract")
        .0
    }

    fn local_pc_training_config() -> TrainingConfig {
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let mut config = burn_dragon_language::load_training_config(&[manifest_dir
            .join("../../config/language/experiments/predictive_coding/local-pc-1m.toml")])
        .expect("load local-PC training profile");
        config.dataset.source = DatasetSourceConfig::NemotronClimbMix {
            revision: Some("contract-test".into()),
            max_records: Some(1),
        };
        config
    }

    #[test]
    fn local_pc_and_temporal_credit_semantics_are_contract_bound() {
        let baseline_config = local_pc_training_config();
        let baseline = local_training_contract(&baseline_config);
        assert_eq!(
            baseline.recurrent_state_policy,
            RecurrentStatePolicy::Ephemeral
        );
        assert!(
            baseline
                .extensions
                .contains_key(DRAGON_LOCAL_PC_PROGRAM_CONTRACT_EXTENSION)
        );

        let mut solver_drift = baseline_config.clone();
        solver_drift
            .training
            .local_predictive_coding
            .inference
            .steps += 1;
        let solver_drift = local_training_contract(&solver_drift);
        assert_ne!(baseline.objective_hash, solver_drift.objective_hash);
        assert_ne!(
            baseline
                .extensions
                .get(DRAGON_LOCAL_PC_PROGRAM_CONTRACT_EXTENSION),
            solver_drift
                .extensions
                .get(DRAGON_LOCAL_PC_PROGRAM_CONTRACT_EXTENSION)
        );
        assert_ne!(
            baseline.contract_id().expect("baseline contract"),
            solver_drift.contract_id().expect("solver-drift contract")
        );

        let mut temporal_drift = baseline_config.clone();
        temporal_drift.training.tbptt_chunk_size = Some(64);
        temporal_drift.training.tbptt_credit_window_chunks = 2;
        temporal_drift.training.tbptt_persist_across_steps = true;
        temporal_drift.training.sequence_batching =
            burn_dragon_language::config::SequenceBatchingMode::Streaming;
        let temporal_drift = local_training_contract(&temporal_drift);
        assert_ne!(
            baseline.preprocessing_hash,
            temporal_drift.preprocessing_hash
        );
        assert_eq!(
            temporal_drift.recurrent_state_policy,
            RecurrentStatePolicy::LeaseScoped
        );
        assert_ne!(
            baseline.contract_id().expect("baseline contract"),
            temporal_drift
                .contract_id()
                .expect("temporal-drift contract")
        );

        let mut hardware_local_batch = baseline_config;
        hardware_local_batch.training.batch_size =
            hardware_local_batch.training.batch_size.saturating_add(7);
        let hardware_local_batch = local_training_contract(&hardware_local_batch);
        assert_eq!(
            baseline.contract_id().expect("baseline contract"),
            hardware_local_batch
                .contract_id()
                .expect("hardware-local batch contract"),
            "peer-local batch calibration must remain outside semantic revision identity"
        );
    }

    #[test]
    fn ruliad_contract_binds_content_not_local_config_path() {
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let relative = manifest_dir.join("deploy/profiles/../profiles/ruliad-r3.corpus.toml");
        let absolute = relative.canonicalize().expect("canonical profile path");
        let mut relative_config = burn_dragon_language::load_training_config(&[
            manifest_dir.join("deploy/profiles/ruliad-r3.training.toml")
        ])
        .expect("load ruliad training profile");
        relative_config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: relative.clone(),
        };
        let mut absolute_config = relative_config.clone();
        absolute_config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: absolute.clone(),
        };
        let relative_contract = local_training_contract(&relative_config);
        let absolute_contract = local_training_contract(&absolute_config);
        assert_eq!(
            relative_contract
                .extensions
                .get(DRAGON_RULIAD_SEMANTIC_CONTRACT_EXTENSION),
            absolute_contract
                .extensions
                .get(DRAGON_RULIAD_SEMANTIC_CONTRACT_EXTENSION)
        );
        assert_eq!(
            relative_contract.preprocessing_hash,
            absolute_contract.preprocessing_hash
        );
        assert_eq!(
            relative_contract.contract_id().expect("relative id"),
            absolute_contract.contract_id().expect("absolute id")
        );

        let dir = tempfile::tempdir().expect("tempdir");
        let changed_path = dir.path().join("changed-ruliad.toml");
        let mut changed =
            burn_dragon_universality::load_ruliad_config(&relative).expect("load ruliad profile");
        changed.seed = changed.seed.wrapping_add(1);
        std::fs::write(
            &changed_path,
            toml::to_string_pretty(&changed).expect("changed config toml"),
        )
        .expect("write changed config");
        let mut changed_config = relative_config;
        changed_config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: changed_path,
        };
        let changed_contract = local_training_contract(&changed_config);
        assert_ne!(
            relative_contract
                .extensions
                .get(DRAGON_RULIAD_SEMANTIC_CONTRACT_EXTENSION),
            changed_contract
                .extensions
                .get(DRAGON_RULIAD_SEMANTIC_CONTRACT_EXTENSION)
        );
        assert_ne!(
            relative_contract.contract_id().expect("relative id"),
            changed_contract.contract_id().expect("changed id")
        );

        let task_mix_path = dir.path().join("changed-ruliad-task-mix.toml");
        let mut changed_task_mix =
            burn_dragon_universality::load_ruliad_config(&absolute).expect("load Ruliad profile");
        changed_task_mix
            .source_selection
            .formal_task_mix
            .advance_proof_weight += 1;
        std::fs::write(
            &task_mix_path,
            toml::to_string_pretty(&changed_task_mix).expect("changed task mix TOML"),
        )
        .expect("write changed task mix");
        let mut task_mix_config = absolute_config;
        task_mix_config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: task_mix_path,
        };
        let task_mix_contract = local_training_contract(&task_mix_config);
        assert_ne!(
            absolute_contract
                .extensions
                .get(DRAGON_RULIAD_SEMANTIC_CONTRACT_EXTENSION),
            task_mix_contract
                .extensions
                .get(DRAGON_RULIAD_SEMANTIC_CONTRACT_EXTENSION)
        );
        assert_ne!(
            absolute_contract.contract_id().expect("absolute id"),
            task_mix_contract.contract_id().expect("task mix id")
        );
    }

    #[test]
    fn random_scaffold_revision_binds_mutable_catalog_and_wire_encoding() {
        let mut model_config = DragonConfig::default();
        model_config.random_scaffold.enabled = true;
        model_config.random_scaffold.seed = 23;
        model_config.random_scaffold.rank = 8;
        let catalog = ParameterSubsetCatalog::new(
            dragon_model_schema_hash(&model_config),
            vec![burn_p2p::ParameterSubsetEntry {
                path: "model.random_scaffold_adapters.fast.encoder.a".into(),
                shape: vec![4, 256, 8],
            }],
        );
        let merge_topology =
            dragon_diffusion_merge_topology(DragonExperimentKind::NcaPrepretraining);
        let (contract, _) = dragon_training_contract(DragonTrainingContractInput {
            experiment_kind: DragonExperimentKind::NcaPrepretraining,
            model_config: &model_config,
            training_config: None,
            dataset_view_id: DatasetViewId::new("dataset"),
            checkpoint_format_hash: ContentId::new("checkpoint"),
            merge_topology_policy: &merge_topology,
            root_ema_update_basis_points: 10_000,
            browser_profile: None,
            random_scaffold_catalog: Some(&catalog),
            random_scaffold_update_encoding: burn_p2p::CompactScalarEncoding::SymmetricInt16,
            training_protocol: &TrainingProtocol::ArtifactWindows,
        })
        .expect("random-scaffold contract");

        assert_eq!(
            contract.update_codec,
            UpdateCodec::MutableSubsetParameters {
                parameter_catalog_hash: catalog.catalog_id().expect("catalog id"),
                parameter_count: catalog.parameter_count().expect("parameter count"),
                encoding: burn_p2p::CompactScalarEncoding::SymmetricInt16,
            }
        );
        assert!(contract.extensions.contains_key("dragon_random_scaffold"));
        assert_eq!(
            contract.optimizer_state_policy,
            LocalOptimizerStatePolicy::ResetPerWindow
        );
        assert_eq!(
            contract.scheduler_state_policy,
            SchedulerStatePolicy::ResetPerWindow
        );

        let error = dragon_training_contract(DragonTrainingContractInput {
            experiment_kind: DragonExperimentKind::NcaPrepretraining,
            model_config: &model_config,
            training_config: None,
            dataset_view_id: DatasetViewId::new("dataset"),
            checkpoint_format_hash: ContentId::new("checkpoint"),
            merge_topology_policy: &merge_topology,
            root_ema_update_basis_points: 10_000,
            browser_profile: None,
            random_scaffold_catalog: None,
            random_scaffold_update_encoding: burn_p2p::CompactScalarEncoding::Fp32,
            training_protocol: &TrainingProtocol::ArtifactWindows,
        })
        .expect_err("missing catalog must fail");
        assert!(error.to_string().contains("requires a canonical"));
    }

    #[test]
    fn manifests_publish_backend_neutral_revision_requirements() {
        let model_config = DragonConfig::default();
        let footprint = DragonTrainingFootprint {
            estimated_parameter_bytes: 1024,
            estimated_optimizer_state_bytes: 2048,
            estimated_activation_bytes: 4096,
            estimated_training_bytes: 8192,
            estimated_checkpoint_bytes: 4096,
            estimated_shard_bytes: 2048,
            estimated_tokens_per_second: 1234.0,
        };
        let bundle = build_manifest_bundle(
            &seed(),
            DragonExperimentKind::NcaPrepretraining,
            "wgpu",
            &model_config,
            None,
            None,
            &DragonExperimentProfile {
                version: 1,
                experiment_kind: DragonExperimentKind::NcaPrepretraining,
                native: crate::profile::DragonNativeExperimentProfile {
                    training_toml: String::new(),
                    nca_corpus_toml: None,
                    ruliad_corpus_toml: None,
                },
                browser: None,
            },
            DatasetViewId::new("dataset-view"),
            &footprint,
            Version::parse(env!("CARGO_PKG_VERSION")).expect("valid burn_dragon version"),
            "test",
            "native,wgpu",
        )
        .expect("manifest bundle");
        let requirements = &bundle.experiment_directory[0].resource_requirements;
        assert_eq!(requirements.minimum_device_memory_bytes, None);
        assert_eq!(
            requirements.minimum_system_memory_bytes,
            Some(
                footprint
                    .estimated_checkpoint_bytes
                    .saturating_add(footprint.estimated_shard_bytes)
                    .max(512 * 1024 * 1024)
            )
        );
        assert_eq!(
            requirements.estimated_download_bytes,
            footprint
                .estimated_checkpoint_bytes
                .saturating_add(footprint.estimated_shard_bytes)
        );
        assert!(
            bundle.experiment_directory[0]
                .allowed_roles
                .contains(&PeerRole::TrainerCpu)
        );
        assert!(
            bundle.experiment_directory[0]
                .allowed_roles
                .contains(&PeerRole::TrainerGpu)
        );
    }

    #[test]
    fn manifest_seed_timestamps_are_stable_across_builds() {
        let model_config = DragonConfig::default();
        let footprint = DragonTrainingFootprint {
            estimated_parameter_bytes: 1024,
            estimated_optimizer_state_bytes: 2048,
            estimated_activation_bytes: 4096,
            estimated_training_bytes: 8192,
            estimated_checkpoint_bytes: 4096,
            estimated_shard_bytes: 2048,
            estimated_tokens_per_second: 1234.0,
        };
        let seed = seed();
        let bundle = build_manifest_bundle(
            &seed,
            DragonExperimentKind::NcaPrepretraining,
            "cpu",
            &model_config,
            None,
            None,
            &DragonExperimentProfile {
                version: 1,
                experiment_kind: DragonExperimentKind::NcaPrepretraining,
                native: crate::profile::DragonNativeExperimentProfile {
                    training_toml: String::new(),
                    nca_corpus_toml: None,
                    ruliad_corpus_toml: None,
                },
                browser: None,
            },
            DatasetViewId::new("dataset-view"),
            &footprint,
            Version::parse(env!("CARGO_PKG_VERSION")).expect("valid burn_dragon version"),
            "test",
            "native,cpu",
        )
        .expect("manifest bundle");

        assert_eq!(bundle.network_manifest.created_at, seed.created_at);
        assert_eq!(bundle.release_manifest.built_at, seed.release_built_at);
    }

    #[test]
    fn manifest_publishes_browser_trainer_when_profile_budget_fits() {
        let model_config = DragonConfig::default();
        let footprint = DragonTrainingFootprint {
            estimated_parameter_bytes: 1024,
            estimated_optimizer_state_bytes: 2048,
            estimated_activation_bytes: 4096,
            estimated_training_bytes: 8192,
            estimated_checkpoint_bytes: 4096,
            estimated_shard_bytes: 2048,
            estimated_tokens_per_second: 1234.0,
        };
        let capability_policy = crate::config::DragonCapabilityPolicy {
            browser_wgpu_memory_budget_bytes: Some(16_384),
            ..crate::config::DragonCapabilityPolicy::default()
        };
        let profile = DragonExperimentProfile {
            version: 1,
            experiment_kind: DragonExperimentKind::NcaPrepretraining,
            native: crate::profile::DragonNativeExperimentProfile {
                training_toml: String::new(),
                nca_corpus_toml: None,
                ruliad_corpus_toml: None,
            },
            browser: Some(crate::profile::DragonBrowserExperimentProfile {
                model_config: model_config.clone(),
                training_objective: Default::default(),
                optimizer: Default::default(),
                execution_backend: crate::config::DragonBrowserExecutionBackend::Auto,
                block_size: 8,
                tbptt_chunk_size: None,
                tbptt_persist_across_steps: false,
                learning_rate: 1.0e-3,
                weight_decay: 0.0,
                batch_size: 1,
                max_train_batches: Some(1),
                max_eval_batches: Some(1),
                capability_policy,
                trainer_support: Default::default(),
                train_source: crate::profile::DragonBrowserProfileTokenSource::Inline {
                    records: Vec::new(),
                },
                eval_source: None,
            }),
        };
        let bundle = build_manifest_bundle(
            &seed(),
            DragonExperimentKind::NcaPrepretraining,
            "cpu",
            &model_config,
            None,
            None,
            &profile,
            DatasetViewId::new("dataset-view"),
            &footprint,
            Version::parse(env!("CARGO_PKG_VERSION")).expect("valid burn_dragon version"),
            "test",
            "native,cpu",
        )
        .expect("manifest bundle");

        let entry = &bundle.experiment_directory[0];
        assert!(entry.allowed_roles.contains(&PeerRole::BrowserTrainerWgpu));
        assert!(entry.allowed_roles.contains(&PeerRole::BrowserVerifier));
        assert!(entry.browser_role_policy().trainer_wgpu);
        assert!(entry.browser_role_policy().verifier);
        assert_eq!(
            bundle
                .training_contract
                .extensions
                .get(DRAGON_BROWSER_EXECUTION_CONTRACT_EXTENSION),
            Some(
                &browser_profile_execution_contract_hash(
                    profile.experiment_kind,
                    profile.browser.as_ref().expect("browser profile"),
                )
                .expect("browser execution contract")
            ),
        );
        assert!(entry.allowed_scopes.contains(&ExperimentScope::Validate {
            experiment_id: entry.experiment_id.clone(),
        }));
        let browser_training =
            crate::profile::browser_training_config_from_profile(entry, &profile)
                .expect("browser training profile")
                .expect("browser trainer should be configured");
        let live = browser_training
            .live_participant
            .expect("browser live participant config");
        assert!(live.publish_canonical_update);
        assert!(live.load_active_head_artifact);

        let mut observer_profile = profile.clone();
        observer_profile
            .browser
            .as_mut()
            .expect("browser profile")
            .trainer_support = crate::profile::DragonBrowserTrainerSupport::ObserverOnly {
            reason: "native-only auxiliary objective".into(),
        };
        let observer_bundle = build_manifest_bundle(
            &seed(),
            DragonExperimentKind::NcaPrepretraining,
            "cpu",
            &model_config,
            None,
            None,
            &observer_profile,
            DatasetViewId::new("dataset-view"),
            &footprint,
            Version::parse(env!("CARGO_PKG_VERSION")).expect("valid burn_dragon version"),
            "test",
            "native,cpu",
        )
        .expect("observer-only manifest bundle");
        let observer_entry = &observer_bundle.experiment_directory[0];
        assert!(
            !observer_entry
                .allowed_roles
                .contains(&PeerRole::BrowserTrainerWgpu)
        );
        assert!(
            observer_entry
                .allowed_roles
                .contains(&PeerRole::BrowserVerifier)
        );
        assert!(
            crate::profile::browser_training_config_from_profile(
                observer_entry,
                &observer_profile,
            )
            .expect("observer profile")
            .is_none()
        );
    }

    #[test]
    fn heterogeneous_backends_share_workload_revision_and_training_contract() {
        let model_config = DragonConfig::default();
        let footprint = DragonTrainingFootprint {
            estimated_parameter_bytes: 1024,
            estimated_optimizer_state_bytes: 2048,
            estimated_activation_bytes: 4096,
            estimated_training_bytes: 8192,
            estimated_checkpoint_bytes: 4096,
            estimated_shard_bytes: 2048,
            estimated_tokens_per_second: 1234.0,
        };
        let profile = DragonExperimentProfile {
            version: 1,
            experiment_kind: DragonExperimentKind::NcaPrepretraining,
            native: crate::profile::DragonNativeExperimentProfile {
                training_toml: String::new(),
                nca_corpus_toml: None,
                ruliad_corpus_toml: None,
            },
            browser: None,
        };
        let build = |backend| {
            build_manifest_bundle(
                &seed(),
                DragonExperimentKind::NcaPrepretraining,
                backend,
                &model_config,
                None,
                None,
                &profile,
                DatasetViewId::new("dataset-view"),
                &footprint,
                Version::parse(env!("CARGO_PKG_VERSION")).expect("version"),
                "test",
                backend,
            )
            .expect("manifest bundle")
        };

        let cpu = build("cpu");
        let wgpu = build("wgpu");
        assert_eq!(
            cpu.supported_workload.workload_id,
            wgpu.supported_workload.workload_id
        );
        assert_eq!(
            cpu.supported_workload.model_program_hash,
            wgpu.supported_workload.model_program_hash
        );
        assert_eq!(
            cpu.supported_workload.supported_revision_family,
            wgpu.supported_workload.supported_revision_family
        );
        assert_eq!(
            cpu.release_manifest.release_train_hash,
            wgpu.release_manifest.release_train_hash
        );
        assert_eq!(cpu.training_contract_id, wgpu.training_contract_id);
        assert_eq!(cpu.training_contract, wgpu.training_contract);
        assert_eq!(
            cpu.revision_manifest.training_config_hash,
            cpu.training_contract_id
        );
        assert_eq!(cpu.revision_manifest, wgpu.revision_manifest);
        assert_eq!(
            cpu.workload_config.model_schema_hash.as_ref(),
            Some(&cpu.training_contract.model_schema_hash),
        );
        assert_eq!(
            wgpu.workload_config.model_schema_hash.as_ref(),
            Some(&wgpu.training_contract.model_schema_hash),
        );
        assert_eq!(
            cpu.network_manifest.allowed_target_artifact_hashes,
            wgpu.network_manifest.allowed_target_artifact_hashes
        );
        assert_eq!(cpu.experiment_directory, wgpu.experiment_directory);
        assert_ne!(
            cpu.release_manifest.target_artifact_hash, wgpu.release_manifest.target_artifact_hash,
            "hardware builds remain distinct release artifacts"
        );
    }

    #[test]
    fn dedicated_validator_topology_is_revision_bound_and_fail_closed() {
        let promotion = DragonPromotionConfig {
            mode: DragonPromotionMode::ValidatorQuorum,
            validator_quorum: 3,
        };
        let topology = dragon_merge_topology(DragonExperimentKind::RuliadPretraining, &promotion)
            .expect("validator topology");

        assert_eq!(
            topology.strategy,
            MergeStrategy::MicrocohortReducePlusValidatorPromotion
        );
        assert_eq!(
            topology.promotion_policy.mode,
            HeadPromotionMode::ValidatorQuorum
        );
        assert_eq!(topology.promotion_policy.validator_quorum, 3);
        assert!(topology.promotion_policy.diffusion.is_none());
        assert!(
            dragon_merge_topology(
                DragonExperimentKind::RuliadPretraining,
                &DragonPromotionConfig {
                    mode: DragonPromotionMode::ValidatorQuorum,
                    validator_quorum: 0,
                },
            )
            .is_err()
        );
    }

    #[test]
    fn manifests_default_to_trainer_only_diffusion_topology() {
        let model_config = DragonConfig::default();
        let footprint = DragonTrainingFootprint {
            estimated_parameter_bytes: 1024,
            estimated_optimizer_state_bytes: 2048,
            estimated_activation_bytes: 4096,
            estimated_training_bytes: 8192,
            estimated_checkpoint_bytes: 4096,
            estimated_shard_bytes: 2048,
            estimated_tokens_per_second: 1234.0,
        };
        let bundle = build_manifest_bundle(
            &seed(),
            DragonExperimentKind::NcaPrepretraining,
            "cpu",
            &model_config,
            None,
            None,
            &DragonExperimentProfile {
                version: 1,
                experiment_kind: DragonExperimentKind::NcaPrepretraining,
                native: crate::profile::DragonNativeExperimentProfile {
                    training_toml: String::new(),
                    nca_corpus_toml: None,
                    ruliad_corpus_toml: None,
                },
                browser: None,
            },
            DatasetViewId::new("dataset-view"),
            &footprint,
            Version::parse(env!("CARGO_PKG_VERSION")).expect("valid burn_dragon version"),
            "test",
            "native,cpu",
        )
        .expect("manifest bundle");

        let entry = &bundle.experiment_directory[0];
        assert_eq!(entry.training_protocol(), TrainingProtocol::ArtifactWindows);
        assert_eq!(
            bundle.revision_manifest.training_protocol,
            TrainingProtocol::ArtifactWindows
        );
        assert_eq!(
            entry.metadata.get("training_protocol").map(String::as_str),
            Some("artifact_windows")
        );
        assert!(entry.allowed_roles.contains(&PeerRole::Validator));
        assert!(entry.allowed_roles.contains(&PeerRole::Evaluator));
        assert!(!entry.allowed_roles.contains(&PeerRole::BrowserVerifier));
        assert!(!entry.allowed_roles.contains(&PeerRole::BrowserTrainerWgpu));
        assert!(!entry.browser_role_policy().trainer_wgpu);
        assert!(entry.allowed_scopes.contains(&ExperimentScope::Validate {
            experiment_id: entry.experiment_id.clone(),
        }));

        let topology = entry
            .merge_topology_policy()
            .expect("diffusion merge topology");
        assert_eq!(topology.strategy, MergeStrategy::KRegularGossip);
        assert_eq!(
            topology.promotion_policy.mode,
            HeadPromotionMode::DiffusionSteadyState
        );
        assert_eq!(topology.promotion_policy.validator_quorum, 1);
        assert!(
            topology
                .promotion_policy
                .diffusion
                .as_ref()
                .is_some_and(|policy| policy.allow_solo_promotion)
        );
        assert_eq!(
            topology
                .promotion_policy
                .diffusion
                .as_ref()
                .expect("diffusion policy")
                .artifact_sync_timeout_secs,
            DRAGON_DIFFUSION_ARTIFACT_SYNC_TIMEOUT_SECS
        );

        let robustness = entry.robustness_policy().expect("robustness policy");
        assert_eq!(
            robustness.validator_canary_policy.minimum_evaluator_quorum,
            topology.promotion_policy.validator_quorum
        );
        assert_eq!(
            robustness.validator_canary_policy.maximum_regression_delta,
            1.0
        );
    }

    #[test]
    fn diloco_protocol_is_validated_and_bound_to_directory_and_revision() {
        let model_config = DragonConfig::default();
        let footprint = DragonTrainingFootprint {
            estimated_parameter_bytes: 1024,
            estimated_optimizer_state_bytes: 2048,
            estimated_activation_bytes: 4096,
            estimated_training_bytes: 8192,
            estimated_checkpoint_bytes: 4096,
            estimated_shard_bytes: 2048,
            estimated_tokens_per_second: 1234.0,
        };
        let profile = DragonExperimentProfile {
            version: 1,
            experiment_kind: DragonExperimentKind::NcaPrepretraining,
            native: crate::profile::DragonNativeExperimentProfile {
                training_toml: String::new(),
                nca_corpus_toml: None,
                ruliad_corpus_toml: None,
            },
            browser: None,
        };
        let policy = DiLoCoPolicy {
            num_inner_steps: 7,
            target_group_size: 3,
            minimum_group_size: 2,
            checkpoint_interval_rounds: 4,
            codec: GradientCodec::Fp32,
            outer_optimizer_policy: OuterOptimizerPolicy::Sgd {
                learning_rate_micros: 750_000,
                momentum_micros: Some(500_000),
                nesterov: true,
                weight_decay_micros: None,
                max_pseudo_gradient_rms_ratio_micros: None,
            },
            ..DiLoCoPolicy::default()
        };
        let mut diloco_seed = seed();
        diloco_seed.training_protocol = TrainingProtocol::DiLoCo(policy.clone());

        let bundle = build_manifest_bundle(
            &diloco_seed,
            DragonExperimentKind::NcaPrepretraining,
            "cpu",
            &model_config,
            None,
            None,
            &profile,
            DatasetViewId::new("dataset-view"),
            &footprint,
            Version::parse(env!("CARGO_PKG_VERSION")).expect("valid burn_dragon version"),
            "test",
            "native,cpu",
        )
        .expect("DiLoCo manifest bundle");

        let expected = TrainingProtocol::DiLoCo(policy);
        let entry = &bundle.experiment_directory[0];
        assert_eq!(entry.training_protocol(), expected);
        assert_eq!(bundle.revision_manifest.training_protocol, expected);
        assert_eq!(
            bundle.training_contract.optimizer_state_policy,
            LocalOptimizerStatePolicy::PeerLocalPersistent
        );
        assert_eq!(
            bundle.training_contract.scheduler_state_policy,
            SchedulerStatePolicy::PeerLocalPersistent
        );
        assert_eq!(
            entry.metadata.get("training_protocol").map(String::as_str),
            Some("diloco")
        );
        assert_eq!(
            entry
                .metadata
                .get("diloco_num_inner_steps")
                .map(String::as_str),
            Some("7")
        );
        assert_eq!(
            entry
                .metadata
                .get("diloco_target_group_size")
                .map(String::as_str),
            Some("3")
        );
        assert_eq!(
            entry
                .metadata
                .get("diloco_minimum_group_size")
                .map(String::as_str),
            Some("2")
        );

        let mut invalid_seed = diloco_seed;
        let TrainingProtocol::DiLoCo(policy) = &mut invalid_seed.training_protocol else {
            unreachable!("test seed is DiLoCo");
        };
        policy.num_inner_steps = 0;
        let error = build_manifest_bundle(
            &invalid_seed,
            DragonExperimentKind::NcaPrepretraining,
            "cpu",
            &model_config,
            None,
            None,
            &profile,
            DatasetViewId::new("dataset-view"),
            &footprint,
            Version::parse(env!("CARGO_PKG_VERSION")).expect("valid burn_dragon version"),
            "test",
            "native,cpu",
        )
        .expect_err("invalid DiLoCo policy must fail at manifest construction");
        assert!(error.to_string().contains("num_inner_steps"));
    }

    #[test]
    fn aggregation_weight_is_revision_bound_and_matches_runtime_merge() {
        let model_config = DragonConfig::default();
        let footprint = DragonTrainingFootprint {
            estimated_parameter_bytes: 1024,
            estimated_optimizer_state_bytes: 2048,
            estimated_activation_bytes: 4096,
            estimated_training_bytes: 8192,
            estimated_checkpoint_bytes: 4096,
            estimated_shard_bytes: 2048,
            estimated_tokens_per_second: 1234.0,
        };
        let profile = DragonExperimentProfile {
            version: 1,
            experiment_kind: DragonExperimentKind::NcaPrepretraining,
            native: crate::profile::DragonNativeExperimentProfile {
                training_toml: String::new(),
                nca_corpus_toml: None,
                ruliad_corpus_toml: None,
            },
            browser: None,
        };
        let build = |root_ema_update_basis_points| {
            let mut seed = seed();
            seed.aggregation.root_ema_update_basis_points = root_ema_update_basis_points;
            build_manifest_bundle(
                &seed,
                DragonExperimentKind::NcaPrepretraining,
                "cpu",
                &model_config,
                None,
                None,
                &profile,
                DatasetViewId::new("dataset-view"),
                &footprint,
                Version::parse(env!("CARGO_PKG_VERSION")).expect("version"),
                "test",
                "native,cpu",
            )
        };

        let smoothed = build(3_500).expect("smoothed manifest");
        let direct = build(10_000).expect("direct manifest");

        assert_ne!(
            smoothed.training_contract.aggregation_hash,
            direct.training_contract.aggregation_hash
        );
        assert_ne!(smoothed.training_contract_id, direct.training_contract_id);
        assert_eq!(
            smoothed.experiment_directory[0]
                .metadata
                .get("root_ema_update_basis_points")
                .map(String::as_str),
            Some("3500")
        );
        assert_eq!(
            direct.experiment_directory[0]
                .metadata
                .get("root_ema_update_basis_points")
                .map(String::as_str),
            Some("10000")
        );
        match smoothed.workload_config.merge {
            BurnMergeConfig::WeightedMeanWithRootEma { decay } => {
                assert!((decay - 0.35).abs() < f64::EPSILON);
            }
            _ => panic!("expected root-EMA merge"),
        }
        match direct.workload_config.merge {
            BurnMergeConfig::WeightedMeanWithRootEma { decay } => {
                assert!((decay - 1.0).abs() < f64::EPSILON);
            }
            _ => panic!("expected root-EMA merge"),
        }
    }

    #[test]
    fn aggregation_weight_rejects_values_above_one() {
        let model_config = DragonConfig::default();
        let footprint = DragonTrainingFootprint {
            estimated_parameter_bytes: 1024,
            estimated_optimizer_state_bytes: 2048,
            estimated_activation_bytes: 4096,
            estimated_training_bytes: 8192,
            estimated_checkpoint_bytes: 4096,
            estimated_shard_bytes: 2048,
            estimated_tokens_per_second: 1234.0,
        };
        let mut invalid_seed = seed();
        invalid_seed.aggregation.root_ema_update_basis_points = 10_001;
        let result = build_manifest_bundle(
            &invalid_seed,
            DragonExperimentKind::NcaPrepretraining,
            "cpu",
            &model_config,
            None,
            None,
            &DragonExperimentProfile {
                version: 1,
                experiment_kind: DragonExperimentKind::NcaPrepretraining,
                native: crate::profile::DragonNativeExperimentProfile {
                    training_toml: String::new(),
                    nca_corpus_toml: None,
                    ruliad_corpus_toml: None,
                },
                browser: None,
            },
            DatasetViewId::new("dataset-view"),
            &footprint,
            Version::parse(env!("CARGO_PKG_VERSION")).expect("version"),
            "test",
            "native,cpu",
        );

        let error = result.expect_err("invalid root-EMA weight should fail");
        assert!(error.to_string().contains("at most 10000 basis points"));
    }
}
