use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use burn::data::dataloader::batcher::Batcher;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData};
use burn::train::LearningComponentsMarker;
use burn_dragon_language::api::checkpoint::apply_init_checkpoint_to_language_core;
use burn_dragon_language::api::inference::build_model_config_with_tokenizer;
use burn_dragon_language::config::ValidationDatasetConfig;
use burn_dragon_language::dataset::{
    Dataset, DatasetSplit, RandomDataLoader, SequenceBatch, StreamingDataLoader,
    TokenSequenceDataset,
};
use burn_dragon_language::summary_event_mask_tensor;
use burn_dragon_language::tokenizer::{SharedTokenizer, Tokenizer};
use burn_dragon_language::train::schedule::{resolve_lr_scheduler, resolve_train_schedule};
use burn_dragon_language::train::steps::LanguageTrainModel;
use burn_dragon_language::train::utils::prepare_datasets;
use burn_dragon_language::train::{
    LanguageOptimizer, resolve_dragon_language_optimizer, validate_dragon_continual_backprop,
};
use burn_dragon_language::{
    DatasetConfig, DragonConfig, DragonModel, TrainingConfig, TrainingHyperparameters,
};
use burn_dragon_train::train::constants::ValidBackend;
use burn_dragon_train::train::metrics::{
    LanguageModelOutput, LanguageModelTrainItem, LossValue, ScalarValue,
};
use burn_dragon_train::train::pipeline::ResolvedLrScheduler;
use burn_p2p::burn::{
    BurnLearnerDataPipeline, BurnLearnerProject, BurnLearnerProjectBuilder, BurnTrainLoader,
    BurnValidationLoader, BurnWorkloadAdapter, connect, from_stateful_components,
    from_stateful_loaders,
};
use burn_p2p::{
    DatasetViewId, EvalSplit, GeneratedWorkloadInputProvider, LeaseDataPipeline,
    LeaseDataPipelineDescriptor, LeaseDataPipelineKind, MetricReport, MetricValue, NodeBuilder,
    PeerRole, PeerRoleSet, SelectedWorkloadProject, SingleWorkloadProjectFamily,
};
use burn_train::InferenceStep;
use burn_train::metric::{Adaptor, ItemLazy};

use crate::auth::compose_auth_config;
use crate::capability::{
    DragonCapabilityClass, DragonNativeCapabilityAssessment, DragonNativeTargetDecision,
    DragonTrainingFootprint, decide_native_target, estimate_language_training_footprint,
};
use crate::capability_state::{
    NativeDowngradeObservation, NativeDowngradeScope, apply_native_downgrade_state,
    clear_native_downgrade, load_matching_native_downgrade, persist_native_downgrade,
};
use crate::config::{
    DragonExistingShardDatasetConfig, DragonExperimentKind, DragonManifestBundle,
    DragonNativeAuthBundle, DragonNativePeerConfig, DragonShardExportConfig, TokenWindowRecord,
    dragon_model_schema_hash,
};
use crate::manifests::build_manifest_bundle;
use crate::profile::resolve_native_training_profile;
use crate::random_scaffold::{
    DragonRandomScaffoldP2pContract, apply_random_scaffold_update,
    dragon_random_scaffold_p2p_contract, load_random_scaffold_genesis, load_random_scaffold_head,
    materialize_random_scaffold_genesis, materialize_random_scaffold_head,
    materialize_random_scaffold_update, random_scaffold_genesis_materialization,
    validate_random_scaffold_update,
};

pub type DragonLearningComponents<B> =
    LearningComponentsMarker<B, ResolvedLrScheduler, LanguageTrainModel<B>, LanguageOptimizer<B>>;

type DragonLearnerProjectBuilder<B> = BurnLearnerProjectBuilder<DragonLearningComponents<B>>;
type DragonValidationSource<B> = (
    BurnValidationLoader<DragonLearningComponents<B>>,
    Arc<Dataset>,
);

pub type DragonProjectFamily<B> = SingleWorkloadProjectFamily<
    BurnWorkloadAdapter<BurnLearnerProject<DragonLearningComponents<B>>>,
>;

pub type DragonNodeBuilder<B> = NodeBuilder<SelectedWorkloadProject<DragonProjectFamily<B>>>;

fn attach_dragon_workload_update_applier<B>(
    builder: DragonLearnerProjectBuilder<B>,
    config: &TrainingConfig,
    random_scaffold: Option<&DragonRandomScaffoldP2pContract>,
) -> DragonLearnerProjectBuilder<B>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    if let Some(random_scaffold) = random_scaffold {
        let materialize_catalog = random_scaffold.catalog.clone();
        let apply_catalog = random_scaffold.catalog.clone();
        let validate_catalog = random_scaffold.catalog.clone();
        let materialize_genesis_contract = random_scaffold.clone();
        let load_genesis_contract = random_scaffold.clone();
        let materialize_head_contract = random_scaffold.clone();
        let load_head_contract = random_scaffold.clone();
        return builder
            .with_genesis_materializer(move |context| {
                materialize_random_scaffold_genesis::<B>(context, &materialize_genesis_contract)
            })
            .with_genesis_loader(move |model, context| {
                load_random_scaffold_genesis::<B>(model, context, &load_genesis_contract)
            })
            .with_model_artifact_materializer(
                move |model, artifact_kind, head_id, base_head_id, store, model_schema_hash| {
                    materialize_random_scaffold_head::<B>(
                        model,
                        artifact_kind,
                        head_id,
                        base_head_id,
                        store,
                        model_schema_hash,
                        &materialize_head_contract,
                    )
                },
            )
            .with_model_artifact_loader(
                move |model, descriptor, store, device, model_schema_hash| {
                    load_random_scaffold_head::<B>(
                        model,
                        descriptor,
                        store,
                        device,
                        model_schema_hash,
                        &load_head_contract,
                    )
                },
            )
            .with_workload_update_materializer(move |context| {
                materialize_random_scaffold_update::<B>(context, &materialize_catalog)
            })
            .with_workload_update_applier(
                move |base_model, descriptor, envelope, contract, store, _device| {
                    apply_random_scaffold_update::<B>(
                        base_model,
                        descriptor,
                        envelope,
                        contract,
                        store,
                        &apply_catalog,
                    )
                },
            )
            .with_workload_update_validator(move |base_model, context| {
                validate_random_scaffold_update::<B>(base_model, context, &validate_catalog)
            });
    }
    if config.optimizer.name != burn_dragon_train::OptimizerKind::Eggroll {
        return builder;
    }
    let eggroll = config.optimizer.effective_eggroll_config();
    let apply_eggroll = eggroll.clone();
    let builder = builder.with_workload_update_applier(
        move |mut base_model, descriptor, envelope, contract, store, _device| {
            let bytes = store.materialize_artifact_bytes(descriptor)?;
            let update = burn_p2p_workload::decode_compact_update(
                &bytes,
                &envelope.training_contract_id,
                contract,
            )?;
            base_model.model = crate::seeded_fitness::replay_dragon_seeded_fitness_update(
                base_model.model.clone(),
                &apply_eggroll,
                &contract.optimizer_hash,
                &update,
            )?;
            Ok(base_model)
        },
    );
    let tbptt_chunk_size = config.training.tbptt_chunk_size;
    let persist_across_steps = config.training.tbptt_persist_across_steps;
    builder.with_workload_update_validator(move |mut base_model, context| {
        let burn_p2p::WorkloadUpdateValidationContext {
            descriptor,
            update: envelope,
            contract,
            store,
            device,
            replay,
        } = context;
        let burn_p2p::UpdateCodec::SeededFitness {
            replay: replay_policy,
            ..
        } = &contract.update_codec
        else {
            anyhow::bail!("Dragon seeded-fitness validator received a different update codec");
        };
        let bytes = store.materialize_artifact_bytes(descriptor)?;
        let update = burn_p2p_workload::decode_compact_update(
            &bytes,
            &envelope.training_contract_id,
            contract,
        )?;
        let records =
            crate::seeded_fitness::load_replay_token_window_records(replay.cached_microshards)?;
        let (model, replay_stats) =
            crate::seeded_fitness::validate_and_replay_dragon_seeded_fitness_update(
                base_model.model.clone(),
                &eggroll,
                &contract.optimizer_hash,
                &update,
                records,
                replay_policy,
                tbptt_chunk_size,
                persist_across_steps,
                device,
            )?;
        base_model.model = model;
        Ok(burn_p2p::ValidatedWorkloadUpdate {
            model: base_model,
            evidence: burn_p2p::ValidatedUpdateEvidence {
                update_envelope_id: burn_p2p::ContentId::derive(envelope)?,
                norm_stats: None,
                feature_sketch: None,
                reconstruction_verified: true,
                replay_verified: true,
                replay_stats: Some(replay_stats),
                validator_peer_id: replay.validator_peer_id.clone(),
                validated_at: chrono::Utc::now(),
            },
        })
    })
}
pub type DragonBurnProject<B> = BurnLearnerProject<DragonLearningComponents<B>>;

#[derive(Clone)]
pub struct PreparedNativePeer<B>
where
    B: AutodiffBackend + Clone + 'static,
{
    pub project: DragonBurnProject<B>,
    pub builder: DragonNodeBuilder<B>,
    pub manifests: DragonManifestBundle,
    pub config: TrainingConfig,
    pub storage_root: PathBuf,
    pub experiment_kind: DragonExperimentKind,
    pub backend_label: String,
    pub model_config: DragonConfig,
    pub footprint: DragonTrainingFootprint,
    pub target_decision: DragonNativeTargetDecision,
    pub capability_reprobe_policy: crate::config::DragonNativeCapabilityReprobePolicy,
    pub genesis_materialization: burn_p2p::GenesisMaterialization,
}

impl<B> PreparedNativePeer<B>
where
    B: AutodiffBackend + Clone + 'static,
{
    fn runtime_downgrade_target(&self) -> &'static str {
        match self.target_decision.effective_target {
            crate::config::DragonNativeTarget::Reducer => "reducer",
            crate::config::DragonNativeTarget::Validator => "validator",
            crate::config::DragonNativeTarget::Auto
            | crate::config::DragonNativeTarget::Trainer => "observer",
        }
    }

    fn downgrade_scope(&self) -> NativeDowngradeScope<'_, DragonConfig> {
        NativeDowngradeScope {
            storage_root: &self.storage_root,
            experiment_kind: self.experiment_kind,
            backend_label: &self.backend_label,
            model_config: &self.model_config,
            batch_size: self.config.training.batch_size,
            block_size: self.config.training.block_size,
        }
    }

    pub fn persist_runtime_training_failure_with_source(
        &self,
        reason: &str,
        source: &str,
    ) -> Result<()> {
        let _ = persist_native_downgrade(
            self.downgrade_scope(),
            NativeDowngradeObservation {
                footprint: &self.footprint,
                trainer_budget_bytes: self.target_decision.trainer_memory_budget_bytes,
                downgrade_to: self.runtime_downgrade_target(),
                reason,
                source,
            },
        )?;
        Ok(())
    }

    pub fn record_runtime_training_failure(&self, reason: &str) -> Result<()> {
        self.persist_runtime_training_failure_with_source(reason, "runtime")?;
        Ok(())
    }

    pub fn clear_runtime_downgrade(&self) -> Result<()> {
        clear_native_downgrade(self.downgrade_scope())
    }

    pub fn runtime_downgrade_failure_count(&self) -> Result<u32> {
        Ok(load_matching_native_downgrade(
            self.downgrade_scope(),
            self.target_decision.trainer_memory_budget_bytes,
        )?
        .map(|record| record.failure_count)
        .unwrap_or_default())
    }
}

#[derive(Clone, Debug)]
pub struct TokenWindowBatcher {
    summary_event_token_ids: Option<Vec<u32>>,
}

impl TokenWindowBatcher {
    pub fn new(summary_event_token_ids: Option<Vec<u32>>) -> Self {
        Self {
            summary_event_token_ids,
        }
    }
}

#[derive(Clone, Debug)]
struct DragonGeneratedInputDescriptor {
    provider: &'static str,
    metadata: BTreeMap<String, String>,
}

impl GeneratedWorkloadInputProvider for DragonGeneratedInputDescriptor {
    fn provider_id(&self) -> String {
        self.provider.into()
    }

    fn metadata(&self) -> BTreeMap<String, String> {
        self.metadata.clone()
    }
}

fn trim_http_base(base_url: &str) -> String {
    base_url.trim_end_matches('/').to_owned()
}

fn dragon_sharded_input_descriptor(
    experiment_kind: DragonExperimentKind,
    dataset_source: &burn_dragon_language::DatasetSourceConfig,
    registration: &burn_p2p::DatasetRegistration,
    shard_count: usize,
    http_upstream: Option<&str>,
) -> LeaseDataPipelineDescriptor {
    let mut descriptor = LeaseDataPipelineDescriptor::new(
        format!("dragon-{}-shards", experiment_kind.workload_slug()),
        LeaseDataPipelineKind::ShardedStatic,
    )
    .with_metadata_entry("experiment_kind", experiment_kind.workload_slug())
    .with_metadata_entry("dataset_id", registration.manifest.dataset_id.as_str())
    .with_metadata_entry(
        "dataset_view_id",
        registration.view.dataset_view_id.as_str(),
    )
    .with_metadata_entry("source_uri", registration.manifest.source_uri.clone())
    .with_metadata_entry("format", registration.manifest.format.clone());

    if let Some(base_url) = http_upstream {
        return descriptor
            .with_shard_manifest_http_source(
                format!("{}/fetch-manifest.json", trim_http_base(base_url)),
                Some(shard_count as u64),
            )
            .with_metadata_entry("upstream", "http");
    }

    descriptor = descriptor.with_metadata_entry("upstream", "local");
    match dataset_source {
        burn_dragon_language::DatasetSourceConfig::UniversalityNca { config } => {
            let provider = DragonGeneratedInputDescriptor {
                provider: "burn_dragon_universality_nca",
                metadata: BTreeMap::from([
                    ("config_path".into(), config.display().to_string()),
                    (
                        "experiment_kind".into(),
                        experiment_kind.workload_slug().into(),
                    ),
                ]),
            };
            descriptor.with_generated_input_source(&provider)
        }
        burn_dragon_language::DatasetSourceConfig::UniversalityRuliad { config } => {
            let provider = DragonGeneratedInputDescriptor {
                provider: "burn_dragon_universality_ruliad",
                metadata: BTreeMap::from([
                    ("config_path".into(), config.display().to_string()),
                    (
                        "experiment_kind".into(),
                        experiment_kind.workload_slug().into(),
                    ),
                ]),
            };
            descriptor.with_generated_input_source(&provider)
        }
        burn_dragon_language::DatasetSourceConfig::UniversalityManifest { manifest } => descriptor
            .with_custom_input_source(
                "universality-manifest",
                BTreeMap::from([
                    ("manifest_path".into(), manifest.display().to_string()),
                    (
                        "experiment_kind".into(),
                        experiment_kind.workload_slug().into(),
                    ),
                ]),
            ),
        burn_dragon_language::DatasetSourceConfig::NemotronClimbMix {
            revision,
            max_records,
        } => {
            let mut metadata = BTreeMap::from([(
                "experiment_kind".into(),
                experiment_kind.workload_slug().into(),
            )]);
            if let Some(revision) = revision {
                metadata.insert("revision".into(), revision.clone());
            }
            if let Some(max_records) = max_records {
                metadata.insert("max_records".into(), max_records.to_string());
            }
            descriptor.with_custom_input_source("nemotron-climbmix", metadata)
        }
    }
}

fn dragon_sharded_data_pipeline<B>(
    descriptor: LeaseDataPipelineDescriptor,
    dataset: burn_p2p::burn::BurnShardedDataset<TokenWindowRecord>,
    batcher: TokenWindowBatcher,
    batch_size: usize,
    max_train_batches: usize,
) -> BurnLearnerDataPipeline<DragonLearningComponents<B>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let registration = dataset.registration().clone();
    let microshard_plan = dataset.microshard_plan().clone();
    LeaseDataPipeline::new(
        descriptor,
        move || Ok(registration.clone()),
        move |_registration| Ok(microshard_plan.clone()),
        move |lease, cached_microshards, device| {
            let records = dataset.load_records(cached_microshards)?;
            batcher.stream_aligned_batches::<B>(
                records,
                batch_size,
                Some(max_train_batches),
                Some(lease.window_id.0),
                device,
            )
        },
    )
}

impl TokenWindowBatcher {
    fn batch_items<B: Backend>(
        &self,
        items: Vec<TokenWindowRecord>,
        reset_stream_state: bool,
        device: &B::Device,
    ) -> SequenceBatch<B> {
        let batch_size = items.len().max(1);
        let block_size = items
            .first()
            .map(|item| item.inputs.len())
            .unwrap_or_default()
            .max(1);
        let mut inputs = Vec::with_capacity(batch_size * block_size);
        let mut targets = Vec::with_capacity(batch_size * block_size);
        let mut loss_mask = items
            .iter()
            .any(|item| item.loss_mask.is_some())
            .then(|| Vec::with_capacity(batch_size * block_size));
        for item in items {
            assert_eq!(
                item.inputs.len(),
                block_size,
                "token-window input lengths must match within one stream batch"
            );
            assert_eq!(
                item.targets.len(),
                block_size,
                "token-window target lengths must match within one stream batch"
            );
            if let Some(item_loss_mask) = item.loss_mask.as_ref() {
                assert_eq!(
                    item_loss_mask.len(),
                    block_size,
                    "token-window loss-mask lengths must match inputs and targets"
                );
            }
            if let Some(batch_loss_mask) = loss_mask.as_mut() {
                if let Some(item_loss_mask) = item.loss_mask.as_ref() {
                    batch_loss_mask.extend(item_loss_mask);
                } else {
                    batch_loss_mask.extend(std::iter::repeat_n(1, block_size));
                }
            }
            inputs.extend(item.inputs);
            targets.extend(item.targets);
        }
        let summary_event_mask = summary_event_mask_tensor::<B>(
            &inputs,
            batch_size,
            block_size,
            self.summary_event_token_ids.as_deref(),
            device,
        );
        SequenceBatch::<B> {
            inputs: Tensor::<B, 2, Int>::from_data(
                TensorData::new(inputs, [batch_size, block_size]),
                device,
            ),
            targets: Tensor::<B, 2, Int>::from_data(
                TensorData::new(targets, [batch_size, block_size]),
                device,
            ),
            loss_mask: loss_mask.map(|loss_mask| {
                Tensor::<B, 2, Int>::from_data(
                    TensorData::new(loss_mask, [batch_size, block_size]),
                    device,
                )
            }),
            summary_event_mask,
            ruliad_policy_batch: None,
            reset_stream_state,
        }
    }

    fn stream_aligned_batches<B: Backend>(
        &self,
        records: Vec<TokenWindowRecord>,
        batch_size: usize,
        max_batches: Option<usize>,
        window_id: Option<u64>,
        device: &B::Device,
    ) -> Result<Vec<SequenceBatch<B>>> {
        let plan = crate::stream_batch::plan_windowed_stream_batches(
            &records,
            batch_size,
            max_batches,
            window_id,
        )?;
        let mut batches = Vec::with_capacity(plan.len());
        for planned in plan {
            let items = planned
                .record_indices
                .iter()
                .map(|index| records[*index].clone())
                .collect();
            batches.push(self.batch_items::<B>(items, planned.reset_stream_state, device));
        }
        Ok(batches)
    }
}

impl<B: Backend> Batcher<B, TokenWindowRecord, SequenceBatch<B>> for TokenWindowBatcher {
    fn batch(&self, items: Vec<TokenWindowRecord>, device: &B::Device) -> SequenceBatch<B> {
        let reset_stream_state = items
            .first()
            .map(|first| {
                if items
                    .iter()
                    .all(|item| item.reset_stream_state == first.reset_stream_state)
                {
                    first.reset_stream_state
                } else {
                    true
                }
            })
            .unwrap_or(true);
        self.batch_items::<B>(items, reset_stream_state, device)
    }
}

fn inline_dataset_view_id(dataset: &Dataset) -> Result<DatasetViewId> {
    DatasetViewId::derive(&(
        "burn-dragon-p2p-inline-dataset-view",
        dataset.block_size(),
        dataset.batch_size(),
        dataset.token_count(),
        dataset.train_split_ratio().to_bits(),
    ))
    .map_err(Into::into)
}

fn summary_event_token_ids(dataset: &Arc<Dataset>) -> Option<Vec<u32>> {
    summary_event_token_ids_for_tokenizer(dataset.tokenizer().as_ref())
}

fn summary_event_token_ids_for_tokenizer(tokenizer: &dyn Tokenizer) -> Option<Vec<u32>> {
    let ids = [tokenizer.bos_id(), tokenizer.eos_id()]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    (!ids.is_empty()).then_some(ids)
}

fn validation_dataset_config_for(
    dataset_cfg: &DatasetConfig,
    validation_cfg: &ValidationDatasetConfig,
) -> DatasetConfig {
    DatasetConfig {
        cache_dir: validation_cfg
            .cache_dir
            .clone()
            .unwrap_or_else(|| dataset_cfg.cache_dir.join("validation")),
        train_split_ratio: validation_cfg
            .train_split_ratio
            .unwrap_or(dataset_cfg.train_split_ratio),
        validation: None,
        source: validation_cfg.source.clone(),
        tokenizer: dataset_cfg.tokenizer.clone(),
    }
}

fn load_tokenizer_without_dataset(config: &TrainingConfig) -> Result<SharedTokenizer> {
    let tokenizer_cfg = &config.dataset.tokenizer;
    match tokenizer_cfg.storage_path(&config.dataset.cache_dir) {
        Some(path) if path.is_file() => tokenizer_cfg.load(&path),
        Some(path) => bail!(
            "shard-first p2p setup requires a persisted tokenizer at {}",
            path.display()
        ),
        None => tokenizer_cfg.fit(std::iter::empty::<&str>()),
    }
}

fn resolve_model_config_for_capability(config: &TrainingConfig) -> Result<DragonConfig> {
    let tokenizer = match load_tokenizer_without_dataset(config) {
        Ok(tokenizer) => tokenizer,
        Err(_) => prepare_datasets(&config.dataset, &config.training)?
            .train
            .tokenizer()
            .clone(),
    };
    build_model_config_with_tokenizer(
        &config.model,
        config.training.block_size,
        tokenizer.as_ref(),
    )
}

fn assess_loaded_native_training_config(
    config: &TrainingConfig,
    requested_target: crate::config::DragonNativeTarget,
    experiment_kind: DragonExperimentKind,
    backend_label: &str,
    capability_policy: &crate::config::DragonCapabilityPolicy,
) -> Result<DragonNativeCapabilityAssessment> {
    ensure_supported_training_mode(config, experiment_kind)?;
    let model_config = resolve_model_config_for_capability(config)?;
    let footprint = estimate_language_training_footprint(
        &model_config,
        config.training.batch_size,
        config.training.block_size,
        DragonCapabilityClass::from_backend_label(backend_label),
    );
    let target_decision = decide_native_target(
        requested_target,
        capability_policy,
        DragonCapabilityClass::from_backend_label(backend_label),
        &footprint,
    );

    Ok(DragonNativeCapabilityAssessment {
        experiment_kind,
        backend_label: backend_label.to_owned(),
        model_config,
        batch_size: config.training.batch_size,
        block_size: config.training.block_size,
        footprint,
        target_decision,
    })
}

pub fn assess_native_peer_for_backend(
    native: &DragonNativePeerConfig,
    experiment_kind: DragonExperimentKind,
    backend_label: &str,
) -> Result<DragonNativeCapabilityAssessment> {
    let resolved = resolve_native_training_profile(native, experiment_kind, true)?;
    let config = resolved.config;
    let assessment = assess_loaded_native_training_config(
        &config,
        native.target_or_default(),
        experiment_kind,
        backend_label,
        &native.capability_policy,
    )?;
    apply_native_downgrade_state(&native.storage_root, &config, assessment)
}

fn ensure_supported_training_mode(
    config: &TrainingConfig,
    experiment_kind: DragonExperimentKind,
) -> Result<()> {
    if !matches!(
        config.parallel.mode,
        burn_dragon_train::ParallelismKind::Single
    ) {
        bail!("burn_dragon_p2p currently requires parallel.mode = \"single\"");
    }
    if config.parallel.pipeline.enabled {
        bail!("burn_dragon_p2p does not support pipeline parallel training");
    }
    match (&config.dataset.source, experiment_kind) {
        (
            burn_dragon_language::DatasetSourceConfig::UniversalityManifest { .. },
            DragonExperimentKind::NcaPrepretraining,
        )
        | (
            burn_dragon_language::DatasetSourceConfig::UniversalityNca { .. },
            DragonExperimentKind::NcaPrepretraining,
        )
        | (
            burn_dragon_language::DatasetSourceConfig::UniversalityRuliad { .. },
            DragonExperimentKind::RuliadPretraining,
        )
        | (
            burn_dragon_language::DatasetSourceConfig::NemotronClimbMix { .. },
            DragonExperimentKind::ClimbMixPretraining,
        ) => {}
        (source, DragonExperimentKind::NcaPrepretraining) => {
            bail!(
                "NCA p2p peers require universality_nca data, found {:?}",
                source
            )
        }
        (source, DragonExperimentKind::RuliadPretraining) => {
            bail!(
                "Ruliad p2p peers require universality_ruliad data, found {:?}",
                source
            )
        }
        (source, DragonExperimentKind::ClimbMixPretraining) => {
            bail!(
                "ClimbMix p2p peers require nemotron_climbmix data, found {:?}",
                source
            )
        }
    }
    Ok(())
}

fn mean_loss_from_valid_output<B: Backend>(output: LanguageModelOutput<B>) -> f64 {
    mean_loss_from_output_ref(&output)
}

fn mean_loss_from_output_ref<B: Backend>(output: &LanguageModelOutput<B>) -> f64 {
    let loss_value: LossValue<B> = output.adapt();
    let values = loss_value
        .value()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("loss tensor");
    if values.is_empty() {
        0.0
    } else {
        values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64
    }
}

fn mean_loss_from_train_output_ref<B: AutodiffBackend>(output: &LanguageModelTrainItem<B>) -> f64 {
    mean_loss_from_output_ref(&output.clone().sync())
}

fn insert_train_loss_metrics(
    metrics: &mut BTreeMap<String, MetricValue>,
    step_index: usize,
    loss: f64,
) {
    let previous_mean = match metrics.get("train_loss_mean") {
        Some(MetricValue::Float(value)) => *value,
        Some(MetricValue::Integer(value)) => *value as f64,
        _ => 0.0,
    };
    let count = (step_index + 1) as f64;
    let mean = previous_mean + (loss - previous_mean) / count;
    metrics.insert("train_loss".into(), MetricValue::Float(mean));
    metrics.insert("train_loss_mean".into(), MetricValue::Float(mean));
    metrics.insert("train_loss_last".into(), MetricValue::Float(loss));
}

fn insert_ruliad_source_selection_metrics(
    metrics: &mut BTreeMap<String, MetricValue>,
    snapshot: &burn_dragon_universality::RuliadMetricSnapshot,
) {
    metrics.insert(
        "ruliad_source_selection_entropy_bits".into(),
        MetricValue::Float(snapshot.sampler_entropy_bits as f64),
    );
    metrics.insert(
        "ruliad_source_selection_active_candidate_count".into(),
        MetricValue::Integer(snapshot.active_candidate_count as i64),
    );
    metrics.insert(
        "ruliad_source_selection_active_max_entropy_bits".into(),
        MetricValue::Float(snapshot.active_max_entropy_bits as f64),
    );
    metrics.insert(
        "ruliad_source_selection_normalized_entropy".into(),
        MetricValue::Float(snapshot.normalized_sampler_entropy as f64),
    );
    metrics.insert(
        "ruliad_source_selection_hash_noise_probability".into(),
        MetricValue::Float(snapshot.hash_noise_probability as f64),
    );
    metrics.insert(
        "ruliad_source_selection_mean_loss".into(),
        MetricValue::Float(snapshot.mean_loss as f64),
    );
    metrics.insert(
        "ruliad_source_selection_mean_learning_progress".into(),
        MetricValue::Float(snapshot.mean_learning_progress as f64),
    );
    metrics.insert(
        "ruliad_source_selection_frontier_loss".into(),
        MetricValue::Float(snapshot.frontier_loss as f64),
    );
    metrics.insert(
        "ruliad_source_selection_target_loss".into(),
        MetricValue::Float(snapshot.target_loss as f64),
    );
    metrics.insert(
        "ruliad_source_selection_target_difficulty_score".into(),
        MetricValue::Float(snapshot.target_difficulty_score as f64),
    );
    metrics.insert(
        "ruliad_source_selection_max_difficulty_level".into(),
        MetricValue::Integer(snapshot.max_difficulty_level as i64),
    );
    metrics.insert(
        "ruliad_source_selection_mean_difficulty_level".into(),
        MetricValue::Float(snapshot.mean_difficulty_level as f64),
    );
    metrics.insert(
        "ruliad_source_selection_normalized_difficulty_score".into(),
        MetricValue::Float(snapshot.normalized_difficulty_score as f64),
    );
    metrics.insert(
        "ruliad_source_selection_max_difficulty_probability".into(),
        MetricValue::Float(snapshot.max_difficulty_probability as f64),
    );
    metrics.insert(
        "ruliad_source_selection_mastered_probability".into(),
        MetricValue::Float(snapshot.mastered_probability as f64),
    );
    metrics.insert(
        "ruliad_source_selection_capability_feedback_probability".into(),
        MetricValue::Float(snapshot.capability_feedback_probability as f64),
    );
    metrics.insert(
        "ruliad_source_selection_capability_verifier_ema".into(),
        MetricValue::Float(snapshot.capability_verifier_ema as f64),
    );
    metrics.insert(
        "ruliad_source_selection_capability_completion_health_ema".into(),
        MetricValue::Float(snapshot.capability_completion_health_ema as f64),
    );
    metrics.insert(
        "ruliad_source_selection_capability_schema_wrong_ema".into(),
        MetricValue::Float(snapshot.capability_schema_wrong_ema as f64),
    );
    metrics.insert(
        "ruliad_source_selection_capability_malformed_ema".into(),
        MetricValue::Float(snapshot.capability_malformed_ema as f64),
    );
    metrics.insert(
        "ruliad_source_selection_capability_missing_ema".into(),
        MetricValue::Float(snapshot.capability_missing_ema as f64),
    );
    metrics.insert(
        "ruliad_source_selection_capability_lagging_probability".into(),
        MetricValue::Float(snapshot.capability_lagging_probability as f64),
    );
    metrics.insert(
        "ruliad_source_selection_verifier_failures".into(),
        MetricValue::Integer(snapshot.verifier_failures as i64),
    );
}

#[derive(Clone)]
struct RuliadP2pEvaluationContext<D> {
    dataset: Arc<Dataset>,
    training: TrainingHyperparameters,
    device: D,
}

fn ruliad_p2p_evaluation_context<D: Clone>(
    dataset: Arc<Dataset>,
    config: &TrainingConfig,
    device: &D,
    formal_evaluation_enabled: bool,
) -> Option<RuliadP2pEvaluationContext<D>> {
    if !formal_evaluation_enabled {
        return None;
    }
    let item_count = config.training.events.ruliad_correctness_probe_items;
    if item_count == 0
        || dataset
            .sample_ruliad_validation_probe_items(0, 0, 1)
            .is_empty()
    {
        return None;
    }
    Some(RuliadP2pEvaluationContext {
        dataset,
        training: config.training.clone(),
        device: device.clone(),
    })
}

fn metric_key_component(label: &str) -> String {
    let mut normalized = String::with_capacity(label.len());
    let mut separator = false;
    for character in label.chars() {
        if character.is_ascii_alphanumeric() {
            normalized.push(character.to_ascii_lowercase());
            separator = false;
        } else if !separator && !normalized.is_empty() {
            normalized.push('_');
            separator = true;
        }
    }
    while normalized.ends_with('_') {
        normalized.pop();
    }
    normalized
}

fn insert_ruliad_eval_group_metrics(
    metrics: &mut BTreeMap<String, MetricValue>,
    scope: &str,
    groups: &[burn_dragon_universality::RuliadEvalGroupScore],
) {
    for group in groups {
        let label = metric_key_component(&group.label);
        if label.is_empty() {
            continue;
        }
        let prefix = format!("ruliad_{scope}_{label}");
        metrics.insert(
            format!("{prefix}_items"),
            MetricValue::Integer(group.count as i64),
        );
        metrics.insert(
            format!("{prefix}_verifier_accuracy"),
            MetricValue::Float(group.verifier_accuracy as f64),
        );
        metrics.insert(
            format!("{prefix}_partial_credit_rate"),
            MetricValue::Float(group.partial_credit_rate as f64),
        );
        metrics.insert(
            format!("{prefix}_answer_field_accuracy"),
            MetricValue::Float(group.answer_field_accuracy as f64),
        );
        metrics.insert(
            format!("{prefix}_completion_quality"),
            MetricValue::Float(group.mean_completion_quality as f64),
        );
    }
}

fn insert_ruliad_model_evaluation_metrics(
    metrics: &mut BTreeMap<String, MetricValue>,
    evaluation: &burn_dragon_language::train::schedule::RuliadModelEvaluation,
) {
    let report = &evaluation.report;
    let item_count = report.item_count.max(1) as f64;
    for (key, value) in [
        ("ruliad_exact_accuracy", report.exact_accuracy as f64),
        ("ruliad_semantic_accuracy", report.semantic_accuracy as f64),
        ("ruliad_verifier_accuracy", report.verifier_accuracy as f64),
        (
            "ruliad_partial_credit_rate",
            report.partial_credit_rate as f64,
        ),
        (
            "ruliad_mean_partial_progress",
            report.mean_partial_progress as f64,
        ),
        (
            "ruliad_answer_field_accuracy",
            report.answer_field_accuracy as f64,
        ),
        (
            "ruliad_answer_field_coverage",
            report.answer_field_coverage as f64,
        ),
        (
            "ruliad_answer_termination_rate",
            report.answer_termination_rate as f64,
        ),
        (
            "ruliad_mean_completion_quality",
            report.mean_completion_quality as f64,
        ),
        (
            "ruliad_actual_answer_distinct_fraction",
            report.actual_answer_distinct_fraction as f64,
        ),
        (
            "ruliad_actual_answer_dominant_fraction",
            report.actual_answer_dominant_fraction as f64,
        ),
        (
            "ruliad_malformed_completion_rate",
            report.malformed_completion_count as f64 / item_count,
        ),
        (
            "ruliad_missing_completion_rate",
            report.missing_completion_count as f64 / item_count,
        ),
        ("ruliad_probe_elapsed_ms", evaluation.elapsed_ms),
        (
            "ruliad_probe_generation_mean_batch_rows",
            evaluation.generation_mean_batch_rows,
        ),
        (
            "ruliad_probe_generation_batched_row_fraction",
            evaluation.generation_batched_row_fraction,
        ),
    ] {
        metrics.insert(key.into(), MetricValue::Float(value));
    }
    for (key, value) in [
        ("ruliad_evaluation_items", evaluation.item_count),
        (
            "ruliad_probe_generation_maximum_batch_rows",
            evaluation.generation_maximum_batch_rows,
        ),
        (
            "ruliad_probe_generation_maximum_in_flight_rows",
            evaluation.generation_maximum_in_flight_rows,
        ),
    ] {
        metrics.insert(key.into(), MetricValue::Integer(value as i64));
    }
    insert_ruliad_eval_group_metrics(metrics, "difficulty", &report.difficulty_scores);
    insert_ruliad_eval_group_metrics(metrics, "task", &report.task_scores);
}

fn language_evaluate<B>(
    model: &LanguageTrainModel<ValidBackend<B>>,
    validation_loader: BurnValidationLoader<DragonLearningComponents<B>>,
    max_batches: Option<usize>,
    ruliad: Option<RuliadP2pEvaluationContext<B::Device>>,
    split: EvalSplit,
) -> MetricReport
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let mut total = 0.0;
    let mut count = 0usize;
    for item in validation_loader.iter() {
        if max_batches.is_some_and(|limit| count >= limit) {
            break;
        }
        total += mean_loss_from_valid_output(model.step(item));
        count += 1;
    }
    let mut metrics = std::collections::BTreeMap::from([
        (
            "loss".into(),
            MetricValue::Float(if count == 0 {
                0.0
            } else {
                total / count as f64
            }),
        ),
        (
            "evaluation_batches".into(),
            MetricValue::Integer(count as i64),
        ),
    ]);
    if !matches!(split, EvalSplit::Train)
        && let Some(ruliad) = ruliad
    {
        match burn_dragon_language::train::schedule::evaluate_ruliad_model_free_run(
            &ruliad.dataset,
            model,
            &ruliad.training,
            0,
            0,
            ruliad.training.events.ruliad_correctness_probe_items,
            ruliad.training.batch_size,
            "burn_dragon_p2p_ruliad_validation_v1",
            &ruliad.device,
        ) {
            Ok(Some(evaluation)) => {
                metrics.insert(
                    "ruliad_evaluation_completed".into(),
                    MetricValue::Bool(true),
                );
                insert_ruliad_model_evaluation_metrics(&mut metrics, &evaluation);
            }
            Ok(None) => {
                metrics.insert(
                    "ruliad_evaluation_completed".into(),
                    MetricValue::Bool(false),
                );
                metrics.insert(
                    "ruliad_evaluation_error".into(),
                    MetricValue::Text("formal validation panel is empty".into()),
                );
            }
            Err(error) => {
                metrics.insert(
                    "ruliad_evaluation_completed".into(),
                    MetricValue::Bool(false),
                );
                metrics.insert(
                    "ruliad_evaluation_error".into(),
                    MetricValue::Text(error.to_string()),
                );
            }
        }
    }
    MetricReport {
        metrics,
        captured_at: chrono::Utc::now(),
    }
}

fn build_train_loader<B>(
    datasets: &burn_dragon_language::train::utils::PreparedDatasets,
    config: &TrainingConfig,
    steps_per_epoch: usize,
    total_steps: usize,
    device: &B::Device,
    summary_event_token_ids: Option<Vec<u32>>,
) -> BurnTrainLoader<DragonLearningComponents<B>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    if config
        .training
        .sequence_batching
        .uses_streaming_loader(config.training.tbptt_persist_across_steps)
    {
        Arc::new(
            StreamingDataLoader::<B>::new(
                Arc::clone(&datasets.train),
                DatasetSplit::Train,
                device,
                steps_per_epoch,
                Some(total_steps),
                config.training.min_logical_block_size,
                config.training.seed,
            )
            .with_summary_event_token_ids(summary_event_token_ids),
        )
    } else {
        Arc::new(
            RandomDataLoader::<B>::new(
                Arc::clone(&datasets.train),
                DatasetSplit::Train,
                device,
                steps_per_epoch,
                Some(total_steps),
            )
            .with_summary_event_token_ids(summary_event_token_ids),
        )
    }
}

fn build_valid_loader<B>(
    datasets: &burn_dragon_language::train::utils::PreparedDatasets,
    _config: &TrainingConfig,
    device: &burn::tensor::Device<ValidBackend<B>>,
    summary_event_token_ids: Option<Vec<u32>>,
) -> BurnValidationLoader<DragonLearningComponents<B>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let valid_steps = datasets.valid.steps_per_epoch(DatasetSplit::Val);
    Arc::new(
        RandomDataLoader::<ValidBackend<B>>::new(
            Arc::clone(&datasets.valid),
            DatasetSplit::Val,
            device,
            valid_steps,
            None,
        )
        .with_summary_event_token_ids(summary_event_token_ids),
    )
}

fn build_valid_loader_for_dataset<B>(
    dataset: Arc<Dataset>,
    device: &burn::tensor::Device<ValidBackend<B>>,
    summary_event_token_ids: Option<Vec<u32>>,
) -> BurnValidationLoader<DragonLearningComponents<B>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let valid_steps = dataset.steps_per_epoch(DatasetSplit::Val);
    Arc::new(
        RandomDataLoader::<ValidBackend<B>>::new(
            dataset,
            DatasetSplit::Val,
            device,
            valid_steps,
            None,
        )
        .with_summary_event_token_ids(summary_event_token_ids),
    )
}

fn window_records_from_dataset(
    dataset: &Dataset,
    split: DatasetSplit,
    max_records: Option<usize>,
    batch_size: usize,
) -> Vec<TokenWindowRecord> {
    let (offset, span) = dataset.split_offset_and_span(split);
    let block_size = dataset.block_size();
    if block_size == 0 || span <= block_size {
        return Vec::new();
    }
    let logical_document_tokens = dataset
        .preferred_logical_document_tokens(split)
        .unwrap_or_else(|| span.saturating_sub(1).max(block_size));
    let document_span = logical_document_tokens.saturating_add(1);
    let num_documents = (span / document_span).max(1);
    let chunks_per_document = logical_document_tokens.div_ceil(block_size).max(1);
    let batch_size = batch_size.max(1);
    let mut records = Vec::new();
    for (group_id, document_group_start) in (0..num_documents).step_by(batch_size).enumerate() {
        let group_rows = (num_documents - document_group_start).min(batch_size);
        for chunk_index in 0..chunks_per_document {
            let mut chunk_records = Vec::with_capacity(group_rows);
            let mut chunk_has_supervision = !dataset.uses_target_loss_mask();
            for stream_row in 0..group_rows {
                let document_index = document_group_start + stream_row;
                let start = offset
                    + document_index.saturating_mul(document_span)
                    + chunk_index.saturating_mul(block_size);
                let mut sample = vec![0_u32; block_size + 1];
                dataset.copy_token_range(start, &mut sample);
                let loss_mask = dataset.uses_target_loss_mask().then(|| {
                    let mut mask = vec![0_i64; block_size];
                    dataset.target_loss_mask_for_window(&sample, &mut mask);
                    mask
                });
                chunk_has_supervision |= loss_mask
                    .as_ref()
                    .is_some_and(|mask| mask.iter().any(|weight| *weight != 0));
                chunk_records.push(TokenWindowRecord {
                    inputs: sample[..block_size]
                        .iter()
                        .map(|token| *token as i64)
                        .collect(),
                    targets: sample[1..].iter().map(|token| *token as i64).collect(),
                    loss_mask,
                    reset_stream_state: chunk_index == 0,
                    stream_group_id: Some(group_id as u64),
                    stream_row: Some(stream_row),
                    chunk_index: Some(chunk_index),
                });
            }
            if !chunk_has_supervision {
                continue;
            }
            if max_records.is_some_and(|limit| records.len() + chunk_records.len() > limit) {
                return records;
            }
            records.extend(chunk_records);
            if max_records.is_some_and(|limit| records.len() >= limit) {
                return records;
            }
        }
    }
    records
}

fn stream_segment_partition_key(
    record_index: usize,
    record: &TokenWindowRecord,
    max_segment_chunks: usize,
) -> (u8, u64, usize) {
    match (record.stream_group_id, record.chunk_index) {
        (Some(group_id), Some(chunk_index)) => {
            (0, group_id, chunk_index / max_segment_chunks.max(1))
        }
        _ => (1, record_index as u64, 0),
    }
}

fn attach_sharded_dataset<B>(
    builder: burn_p2p::burn::BurnLearnerProjectBuilder<DragonLearningComponents<B>>,
    experiment_kind: DragonExperimentKind,
    dataset_source: &burn_dragon_language::DatasetSourceConfig,
    datasets: &burn_dragon_language::train::utils::PreparedDatasets,
    shard_export: &DragonShardExportConfig,
    summary_event_token_ids: Option<Vec<u32>>,
    max_train_batches: usize,
) -> Result<(
    burn_p2p::burn::BurnLearnerProjectBuilder<DragonLearningComponents<B>>,
    DatasetViewId,
)>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let records = window_records_from_dataset(
        &datasets.train,
        DatasetSplit::Train,
        shard_export.max_records,
        datasets.train.batch_size(),
    );
    if records.is_empty() {
        bail!(
            "shard export for {} produced no records",
            shard_export.root.display()
        );
    }
    let dataset_name = shard_export
        .dataset_name
        .clone()
        .unwrap_or_else(|| "burn-dragon-p2p-dataset".into());
    let mut config = burn_p2p::burn::BurnShardedDatasetConfig::new(dataset_name)
        .with_source_uri(shard_export.root.display().to_string())
        .with_view_metadata_entry("dataset_kind", "language-token-windows");
    if let Some(count) = shard_export.microshards {
        config = config.with_microshards(count);
    }
    let sharded = burn_p2p::burn::BurnShardedDataset::write_local_grouped_by(
        &shard_export.root,
        &records,
        config,
        "dragon-bounded-stream-segment-balanced-v3-target-masks",
        |record_index, record| {
            stream_segment_partition_key(record_index, record, max_train_batches)
        },
    )?;
    let sharded = if let Some(base_url) = &shard_export.http_upstream {
        sharded.with_http_upstream(base_url.clone())
    } else {
        sharded
    };
    let descriptor = dragon_sharded_input_descriptor(
        experiment_kind,
        dataset_source,
        sharded.registration(),
        sharded.microshard_plan().microshards.len(),
        shard_export.http_upstream.as_deref(),
    );
    let dataset_view_id = sharded.registration().view.dataset_view_id.clone();
    Ok((
        builder.with_data_pipeline(dragon_sharded_data_pipeline::<B>(
            descriptor,
            sharded,
            TokenWindowBatcher::new(summary_event_token_ids),
            datasets.train.batch_size(),
            max_train_batches,
        )),
        dataset_view_id,
    ))
}

fn attach_existing_sharded_dataset<B>(
    builder: burn_p2p::burn::BurnLearnerProjectBuilder<DragonLearningComponents<B>>,
    experiment_kind: DragonExperimentKind,
    dataset_source: &burn_dragon_language::DatasetSourceConfig,
    shard_dataset: &DragonExistingShardDatasetConfig,
    batch_size: usize,
    max_train_batches: usize,
    summary_event_token_ids: Option<Vec<u32>>,
) -> Result<burn_p2p::burn::BurnLearnerProjectBuilder<DragonLearningComponents<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let sharded =
        burn_p2p::burn::BurnShardedDataset::<TokenWindowRecord>::read_local(&shard_dataset.root)?;
    let sharded = if let Some(base_url) = &shard_dataset.http_upstream {
        sharded.with_http_upstream(base_url.clone())
    } else {
        sharded.with_local_upstream(shard_dataset.root.display().to_string())
    };
    let descriptor = dragon_sharded_input_descriptor(
        experiment_kind,
        dataset_source,
        sharded.registration(),
        sharded.microshard_plan().microshards.len(),
        shard_dataset.http_upstream.as_deref(),
    );
    Ok(
        builder.with_data_pipeline(dragon_sharded_data_pipeline::<B>(
            descriptor,
            sharded,
            TokenWindowBatcher::new(summary_event_token_ids),
            batch_size,
            max_train_batches,
        )),
    )
}

fn build_language_learning_components<B>(
    config: &TrainingConfig,
    backend_label: &str,
    model_config: &DragonConfig,
    total_steps: usize,
    scheduler_iters: Option<usize>,
    device: &B::Device,
) -> Result<(
    LanguageTrainModel<B>,
    LanguageOptimizer<B>,
    ResolvedLrScheduler,
)>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    B::seed(device, config.training.seed);
    let mut base_model = DragonModel::<B>::new(model_config.clone(), device);
    let fresh_model = base_model.clone();
    if let Some(checkpoint_path) = &config.training.init_checkpoint_path {
        base_model = apply_init_checkpoint_to_language_core(
            &base_model,
            config,
            checkpoint_path,
            config.training.init_checkpoint_epoch,
            backend_label,
            device,
        )?;
    }
    validate_dragon_continual_backprop(&config.training, &base_model, 1)?;

    let model = LanguageTrainModel::new(base_model)
        .with_training_configuration(&config.training, total_steps)
        .with_pipeline_plan(None);
    let optimizer = resolve_dragon_language_optimizer::<B>(
        &config.training,
        &config.optimizer,
        total_steps,
        fresh_model,
    )?;
    let scheduler = resolve_lr_scheduler(
        &config.optimizer,
        total_steps,
        scheduler_iters,
        model_config,
    )?;
    Ok((model, optimizer, scheduler))
}

fn shard_dataset_upstream(
    shard_export: Option<&DragonShardExportConfig>,
    existing_shard_dataset: Option<&DragonExistingShardDatasetConfig>,
) -> Result<Option<burn_p2p::UpstreamAdapter>> {
    if shard_export.is_some() && existing_shard_dataset.is_some() {
        bail!("configure at most one of shard_export or existing_shard_dataset");
    }
    Ok(match (shard_export, existing_shard_dataset) {
        (Some(shard_export), None) => Some(if let Some(base_url) = &shard_export.http_upstream {
            burn_p2p::UpstreamAdapter::Http {
                base_url: base_url.clone(),
            }
        } else {
            burn_p2p::UpstreamAdapter::Local {
                root: shard_export.root.display().to_string(),
            }
        }),
        (None, Some(shard_dataset)) => Some(if let Some(base_url) = &shard_dataset.http_upstream {
            burn_p2p::UpstreamAdapter::Http {
                base_url: base_url.clone(),
            }
        } else {
            burn_p2p::UpstreamAdapter::Local {
                root: shard_dataset.root.display().to_string(),
            }
        }),
        (None, None) => None,
        (Some(_), Some(_)) => unreachable!(),
    })
}

fn ensure_tokenizer_compatible(
    train_tokenizer: &dyn Tokenizer,
    valid_tokenizer: &dyn Tokenizer,
    tokenizer_label: &str,
) -> Result<()> {
    if train_tokenizer.len() != valid_tokenizer.len() {
        bail!(
            "validation dataset tokenizer is incompatible with the training tokenizer: vocab sizes differ (train={}, valid={}, tokenizer={tokenizer_label})",
            train_tokenizer.len(),
            valid_tokenizer.len(),
        );
    }
    if train_tokenizer.bos_id() != valid_tokenizer.bos_id()
        || train_tokenizer.eos_id() != valid_tokenizer.eos_id()
        || train_tokenizer.pad_id() != valid_tokenizer.pad_id()
        || train_tokenizer.unk_id() != valid_tokenizer.unk_id()
    {
        bail!(
            "validation dataset tokenizer is incompatible with the training tokenizer: special token ids differ (tokenizer={tokenizer_label})"
        );
    }
    Ok(())
}

fn prepare_validation_loader_only<B>(
    config: &TrainingConfig,
    device: &burn::tensor::Device<ValidBackend<B>>,
    base_tokenizer: &dyn Tokenizer,
    summary_event_token_ids: Option<Vec<u32>>,
) -> Result<Option<DragonValidationSource<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let effective_cfg = match &config.dataset.validation {
        Some(validation_cfg) => validation_dataset_config_for(&config.dataset, validation_cfg),
        None => match &config.dataset.source {
            burn_dragon_language::DatasetSourceConfig::UniversalityNca { .. }
            | burn_dragon_language::DatasetSourceConfig::UniversalityRuliad { .. }
            | burn_dragon_language::DatasetSourceConfig::UniversalityManifest { .. } => {
                config.dataset.clone()
            }
            burn_dragon_language::DatasetSourceConfig::NemotronClimbMix { .. } => return Ok(None),
        },
    };
    let prepared = prepare_datasets(&effective_cfg, &config.training)?;
    ensure_tokenizer_compatible(
        base_tokenizer,
        prepared.valid.tokenizer().as_ref(),
        config.dataset.tokenizer.kind_name(),
    )?;
    let dataset = prepared.valid;
    Ok(Some((
        build_valid_loader_for_dataset::<B>(Arc::clone(&dataset), device, summary_event_token_ids),
        dataset,
    )))
}

pub fn prepare_language_peer_for_backend<B>(
    native: &DragonNativePeerConfig,
    experiment_kind: DragonExperimentKind,
    backend_label: &str,
    device: B::Device,
    auth_bundle: Option<&DragonNativeAuthBundle>,
) -> Result<PreparedNativePeer<B>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    native.capability_policy.native_reprobe.validate()?;
    let resolved = resolve_native_training_profile(native, experiment_kind, true)?;
    let config = resolved.config;
    if native
        .training_overrides
        .max_eval_batches
        .is_some_and(|max_eval_batches| max_eval_batches == 0)
    {
        bail!("native training override max_eval_batches must be > 0");
    }
    let capability_assessment = apply_native_downgrade_state(
        &native.storage_root,
        &config,
        assess_loaded_native_training_config(
            &config,
            native.target_or_default(),
            experiment_kind,
            backend_label,
            &native.capability_policy,
        )?,
    )?;
    let use_existing_shards = native.existing_shard_dataset.as_ref();
    let dataset_upstream = shard_dataset_upstream(
        native.shard_export.as_ref(),
        native.existing_shard_dataset.as_ref(),
    )?;
    let model_config = capability_assessment.model_config.clone();
    let footprint = capability_assessment.footprint.clone();
    let target_decision = capability_assessment.target_decision.clone();
    let formal_evaluation_enabled = matches!(
        target_decision.effective_target,
        crate::config::DragonNativeTarget::Validator
    );

    let (project, dataset_view_id, random_scaffold) = if let Some(existing_shards) =
        use_existing_shards
    {
        let sharded = burn_p2p::burn::BurnShardedDataset::<TokenWindowRecord>::read_local(
            &existing_shards.root,
        )?;
        let total_examples = sharded
            .shard_examples()
            .values()
            .copied()
            .sum::<usize>()
            .max(1);
        let steps_per_epoch = total_examples.div_ceil(config.training.batch_size.max(1));
        let train_schedule = resolve_train_schedule(&config.training, steps_per_epoch)?;
        let total_steps = train_schedule.total_steps.max(1);
        let scheduler_iters = match train_schedule.source {
            burn_dragon_train::train::pipeline::ScheduleSource::Epochs => Some(total_steps),
            burn_dragon_train::train::pipeline::ScheduleSource::MaxIters => None,
        };
        let tokenizer = load_tokenizer_without_dataset(&config)?;
        let summary_event_token_ids = summary_event_token_ids_for_tokenizer(tokenizer.as_ref());
        let (model, optimizer, scheduler) = build_language_learning_components::<B>(
            &config,
            backend_label,
            &model_config,
            total_steps,
            scheduler_iters,
            &device,
        )?;
        let random_scaffold = dragon_random_scaffold_p2p_contract::<B>(
            &model,
            dragon_model_schema_hash(&model_config),
        )?;
        let backend_label_owned = backend_label.to_owned();
        let estimated_tokens_per_second = footprint.estimated_tokens_per_second;
        let mut builder = from_stateful_components(
            model,
            optimizer,
            scheduler,
            config.training.gradient_accumulation_steps,
            device.clone(),
        )
        .with_benchmark(move |model, _device| {
            let inventory = burn_p2p::burn::inspect_module::<B, _>(model);
            burn_p2p::CapabilityEstimate {
                preferred_backends: vec![backend_label_owned.clone()],
                work_units_per_second: estimated_tokens_per_second
                    .max((inventory.total_scalar_parameters.max(1) as f64).sqrt()),
                target_window_seconds: 30,
            }
        })
        .with_step_metrics(|step_index, output, metrics| {
            metrics.insert(
                "train_steps".into(),
                MetricValue::Integer((step_index + 1) as i64),
            );
            insert_train_loss_metrics(metrics, step_index, mean_loss_from_train_output_ref(output));
            Ok(())
        });
        if let Some((validation_loader, validation_dataset)) = prepare_validation_loader_only::<B>(
            &config,
            &device,
            tokenizer.as_ref(),
            summary_event_token_ids.clone(),
        )? {
            let validation_for_eval = validation_loader.clone();
            let max_eval_batches = native.training_overrides.max_eval_batches;
            let ruliad_evaluation = ruliad_p2p_evaluation_context(
                validation_dataset,
                &config,
                &device,
                formal_evaluation_enabled,
            );
            builder = builder
                .with_validation_loader(validation_loader)
                .with_evaluate(move |model, split| {
                    language_evaluate::<B>(
                        model,
                        validation_for_eval.clone(),
                        max_eval_batches,
                        ruliad_evaluation.clone(),
                        split,
                    )
                });
        }
        builder = attach_existing_sharded_dataset::<B>(
            builder,
            experiment_kind,
            &config.dataset.source,
            existing_shards,
            config.training.batch_size,
            config.training.max_iters,
            summary_event_token_ids,
        )?;
        builder = attach_dragon_workload_update_applier(builder, &config, random_scaffold.as_ref());
        (
            builder.build()?,
            sharded.registration().view.dataset_view_id.clone(),
            random_scaffold,
        )
    } else {
        let datasets = prepare_datasets(&config.dataset, &config.training)?;
        let summary_event_token_ids = summary_event_token_ids(&datasets.train);
        let steps_per_epoch = datasets.train.steps_per_epoch(DatasetSplit::Train);
        let train_schedule = resolve_train_schedule(&config.training, steps_per_epoch)?;
        let total_steps = train_schedule.total_steps.max(1);
        let scheduler_iters = match train_schedule.source {
            burn_dragon_train::train::pipeline::ScheduleSource::Epochs => Some(total_steps),
            burn_dragon_train::train::pipeline::ScheduleSource::MaxIters => None,
        };

        let train_loader = build_train_loader::<B>(
            &datasets,
            &config,
            train_schedule.steps_per_epoch,
            total_steps,
            &device,
            summary_event_token_ids.clone(),
        );
        let valid_device = device.clone();
        let validation_loader = build_valid_loader::<B>(
            &datasets,
            &config,
            &valid_device,
            summary_event_token_ids.clone(),
        );
        let (model, optimizer, scheduler) = build_language_learning_components::<B>(
            &config,
            backend_label,
            &model_config,
            total_steps,
            scheduler_iters,
            &device,
        )?;
        let random_scaffold = dragon_random_scaffold_p2p_contract::<B>(
            &model,
            dragon_model_schema_hash(&model_config),
        )?;
        let validation_for_eval = validation_loader.clone();
        let ruliad_evaluation = ruliad_p2p_evaluation_context(
            Arc::clone(&datasets.valid),
            &config,
            &device,
            formal_evaluation_enabled,
        );
        let max_eval_batches = native.training_overrides.max_eval_batches;
        let backend_label_owned = backend_label.to_owned();
        let estimated_tokens_per_second = footprint.estimated_tokens_per_second;
        let source_selection_dataset = Arc::clone(&datasets.train);
        let mut builder = from_stateful_loaders(
            model,
            optimizer,
            scheduler,
            config.training.gradient_accumulation_steps,
            device.clone(),
            train_loader,
            validation_loader,
        )
        .with_benchmark(move |model, _device| {
            let inventory = burn_p2p::burn::inspect_module::<B, _>(model);
            burn_p2p::CapabilityEstimate {
                preferred_backends: vec![backend_label_owned.clone()],
                work_units_per_second: estimated_tokens_per_second
                    .max((inventory.total_scalar_parameters.max(1) as f64).sqrt()),
                target_window_seconds: 30,
            }
        })
        .with_evaluate(move |model, split| {
            language_evaluate::<B>(
                model,
                validation_for_eval.clone(),
                max_eval_batches,
                ruliad_evaluation.clone(),
                split,
            )
        })
        .with_step_metrics(move |step_index, output, metrics| {
            let train_loss = mean_loss_from_train_output_ref(output);
            metrics.insert(
                "train_steps".into(),
                MetricValue::Integer((step_index + 1) as i64),
            );
            insert_train_loss_metrics(metrics, step_index, train_loss);
            if let Some(snapshot) =
                source_selection_dataset.record_source_selection_loss(step_index, train_loss as f32)
            {
                insert_ruliad_source_selection_metrics(metrics, &snapshot);
            }
            Ok(())
        });

        let exported_dataset_view_id = if let Some(shard_export) = &native.shard_export {
            let (next_builder, dataset_view_id) = attach_sharded_dataset::<B>(
                builder,
                experiment_kind,
                &config.dataset.source,
                &datasets,
                shard_export,
                summary_event_token_ids.clone(),
                config.training.max_iters,
            )?;
            builder = next_builder;
            Some(dataset_view_id)
        } else {
            None
        };
        builder = attach_dragon_workload_update_applier(builder, &config, random_scaffold.as_ref());
        (
            builder.build()?,
            match exported_dataset_view_id {
                Some(dataset_view_id) => dataset_view_id,
                None => inline_dataset_view_id(&datasets.train)?,
            },
            random_scaffold,
        )
    };

    let git_commit = native.git_commit.as_deref().unwrap_or("unknown");
    let enabled_features = native
        .enabled_features_label
        .as_deref()
        .unwrap_or(backend_label);
    let mut manifest_seed = resolved.manifest_seed;
    let effective_seed_node_urls = native.effective_seed_node_urls();
    if !effective_seed_node_urls.is_empty() {
        manifest_seed.bootstrap_addrs = effective_seed_node_urls;
    }
    if let Some(random_scaffold) = &random_scaffold {
        let mutable_parameter_count = random_scaffold.catalog.parameter_count()?;
        log::info!(
            "random-scaffold P2P contract: scaffold={} mutable_params={} frozen_params={} encoding={:?}",
            random_scaffold.scaffold_contract_hash.as_str(),
            mutable_parameter_count,
            random_scaffold.frozen_parameter_count,
            manifest_seed.random_scaffold_update_encoding,
        );
    }
    let genesis_materialization = random_scaffold
        .as_ref()
        .map(random_scaffold_genesis_materialization)
        .transpose()?
        .unwrap_or_default();
    let manifests = build_manifest_bundle(
        &manifest_seed,
        experiment_kind,
        backend_label,
        &model_config,
        Some(&config),
        random_scaffold.as_ref().map(|contract| &contract.catalog),
        &resolved.profile,
        dataset_view_id,
        &footprint,
        native.app_semver.clone(),
        git_commit,
        enabled_features,
    )?;

    let auth_available = auth_bundle.is_some()
        || native.auth.as_ref().is_some_and(|auth| {
            auth.local_peer_auth.is_some() && !auth.trust_bundle_endpoints.is_empty()
        });
    if !auth_available {
        bail!("burn_dragon_p2p peers require an authenticated edge auth bundle");
    }

    let mut node_builder = connect(
        target_decision.burn_target(DragonCapabilityClass::from_backend_label(backend_label)),
        manifests.release_manifest.clone(),
        project.clone(),
        manifests.workload_config.clone(),
    )?;
    node_builder = node_builder
        .with_mainnet(burn_p2p::GenesisSpec {
            network_id: manifests.network_manifest.network_id.clone(),
            protocol_version: semver::Version::new(
                u64::from(manifests.network_manifest.protocol_major),
                0,
                0,
            ),
            display_name: manifests.network_manifest.description.clone(),
            created_at: manifests.network_manifest.created_at,
            metadata: Default::default(),
        })
        .with_storage(native.storage_root.clone())
        .with_identity(native.identity.clone());
    let mut node_builder = node_builder.with_network(manifests.network_manifest.clone())?;
    for peer in native.effective_bootstrap_peers()? {
        node_builder = node_builder.with_bootstrap_peer(peer);
    }
    for address in native.effective_listen_addresses().iter().cloned() {
        node_builder = node_builder.with_listen_address(address);
    }
    for address in native.effective_external_addresses().iter().cloned() {
        node_builder = node_builder.with_external_address(address);
    }
    let auth_config = compose_auth_config(
        native.auth.clone(),
        auth_bundle,
        &manifests.experiment_directory,
    );
    node_builder = node_builder.with_auth(auth_config);
    if matches!(
        target_decision.requested_target,
        crate::config::DragonNativeTarget::Auto | crate::config::DragonNativeTarget::Trainer
    ) {
        let gpu = !backend_label.eq_ignore_ascii_case("cpu")
            && !backend_label.eq_ignore_ascii_case("ndarray");
        let mut roles = target_decision
            .requested_target
            .roles(gpu)
            .roles
            .into_iter()
            .collect::<Vec<_>>();
        roles.extend(target_decision.effective_target.roles(gpu).roles);
        roles.push(PeerRole::Viewer);
        node_builder = node_builder.with_role_capabilities(PeerRoleSet::new(roles));
    }
    if let Some(path) = native.manifest.revision_contract_path.as_ref() {
        for authority_public_key_hex in &native.manifest.authority_public_keys {
            let public_key_bytes = hex::decode(authority_public_key_hex)
                .with_context(|| "decode revision authority public key as protobuf hex")?;
            let public_key = libp2p_identity::PublicKey::try_decode_protobuf(&public_key_bytes)
                .with_context(|| "decode revision authority protobuf public key")?;
            let issuer_peer_id = burn_p2p::PeerId::new(
                libp2p_identity::PeerId::from_public_key(&public_key).to_string(),
            );
            node_builder =
                node_builder.with_revision_contract_trusted_issuer(burn_p2p::TrustedIssuer {
                    issuer_peer_id,
                    issuer_public_key_hex: authority_public_key_hex.clone(),
                })?;
        }
        let bytes = std::fs::read(path).with_context(|| {
            format!(
                "read signed revision contract bundle from {}",
                path.display()
            )
        })?;
        let contract: burn_p2p::RevisionContractBundle = serde_json::from_slice(&bytes)
            .with_context(|| {
                format!(
                    "decode signed revision contract bundle from {}",
                    path.display()
                )
            })?;
        contract
            .validate()
            .context("validate signed Dragon revision contract bundle")?;
        ensure!(
            contract.revision.experiment_id.as_str() == manifest_seed.experiment_id
                && contract.revision.revision_id.as_str() == manifest_seed.revision_id,
            "signed revision contract does not match selected Dragon experiment/revision"
        );
        ensure!(
            contract.training_contract_id == manifests.training_contract_id
                && contract.training == manifests.training_contract,
            "signed revision contract semantic training identity does not match local Dragon configuration"
        );
        node_builder = node_builder.with_revision_contract(contract)?;
    } else if native.manifest.require_signed_revision_contracts {
        bail!(
            "signed revision contracts are required, but manifest.revision_contract_path is unset"
        );
    }
    node_builder = node_builder
        .require_signed_revision_contracts(native.manifest.require_signed_revision_contracts);
    if let Some(upstream) = dataset_upstream {
        node_builder = node_builder.with_dataset(upstream);
    }

    Ok(PreparedNativePeer {
        project,
        builder: node_builder,
        manifests,
        config,
        storage_root: native.storage_root.clone(),
        experiment_kind,
        backend_label: backend_label.to_owned(),
        model_config,
        footprint,
        target_decision,
        capability_reprobe_policy: native.capability_policy.native_reprobe.clone(),
        genesis_materialization,
    })
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;

    use super::*;

    fn record(
        group: Option<u64>,
        row: Option<usize>,
        chunk: Option<usize>,
        token: i64,
        reset: bool,
    ) -> TokenWindowRecord {
        TokenWindowRecord {
            inputs: vec![token, token + 1],
            targets: vec![token + 1, token + 2],
            loss_mask: None,
            reset_stream_state: reset,
            stream_group_id: group,
            stream_row: row,
            chunk_index: chunk,
        }
    }

    #[test]
    fn sharded_batches_restore_stream_group_and_row_order() {
        let device = burn::tensor::Device::<NdArray<f32>>::default();
        let batcher = TokenWindowBatcher::new(None);
        let batches = batcher
            .stream_aligned_batches::<NdArray<f32>>(
                vec![
                    record(Some(7), Some(1), Some(1), 31, false),
                    record(Some(8), Some(0), Some(5), 50, false),
                    record(Some(7), Some(0), Some(0), 10, true),
                    record(Some(7), Some(0), Some(1), 30, false),
                    record(Some(7), Some(1), Some(0), 11, true),
                ],
                2,
                None,
                None,
                &device,
            )
            .expect("aligned batches");

        assert_eq!(batches.len(), 3);
        assert!(batches[0].reset_stream_state);
        assert!(!batches[1].reset_stream_state);
        assert!(batches[2].reset_stream_state);
        assert_eq!(
            batches[1]
                .inputs
                .clone()
                .into_data()
                .into_vec::<i64>()
                .expect("tokens"),
            vec![30, 31, 31, 32]
        );
    }

    #[test]
    fn sharded_batches_preserve_loss_masks_and_default_legacy_rows_to_supervised() {
        let device = burn::tensor::Device::<NdArray<f32>>::default();
        let batcher = TokenWindowBatcher::new(None);
        let mut masked = record(Some(7), Some(0), Some(0), 10, true);
        masked.loss_mask = Some(vec![1, 0]);
        let legacy = record(Some(7), Some(1), Some(0), 20, true);
        let batch = batcher.batch_items::<NdArray<f32>>(vec![masked, legacy], true, &device);

        assert_eq!(
            batch
                .loss_mask
                .expect("mixed masked and legacy rows should emit a mask")
                .into_data()
                .into_vec::<i64>()
                .expect("mask values"),
            vec![1, 0, 1, 1]
        );
    }

    #[test]
    fn legacy_shards_reset_each_batch_instead_of_cross_wiring_streams() {
        let device = burn::tensor::Device::<NdArray<f32>>::default();
        let batcher = TokenWindowBatcher::new(None);
        let batches = batcher
            .stream_aligned_batches::<NdArray<f32>>(
                vec![
                    record(None, None, None, 1, true),
                    record(None, None, None, 2, false),
                    record(None, None, None, 3, false),
                ],
                2,
                None,
                None,
                &device,
            )
            .expect("legacy batches");
        assert_eq!(batches.len(), 2);
        assert!(batches.iter().all(|batch| batch.reset_stream_state));
    }

    #[test]
    fn shard_partition_preserves_bounded_contiguous_stream_segments() {
        let key = |index, group, chunk| {
            stream_segment_partition_key(
                index,
                &record(group, Some(0), chunk, index as i64, chunk == Some(0)),
                3,
            )
        };

        assert_eq!(key(0, Some(7), Some(0)), key(1, Some(7), Some(2)));
        assert_ne!(key(0, Some(7), Some(2)), key(1, Some(7), Some(3)));
        assert_ne!(key(0, Some(7), Some(0)), key(1, Some(8), Some(0)));
        assert_ne!(key(0, None, None), key(1, None, None));
    }

    #[test]
    fn train_loss_metrics_keep_mean_and_last_batch_distinct() {
        let mut metrics = BTreeMap::new();
        for (step_index, loss) in [3.0, 1.0, 4.0].into_iter().enumerate() {
            insert_train_loss_metrics(&mut metrics, step_index, loss);
        }
        assert_eq!(
            metrics.get("train_loss"),
            Some(&MetricValue::Float(8.0 / 3.0))
        );
        assert_eq!(
            metrics.get("train_loss_mean"),
            Some(&MetricValue::Float(8.0 / 3.0))
        );
        assert_eq!(
            metrics.get("train_loss_last"),
            Some(&MetricValue::Float(4.0))
        );
    }

    #[test]
    fn metric_key_components_are_stable_and_path_safe() {
        assert_eq!(
            metric_key_component("Formal Proof / Select-Action:d=17"),
            "formal_proof_select_action_d_17"
        );
        assert_eq!(metric_key_component("___"), "");
    }

    #[test]
    fn p2p_ruliad_metrics_preserve_verifier_and_generation_evidence() {
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "metric-fixture".into(),
            sample_index: 0,
            split: burn_dragon_universality::SampleSplit::Validation,
            family: "formal_proof".into(),
            task_kind: "select_proof_action".into(),
            math_domains: vec!["category_theory".into()],
            reasoning_modes: vec!["equational_reasoning".into()],
            prompt: "[R3 metric-fixture v1 P/thm/proof]\nA:ok,l,r\n!:".into(),
            expected_answer: "ok=1;l=2;r=2".into(),
            difficulty_level: Some(17),
            spec: None,
        };
        let completion = burn_dragon_universality::RuliadCompletionRecord {
            oracle_hash: "metric-fixture".into(),
            completion: "!:ok=1;l=2;r=2\n[/R3]\n".into(),
        };
        let report = burn_dragon_universality::evaluate_completions(
            "p2p-metric-fixture",
            &[item],
            &[completion],
        );
        let evaluation = burn_dragon_language::train::schedule::RuliadModelEvaluation {
            report: report.clone(),
            item_count: 1,
            elapsed_ms: 12.5,
            generation_mean_batch_rows: 4.0,
            generation_maximum_batch_rows: 8,
            generation_maximum_in_flight_rows: 4,
            generation_batched_row_fraction: 1.0,
        };
        let mut metrics = BTreeMap::new();
        insert_ruliad_model_evaluation_metrics(&mut metrics, &evaluation);

        assert_eq!(
            metrics.get("ruliad_semantic_accuracy"),
            Some(&MetricValue::Float(report.semantic_accuracy as f64))
        );
        assert_eq!(
            metrics.get("ruliad_verifier_accuracy"),
            Some(&MetricValue::Float(report.verifier_accuracy as f64))
        );
        assert_eq!(
            metrics.get("ruliad_probe_generation_maximum_batch_rows"),
            Some(&MetricValue::Integer(8))
        );
        assert!(metrics.keys().any(|key| {
            key.starts_with("ruliad_difficulty_") && key.ends_with("_verifier_accuracy")
        }));
        assert!(metrics.keys().any(|key| {
            key.starts_with("ruliad_task_") && key.ends_with("_answer_field_accuracy")
        }));
    }
}
