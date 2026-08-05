use std::collections::{BTreeMap, BTreeSet};
use std::io::Cursor;

#[cfg(feature = "native")]
use anyhow::bail;
use anyhow::{Context, Result, ensure};
use burn::module::Module;
#[cfg(feature = "native")]
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::backend::Backend;
use burn_dragon_core::DragonModel;
#[cfg(feature = "native")]
use burn_dragon_language::train::steps::LanguageTrainModel;
#[cfg(feature = "native")]
use burn_p2p::burn_module::{
    diff_module_float_parameter_subset, flatten_module_float_parameter_subset, module_tensor_digest,
};
use burn_p2p::burn_module::{
    inspect_module, module_float_parameter_subset_catalog, replace_module_float_parameter_subset,
};
#[cfg(feature = "native")]
use burn_p2p::{
    ArtifactBuildSpec, COMPACT_UPDATE_PAYLOAD_VERSION, ChunkingScheme, CompactUpdateBody,
    CompactUpdatePayload, GenesisArtifactLoadContext, GenesisArtifactMaterializationContext,
    MaterializedWorkloadUpdate, Precision, UpdateCodec, UpdateNormStats, ValidatedUpdateEvidence,
    ValidatedWorkloadUpdate, WorkloadUpdateEnvelope, WorkloadUpdateMaterializationContext,
    WorkloadUpdateValidationContext,
};
use burn_p2p::{
    ArtifactKind, CompactScalarEncoding, CompactScalarVector, ContentId, GenesisMaterialization,
    ParameterSubsetCatalog,
};
use serde::{Deserialize, Serialize};

#[cfg(feature = "native")]
const MUTABLE_SUBSET_RECORD_FORMAT: &str = "burn-p2p-compact-update-cbor-v1";
#[cfg(feature = "native")]
const MUTABLE_SUBSET_CHUNK_BYTES: u64 = 256 * 1024;
const RANDOM_SCAFFOLD_GENESIS_RECORD_FORMAT: &str = "burn-dragon-random-scaffold-genesis-cbor-v1";
const RANDOM_SCAFFOLD_GENESIS_VERSION: u16 = 1;
pub(crate) const RANDOM_SCAFFOLD_HEAD_RECORD_FORMAT: &str =
    "burn-dragon-random-scaffold-head-cbor-v1";
const RANDOM_SCAFFOLD_HEAD_VERSION: u16 = 1;
#[cfg(feature = "native")]
const VERIFY_CATALOG_ENV: &str = "BURN_DRAGON_P2P_VERIFY_RANDOM_SCAFFOLD_CATALOG";

#[derive(Clone, Debug)]
pub(crate) struct DragonRandomScaffoldP2pContract {
    pub catalog: ParameterSubsetCatalog,
    pub immutable_catalog: ParameterSubsetCatalog,
    pub scaffold_contract_hash: ContentId,
    #[cfg(feature = "native")]
    pub frozen_parameter_count: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DragonRandomScaffoldGenesisPayload {
    version: u16,
    training_contract_id: ContentId,
    model_schema_hash: ContentId,
    reconstruction_contract_hash: ContentId,
    immutable_parameter_catalog_hash: ContentId,
    mutable_parameter_catalog_hash: ContentId,
    mutable_parameter_count: u64,
    values: CompactScalarVector,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DragonRandomScaffoldHeadPayload {
    version: u16,
    model_schema_hash: ContentId,
    reconstruction_contract_hash: ContentId,
    immutable_parameter_catalog_hash: ContentId,
    mutable_parameter_catalog_hash: ContentId,
    mutable_parameter_count: u64,
    values: CompactScalarVector,
}

pub(crate) fn dragon_random_scaffold_p2p_contract_for_module<B, M>(
    module: &M,
    model: &DragonModel<B>,
    model_schema_hash: ContentId,
) -> Result<Option<DragonRandomScaffoldP2pContract>>
where
    B: Backend,
    M: Module<B>,
{
    let Some(report) = model.random_scaffold_report() else {
        return Ok(None);
    };
    let inventory = inspect_module::<B, _>(module);
    let by_id = inventory
        .parameters
        .iter()
        .map(|parameter| (parameter.param_id.as_str(), parameter))
        .collect::<BTreeMap<_, _>>();
    let frozen_ids = model
        .random_scaffold_frozen_param_ids()
        .into_iter()
        .map(|id| id.to_string())
        .collect::<BTreeSet<_>>();
    let frozen_paths = frozen_ids
        .iter()
        .map(|id| {
            by_id
                .get(id.as_str())
                .map(|parameter| parameter.path.clone())
                .ok_or_else(|| anyhow::anyhow!("immutable scaffold parameter {id} is not in model"))
        })
        .collect::<Result<BTreeSet<_>>>()?;
    let trainable_gain = report.manifest.adapter.trainable_gain;
    let expected_immutable_tensors =
        report.manifest.tensors.len() * if trainable_gain { 1 } else { 2 };
    ensure!(
        frozen_paths.len() == expected_immutable_tensors,
        "random-scaffold contract expected {expected_immutable_tensors} immutable tensors, found {}",
        frozen_paths.len()
    );

    let catalog =
        module_float_parameter_subset_catalog::<B, _>(module, model_schema_hash.clone(), |path| {
            !frozen_paths.contains(path)
                && (trainable_gain
                    || !(path.contains("random_scaffold_adapters") && path.ends_with(".gain")))
        })?;
    let immutable_catalog =
        module_float_parameter_subset_catalog::<B, _>(module, model_schema_hash, |path| {
            frozen_paths.contains(path)
        })?;
    let mutable_paths = catalog
        .entries
        .iter()
        .map(|entry| entry.path.as_str())
        .collect::<BTreeSet<_>>();
    let immutable_paths = immutable_catalog
        .entries
        .iter()
        .map(|entry| entry.path.as_str())
        .collect::<BTreeSet<_>>();
    ensure!(
        frozen_paths
            .iter()
            .all(|path| !mutable_paths.contains(path.as_str())),
        "immutable random-scaffold tensor entered the trainable parameter catalog"
    );
    let full_catalog = module_float_parameter_subset_catalog::<B, _>(
        module,
        catalog.model_schema_hash.clone(),
        |_| true,
    )?;
    let full_paths = full_catalog
        .entries
        .iter()
        .map(|entry| entry.path.as_str())
        .collect::<BTreeSet<_>>();
    let partitioned_paths = mutable_paths
        .union(&immutable_paths)
        .copied()
        .collect::<BTreeSet<_>>();
    ensure!(
        partitioned_paths == full_paths,
        "random-scaffold mutable and immutable catalogs do not partition all model parameters"
    );
    for id in model.random_scaffold_trainable_param_ids() {
        let parameter = by_id.get(id.to_string().as_str()).ok_or_else(|| {
            anyhow::anyhow!("random-scaffold adapter parameter {id} is not in model")
        })?;
        ensure!(
            mutable_paths.contains(parameter.path.as_str()),
            "trainable random-scaffold adapter {} is absent from mutable catalog",
            parameter.path
        );
    }

    let frozen_parameter_count: u64 = inventory
        .parameters
        .iter()
        .filter(|parameter| frozen_paths.contains(parameter.path.as_str()))
        .map(|parameter| parameter.num_elements as u64)
        .sum();
    let expected_frozen_parameter_count = report.frozen_scaffold_elements as u64
        + if trainable_gain {
            0
        } else {
            report.manifest.tensors.len() as u64
        };
    ensure!(
        frozen_parameter_count == expected_frozen_parameter_count,
        "random-scaffold frozen parameter count mismatch: inventory={frozen_parameter_count}, expected={expected_frozen_parameter_count}"
    );
    ensure!(
        immutable_catalog.parameter_count()? == frozen_parameter_count,
        "random-scaffold immutable parameter catalog count mismatch"
    );
    Ok(Some(DragonRandomScaffoldP2pContract {
        catalog,
        immutable_catalog,
        scaffold_contract_hash: ContentId::derive(&report.manifest)
            .context("derive random-scaffold contract hash")?,
        #[cfg(feature = "native")]
        frozen_parameter_count,
    }))
}

#[cfg(feature = "native")]
pub(crate) fn dragon_random_scaffold_p2p_contract<B: Backend>(
    model: &LanguageTrainModel<B>,
    model_schema_hash: ContentId,
) -> Result<Option<DragonRandomScaffoldP2pContract>>
where
    LanguageTrainModel<B>: Module<B>,
{
    dragon_random_scaffold_p2p_contract_for_module(model, &model.model, model_schema_hash)
}

pub(crate) fn random_scaffold_genesis_materialization(
    contract: &DragonRandomScaffoldP2pContract,
) -> Result<GenesisMaterialization> {
    Ok(GenesisMaterialization::DeterministicReconstruction {
        generator_id: burn_eggroll::PORTABLE_SCAFFOLD_GENERATOR_ID.into(),
        reconstruction_contract_hash: contract.scaffold_contract_hash.clone(),
        immutable_parameter_catalog_hash: contract.immutable_catalog.catalog_id()?,
        immutable_parameter_count: contract.immutable_catalog.parameter_count()?,
        mutable_parameter_catalog_hash: contract.catalog.catalog_id()?,
        mutable_parameter_count: contract.catalog.parameter_count()?,
    })
}

fn validate_genesis_materialization(
    materialization: &GenesisMaterialization,
    scaffold: &DragonRandomScaffoldP2pContract,
) -> Result<()> {
    let expected = random_scaffold_genesis_materialization(scaffold)?;
    ensure!(
        materialization == &expected,
        "random-scaffold genesis reconstruction contract mismatch"
    );
    Ok(())
}

#[cfg(feature = "native")]
pub(crate) fn materialize_random_scaffold_genesis<B: AutodiffBackend>(
    context: GenesisArtifactMaterializationContext<'_, LanguageTrainModel<B>>,
    scaffold: &DragonRandomScaffoldP2pContract,
) -> Result<Option<burn_p2p::ArtifactDescriptor>>
where
    LanguageTrainModel<B>: Module<B>,
{
    validate_genesis_materialization(context.materialization, scaffold)?;
    ensure!(
        context.contract.model_schema_hash == scaffold.catalog.model_schema_hash,
        "random-scaffold genesis model schema mismatch"
    );
    let parameter_values =
        flatten_module_float_parameter_subset::<B, _>(context.model, &scaffold.catalog)?;
    let payload = DragonRandomScaffoldGenesisPayload {
        version: RANDOM_SCAFFOLD_GENESIS_VERSION,
        training_contract_id: context.training_contract_id.clone(),
        model_schema_hash: context.contract.model_schema_hash.clone(),
        reconstruction_contract_hash: scaffold.scaffold_contract_hash.clone(),
        immutable_parameter_catalog_hash: scaffold.immutable_catalog.catalog_id()?,
        mutable_parameter_catalog_hash: scaffold.catalog.catalog_id()?,
        mutable_parameter_count: scaffold.catalog.parameter_count()?,
        values: CompactScalarVector::encode(&parameter_values, CompactScalarEncoding::Fp32)?,
    };
    let mut bytes = Vec::new();
    ciborium::ser::into_writer(&payload, &mut bytes)
        .context("encode random-scaffold genesis payload")?;
    let spec = ArtifactBuildSpec::new(
        ArtifactKind::FullHead,
        Precision::Custom("random-scaffold-mutable-fp32".into()),
        context.contract.model_schema_hash.clone(),
        RANDOM_SCAFFOLD_GENESIS_RECORD_FORMAT,
    )
    .with_head(context.head_id);
    let artifact = context.store.store_artifact_reader(
        &spec,
        Cursor::new(bytes),
        ChunkingScheme::new(MUTABLE_SUBSET_CHUNK_BYTES)?,
    )?;
    Ok(Some(artifact))
}

pub(crate) fn decode_random_scaffold_genesis_bytes<B, M>(
    model: M,
    descriptor: &burn_p2p::ArtifactDescriptor,
    bytes: &[u8],
    training_contract_id: &ContentId,
    contract: &burn_p2p::TrainingContractManifest,
    materialization: &GenesisMaterialization,
    scaffold: &DragonRandomScaffoldP2pContract,
) -> Result<M>
where
    B: Backend,
    M: Module<B>,
{
    validate_genesis_materialization(materialization, scaffold)?;
    ensure!(
        descriptor.kind == ArtifactKind::FullHead
            && descriptor.base_head_id.is_none()
            && descriptor.record_format == RANDOM_SCAFFOLD_GENESIS_RECORD_FORMAT
            && descriptor.model_schema_hash == contract.model_schema_hash,
        "random-scaffold genesis artifact descriptor mismatch"
    );
    let payload: DragonRandomScaffoldGenesisPayload = ciborium::de::from_reader(Cursor::new(bytes))
        .context("decode random-scaffold genesis payload")?;
    ensure!(
        payload.version == RANDOM_SCAFFOLD_GENESIS_VERSION
            && payload.training_contract_id == *training_contract_id
            && payload.model_schema_hash == contract.model_schema_hash
            && payload.reconstruction_contract_hash == scaffold.scaffold_contract_hash
            && payload.immutable_parameter_catalog_hash
                == scaffold.immutable_catalog.catalog_id()?
            && payload.mutable_parameter_catalog_hash == scaffold.catalog.catalog_id()?
            && payload.mutable_parameter_count == scaffold.catalog.parameter_count()?
            && payload.values.encoding == CompactScalarEncoding::Fp32,
        "random-scaffold genesis payload contract mismatch"
    );
    let values = payload.values.decode()?;
    replace_module_float_parameter_subset::<B, _>(&model, &scaffold.catalog, &values)
        .map_err(Into::into)
}

pub(crate) fn decode_random_scaffold_head_bytes<B, M>(
    model: M,
    descriptor: &burn_p2p::ArtifactDescriptor,
    bytes: &[u8],
    scaffold: &DragonRandomScaffoldP2pContract,
) -> Result<M>
where
    B: Backend,
    M: Module<B>,
{
    ensure!(
        descriptor.record_format == RANDOM_SCAFFOLD_HEAD_RECORD_FORMAT
            && descriptor.model_schema_hash == scaffold.catalog.model_schema_hash,
        "random-scaffold head artifact descriptor mismatch"
    );
    let payload: DragonRandomScaffoldHeadPayload = ciborium::de::from_reader(Cursor::new(bytes))
        .context("decode random-scaffold head payload")?;
    ensure!(
        payload.version == RANDOM_SCAFFOLD_HEAD_VERSION
            && payload.model_schema_hash == descriptor.model_schema_hash
            && payload.reconstruction_contract_hash == scaffold.scaffold_contract_hash
            && payload.immutable_parameter_catalog_hash
                == scaffold.immutable_catalog.catalog_id()?
            && payload.mutable_parameter_catalog_hash == scaffold.catalog.catalog_id()?
            && payload.mutable_parameter_count == scaffold.catalog.parameter_count()?
            && payload.values.encoding == CompactScalarEncoding::Fp32,
        "random-scaffold head payload contract mismatch"
    );
    let values = payload.values.decode()?;
    replace_module_float_parameter_subset::<B, _>(&model, &scaffold.catalog, &values)
        .map_err(Into::into)
}

#[cfg(feature = "native")]
pub(crate) fn materialize_random_scaffold_head<B: AutodiffBackend>(
    model: &LanguageTrainModel<B>,
    artifact_kind: ArtifactKind,
    head_id: &burn_p2p::HeadId,
    base_head_id: Option<&burn_p2p::HeadId>,
    store: &burn_p2p::FsArtifactStore,
    model_schema_hash: &ContentId,
    scaffold: &DragonRandomScaffoldP2pContract,
) -> Result<Option<burn_p2p::ArtifactDescriptor>>
where
    LanguageTrainModel<B>: Module<B>,
{
    ensure!(
        model_schema_hash == &scaffold.catalog.model_schema_hash,
        "random-scaffold head model schema mismatch"
    );
    let parameter_values = flatten_module_float_parameter_subset::<B, _>(model, &scaffold.catalog)?;
    let payload = DragonRandomScaffoldHeadPayload {
        version: RANDOM_SCAFFOLD_HEAD_VERSION,
        model_schema_hash: model_schema_hash.clone(),
        reconstruction_contract_hash: scaffold.scaffold_contract_hash.clone(),
        immutable_parameter_catalog_hash: scaffold.immutable_catalog.catalog_id()?,
        mutable_parameter_catalog_hash: scaffold.catalog.catalog_id()?,
        mutable_parameter_count: scaffold.catalog.parameter_count()?,
        values: CompactScalarVector::encode(&parameter_values, CompactScalarEncoding::Fp32)?,
    };
    let mut bytes = Vec::new();
    ciborium::ser::into_writer(&payload, &mut bytes)
        .context("encode random-scaffold head payload")?;
    let mut spec = ArtifactBuildSpec::new(
        artifact_kind,
        Precision::Custom("random-scaffold-mutable-fp32".into()),
        model_schema_hash.clone(),
        RANDOM_SCAFFOLD_HEAD_RECORD_FORMAT,
    )
    .with_head(head_id.clone());
    if let Some(base_head_id) = base_head_id {
        spec = spec.with_base_head(base_head_id.clone());
    }
    let artifact = store.store_artifact_reader(
        &spec,
        Cursor::new(bytes),
        ChunkingScheme::new(MUTABLE_SUBSET_CHUNK_BYTES)?,
    )?;
    Ok(Some(artifact))
}

#[cfg(feature = "native")]
pub(crate) fn load_random_scaffold_head<B: AutodiffBackend>(
    model: &LanguageTrainModel<B>,
    descriptor: &burn_p2p::ArtifactDescriptor,
    store: &burn_p2p::FsArtifactStore,
    _device: &B::Device,
    model_schema_hash: &ContentId,
    scaffold: &DragonRandomScaffoldP2pContract,
) -> Result<Option<LanguageTrainModel<B>>>
where
    LanguageTrainModel<B>: Module<B>,
{
    if descriptor.record_format != RANDOM_SCAFFOLD_HEAD_RECORD_FORMAT {
        return Ok(None);
    }
    ensure!(
        &descriptor.model_schema_hash == model_schema_hash,
        "random-scaffold head loader model schema mismatch"
    );
    let bytes = store.materialize_artifact_bytes(descriptor)?;
    decode_random_scaffold_head_bytes::<B, _>(model.clone(), descriptor, &bytes, scaffold).map(Some)
}

#[cfg(feature = "native")]
pub(crate) fn load_random_scaffold_genesis<B: AutodiffBackend>(
    model: LanguageTrainModel<B>,
    context: GenesisArtifactLoadContext<'_, B::Device>,
    scaffold: &DragonRandomScaffoldP2pContract,
) -> Result<Option<LanguageTrainModel<B>>>
where
    LanguageTrainModel<B>: Module<B>,
{
    let bytes = context
        .store
        .materialize_artifact_bytes(context.descriptor)?;
    let model = decode_random_scaffold_genesis_bytes::<B, _>(
        model,
        context.descriptor,
        &bytes,
        context.training_contract_id,
        context.contract,
        context.materialization,
        scaffold,
    )?;
    Ok(Some(model))
}

#[cfg(feature = "native")]
fn codec_encoding(
    contract: &burn_p2p::TrainingContractManifest,
    catalog: &ParameterSubsetCatalog,
) -> Result<CompactScalarEncoding> {
    let UpdateCodec::MutableSubsetParameters {
        parameter_catalog_hash,
        parameter_count,
        encoding,
    } = &contract.update_codec
    else {
        bail!("random-scaffold update requires the mutable-subset codec");
    };
    ensure!(
        parameter_catalog_hash == &catalog.catalog_id()?,
        "random-scaffold mutable parameter catalog hash mismatch"
    );
    ensure!(
        *parameter_count == catalog.parameter_count()?,
        "random-scaffold mutable parameter count mismatch"
    );
    ensure!(
        contract.model_schema_hash == catalog.model_schema_hash,
        "random-scaffold mutable catalog model schema mismatch"
    );
    Ok(*encoding)
}

#[cfg(feature = "native")]
fn update_norm(values: &[f32]) -> Result<UpdateNormStats> {
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "random-scaffold update contains a non-finite value"
    );
    let l2_norm = values
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();
    let max_abs = values
        .iter()
        .map(|value| f64::from(value.abs()))
        .fold(0.0_f64, f64::max);
    Ok(UpdateNormStats {
        l2_norm,
        max_abs,
        clipped: false,
        non_finite_tensors: 0,
    })
}

#[cfg(feature = "native")]
fn norm_stats_match(left: &UpdateNormStats, right: &UpdateNormStats) -> bool {
    fn close(left: f64, right: f64) -> bool {
        let scale = left.abs().max(right.abs()).max(1.0);
        (left - right).abs() <= scale * 1.0e-12
    }

    close(left.l2_norm, right.l2_norm)
        && close(left.max_abs, right.max_abs)
        && left.clipped == right.clipped
        && left.non_finite_tensors == right.non_finite_tensors
}

#[cfg(feature = "native")]
fn verify_catalog_covers_all_changed_parameters<B: Backend>(
    base_model: &LanguageTrainModel<B>,
    trained_model: &LanguageTrainModel<B>,
    catalog: &ParameterSubsetCatalog,
) -> Result<()>
where
    LanguageTrainModel<B>: Module<B>,
{
    let full_catalog = module_float_parameter_subset_catalog::<B, _>(
        base_model,
        catalog.model_schema_hash.clone(),
        |_| true,
    )?;
    let full_delta =
        diff_module_float_parameter_subset::<B, _>(base_model, trained_model, &full_catalog)?;
    let mutable_paths = catalog
        .entries
        .iter()
        .map(|entry| entry.path.as_str())
        .collect::<BTreeSet<_>>();
    let mut offset = 0_usize;
    let mut omitted_drift = Vec::new();
    for entry in &full_catalog.entries {
        let count = entry.shape.iter().try_fold(1_usize, |count, dimension| {
            count.checked_mul(usize::try_from(*dimension).ok()?)
        });
        let count = count.ok_or_else(|| anyhow::anyhow!("parameter shape exceeds usize"))?;
        let end = offset
            .checked_add(count)
            .ok_or_else(|| anyhow::anyhow!("parameter offset overflow"))?;
        let max_abs = full_delta[offset..end]
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f32, f32::max);
        if max_abs > 0.0 && !mutable_paths.contains(entry.path.as_str()) {
            omitted_drift.push((entry.path.clone(), max_abs));
        }
        offset = end;
    }
    ensure!(
        omitted_drift.is_empty(),
        "random-scaffold mutable catalog omitted changed parameters: {omitted_drift:?}"
    );
    Ok(())
}

#[cfg(feature = "native")]
fn decode_mutable_parameters(
    bytes: &[u8],
    envelope: &WorkloadUpdateEnvelope,
    contract: &burn_p2p::TrainingContractManifest,
    catalog: &ParameterSubsetCatalog,
) -> Result<Vec<f32>> {
    codec_encoding(contract, catalog)?;
    let update =
        burn_p2p_workload::decode_compact_update(bytes, &envelope.training_contract_id, contract)?;
    ensure!(
        update.payload.parameter_catalog_hash == catalog.catalog_id()?,
        "random-scaffold payload parameter catalog mismatch"
    );
    let CompactUpdateBody::MutableSubsetParameters { values } = update.payload.body else {
        bail!("random-scaffold payload has a different compact-update body");
    };
    let values = values.decode().context("decode mutable-subset values")?;
    ensure!(
        values.len() as u64 == catalog.parameter_count()?,
        "random-scaffold decoded mutable parameter count mismatch"
    );
    Ok(values)
}

#[cfg(feature = "native")]
pub(crate) fn materialize_random_scaffold_update<B: AutodiffBackend>(
    context: WorkloadUpdateMaterializationContext<'_, B::Device, LanguageTrainModel<B>>,
    catalog: &ParameterSubsetCatalog,
) -> Result<Option<MaterializedWorkloadUpdate>>
where
    LanguageTrainModel<B>: Module<B>,
{
    let encoding = codec_encoding(context.contract, catalog)?;
    if std::env::var(VERIFY_CATALOG_ENV)
        .ok()
        .is_some_and(|value| matches!(value.trim(), "1" | "true" | "yes" | "on"))
    {
        verify_catalog_covers_all_changed_parameters::<B>(
            context.base_model,
            context.trained_model,
            catalog,
        )?;
    }
    let parameter_values =
        flatten_module_float_parameter_subset::<B, _>(context.trained_model, catalog)?;
    let encoded = CompactScalarVector::encode(&parameter_values, encoding)
        .context("encode random-scaffold mutable parameters")?;
    let decoded = encoded
        .decode()
        .context("decode local random-scaffold update for telemetry")?;
    let base_values = flatten_module_float_parameter_subset::<B, _>(context.base_model, catalog)?;
    let decoded_delta = decoded
        .iter()
        .zip(base_values)
        .map(|(value, base)| value - base)
        .collect::<Vec<_>>();
    let norm_stats = update_norm(&decoded_delta)?;
    let decoded_model =
        replace_module_float_parameter_subset::<B, _>(context.base_model, catalog, &decoded)?;
    let decoded_tensor_digest =
        module_tensor_digest::<B, _>(&decoded_model, context.contract.model_schema_hash.clone())?;
    let payload = CompactUpdatePayload {
        version: COMPACT_UPDATE_PAYLOAD_VERSION,
        training_contract_id: context.training_contract_id.clone(),
        model_schema_hash: context.contract.model_schema_hash.clone(),
        parameter_catalog_hash: catalog.catalog_id()?,
        parameter_count: catalog.parameter_count()?,
        body: CompactUpdateBody::MutableSubsetParameters { values: encoded },
    };
    let bytes = burn_p2p_workload::encode_compact_update(
        &payload,
        context.training_contract_id,
        context.contract,
    )
    .context("encode random-scaffold compact update")?;
    let precision = match encoding {
        CompactScalarEncoding::Fp32 => "mutable-subset-fp32",
        CompactScalarEncoding::SymmetricInt8 => "mutable-subset-int8",
        CompactScalarEncoding::SymmetricInt16 => "mutable-subset-int16",
    };
    let spec = ArtifactBuildSpec::new(
        ArtifactKind::DeltaPack,
        Precision::Custom(precision.into()),
        context.contract.model_schema_hash.clone(),
        MUTABLE_SUBSET_RECORD_FORMAT,
    )
    .with_head(context.candidate_head_id.clone())
    .with_base_head(context.base_head_id.clone());
    let artifact = context.store.store_artifact_reader(
        &spec,
        Cursor::new(bytes),
        ChunkingScheme::new(MUTABLE_SUBSET_CHUNK_BYTES)?,
    )?;
    let envelope = WorkloadUpdateEnvelope {
        training_contract_id: context.training_contract_id.clone(),
        revision_id: context.revision_id.clone(),
        base_head_id: context.base_head_id.clone(),
        window_id: context.window_id,
        lease_id: context.lease_id.clone(),
        codec: context.contract.update_codec.clone(),
        routing_context: None,
        artifact: artifact.clone(),
        decoded_tensor_digest: Some(decoded_tensor_digest),
        claimed_norm_stats: Some(norm_stats),
        claimed_feature_sketch: None,
    };
    envelope
        .validate_against(context.training_contract_id, context.contract)
        .context("validate local random-scaffold update envelope")?;
    Ok(Some(MaterializedWorkloadUpdate { artifact, envelope }))
}

#[cfg(feature = "native")]
pub(crate) fn apply_random_scaffold_update<B: AutodiffBackend>(
    base_model: LanguageTrainModel<B>,
    descriptor: &burn_p2p::ArtifactDescriptor,
    envelope: &WorkloadUpdateEnvelope,
    contract: &burn_p2p::TrainingContractManifest,
    store: &burn_p2p::FsArtifactStore,
    catalog: &ParameterSubsetCatalog,
) -> Result<LanguageTrainModel<B>>
where
    LanguageTrainModel<B>: Module<B>,
{
    let bytes = store.materialize_artifact_bytes(descriptor)?;
    let values = decode_mutable_parameters(&bytes, envelope, contract, catalog)?;
    replace_module_float_parameter_subset::<B, _>(&base_model, catalog, &values).map_err(Into::into)
}

#[cfg(feature = "native")]
pub(crate) fn validate_random_scaffold_update<B: AutodiffBackend>(
    base_model: LanguageTrainModel<B>,
    context: WorkloadUpdateValidationContext<'_, B::Device>,
    catalog: &ParameterSubsetCatalog,
) -> Result<ValidatedWorkloadUpdate<LanguageTrainModel<B>>>
where
    LanguageTrainModel<B>: Module<B>,
{
    let bytes = context
        .store
        .materialize_artifact_bytes(context.descriptor)?;
    let values = decode_mutable_parameters(&bytes, context.update, context.contract, catalog)?;
    let base_values = flatten_module_float_parameter_subset::<B, _>(&base_model, catalog)?;
    let delta = values
        .iter()
        .zip(base_values)
        .map(|(value, base)| value - base)
        .collect::<Vec<_>>();
    let norm_stats = update_norm(&delta)?;
    if let Some(claimed) = &context.update.claimed_norm_stats {
        ensure!(
            norm_stats_match(claimed, &norm_stats),
            "random-scaffold update norm claim does not match decoded payload"
        );
    }
    let model = replace_module_float_parameter_subset::<B, _>(&base_model, catalog, &values)?;
    let tensor_digest =
        module_tensor_digest::<B, _>(&model, context.contract.model_schema_hash.clone())?;
    if let Some(expected) = &context.update.decoded_tensor_digest {
        ensure!(
            expected == &tensor_digest,
            "random-scaffold reconstructed model digest mismatch"
        );
    }
    Ok(ValidatedWorkloadUpdate {
        model,
        evidence: ValidatedUpdateEvidence {
            update_envelope_id: ContentId::derive(context.update)?,
            norm_stats: Some(norm_stats),
            feature_sketch: None,
            reconstruction_verified: true,
            replay_verified: true,
            replay_stats: None,
            validator_peer_id: context.replay.validator_peer_id.clone(),
            validated_at: chrono::Utc::now(),
        },
    })
}

#[cfg(all(test, feature = "native"))]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use burn::backend::{Autodiff, NdArray};
    use burn_dragon_core::{DragonConfig, DragonModel};
    use burn_p2p::burn_module::{
        flatten_module_float_parameter_subset, module_tensor_digest,
        replace_module_float_parameter_subset,
    };
    use burn_p2p::{
        AssignmentLease, DatasetViewId, ExperimentId, LeaseId, LocalOptimizerStatePolicy,
        MicroShardId, NetworkId, PeerId, RecurrentStatePolicy, RevisionId, SchedulerStatePolicy,
        StudyId, TRAINING_CONTRACT_VERSION, TrainingContractManifest, WindowId, WorkloadId,
        WorkloadUpdateReplayContext,
    };
    use chrono::{Duration, Utc};
    use tempfile::tempdir;

    use super::*;

    type TestBackend = Autodiff<NdArray<f32>>;

    fn model_with_trainable_gain(trainable_gain: bool) -> LanguageTrainModel<TestBackend> {
        let device = burn::tensor::Device::<TestBackend>::default();
        let mut config = DragonConfig {
            n_layer: 1,
            n_embd: 16,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 32,
            dropout: 0.0,
            ..DragonConfig::default()
        };
        config.random_scaffold.enabled = true;
        config.random_scaffold.seed = 17;
        config.random_scaffold.rank = 4;
        config.random_scaffold.alpha = 16.0;
        config.random_scaffold.trainable_gain = trainable_gain;
        LanguageTrainModel::new(DragonModel::new(config, &device))
    }

    fn model() -> LanguageTrainModel<TestBackend> {
        model_with_trainable_gain(true)
    }

    fn contract_with_encoding(
        catalog: &ParameterSubsetCatalog,
        encoding: CompactScalarEncoding,
    ) -> TrainingContractManifest {
        TrainingContractManifest {
            version: TRAINING_CONTRACT_VERSION,
            workload_id: WorkloadId::new("dragon-test"),
            model_program_hash: ContentId::new("program"),
            model_schema_hash: catalog.model_schema_hash.clone(),
            checkpoint_format_hash: ContentId::new("checkpoint"),
            dataset_view_id: DatasetViewId::new("dataset"),
            tokenizer_hash: ContentId::new("tokenizer"),
            preprocessing_hash: ContentId::new("preprocess"),
            objective_hash: ContentId::new("objective"),
            optimizer_hash: ContentId::new("optimizer"),
            scheduler_hash: ContentId::new("scheduler"),
            optimizer_state_policy: LocalOptimizerStatePolicy::ResetPerWindow,
            scheduler_state_policy: SchedulerStatePolicy::ResetPerWindow,
            recurrent_state_policy: RecurrentStatePolicy::LeaseScoped,
            update_codec: UpdateCodec::MutableSubsetParameters {
                parameter_catalog_hash: catalog.catalog_id().expect("catalog id"),
                parameter_count: catalog.parameter_count().expect("parameter count"),
                encoding,
            },
            aggregation_hash: ContentId::new("aggregation"),
            validation_hash: ContentId::new("validation"),
            initialization_hash: ContentId::new("initialization"),
            extensions: BTreeMap::new(),
        }
    }

    fn contract(catalog: &ParameterSubsetCatalog) -> TrainingContractManifest {
        contract_with_encoding(catalog, CompactScalarEncoding::Fp32)
    }

    fn lease(now: chrono::DateTime<Utc>) -> AssignmentLease {
        AssignmentLease {
            lease_id: LeaseId::new("lease"),
            network_id: NetworkId::new("network"),
            study_id: StudyId::new("study"),
            experiment_id: ExperimentId::new("experiment"),
            revision_id: RevisionId::new("revision"),
            peer_id: PeerId::new("trainer"),
            dataset_view_id: DatasetViewId::new("dataset"),
            window_id: WindowId(1),
            granted_at: now,
            expires_at: now + Duration::minutes(1),
            budget_work_units: 1,
            microshards: vec![MicroShardId::new("shard")],
            assignment_hash: ContentId::new("assignment"),
        }
    }

    fn max_abs_difference(left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn scaffold_catalog_contains_adapters_and_excludes_immutable_backbone() {
        let model = model();
        let scaffold = dragon_random_scaffold_p2p_contract(&model, ContentId::new("model-schema"))
            .expect("scaffold contract")
            .expect("enabled scaffold");

        assert_eq!(
            scaffold.frozen_parameter_count,
            model
                .model
                .random_scaffold_report()
                .expect("report")
                .frozen_scaffold_elements as u64
        );
        assert!(
            scaffold
                .catalog
                .entries
                .iter()
                .any(|entry| entry.path.contains("random_scaffold_adapters"))
        );
        let inventory = inspect_module::<TestBackend, _>(&model);
        let frozen_ids = model
            .model
            .random_scaffold_frozen_param_ids()
            .into_iter()
            .map(|id| id.to_string())
            .collect::<BTreeSet<_>>();
        assert!(inventory.parameters.iter().all(|parameter| {
            !frozen_ids.contains(&parameter.param_id)
                || !scaffold
                    .catalog
                    .entries
                    .iter()
                    .any(|entry| entry.path == parameter.path)
        }));
    }

    #[test]
    fn fixed_gains_are_immutable_and_catalogs_partition_every_parameter() {
        let model = model_with_trainable_gain(false);
        let schema = ContentId::new("model-schema");
        let scaffold = dragon_random_scaffold_p2p_contract(&model, schema.clone())
            .expect("scaffold contract")
            .expect("enabled scaffold");
        let full_catalog =
            module_float_parameter_subset_catalog::<TestBackend, _>(&model, schema, |_| true)
                .expect("full catalog");
        assert_eq!(
            scaffold.catalog.parameter_count().expect("mutable count")
                + scaffold
                    .immutable_catalog
                    .parameter_count()
                    .expect("immutable count"),
            full_catalog.parameter_count().expect("full count")
        );
        assert!(
            scaffold
                .immutable_catalog
                .entries
                .iter()
                .any(|entry| entry.path.ends_with(".gain"))
        );
        assert!(
            scaffold
                .catalog
                .entries
                .iter()
                .all(|entry| !entry.path.ends_with(".gain"))
        );
        let report = model.model.random_scaffold_report().expect("report");
        assert_eq!(
            scaffold.frozen_parameter_count,
            report.frozen_scaffold_elements as u64 + report.manifest.tensors.len() as u64
        );
    }

    #[test]
    fn deterministic_genesis_transmits_only_mutable_state_and_round_trips_exactly() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let model = model();
        let scaffold = dragon_random_scaffold_p2p_contract(&model, ContentId::new("model-schema"))
            .expect("scaffold contract")
            .expect("enabled scaffold");
        let training_contract = contract(&scaffold.catalog);
        let training_contract_id = training_contract.contract_id().expect("contract id");
        let materialization =
            random_scaffold_genesis_materialization(&scaffold).expect("materialization");
        let root = tempdir().expect("artifact root");
        let store = burn_p2p::FsArtifactStore::new(root.path());
        let descriptor = materialize_random_scaffold_genesis::<TestBackend>(
            GenesisArtifactMaterializationContext {
                model: &model,
                head_id: burn_p2p::HeadId::new("genesis"),
                training_contract_id: &training_contract_id,
                contract: &training_contract,
                materialization: &materialization,
                store: &store,
            },
            &scaffold,
        )
        .expect("materialize genesis")
        .expect("deterministic genesis");
        let bytes = store
            .materialize_artifact_bytes(&descriptor)
            .expect("genesis bytes");
        let mutable_bytes = usize::try_from(scaffold.catalog.parameter_count().expect("count"))
            .expect("usize")
            * core::mem::size_of::<f32>();
        assert!(bytes.len() >= mutable_bytes);
        assert!(bytes.len() < mutable_bytes + 4096);
        assert!(
            bytes.len() < model.num_params() * core::mem::size_of::<f32>(),
            "deterministic genesis should omit immutable scaffold tensors"
        );

        let loaded = load_random_scaffold_genesis::<TestBackend>(
            model.clone(),
            GenesisArtifactLoadContext {
                descriptor: &descriptor,
                training_contract_id: &training_contract_id,
                contract: &training_contract,
                materialization: &materialization,
                store: &store,
                device: &device,
            },
            &scaffold,
        )
        .expect("load genesis")
        .expect("deterministic genesis");
        assert_eq!(
            module_tensor_digest::<TestBackend, _>(
                &model,
                training_contract.model_schema_hash.clone(),
            )
            .expect("source digest"),
            module_tensor_digest::<TestBackend, _>(
                &loaded,
                training_contract.model_schema_hash.clone(),
            )
            .expect("loaded digest"),
        );
    }

    #[test]
    fn canonical_scaffold_heads_ignore_ephemeral_param_ids_and_round_trip_exactly() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let first = model();
        let second = model();
        let schema = ContentId::new("model-schema");
        let first_contract = dragon_random_scaffold_p2p_contract(&first, schema.clone())
            .expect("first scaffold contract")
            .expect("enabled scaffold");
        let second_contract = dragon_random_scaffold_p2p_contract(&second, schema.clone())
            .expect("second scaffold contract")
            .expect("enabled scaffold");
        assert_eq!(
            first_contract.catalog.catalog_id().expect("first catalog"),
            second_contract
                .catalog
                .catalog_id()
                .expect("second catalog")
        );
        let first_values = flatten_module_float_parameter_subset::<TestBackend, _>(
            &first,
            &first_contract.catalog,
        )
        .expect("first mutable values");
        let second = replace_module_float_parameter_subset::<TestBackend, _>(
            &second,
            &second_contract.catalog,
            &first_values,
        )
        .expect("align mutable values");
        let expected_digest =
            module_tensor_digest::<TestBackend, _>(&first, schema.clone()).expect("first digest");
        assert_eq!(
            expected_digest,
            module_tensor_digest::<TestBackend, _>(&second, schema.clone()).expect("second digest")
        );

        let first_root = tempdir().expect("first artifact root");
        let second_root = tempdir().expect("second artifact root");
        let first_store = burn_p2p::FsArtifactStore::new(first_root.path());
        let second_store = burn_p2p::FsArtifactStore::new(second_root.path());
        let head_id = burn_p2p::HeadId::new("canonical-head");
        let base_head_id = burn_p2p::HeadId::new("base-head");
        let first_artifact = materialize_random_scaffold_head::<TestBackend>(
            &first,
            ArtifactKind::FullHead,
            &head_id,
            Some(&base_head_id),
            &first_store,
            &schema,
            &first_contract,
        )
        .expect("first compact head")
        .expect("handled scaffold head");
        let second_artifact = materialize_random_scaffold_head::<TestBackend>(
            &second,
            ArtifactKind::FullHead,
            &head_id,
            Some(&base_head_id),
            &second_store,
            &schema,
            &second_contract,
        )
        .expect("second compact head")
        .expect("handled scaffold head");
        assert_eq!(first_artifact, second_artifact);
        assert!(
            first_artifact.bytes_len < (first.num_params() * core::mem::size_of::<f32>()) as u64
        );

        let loaded = load_random_scaffold_head::<TestBackend>(
            &model(),
            &first_artifact,
            &first_store,
            &device,
            &schema,
            &first_contract,
        )
        .expect("load compact head")
        .expect("handled scaffold head");
        assert_eq!(
            expected_digest,
            module_tensor_digest::<TestBackend, _>(&loaded, schema).expect("loaded digest")
        );
    }

    #[test]
    fn fp32_mutable_update_round_trip_is_exact_and_digest_verified() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let base = model();
        let scaffold = dragon_random_scaffold_p2p_contract(&base, ContentId::new("model-schema"))
            .expect("scaffold contract")
            .expect("enabled scaffold");
        let base_values =
            flatten_module_float_parameter_subset::<TestBackend, _>(&base, &scaffold.catalog)
                .expect("base values");
        let trained_values = base_values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                value
                    + if index.is_multiple_of(2) {
                        1.0e-3
                    } else {
                        -1.0e-3
                    }
            })
            .collect::<Vec<_>>();
        let trained = replace_module_float_parameter_subset::<TestBackend, _>(
            &base,
            &scaffold.catalog,
            &trained_values,
        )
        .expect("trained fixture");
        let contract = contract(&scaffold.catalog);
        let contract_id = contract.contract_id().expect("contract id");
        let run_dir = tempdir().expect("run dir");
        let store = burn_p2p::FsArtifactStore::new(run_dir.path());
        let materialized = materialize_random_scaffold_update::<TestBackend>(
            WorkloadUpdateMaterializationContext {
                base_model: &base,
                trained_model: &trained,
                training_contract_id: &contract_id,
                contract: &contract,
                revision_id: &RevisionId::new("revision"),
                base_head_id: &burn_p2p::HeadId::new("base"),
                candidate_head_id: &burn_p2p::HeadId::new("candidate"),
                window_id: WindowId(1),
                lease_id: &LeaseId::new("lease"),
                store: &store,
                device: &device,
            },
            &scaffold.catalog,
        )
        .expect("materialize update")
        .expect("typed update");
        assert_eq!(materialized.artifact.kind, ArtifactKind::DeltaPack);

        let now = Utc::now();
        let assignment = lease(now);
        let validator = PeerId::new("validator");
        let validated = validate_random_scaffold_update::<TestBackend>(
            base.clone(),
            WorkloadUpdateValidationContext {
                descriptor: &materialized.artifact,
                update: &materialized.envelope,
                contract: &contract,
                store: &store,
                device: &device,
                replay: WorkloadUpdateReplayContext {
                    lease: &assignment,
                    cached_microshards: &[],
                    validator_peer_id: &validator,
                },
            },
            &scaffold.catalog,
        )
        .expect("validate update");
        let reconstructed = flatten_module_float_parameter_subset::<TestBackend, _>(
            &validated.model,
            &scaffold.catalog,
        )
        .expect("reconstructed values");
        assert_eq!(max_abs_difference(&trained_values, &reconstructed), 0.0);
        assert_eq!(
            module_tensor_digest::<TestBackend, _>(
                &validated.model,
                contract.model_schema_hash.clone()
            )
            .expect("reconstructed digest"),
            materialized
                .envelope
                .decoded_tensor_digest
                .expect("declared digest")
        );
        let dense_fp32_bytes = inspect_module::<TestBackend, _>(&trained)
            .total_scalar_parameters
            .saturating_mul(size_of::<f32>()) as u64;
        assert!(
            materialized.artifact.bytes_len < dense_fp32_bytes,
            "compact update {} bytes should be smaller than dense fp32 {} bytes",
            materialized.artifact.bytes_len,
            dense_fp32_bytes
        );
    }

    #[test]
    fn int16_mutable_update_binds_digest_to_quantized_reconstruction() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let base = model();
        let scaffold = dragon_random_scaffold_p2p_contract(&base, ContentId::new("model-schema"))
            .expect("scaffold contract")
            .expect("enabled scaffold");
        let base_values =
            flatten_module_float_parameter_subset::<TestBackend, _>(&base, &scaffold.catalog)
                .expect("base values");
        let trained_values = base_values
            .iter()
            .enumerate()
            .map(|(index, value)| value + (index % 17) as f32 * 1.0e-4)
            .collect::<Vec<_>>();
        let trained = replace_module_float_parameter_subset::<TestBackend, _>(
            &base,
            &scaffold.catalog,
            &trained_values,
        )
        .expect("trained fixture");
        let contract =
            contract_with_encoding(&scaffold.catalog, CompactScalarEncoding::SymmetricInt16);
        let contract_id = contract.contract_id().expect("contract id");
        let run_dir = tempdir().expect("run dir");
        let store = burn_p2p::FsArtifactStore::new(run_dir.path());
        let materialized = materialize_random_scaffold_update::<TestBackend>(
            WorkloadUpdateMaterializationContext {
                base_model: &base,
                trained_model: &trained,
                training_contract_id: &contract_id,
                contract: &contract,
                revision_id: &RevisionId::new("revision"),
                base_head_id: &burn_p2p::HeadId::new("base"),
                candidate_head_id: &burn_p2p::HeadId::new("candidate"),
                window_id: WindowId(1),
                lease_id: &LeaseId::new("lease"),
                store: &store,
                device: &device,
            },
            &scaffold.catalog,
        )
        .expect("materialize update")
        .expect("typed update");
        let assignment = lease(Utc::now());
        let validator = PeerId::new("validator");
        let validated = validate_random_scaffold_update::<TestBackend>(
            base,
            WorkloadUpdateValidationContext {
                descriptor: &materialized.artifact,
                update: &materialized.envelope,
                contract: &contract,
                store: &store,
                device: &device,
                replay: WorkloadUpdateReplayContext {
                    lease: &assignment,
                    cached_microshards: &[],
                    validator_peer_id: &validator,
                },
            },
            &scaffold.catalog,
        )
        .expect("validate update");
        let reconstructed = flatten_module_float_parameter_subset::<TestBackend, _>(
            &validated.model,
            &scaffold.catalog,
        )
        .expect("reconstructed values");
        let quantization_error = max_abs_difference(&trained_values, &reconstructed);
        assert!(quantization_error > 0.0);
        assert!(quantization_error < 1.0e-4);
        assert_eq!(
            module_tensor_digest::<TestBackend, _>(
                &validated.model,
                contract.model_schema_hash.clone()
            )
            .expect("reconstructed digest"),
            materialized
                .envelope
                .decoded_tensor_digest
                .expect("declared digest")
        );
    }

    #[test]
    fn validator_rejects_a_false_reconstructed_model_digest() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let base = model();
        let scaffold = dragon_random_scaffold_p2p_contract(&base, ContentId::new("model-schema"))
            .expect("scaffold contract")
            .expect("enabled scaffold");
        let trained_values =
            flatten_module_float_parameter_subset::<TestBackend, _>(&base, &scaffold.catalog)
                .expect("base values")
                .into_iter()
                .map(|value| value + 1.0e-3)
                .collect::<Vec<_>>();
        let trained = replace_module_float_parameter_subset::<TestBackend, _>(
            &base,
            &scaffold.catalog,
            &trained_values,
        )
        .expect("trained fixture");
        let contract = contract(&scaffold.catalog);
        let contract_id = contract.contract_id().expect("contract id");
        let run_dir = tempdir().expect("run dir");
        let store = burn_p2p::FsArtifactStore::new(run_dir.path());
        let mut materialized = materialize_random_scaffold_update::<TestBackend>(
            WorkloadUpdateMaterializationContext {
                base_model: &base,
                trained_model: &trained,
                training_contract_id: &contract_id,
                contract: &contract,
                revision_id: &RevisionId::new("revision"),
                base_head_id: &burn_p2p::HeadId::new("base"),
                candidate_head_id: &burn_p2p::HeadId::new("candidate"),
                window_id: WindowId(1),
                lease_id: &LeaseId::new("lease"),
                store: &store,
                device: &device,
            },
            &scaffold.catalog,
        )
        .expect("materialize update")
        .expect("typed update");
        materialized.envelope.decoded_tensor_digest = Some(ContentId::new("false-digest"));
        let now = Utc::now();
        let assignment = lease(now);
        let validator = PeerId::new("validator");
        let error = match validate_random_scaffold_update::<TestBackend>(
            base,
            WorkloadUpdateValidationContext {
                descriptor: &materialized.artifact,
                update: &materialized.envelope,
                contract: &contract,
                store: &store,
                device: &device,
                replay: WorkloadUpdateReplayContext {
                    lease: &assignment,
                    cached_microshards: &[],
                    validator_peer_id: &validator,
                },
            },
            &scaffold.catalog,
        ) {
            Ok(_) => panic!("false model digest must fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("digest mismatch"));
    }
}
