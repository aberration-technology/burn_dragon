use anyhow::{Result, anyhow, bail};
use burn::backend::NdArray;
use burn::module::Module;
use burn::record::{
    BinBytesRecorder, FullPrecisionSettings, HalfPrecisionSettings, NamedMpkBytesRecorder, Recorder,
};
use burn::tensor::backend::Backend;
use burn_dragon_core::{DragonConfig, DragonModel};
use burn_p2p::{
    ArtifactDescriptor, ContentId, GenesisMaterialization, Precision, TrainingContractManifest,
};
use log::info;

use crate::random_scaffold::{
    DragonRandomScaffoldP2pContract, RANDOM_SCAFFOLD_HEAD_RECORD_FORMAT,
    decode_random_scaffold_genesis_bytes, decode_random_scaffold_head_bytes,
    dragon_random_scaffold_p2p_contract_for_module,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BrowserBurnRecordBytesFormat {
    Bin,
    NamedMpk,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BrowserBurnRecordPrecision {
    Full,
    Half,
}

#[derive(Module, Debug)]
pub(crate) struct BrowserNativeTrainModelArtifact<B: Backend> {
    pub(crate) model: DragonModel<B>,
}

pub(crate) fn browser_random_scaffold_contract<B: Backend>(
    model: &DragonModel<B>,
    model_schema_hash: ContentId,
) -> Result<Option<DragonRandomScaffoldP2pContract>>
where
    DragonModel<B>: Module<B>,
{
    let wrapped = BrowserNativeTrainModelArtifact {
        model: model.clone(),
    };
    dragon_random_scaffold_p2p_contract_for_module(&wrapped, model, model_schema_hash)
}

pub(crate) async fn flatten_browser_random_scaffold_mutable<B: Backend>(
    model: &DragonModel<B>,
    scaffold: &DragonRandomScaffoldP2pContract,
) -> Result<Vec<f32>>
where
    DragonModel<B>: Module<B>,
{
    let wrapped = BrowserNativeTrainModelArtifact {
        model: model.clone(),
    };
    burn_p2p::burn_module::flatten_module_float_parameter_subset_async::<B, _>(
        &wrapped,
        &scaffold.catalog,
    )
    .await
    .map_err(Into::into)
}

pub(crate) fn browser_random_scaffold_tensor_digest_from_mutable(
    model_config: &DragonConfig,
    model_schema_hash: ContentId,
    values: &[f32],
) -> Result<ContentId> {
    type DigestBackend = NdArray<f32>;

    let device = burn::tensor::Device::<DigestBackend>::default();
    let wrapped = BrowserNativeTrainModelArtifact {
        model: DragonModel::<DigestBackend>::new(model_config.clone(), &device),
    };
    let scaffold = dragon_random_scaffold_p2p_contract_for_module(
        &wrapped,
        &wrapped.model,
        model_schema_hash.clone(),
    )?
    .ok_or_else(|| anyhow!("mutable scaffold digest requires random-scaffold mode"))?;
    let wrapped = burn_p2p::burn_module::replace_module_float_parameter_subset::<DigestBackend, _>(
        &wrapped,
        &scaffold.catalog,
        values,
    )?;
    burn_p2p::burn_module::module_tensor_digest::<DigestBackend, _>(&wrapped, model_schema_hash)
        .map_err(Into::into)
}

pub(crate) fn browser_record_bytes_format(
    record_format: &str,
) -> Result<BrowserBurnRecordBytesFormat> {
    match record_format {
        "burn-record:bytes-mpk" => Ok(BrowserBurnRecordBytesFormat::NamedMpk),
        "burn-record:bytes-bin" => Ok(BrowserBurnRecordBytesFormat::Bin),
        other => bail!("browser active head artifact format {other} is not supported"),
    }
}

pub(crate) fn browser_record_precision(
    precision: &Precision,
) -> Result<BrowserBurnRecordPrecision> {
    match precision {
        Precision::Fp32 => Ok(BrowserBurnRecordPrecision::Full),
        Precision::Fp16 => Ok(BrowserBurnRecordPrecision::Half),
        other => bail!("browser active head artifact precision {other:?} is not supported"),
    }
}

pub(crate) fn browser_record_precision_descriptor(
    precision: BrowserBurnRecordPrecision,
) -> Precision {
    match precision {
        BrowserBurnRecordPrecision::Full => Precision::Fp32,
        BrowserBurnRecordPrecision::Half => Precision::Fp16,
    }
}

pub(crate) fn browser_record_format_name(format: BrowserBurnRecordBytesFormat) -> &'static str {
    match format {
        BrowserBurnRecordBytesFormat::Bin => "burn-record:bytes-bin",
        BrowserBurnRecordBytesFormat::NamedMpk => "burn-record:bytes-mpk",
    }
}

pub(crate) fn encode_browser_record_bytes<B, M>(
    module: M,
    format: BrowserBurnRecordBytesFormat,
    precision: BrowserBurnRecordPrecision,
) -> Result<Vec<u8>>
where
    B: Backend,
    M: Module<B>,
{
    match (format, precision) {
        (BrowserBurnRecordBytesFormat::Bin, BrowserBurnRecordPrecision::Full) => {
            record_browser_module::<B, M, BinBytesRecorder<FullPrecisionSettings>>(module)
        }
        (BrowserBurnRecordBytesFormat::Bin, BrowserBurnRecordPrecision::Half) => {
            record_browser_module::<B, M, BinBytesRecorder<HalfPrecisionSettings>>(module)
        }
        (BrowserBurnRecordBytesFormat::NamedMpk, BrowserBurnRecordPrecision::Full) => {
            record_browser_module::<B, M, NamedMpkBytesRecorder<FullPrecisionSettings>>(module)
        }
        (BrowserBurnRecordBytesFormat::NamedMpk, BrowserBurnRecordPrecision::Half) => {
            record_browser_module::<B, M, NamedMpkBytesRecorder<HalfPrecisionSettings>>(module)
        }
    }
}

fn record_browser_module<B, M, R>(module: M) -> Result<Vec<u8>>
where
    B: Backend,
    M: Module<B>,
    R: Recorder<B, RecordArgs = (), RecordOutput = Vec<u8>, LoadArgs = Vec<u8>>,
{
    R::default()
        .record(module.into_record(), ())
        .map_err(|error| anyhow!("failed to encode browser model record: {error}"))
}

fn load_browser_record_bytes<B, R>(
    model: DragonModel<B>,
    bytes: Vec<u8>,
    device: &B::Device,
) -> Result<DragonModel<B>>
where
    B: Backend,
    R: Recorder<B, RecordArgs = (), RecordOutput = Vec<u8>, LoadArgs = Vec<u8>>,
{
    match R::default().load(bytes.clone(), device) {
        Ok(record) => Ok(model.load_record(record)),
        Err(direct_error) => {
            let wrapped = BrowserNativeTrainModelArtifact { model };
            let record = R::default().load(bytes, device).map_err(|wrapped_error| {
                anyhow!(
                    "failed to decode browser model record as DragonModel or native training wrapper: direct={direct_error}; wrapped={wrapped_error}"
                )
            })?;
            let loaded = wrapped.load_record(record);
            info!("browser active head artifact decoded as native training wrapper");
            Ok(loaded.model)
        }
    }
}

pub(crate) fn load_browser_active_head_model<B>(
    model: DragonModel<B>,
    descriptor: &ArtifactDescriptor,
    bytes: Vec<u8>,
    device: &B::Device,
) -> Result<DragonModel<B>>
where
    B: Backend,
    DragonModel<B>: Module<B>,
{
    if descriptor.record_format == RANDOM_SCAFFOLD_HEAD_RECORD_FORMAT {
        let wrapped = BrowserNativeTrainModelArtifact { model };
        let scaffold = dragon_random_scaffold_p2p_contract_for_module(
            &wrapped,
            &wrapped.model,
            descriptor.model_schema_hash.clone(),
        )?
        .ok_or_else(|| anyhow!("compact scaffold head requires random-scaffold mode"))?;
        return Ok(decode_random_scaffold_head_bytes::<B, _>(
            wrapped, descriptor, &bytes, &scaffold,
        )?
        .model);
    }
    let format = browser_record_bytes_format(&descriptor.record_format)?;
    let precision = browser_record_precision(&descriptor.precision)?;
    match (format, precision) {
        (BrowserBurnRecordBytesFormat::Bin, BrowserBurnRecordPrecision::Full) => {
            load_browser_record_bytes::<B, BinBytesRecorder<FullPrecisionSettings>>(
                model, bytes, device,
            )
        }
        (BrowserBurnRecordBytesFormat::Bin, BrowserBurnRecordPrecision::Half) => {
            load_browser_record_bytes::<B, BinBytesRecorder<HalfPrecisionSettings>>(
                model, bytes, device,
            )
        }
        (BrowserBurnRecordBytesFormat::NamedMpk, BrowserBurnRecordPrecision::Full) => {
            load_browser_record_bytes::<B, NamedMpkBytesRecorder<FullPrecisionSettings>>(
                model, bytes, device,
            )
        }
        (BrowserBurnRecordBytesFormat::NamedMpk, BrowserBurnRecordPrecision::Half) => {
            load_browser_record_bytes::<B, NamedMpkBytesRecorder<HalfPrecisionSettings>>(
                model, bytes, device,
            )
        }
    }
}

pub(crate) fn load_browser_genesis_model<B>(
    model: DragonModel<B>,
    descriptor: &ArtifactDescriptor,
    bytes: Vec<u8>,
    training_contract_id: &ContentId,
    contract: &TrainingContractManifest,
    materialization: &GenesisMaterialization,
    device: &B::Device,
) -> Result<DragonModel<B>>
where
    B: Backend,
    DragonModel<B>: Module<B>,
{
    match materialization {
        GenesisMaterialization::FullArtifact => {
            load_browser_active_head_model(model, descriptor, bytes, device)
        }
        GenesisMaterialization::DeterministicReconstruction { .. } => {
            let wrapped = BrowserNativeTrainModelArtifact { model };
            let scaffold = dragon_random_scaffold_p2p_contract_for_module(
                &wrapped,
                &wrapped.model,
                contract.model_schema_hash.clone(),
            )?
            .ok_or_else(|| anyhow!("signed deterministic genesis requires random-scaffold mode"))?;
            let wrapped = decode_random_scaffold_genesis_bytes::<B, _>(
                wrapped,
                descriptor,
                &bytes,
                training_contract_id,
                contract,
                materialization,
                &scaffold,
            )?;
            Ok(wrapped.model)
        }
    }
}

pub(crate) fn verify_browser_signed_genesis_tensor_digest(
    model_config: &DragonConfig,
    descriptor: &ArtifactDescriptor,
    bytes: &[u8],
    training_contract_id: &ContentId,
    contract: &TrainingContractManifest,
    materialization: &GenesisMaterialization,
    expected: &ContentId,
) -> Result<()> {
    if let GenesisMaterialization::DeterministicReconstruction { .. } = materialization {
        type DigestBackend = NdArray<f32>;

        let device = burn::tensor::Device::<DigestBackend>::default();
        let wrapped = BrowserNativeTrainModelArtifact {
            model: DragonModel::<DigestBackend>::new(model_config.clone(), &device),
        };
        let scaffold = dragon_random_scaffold_p2p_contract_for_module(
            &wrapped,
            &wrapped.model,
            contract.model_schema_hash.clone(),
        )?
        .ok_or_else(|| anyhow!("signed deterministic genesis requires random-scaffold mode"))?;
        let wrapped = decode_random_scaffold_genesis_bytes::<DigestBackend, _>(
            wrapped,
            descriptor,
            bytes,
            training_contract_id,
            contract,
            materialization,
            &scaffold,
        )?;
        let actual = burn_p2p::burn_module::module_tensor_digest::<DigestBackend, _>(
            &wrapped,
            descriptor.model_schema_hash.clone(),
        )?;
        if &actual != expected {
            bail!(
                "browser decoded genesis tensor digest {} does not match signed digest {}",
                actual.as_str(),
                expected.as_str()
            );
        }
        return Ok(());
    }

    verify_browser_genesis_tensor_digest(model_config, descriptor, bytes, expected)
}

pub(crate) fn verify_browser_genesis_tensor_digest(
    model_config: &DragonConfig,
    descriptor: &ArtifactDescriptor,
    bytes: &[u8],
    expected: &ContentId,
) -> Result<()> {
    let format = browser_record_bytes_format(&descriptor.record_format)?;
    let precision = browser_record_precision(&descriptor.precision)?;
    match (format, precision) {
        (BrowserBurnRecordBytesFormat::Bin, BrowserBurnRecordPrecision::Full) => {
            verify_browser_record_tensor_digest::<BinBytesRecorder<FullPrecisionSettings>>(
                model_config,
                descriptor,
                bytes,
                expected,
            )
        }
        (BrowserBurnRecordBytesFormat::Bin, BrowserBurnRecordPrecision::Half) => {
            verify_browser_record_tensor_digest::<BinBytesRecorder<HalfPrecisionSettings>>(
                model_config,
                descriptor,
                bytes,
                expected,
            )
        }
        (BrowserBurnRecordBytesFormat::NamedMpk, BrowserBurnRecordPrecision::Full) => {
            verify_browser_record_tensor_digest::<NamedMpkBytesRecorder<FullPrecisionSettings>>(
                model_config,
                descriptor,
                bytes,
                expected,
            )
        }
        (BrowserBurnRecordBytesFormat::NamedMpk, BrowserBurnRecordPrecision::Half) => {
            verify_browser_record_tensor_digest::<NamedMpkBytesRecorder<HalfPrecisionSettings>>(
                model_config,
                descriptor,
                bytes,
                expected,
            )
        }
    }
}

fn verify_browser_record_tensor_digest<R>(
    model_config: &DragonConfig,
    descriptor: &ArtifactDescriptor,
    bytes: &[u8],
    expected: &ContentId,
) -> Result<()>
where
    R: Recorder<NdArray<f32>, RecordArgs = (), RecordOutput = Vec<u8>, LoadArgs = Vec<u8>>,
{
    type DigestBackend = NdArray<f32>;

    let device = burn::tensor::Device::<DigestBackend>::default();
    let wrapped = BrowserNativeTrainModelArtifact {
        model: DragonModel::<DigestBackend>::new(model_config.clone(), &device),
    };
    let wrapped_result = R::default()
        .load(bytes.to_vec(), &device)
        .map(|record| wrapped.load_record(record))
        .map_err(|error| error.to_string())
        .and_then(|model| {
            burn_p2p::tensor_identity::module_tensor_digest::<DigestBackend, _>(
                &model,
                descriptor.model_schema_hash.clone(),
            )
            .map_err(|error| error.to_string())
        });
    if wrapped_result.as_ref() == Ok(expected) {
        return Ok(());
    }

    let direct = DragonModel::<DigestBackend>::new(model_config.clone(), &device);
    let direct_result = R::default()
        .load(bytes.to_vec(), &device)
        .map(|record| direct.load_record(record))
        .map_err(|error| error.to_string())
        .and_then(|model| {
            burn_p2p::tensor_identity::module_tensor_digest::<DigestBackend, _>(
                &model,
                descriptor.model_schema_hash.clone(),
            )
            .map_err(|error| error.to_string())
        });
    if direct_result.as_ref() == Ok(expected) {
        return Ok(());
    }

    bail!(
        "decoded browser genesis tensors do not match authority-signed digest {}: native-wrapper={}; direct={}",
        expected.as_str(),
        digest_result_label(&wrapped_result),
        digest_result_label(&direct_result),
    )
}

fn digest_result_label(result: &std::result::Result<ContentId, String>) -> String {
    match result {
        Ok(digest) => digest.as_str().to_owned(),
        Err(error) => format!("decode-error:{error}"),
    }
}

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "native",
    feature = "wasm-peer"
))]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use burn::backend::NdArray;
    use burn_autodiff::Autodiff;
    use burn_dragon_core::{DragonConfig, LanguageHeadConfig};
    use burn_dragon_language::train::steps::LanguageTrainModel;
    use burn_p2p::{
        ArtifactKind, ChunkingScheme, CompactScalarEncoding, ContentId, DatasetViewId, HeadId,
        LocalOptimizerStatePolicy, RecurrentStatePolicy, SchedulerStatePolicy,
        TRAINING_CONTRACT_VERSION, TrainingContractManifest, UpdateCodec, WorkloadId,
    };
    use burn_p2p_checkpoint::{ArtifactBuildSpec, build_artifact_descriptor_from_bytes};
    use tempfile::tempdir;

    use crate::random_scaffold::{
        dragon_random_scaffold_p2p_contract, materialize_random_scaffold_genesis,
        materialize_random_scaffold_head, random_scaffold_genesis_materialization,
    };

    type TestBackend = Autodiff<NdArray<f32>>;

    #[test]
    fn browser_active_head_loader_accepts_native_training_wrapper_record() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let model_config = tiny_factorized_nca_model_config();
        let source = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            model_config.clone(),
            &device,
        ));
        let format = BrowserBurnRecordBytesFormat::NamedMpk;
        let precision = BrowserBurnRecordPrecision::Full;
        let bytes = encode_browser_record_bytes::<TestBackend, _>(source, format, precision)
            .expect("native training wrapper record should encode");
        let descriptor = descriptor_for_bytes(&bytes, format, precision);
        let target = DragonModel::<TestBackend>::new(model_config, &device);

        load_browser_active_head_model(target, &descriptor, bytes, &device)
            .expect("browser should load native learner-wrapper head artifacts");
    }

    #[test]
    fn browser_genesis_verifier_matches_native_training_wrapper_tensors() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let model_config = tiny_factorized_nca_model_config();
        let source = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            model_config.clone(),
            &device,
        ));
        let semantic_schema = ContentId::new("semantic-schema");
        let expected = burn_p2p::burn_module::module_tensor_digest::<TestBackend, _>(
            &source,
            semantic_schema.clone(),
        )
        .expect("native tensor digest");
        let format = BrowserBurnRecordBytesFormat::NamedMpk;
        let precision = BrowserBurnRecordPrecision::Full;
        let bytes = encode_browser_record_bytes::<TestBackend, _>(source, format, precision)
            .expect("native training wrapper record should encode");
        let mut descriptor = descriptor_for_bytes(&bytes, format, precision);
        descriptor.model_schema_hash = semantic_schema;

        verify_browser_genesis_tensor_digest(&model_config, &descriptor, &bytes, &expected)
            .expect("browser verifier should reproduce native digest");
        let error = verify_browser_genesis_tensor_digest(
            &model_config,
            &descriptor,
            &bytes,
            &ContentId::new("wrong"),
        )
        .expect_err("wrong signed digest must fail");
        assert!(error.to_string().contains("do not match authority-signed"));
    }

    #[test]
    fn browser_active_head_loader_accepts_browser_dragon_record() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let model_config = tiny_factorized_nca_model_config();
        let source = DragonModel::<TestBackend>::new(model_config.clone(), &device);
        let format = BrowserBurnRecordBytesFormat::NamedMpk;
        let precision = BrowserBurnRecordPrecision::Half;
        let bytes = encode_browser_record_bytes::<TestBackend, _>(source, format, precision)
            .expect("browser dragon record should encode");
        let descriptor = descriptor_for_bytes(&bytes, format, precision);
        let target = DragonModel::<TestBackend>::new(model_config, &device);

        load_browser_active_head_model(target, &descriptor, bytes, &device)
            .expect("browser should keep loading browser-published head artifacts");
    }

    #[test]
    fn browser_async_mutable_readback_matches_native_catalog_order_and_digest() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let model_config = tiny_random_scaffold_model_config();
        let source = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            model_config.clone(),
            &device,
        ));
        let schema = ContentId::new("random-scaffold-browser-schema");
        let native = dragon_random_scaffold_p2p_contract(&source, schema.clone())
            .expect("native scaffold contract")
            .expect("enabled scaffold");
        let browser = browser_random_scaffold_contract(&source.model, schema.clone())
            .expect("browser scaffold contract")
            .expect("enabled scaffold");
        assert_eq!(
            native.catalog.catalog_id().expect("native catalog id"),
            browser.catalog.catalog_id().expect("browser catalog id")
        );
        assert_eq!(
            native
                .immutable_catalog
                .catalog_id()
                .expect("native immutable id"),
            browser
                .immutable_catalog
                .catalog_id()
                .expect("browser immutable id")
        );

        let expected =
            burn_p2p::burn_module::flatten_module_float_parameter_subset::<TestBackend, _>(
                &source,
                &native.catalog,
            )
            .expect("native mutable values");
        let actual = futures::executor::block_on(flatten_browser_random_scaffold_mutable(
            &source.model,
            &browser,
        ))
        .expect("browser async mutable values");
        assert_eq!(expected, actual);

        let expected_digest =
            burn_p2p::burn_module::module_tensor_digest::<TestBackend, _>(&source, schema.clone())
                .expect("source digest");
        let browser_digest =
            browser_random_scaffold_tensor_digest_from_mutable(&model_config, schema, &actual)
                .expect("browser reconstructed digest");
        assert_eq!(expected_digest, browser_digest);
    }

    #[test]
    fn browser_reconstructs_compact_signed_scaffold_genesis_exactly() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let model_config = tiny_random_scaffold_model_config();
        let source = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            model_config.clone(),
            &device,
        ));
        let schema = ContentId::new("random-scaffold-browser-schema");
        let scaffold = dragon_random_scaffold_p2p_contract(&source, schema.clone())
            .expect("scaffold contract")
            .expect("enabled scaffold");
        let contract = scaffold_training_contract(&scaffold.catalog);
        let contract_id = contract.contract_id().expect("training contract id");
        let materialization =
            random_scaffold_genesis_materialization(&scaffold).expect("materialization");
        let root = tempdir().expect("artifact root");
        let store = burn_p2p::FsArtifactStore::new(root.path());
        let descriptor = materialize_random_scaffold_genesis::<TestBackend>(
            burn_p2p::GenesisArtifactMaterializationContext {
                model: &source,
                head_id: HeadId::new("genesis"),
                training_contract_id: &contract_id,
                contract: &contract,
                materialization: &materialization,
                store: &store,
            },
            &scaffold,
        )
        .expect("materialize compact genesis")
        .expect("deterministic scaffold genesis");
        let bytes = store
            .materialize_artifact_bytes(&descriptor)
            .expect("genesis bytes");
        let expected =
            burn_p2p::burn_module::module_tensor_digest::<TestBackend, _>(&source, schema.clone())
                .expect("source digest");

        verify_browser_signed_genesis_tensor_digest(
            &model_config,
            &descriptor,
            &bytes,
            &contract_id,
            &contract,
            &materialization,
            &expected,
        )
        .expect("browser signed-genesis verification");
        let target = DragonModel::<TestBackend>::new(model_config, &device);
        let loaded = load_browser_genesis_model(
            target,
            &descriptor,
            bytes,
            &contract_id,
            &contract,
            &materialization,
            &device,
        )
        .expect("browser compact-genesis load");
        let loaded = BrowserNativeTrainModelArtifact { model: loaded };
        assert_eq!(
            expected,
            burn_p2p::burn_module::module_tensor_digest::<TestBackend, _>(&loaded, schema)
                .expect("loaded digest")
        );
    }

    #[test]
    fn browser_loads_compact_promoted_scaffold_head_exactly() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let model_config = tiny_random_scaffold_model_config();
        let source = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            model_config.clone(),
            &device,
        ));
        let schema = ContentId::new("random-scaffold-browser-schema");
        let scaffold = dragon_random_scaffold_p2p_contract(&source, schema.clone())
            .expect("scaffold contract")
            .expect("enabled scaffold");
        let root = tempdir().expect("artifact root");
        let store = burn_p2p::FsArtifactStore::new(root.path());
        let descriptor = materialize_random_scaffold_head::<TestBackend>(
            &source,
            ArtifactKind::FullHead,
            &HeadId::new("promoted-head"),
            Some(&HeadId::new("genesis")),
            &store,
            &schema,
            &scaffold,
        )
        .expect("materialize compact head")
        .expect("handled scaffold head");
        let bytes = store
            .materialize_artifact_bytes(&descriptor)
            .expect("head bytes");
        let expected =
            burn_p2p::burn_module::module_tensor_digest::<TestBackend, _>(&source, schema.clone())
                .expect("source digest");
        let target = DragonModel::<TestBackend>::new(model_config, &device);
        let loaded = load_browser_active_head_model(target, &descriptor, bytes, &device)
            .expect("browser compact-head load");
        let wrapped = BrowserNativeTrainModelArtifact { model: loaded };
        assert_eq!(
            expected,
            burn_p2p::burn_module::module_tensor_digest::<TestBackend, _>(&wrapped, schema)
                .expect("loaded digest")
        );
    }

    fn tiny_factorized_nca_model_config() -> DragonConfig {
        DragonConfig {
            n_layer: 1,
            n_embd: 16,
            dropout: 0.0,
            n_head: 1,
            mlp_internal_dim_multiplier: 2,
            n_expert: 1,
            vocab_size: 256,
            language_head: LanguageHeadConfig::NcaFactorizedPatch {
                state_count: 2,
                patch_size: 2,
                frame_special_tokens: true,
                eos_id: Some(255),
            },
            ..DragonConfig::default()
        }
    }

    fn tiny_random_scaffold_model_config() -> DragonConfig {
        let mut config = DragonConfig {
            n_layer: 1,
            n_embd: 16,
            dropout: 0.0,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            n_expert: 1,
            vocab_size: 32,
            ..DragonConfig::default()
        };
        config.random_scaffold.enabled = true;
        config.random_scaffold.seed = 17;
        config.random_scaffold.rank = 4;
        config
    }

    fn scaffold_training_contract(
        catalog: &burn_p2p::ParameterSubsetCatalog,
    ) -> TrainingContractManifest {
        TrainingContractManifest {
            version: TRAINING_CONTRACT_VERSION,
            workload_id: WorkloadId::new("dragon-browser-scaffold-test"),
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
                encoding: CompactScalarEncoding::Fp32,
            },
            aggregation_hash: ContentId::new("aggregation"),
            validation_hash: ContentId::new("validation"),
            initialization_hash: ContentId::new("initialization"),
            extensions: BTreeMap::new(),
        }
    }

    fn descriptor_for_bytes(
        bytes: &[u8],
        format: BrowserBurnRecordBytesFormat,
        precision: BrowserBurnRecordPrecision,
    ) -> ArtifactDescriptor {
        build_artifact_descriptor_from_bytes(
            &ArtifactBuildSpec::new(
                ArtifactKind::FullHead,
                browser_record_precision_descriptor(precision),
                ContentId::new("test-dragon-browser-model-schema"),
                browser_record_format_name(format),
            )
            .with_head(HeadId::new("test-head")),
            bytes,
            ChunkingScheme::new(1024 * 1024).expect("chunk size"),
        )
        .expect("descriptor")
    }
}
