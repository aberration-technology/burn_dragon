use anyhow::{Result, anyhow, bail};
use burn::module::Module;
use burn::record::{
    BinBytesRecorder, FullPrecisionSettings, HalfPrecisionSettings, NamedMpkBytesRecorder, Recorder,
};
use burn::tensor::backend::Backend;
use burn_dragon_core::DragonModel;
use burn_p2p::{ArtifactDescriptor, Precision};
use log::info;

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
struct BrowserNativeTrainModelArtifact<B: Backend> {
    model: DragonModel<B>,
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

#[cfg(all(
    test,
    not(target_arch = "wasm32"),
    feature = "native",
    feature = "wasm-peer"
))]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn_autodiff::Autodiff;
    use burn_dragon_core::{DragonConfig, LanguageHeadConfig};
    use burn_dragon_language::train::steps::LanguageTrainModel;
    use burn_p2p::{ArtifactKind, ChunkingScheme, ContentId, HeadId};
    use burn_p2p_checkpoint::{ArtifactBuildSpec, build_artifact_descriptor_from_bytes};

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
