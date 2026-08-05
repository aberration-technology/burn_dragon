use burn::module::{
    AutodiffModule, Content, Devices, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor,
};
use burn::tensor::backend::{AutodiffBackend, Backend};
use serde::de::Deserializer;
use serde::ser::{SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};

pub const DEFAULT_DENSE_SCORE_ROW_CHUNK: usize = 512;

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SequenceMemorySystem {
    #[default]
    LinearAttention,
    Mamba3StateSpaceDuality,
    #[serde(alias = "gated_deltanet2", alias = "gdn2")]
    GatedDeltaNet2,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SequenceTrainingExecutor {
    #[default]
    Reference,
    DenseScoreShortContext,
    #[serde(alias = "gdn2_chunk_wy")]
    GatedDeltaChunkWy,
}

impl SequenceMemorySystem {
    pub const fn default_executor(self) -> SequenceTrainingExecutor {
        match self {
            Self::LinearAttention | Self::Mamba3StateSpaceDuality | Self::GatedDeltaNet2 => {
                SequenceTrainingExecutor::Reference
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequenceKernelConfig {
    pub memory_system: SequenceMemorySystem,
    pub executor: SequenceTrainingExecutor,
    pub dense_score_row_chunk: usize,
}

impl Default for SequenceKernelConfig {
    fn default() -> Self {
        Self::reference(SequenceMemorySystem::LinearAttention)
    }
}

impl SequenceKernelConfig {
    pub const fn new(
        memory_system: SequenceMemorySystem,
        executor: SequenceTrainingExecutor,
    ) -> Self {
        Self {
            memory_system,
            executor,
            dense_score_row_chunk: DEFAULT_DENSE_SCORE_ROW_CHUNK,
        }
    }

    pub const fn with_dense_score_row_chunk(mut self, row_chunk: usize) -> Self {
        self.dense_score_row_chunk = row_chunk;
        self
    }

    pub const fn reference(memory_system: SequenceMemorySystem) -> Self {
        Self::new(memory_system, memory_system.default_executor())
    }

    pub const fn dense_score_short_context() -> Self {
        Self::new(
            SequenceMemorySystem::LinearAttention,
            SequenceTrainingExecutor::DenseScoreShortContext,
        )
    }

    pub const fn gated_delta_chunk_wy() -> Self {
        Self::new(
            SequenceMemorySystem::GatedDeltaNet2,
            SequenceTrainingExecutor::GatedDeltaChunkWy,
        )
    }

    pub const fn is_supported(self) -> bool {
        matches!(
            (self.memory_system, self.executor),
            (
                SequenceMemorySystem::LinearAttention,
                SequenceTrainingExecutor::Reference
                    | SequenceTrainingExecutor::DenseScoreShortContext
            ) | (
                SequenceMemorySystem::Mamba3StateSpaceDuality,
                SequenceTrainingExecutor::Reference
            ) | (
                SequenceMemorySystem::GatedDeltaNet2,
                SequenceTrainingExecutor::Reference | SequenceTrainingExecutor::GatedDeltaChunkWy
            )
        )
    }

    pub fn validate(self) -> Result<(), String> {
        if self.dense_score_row_chunk == 0 {
            return Err("dense_score_row_chunk must be > 0".to_string());
        }
        if self.is_supported() {
            Ok(())
        } else {
            Err(format!(
                "sequence kernel family {:?} with executor {:?} is not supported",
                self.memory_system, self.executor
            ))
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum SequenceKernelConfigSerde {
    MemorySystem(SequenceMemorySystem),
    Config {
        #[serde(alias = "family")]
        memory_system: SequenceMemorySystem,
        #[serde(default)]
        executor: Option<SequenceTrainingExecutor>,
        #[serde(default = "default_dense_score_row_chunk")]
        dense_score_row_chunk: usize,
    },
}

const fn default_dense_score_row_chunk() -> usize {
    DEFAULT_DENSE_SCORE_ROW_CHUNK
}

impl From<SequenceKernelConfigSerde> for SequenceKernelConfig {
    fn from(value: SequenceKernelConfigSerde) -> Self {
        match value {
            SequenceKernelConfigSerde::MemorySystem(memory_system) => {
                Self::reference(memory_system)
            }
            SequenceKernelConfigSerde::Config {
                memory_system,
                executor,
                dense_score_row_chunk,
            } => Self::new(
                memory_system,
                executor.unwrap_or_else(|| memory_system.default_executor()),
            )
            .with_dense_score_row_chunk(dense_score_row_chunk),
        }
    }
}

impl<'de> Deserialize<'de> for SequenceKernelConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        SequenceKernelConfigSerde::deserialize(deserializer).map(Into::into)
    }
}

impl Serialize for SequenceKernelConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if self.executor == self.memory_system.default_executor()
            && self.dense_score_row_chunk == DEFAULT_DENSE_SCORE_ROW_CHUNK
        {
            return self.memory_system.serialize(serializer);
        }

        let mut state = serializer.serialize_struct("SequenceKernelConfig", 3)?;
        state.serialize_field("memory_system", &self.memory_system)?;
        state.serialize_field("executor", &self.executor)?;
        state.serialize_field("dense_score_row_chunk", &self.dense_score_row_chunk)?;
        state.end()
    }
}

impl<B: Backend> Module<B> for SequenceMemorySystem {
    type Record = ();

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for SequenceMemorySystem {
    type InnerModule = SequenceMemorySystem;

    fn valid(&self) -> Self::InnerModule {
        *self
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for SequenceMemorySystem {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .set_top_level_type("SequenceMemorySystem")
            .add_formatted(&format!("{self:?}"))
            .optional()
    }
}

impl ModuleDisplay for SequenceMemorySystem {}

impl<B: Backend> Module<B> for SequenceTrainingExecutor {
    type Record = ();

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for SequenceTrainingExecutor {
    type InnerModule = SequenceTrainingExecutor;

    fn valid(&self) -> Self::InnerModule {
        *self
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for SequenceTrainingExecutor {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .set_top_level_type("SequenceTrainingExecutor")
            .add_formatted(&format!("{self:?}"))
            .optional()
    }
}

impl ModuleDisplay for SequenceTrainingExecutor {}

impl<B: Backend> Module<B> for SequenceKernelConfig {
    type Record = ();

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for SequenceKernelConfig {
    type InnerModule = SequenceKernelConfig;

    fn valid(&self) -> Self::InnerModule {
        *self
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for SequenceKernelConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .set_top_level_type("SequenceKernelConfig")
            .add_formatted(&format!(
                "memory_system={:?}, executor={:?}, dense_score_row_chunk={}",
                self.memory_system, self.executor, self.dense_score_row_chunk
            ))
            .optional()
    }
}

impl ModuleDisplay for SequenceKernelConfig {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_executor_is_reference_for_memory_systems() {
        assert_eq!(
            SequenceKernelConfig::reference(SequenceMemorySystem::LinearAttention),
            SequenceKernelConfig {
                memory_system: SequenceMemorySystem::LinearAttention,
                executor: SequenceTrainingExecutor::Reference,
                dense_score_row_chunk: DEFAULT_DENSE_SCORE_ROW_CHUNK,
            }
        );
    }

    #[test]
    fn dense_score_short_context_is_explicit() {
        assert_eq!(
            SequenceKernelConfig::dense_score_short_context(),
            SequenceKernelConfig {
                memory_system: SequenceMemorySystem::LinearAttention,
                executor: SequenceTrainingExecutor::DenseScoreShortContext,
                dense_score_row_chunk: DEFAULT_DENSE_SCORE_ROW_CHUNK,
            },
        );
    }

    #[test]
    fn gated_delta_chunk_wy_is_explicit() {
        assert_eq!(
            SequenceKernelConfig::gated_delta_chunk_wy(),
            SequenceKernelConfig {
                memory_system: SequenceMemorySystem::GatedDeltaNet2,
                executor: SequenceTrainingExecutor::GatedDeltaChunkWy,
                dense_score_row_chunk: DEFAULT_DENSE_SCORE_ROW_CHUNK,
            },
        );
    }

    #[test]
    fn dense_score_row_chunk_round_trips_when_overridden() {
        let json = r#"{"memory_system":"linear_attention","executor":"dense_score_short_context","dense_score_row_chunk":256}"#;
        let parsed: SequenceKernelConfig = serde_json::from_str(json).expect("sequence config");
        assert_eq!(parsed.dense_score_row_chunk, 256);
        assert_eq!(
            serde_json::from_str::<SequenceKernelConfig>(
                &serde_json::to_string(&parsed).expect("serialize sequence config")
            )
            .expect("round-trip sequence config"),
            parsed
        );
    }

    #[test]
    fn dense_score_row_chunk_rejects_zero() {
        let config =
            SequenceKernelConfig::dense_score_short_context().with_dense_score_row_chunk(0);
        assert_eq!(
            config.validate().expect_err("zero row chunk must fail"),
            "dense_score_row_chunk must be > 0"
        );
    }

    #[test]
    fn gated_delta_net2_aliases_parse() {
        let bare: SequenceKernelConfig =
            serde_json::from_str("\"gated_deltanet2\"").expect("gdn2 bare alias");
        assert_eq!(
            bare,
            SequenceKernelConfig::reference(SequenceMemorySystem::GatedDeltaNet2)
        );
        let chunk: SequenceKernelConfig =
            serde_json::from_str(r#"{"memory_system":"gdn2","executor":"gdn2_chunk_wy"}"#)
                .expect("gdn2 chunk alias");
        assert_eq!(chunk, SequenceKernelConfig::gated_delta_chunk_wy());
    }
}
