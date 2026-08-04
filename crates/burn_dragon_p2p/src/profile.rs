use std::collections::BTreeMap;

#[cfg(feature = "native")]
use anyhow::bail;
use anyhow::{Result, anyhow};
use burn_p2p::{
    BrowserRole, ExperimentDirectoryEntry, ExperimentDirectoryPolicyExt, TrainingProtocol,
};
use burn_p2p_workload::{
    DirectoryMetadataAttachment, find_matching_directory_entry_with_predicate,
};
use serde::{Deserialize, Serialize};

#[cfg(feature = "native")]
use std::fs;
#[cfg(feature = "native")]
use std::path::{Path, PathBuf};

#[cfg(all(not(feature = "native"), feature = "wasm-peer"))]
use burn_dragon_core::DragonConfig;
#[cfg(feature = "native")]
use burn_dragon_language::api::inference::build_model_config_with_tokenizer;
#[cfg(feature = "native")]
use burn_dragon_language::config::ValidationDatasetConfig;
#[cfg(feature = "native")]
use burn_dragon_language::{
    DatasetSourceConfig, DragonConfig, TrainingConfig, load_training_config,
};
#[cfg(any(feature = "wasm-peer", feature = "native"))]
use burn_dragon_universality::{NcaCorpusConfig, RuliadCorpusConfig};
#[cfg(feature = "native")]
use burn_p2p::BrowserEdgeSnapshot;

#[cfg(feature = "native")]
use crate::auth::fetch_edge_snapshot;
use crate::config::{
    DragonBrowserDatasetSplit, DragonBrowserExecutionBackend, DragonBrowserShardSelectionPolicy,
    DragonCapabilityPolicy, DragonExperimentKind, TokenWindowRecord,
};
#[cfg(any(feature = "wasm-peer", feature = "native"))]
use crate::config::{
    DragonBrowserLiveParticipantConfig, DragonBrowserOptimizerConfig, DragonBrowserTokenSource,
    DragonBrowserTrainingConfig, DragonBrowserTrainingObjectiveConfig,
};
#[cfg(feature = "native")]
use crate::config::{
    DragonManifestSeed, DragonNativePeerConfig, DragonNativeTrainingOverrides,
    DragonPromotionConfig, DragonPromotionMode,
};

pub const DRAGON_PROFILE_VERSION_METADATA_KEY: &str = "dragon_profile_version";
pub const DRAGON_PROFILE_JSON_METADATA_KEY: &str = "dragon_profile_json";
pub const DRAGON_BROWSER_EXECUTION_CONTRACT_EXTENSION: &str = "dragon.browser_execution.v1";
const DRAGON_PROFILE_VERSION: u32 = 1;
#[cfg(feature = "native")]
const DEFAULT_BROWSER_CLIMBMIX_MAX_SHARDS_PER_WINDOW: usize = 4;
#[cfg(feature = "native")]
const NCA_BROWSER_WGPU_BATCH_SIZE_CAP: usize = 1;
#[cfg(feature = "native")]
const NCA_BROWSER_WGPU_MAX_TRAIN_BATCHES_CAP: usize = 8;
#[cfg(feature = "native")]
const DEFAULT_NCA_BROWSER_WGPU_MEMORY_BUDGET_BYTES: u64 = 6 * 1024 * 1024 * 1024;
#[cfg(feature = "native")]
const NCA_BROWSER_MIN_TRAIN_DOCUMENT_POOL: usize = 64;
#[cfg(feature = "native")]
const NCA_BROWSER_MIN_EVAL_DOCUMENT_POOL: usize = 8;
#[cfg(feature = "native")]
const PORTABLE_NCA_CORPUS_FILE_NAME: &str = "nca-corpus.toml";
#[cfg(feature = "native")]
const PORTABLE_RULIAD_CORPUS_FILE_NAME: &str = "ruliad-corpus.toml";
#[cfg(feature = "native")]
const PORTABLE_CACHE_DIR_NAME: &str = "__dragon_network_profile_cache__";
#[cfg(feature = "native")]
const BUILTIN_NCA_R1_PROFILE_JSON: &str = include_str!("../deploy/profiles/nca-r1.profile.json");

#[cfg(feature = "native")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PortableUniversalityCorpusKind {
    Nca,
    Ruliad,
}

#[cfg(feature = "native")]
impl PortableUniversalityCorpusKind {
    fn from_source(source: &DatasetSourceConfig) -> Option<Self> {
        match source {
            DatasetSourceConfig::UniversalityNca { .. } => Some(Self::Nca),
            DatasetSourceConfig::UniversalityRuliad { .. } => Some(Self::Ruliad),
            _ => None,
        }
    }

    fn config_path(self, source: &DatasetSourceConfig) -> Option<&Path> {
        match (self, source) {
            (Self::Nca, DatasetSourceConfig::UniversalityNca { config })
            | (Self::Ruliad, DatasetSourceConfig::UniversalityRuliad { config }) => {
                Some(config.as_path())
            }
            _ => None,
        }
    }

    fn file_name(self) -> &'static str {
        match self {
            Self::Nca => PORTABLE_NCA_CORPUS_FILE_NAME,
            Self::Ruliad => PORTABLE_RULIAD_CORPUS_FILE_NAME,
        }
    }

    fn generated_dir_name(self) -> &'static str {
        match self {
            Self::Nca => "nca-generated",
            Self::Ruliad => "ruliad-generated",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Nca => "NCA",
            Self::Ruliad => "ruliad",
        }
    }

    fn source_config(self, path: PathBuf) -> DatasetSourceConfig {
        match self {
            Self::Nca => DatasetSourceConfig::UniversalityNca { config: path },
            Self::Ruliad => DatasetSourceConfig::UniversalityRuliad { config: path },
        }
    }

    fn matches_source(self, source: &DatasetSourceConfig) -> bool {
        matches!(
            (self, source),
            (Self::Nca, DatasetSourceConfig::UniversalityNca { .. })
                | (Self::Ruliad, DatasetSourceConfig::UniversalityRuliad { .. })
        )
    }

    fn apply_source_path(self, config: &mut TrainingConfig, path: PathBuf) {
        config.dataset.source = self.source_config(path.clone());
        if let Some(validation) = config.dataset.validation.as_mut()
            && self.matches_source(&validation.source)
        {
            validation.source = self.source_config(path);
        }
    }
}

#[cfg(feature = "native")]
struct PortableUniversalityCorpus {
    kind: PortableUniversalityCorpusKind,
    toml: String,
}

#[cfg(feature = "native")]
fn resolve_local_profile_path(path: &Path) -> PathBuf {
    if path.is_absolute() || path.is_file() {
        return path.to_path_buf();
    }

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let package_relative = manifest_dir.join(path);
    if package_relative.is_file() {
        return package_relative;
    }

    // Repository profiles historically use workspace-relative paths. Cargo test
    // executes from the package directory, while operators commonly launch from
    // the workspace root, so accept both without making the published profile
    // depend on process cwd.
    let workspace_relative = manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(|workspace| workspace.join(path));
    if let Some(workspace_relative) = workspace_relative
        && workspace_relative.is_file()
    {
        return workspace_relative;
    }

    path.to_path_buf()
}

#[cfg(feature = "native")]
fn portable_universality_corpus(
    config: &TrainingConfig,
) -> Result<Option<PortableUniversalityCorpus>> {
    let Some(kind) = PortableUniversalityCorpusKind::from_source(&config.dataset.source) else {
        return Ok(None);
    };
    let config_path = kind
        .config_path(&config.dataset.source)
        .expect("portable corpus source path");
    let resolved_path = resolve_local_profile_path(config_path);
    let toml = fs::read_to_string(&resolved_path).map_err(|error| {
        anyhow!(
            "failed to read portable {} corpus config {} (declared as {}): {error}",
            kind.label(),
            resolved_path.display(),
            config_path.display(),
        )
    })?;
    Ok(Some(PortableUniversalityCorpus { kind, toml }))
}

#[cfg(feature = "native")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DragonBrowserWindowTuning {
    batch_size: usize,
    max_train_batches: usize,
    max_eval_batches: usize,
    train_document_pool: usize,
    eval_document_pool: usize,
}

#[cfg(feature = "native")]
impl DragonBrowserWindowTuning {
    fn nca_wgpu_from_native(config: &TrainingConfig) -> Self {
        let batch_size = config
            .training
            .batch_size
            .clamp(1, NCA_BROWSER_WGPU_BATCH_SIZE_CAP);
        let max_train_batches = config
            .training
            .max_iters
            .clamp(1, NCA_BROWSER_WGPU_MAX_TRAIN_BATCHES_CAP);
        let native_window_examples = config
            .training
            .batch_size
            .saturating_mul(config.training.max_iters)
            .max(1);
        let train_document_pool = native_window_examples.max(NCA_BROWSER_MIN_TRAIN_DOCUMENT_POOL);
        let eval_document_pool = config
            .training
            .batch_size
            .max(NCA_BROWSER_MIN_EVAL_DOCUMENT_POOL);

        Self {
            batch_size,
            max_train_batches,
            max_eval_batches: 1,
            train_document_pool,
            eval_document_pool,
        }
    }
}

fn dragon_profile_attachment() -> DirectoryMetadataAttachment {
    DirectoryMetadataAttachment::new(
        DRAGON_PROFILE_VERSION_METADATA_KEY,
        DRAGON_PROFILE_JSON_METADATA_KEY,
        DRAGON_PROFILE_VERSION.to_string(),
    )
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DragonExperimentProfile {
    pub version: u32,
    pub experiment_kind: DragonExperimentKind,
    pub native: DragonNativeExperimentProfile,
    #[serde(default)]
    pub browser: Option<DragonBrowserExperimentProfile>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DragonNativeExperimentProfile {
    pub training_toml: String,
    #[serde(default)]
    pub nca_corpus_toml: Option<String>,
    #[serde(default)]
    pub ruliad_corpus_toml: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DragonBrowserExperimentProfile {
    pub model_config: DragonConfig,
    #[serde(default)]
    pub training_objective: DragonBrowserTrainingObjectiveConfig,
    #[serde(default)]
    pub optimizer: DragonBrowserOptimizerConfig,
    #[serde(default)]
    pub execution_backend: DragonBrowserExecutionBackend,
    pub block_size: usize,
    #[serde(default)]
    pub tbptt_chunk_size: Option<usize>,
    #[serde(default)]
    pub tbptt_persist_across_steps: bool,
    pub learning_rate: f64,
    #[serde(default)]
    pub weight_decay: f32,
    pub batch_size: usize,
    #[serde(default)]
    pub max_train_batches: Option<usize>,
    #[serde(default)]
    pub max_eval_batches: Option<usize>,
    #[serde(default)]
    pub capability_policy: DragonCapabilityPolicy,
    #[serde(default)]
    pub trainer_support: DragonBrowserTrainerSupport,
    pub train_source: DragonBrowserProfileTokenSource,
    #[serde(default)]
    pub eval_source: Option<DragonBrowserProfileTokenSource>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum DragonBrowserTrainerSupport {
    #[default]
    Supported,
    ObserverOnly {
        reason: String,
    },
}

impl DragonBrowserTrainerSupport {
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Supported)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DragonBrowserProfileTokenSource {
    Inline {
        records: Vec<TokenWindowRecord>,
    },
    HttpJson {
        url: String,
    },
    ShardManifestHttp {
        manifest_url: String,
        #[serde(default)]
        selection: DragonBrowserShardSelectionPolicy,
        #[serde(default)]
        max_shards_per_window: Option<usize>,
    },
    GeneratedNca {
        corpus_toml: String,
        split: DragonBrowserDatasetSplit,
        #[serde(default)]
        max_documents: Option<usize>,
    },
    GeneratedRuliad {
        corpus_toml: String,
        split: DragonBrowserDatasetSplit,
        #[serde(default)]
        max_documents: Option<usize>,
        #[serde(default)]
        supervision: burn_dragon_universality::ruliad::RuliadTokenSupervisionConfig,
    },
}

fn normalized_browser_profile_source(
    source: &DragonBrowserProfileTokenSource,
) -> Result<serde_json::Value> {
    Ok(match source {
        DragonBrowserProfileTokenSource::Inline { records } => {
            serde_json::json!({"type": "inline", "records": records})
        }
        DragonBrowserProfileTokenSource::HttpJson { url } => {
            serde_json::json!({"type": "http_json", "url": url})
        }
        DragonBrowserProfileTokenSource::ShardManifestHttp {
            manifest_url,
            selection,
            max_shards_per_window,
        } => serde_json::json!({
            "type": "shard_manifest_http",
            "manifest_url": manifest_url,
            "selection": selection,
            "max_shards_per_window": max_shards_per_window,
        }),
        DragonBrowserProfileTokenSource::GeneratedNca {
            corpus_toml,
            split,
            max_documents,
        } => {
            let corpus: NcaCorpusConfig = toml::from_str(corpus_toml)
                .map_err(|error| anyhow!("invalid browser NCA corpus TOML: {error}"))?;
            serde_json::json!({
                "type": "generated_nca",
                "corpus": corpus,
                "split": split,
                "max_documents": max_documents,
            })
        }
        DragonBrowserProfileTokenSource::GeneratedRuliad {
            corpus_toml,
            split,
            max_documents,
            supervision,
        } => {
            let corpus: RuliadCorpusConfig = toml::from_str(corpus_toml)
                .map_err(|error| anyhow!("invalid browser Ruliad corpus TOML: {error}"))?;
            serde_json::json!({
                "type": "generated_ruliad",
                "corpus": corpus,
                "split": split,
                "max_documents": max_documents,
                "supervision": supervision,
            })
        }
    })
}

#[cfg(any(feature = "wasm-peer", feature = "native"))]
fn normalized_browser_runtime_source(source: &DragonBrowserTokenSource) -> serde_json::Value {
    match source {
        DragonBrowserTokenSource::Inline { records } => {
            serde_json::json!({"type": "inline", "records": records})
        }
        DragonBrowserTokenSource::HttpJson { url } => {
            serde_json::json!({"type": "http_json", "url": url})
        }
        DragonBrowserTokenSource::ShardManifestHttp {
            manifest_url,
            selection,
            max_shards_per_window,
        } => serde_json::json!({
            "type": "shard_manifest_http",
            "manifest_url": manifest_url,
            "selection": selection,
            "max_shards_per_window": max_shards_per_window,
        }),
        DragonBrowserTokenSource::GeneratedNca {
            corpus,
            split,
            max_documents,
        } => serde_json::json!({
            "type": "generated_nca",
            "corpus": corpus,
            "split": split,
            "max_documents": max_documents,
        }),
        DragonBrowserTokenSource::GeneratedRuliad {
            corpus,
            split,
            max_documents,
            supervision,
        } => serde_json::json!({
            "type": "generated_ruliad",
            "corpus": corpus,
            "split": split,
            "max_documents": max_documents,
            "supervision": supervision,
        }),
    }
}

fn browser_execution_contract_hash(execution: serde_json::Value) -> Result<burn_p2p::ContentId> {
    use sha2::{Digest, Sha256};

    let bytes = serde_json::to_vec(&execution)?;
    let mut hasher = Sha256::new();
    hasher.update(DRAGON_BROWSER_EXECUTION_CONTRACT_EXTENSION.as_bytes());
    hasher.update([0]);
    hasher.update(bytes);
    Ok(burn_p2p::ContentId::new(format!(
        "dragon-browser-execution-{:x}",
        hasher.finalize()
    )))
}

pub fn browser_profile_execution_contract_hash(
    experiment_kind: DragonExperimentKind,
    profile: &DragonBrowserExperimentProfile,
) -> Result<burn_p2p::ContentId> {
    browser_execution_contract_hash(serde_json::json!({
        "version": 1,
        "experiment_kind": experiment_kind,
        "model_config": profile.model_config,
        "training_objective": profile.training_objective,
        "optimizer": profile.optimizer,
        "block_size": profile.block_size,
        "tbptt_chunk_size": profile.tbptt_chunk_size,
        "tbptt_persist_across_steps": profile.tbptt_persist_across_steps,
        "learning_rate": profile.learning_rate,
        "weight_decay": profile.weight_decay,
        "batch_size": profile.batch_size,
        "max_train_batches": profile.max_train_batches,
        "max_eval_batches": profile.max_eval_batches,
        "train_source": normalized_browser_profile_source(&profile.train_source)?,
        "eval_source": profile
            .eval_source
            .as_ref()
            .map(normalized_browser_profile_source)
            .transpose()?,
    }))
}

#[cfg(any(feature = "wasm-peer", feature = "native"))]
pub fn browser_runtime_execution_contract_hash(
    config: &DragonBrowserTrainingConfig,
) -> Result<burn_p2p::ContentId> {
    browser_execution_contract_hash(serde_json::json!({
        "version": 1,
        "experiment_kind": config.experiment_kind,
        "model_config": config.model_config,
        "training_objective": config.training_objective,
        "optimizer": config.optimizer,
        "block_size": config.block_size,
        "tbptt_chunk_size": config.tbptt_chunk_size,
        "tbptt_persist_across_steps": config.tbptt_persist_across_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "max_train_batches": config.max_train_batches,
        "max_eval_batches": config.max_eval_batches,
        "train_source": normalized_browser_runtime_source(&config.train_source),
        "eval_source": config
            .eval_source
            .as_ref()
            .map(normalized_browser_runtime_source),
    }))
}

#[cfg(feature = "native")]
#[derive(Clone, Debug)]
pub struct ResolvedNativeTrainingProfile {
    pub config: TrainingConfig,
    pub manifest_seed: DragonManifestSeed,
    pub profile: DragonExperimentProfile,
    pub directory_entry: Option<ExperimentDirectoryEntry>,
    pub source: DragonResolvedProfileSource,
}

#[cfg(feature = "native")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DragonResolvedProfileSource {
    NetworkPublished,
    BuiltinFallback,
    LocalConfig,
}

impl DragonExperimentProfile {
    pub fn metadata_entries(&self) -> Result<BTreeMap<String, String>> {
        Ok(BTreeMap::from([
            (
                DRAGON_PROFILE_VERSION_METADATA_KEY.into(),
                self.version.to_string(),
            ),
            (
                DRAGON_PROFILE_JSON_METADATA_KEY.into(),
                serde_json::to_string(self)?,
            ),
        ]))
    }

    pub fn attach_to_entry(&self, entry: &mut ExperimentDirectoryEntry) -> Result<()> {
        dragon_profile_attachment()
            .attach(entry, self)
            .map_err(|error| {
                anyhow!(
                    "failed to attach Dragon experiment profile for {}: {error}",
                    entry.experiment_id.as_str()
                )
            })
    }

    pub fn from_entry_metadata(entry: &ExperimentDirectoryEntry) -> Result<Option<Self>> {
        dragon_profile_attachment().decode(entry).map_err(|error| {
            anyhow!(
                "failed to decode Dragon experiment profile for {}: {error}",
                entry.experiment_id.as_str()
            )
        })
    }
}

pub fn find_matching_entry<'a>(
    entries: &'a [ExperimentDirectoryEntry],
    selected_experiment_id: Option<&str>,
    selected_revision_id: Option<&str>,
    experiment_kind: Option<DragonExperimentKind>,
) -> Result<Option<&'a ExperimentDirectoryEntry>> {
    find_matching_directory_entry_with_predicate::<DragonExperimentProfile, _>(
        entries,
        &dragon_profile_attachment(),
        selected_experiment_id,
        selected_revision_id,
        |profile| {
            experiment_kind
                .map(|experiment_kind| profile.experiment_kind == experiment_kind)
                .unwrap_or(true)
        },
    )
}

#[cfg(feature = "native")]
fn ensure_portable_native_profile(
    config: &TrainingConfig,
    experiment_kind: DragonExperimentKind,
) -> Result<()> {
    match (&config.dataset.source, experiment_kind) {
        (DatasetSourceConfig::UniversalityNca { .. }, DragonExperimentKind::NcaPrepretraining)
        | (
            DatasetSourceConfig::UniversalityRuliad { .. },
            DragonExperimentKind::RuliadPretraining,
        )
        | (
            DatasetSourceConfig::NemotronClimbMix { .. },
            DragonExperimentKind::ClimbMixPretraining,
        ) => {}
        _ => bail!(
            "network-published Dragon profiles currently support only universality_nca, universality_ruliad, and nemotron_climb_mix datasets"
        ),
    }

    if config.training.resume_run_dir.is_some() {
        bail!("network-published Dragon profiles do not support training.resume_run_dir");
    }
    if config.training.init_checkpoint_path.is_some() {
        bail!("network-published Dragon profiles do not support training.init_checkpoint_path");
    }
    if config
        .training
        .init_transfer
        .interface_checkpoint_path
        .is_some()
    {
        bail!(
            "network-published Dragon profiles do not support training.init_transfer.interface_checkpoint_path"
        );
    }
    Ok(())
}

#[cfg(feature = "native")]
fn portable_training_template(
    config: &TrainingConfig,
    corpus_kind: Option<PortableUniversalityCorpusKind>,
) -> Result<String> {
    let mut portable = config.clone();
    portable.dataset.cache_dir = PathBuf::from(PORTABLE_CACHE_DIR_NAME);
    if let Some(validation) = portable.dataset.validation.as_mut() {
        validation.cache_dir = None;
    }
    if let Some(kind) = corpus_kind {
        kind.apply_source_path(&mut portable, PathBuf::from(kind.file_name()));
    }
    toml::to_string(&portable).map_err(Into::into)
}

#[cfg(feature = "native")]
fn browser_profile_from_native_config(
    config: &TrainingConfig,
    experiment_kind: DragonExperimentKind,
    model_config: &DragonConfig,
    portable_corpus: Option<&PortableUniversalityCorpus>,
    revision_id: Option<&str>,
    browser_climbmix_manifest_url: Option<&str>,
) -> Result<Option<DragonBrowserExperimentProfile>> {
    let optimizer = match config.optimizer.name {
        burn_dragon_train::OptimizerKind::Adamw => DragonBrowserOptimizerConfig::Adamw,
        burn_dragon_train::OptimizerKind::Eggroll => DragonBrowserOptimizerConfig::SeededFitness {
            eggroll: config.optimizer.effective_eggroll_config(),
            scalar_encoding: burn_p2p::CompactScalarEncoding::SymmetricInt16,
        },
        burn_dragon_train::OptimizerKind::PredictiveCoding => return Ok(None),
    };
    match (&config.dataset.source, experiment_kind) {
        (
            DatasetSourceConfig::UniversalityNca {
                config: nca_config_path,
            },
            DragonExperimentKind::NcaPrepretraining,
        ) => {
            let corpus_toml = match portable_corpus {
                Some(PortableUniversalityCorpus {
                    kind: PortableUniversalityCorpusKind::Nca,
                    toml,
                }) => toml.clone(),
                _ => fs::read_to_string(nca_config_path).map_err(|error| {
                    anyhow!(
                        "failed to read portable NCA corpus config {}: {error}",
                        nca_config_path.display()
                    )
                })?,
            };
            let window_tuning = DragonBrowserWindowTuning::nca_wgpu_from_native(config);
            let capability_policy = DragonCapabilityPolicy {
                browser_wgpu_memory_budget_bytes: Some(
                    DEFAULT_NCA_BROWSER_WGPU_MEMORY_BUDGET_BYTES,
                ),
                ..DragonCapabilityPolicy::default()
            };
            Ok(Some(DragonBrowserExperimentProfile {
                model_config: model_config.clone(),
                training_objective: config.training.objective.clone(),
                optimizer: optimizer.clone(),
                execution_backend: DragonBrowserExecutionBackend::Auto,
                block_size: config.training.block_size,
                tbptt_chunk_size: config.training.tbptt_chunk_size,
                tbptt_persist_across_steps: config.training.tbptt_persist_across_steps,
                learning_rate: config.optimizer.learning_rate,
                weight_decay: config.optimizer.weight_decay,
                batch_size: window_tuning.batch_size,
                max_train_batches: Some(window_tuning.max_train_batches),
                max_eval_batches: Some(window_tuning.max_eval_batches),
                capability_policy,
                trainer_support: DragonBrowserTrainerSupport::Supported,
                train_source: DragonBrowserProfileTokenSource::GeneratedNca {
                    corpus_toml: corpus_toml.clone(),
                    split: DragonBrowserDatasetSplit::Train,
                    max_documents: Some(window_tuning.train_document_pool),
                },
                eval_source: Some(DragonBrowserProfileTokenSource::GeneratedNca {
                    corpus_toml,
                    split: DragonBrowserDatasetSplit::Validation,
                    max_documents: Some(window_tuning.eval_document_pool),
                }),
            }))
        }
        (
            DatasetSourceConfig::NemotronClimbMix { .. },
            DragonExperimentKind::ClimbMixPretraining,
        ) => Ok(Some(DragonBrowserExperimentProfile {
            model_config: model_config.clone(),
            training_objective: config.training.objective.clone(),
            optimizer,
            execution_backend: DragonBrowserExecutionBackend::Auto,
            block_size: config.training.block_size,
            tbptt_chunk_size: config.training.tbptt_chunk_size,
            tbptt_persist_across_steps: config.training.tbptt_persist_across_steps,
            learning_rate: config.optimizer.learning_rate,
            weight_decay: config.optimizer.weight_decay,
            batch_size: config.training.batch_size,
            max_train_batches: Some(config.training.max_iters.max(1)),
            max_eval_batches: None,
            capability_policy: DragonCapabilityPolicy::default(),
            trainer_support: DragonBrowserTrainerSupport::Supported,
            train_source: DragonBrowserProfileTokenSource::ShardManifestHttp {
                manifest_url: browser_climbmix_manifest_url
                    .map(str::trim)
                    .filter(|url| !url.is_empty())
                    .map(ToOwned::to_owned)
                    .unwrap_or_else(|| {
                        default_browser_climbmix_manifest_url(experiment_kind, revision_id)
                    }),
                selection: DragonBrowserShardSelectionPolicy::DeterministicPeer,
                max_shards_per_window: Some(DEFAULT_BROWSER_CLIMBMIX_MAX_SHARDS_PER_WINDOW),
            },
            eval_source: None,
        })),
        (
            DatasetSourceConfig::UniversalityRuliad {
                config: ruliad_config_path,
            },
            DragonExperimentKind::RuliadPretraining,
        ) => {
            let corpus_toml = match portable_corpus {
                Some(PortableUniversalityCorpus {
                    kind: PortableUniversalityCorpusKind::Ruliad,
                    toml,
                }) => toml.clone(),
                _ => fs::read_to_string(ruliad_config_path).map_err(|error| {
                    anyhow!(
                        "failed to read portable Ruliad corpus config {}: {error}",
                        ruliad_config_path.display()
                    )
                })?,
            };
            let browser_ruliad_corpus: RuliadCorpusConfig = toml::from_str(&corpus_toml)
                .map_err(|error| anyhow!("invalid browser Ruliad corpus TOML: {error}"))?;
            let window_tuning = DragonBrowserWindowTuning::nca_wgpu_from_native(config);
            let capability_policy = DragonCapabilityPolicy {
                browser_wgpu_memory_budget_bytes: Some(
                    DEFAULT_NCA_BROWSER_WGPU_MEMORY_BUDGET_BYTES,
                ),
                ..DragonCapabilityPolicy::default()
            };
            Ok(Some(DragonBrowserExperimentProfile {
                model_config: model_config.clone(),
                training_objective: config.training.objective.clone(),
                optimizer,
                execution_backend: DragonBrowserExecutionBackend::Auto,
                block_size: config.training.block_size,
                tbptt_chunk_size: config.training.tbptt_chunk_size,
                tbptt_persist_across_steps: config.training.tbptt_persist_across_steps,
                learning_rate: config.optimizer.learning_rate,
                weight_decay: config.optimizer.weight_decay,
                batch_size: window_tuning.batch_size,
                max_train_batches: Some(window_tuning.max_train_batches),
                max_eval_batches: Some(window_tuning.max_eval_batches),
                capability_policy,
                trainer_support: if config
                    .training
                    .ruliad_supervision
                    .needs_ruliad_policy_batch()
                {
                    DragonBrowserTrainerSupport::ObserverOnly {
                        reason: "the signed Ruliad objective requires verifier or denoising policy batches that are not implemented by the browser trainer".into(),
                    }
                } else if browser_ruliad_corpus.source_selection.enabled {
                    DragonBrowserTrainerSupport::ObserverOnly {
                        reason: "the signed Ruliad revision uses adaptive live source selection whose capability feedback state is not synchronized into browser leases".into(),
                    }
                } else {
                    DragonBrowserTrainerSupport::Supported
                },
                train_source: DragonBrowserProfileTokenSource::GeneratedRuliad {
                    corpus_toml: corpus_toml.clone(),
                    split: DragonBrowserDatasetSplit::Train,
                    max_documents: Some(window_tuning.train_document_pool),
                    supervision: config.training.ruliad_supervision.token_supervision(),
                },
                eval_source: Some(DragonBrowserProfileTokenSource::GeneratedRuliad {
                    corpus_toml,
                    split: DragonBrowserDatasetSplit::Validation,
                    max_documents: Some(window_tuning.eval_document_pool),
                    supervision: config.training.ruliad_supervision.token_supervision(),
                }),
            }))
        }
        _ => Ok(None),
    }
}

#[cfg(feature = "native")]
fn default_browser_climbmix_manifest_url(
    experiment_kind: DragonExperimentKind,
    revision_id: Option<&str>,
) -> String {
    match revision_id {
        Some(revision_id) if !revision_id.trim().is_empty() => format!(
            "/dragon-datasets/{}/{}/fetch-manifest.json",
            experiment_kind.workload_slug(),
            revision_id.trim()
        ),
        _ => format!(
            "/dragon-datasets/{}/fetch-manifest.json",
            experiment_kind.workload_slug()
        ),
    }
}

#[cfg(feature = "native")]
pub fn build_profile_from_local_config(
    config: &TrainingConfig,
    experiment_kind: DragonExperimentKind,
    revision_id: Option<&str>,
    browser_climbmix_manifest_url: Option<&str>,
) -> Result<DragonExperimentProfile> {
    ensure_portable_native_profile(config, experiment_kind)?;
    let model_config = build_model_config_with_tokenizer(
        &config.model,
        config.training.block_size,
        config
            .dataset
            .tokenizer
            .fit(std::iter::empty::<&str>())?
            .as_ref(),
    )?;
    let portable_corpus = portable_universality_corpus(config)?;
    let portable_kind = portable_corpus.as_ref().map(|corpus| corpus.kind);
    let nca_corpus_toml = portable_corpus
        .as_ref()
        .filter(|corpus| corpus.kind == PortableUniversalityCorpusKind::Nca)
        .map(|corpus| corpus.toml.clone());
    let ruliad_corpus_toml = portable_corpus
        .as_ref()
        .filter(|corpus| corpus.kind == PortableUniversalityCorpusKind::Ruliad)
        .map(|corpus| corpus.toml.clone());
    Ok(DragonExperimentProfile {
        version: DRAGON_PROFILE_VERSION,
        experiment_kind,
        native: DragonNativeExperimentProfile {
            training_toml: portable_training_template(config, portable_kind)?,
            nca_corpus_toml,
            ruliad_corpus_toml,
        },
        browser: browser_profile_from_native_config(
            config,
            experiment_kind,
            &model_config,
            portable_corpus.as_ref(),
            revision_id,
            browser_climbmix_manifest_url,
        )?,
    })
}

#[cfg(feature = "native")]
fn profile_storage_root_for_ids(
    storage_root: &Path,
    study_id: &str,
    experiment_id: &str,
    revision_id: &str,
) -> PathBuf {
    storage_root
        .join("network_profiles")
        .join(study_id)
        .join(experiment_id)
        .join(revision_id)
}

#[cfg(feature = "native")]
fn validation_cache_dir_for(cache_dir: &Path, validation: &mut ValidationDatasetConfig) {
    validation.cache_dir = Some(cache_dir.join("validation"));
}

#[cfg(feature = "native")]
fn materialize_portable_universality_corpus(
    config: &mut TrainingConfig,
    profile_root: &Path,
    experiment_id: &str,
    kind: PortableUniversalityCorpusKind,
    corpus_toml: &str,
) -> Result<()> {
    let output_dir = profile_root.join(kind.generated_dir_name());
    let corpus_path = profile_root.join(kind.file_name());
    let serialized = match kind {
        PortableUniversalityCorpusKind::Nca => {
            let mut corpus = toml::from_str::<NcaCorpusConfig>(corpus_toml).map_err(|error| {
                anyhow!(
                    "failed to decode portable {} corpus config for {experiment_id}: {error}",
                    kind.label()
                )
            })?;
            corpus.output_dir = output_dir;
            toml::to_string(&corpus)?
        }
        PortableUniversalityCorpusKind::Ruliad => {
            let mut corpus =
                toml::from_str::<RuliadCorpusConfig>(corpus_toml).map_err(|error| {
                    anyhow!(
                        "failed to decode portable {} corpus config for {experiment_id}: {error}",
                        kind.label()
                    )
                })?;
            corpus.output_dir = output_dir;
            toml::to_string(&corpus)?
        }
    };
    fs::write(&corpus_path, serialized)?;
    kind.apply_source_path(config, corpus_path);
    Ok(())
}

#[cfg(feature = "native")]
pub fn materialize_native_training_config(
    storage_root: &Path,
    entry: &ExperimentDirectoryEntry,
    profile: &DragonExperimentProfile,
) -> Result<TrainingConfig> {
    materialize_native_training_config_for_ids(
        storage_root,
        entry.study_id.as_str(),
        entry.experiment_id.as_str(),
        entry.current_revision_id.as_str(),
        profile,
    )
}

#[cfg(feature = "native")]
fn materialize_native_training_config_for_ids(
    storage_root: &Path,
    study_id: &str,
    experiment_id: &str,
    revision_id: &str,
    profile: &DragonExperimentProfile,
) -> Result<TrainingConfig> {
    let mut config =
        toml::from_str::<TrainingConfig>(&profile.native.training_toml).map_err(|error| {
            anyhow!("failed to decode native Dragon training config for {experiment_id}: {error}")
        })?;
    let profile_root =
        profile_storage_root_for_ids(storage_root, study_id, experiment_id, revision_id);
    let cache_dir = profile_root.join("cache");
    fs::create_dir_all(&cache_dir)?;
    config.dataset.cache_dir = cache_dir.clone();
    if let Some(validation) = config.dataset.validation.as_mut() {
        validation_cache_dir_for(&cache_dir, validation);
    }
    if profile.native.nca_corpus_toml.is_some() && profile.native.ruliad_corpus_toml.is_some() {
        bail!(
            "native Dragon profile for {experiment_id} must include at most one portable universality corpus"
        );
    }

    if let Some(corpus_toml) = profile.native.nca_corpus_toml.as_ref() {
        materialize_portable_universality_corpus(
            &mut config,
            &profile_root,
            experiment_id,
            PortableUniversalityCorpusKind::Nca,
            corpus_toml,
        )?;
    }
    if let Some(corpus_toml) = profile.native.ruliad_corpus_toml.as_ref() {
        materialize_portable_universality_corpus(
            &mut config,
            &profile_root,
            experiment_id,
            PortableUniversalityCorpusKind::Ruliad,
            corpus_toml,
        )?;
    }

    Ok(config)
}

#[cfg(feature = "native")]
fn apply_native_training_overrides(
    mut config: TrainingConfig,
    overrides: &DragonNativeTrainingOverrides,
) -> Result<TrainingConfig> {
    if let Some(batch_size) = overrides.batch_size {
        if batch_size == 0 {
            bail!("native training override batch_size must be > 0");
        }
        config.training.batch_size = batch_size;
        if let Some(target_effective_batch_size) = config.training.target_effective_batch_size
            && target_effective_batch_size < batch_size
        {
            config.training.target_effective_batch_size = Some(batch_size);
        }
    }
    if let Some(max_iters) = overrides.max_iters {
        if max_iters == 0 {
            bail!("native training override max_iters must be > 0");
        }
        config.training.max_iters = max_iters;
        config.training.checkpoint_interval_iters = config
            .training
            .checkpoint_interval_iters
            .clamp(1, max_iters);
        config.training.log_frequency = config.training.log_frequency.clamp(1, max_iters);
    }
    config.validate()?;
    Ok(config)
}

#[cfg(feature = "native")]
fn builtin_native_training_profile(
    native: &DragonNativePeerConfig,
    experiment_kind: DragonExperimentKind,
) -> Result<Option<DragonExperimentProfile>> {
    match (
        experiment_kind,
        native.manifest.experiment_id.as_str(),
        native.manifest.revision_id.as_str(),
    ) {
        (DragonExperimentKind::NcaPrepretraining, "nca-prepretraining", "nca-r1") => {
            Ok(Some(serde_json::from_str(BUILTIN_NCA_R1_PROFILE_JSON)?))
        }
        _ => Ok(None),
    }
}

#[cfg(feature = "native")]
fn manifest_seed_from_entry(
    default_seed: &DragonManifestSeed,
    entry: &ExperimentDirectoryEntry,
) -> Result<DragonManifestSeed> {
    let mut seed = default_seed.clone();
    seed.network_id = entry.network_id.as_str().to_owned();
    seed.study_id = entry.study_id.as_str().to_owned();
    seed.experiment_id = entry.experiment_id.as_str().to_owned();
    seed.revision_id = entry.current_revision_id.as_str().to_owned();
    seed.display_name = entry.display_name.clone();
    if let Some(topology) = entry.merge_topology_policy() {
        seed.promotion = match topology.promotion_policy.mode {
            burn_p2p::HeadPromotionMode::DiffusionSteadyState => DragonPromotionConfig {
                mode: DragonPromotionMode::DiffusionSteadyState,
                validator_quorum: 1,
            },
            burn_p2p::HeadPromotionMode::ValidatorQuorum => DragonPromotionConfig {
                mode: DragonPromotionMode::ValidatorQuorum,
                validator_quorum: topology.promotion_policy.validator_quorum,
            },
            burn_p2p::HeadPromotionMode::ReducerAuthority => {
                anyhow::bail!(
                    "network profile requests reducer-authority promotion, which Dragon does not expose"
                )
            }
        };
    }
    Ok(seed)
}

#[cfg(feature = "native")]
fn fetch_matching_profile_entry(
    snapshot: &BrowserEdgeSnapshot,
    native: &DragonNativePeerConfig,
    experiment_kind: DragonExperimentKind,
) -> Result<Option<(ExperimentDirectoryEntry, DragonExperimentProfile)>> {
    let Some(entry) = find_matching_entry(
        &snapshot.directory.entries,
        Some(&native.manifest.experiment_id),
        Some(&native.manifest.revision_id),
        Some(experiment_kind),
    )?
    else {
        return Ok(None);
    };
    let Some(profile) = DragonExperimentProfile::from_entry_metadata(entry)? else {
        return Ok(None);
    };
    Ok(Some((entry.clone(), profile)))
}

#[cfg(feature = "native")]
pub fn resolve_native_training_profile(
    native: &DragonNativePeerConfig,
    experiment_kind: DragonExperimentKind,
    use_network_profile: bool,
) -> Result<ResolvedNativeTrainingProfile> {
    let has_local_training = !native.training_config_paths.is_empty();

    if use_network_profile && let Some(edge_base_url) = native.effective_edge_base_url() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        match runtime.block_on(fetch_edge_snapshot(edge_base_url)) {
            Ok(snapshot) => {
                if let Some((entry, profile)) =
                    fetch_matching_profile_entry(&snapshot, native, experiment_kind)?
                {
                    let config =
                        materialize_native_training_config(&native.storage_root, &entry, &profile)?;
                    let config =
                        apply_native_training_overrides(config, &native.training_overrides)?;
                    return Ok(ResolvedNativeTrainingProfile {
                        config,
                        manifest_seed: manifest_seed_from_entry(&native.manifest, &entry)?,
                        profile,
                        directory_entry: Some(entry),
                        source: DragonResolvedProfileSource::NetworkPublished,
                    });
                }
            }
            Err(error) if !has_local_training => return Err(error),
            Err(_) => {}
        }
    }

    if has_local_training {
        let config = load_training_config(&native.training_config_paths)?;
        let config = apply_native_training_overrides(config, &native.training_overrides)?;
        let profile = build_profile_from_local_config(
            &config,
            experiment_kind,
            Some(&native.manifest.revision_id),
            None,
        )?;
        return Ok(ResolvedNativeTrainingProfile {
            config,
            manifest_seed: native.manifest.clone(),
            profile,
            directory_entry: None,
            source: DragonResolvedProfileSource::LocalConfig,
        });
    }

    if let Some(profile) = builtin_native_training_profile(native, experiment_kind)? {
        let config = materialize_native_training_config_for_ids(
            &native.storage_root,
            &native.manifest.study_id,
            &native.manifest.experiment_id,
            &native.manifest.revision_id,
            &profile,
        )?;
        let config = apply_native_training_overrides(config, &native.training_overrides)?;
        return Ok(ResolvedNativeTrainingProfile {
            config,
            manifest_seed: native.manifest.clone(),
            profile,
            directory_entry: None,
            source: DragonResolvedProfileSource::BuiltinFallback,
        });
    }

    bail!(
        "no network-published Dragon profile was available and native.training_config_paths is empty"
    )
}

#[cfg(any(feature = "wasm-peer", feature = "native"))]
fn browser_source_from_profile(
    source: DragonBrowserProfileTokenSource,
) -> Result<DragonBrowserTokenSource> {
    match source {
        DragonBrowserProfileTokenSource::Inline { records } => {
            Ok(DragonBrowserTokenSource::Inline { records })
        }
        DragonBrowserProfileTokenSource::HttpJson { url } => {
            Ok(DragonBrowserTokenSource::HttpJson { url })
        }
        DragonBrowserProfileTokenSource::ShardManifestHttp {
            manifest_url,
            selection,
            max_shards_per_window,
        } => Ok(DragonBrowserTokenSource::ShardManifestHttp {
            manifest_url,
            selection,
            max_shards_per_window,
        }),
        DragonBrowserProfileTokenSource::GeneratedNca {
            corpus_toml,
            split,
            max_documents,
        } => Ok(DragonBrowserTokenSource::GeneratedNca {
            corpus: toml::from_str(&corpus_toml)?,
            split,
            max_documents,
        }),
        DragonBrowserProfileTokenSource::GeneratedRuliad {
            corpus_toml,
            split,
            max_documents,
            supervision,
        } => Ok(DragonBrowserTokenSource::GeneratedRuliad {
            corpus: Box::new(toml::from_str(&corpus_toml)?),
            split,
            max_documents,
            supervision,
        }),
    }
}

#[cfg(any(feature = "wasm-peer", feature = "native"))]
pub fn browser_training_config_from_profile(
    entry: &ExperimentDirectoryEntry,
    profile: &DragonExperimentProfile,
) -> Result<Option<DragonBrowserTrainingConfig>> {
    if !browser_training_protocol_supported(entry) {
        return Ok(None);
    }
    if !entry.browser_role_allowed(BrowserRole::TrainerWgpu) {
        return Ok(None);
    }
    let Some(browser) = profile.browser.clone() else {
        return Ok(None);
    };
    if !browser.trainer_support.is_supported() {
        return Ok(None);
    }
    Ok(Some(DragonBrowserTrainingConfig {
        experiment_kind: profile.experiment_kind,
        model_config: browser.model_config,
        training_objective: browser.training_objective,
        optimizer: browser.optimizer,
        execution_backend: browser.execution_backend,
        block_size: browser.block_size,
        tbptt_chunk_size: browser.tbptt_chunk_size,
        tbptt_persist_across_steps: browser.tbptt_persist_across_steps,
        learning_rate: browser.learning_rate,
        weight_decay: browser.weight_decay,
        batch_size: browser.batch_size,
        max_train_batches: browser.max_train_batches,
        max_eval_batches: browser.max_eval_batches,
        capability_policy: browser.capability_policy,
        training_lease: None,
        train_source: browser_source_from_profile(browser.train_source)?,
        eval_source: match browser.eval_source {
            Some(source) => Some(browser_source_from_profile(source)?),
            None => None,
        },
        live_participant: Some(DragonBrowserLiveParticipantConfig {
            principal_id: None,
            study_id: entry.study_id.as_str().to_owned(),
            experiment_id: entry.experiment_id.as_str().to_owned(),
            revision_id: entry.current_revision_id.as_str().to_owned(),
            workload_id: entry.workload_id.as_str().to_owned(),
            publish_canonical_update: true,
            load_active_head_artifact: true,
            revision_contract: None,
        }),
    }))
}

fn browser_training_protocol_supported(entry: &ExperimentDirectoryEntry) -> bool {
    matches!(&entry.training_protocol, TrainingProtocol::ArtifactWindows)
}

#[cfg(feature = "native")]
pub fn browser_training_config_from_directory_entries(
    entries: &[ExperimentDirectoryEntry],
    selected_experiment_id: Option<&str>,
    selected_revision_id: Option<&str>,
) -> Result<Option<DragonBrowserTrainingConfig>> {
    let Some(entry) =
        find_matching_entry(entries, selected_experiment_id, selected_revision_id, None)?
    else {
        return Ok(None);
    };

    if let Some(profile) = DragonExperimentProfile::from_entry_metadata(entry)? {
        return browser_training_config_from_profile(entry, &profile);
    }

    match (
        entry.experiment_id.as_str(),
        entry.current_revision_id.as_str(),
    ) {
        ("nca-prepretraining", "nca-r1") => {
            let profile: DragonExperimentProfile =
                serde_json::from_str(BUILTIN_NCA_R1_PROFILE_JSON)?;
            browser_training_config_from_profile(entry, &profile)
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    use burn_p2p::{
        ContentId, DatasetViewId, ExperimentId, ExperimentOptInPolicy,
        ExperimentResourceRequirements, ExperimentScope, ExperimentVisibility, NetworkId, PeerRole,
        PeerRoleSet, RevisionId, StudyId, WorkloadId,
    };

    fn sample_entry() -> ExperimentDirectoryEntry {
        ExperimentDirectoryEntry {
            network_id: NetworkId::new("dragon-net"),
            study_id: StudyId::new("dragon-study"),
            experiment_id: ExperimentId::new("nca-prepretraining"),
            workload_id: WorkloadId::new("dragon-nca"),
            display_name: "NCA".into(),
            model_schema_hash: ContentId::new("schema"),
            dataset_view_id: DatasetViewId::new("view"),
            resource_requirements: ExperimentResourceRequirements {
                minimum_roles: BTreeSet::from([PeerRole::TrainerGpu]),
                minimum_device_memory_bytes: None,
                minimum_system_memory_bytes: None,
                estimated_download_bytes: 0,
                estimated_window_seconds: 30,
            },
            visibility: ExperimentVisibility::Public,
            opt_in_policy: ExperimentOptInPolicy::Open,
            current_revision_id: RevisionId::new("r1"),
            current_head_id: None,
            allowed_roles: PeerRoleSet::new([PeerRole::TrainerGpu]),
            allowed_scopes: BTreeSet::from([ExperimentScope::Connect]),
            training_protocol: Default::default(),
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn profile_metadata_round_trip_decodes_from_directory_entry() {
        let profile = DragonExperimentProfile {
            version: DRAGON_PROFILE_VERSION,
            experiment_kind: DragonExperimentKind::NcaPrepretraining,
            native: DragonNativeExperimentProfile {
                training_toml: "[training]\nblock_size = 64\nbatch_size = 2\n".into(),
                nca_corpus_toml: Some("seed = 1337\n".into()),
                ruliad_corpus_toml: None,
            },
            browser: None,
        };
        let mut entry = sample_entry();
        entry.metadata.extend(
            profile
                .metadata_entries()
                .expect("profile metadata should serialize"),
        );

        let decoded = DragonExperimentProfile::from_entry_metadata(&entry)
            .expect("profile metadata should decode")
            .expect("profile should be present");

        assert_eq!(decoded, profile);
    }

    #[test]
    fn browser_training_is_hidden_for_diloco_revisions() {
        let mut entry = sample_entry();
        assert!(browser_training_protocol_supported(&entry));
        entry.training_protocol = TrainingProtocol::DiLoCo(burn_p2p::DiLoCoPolicy::default());
        assert!(!browser_training_protocol_supported(&entry));
    }

    #[test]
    fn browser_profile_and_runtime_execution_contracts_match_exactly() {
        let browser = DragonBrowserExperimentProfile {
            model_config: DragonConfig::default(),
            training_objective: Default::default(),
            optimizer: Default::default(),
            execution_backend: DragonBrowserExecutionBackend::Auto,
            block_size: 8,
            tbptt_chunk_size: Some(4),
            tbptt_persist_across_steps: true,
            learning_rate: 1.0e-3,
            weight_decay: 0.01,
            batch_size: 2,
            max_train_batches: Some(3),
            max_eval_batches: Some(1),
            capability_policy: DragonCapabilityPolicy::default(),
            trainer_support: DragonBrowserTrainerSupport::Supported,
            train_source: DragonBrowserProfileTokenSource::Inline {
                records: vec![TokenWindowRecord {
                    inputs: vec![1; 8],
                    targets: vec![2; 8],
                    loss_mask: None,
                    reset_stream_state: true,
                    stream_group_id: Some(1),
                    stream_row: Some(0),
                    chunk_index: Some(0),
                }],
            },
            eval_source: None,
        };
        let profile = DragonExperimentProfile {
            version: DRAGON_PROFILE_VERSION,
            experiment_kind: DragonExperimentKind::NcaPrepretraining,
            native: DragonNativeExperimentProfile {
                training_toml: String::new(),
                nca_corpus_toml: None,
                ruliad_corpus_toml: None,
            },
            browser: Some(browser.clone()),
        };
        let runtime = DragonBrowserTrainingConfig {
            experiment_kind: profile.experiment_kind,
            model_config: browser.model_config.clone(),
            training_objective: browser.training_objective.clone(),
            optimizer: browser.optimizer.clone(),
            execution_backend: browser.execution_backend,
            block_size: browser.block_size,
            tbptt_chunk_size: browser.tbptt_chunk_size,
            tbptt_persist_across_steps: browser.tbptt_persist_across_steps,
            learning_rate: browser.learning_rate,
            weight_decay: browser.weight_decay,
            batch_size: browser.batch_size,
            max_train_batches: browser.max_train_batches,
            max_eval_batches: browser.max_eval_batches,
            capability_policy: browser.capability_policy.clone(),
            training_lease: None,
            train_source: browser_source_from_profile(browser.train_source.clone())
                .expect("runtime train source"),
            eval_source: browser
                .eval_source
                .clone()
                .map(browser_source_from_profile)
                .transpose()
                .expect("runtime eval source"),
            live_participant: None,
        };

        assert_eq!(
            browser_profile_execution_contract_hash(profile.experiment_kind, &browser)
                .expect("profile hash"),
            browser_runtime_execution_contract_hash(&runtime).expect("runtime hash"),
        );

        let mut changed = runtime;
        changed.tbptt_chunk_size = Some(2);
        assert_ne!(
            browser_profile_execution_contract_hash(profile.experiment_kind, &browser)
                .expect("profile hash"),
            browser_runtime_execution_contract_hash(&changed).expect("changed runtime hash"),
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn climbmix_profile_builds_browser_shard_manifest_source() {
        let config: TrainingConfig = toml::from_str(
            r#"
[dataset]
cache_dir = "./cache/climbmix-r1"
train_split_ratio = 0.9
type = "nemotron_climb_mix"
max_records = 256

[dataset.tokenizer]
type = "pretokenized"
vocab_size = 50257
eos_id = 50256

[model]
n_layer = 6
n_embd = 96
n_head = 8
latent_total = 192

[training]
block_size = 128
batch_size = 4
max_iters = 32
checkpoint_interval_iters = 4
log_frequency = 1
seed = 1337

[optimizer]
learning_rate = 0.003
weight_decay = 0.0

[generation]
prompt = "1 2 3"
"#,
        )
        .expect("training config");

        let profile = build_profile_from_local_config(
            &config,
            DragonExperimentKind::ClimbMixPretraining,
            Some("climbmix-r1"),
            None,
        )
        .expect("profile");

        match profile.browser.expect("browser profile").train_source {
            DragonBrowserProfileTokenSource::ShardManifestHttp {
                manifest_url,
                selection,
                max_shards_per_window,
            } => {
                assert_eq!(
                    manifest_url,
                    "/dragon-datasets/climbmix-pretraining/climbmix-r1/fetch-manifest.json"
                );
                assert_eq!(
                    selection,
                    DragonBrowserShardSelectionPolicy::DeterministicPeer
                );
                assert_eq!(
                    max_shards_per_window,
                    Some(DEFAULT_BROWSER_CLIMBMIX_MAX_SHARDS_PER_WINDOW)
                );
            }
            other => panic!("expected shard-manifest browser source, got {other:?}"),
        }
    }

    #[cfg(feature = "native")]
    #[test]
    fn ruliad_profile_materializes_portable_native_corpus() {
        use burn_dragon_universality::{
            RuliadSerializationConfig, RuliadSourceSelectionConfig, RuliadTokenizationConfig,
            compact_ruliad_families,
        };
        use tempfile::tempdir;

        let dir = tempdir().expect("config dir");
        let ruliad_config_path = dir.path().join("ruliad.toml");
        let ruliad_corpus = RuliadCorpusConfig {
            output_dir: dir.path().join("generated"),
            seed: 1337,
            name: "profile-ruliad".into(),
            train_samples: 8,
            validation_samples: 4,
            chunk_token_capacity: 1024,
            serialization: RuliadSerializationConfig {
                document_tokens: 513,
                preview_samples: 1,
                ..RuliadSerializationConfig::default()
            },
            tokenization: RuliadTokenizationConfig::default(),
            formal_generalization: Default::default(),
            source_selection: RuliadSourceSelectionConfig::default(),
            families: compact_ruliad_families(),
            proof_tasks: None,
            lean_task_limit: None,
        };
        fs::write(
            &ruliad_config_path,
            toml::to_string_pretty(&ruliad_corpus).expect("ruliad corpus toml"),
        )
        .expect("write ruliad corpus config");
        let config: TrainingConfig = toml::from_str(&format!(
            r#"
[dataset]
cache_dir = "./cache/ruliad"
train_split_ratio = 0.9
type = "universality_ruliad"
config = "{}"

[dataset.tokenizer]
type = "pretokenized"
vocab_size = 50257
eos_id = 50256

[model]
n_layer = 2
n_embd = 64
n_head = 4
latent_total = 128

[model.language_head]
type = "standard_token_classification"

[training]
block_size = 512
batch_size = 2
max_iters = 8
checkpoint_interval_iters = 4
log_frequency = 1
seed = 1337

[optimizer]
learning_rate = 0.001
weight_decay = 0.0

[generation]
prompt = "[R2"
"#,
            ruliad_config_path.display()
        ))
        .expect("training config");

        let profile = build_profile_from_local_config(
            &config,
            DragonExperimentKind::RuliadPretraining,
            Some("ruliad-r1"),
            None,
        )
        .expect("profile");

        assert!(profile.native.nca_corpus_toml.is_none());
        assert!(profile.native.ruliad_corpus_toml.is_some());
        let browser = profile.browser.as_ref().expect("Ruliad browser profile");
        assert_eq!(
            browser.trainer_support,
            DragonBrowserTrainerSupport::Supported
        );
        assert!(matches!(
            &browser.train_source,
            DragonBrowserProfileTokenSource::GeneratedRuliad {
                split: DragonBrowserDatasetSplit::Train,
                supervision,
                ..
            } if *supervision == config.training.ruliad_supervision.token_supervision()
        ));
        assert!(matches!(
            browser.eval_source.as_ref(),
            Some(DragonBrowserProfileTokenSource::GeneratedRuliad {
                split: DragonBrowserDatasetSplit::Validation,
                ..
            })
        ));
        let portable_training: TrainingConfig =
            toml::from_str(&profile.native.training_toml).expect("portable training config");
        assert!(matches!(
            &portable_training.dataset.source,
            DatasetSourceConfig::UniversalityRuliad { .. }
        ));
        if let DatasetSourceConfig::UniversalityRuliad { config } =
            &portable_training.dataset.source
        {
            assert_eq!(config, &PathBuf::from(PORTABLE_RULIAD_CORPUS_FILE_NAME));
        }

        let storage = tempdir().expect("storage");
        let materialized = materialize_native_training_config_for_ids(
            storage.path(),
            "study",
            "experiment",
            "r1",
            &profile,
        )
        .expect("materialized config");
        let DatasetSourceConfig::UniversalityRuliad {
            config: materialized_config_path,
        } = materialized.dataset.source
        else {
            panic!("expected materialized ruliad source");
        };
        assert!(materialized_config_path.is_file());
        let corpus: burn_dragon_universality::RuliadCorpusConfig =
            toml::from_str(&fs::read_to_string(&materialized_config_path).expect("read corpus"))
                .expect("materialized corpus config");
        assert!(corpus.output_dir.ends_with("ruliad-generated"));

        let adaptive_config_path = dir.path().join("ruliad-adaptive.toml");
        let mut adaptive_corpus = ruliad_corpus.clone();
        adaptive_corpus.source_selection.enabled = true;
        fs::write(
            &adaptive_config_path,
            toml::to_string_pretty(&adaptive_corpus).expect("adaptive ruliad corpus toml"),
        )
        .expect("write adaptive ruliad corpus config");
        let mut adaptive_config = config.clone();
        adaptive_config.dataset.source = DatasetSourceConfig::UniversalityRuliad {
            config: adaptive_config_path,
        };
        let adaptive_profile = build_profile_from_local_config(
            &adaptive_config,
            DragonExperimentKind::RuliadPretraining,
            Some("ruliad-r1-adaptive"),
            None,
        )
        .expect("adaptive profile");
        assert!(matches!(
            adaptive_profile
                .browser
                .expect("adaptive browser profile")
                .trainer_support,
            DragonBrowserTrainerSupport::ObserverOnly { reason }
                if reason.contains("adaptive live source selection")
        ));

        let mut auxiliary_config = config.clone();
        auxiliary_config
            .training
            .ruliad_supervision
            .verifier_reward
            .enabled = true;
        auxiliary_config
            .training
            .ruliad_supervision
            .verifier_reward
            .weight = 0.1;
        let auxiliary_profile = build_profile_from_local_config(
            &auxiliary_config,
            DragonExperimentKind::RuliadPretraining,
            Some("ruliad-r1-auxiliary"),
            None,
        )
        .expect("auxiliary profile");
        assert!(matches!(
            auxiliary_profile
                .browser
                .expect("observer browser profile")
                .trainer_support,
            DragonBrowserTrainerSupport::ObserverOnly { .. }
        ));
    }

    #[cfg(feature = "native")]
    #[test]
    fn builtin_r3_profile_binds_streaming_supervision_and_fails_closed_in_browser() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let config = burn_dragon_language::load_training_config(&[
            manifest_dir.join("deploy/profiles/ruliad-r3.training.toml")
        ])
        .expect("load builtin R3 profile");
        let DatasetSourceConfig::UniversalityRuliad {
            config: corpus_path,
        } = &config.dataset.source
        else {
            panic!("R3 profile must use the Ruliad source");
        };
        let corpus =
            burn_dragon_universality::load_ruliad_config(&resolve_local_profile_path(corpus_path))
                .expect("load builtin R3 corpus");
        assert_eq!(
            corpus.source_selection.formal_task_mix,
            burn_dragon_universality::RuliadFormalTaskMixConfig {
                advance_proof_weight: 2,
                select_proof_action_weight: 0,
                construct_proof_weight: 1,
                check_proof_weight: 1,
                proof_action_answer_contract: Default::default(),
            }
        );
        let profile = build_profile_from_local_config(
            &config,
            DragonExperimentKind::RuliadPretraining,
            Some("ruliad-r3"),
            None,
        )
        .expect("build R3 P2P profile");
        let browser = profile.browser.expect("R3 browser observer profile");

        assert_eq!(browser.block_size, 512);
        assert_eq!(browser.tbptt_chunk_size, Some(512));
        assert!(browser.tbptt_persist_across_steps);
        assert!(matches!(
            browser.trainer_support,
            DragonBrowserTrainerSupport::ObserverOnly { reason }
                if reason.contains("adaptive live source selection")
        ));
        assert!(matches!(
            browser.train_source,
            DragonBrowserProfileTokenSource::GeneratedRuliad {
                supervision: burn_dragon_universality::ruliad::RuliadTokenSupervisionConfig {
                    mode:
                        burn_dragon_universality::ruliad::RuliadTokenSupervisionMode::TraceAndAnswer,
                    mask_high_entropy_spans: true,
                    ..
                },
                ..
            }
        ));
        let portable_corpus: burn_dragon_universality::RuliadCorpusConfig = toml::from_str(
            profile
                .native
                .ruliad_corpus_toml
                .as_deref()
                .expect("embedded R3 corpus"),
        )
        .expect("parse embedded R3 corpus");
        assert_eq!(
            portable_corpus.source_selection.formal_task_mix,
            corpus.source_selection.formal_task_mix
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn builtin_nca_profile_fallback_materializes_without_local_or_network_profile() {
        use crate::config::{DragonNativeTrainingOverrides, DragonPeerNetworkConfig};
        use tempfile::tempdir;

        let storage = tempdir().expect("storage");
        let native = DragonNativePeerConfig {
            training_overrides: DragonNativeTrainingOverrides::default(),
            training_config_paths: Vec::new(),
            storage_root: storage.path().to_path_buf(),
            network: DragonPeerNetworkConfig::default(),
            target: None,
            identity: Default::default(),
            bootstrap_peers: Vec::new(),
            manifest: DragonManifestSeed {
                study_id: "burn-dragon-mainnet".into(),
                experiment_id: "nca-prepretraining".into(),
                revision_id: "nca-r1".into(),
                ..DragonManifestSeed::default()
            },
            app_semver: semver::Version::parse(env!("CARGO_PKG_VERSION"))
                .expect("valid burn_dragon version"),
            git_commit: None,
            enabled_features_label: Some("native".into()),
            auth: None,
            capability_policy: DragonCapabilityPolicy::default(),
            shard_export: None,
            existing_shard_dataset: None,
        };

        let resolved = resolve_native_training_profile(
            &native,
            DragonExperimentKind::NcaPrepretraining,
            false,
        )
        .expect("builtin fallback should resolve");

        assert_eq!(
            resolved.manifest_seed.experiment_id,
            "nca-prepretraining".to_owned()
        );
        assert_eq!(resolved.manifest_seed.revision_id, "nca-r1".to_owned());
        assert_eq!(
            resolved.source,
            DragonResolvedProfileSource::BuiltinFallback
        );
        assert_eq!(resolved.config.training.block_size, 512);
        assert_eq!(resolved.config.training.batch_size, 6);
        assert!(matches!(
            resolved.config.dataset.source,
            DatasetSourceConfig::UniversalityNca { .. }
        ));
    }

    #[cfg(feature = "native")]
    #[test]
    fn builtin_nca_browser_window_uses_native_profile_tuning() {
        let profile: DragonExperimentProfile =
            serde_json::from_str(BUILTIN_NCA_R1_PROFILE_JSON).expect("builtin NCA profile");
        let native_config: TrainingConfig =
            toml::from_str(&profile.native.training_toml).expect("native training config");
        let expected = DragonBrowserWindowTuning::nca_wgpu_from_native(&native_config);
        let browser = profile.browser.expect("browser profile");

        assert_eq!(browser.block_size, native_config.training.block_size);
        assert_eq!(browser.learning_rate, native_config.optimizer.learning_rate);
        assert_eq!(browser.weight_decay, native_config.optimizer.weight_decay);
        assert_eq!(browser.batch_size, expected.batch_size);
        assert_eq!(browser.max_train_batches, Some(expected.max_train_batches));
        assert_eq!(browser.max_eval_batches, Some(expected.max_eval_batches));

        match browser.train_source {
            DragonBrowserProfileTokenSource::GeneratedNca { max_documents, .. } => {
                assert_eq!(max_documents, Some(expected.train_document_pool));
            }
            other => panic!("expected generated NCA train source, got {other:?}"),
        }
        match browser.eval_source.expect("eval source") {
            DragonBrowserProfileTokenSource::GeneratedNca { max_documents, .. } => {
                assert_eq!(max_documents, Some(expected.eval_document_pool));
            }
            other => panic!("expected generated NCA eval source, got {other:?}"),
        }
    }

    #[cfg(feature = "native")]
    #[test]
    fn native_training_overrides_apply_to_builtin_profile() {
        use crate::config::{DragonNativeTrainingOverrides, DragonPeerNetworkConfig};
        use tempfile::tempdir;

        let storage = tempdir().expect("storage");
        let native = DragonNativePeerConfig {
            training_overrides: DragonNativeTrainingOverrides {
                batch_size: Some(1),
                max_iters: Some(4),
                max_eval_batches: Some(1),
            },
            training_config_paths: Vec::new(),
            storage_root: storage.path().to_path_buf(),
            network: DragonPeerNetworkConfig::default(),
            target: None,
            identity: Default::default(),
            bootstrap_peers: Vec::new(),
            manifest: DragonManifestSeed {
                study_id: "burn-dragon-mainnet".into(),
                experiment_id: "nca-prepretraining".into(),
                revision_id: "nca-r1".into(),
                ..DragonManifestSeed::default()
            },
            app_semver: semver::Version::parse(env!("CARGO_PKG_VERSION"))
                .expect("valid burn_dragon version"),
            git_commit: None,
            enabled_features_label: Some("native".into()),
            auth: None,
            capability_policy: DragonCapabilityPolicy::default(),
            shard_export: None,
            existing_shard_dataset: None,
        };

        let resolved = resolve_native_training_profile(
            &native,
            DragonExperimentKind::NcaPrepretraining,
            false,
        )
        .expect("builtin fallback should accept resource-only training overrides");

        assert_eq!(
            resolved.source,
            DragonResolvedProfileSource::BuiltinFallback
        );
        assert_eq!(resolved.config.training.batch_size, 1);
        assert_eq!(resolved.config.training.max_iters, 4);
        assert_eq!(resolved.config.model.n_embd, Some(512));
    }

    #[cfg(feature = "native")]
    #[test]
    fn local_training_config_wins_over_builtin_nca_profile_fallback() {
        use crate::config::DragonPeerNetworkConfig;
        use tempfile::tempdir;

        let storage = tempdir().expect("storage");
        let config_dir = tempdir().expect("config");
        let corpus_path = config_dir.path().join("nca-corpus.toml");
        let training_path = config_dir.path().join("nca-training.toml");
        fs::write(
            &corpus_path,
            format!(
                r#"
output_dir = "{}"
seed = 1337
name = "local-nca"
train_samples = 8
validation_samples = 4
chunk_token_capacity = 4096
"#,
                config_dir.path().join("generated").display()
            ),
        )
        .expect("write corpus");
        fs::write(
            &training_path,
            format!(
                r#"
[dataset]
cache_dir = "{}"
train_split_ratio = 0.9
type = "universality_nca"
config = "{}"

[dataset.tokenizer]
type = "pretokenized"
vocab_size = 50257
eos_id = 50256

[model]
n_layer = 8
n_embd = 512
n_head = 8
latent_total = 1024

[model.language_head]
type = "nca_factorized_patch"
state_count = 10
patch_size = 2
frame_special_tokens = true
eos_id = 50256

[training]
block_size = 512
batch_size = 6
max_iters = 24
checkpoint_interval_iters = 8
log_frequency = 1
seed = 1337

[optimizer]
learning_rate = 0.001
weight_decay = 0.0

[generation]
prompt = "0 0 0"
"#,
                config_dir.path().join("cache").display(),
                corpus_path.display()
            ),
        )
        .expect("write training config");

        let native = DragonNativePeerConfig {
            training_overrides: Default::default(),
            training_config_paths: vec![training_path],
            storage_root: storage.path().to_path_buf(),
            network: DragonPeerNetworkConfig::default(),
            target: None,
            identity: Default::default(),
            bootstrap_peers: Vec::new(),
            manifest: DragonManifestSeed {
                study_id: "burn-dragon-mainnet".into(),
                experiment_id: "nca-prepretraining".into(),
                revision_id: "nca-r1".into(),
                ..DragonManifestSeed::default()
            },
            app_semver: semver::Version::parse(env!("CARGO_PKG_VERSION"))
                .expect("valid burn_dragon version"),
            git_commit: None,
            enabled_features_label: Some("native".into()),
            auth: None,
            capability_policy: DragonCapabilityPolicy::default(),
            shard_export: None,
            existing_shard_dataset: None,
        };

        let resolved = resolve_native_training_profile(
            &native,
            DragonExperimentKind::NcaPrepretraining,
            false,
        )
        .expect("local profile should resolve");

        assert_eq!(resolved.source, DragonResolvedProfileSource::LocalConfig);
        assert_eq!(resolved.config.training.block_size, 512);
        assert_eq!(resolved.config.training.batch_size, 6);
        assert_eq!(resolved.config.model.n_layer, Some(8));
        assert_eq!(resolved.config.model.n_embd, Some(512));
        assert_eq!(resolved.config.model.latent_total, Some(1024));
    }
}
