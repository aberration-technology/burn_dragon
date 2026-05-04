use std::collections::BTreeMap;

#[cfg(feature = "native")]
use anyhow::bail;
use anyhow::{Result, anyhow};
use burn_p2p::{BrowserRole, ExperimentDirectoryEntry, ExperimentDirectoryPolicyExt};
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
#[cfg(feature = "native")]
use burn_dragon_universality::NcaCorpusConfig;
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
    DragonBrowserLiveParticipantConfig, DragonBrowserTokenSource, DragonBrowserTrainingConfig,
    DragonBrowserTrainingObjectiveConfig,
};
#[cfg(feature = "native")]
use crate::config::{DragonManifestSeed, DragonNativePeerConfig, DragonNativeTrainingOverrides};

pub const DRAGON_PROFILE_VERSION_METADATA_KEY: &str = "dragon_profile_version";
pub const DRAGON_PROFILE_JSON_METADATA_KEY: &str = "dragon_profile_json";
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
const PORTABLE_CACHE_DIR_NAME: &str = "__dragon_network_profile_cache__";
#[cfg(feature = "native")]
const BUILTIN_NCA_R1_PROFILE_JSON: &str = include_str!("../deploy/profiles/nca-r1.profile.json");

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
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DragonBrowserExperimentProfile {
    pub model_config: DragonConfig,
    #[serde(default)]
    pub execution_backend: DragonBrowserExecutionBackend,
    pub block_size: usize,
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
    pub train_source: DragonBrowserProfileTokenSource,
    #[serde(default)]
    pub eval_source: Option<DragonBrowserProfileTokenSource>,
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
            DatasetSourceConfig::NemotronClimbMix { .. },
            DragonExperimentKind::ClimbMixPretraining,
        ) => {}
        _ => bail!(
            "network-published Dragon profiles currently support only universality_nca and nemotron_climb_mix datasets"
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
    nca_corpus_toml: Option<&str>,
) -> Result<String> {
    let mut portable = config.clone();
    portable.dataset.cache_dir = PathBuf::from(PORTABLE_CACHE_DIR_NAME);
    if let Some(validation) = portable.dataset.validation.as_mut() {
        validation.cache_dir = None;
    }
    if nca_corpus_toml.is_some() {
        portable.dataset.source = DatasetSourceConfig::UniversalityNca {
            config: PathBuf::from(PORTABLE_NCA_CORPUS_FILE_NAME),
        };
        if let Some(validation) = portable.dataset.validation.as_mut()
            && matches!(
                validation.source,
                DatasetSourceConfig::UniversalityNca { .. }
            )
        {
            validation.source = DatasetSourceConfig::UniversalityNca {
                config: PathBuf::from(PORTABLE_NCA_CORPUS_FILE_NAME),
            };
        }
    }
    toml::to_string(&portable).map_err(Into::into)
}

#[cfg(feature = "native")]
fn browser_profile_from_native_config(
    config: &TrainingConfig,
    experiment_kind: DragonExperimentKind,
    model_config: &DragonConfig,
    revision_id: Option<&str>,
    browser_climbmix_manifest_url: Option<&str>,
) -> Result<Option<DragonBrowserExperimentProfile>> {
    match (&config.dataset.source, experiment_kind) {
        (
            DatasetSourceConfig::UniversalityNca {
                config: nca_config_path,
            },
            DragonExperimentKind::NcaPrepretraining,
        ) => {
            let corpus_toml = fs::read_to_string(nca_config_path).map_err(|error| {
                anyhow!(
                    "failed to read portable NCA corpus config {}: {error}",
                    nca_config_path.display()
                )
            })?;
            let window_tuning = DragonBrowserWindowTuning::nca_wgpu_from_native(config);
            let capability_policy = DragonCapabilityPolicy {
                browser_wgpu_memory_budget_bytes: Some(
                    DEFAULT_NCA_BROWSER_WGPU_MEMORY_BUDGET_BYTES,
                ),
                ..DragonCapabilityPolicy::default()
            };
            Ok(Some(DragonBrowserExperimentProfile {
                model_config: model_config.clone(),
                execution_backend: DragonBrowserExecutionBackend::Auto,
                block_size: config.training.block_size,
                learning_rate: config.optimizer.learning_rate,
                weight_decay: config.optimizer.weight_decay,
                batch_size: window_tuning.batch_size,
                max_train_batches: Some(window_tuning.max_train_batches),
                max_eval_batches: Some(window_tuning.max_eval_batches),
                capability_policy,
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
            execution_backend: DragonBrowserExecutionBackend::Auto,
            block_size: config.training.block_size,
            learning_rate: config.optimizer.learning_rate,
            weight_decay: config.optimizer.weight_decay,
            batch_size: config.training.batch_size,
            max_train_batches: Some(config.training.max_iters.max(1)),
            max_eval_batches: None,
            capability_policy: DragonCapabilityPolicy::default(),
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
    let nca_corpus_toml = match &config.dataset.source {
        DatasetSourceConfig::UniversalityNca {
            config: nca_config_path,
        } => Some(fs::read_to_string(nca_config_path).map_err(|error| {
            anyhow!(
                "failed to read portable NCA corpus config {}: {error}",
                nca_config_path.display()
            )
        })?),
        _ => None,
    };
    Ok(DragonExperimentProfile {
        version: DRAGON_PROFILE_VERSION,
        experiment_kind,
        native: DragonNativeExperimentProfile {
            training_toml: portable_training_template(config, nca_corpus_toml.as_deref())?,
            nca_corpus_toml,
        },
        browser: browser_profile_from_native_config(
            config,
            experiment_kind,
            &model_config,
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

    if let Some(corpus_toml) = profile.native.nca_corpus_toml.as_ref() {
        let mut corpus = toml::from_str::<NcaCorpusConfig>(corpus_toml).map_err(|error| {
            anyhow!("failed to decode portable NCA corpus config for {experiment_id}: {error}")
        })?;
        corpus.output_dir = profile_root.join("nca-generated");
        let corpus_path = profile_root.join(PORTABLE_NCA_CORPUS_FILE_NAME);
        fs::write(&corpus_path, toml::to_string(&corpus)?)?;
        config.dataset.source = DatasetSourceConfig::UniversalityNca {
            config: corpus_path.clone(),
        };
        if let Some(validation) = config.dataset.validation.as_mut()
            && matches!(
                validation.source,
                DatasetSourceConfig::UniversalityNca { .. }
            )
        {
            validation.source = DatasetSourceConfig::UniversalityNca {
                config: corpus_path,
            };
        }
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
) -> DragonManifestSeed {
    let mut seed = default_seed.clone();
    seed.network_id = entry.network_id.as_str().to_owned();
    seed.study_id = entry.study_id.as_str().to_owned();
    seed.experiment_id = entry.experiment_id.as_str().to_owned();
    seed.revision_id = entry.current_revision_id.as_str().to_owned();
    seed.display_name = entry.display_name.clone();
    seed
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
                        manifest_seed: manifest_seed_from_entry(&native.manifest, &entry),
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
    }
}

#[cfg(any(feature = "wasm-peer", feature = "native"))]
pub fn browser_training_config_from_profile(
    entry: &ExperimentDirectoryEntry,
    profile: &DragonExperimentProfile,
) -> Result<Option<DragonBrowserTrainingConfig>> {
    if !entry.browser_role_allowed(BrowserRole::TrainerWgpu) {
        return Ok(None);
    }
    let Some(browser) = profile.browser.clone() else {
        return Ok(None);
    };
    Ok(Some(DragonBrowserTrainingConfig {
        experiment_kind: profile.experiment_kind,
        model_config: browser.model_config,
        training_objective: DragonBrowserTrainingObjectiveConfig::default(),
        execution_backend: browser.execution_backend,
        block_size: browser.block_size,
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
        }),
    }))
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
    fn native_training_overrides_bound_runtime_without_changing_model_profile() {
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
        .expect("builtin fallback should resolve");

        assert_eq!(resolved.config.training.batch_size, 1);
        assert_eq!(resolved.config.training.max_iters, 4);
        assert_eq!(resolved.config.training.checkpoint_interval_iters, 4);
        assert_eq!(resolved.config.model.n_layer, Some(8));
        assert_eq!(resolved.config.model.n_embd, Some(512));
        assert_eq!(resolved.config.model.latent_total, Some(1024));
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
