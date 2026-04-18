use std::collections::BTreeSet;
#[cfg(feature = "wgpu")]
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Context;
use anyhow::{Result, anyhow, bail};
use burn::backend::NdArray;
use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
use burn_autodiff::Autodiff;
use burn_dragon_core::DragonModel;
use burn_dragon_universality::{OnlineNcaCorpus, SampleSplit};
use burn_p2p::{
    ContentId, ExperimentId, ExperimentScope, RevisionId, StudyId, WorkloadId,
    WorkloadTrainingLease,
};
use burn_p2p_browser::{
    BrowserCapabilityReport, BrowserRuntimeRole, BrowserSessionRuntimeConfig,
    BrowserSessionRuntimeError, BrowserSessionRuntimeHandle, BrowserSessionState,
    BrowserTrainingBudget, BrowserTrainingPlan,
};
use burn_p2p_core::codec::multihash_sha256;
use burn_p2p_dataloader::ShardFetchManifest;
#[cfg(feature = "wgpu")]
use burn_wgpu::{RuntimeOptions, graphics};
use gloo_net::http::Request;
use serde::{Deserialize, Serialize};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
use url::Url;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use crate::auth::{browser_github_enrollment_config, fetch_edge_snapshot, load_browser_session};
use crate::capability::{decide_browser_capability, detect_browser_host_capabilities};
#[cfg(target_arch = "wasm32")]
use crate::capability_state::{
    apply_browser_downgrade_state, clear_browser_downgrade, persist_browser_downgrade,
};
use crate::config::{
    DragonBrowserDatasetSplit, DragonBrowserExecutionBackend, DragonBrowserShardSelectionPolicy,
    DragonBrowserTokenSource, DragonBrowserTrainingConfig, TokenWindowRecord,
};

type BrowserCpuEvalBackend = NdArray<f32>;
type BrowserCpuTrainBackend = Autodiff<BrowserCpuEvalBackend>;

#[cfg(feature = "wgpu")]
type BrowserWgpuEvalBackend = burn_wgpu::Wgpu<f32>;
#[cfg(feature = "wgpu")]
type BrowserWgpuTrainBackend = Autodiff<BrowserWgpuEvalBackend>;
#[cfg(feature = "wgpu")]
type BrowserWgpuTrainDevice = <BrowserWgpuTrainBackend as Backend>::Device;

#[cfg(feature = "wgpu")]
static WEBGPU_RUNTIME_READY: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BrowserTrainingBackendKind {
    Cpu,
    #[cfg(feature = "wgpu")]
    Wgpu,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DragonBrowserTrainingResult {
    pub backend: String,
    pub experiment_kind_label: String,
    pub train_batches: usize,
    pub train_examples: usize,
    pub train_tokens: usize,
    pub train_loss_mean: f64,
    pub eval_examples: usize,
    pub eval_loss: Option<f64>,
    pub setup_time_ms: u64,
    pub training_time_ms: u64,
    pub eval_time_ms: u64,
    pub total_time_ms: u64,
    pub tokens_per_second: Option<f64>,
    #[serde(default)]
    pub live_participant: Option<DragonBrowserLiveParticipantResult>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DragonBrowserLiveParticipantResult {
    pub receipt_submission_accepted: bool,
    pub accepted_receipt_ids: Vec<String>,
    pub emitted_receipt_id: Option<String>,
    pub runtime_state: Option<String>,
    pub transport: Option<String>,
}

#[derive(Clone, Debug)]
struct TokenWindowBatch<B: Backend> {
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    token_count: usize,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(untagged)]
enum TokenWindowPayload {
    Records(Vec<TokenWindowRecord>),
    Wrapped {
        records: Vec<TokenWindowRecord>,
    },
    #[default]
    Empty,
}

impl TokenWindowPayload {
    fn into_records(self) -> Vec<TokenWindowRecord> {
        match self {
            Self::Records(records) => records,
            Self::Wrapped { records } => records,
            Self::Empty => Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
struct TokenRecordLoadPolicy {
    record_limit: Option<usize>,
    shard_selection_key: Option<String>,
    training_lease: Option<WorkloadTrainingLease>,
}

struct LiveBrowserParticipantHandle {
    session_runtime: BrowserSessionRuntimeHandle,
    training_budget: BrowserTrainingBudget,
}

#[derive(Clone)]
struct BrowserTrainingRunContext<'a> {
    edge_base_url: &'a str,
    config: &'a DragonBrowserTrainingConfig,
    release_manifest: &'a burn_p2p::ClientReleaseManifest,
    backend_label: &'a str,
    backend_kind: BrowserTrainingBackendKind,
    setup_time_ms: u64,
    live_browser_session: Option<BrowserSessionState>,
}

impl<'a> BrowserTrainingRunContext<'a> {
    fn live_session_principal_id(&self) -> Option<&str> {
        live_session_principal_id(self.live_browser_session.as_ref())
    }

    fn token_record_load_policy(
        &self,
        stage: &str,
        record_limit: Option<usize>,
        training_lease: Option<WorkloadTrainingLease>,
    ) -> TokenRecordLoadPolicy {
        TokenRecordLoadPolicy {
            record_limit,
            shard_selection_key: Some(browser_shard_selection_key(
                self.edge_base_url,
                self.config,
                self.live_session_principal_id(),
                stage,
            )),
            training_lease,
        }
    }
}

struct ShardManifestLoadRequest<'a> {
    manifest_url: &'a str,
    edge_base_url: &'a str,
    block_size: usize,
    record_limit: Option<usize>,
    selection: DragonBrowserShardSelectionPolicy,
    max_shards_per_window: Option<usize>,
    selection_key: Option<&'a str>,
    training_lease: Option<&'a WorkloadTrainingLease>,
}

pub async fn run_browser_training_with_release_manifest(
    edge_base_url: &str,
    config: &DragonBrowserTrainingConfig,
    release_manifest: &burn_p2p::ClientReleaseManifest,
) -> Result<DragonBrowserTrainingResult> {
    let backend_kind = resolve_browser_training_backend(config)?;
    let backend_label = match backend_kind {
        BrowserTrainingBackendKind::Cpu => "cpu",
        #[cfg(feature = "wgpu")]
        BrowserTrainingBackendKind::Wgpu => "wgpu",
    };
    let browser_training_requires_webgpu = match backend_kind {
        BrowserTrainingBackendKind::Cpu => false,
        #[cfg(feature = "wgpu")]
        BrowserTrainingBackendKind::Wgpu => true,
    };
    let browser_capability_decision = apply_browser_downgrade_state(
        edge_base_url,
        config,
        backend_label,
        decide_browser_capability(Some(config), &detect_browser_host_capabilities()),
    );
    if browser_training_requires_webgpu && !browser_capability_decision.can_train {
        bail!(
            "{}",
            browser_capability_decision
                .downgrade_reason
                .unwrap_or_else(
                    || "browser trainer capability assessment rejected local training".into()
                )
        );
    }
    let live_browser_session = if config.live_participant.is_some() {
        Some(load_browser_session(edge_base_url).await?)
    } else {
        None
    };
    let result = match backend_kind {
        BrowserTrainingBackendKind::Cpu => {
            let train_device = <BrowserCpuTrainBackend as Backend>::Device::default();
            let eval_device = <BrowserCpuEvalBackend as Backend>::Device::default();
            let setup_started_at = Instant::now();
            BrowserCpuEvalBackend::seed(&eval_device, 1337);
            let setup_time_ms = elapsed_ms(setup_started_at);
            run_browser_training_inner::<BrowserCpuTrainBackend, BrowserCpuEvalBackend>(
                BrowserTrainingRunContext {
                    edge_base_url,
                    config,
                    release_manifest,
                    backend_label: "burn-ndarray-wasm",
                    backend_kind,
                    setup_time_ms,
                    live_browser_session: live_browser_session.clone(),
                },
                &train_device,
                &eval_device,
            )
            .await
        }
        #[cfg(feature = "wgpu")]
        BrowserTrainingBackendKind::Wgpu => {
            let train_device = BrowserWgpuTrainDevice::default();
            let eval_device = <BrowserWgpuEvalBackend as Backend>::Device::default();
            let setup_started_at = Instant::now();
            ensure_webgpu_runtime_ready(&train_device).await;
            BrowserWgpuEvalBackend::seed(&eval_device, 1337);
            let setup_time_ms = elapsed_ms(setup_started_at);
            run_browser_training_inner::<BrowserWgpuTrainBackend, BrowserWgpuEvalBackend>(
                BrowserTrainingRunContext {
                    edge_base_url,
                    config,
                    release_manifest,
                    backend_label: "burn-webgpu-wasm",
                    backend_kind,
                    setup_time_ms,
                    live_browser_session,
                },
                &train_device,
                &eval_device,
            )
            .await
        }
    };

    #[cfg(target_arch = "wasm32")]
    match &result {
        Ok(_) if browser_training_requires_webgpu => {
            let _ = clear_browser_downgrade(edge_base_url, config, backend_label);
        }
        Err(error) if browser_training_requires_webgpu => {
            let _ = persist_browser_downgrade(
                edge_base_url,
                config,
                backend_label,
                &browser_capability_decision,
                &error.to_string(),
                "runtime",
            );
        }
        _ => {}
    }

    result
}

#[cfg(feature = "wgpu")]
async fn ensure_webgpu_runtime_ready(device: &BrowserWgpuTrainDevice) {
    if !WEBGPU_RUNTIME_READY.swap(true, Ordering::SeqCst) {
        burn_wgpu::init_setup_async::<graphics::WebGpu>(device, RuntimeOptions::default()).await;
    }
}

fn resolve_browser_training_backend(
    config: &DragonBrowserTrainingConfig,
) -> Result<BrowserTrainingBackendKind> {
    match config.execution_backend {
        DragonBrowserExecutionBackend::Auto => {
            #[cfg(feature = "wgpu")]
            {
                Ok(BrowserTrainingBackendKind::Wgpu)
            }
            #[cfg(not(feature = "wgpu"))]
            {
                Ok(BrowserTrainingBackendKind::Cpu)
            }
        }
        DragonBrowserExecutionBackend::Cpu => Ok(BrowserTrainingBackendKind::Cpu),
        DragonBrowserExecutionBackend::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                Ok(BrowserTrainingBackendKind::Wgpu)
            }
            #[cfg(not(feature = "wgpu"))]
            {
                bail!(
                    "browser training requested webgpu backend but the `wgpu` feature is disabled"
                )
            }
        }
    }
}

async fn run_browser_training_inner<TrainB, EvalB>(
    context: BrowserTrainingRunContext<'_>,
    train_device: &TrainB::Device,
    eval_device: &EvalB::Device,
) -> Result<DragonBrowserTrainingResult>
where
    TrainB: AutodiffBackend<InnerBackend = EvalB> + Clone,
    EvalB: Backend + Clone,
{
    validate_browser_training_config(context.config)?;
    validate_live_training_backend(context.config, context.backend_kind)?;

    let total_started_at = Instant::now();

    let train_records = load_token_records(
        context.edge_base_url,
        &context.config.train_source,
        context.config.block_size,
        context.token_record_load_policy(
            "train",
            max_record_limit(context.config.batch_size, context.config.max_train_batches),
            context.config.training_lease.clone(),
        ),
    )
    .await?;
    if train_records.is_empty() {
        bail!("browser training source produced no train records");
    }
    let eval_records = match &context.config.eval_source {
        Some(source) => {
            load_token_records(
                context.edge_base_url,
                source,
                context.config.block_size,
                context.token_record_load_policy(
                    "eval",
                    max_record_limit(context.config.batch_size, context.config.max_eval_batches),
                    None,
                ),
            )
            .await?
        }
        None => Vec::new(),
    };

    let train_batches = build_batches::<TrainB>(
        &train_records,
        context.config.batch_size,
        context.config.block_size,
        train_device,
    )?;
    let eval_batches = build_batches::<EvalB>(
        &eval_records,
        context.config.batch_size,
        context.config.block_size,
        eval_device,
    )?;

    let mut live_participant = start_live_browser_participant(
        context.edge_base_url,
        context.config,
        context.release_manifest,
        context.live_browser_session.as_ref(),
    )
    .await?;

    let training_started_at = Instant::now();
    let mut model = DragonModel::<TrainB>::new(context.config.model_config.clone(), train_device);
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(context.config.weight_decay)
        .init();
    let mut train_loss_sum = 0.0;
    let mut train_batch_count = 0usize;
    let mut train_token_count = 0usize;
    for (batch_index, batch) in train_batches.into_iter().enumerate() {
        if context
            .config
            .max_train_batches
            .is_some_and(|max_batches| batch_index >= max_batches)
        {
            break;
        }
        let hidden = model.forward_hidden(batch.inputs);
        let loss = model.language_loss_from_hidden(hidden, batch.targets);
        train_loss_sum += scalar_from_loss_async(loss.clone()).await?;
        train_token_count = train_token_count.saturating_add(batch.token_count);
        train_batch_count = train_batch_count.saturating_add(1);
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(context.config.learning_rate, model, grads);
    }
    let training_time_ms = elapsed_ms(training_started_at);
    let train_batch_count = train_batch_count.max(1);
    let train_loss_mean = train_loss_sum / train_batch_count as f64;

    let eval_started_at = Instant::now();
    let eval_loss = if eval_batches.is_empty() {
        None
    } else {
        let eval_model = model.valid();
        let mut total = 0.0;
        let mut count = 0usize;
        for (batch_index, batch) in eval_batches.into_iter().enumerate() {
            if context
                .config
                .max_eval_batches
                .is_some_and(|max_batches| batch_index >= max_batches)
            {
                break;
            }
            let hidden = eval_model.forward_hidden(batch.inputs);
            let loss = eval_model.language_loss_from_hidden(hidden, batch.targets);
            total += scalar_from_loss_async(loss).await?;
            count = count.saturating_add(1);
        }
        (count > 0).then_some(total / count as f64)
    };
    let eval_time_ms = elapsed_ms(eval_started_at);

    let live_participant = finish_live_browser_participant(
        context.edge_base_url,
        context.config,
        live_participant.as_mut(),
    )
    .await?;

    Ok(DragonBrowserTrainingResult {
        backend: context.backend_label.into(),
        experiment_kind_label: context.config.experiment_kind.display_name().into(),
        train_batches: train_batch_count,
        train_examples: train_records.len(),
        train_tokens: train_token_count,
        train_loss_mean,
        eval_examples: eval_records.len(),
        eval_loss,
        setup_time_ms: context.setup_time_ms,
        training_time_ms,
        eval_time_ms,
        total_time_ms: context.setup_time_ms + elapsed_ms(total_started_at),
        tokens_per_second: (training_time_ms > 0)
            .then_some(train_token_count as f64 / (training_time_ms as f64 / 1000.0)),
        live_participant,
    })
}

fn validate_live_training_backend(
    config: &DragonBrowserTrainingConfig,
    backend_kind: BrowserTrainingBackendKind,
) -> Result<()> {
    if config.live_participant.is_some() && !backend_supports_live_participant(backend_kind) {
        bail!("browser live training requires the webgpu backend");
    }
    Ok(())
}

fn backend_supports_live_participant(backend_kind: BrowserTrainingBackendKind) -> bool {
    match backend_kind {
        BrowserTrainingBackendKind::Cpu => false,
        #[cfg(feature = "wgpu")]
        BrowserTrainingBackendKind::Wgpu => true,
    }
}

fn validate_browser_training_config(config: &DragonBrowserTrainingConfig) -> Result<()> {
    if config.block_size == 0 {
        bail!("browser training block_size must be > 0");
    }
    if config.batch_size == 0 {
        bail!("browser training batch_size must be > 0");
    }
    if config.model_config.vocab_size == 0 {
        bail!("browser training model_config.vocab_size must be > 0");
    }
    Ok(())
}

fn max_record_limit(batch_size: usize, max_batches: Option<usize>) -> Option<usize> {
    max_batches.and_then(|max_batches| max_batches.checked_mul(batch_size.max(1)))
}

fn live_session_principal_id(session_state: Option<&BrowserSessionState>) -> Option<&str> {
    session_state
        .and_then(|session_state| session_state.session.as_ref())
        .map(|session| session.claims.principal_id.as_str())
}

fn browser_shard_selection_key(
    edge_base_url: &str,
    config: &DragonBrowserTrainingConfig,
    session_principal_id: Option<&str>,
    stage: &str,
) -> String {
    if let Some(live) = config.live_participant.as_ref() {
        let participant_id = session_principal_id
            .or(live.principal_id.as_deref())
            .unwrap_or("browser-live-session");
        return format!(
            "live|{}|{}|{}|{}|{}|{}",
            edge_base_url.trim_end_matches('/'),
            participant_id,
            live.study_id,
            live.experiment_id,
            live.revision_id,
            stage,
        );
    }

    format!(
        "local|{}|{}|{}|{}|{}|{}",
        edge_base_url.trim_end_matches('/'),
        config.experiment_kind.workload_slug(),
        config.block_size,
        config.batch_size,
        config.max_train_batches.unwrap_or(0),
        stage,
    )
}

async fn load_token_records(
    edge_base_url: &str,
    source: &DragonBrowserTokenSource,
    block_size: usize,
    policy: TokenRecordLoadPolicy,
) -> Result<Vec<TokenWindowRecord>> {
    let records = match source {
        DragonBrowserTokenSource::Inline { records } => records.clone(),
        DragonBrowserTokenSource::HttpJson { url } => {
            let resolved_url = resolve_browser_source_url(url, edge_base_url)?;
            let response = Request::get(&resolved_url).send().await.map_err(|error| {
                anyhow!("failed to fetch browser shard {resolved_url}: {error}")
            })?;
            let payload = response
                .json::<TokenWindowPayload>()
                .await
                .map_err(|error| {
                    anyhow!("failed to decode browser shard {resolved_url}: {error}")
                })?;
            payload.into_records()
        }
        DragonBrowserTokenSource::ShardManifestHttp {
            manifest_url,
            selection,
            max_shards_per_window,
        } => {
            load_shard_manifest_records(ShardManifestLoadRequest {
                manifest_url,
                edge_base_url,
                block_size,
                record_limit: policy.record_limit,
                selection: *selection,
                max_shards_per_window: *max_shards_per_window,
                selection_key: policy.shard_selection_key.as_deref(),
                training_lease: policy.training_lease.as_ref(),
            })
            .await?
        }
        DragonBrowserTokenSource::GeneratedNca {
            corpus,
            split,
            max_documents,
        } => load_generated_nca_records(corpus, split.clone(), *max_documents, block_size)?,
    };
    validate_token_records(&records, block_size)?;
    Ok(records)
}

fn resolve_browser_source_url(url_or_path: &str, edge_base_url: &str) -> Result<String> {
    if url_or_path.starts_with("data:")
        || url_or_path.starts_with("blob:")
        || Url::parse(url_or_path).is_ok()
    {
        return Ok(url_or_path.to_owned());
    }
    let base = Url::parse(edge_base_url)
        .with_context(|| format!("invalid browser edge base URL {edge_base_url}"))?;
    Ok(base
        .join(url_or_path)
        .with_context(|| format!("failed to resolve browser source {url_or_path}"))?
        .into())
}

fn resolve_shard_entry_url(manifest_url: &str, locator: &str) -> Result<String> {
    if locator.starts_with("data:") || locator.starts_with("blob:") || Url::parse(locator).is_ok() {
        return Ok(locator.to_owned());
    }
    let manifest = Url::parse(manifest_url).with_context(|| {
        format!("shard manifest URL must be absolute when locators are relative: {manifest_url}")
    })?;
    Ok(manifest
        .join(locator)
        .with_context(|| format!("failed to resolve shard locator {locator} from {manifest_url}"))?
        .into())
}

fn verify_shard_entry_bytes(
    manifest_url: &str,
    entry: &burn_p2p_dataloader::ShardFetchEntry,
    bytes: &[u8],
) -> Result<()> {
    if entry.bytes_len != bytes.len() as u64 {
        bail!(
            "browser shard {} from {} had {} bytes, expected {}",
            entry.locator,
            manifest_url,
            bytes.len(),
            entry.bytes_len
        );
    }
    let actual = ContentId::from_multihash(multihash_sha256(bytes));
    if actual != entry.content_hash {
        bail!(
            "browser shard {} from {} failed content hash verification",
            entry.locator,
            manifest_url
        );
    }
    Ok(())
}

fn shard_selection_rank(selection_key: &str, entry: &burn_p2p_dataloader::ShardFetchEntry) -> u64 {
    let material = format!(
        "{selection_key}\0{}\0{}",
        entry.microshard_id.as_str(),
        entry.ordinal
    );
    let digest = multihash_sha256(material.as_bytes());
    let bytes = digest.get(2..10).unwrap_or(&digest[..digest.len().min(8)]);
    let mut rank = [0_u8; 8];
    for (index, byte) in bytes.iter().enumerate() {
        rank[index] = *byte;
    }
    u64::from_be_bytes(rank)
}

fn ordered_manifest_entries<'a>(
    manifest: &'a ShardFetchManifest,
    selection: DragonBrowserShardSelectionPolicy,
    selection_key: Option<&str>,
) -> Vec<&'a burn_p2p_dataloader::ShardFetchEntry> {
    let mut entries = manifest.entries.iter().collect::<Vec<_>>();
    match selection {
        DragonBrowserShardSelectionPolicy::Sequential => {
            entries.sort_by_key(|entry| (entry.ordinal, entry.microshard_id.as_str()))
        }
        DragonBrowserShardSelectionPolicy::DeterministicPeer => {
            let selection_key = selection_key.unwrap_or(manifest.dataset_view_id.as_str());
            entries.sort_by_key(|entry| {
                (
                    shard_selection_rank(selection_key, entry),
                    entry.ordinal,
                    entry.microshard_id.as_str(),
                )
            });
        }
    }
    entries
}

async fn load_shard_manifest_records(
    request: ShardManifestLoadRequest<'_>,
) -> Result<Vec<TokenWindowRecord>> {
    let manifest_url = resolve_browser_source_url(request.manifest_url, request.edge_base_url)?;
    let response = Request::get(&manifest_url)
        .send()
        .await
        .map_err(|error| anyhow!("failed to fetch shard manifest {manifest_url}: {error}"))?;
    let manifest_bytes = response
        .binary()
        .await
        .map_err(|error| anyhow!("failed to read shard manifest {manifest_url}: {error}"))?;
    let manifest: ShardFetchManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| anyhow!("failed to decode shard manifest {manifest_url}: {error}"))?;

    let leased_microshard_ids = request.training_lease.map(|lease| {
        lease
            .microshards
            .iter()
            .map(|microshard_id| microshard_id.as_str().to_owned())
            .collect::<std::collections::BTreeSet<_>>()
    });

    let mut records = Vec::new();
    let filtered_entries = manifest
        .entries
        .iter()
        .filter(|entry| {
            leased_microshard_ids
                .as_ref()
                .is_none_or(|ids| ids.contains(entry.microshard_id.as_str()))
        })
        .cloned()
        .collect::<Vec<_>>();
    if let Some(ids) = leased_microshard_ids.as_ref()
        && !ids.is_empty()
        && filtered_entries.is_empty()
    {
        bail!(
            "browser shard manifest {manifest_url} did not contain any leased microshards from the active assignment"
        );
    }
    let filtered_manifest = ShardFetchManifest {
        dataset_view_id: manifest.dataset_view_id.clone(),
        entries: filtered_entries,
    };
    let ordered_entries =
        ordered_manifest_entries(&filtered_manifest, request.selection, request.selection_key);
    let shard_limit = request.max_shards_per_window.unwrap_or(usize::MAX);
    for entry in ordered_entries.into_iter().take(shard_limit) {
        let shard_url = resolve_shard_entry_url(&manifest_url, &entry.locator)?;
        let response = Request::get(&shard_url)
            .send()
            .await
            .map_err(|error| anyhow!("failed to fetch browser shard {shard_url}: {error}"))?;
        let shard_bytes = response
            .binary()
            .await
            .map_err(|error| anyhow!("failed to read browser shard {shard_url}: {error}"))?;
        verify_shard_entry_bytes(&manifest_url, entry, &shard_bytes)?;
        let mut shard_records = serde_json::from_slice::<Vec<TokenWindowRecord>>(&shard_bytes)
            .map_err(|error| anyhow!("failed to decode browser shard {shard_url}: {error}"))?;
        records.append(&mut shard_records);
        if let Some(limit) = request.record_limit
            && records.len() >= limit
        {
            records.truncate(limit);
            break;
        }
    }

    validate_token_records(&records, request.block_size)?;
    Ok(records)
}

fn load_generated_nca_records(
    corpus: &burn_dragon_universality::NcaCorpusConfig,
    split: DragonBrowserDatasetSplit,
    max_documents: Option<usize>,
    block_size: usize,
) -> Result<Vec<TokenWindowRecord>> {
    let logical_document_tokens = block_size.saturating_add(1);
    let runtime = OnlineNcaCorpus::new_with_min_logical_document_tokens(
        corpus.clone(),
        Some(logical_document_tokens),
    )?;
    let split = match split {
        DragonBrowserDatasetSplit::Train => SampleSplit::Train,
        DragonBrowserDatasetSplit::Validation => SampleSplit::Validation,
    };
    let document_count = runtime
        .sample_count(split)
        .min(max_documents.unwrap_or(usize::MAX));
    let mut records = Vec::new();
    for sample_index in 0..document_count {
        let tokens = runtime.generate_document_tokens(split, sample_index)?;
        records.extend(token_windows_from_tokens(&tokens, block_size));
    }
    Ok(records)
}

fn token_windows_from_tokens(tokens: &[u32], block_size: usize) -> Vec<TokenWindowRecord> {
    if tokens.len() <= block_size {
        return Vec::new();
    }
    let max_start = tokens.len() - (block_size + 1);
    let mut records = Vec::new();
    let mut start = 0usize;
    loop {
        let window = &tokens[start..start + block_size + 1];
        records.push(TokenWindowRecord {
            inputs: window[..block_size]
                .iter()
                .map(|token| i64::from(*token))
                .collect(),
            targets: window[1..].iter().map(|token| i64::from(*token)).collect(),
            reset_stream_state: start == 0,
        });
        if start >= max_start {
            break;
        }
        start = start.saturating_add(block_size).min(max_start);
    }
    records
}

fn validate_token_records(records: &[TokenWindowRecord], block_size: usize) -> Result<()> {
    for (index, record) in records.iter().enumerate() {
        if record.inputs.len() != block_size {
            bail!(
                "token window record {index} inputs length {} does not match block_size {}",
                record.inputs.len(),
                block_size
            );
        }
        if record.targets.len() != block_size {
            bail!(
                "token window record {index} targets length {} does not match block_size {}",
                record.targets.len(),
                block_size
            );
        }
    }
    Ok(())
}

fn build_batches<B: Backend>(
    records: &[TokenWindowRecord],
    batch_size: usize,
    block_size: usize,
    device: &B::Device,
) -> Result<Vec<TokenWindowBatch<B>>> {
    if records.is_empty() {
        return Ok(Vec::new());
    }
    let mut batches = Vec::new();
    for chunk in records.chunks(batch_size.max(1)) {
        let mut inputs = Vec::with_capacity(chunk.len() * block_size);
        let mut targets = Vec::with_capacity(chunk.len() * block_size);
        for record in chunk {
            inputs.extend(record.inputs.iter().copied());
            targets.extend(record.targets.iter().copied());
        }
        batches.push(TokenWindowBatch {
            inputs: Tensor::<B, 2, Int>::from_data(
                TensorData::new(inputs, [chunk.len(), block_size]),
                device,
            ),
            targets: Tensor::<B, 2, Int>::from_data(
                TensorData::new(targets, [chunk.len(), block_size]),
                device,
            ),
            token_count: chunk.len() * block_size,
        });
    }
    Ok(batches)
}

async fn scalar_from_loss_async<B: Backend>(loss: Tensor<B, 1>) -> Result<f64> {
    loss.into_scalar_async()
        .await
        .map(|scalar| scalar.elem::<f64>())
        .map_err(|error| anyhow!("failed to read browser loss scalar: {error}"))
}

async fn start_live_browser_participant(
    edge_base_url: &str,
    config: &DragonBrowserTrainingConfig,
    release_manifest: &burn_p2p::ClientReleaseManifest,
    preloaded_session: Option<&BrowserSessionState>,
) -> Result<Option<LiveBrowserParticipantHandle>> {
    let Some(live) = config.live_participant.as_ref() else {
        return Ok(None);
    };
    let snapshot = fetch_edge_snapshot(edge_base_url).await?;
    let requested_scopes = BTreeSet::from([
        ExperimentScope::Connect,
        ExperimentScope::Train {
            experiment_id: ExperimentId::new(live.experiment_id.clone()),
        },
    ]);
    let _ = browser_github_enrollment_config(&snapshot, release_manifest, requested_scopes, 900)?;
    let session = match preloaded_session {
        Some(session) => session.clone(),
        None => load_browser_session(edge_base_url).await?,
    };
    let _claims = session
        .session
        .as_ref()
        .ok_or_else(|| anyhow!("browser live training requires an authenticated session"))?;

    let capability_decision = apply_browser_downgrade_state(
        edge_base_url,
        config,
        config.execution_backend.backend_label(),
        decide_browser_capability(Some(config), &detect_browser_host_capabilities()),
    );
    let capability = BrowserCapabilityReport {
        ..capability_decision.capability
    };
    if capability.recommended_role != BrowserRuntimeRole::BrowserTrainerWgpu {
        bail!(
            "browser live training capability downgraded to {}; reconnect as verifier instead of trainer",
            match capability.recommended_role {
                BrowserRuntimeRole::BrowserVerifier => "browser_verifier",
                BrowserRuntimeRole::BrowserObserver => "browser_observer",
                BrowserRuntimeRole::BrowserFallback => "browser_fallback",
                BrowserRuntimeRole::Viewer => "viewer",
                BrowserRuntimeRole::BrowserTrainerWgpu => "browser_trainer_wgpu",
            }
        );
    }
    let session_runtime = BrowserSessionRuntimeHandle::start(
        &snapshot,
        BrowserSessionRuntimeConfig {
            edge_base_url: edge_base_url.to_owned(),
            release_train_hash: release_manifest.release_train_hash.clone(),
            target_artifact_id: release_manifest.target_artifact_id.clone(),
            target_artifact_hash: release_manifest.target_artifact_hash.clone(),
            role: BrowserRuntimeRole::BrowserTrainerWgpu,
            transport: burn_p2p_browser::BrowserTransportPolicy::from(
                burn_p2p::RuntimeTransportPolicy::browser_for_roles(&burn_p2p::PeerRoleSet::new([
                    burn_p2p::PeerRole::BrowserTrainerWgpu,
                ])),
            ),
            selected_experiment: Some(ExperimentId::new(live.experiment_id.clone())),
            selected_revision: Some(RevisionId::new(live.revision_id.clone())),
            capability,
            include_leaderboard: true,
        },
        session,
    )
    .await
    .map_err(map_browser_session_runtime_error)?;

    Ok(Some(LiveBrowserParticipantHandle {
        session_runtime,
        training_budget: capability_decision.training_budget.unwrap_or_else(|| {
            BrowserTrainingBudget {
                max_window_secs: 30,
                requires_webgpu: true,
                max_batch_size: Some(config.batch_size as u32),
                ..BrowserTrainingBudget::default()
            }
        }),
    }))
}

async fn finish_live_browser_participant(
    edge_base_url: &str,
    config: &DragonBrowserTrainingConfig,
    handle: Option<&mut LiveBrowserParticipantHandle>,
) -> Result<Option<DragonBrowserLiveParticipantResult>> {
    let Some(handle) = handle else {
        return Ok(None);
    };
    let assignment = handle
        .session_runtime
        .runtime
        .storage
        .active_assignment
        .clone()
        .ok_or_else(|| anyhow!("browser runtime has no active assignment for live training"))?;
    let outcome = handle
        .session_runtime
        .run_training_plan(BrowserTrainingPlan {
            study_id: StudyId::new(assignment.study_id.as_str().to_owned()),
            experiment_id: ExperimentId::new(assignment.experiment_id.as_str().to_owned()),
            revision_id: RevisionId::new(assignment.revision_id.as_str().to_owned()),
            workload_id: WorkloadId::new("browser-dragon-training"),
            budget: handle.training_budget.clone(),
            lease: config.training_lease.clone(),
        })
        .await
        .map_err(|error| match error {
            BrowserSessionRuntimeError::Worker(message) => {
                let capability_decision = apply_browser_downgrade_state(
                    edge_base_url,
                    config,
                    config.execution_backend.backend_label(),
                    decide_browser_capability(Some(config), &detect_browser_host_capabilities()),
                );
                let _ = persist_browser_downgrade(
                    edge_base_url,
                    config,
                    config.execution_backend.backend_label(),
                    &capability_decision,
                    &message,
                    "browser-worker-runtime",
                );
                anyhow!("browser worker training failed: {message}")
            }
            other => anyhow!(other),
        })?;
    Ok(Some(DragonBrowserLiveParticipantResult {
        receipt_submission_accepted: outcome.receipt_submission_accepted,
        accepted_receipt_ids: outcome.accepted_receipt_ids,
        emitted_receipt_id: outcome.emitted_receipt_id,
        runtime_state: outcome.runtime_state.as_ref().map(|state| state.label()),
        transport: outcome
            .transport
            .as_ref()
            .map(|kind| kind.label().to_owned()),
    }))
}

fn map_browser_session_runtime_error(error: BrowserSessionRuntimeError) -> anyhow::Error {
    match error {
        BrowserSessionRuntimeError::MissingSession => {
            anyhow!("browser live training requires an authenticated session")
        }
        BrowserSessionRuntimeError::Client(error) => {
            anyhow!("failed to synchronize browser runtime before training: {error}")
        }
        BrowserSessionRuntimeError::Worker(message) => {
            anyhow!("browser worker runtime failed during bootstrap: {message}")
        }
    }
}

fn elapsed_ms(started_at: Instant) -> u64 {
    started_at.elapsed().as_millis() as u64
}

#[cfg(all(test, target_arch = "wasm32", feature = "wasm-peer"))]
mod tests {
    use super::*;
    use crate::config::DragonBrowserLiveParticipantConfig;
    use burn_dragon_core::{DragonConfig, LanguageHeadConfig};
    use burn_dragon_universality::{
        NcaCorpusConfig, NcaFamilyConfig, NcaFamilyKind, NcaSerializationConfig,
        NcaTokenizationConfig, UsizeRangeConfig,
    };
    use burn_p2p::{
        ClientPlatform, ClientReleaseManifest, ContentId, DatasetViewId, MicroShardId,
        ProjectFamilyId,
    };
    use burn_p2p_dataloader::{ShardFetchEntry, ShardFetchManifest};
    use js_sys::encode_uri_component;
    use serde_json::json;
    use std::path::PathBuf;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn browser_live_shard_selection_prefers_authenticated_session_principal() {
        let mut config = sample_browser_training_config();
        config.live_participant = Some(DragonBrowserLiveParticipantConfig {
            principal_id: Some("configured-live-principal".into()),
            study_id: "dragon-study".into(),
            experiment_id: "dragon-experiment".into(),
            revision_id: "dragon-revision".into(),
            workload_id: "dragon-workload".into(),
        });

        let shard_key = browser_shard_selection_key(
            "https://edge.example.invalid",
            &config,
            Some("session-principal"),
            "train",
        );

        assert!(shard_key.contains("session-principal"));
        assert!(!shard_key.contains("configured-live-principal"));
    }

    #[wasm_bindgen_test]
    fn browser_live_shard_selection_falls_back_to_config_then_default() {
        let mut config = sample_browser_training_config();
        config.live_participant = Some(DragonBrowserLiveParticipantConfig {
            principal_id: Some("configured-live-principal".into()),
            study_id: "dragon-study".into(),
            experiment_id: "dragon-experiment".into(),
            revision_id: "dragon-revision".into(),
            workload_id: "dragon-workload".into(),
        });

        let configured_key =
            browser_shard_selection_key("https://edge.example.invalid", &config, None, "train");
        assert!(configured_key.contains("configured-live-principal"));

        config
            .live_participant
            .as_mut()
            .expect("live participant")
            .principal_id = None;
        let default_key =
            browser_shard_selection_key("https://edge.example.invalid", &config, None, "train");
        assert!(default_key.contains("browser-live-session"));
    }

    #[wasm_bindgen_test(async)]
    async fn browser_training_smoke_generated_nca() {
        #[cfg(feature = "wgpu")]
        let execution_backend = DragonBrowserExecutionBackend::Wgpu;
        #[cfg(not(feature = "wgpu"))]
        let execution_backend = DragonBrowserExecutionBackend::Cpu;
        let config = DragonBrowserTrainingConfig {
            experiment_kind: crate::config::DragonExperimentKind::NcaPrepretraining,
            model_config: tiny_model_config(256),
            execution_backend,
            block_size: 8,
            learning_rate: 1.0e-3,
            weight_decay: 0.0,
            batch_size: 2,
            max_train_batches: Some(1),
            max_eval_batches: Some(1),
            capability_policy: Default::default(),
            training_lease: None,
            train_source: DragonBrowserTokenSource::GeneratedNca {
                corpus: tiny_nca_corpus_config(),
                split: DragonBrowserDatasetSplit::Train,
                max_documents: Some(1),
            },
            eval_source: Some(DragonBrowserTokenSource::GeneratedNca {
                corpus: tiny_nca_corpus_config(),
                split: DragonBrowserDatasetSplit::Validation,
                max_documents: Some(1),
            }),
            live_participant: None,
        };
        let result = run_browser_training_with_release_manifest(
            "https://example.invalid",
            &config,
            &dummy_release_manifest(),
        )
        .await
        .expect("generated nca browser training should succeed");
        let expected_backend = match execution_backend.backend_label() {
            "wgpu" => "burn-webgpu-wasm",
            _ => "burn-ndarray-wasm",
        };
        assert_eq!(result.backend, expected_backend);
        assert!(result.train_batches >= 1);
        assert!(result.train_examples >= 1);
        assert!(result.train_loss_mean.is_finite());
    }

    #[wasm_bindgen_test(async)]
    async fn browser_training_supports_factorized_nca_language_head() {
        let mut model_config = tiny_model_config(256);
        model_config.language_head = LanguageHeadConfig::NcaFactorizedPatch {
            state_count: 2,
            patch_size: 2,
            frame_special_tokens: true,
            eos_id: Some(255),
        };
        let config = DragonBrowserTrainingConfig {
            experiment_kind: crate::config::DragonExperimentKind::NcaPrepretraining,
            model_config,
            execution_backend: DragonBrowserExecutionBackend::Cpu,
            block_size: 8,
            learning_rate: 1.0e-3,
            weight_decay: 0.0,
            batch_size: 2,
            max_train_batches: Some(1),
            max_eval_batches: Some(1),
            capability_policy: Default::default(),
            training_lease: None,
            train_source: DragonBrowserTokenSource::GeneratedNca {
                corpus: tiny_nca_corpus_config(),
                split: DragonBrowserDatasetSplit::Train,
                max_documents: Some(1),
            },
            eval_source: Some(DragonBrowserTokenSource::GeneratedNca {
                corpus: tiny_nca_corpus_config(),
                split: DragonBrowserDatasetSplit::Validation,
                max_documents: Some(1),
            }),
            live_participant: None,
        };
        let result = run_browser_training_with_release_manifest(
            "https://example.invalid",
            &config,
            &dummy_release_manifest(),
        )
        .await
        .expect("factorized NCA browser training should succeed");
        assert_eq!(result.backend, "burn-ndarray-wasm");
        assert!(result.train_batches >= 1);
        assert!(result.train_examples >= 1);
        assert!(result.train_loss_mean.is_finite());
    }

    #[wasm_bindgen_test(async)]
    async fn browser_training_smoke_http_json() {
        let records = vec![
            TokenWindowRecord {
                inputs: vec![1, 2, 3, 4, 5, 6, 7, 8],
                targets: vec![2, 3, 4, 5, 6, 7, 8, 9],
                reset_stream_state: true,
            },
            TokenWindowRecord {
                inputs: vec![2, 3, 4, 5, 6, 7, 8, 9],
                targets: vec![3, 4, 5, 6, 7, 8, 9, 10],
                reset_stream_state: false,
            },
        ];
        let payload = serde_json::to_string(&json!({ "records": records })).unwrap();
        let data_url = format!(
            "data:application/json;charset=utf-8,{}",
            encode_uri_component(&payload)
        );
        let config = DragonBrowserTrainingConfig {
            experiment_kind: crate::config::DragonExperimentKind::ClimbMixPretraining,
            model_config: tiny_model_config(256),
            execution_backend: DragonBrowserExecutionBackend::Cpu,
            block_size: 8,
            learning_rate: 1.0e-3,
            weight_decay: 0.0,
            batch_size: 2,
            max_train_batches: Some(1),
            max_eval_batches: None,
            capability_policy: Default::default(),
            training_lease: None,
            train_source: DragonBrowserTokenSource::HttpJson {
                url: data_url.into(),
            },
            eval_source: None,
            live_participant: None,
        };
        let result = run_browser_training_with_release_manifest(
            "https://example.invalid",
            &config,
            &dummy_release_manifest(),
        )
        .await
        .expect("http shard browser training should succeed");
        assert_eq!(result.train_batches, 1);
        assert_eq!(result.train_examples, 2);
        assert!(result.train_loss_mean.is_finite());
    }

    #[wasm_bindgen_test(async)]
    async fn browser_training_smoke_shard_manifest_http() {
        let shard_a = vec![
            TokenWindowRecord {
                inputs: vec![1, 2, 3, 4, 5, 6, 7, 8],
                targets: vec![2, 3, 4, 5, 6, 7, 8, 9],
                reset_stream_state: true,
            },
            TokenWindowRecord {
                inputs: vec![2, 3, 4, 5, 6, 7, 8, 9],
                targets: vec![3, 4, 5, 6, 7, 8, 9, 10],
                reset_stream_state: false,
            },
        ];
        let shard_b = vec![
            TokenWindowRecord {
                inputs: vec![10, 11, 12, 13, 14, 15, 16, 17],
                targets: vec![11, 12, 13, 14, 15, 16, 17, 18],
                reset_stream_state: false,
            },
            TokenWindowRecord {
                inputs: vec![11, 12, 13, 14, 15, 16, 17, 18],
                targets: vec![12, 13, 14, 15, 16, 17, 18, 19],
                reset_stream_state: false,
            },
        ];
        let shard_a_bytes = serde_json::to_vec(&shard_a).expect("shard a bytes");
        let shard_b_bytes = serde_json::to_vec(&shard_b).expect("shard b bytes");
        let manifest = ShardFetchManifest {
            dataset_view_id: DatasetViewId::new("dragon-climbmix-browser"),
            entries: vec![
                ShardFetchEntry {
                    microshard_id: MicroShardId::new("shard-a"),
                    ordinal: 0,
                    locator: json_data_url(&shard_a),
                    content_hash: ContentId::from_multihash(multihash_sha256(&shard_a_bytes)),
                    bytes_len: shard_a_bytes.len() as u64,
                },
                ShardFetchEntry {
                    microshard_id: MicroShardId::new("shard-b"),
                    ordinal: 1,
                    locator: json_data_url(&shard_b),
                    content_hash: ContentId::from_multihash(multihash_sha256(&shard_b_bytes)),
                    bytes_len: shard_b_bytes.len() as u64,
                },
            ],
        };
        let config = DragonBrowserTrainingConfig {
            experiment_kind: crate::config::DragonExperimentKind::ClimbMixPretraining,
            model_config: tiny_model_config(256),
            execution_backend: DragonBrowserExecutionBackend::Cpu,
            block_size: 8,
            learning_rate: 1.0e-3,
            weight_decay: 0.0,
            batch_size: 2,
            max_train_batches: Some(2),
            max_eval_batches: None,
            capability_policy: Default::default(),
            training_lease: None,
            train_source: DragonBrowserTokenSource::ShardManifestHttp {
                manifest_url: json_data_url(&manifest),
                selection: DragonBrowserShardSelectionPolicy::DeterministicPeer,
                max_shards_per_window: Some(4),
            },
            eval_source: None,
            live_participant: None,
        };
        let result = run_browser_training_with_release_manifest(
            "https://example.invalid",
            &config,
            &dummy_release_manifest(),
        )
        .await
        .expect("shard-manifest browser training should succeed");
        assert_eq!(result.train_batches, 2);
        assert_eq!(result.train_examples, 4);
        assert!(result.train_loss_mean.is_finite());
    }

    #[wasm_bindgen_test(async)]
    async fn browser_training_shard_manifest_limits_shards_per_window() {
        let shard_a = vec![
            TokenWindowRecord {
                inputs: vec![1, 2, 3, 4, 5, 6, 7, 8],
                targets: vec![2, 3, 4, 5, 6, 7, 8, 9],
                reset_stream_state: true,
            },
            TokenWindowRecord {
                inputs: vec![2, 3, 4, 5, 6, 7, 8, 9],
                targets: vec![3, 4, 5, 6, 7, 8, 9, 10],
                reset_stream_state: false,
            },
        ];
        let shard_b = vec![
            TokenWindowRecord {
                inputs: vec![10, 11, 12, 13, 14, 15, 16, 17],
                targets: vec![11, 12, 13, 14, 15, 16, 17, 18],
                reset_stream_state: false,
            },
            TokenWindowRecord {
                inputs: vec![11, 12, 13, 14, 15, 16, 17, 18],
                targets: vec![12, 13, 14, 15, 16, 17, 18, 19],
                reset_stream_state: false,
            },
        ];
        let shard_a_bytes = serde_json::to_vec(&shard_a).expect("shard a bytes");
        let shard_b_bytes = serde_json::to_vec(&shard_b).expect("shard b bytes");
        let manifest = ShardFetchManifest {
            dataset_view_id: DatasetViewId::new("dragon-climbmix-browser"),
            entries: vec![
                ShardFetchEntry {
                    microshard_id: MicroShardId::new("shard-a"),
                    ordinal: 0,
                    locator: json_data_url(&shard_a),
                    content_hash: ContentId::from_multihash(multihash_sha256(&shard_a_bytes)),
                    bytes_len: shard_a_bytes.len() as u64,
                },
                ShardFetchEntry {
                    microshard_id: MicroShardId::new("shard-b"),
                    ordinal: 1,
                    locator: json_data_url(&shard_b),
                    content_hash: ContentId::from_multihash(multihash_sha256(&shard_b_bytes)),
                    bytes_len: shard_b_bytes.len() as u64,
                },
            ],
        };
        let config = DragonBrowserTrainingConfig {
            experiment_kind: crate::config::DragonExperimentKind::ClimbMixPretraining,
            model_config: tiny_model_config(256),
            execution_backend: DragonBrowserExecutionBackend::Cpu,
            block_size: 8,
            learning_rate: 1.0e-3,
            weight_decay: 0.0,
            batch_size: 2,
            max_train_batches: Some(4),
            max_eval_batches: None,
            capability_policy: Default::default(),
            training_lease: None,
            train_source: DragonBrowserTokenSource::ShardManifestHttp {
                manifest_url: json_data_url(&manifest),
                selection: DragonBrowserShardSelectionPolicy::Sequential,
                max_shards_per_window: Some(1),
            },
            eval_source: None,
            live_participant: None,
        };
        let result = run_browser_training_with_release_manifest(
            "https://example.invalid",
            &config,
            &dummy_release_manifest(),
        )
        .await
        .expect("limited shard-manifest browser training should succeed");
        assert_eq!(result.train_batches, 1);
        assert_eq!(result.train_examples, 2);
        assert!(result.train_loss_mean.is_finite());
    }

    #[wasm_bindgen_test(async)]
    async fn browser_training_shard_manifest_respects_training_lease_microshards() {
        let shard_a = vec![
            TokenWindowRecord {
                inputs: vec![1, 2, 3, 4, 5, 6, 7, 8],
                targets: vec![2, 3, 4, 5, 6, 7, 8, 9],
                reset_stream_state: true,
            },
            TokenWindowRecord {
                inputs: vec![2, 3, 4, 5, 6, 7, 8, 9],
                targets: vec![3, 4, 5, 6, 7, 8, 9, 10],
                reset_stream_state: false,
            },
        ];
        let shard_b = vec![
            TokenWindowRecord {
                inputs: vec![10, 11, 12, 13, 14, 15, 16, 17],
                targets: vec![11, 12, 13, 14, 15, 16, 17, 18],
                reset_stream_state: true,
            },
            TokenWindowRecord {
                inputs: vec![11, 12, 13, 14, 15, 16, 17, 18],
                targets: vec![12, 13, 14, 15, 16, 17, 18, 19],
                reset_stream_state: false,
            },
        ];
        let shard_c = vec![
            TokenWindowRecord {
                inputs: vec![20, 21, 22, 23, 24, 25, 26, 27],
                targets: vec![21, 22, 23, 24, 25, 26, 27, 28],
                reset_stream_state: true,
            },
            TokenWindowRecord {
                inputs: vec![21, 22, 23, 24, 25, 26, 27, 28],
                targets: vec![22, 23, 24, 25, 26, 27, 28, 29],
                reset_stream_state: false,
            },
        ];
        let shard_a_bytes = serde_json::to_vec(&shard_a).expect("shard a bytes");
        let shard_b_bytes = serde_json::to_vec(&shard_b).expect("shard b bytes");
        let shard_c_bytes = serde_json::to_vec(&shard_c).expect("shard c bytes");
        let manifest = ShardFetchManifest {
            dataset_view_id: DatasetViewId::new("dragon-climbmix-browser"),
            entries: vec![
                ShardFetchEntry {
                    microshard_id: MicroShardId::new("shard-a"),
                    ordinal: 0,
                    locator: json_data_url(&shard_a),
                    content_hash: ContentId::from_multihash(multihash_sha256(&shard_a_bytes)),
                    bytes_len: shard_a_bytes.len() as u64,
                },
                ShardFetchEntry {
                    microshard_id: MicroShardId::new("shard-b"),
                    ordinal: 1,
                    locator: json_data_url(&shard_b),
                    content_hash: ContentId::from_multihash(multihash_sha256(&shard_b_bytes)),
                    bytes_len: shard_b_bytes.len() as u64,
                },
                ShardFetchEntry {
                    microshard_id: MicroShardId::new("shard-c"),
                    ordinal: 2,
                    locator: json_data_url(&shard_c),
                    content_hash: ContentId::from_multihash(multihash_sha256(&shard_c_bytes)),
                    bytes_len: shard_c_bytes.len() as u64,
                },
            ],
        };
        let config = DragonBrowserTrainingConfig {
            experiment_kind: crate::config::DragonExperimentKind::ClimbMixPretraining,
            model_config: tiny_model_config(256),
            execution_backend: DragonBrowserExecutionBackend::Cpu,
            block_size: 8,
            learning_rate: 1.0e-3,
            weight_decay: 0.0,
            batch_size: 2,
            max_train_batches: Some(4),
            max_eval_batches: None,
            capability_policy: Default::default(),
            training_lease: Some(sample_training_lease(&["shard-b"])),
            train_source: DragonBrowserTokenSource::ShardManifestHttp {
                manifest_url: json_data_url(&manifest),
                selection: DragonBrowserShardSelectionPolicy::DeterministicPeer,
                max_shards_per_window: Some(4),
            },
            eval_source: None,
            live_participant: None,
        };
        let result = run_browser_training_with_release_manifest(
            "https://example.invalid",
            &config,
            &dummy_release_manifest(),
        )
        .await
        .expect("leased microshard browser training should succeed");
        assert_eq!(result.train_batches, 1);
        assert_eq!(result.train_examples, 2);
        assert!(result.train_loss_mean.is_finite());
    }

    #[cfg(feature = "wgpu")]
    #[wasm_bindgen_test(async)]
    async fn browser_training_downgrades_cleanly_under_tiny_budget() {
        let config = DragonBrowserTrainingConfig {
            experiment_kind: crate::config::DragonExperimentKind::NcaPrepretraining,
            model_config: tiny_model_config(256),
            execution_backend: DragonBrowserExecutionBackend::Wgpu,
            block_size: 8,
            learning_rate: 1.0e-3,
            weight_decay: 0.0,
            batch_size: 2,
            max_train_batches: Some(1),
            max_eval_batches: Some(1),
            capability_policy: crate::config::DragonCapabilityPolicy {
                browser_wgpu_memory_budget_bytes: Some(1),
                ..Default::default()
            },
            training_lease: None,
            train_source: DragonBrowserTokenSource::GeneratedNca {
                corpus: tiny_nca_corpus_config(),
                split: DragonBrowserDatasetSplit::Train,
                max_documents: Some(1),
            },
            eval_source: None,
            live_participant: None,
        };
        let error = run_browser_training_with_release_manifest(
            "https://example.invalid",
            &config,
            &dummy_release_manifest(),
        )
        .await
        .expect_err("tiny browser budget should downgrade before training starts");
        assert!(
            error.to_string().contains("downgrading to verifier"),
            "unexpected error: {error}"
        );
    }

    fn tiny_model_config(vocab_size: usize) -> DragonConfig {
        DragonConfig {
            n_layer: 1,
            n_embd: 16,
            dropout: 0.0,
            n_head: 1,
            mlp_internal_dim_multiplier: 2,
            n_expert: 1,
            vocab_size,
            ..DragonConfig::default()
        }
    }

    fn sample_browser_training_config() -> DragonBrowserTrainingConfig {
        DragonBrowserTrainingConfig {
            experiment_kind: crate::config::DragonExperimentKind::NcaPrepretraining,
            model_config: tiny_model_config(256),
            execution_backend: DragonBrowserExecutionBackend::Cpu,
            block_size: 8,
            learning_rate: 1.0e-3,
            weight_decay: 0.0,
            batch_size: 2,
            max_train_batches: Some(1),
            max_eval_batches: Some(1),
            capability_policy: Default::default(),
            training_lease: None,
            train_source: DragonBrowserTokenSource::GeneratedNca {
                corpus: tiny_nca_corpus_config(),
                split: DragonBrowserDatasetSplit::Train,
                max_documents: Some(1),
            },
            eval_source: Some(DragonBrowserTokenSource::GeneratedNca {
                corpus: tiny_nca_corpus_config(),
                split: DragonBrowserDatasetSplit::Validation,
                max_documents: Some(1),
            }),
            live_participant: None,
        }
    }

    fn tiny_nca_corpus_config() -> NcaCorpusConfig {
        NcaCorpusConfig {
            output_dir: PathBuf::from("wasm-browser-nca-smoke"),
            seed: 7,
            name: "wasm-browser-nca-smoke".into(),
            train_samples: 1,
            validation_samples: 1,
            chunk_token_capacity: 256,
            serialization: NcaSerializationConfig {
                patch_size: 2,
                ..NcaSerializationConfig::default()
            },
            tokenization: NcaTokenizationConfig::PatchTokenIds {
                vocab_size: 256,
                eos_id: Some(255),
                frame_special_tokens: true,
            },
            families: vec![NcaFamilyConfig {
                kind: NcaFamilyKind::Cyclic,
                weight: 1,
                complexity: Default::default(),
                grid_size: Some(UsizeRangeConfig { min: 4, max: 4 }),
                steps: Some(UsizeRangeConfig { min: 4, max: 4 }),
                state_count: Some(UsizeRangeConfig { min: 2, max: 2 }),
                step_stride: Some(UsizeRangeConfig { min: 1, max: 1 }),
                start_step: Some(UsizeRangeConfig { min: 0, max: 0 }),
                identity_bias: None,
                temperature: None,
                rule_filter: None,
            }],
        }
    }

    fn json_data_url<T: serde::Serialize>(value: &T) -> String {
        let payload = serde_json::to_string(value).expect("json payload");
        format!(
            "data:application/json;charset=utf-8,{}",
            encode_uri_component(&payload)
        )
    }

    fn sample_training_lease(microshard_ids: &[&str]) -> WorkloadTrainingLease {
        WorkloadTrainingLease {
            lease_id: burn_p2p::LeaseId::new("wasm-browser-lease"),
            window_id: burn_p2p::WindowId(1),
            dataset_view_id: burn_p2p::DatasetViewId::new("wasm-browser-view"),
            assignment_hash: ContentId::new("wasm-browser-assignment"),
            microshards: microshard_ids
                .iter()
                .map(|microshard_id| burn_p2p::MicroShardId::new(*microshard_id))
                .collect(),
        }
    }

    fn dummy_release_manifest() -> ClientReleaseManifest {
        serde_json::from_value(json!({
            "project_family_id": "burn-dragon-language",
            "release_train_hash": "browser-smoke-train",
            "target_artifact_id": "browser-wasm",
            "target_artifact_hash": "browser-smoke-artifact",
            "target_platform": "browser",
            "app_semver": "0.21.0-pre.15",
            "git_commit": "smoke",
            "cargo_lock_hash": "browser-smoke-lock",
            "burn_version_string": "0.21.0-pre.3",
            "enabled_features_hash": "browser-smoke-features",
            "protocol_major": 0,
            "supported_workloads": [],
            "built_at": "2026-04-11T00:00:00Z"
        }))
        .unwrap_or_else(|_| ClientReleaseManifest {
            project_family_id: ProjectFamilyId::new("burn-dragon-language"),
            release_train_hash: ContentId::new("browser-smoke-train"),
            target_artifact_id: "browser-wasm".into(),
            target_artifact_hash: ContentId::new("browser-smoke-artifact"),
            target_platform: ClientPlatform::Browser,
            app_semver: semver::Version::parse("0.21.0-pre.15").expect("valid burn_dragon version"),
            git_commit: "smoke".into(),
            cargo_lock_hash: ContentId::new("browser-smoke-lock"),
            burn_version_string: "0.21.0-pre.3".into(),
            enabled_features_hash: ContentId::new("browser-smoke-features"),
            protocol_major: 0,
            supported_workloads: Vec::new(),
            built_at: chrono::Utc::now(),
        })
    }
}
