use std::collections::{BTreeMap, BTreeSet};
#[cfg(feature = "wgpu")]
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Context;
use anyhow::{Result, anyhow, bail, ensure};
use burn::backend::NdArray;
use burn::module::{AutodiffModule, Module};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, AdamWConfig, GradientsAccumulator, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
use burn_autodiff::Autodiff;
use burn_dragon_core::objective::masked_token_mean;
use burn_dragon_core::{DragonModel, ModelState};
use burn_dragon_eggroll::{
    AntitheticFitness, EggrollModuleOptimizerState, apply_antithetic_update_with_allowed_param_ids,
    perturb_module_with_allowed_param_ids,
};
use burn_dragon_time::Instant;
use burn_p2p::{
    ArtifactId, ArtifactKind, COMPACT_UPDATE_PAYLOAD_VERSION, ChunkingScheme,
    CompactScalarEncoding, CompactScalarVector, CompactUpdateBody, CompactUpdatePayload, ContentId,
    ExperimentId, ExperimentScope, HeadDescriptor, HeadId, Precision, RevisionId,
    SeededFitnessGeneration, StudyId, TrainingProtocol, WorkloadId, WorkloadTrainingArtifact,
    WorkloadTrainingArtifactChunk, WorkloadTrainingContribution, WorkloadTrainingLease,
    WorkloadUpdateEnvelope,
};
use burn_p2p_browser::{
    BrowserBootstrapHead, BrowserCapabilityReport, BrowserRuntimeRole, BrowserSessionRuntimeConfig,
    BrowserSessionRuntimeError, BrowserSessionRuntimeHandle, BrowserSessionState,
    BrowserTrainingBudget, BrowserTrainingPlan,
};
use burn_p2p_checkpoint::{ArtifactBuildSpec, build_artifact_descriptor_from_bytes};
use burn_p2p_core::codec::multihash_sha256;
use burn_p2p_dataloader::ShardFetchManifest;
#[cfg(feature = "wgpu")]
use burn_wgpu::{RuntimeOptions, graphics};
use chrono::{DateTime, Duration, Utc};
use gloo_net::http::Request;
use log::info;
use serde::{Deserialize, Serialize};
use url::Url;

use crate::auth::{
    browser_github_enrollment_config, fetch_edge_snapshot, load_or_enroll_browser_session,
};
use crate::browser_data::{
    GeneratedRecordSelection, generated_nca_records, generated_ruliad_records,
};
use crate::browser_record::{
    BrowserBurnRecordBytesFormat, BrowserBurnRecordPrecision, browser_random_scaffold_contract,
    browser_random_scaffold_tensor_digest_from_mutable, browser_record_format_name,
    browser_record_precision_descriptor, encode_browser_record_bytes,
    flatten_browser_random_scaffold_mutable, load_browser_active_head_model,
    load_browser_genesis_model, verify_browser_signed_genesis_tensor_digest,
};
use crate::capability::{decide_browser_capability, detect_browser_host_capabilities};
#[cfg(target_arch = "wasm32")]
use crate::capability_state::{
    apply_browser_downgrade_state, clear_browser_downgrade, is_probable_trainer_fit_failure,
    persist_browser_downgrade,
};
#[cfg(test)]
use crate::config::DragonBrowserDatasetSplit;
use crate::config::{
    DragonBrowserExecutionBackend, DragonBrowserLiveParticipantConfig,
    DragonBrowserOptimizerConfig, DragonBrowserShardSelectionPolicy, DragonBrowserTokenSource,
    DragonBrowserTrainingConfig, TokenWindowRecord, dragon_model_schema_hash,
};
use crate::p2p_adapter::{browser_runtime_role_label, browser_trainer_transport_policy};
use crate::profile::{
    DRAGON_BROWSER_EXECUTION_CONTRACT_EXTENSION, browser_runtime_execution_contract_hash,
};
use crate::seeded_fitness::dragon_seeded_fitness_catalog;

type BrowserCpuEvalBackend = NdArray<f32>;
type BrowserCpuTrainBackend = Autodiff<BrowserCpuEvalBackend>;

const BROWSER_LIVE_SESSION_REFRESH_GRACE_SECS: i64 = 120;

#[cfg(feature = "wgpu")]
type BrowserWgpuEvalBackend = burn_wgpu::Wgpu<f32>;
#[cfg(feature = "wgpu")]
type BrowserWgpuTrainBackend = Autodiff<BrowserWgpuEvalBackend>;
#[cfg(feature = "wgpu")]
type BrowserWgpuTrainDevice = burn::tensor::Device<BrowserWgpuTrainBackend>;

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
    #[serde(default)]
    pub train_loss_observed: bool,
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
    #[serde(default)]
    pub receipt_submission_deferred: bool,
    #[serde(default)]
    pub pending_receipt_count: usize,
    #[serde(default)]
    pub receipt_submission_error: Option<String>,
    pub accepted_receipt_ids: Vec<String>,
    pub emitted_receipt_id: Option<String>,
    #[serde(default)]
    pub artifact_published: bool,
    #[serde(default)]
    pub update_announced: bool,
    pub runtime_state: Option<String>,
    pub transport: Option<String>,
}

#[derive(Clone, Debug)]
struct TokenWindowBatch<B: Backend> {
    inputs: Tensor<B, 2, Int>,
    targets: Tensor<B, 2, Int>,
    loss_mask: Option<Tensor<B, 2, Int>>,
    token_count: usize,
    batch_digest: ContentId,
    record_digests: Vec<ContentId>,
    reset_stream_state: bool,
}

#[derive(Clone, Debug)]
struct BrowserCompactUpdateTrace {
    parameter_catalog_hash: ContentId,
    parameter_count: u64,
    perturbation_generator_hash: ContentId,
    optimizer_update_hash: ContentId,
    generations: Vec<SeededFitnessGeneration>,
}

struct BrowserKernelStep<B: Backend> {
    model: DragonModel<B>,
    losses: Vec<f64>,
}

trait BrowserTrainingKernel<B: Backend> {
    async fn step(
        &mut self,
        model: DragonModel<B>,
        batch: &TokenWindowBatch<B>,
        generation: u64,
    ) -> Result<BrowserKernelStep<B>>;

    fn compact_trace(&self) -> Option<&BrowserCompactUpdateTrace> {
        None
    }

    async fn finish_loss_observation(&mut self) -> Result<Vec<f64>> {
        Ok(Vec::new())
    }

    fn observes_loss(&self) -> bool {
        true
    }

    fn defers_loss_readback(&self) -> bool {
        false
    }
}

struct BrowserAdamwKernel<B: AutodiffBackend>
where
    DragonModel<B>: AutodiffModule<B>,
{
    learning_rate: f64,
    immediate_loss_readback: bool,
    deferred_loss_sum: Option<Tensor<B, 1>>,
    deferred_loss_count: usize,
    tbptt_chunk_size: Option<usize>,
    persist_across_steps: bool,
    state: Option<ModelState<B>>,
    optimizer: OptimizerAdaptor<AdamW, DragonModel<B>, B>,
}

impl<B: AutodiffBackend> BrowserAdamwKernel<B>
where
    DragonModel<B>: AutodiffModule<B>,
{
    fn new(
        learning_rate: f64,
        weight_decay: f32,
        immediate_loss_readback: bool,
        tbptt_chunk_size: Option<usize>,
        persist_across_steps: bool,
    ) -> Self {
        Self {
            learning_rate,
            immediate_loss_readback,
            deferred_loss_sum: None,
            deferred_loss_count: 0,
            tbptt_chunk_size,
            persist_across_steps,
            state: None,
            optimizer: AdamWConfig::new().with_weight_decay(weight_decay).init(),
        }
    }
}

impl<B: AutodiffBackend> BrowserTrainingKernel<B> for BrowserAdamwKernel<B>
where
    DragonModel<B>: AutodiffModule<B>,
{
    async fn step(
        &mut self,
        model: DragonModel<B>,
        batch: &TokenWindowBatch<B>,
        _generation: u64,
    ) -> Result<BrowserKernelStep<B>> {
        let mut state = take_browser_step_state(
            &model,
            &mut self.state,
            batch.reset_stream_state,
            self.persist_across_steps,
        );
        let mut accumulator = GradientsAccumulator::new();
        let mut observed_losses = Vec::new();
        visit_browser_next_token_chunks(&model, batch, &mut state, self.tbptt_chunk_size, |loss| {
            observed_losses.push(loss.clone().detach());
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            accumulator.accumulate(&model, grads);
        });
        let observed_loss = sum_scalar_losses(observed_losses);
        let losses = if self.immediate_loss_readback {
            vec![scalar_from_loss_async(observed_loss).await?]
        } else {
            self.deferred_loss_sum = Some(match self.deferred_loss_sum.take() {
                Some(total) => total + observed_loss,
                None => observed_loss,
            });
            self.deferred_loss_count = self.deferred_loss_count.saturating_add(1);
            Vec::new()
        };
        let grads = accumulator.grads();
        let model = self.optimizer.step(self.learning_rate, model, grads);
        store_browser_step_state(&mut self.state, state, self.persist_across_steps);
        Ok(BrowserKernelStep { model, losses })
    }

    async fn finish_loss_observation(&mut self) -> Result<Vec<f64>> {
        let Some(total) = self.deferred_loss_sum.take() else {
            return Ok(Vec::new());
        };
        let count = std::mem::take(&mut self.deferred_loss_count);
        ensure!(count > 0, "deferred browser loss has no observations");
        Ok(vec![scalar_from_loss_async(total).await? / count as f64])
    }

    fn defers_loss_readback(&self) -> bool {
        !self.immediate_loss_readback
    }
}

struct BrowserSeededFitnessKernel<B: Backend> {
    config: burn_eggroll::EggrollConfig,
    scalar_encoding: CompactScalarEncoding,
    allowed_param_ids: BTreeSet<u64>,
    optimizer_state: EggrollModuleOptimizerState<B>,
    trace: BrowserCompactUpdateTrace,
    tbptt_chunk_size: Option<usize>,
    persist_across_steps: bool,
    state: Option<ModelState<B>>,
}

impl<B: Backend> BrowserSeededFitnessKernel<B> {
    fn new(
        model: &DragonModel<B>,
        config: burn_eggroll::EggrollConfig,
        scalar_encoding: CompactScalarEncoding,
        optimizer_update_hash: ContentId,
        tbptt_chunk_size: Option<usize>,
        persist_across_steps: bool,
    ) -> Result<Self> {
        config.validate()?;
        let catalog = dragon_seeded_fitness_catalog(model, &config)?;
        Ok(Self {
            config,
            scalar_encoding,
            allowed_param_ids: catalog.allowed_param_ids,
            optimizer_state: EggrollModuleOptimizerState::new(),
            trace: BrowserCompactUpdateTrace {
                parameter_catalog_hash: catalog.parameter_catalog_hash,
                parameter_count: catalog.parameter_count,
                perturbation_generator_hash: catalog.perturbation_generator_hash,
                optimizer_update_hash,
                generations: Vec::new(),
            },
            tbptt_chunk_size,
            persist_across_steps,
            state: None,
        })
    }
}

impl<B: Backend> BrowserTrainingKernel<B> for BrowserSeededFitnessKernel<B> {
    async fn step(
        &mut self,
        model: DragonModel<B>,
        batch: &TokenWindowBatch<B>,
        generation: u64,
    ) -> Result<BrowserKernelStep<B>> {
        let base_state = take_browser_step_state(
            &model,
            &mut self.state,
            batch.reset_stream_state,
            self.persist_across_steps,
        );
        let pair_count = self.config.population.population_size / 2;
        let chunk_pairs = (self.config.population.population_chunk_size / 2)
            .max(1)
            .min(pair_count);
        let mut losses = Vec::with_capacity(pair_count * 2);
        let mut fitness = Vec::with_capacity(pair_count);
        for pair_start in (0..pair_count).step_by(chunk_pairs) {
            let pair_end = (pair_start + chunk_pairs).min(pair_count);
            let mut loss_tensors = Vec::with_capacity((pair_end - pair_start) * 2);
            for pair_index in pair_start..pair_end {
                for sign in [
                    burn_eggroll::AntitheticSign::Plus,
                    burn_eggroll::AntitheticSign::Minus,
                ] {
                    let candidate = perturb_module_with_allowed_param_ids(
                        model.clone(),
                        &self.config,
                        generation,
                        pair_index as u64,
                        sign,
                        Some(&self.allowed_param_ids),
                    );
                    let mut candidate_state = base_state.detached_clone();
                    let mut candidate_losses = Vec::new();
                    visit_browser_next_token_chunks(
                        &candidate,
                        batch,
                        &mut candidate_state,
                        self.tbptt_chunk_size,
                        |loss| candidate_losses.push(loss),
                    );
                    loss_tensors.push(sum_scalar_losses(candidate_losses));
                }
            }
            losses.extend(scalar_values_from_loss_tensors_async(loss_tensors).await?);
        }
        for (pair_index, pair_losses) in losses.chunks_exact(2).enumerate() {
            fitness.push(AntitheticFitness {
                pair_index: pair_index as u64,
                plus: -(pair_losses[0] as f32),
                minus: -(pair_losses[1] as f32),
            });
        }
        let transmitted_fitness = fitness
            .iter()
            .flat_map(|item| [item.plus, item.minus])
            .collect::<Vec<_>>();
        self.trace.generations.push(SeededFitnessGeneration {
            generation,
            batch_digest: batch.batch_digest.clone(),
            record_digests: batch.record_digests.clone(),
            reset_stream_state: batch.reset_stream_state,
            fitness: CompactScalarVector::encode(&transmitted_fitness, self.scalar_encoding)
                .map_err(anyhow::Error::msg)?,
        });
        let (model, _) = apply_antithetic_update_with_allowed_param_ids(
            model,
            &self.config,
            generation,
            &fitness,
            &mut self.optimizer_state,
            Some(&self.allowed_param_ids),
        )?;
        let mut next_state = base_state;
        visit_browser_next_token_chunks(
            &model,
            batch,
            &mut next_state,
            self.tbptt_chunk_size,
            drop,
        );
        store_browser_step_state(&mut self.state, next_state, self.persist_across_steps);
        Ok(BrowserKernelStep { model, losses })
    }

    fn compact_trace(&self) -> Option<&BrowserCompactUpdateTrace> {
        Some(&self.trace)
    }
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
    stream_aligned: bool,
}

struct LiveBrowserParticipantHandle {
    session_runtime: BrowserSessionRuntimeHandle,
    training_budget: BrowserTrainingBudget,
    revision_contract: Option<burn_p2p::RevisionContractBundle>,
}

#[derive(Default)]
pub(crate) struct DragonBrowserTrainingSession {
    live_browser_session: Option<BrowserSessionState>,
    live_participant: Option<LiveBrowserParticipantHandle>,
}

impl DragonBrowserTrainingSession {
    fn live_session_principal_id(&self) -> Option<&str> {
        live_session_principal_id(self.live_browser_session.as_ref())
    }

    fn live_session_refresh_deadline(&self) -> DateTime<Utc> {
        let max_window_secs = self
            .live_participant
            .as_ref()
            .map(|participant| participant.training_budget.max_window_secs)
            .unwrap_or(30);
        browser_live_session_refresh_deadline(max_window_secs)
    }

    fn live_participant_matches_config(&self, config: &DragonBrowserTrainingConfig) -> bool {
        let Some(live) = config.live_participant.as_ref() else {
            return self.live_participant.is_none();
        };
        self.live_participant
            .as_ref()
            .and_then(|participant| {
                participant
                    .session_runtime
                    .runtime
                    .storage
                    .active_assignment
                    .as_ref()
            })
            .is_some_and(|assignment| {
                assignment.study_id.as_str() == live.study_id
                    && assignment.experiment_id.as_str() == live.experiment_id
                    && assignment.revision_id.as_str() == live.revision_id
            })
    }

    async fn refresh_live_browser_session_if_needed(
        &mut self,
        edge_base_url: &str,
        config: &DragonBrowserTrainingConfig,
        release_manifest: &burn_p2p::ClientReleaseManifest,
        deadline: DateTime<Utc>,
    ) -> Result<bool> {
        let should_refresh = self
            .live_browser_session
            .as_ref()
            .is_none_or(|session| session.session_expires_before(deadline));
        if !should_refresh {
            return Ok(false);
        }

        info!("browser live participant session refresh starting");
        let requested_scopes = config
            .live_participant
            .as_ref()
            .map(live_browser_training_requested_scopes)
            .unwrap_or_else(|| BTreeSet::from([ExperimentScope::Connect]));
        let session =
            load_or_enroll_browser_session(edge_base_url, release_manifest, requested_scopes, 900)
                .await?;
        if session.session.is_none() {
            bail!("browser live training requires an authenticated session");
        }
        if let Some(participant) = self.live_participant.as_mut() {
            participant.session_runtime.session = session.clone();
            participant
                .session_runtime
                .runtime
                .remember_session(session.clone());
        }
        self.live_browser_session = Some(session);
        info!("browser live participant session refresh complete");
        Ok(true)
    }

    async fn ensure_live_participant(
        &mut self,
        edge_base_url: &str,
        config: &DragonBrowserTrainingConfig,
        release_manifest: &burn_p2p::ClientReleaseManifest,
    ) -> Result<()> {
        if config.live_participant.is_none() {
            self.live_browser_session = None;
            self.live_participant = None;
            return Ok(());
        }

        if !self.live_participant_matches_config(config) {
            self.live_participant = None;
        }
        let refresh_deadline = self.live_session_refresh_deadline();
        self.refresh_live_browser_session_if_needed(
            edge_base_url,
            config,
            release_manifest,
            refresh_deadline,
        )
        .await?;
        if self.live_participant.is_none() {
            info!("browser live participant runtime starting");
            self.live_participant = start_live_browser_participant(
                edge_base_url,
                config,
                release_manifest,
                self.live_browser_session.as_ref(),
            )
            .await?;
        } else {
            info!("browser live participant runtime reused");
        }
        let refresh_deadline = self.live_session_refresh_deadline();
        self.refresh_live_browser_session_if_needed(
            edge_base_url,
            config,
            release_manifest,
            refresh_deadline,
        )
        .await?;

        Ok(())
    }
}

fn browser_live_session_refresh_deadline(max_window_secs: u64) -> DateTime<Utc> {
    let max_window_secs = i64::try_from(max_window_secs)
        .unwrap_or(i64::MAX.saturating_sub(BROWSER_LIVE_SESSION_REFRESH_GRACE_SECS));
    Utc::now()
        + Duration::seconds(max_window_secs.saturating_add(BROWSER_LIVE_SESSION_REFRESH_GRACE_SECS))
}

struct BrowserTrainingRunContext<'a> {
    edge_base_url: &'a str,
    config: &'a DragonBrowserTrainingConfig,
    backend_label: &'a str,
    backend_kind: BrowserTrainingBackendKind,
    setup_time_ms: u64,
    live_session_principal_id: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BrowserCanonicalArtifactPublicationDecision {
    requested: bool,
    should_publish: bool,
    disabled_reason: Option<&'static str>,
}

impl<'a> BrowserTrainingRunContext<'a> {
    fn live_session_principal_id(&self) -> Option<&str> {
        self.live_session_principal_id.as_deref()
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
            stream_aligned: self.config.tbptt_persist_across_steps,
        }
    }
}

fn browser_canonical_artifact_publication_decision(
    requested: bool,
    backend_kind: BrowserTrainingBackendKind,
    compact_update: bool,
) -> BrowserCanonicalArtifactPublicationDecision {
    browser_canonical_artifact_publication_decision_for_platform(
        requested,
        backend_kind,
        compact_update,
        cfg!(target_arch = "wasm32"),
    )
}

fn browser_uses_compact_update(config: &DragonBrowserTrainingConfig) -> bool {
    config.optimizer.is_forward_only()
        || (matches!(config.optimizer, DragonBrowserOptimizerConfig::Adamw)
            && config.model_config.random_scaffold.enabled)
}

fn browser_canonical_artifact_publication_decision_for_platform(
    requested: bool,
    backend_kind: BrowserTrainingBackendKind,
    compact_update: bool,
    target_arch_wasm32: bool,
) -> BrowserCanonicalArtifactPublicationDecision {
    #[cfg(not(feature = "wgpu"))]
    let _ = target_arch_wasm32;

    if !requested {
        return BrowserCanonicalArtifactPublicationDecision {
            requested,
            should_publish: false,
            disabled_reason: None,
        };
    }
    if compact_update {
        return BrowserCanonicalArtifactPublicationDecision {
            requested,
            should_publish: true,
            disabled_reason: None,
        };
    }

    match backend_kind {
        BrowserTrainingBackendKind::Cpu => BrowserCanonicalArtifactPublicationDecision {
            requested,
            should_publish: true,
            disabled_reason: None,
        },
        #[cfg(feature = "wgpu")]
        BrowserTrainingBackendKind::Wgpu if target_arch_wasm32 => {
            BrowserCanonicalArtifactPublicationDecision {
                requested,
                should_publish: false,
                disabled_reason: Some(
                    "Burn 0.21 WebGPU recorder requires synchronous tensor reads, which are unsupported in WASM",
                ),
            }
        }
        #[cfg(feature = "wgpu")]
        BrowserTrainingBackendKind::Wgpu => BrowserCanonicalArtifactPublicationDecision {
            requested,
            should_publish: true,
            disabled_reason: None,
        },
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
    let mut session = DragonBrowserTrainingSession::default();
    run_browser_training_with_session(edge_base_url, config, release_manifest, &mut session).await
}

pub(crate) async fn run_browser_training_with_session(
    edge_base_url: &str,
    config: &DragonBrowserTrainingConfig,
    release_manifest: &burn_p2p::ClientReleaseManifest,
    session: &mut DragonBrowserTrainingSession,
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
    info!(
        "browser training start: experiment={} backend={} block_size={} batch_size={} max_train_batches={:?} max_eval_batches={:?} live_participant={}",
        config.experiment_kind.workload_slug(),
        backend_label,
        config.block_size,
        config.batch_size,
        config.max_train_batches,
        config.max_eval_batches,
        config.live_participant.is_some(),
    );
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
    session
        .ensure_live_participant(edge_base_url, config, release_manifest)
        .await?;
    let live_session_principal_id = session.live_session_principal_id().map(str::to_owned);
    let result = match backend_kind {
        BrowserTrainingBackendKind::Cpu => {
            let setup_started_at = Instant::now();
            match &config.optimizer {
                DragonBrowserOptimizerConfig::Adamw => {
                    let device = burn::tensor::Device::<BrowserCpuTrainBackend>::default();
                    BrowserCpuTrainBackend::seed(&device, 1337);
                    let setup_time_ms = elapsed_ms(setup_started_at);
                    run_browser_training_inner::<
                        BrowserCpuTrainBackend,
                        BrowserAdamwKernel<BrowserCpuTrainBackend>,
                        _,
                    >(
                        BrowserTrainingRunContext {
                            edge_base_url,
                            config,
                            backend_label: "burn-ndarray-wasm",
                            backend_kind,
                            setup_time_ms,
                            live_session_principal_id,
                        },
                        &device,
                        session.live_participant.as_mut(),
                        |_model, _contract| {
                            Ok(BrowserAdamwKernel::<BrowserCpuTrainBackend>::new(
                                config.learning_rate,
                                config.weight_decay,
                                browser_loss_scalar_readback_enabled(backend_kind),
                                config.tbptt_chunk_size,
                                config.tbptt_persist_across_steps,
                            ))
                        },
                    )
                    .await
                }
                DragonBrowserOptimizerConfig::SeededFitness {
                    eggroll,
                    scalar_encoding,
                } => {
                    let device = burn::tensor::Device::<BrowserCpuEvalBackend>::default();
                    BrowserCpuEvalBackend::seed(&device, 1337);
                    let setup_time_ms = elapsed_ms(setup_started_at);
                    run_browser_training_inner::<
                        BrowserCpuEvalBackend,
                        BrowserSeededFitnessKernel<BrowserCpuEvalBackend>,
                        _,
                    >(
                        BrowserTrainingRunContext {
                            edge_base_url,
                            config,
                            backend_label: "burn-ndarray-wasm-forward",
                            backend_kind,
                            setup_time_ms,
                            live_session_principal_id,
                        },
                        &device,
                        session.live_participant.as_mut(),
                        |model, contract| {
                            let optimizer_hash = contract
                                .map(|contract| contract.training.optimizer_hash.clone())
                                .map(Ok)
                                .unwrap_or_else(|| ContentId::derive(eggroll))?;
                            BrowserSeededFitnessKernel::new(
                                model,
                                eggroll.clone(),
                                *scalar_encoding,
                                optimizer_hash,
                                config.tbptt_chunk_size,
                                config.tbptt_persist_across_steps,
                            )
                        },
                    )
                    .await
                }
            }
        }
        #[cfg(feature = "wgpu")]
        BrowserTrainingBackendKind::Wgpu => {
            let setup_started_at = Instant::now();
            match &config.optimizer {
                DragonBrowserOptimizerConfig::Adamw => {
                    let device = BrowserWgpuTrainDevice::default();
                    ensure_webgpu_runtime_ready(&device).await;
                    BrowserWgpuTrainBackend::seed(&device, 1337);
                    let setup_time_ms = elapsed_ms(setup_started_at);
                    run_browser_training_inner::<
                        BrowserWgpuTrainBackend,
                        BrowserAdamwKernel<BrowserWgpuTrainBackend>,
                        _,
                    >(
                        BrowserTrainingRunContext {
                            edge_base_url,
                            config,
                            backend_label: "burn-webgpu-wasm",
                            backend_kind,
                            setup_time_ms,
                            live_session_principal_id,
                        },
                        &device,
                        session.live_participant.as_mut(),
                        |_model, _contract| {
                            Ok(BrowserAdamwKernel::<BrowserWgpuTrainBackend>::new(
                                config.learning_rate,
                                config.weight_decay,
                                browser_loss_scalar_readback_enabled(backend_kind),
                                config.tbptt_chunk_size,
                                config.tbptt_persist_across_steps,
                            ))
                        },
                    )
                    .await
                }
                DragonBrowserOptimizerConfig::SeededFitness {
                    eggroll,
                    scalar_encoding,
                } => {
                    let device = burn::tensor::Device::<BrowserWgpuEvalBackend>::default();
                    ensure_webgpu_runtime_ready(&device).await;
                    BrowserWgpuEvalBackend::seed(&device, 1337);
                    let setup_time_ms = elapsed_ms(setup_started_at);
                    run_browser_training_inner::<
                        BrowserWgpuEvalBackend,
                        BrowserSeededFitnessKernel<BrowserWgpuEvalBackend>,
                        _,
                    >(
                        BrowserTrainingRunContext {
                            edge_base_url,
                            config,
                            backend_label: "burn-webgpu-wasm-forward",
                            backend_kind,
                            setup_time_ms,
                            live_session_principal_id,
                        },
                        &device,
                        session.live_participant.as_mut(),
                        |model, contract| {
                            let optimizer_hash = contract
                                .map(|contract| contract.training.optimizer_hash.clone())
                                .map(Ok)
                                .unwrap_or_else(|| ContentId::derive(eggroll))?;
                            BrowserSeededFitnessKernel::new(
                                model,
                                eggroll.clone(),
                                *scalar_encoding,
                                optimizer_hash,
                                config.tbptt_chunk_size,
                                config.tbptt_persist_across_steps,
                            )
                        },
                    )
                    .await
                }
            }
        }
    };

    #[cfg(target_arch = "wasm32")]
    match &result {
        Ok(_) if browser_training_requires_webgpu => {
            let _ = clear_browser_downgrade(edge_base_url, config, backend_label);
        }
        Err(error)
            if browser_training_requires_webgpu
                && is_probable_trainer_fit_failure(&error.to_string()) =>
        {
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

async fn run_browser_training_inner<B, K, F>(
    context: BrowserTrainingRunContext<'_>,
    device: &B::Device,
    mut live_participant: Option<&mut LiveBrowserParticipantHandle>,
    kernel_factory: F,
) -> Result<DragonBrowserTrainingResult>
where
    B: Backend + Clone,
    DragonModel<B>: Module<B>,
    K: BrowserTrainingKernel<B>,
    F: FnOnce(&DragonModel<B>, Option<&burn_p2p::RevisionContractBundle>) -> Result<K>,
{
    validate_browser_training_config(context.config)?;
    validate_live_training_backend(context.config, context.backend_kind)?;

    let total_started_at = Instant::now();

    let train_record_limit = if context.config.training_lease.is_some()
        && matches!(
            &context.config.train_source,
            DragonBrowserTokenSource::ShardManifestHttp { .. }
        ) {
        None
    } else {
        max_record_limit(context.config.batch_size, context.config.max_train_batches)
    };
    let train_records = load_token_records(
        context.edge_base_url,
        &context.config.train_source,
        context.config.block_size,
        context.token_record_load_policy(
            "train",
            train_record_limit,
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
    info!(
        "browser training records loaded: train_examples={} eval_examples={}",
        train_records.len(),
        eval_records.len(),
    );

    let train_batches = build_batches::<B>(
        &train_records,
        context.config.batch_size,
        context.config.block_size,
        context.config.max_train_batches,
        context
            .config
            .training_lease
            .as_ref()
            .map(|lease| lease.window_id.0),
        device,
    )?;
    let eval_batches = build_batches::<B>(
        &eval_records,
        context.config.batch_size,
        context.config.block_size,
        context.config.max_eval_batches,
        None,
        device,
    )?;
    let train_batches_len = train_batches.len();
    let eval_batches_len = eval_batches.len();
    info!(
        "browser training batches built: train_batches={} eval_batches={}",
        train_batches_len, eval_batches_len,
    );

    let training_window_budget_ms = live_participant
        .as_ref()
        .map(|handle| handle.training_budget.max_window_secs.saturating_mul(1000));

    let load_active_head = context
        .config
        .live_participant
        .as_ref()
        .is_none_or(|config| config.load_active_head_artifact);
    let requested_canonical_update = context
        .config
        .live_participant
        .as_ref()
        .is_some_and(|config| config.publish_canonical_update);
    let artifact_publication_decision = browser_canonical_artifact_publication_decision(
        requested_canonical_update,
        context.backend_kind,
        browser_uses_compact_update(context.config),
    );
    if artifact_publication_decision.should_publish && !load_active_head {
        bail!("browser canonical artifact publication requires loading the active head artifact");
    }

    let active_head_artifact = if load_active_head {
        if let Some(live) = live_participant.as_mut() {
            info!(
                "browser active head artifact sync starting: preferred_transport=p2p fallback=edge-download-ticket"
            );
            let artifact = live
                .session_runtime
                .ensure_active_head_artifact_cached()
                .await
                .map_err(|error| anyhow!("browser active head artifact sync failed: {error}"))?;
            let source = live.session_runtime.runtime.swarm_status().artifact_source;
            info!(
                "browser active head artifact sync complete: head_id={} artifact_id={} bytes={} source={:?}",
                artifact.0.as_str(),
                artifact.1.artifact_id.as_str(),
                artifact.2.len(),
                source,
            );
            Some(artifact)
        } else {
            None
        }
    } else {
        info!(
            "browser active head artifact loading disabled for this training profile; using local initialized model"
        );
        None
    };

    let training_started_at = Instant::now();
    info!("browser training loop starting");
    info!("browser model initialization starting");
    let mut model = DragonModel::<B>::new(context.config.model_config.clone(), device);
    info!("browser model initialization complete");
    let revision_contract = live_participant
        .as_ref()
        .and_then(|handle| handle.revision_contract.as_ref());
    let mut active_model_schema_hash = None;
    if let Some((head_id, descriptor, bytes)) = active_head_artifact {
        info!(
            "browser active head model load starting: head_id={} artifact_id={} bytes={}",
            head_id.as_str(),
            descriptor.artifact_id.as_str(),
            bytes.len(),
        );
        validate_browser_active_head_descriptor(
            context.config,
            revision_contract,
            &head_id,
            &descriptor,
        )?;
        if let Some(contract) = revision_contract
            && descriptor == contract.genesis.payload.payload.artifact
        {
            let genesis = &contract.genesis.payload.payload;
            verify_browser_signed_genesis_tensor_digest(
                &context.config.model_config,
                &descriptor,
                &bytes,
                &contract.training_contract_id,
                &contract.training,
                &genesis.materialization,
                &genesis.tensor_digest,
            )
            .context("verify decoded browser genesis tensors")?;
        }
        active_model_schema_hash = Some(descriptor.model_schema_hash.clone());
        model = if let Some(contract) = revision_contract
            && descriptor == contract.genesis.payload.payload.artifact
        {
            load_browser_genesis_model(
                model,
                &descriptor,
                bytes,
                &contract.training_contract_id,
                &contract.training,
                &contract.genesis.payload.payload.materialization,
                device,
            )?
        } else {
            load_browser_active_head_model(model, &descriptor, bytes, device)?
        };
        info!(
            "browser training loaded active head artifact: head_id={} artifact_id={}",
            head_id.as_str(),
            descriptor.artifact_id.as_str(),
        );
    }
    context
        .config
        .training_objective
        .ensure_browser_supported()
        .map_err(anyhow::Error::msg)?;
    let mut kernel = kernel_factory(&model, revision_contract)?;
    let collect_loss_scalars = kernel.observes_loss();
    if kernel.defers_loss_readback() {
        info!(
            "browser training loss scalar readback deferred for backend={}; aggregating on device for one window-boundary readback",
            context.backend_label,
        );
    }
    let mut train_loss_sum = 0.0;
    let mut train_loss_count = 0usize;
    let mut train_batch_count = 0usize;
    let mut train_example_count = 0usize;
    let mut train_token_count = 0usize;
    let generation_base = context
        .config
        .training_lease
        .as_ref()
        .map(|lease| lease.window_id.0.checked_shl(32).unwrap_or(u64::MAX))
        .unwrap_or(0);
    for (batch_index, batch) in train_batches.into_iter().enumerate() {
        if train_batch_count > 0
            && training_window_budget_ms.is_some_and(|budget_ms| {
                training_started_at.elapsed().as_millis() as u64 >= budget_ms
            })
        {
            info!(
                "browser training window budget reached after {} batch(es); stopping local window before next batch",
                train_batch_count
            );
            break;
        }
        if context
            .config
            .max_train_batches
            .is_some_and(|max_batches| batch_index >= max_batches)
        {
            break;
        }
        if batch_index == 0 {
            info!(
                "browser training first batch starting: token_count={} block_size={} batch_size={}",
                batch.token_count, context.config.block_size, context.config.batch_size,
            );
        }
        let generation = generation_base
            .checked_add(batch_index as u64)
            .ok_or_else(|| anyhow!("browser optimizer generation overflowed"))?;
        let step = kernel.step(model, &batch, generation).await?;
        model = step.model;
        for loss in step.losses {
            train_loss_sum += loss;
            train_loss_count = train_loss_count.saturating_add(1);
        }
        train_example_count = train_example_count.saturating_add(
            batch
                .token_count
                .saturating_div(context.config.block_size.max(1)),
        );
        train_token_count = train_token_count.saturating_add(batch.token_count);
        train_batch_count = train_batch_count.saturating_add(1);
        if batch_index == 0 {
            info!("browser training first batch complete");
        }
    }
    if train_batch_count == 0 {
        bail!("browser training window completed zero batches");
    }
    for loss in kernel.finish_loss_observation().await? {
        train_loss_sum += loss;
        train_loss_count = train_loss_count.saturating_add(1);
    }
    let compact_trace = kernel.compact_trace().cloned();
    let training_time_ms = elapsed_ms(training_started_at);
    let train_loss_mean = if train_loss_count > 0 {
        train_loss_sum / train_loss_count as f64
    } else {
        0.0
    };
    info!(
        "browser training loop complete: train_batches={} train_loss_mean={:.4} train_loss_observed={} training_time_ms={}",
        train_batch_count,
        train_loss_mean,
        train_loss_count > 0,
        training_time_ms,
    );

    let eval_started_at = Instant::now();
    let eval_loss = if eval_batches.is_empty() || !collect_loss_scalars {
        None
    } else {
        let mut total = None;
        let mut count = 0usize;
        let mut eval_state = None;
        for (batch_index, batch) in eval_batches.into_iter().enumerate() {
            if context
                .config
                .max_eval_batches
                .is_some_and(|max_batches| batch_index >= max_batches)
            {
                break;
            }
            let mut step_state = take_browser_step_state(
                &model,
                &mut eval_state,
                batch.reset_stream_state,
                context.config.tbptt_persist_across_steps,
            );
            let mut losses = Vec::new();
            visit_browser_next_token_chunks(
                &model,
                &batch,
                &mut step_state,
                context.config.tbptt_chunk_size,
                |loss| losses.push(loss),
            );
            let loss = sum_scalar_losses(losses);
            total = Some(match total {
                Some(total) => total + loss,
                None => loss,
            });
            store_browser_step_state(
                &mut eval_state,
                step_state,
                context.config.tbptt_persist_across_steps,
            );
            count = count.saturating_add(1);
        }
        match (total, count) {
            (Some(total), count) if count > 0 => {
                Some(scalar_from_loss_async(total).await? / count as f64)
            }
            _ => None,
        }
    };
    let eval_time_ms = elapsed_ms(eval_started_at);
    info!(
        "browser training eval complete: eval_batches={} eval_loss={:?} eval_time_ms={}",
        eval_batches_len, eval_loss, eval_time_ms,
    );

    let total_time_ms = context.setup_time_ms + elapsed_ms(total_started_at);
    let published_update = if let Some(live) = live_participant.as_ref() {
        if !artifact_publication_decision.should_publish {
            if artifact_publication_decision.requested {
                info!(
                    "browser canonical artifact publication skipped: {}; submitting receipt only",
                    artifact_publication_decision
                        .disabled_reason
                        .unwrap_or("artifact publication disabled")
                );
            } else {
                info!(
                    "browser canonical artifact publication disabled for this training profile; submitting receipt only"
                );
            }
            None
        } else {
            info!(
                "browser canonical artifact publication starting: base_head_synced={} backend={}",
                active_model_schema_hash.is_some(),
                context.backend_label,
            );
            let model_schema_hash = active_model_schema_hash
                .unwrap_or_else(|| dragon_model_schema_hash(&context.config.model_config));
            Some(match compact_trace.as_ref() {
                Some(trace) => {
                    browser_training_compact_update(&context, live, trace, model_schema_hash)?
                }
                None if context.config.model_config.random_scaffold.enabled => {
                    browser_training_mutable_subset_update(
                        &context,
                        live,
                        &model,
                        model_schema_hash,
                    )
                    .await?
                }
                None => BrowserPublishedUpdate {
                    artifact: browser_training_head_artifact(
                        &context,
                        live,
                        model,
                        model_schema_hash,
                    )?,
                    workload_update: None,
                },
            })
        }
    } else {
        None
    };
    info!("browser live participant flush starting");
    let contribution = browser_training_contribution(
        &context,
        BrowserTrainingContributionStats {
            train_batch_count,
            train_example_count,
            train_token_count,
            train_loss_observed: train_loss_count > 0,
            train_loss_mean,
            eval_loss,
            training_time_ms,
            eval_time_ms,
            total_time_ms,
        },
        published_update,
    );
    let live_participant = finish_live_browser_participant(
        context.edge_base_url,
        context.config,
        live_participant,
        contribution,
    )
    .await?;
    if let Some(live) = live_participant.as_ref() {
        info!(
            "browser live participant flush complete: receipt_submission_accepted={} accepted_receipts={} transport={:?} runtime_state={:?}",
            live.receipt_submission_accepted,
            live.accepted_receipt_ids.len(),
            live.transport,
            live.runtime_state,
        );
    } else {
        info!("browser local-only training complete");
    }

    let result = DragonBrowserTrainingResult {
        backend: context.backend_label.into(),
        experiment_kind_label: context.config.experiment_kind.display_name().into(),
        train_batches: train_batch_count,
        train_examples: train_example_count,
        train_tokens: train_token_count,
        train_loss_mean,
        train_loss_observed: train_loss_count > 0,
        eval_examples: eval_records.len(),
        eval_loss,
        setup_time_ms: context.setup_time_ms,
        training_time_ms,
        eval_time_ms,
        total_time_ms,
        tokens_per_second: (training_time_ms > 0)
            .then_some(train_token_count as f64 / (training_time_ms as f64 / 1000.0)),
        live_participant,
    };
    info!(
        "browser training finished: total_time_ms={} tokens_per_second={:?}",
        result.total_time_ms, result.tokens_per_second,
    );
    Ok(result)
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

fn browser_loss_scalar_readback_enabled(backend_kind: BrowserTrainingBackendKind) -> bool {
    match backend_kind {
        BrowserTrainingBackendKind::Cpu => true,
        #[cfg(feature = "wgpu")]
        BrowserTrainingBackendKind::Wgpu => !cfg!(target_arch = "wasm32"),
    }
}

fn validate_browser_training_config(config: &DragonBrowserTrainingConfig) -> Result<()> {
    if config.block_size == 0 {
        bail!("browser training block_size must be > 0");
    }
    if config.batch_size == 0 {
        bail!("browser training batch_size must be > 0");
    }
    if let Some(chunk_size) = config.tbptt_chunk_size {
        if chunk_size == 0 {
            bail!("browser training tbptt_chunk_size must be > 0 when set");
        }
        if chunk_size > config.block_size {
            bail!(
                "browser training tbptt_chunk_size must be <= block_size (got {chunk_size} > {})",
                config.block_size
            );
        }
    }
    if config.tbptt_persist_across_steps && config.tbptt_chunk_size.is_none() {
        bail!("browser training tbptt_persist_across_steps requires tbptt_chunk_size");
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
            let response = ensure_browser_success_response(
                Request::get(&resolved_url).send().await.map_err(|error| {
                    anyhow!("failed to fetch browser shard {resolved_url}: {error}")
                })?,
                &resolved_url,
                "browser shard",
            )
            .await?;
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
        } => generated_nca_records(
            corpus,
            *split,
            block_size,
            GeneratedRecordSelection {
                max_documents: *max_documents,
                record_limit: policy.record_limit,
                selection_key: policy.shard_selection_key.as_deref(),
                training_lease: policy.training_lease.as_ref(),
            },
        )?,
        DragonBrowserTokenSource::GeneratedRuliad {
            corpus,
            split,
            max_documents,
            supervision,
        } => generated_ruliad_records(
            corpus,
            *split,
            block_size,
            *supervision,
            policy.stream_aligned,
            GeneratedRecordSelection {
                max_documents: *max_documents,
                record_limit: policy.record_limit,
                selection_key: policy.shard_selection_key.as_deref(),
                training_lease: policy.training_lease.as_ref(),
            },
        )?,
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

fn trim_preview(body: &str) -> String {
    const LIMIT: usize = 240;
    let trimmed = body.trim();
    let preview = trimmed.chars().take(LIMIT).collect::<String>();
    if preview.len() == trimmed.len() {
        preview
    } else {
        format!("{preview}...")
    }
}

async fn ensure_browser_success_response(
    response: gloo_net::http::Response,
    url: &str,
    label: &str,
) -> Result<gloo_net::http::Response> {
    if response.ok() {
        return Ok(response);
    }
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    bail!(
        "failed to fetch {label} {url}: http {} {}",
        status,
        trim_preview(&body)
    );
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
    let response = ensure_browser_success_response(
        Request::get(&manifest_url)
            .send()
            .await
            .map_err(|error| anyhow!("failed to fetch shard manifest {manifest_url}: {error}"))?,
        &manifest_url,
        "shard manifest",
    )
    .await?;
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
        let response = ensure_browser_success_response(
            Request::get(&shard_url)
                .send()
                .await
                .map_err(|error| anyhow!("failed to fetch browser shard {shard_url}: {error}"))?,
            &shard_url,
            "browser shard",
        )
        .await?;
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
        if let Some(loss_mask) = record.loss_mask.as_ref()
            && loss_mask.len() != block_size
        {
            bail!(
                "token window record {index} loss-mask length {} does not match block_size {}",
                loss_mask.len(),
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
    max_batches: Option<usize>,
    window_id: Option<u64>,
    device: &B::Device,
) -> Result<Vec<TokenWindowBatch<B>>> {
    if records.is_empty() {
        return Ok(Vec::new());
    }
    let plan = crate::stream_batch::plan_windowed_stream_batches(
        records,
        batch_size,
        max_batches,
        window_id,
    )?;
    let mut batches = Vec::with_capacity(plan.len());
    for planned in plan {
        let items = planned
            .record_indices
            .iter()
            .map(|index| &records[*index])
            .collect::<Vec<_>>();
        batches.push(build_batch_from_records::<B>(
            &items,
            planned.reset_stream_state,
            block_size,
            device,
        )?);
    }
    Ok(batches)
}

fn build_batch_from_records<B: Backend>(
    records: &[&TokenWindowRecord],
    reset_stream_state: bool,
    block_size: usize,
    device: &B::Device,
) -> Result<TokenWindowBatch<B>> {
    let mut inputs = Vec::with_capacity(records.len() * block_size);
    let mut targets = Vec::with_capacity(records.len() * block_size);
    let mut loss_mask = records
        .iter()
        .any(|record| record.loss_mask.is_some())
        .then(|| Vec::with_capacity(records.len() * block_size));
    for record in records {
        inputs.extend(record.inputs.iter().copied());
        targets.extend(record.targets.iter().copied());
        if let Some(batch_loss_mask) = loss_mask.as_mut() {
            if let Some(record_loss_mask) = record.loss_mask.as_ref() {
                batch_loss_mask.extend(record_loss_mask);
            } else {
                batch_loss_mask.extend(std::iter::repeat_n(1, block_size));
            }
        }
    }
    let batch_digest = ContentId::derive(&(
        "dragon-token-window-batch-v3-target-mask",
        records,
        block_size,
        reset_stream_state,
    ))?;
    let record_digests = records
        .iter()
        .map(|record| ContentId::derive(*record))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(TokenWindowBatch {
        inputs: Tensor::<B, 2, Int>::from_data(
            TensorData::new(inputs, [records.len(), block_size]),
            device,
        ),
        targets: Tensor::<B, 2, Int>::from_data(
            TensorData::new(targets, [records.len(), block_size]),
            device,
        ),
        loss_mask: loss_mask.map(|loss_mask| {
            Tensor::<B, 2, Int>::from_data(
                TensorData::new(loss_mask, [records.len(), block_size]),
                device,
            )
        }),
        token_count: records.len() * block_size,
        batch_digest,
        record_digests,
        reset_stream_state,
    })
}

fn take_browser_step_state<B: Backend>(
    model: &DragonModel<B>,
    state_slot: &mut Option<ModelState<B>>,
    reset_stream_state: bool,
    persist_across_steps: bool,
) -> ModelState<B> {
    if !persist_across_steps {
        return model.init_state_ephemeral();
    }
    if reset_stream_state {
        *state_slot = None;
    }
    state_slot.take().unwrap_or_else(|| model.init_state())
}

fn store_browser_step_state<B: Backend>(
    state_slot: &mut Option<ModelState<B>>,
    mut state: ModelState<B>,
    persist_across_steps: bool,
) {
    if !persist_across_steps {
        return;
    }
    state.detach_in_place();
    *state_slot = Some(state);
}

fn visit_browser_next_token_chunks<B: Backend>(
    model: &DragonModel<B>,
    batch: &TokenWindowBatch<B>,
    state: &mut ModelState<B>,
    tbptt_chunk_size: Option<usize>,
    mut visit: impl FnMut(Tensor<B, 1>),
) {
    let [batch_size, block_size] = batch.inputs.shape().dims();
    let chunk_size = tbptt_chunk_size
        .filter(|chunk_size| *chunk_size > 0)
        .unwrap_or(block_size.max(1))
        .min(block_size.max(1));
    for start in (0..block_size).step_by(chunk_size) {
        let end = (start + chunk_size).min(block_size);
        let inputs = batch.inputs.clone().slice([0..batch_size, start..end]);
        let targets = batch.targets.clone().slice([0..batch_size, start..end]);
        let loss_mask = batch
            .loss_mask
            .clone()
            .map(|mask| mask.slice([0..batch_size, start..end]));
        let hidden = model.forward_hidden_with_state(inputs, state);
        let chunk_weight = (end - start) as f32 / block_size.max(1) as f32;
        let loss = if let Some(loss_mask) = loss_mask {
            masked_token_mean(
                model.language_token_losses_from_hidden(hidden, targets),
                Some(loss_mask),
            )
        } else {
            model.language_loss_from_hidden(hidden, targets)
        };
        visit(loss.mul_scalar(chunk_weight));
        if end < block_size {
            state.detach_in_place();
        }
    }
}

fn sum_scalar_losses<B: Backend>(losses: Vec<Tensor<B, 1>>) -> Tensor<B, 1> {
    Tensor::cat(losses, 0).sum().reshape([1])
}

async fn scalar_from_loss_async<B: Backend>(loss: Tensor<B, 1>) -> Result<f64> {
    loss.into_scalar_async()
        .await
        .map(|scalar| scalar.elem::<f64>())
        .map_err(|error| anyhow!("failed to read browser loss scalar: {error}"))
}

async fn scalar_values_from_loss_tensors_async<B: Backend>(
    losses: Vec<Tensor<B, 1>>,
) -> Result<Vec<f64>> {
    if losses.is_empty() {
        return Ok(Vec::new());
    }
    Tensor::cat(losses, 0)
        .into_data_async()
        .await
        .map_err(|error| anyhow!("failed to read browser population losses: {error}"))?
        .convert::<f32>()
        .into_vec::<f32>()
        .map(|values| values.into_iter().map(f64::from).collect())
        .map_err(|error| anyhow!("failed to decode browser population losses: {error}"))
}

struct BrowserTrainingContributionStats {
    train_batch_count: usize,
    train_example_count: usize,
    train_token_count: usize,
    train_loss_observed: bool,
    train_loss_mean: f64,
    eval_loss: Option<f64>,
    training_time_ms: u64,
    eval_time_ms: u64,
    total_time_ms: u64,
}

struct BrowserPublishedUpdate {
    artifact: WorkloadTrainingArtifact,
    workload_update: Option<WorkloadUpdateEnvelope>,
}

fn materialize_browser_training_artifact(
    descriptor: burn_p2p::ArtifactDescriptor,
    bytes: Vec<u8>,
) -> Result<WorkloadTrainingArtifact> {
    let mut chunks = Vec::with_capacity(descriptor.chunks.len());
    for chunk in &descriptor.chunks {
        let start = usize::try_from(chunk.offset_bytes)
            .map_err(|_| anyhow!("browser artifact chunk offset exceeded local usize"))?;
        let len = usize::try_from(chunk.length_bytes)
            .map_err(|_| anyhow!("browser artifact chunk length exceeded local usize"))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| anyhow!("browser artifact chunk range overflowed"))?;
        let chunk_bytes = bytes
            .get(start..end)
            .ok_or_else(|| anyhow!("browser artifact chunk range exceeded artifact bytes"))?
            .to_vec();
        chunks.push(WorkloadTrainingArtifactChunk {
            chunk: chunk.clone(),
            bytes: chunk_bytes,
        });
    }
    Ok(WorkloadTrainingArtifact { descriptor, chunks })
}

fn browser_training_head_artifact<B>(
    context: &BrowserTrainingRunContext<'_>,
    live: &LiveBrowserParticipantHandle,
    model: DragonModel<B>,
    model_schema_hash: ContentId,
) -> Result<WorkloadTrainingArtifact>
where
    B: Backend,
    DragonModel<B>: Module<B>,
{
    let peer_id = live
        .session_runtime
        .runtime
        .storage
        .stored_certificate_peer_id
        .as_ref()
        .ok_or_else(|| {
            anyhow!("browser canonical training requires an enrolled node certificate")
        })?;
    let base_head_id = live
        .session_runtime
        .runtime
        .storage
        .last_head_id
        .clone()
        .ok_or_else(|| anyhow!("browser canonical training requires a synced active head"))?;
    let window_id = context
        .config
        .training_lease
        .as_ref()
        .map(|lease| lease.window_id.0)
        .unwrap_or(0);
    let head_id = HeadId::new(format!(
        "{}-{}-browser-window-{}-{}",
        context.config.experiment_kind.workload_slug(),
        peer_id.as_str(),
        window_id,
        Utc::now().timestamp_micros()
    ));
    let record_format = BrowserBurnRecordBytesFormat::NamedMpk;
    let record_precision = BrowserBurnRecordPrecision::Half;
    let bytes = encode_browser_record_bytes::<B, _>(model, record_format, record_precision)?;
    let descriptor = build_artifact_descriptor_from_bytes(
        &ArtifactBuildSpec::new(
            ArtifactKind::FullHead,
            browser_record_precision_descriptor(record_precision),
            model_schema_hash,
            browser_record_format_name(record_format),
        )
        .with_head(head_id)
        .with_base_head(base_head_id),
        &bytes,
        ChunkingScheme::new(1024 * 1024)?,
    )
    .map_err(|error| anyhow!("failed to materialize browser training artifact: {error}"))?;
    materialize_browser_training_artifact(descriptor, bytes)
}

fn browser_training_compact_update(
    context: &BrowserTrainingRunContext<'_>,
    live: &LiveBrowserParticipantHandle,
    trace: &BrowserCompactUpdateTrace,
    model_schema_hash: ContentId,
) -> Result<BrowserPublishedUpdate> {
    let contract = live.revision_contract.as_ref().ok_or_else(|| {
        anyhow!("browser compact update publication requires a signed revision contract")
    })?;
    let lease = context.config.training_lease.as_ref().ok_or_else(|| {
        anyhow!("browser compact update publication requires an active training lease")
    })?;
    let base_head_id = live
        .session_runtime
        .runtime
        .storage
        .last_head_id
        .clone()
        .ok_or_else(|| anyhow!("browser compact update publication requires a synced base head"))?;
    if model_schema_hash != contract.training.model_schema_hash {
        bail!("browser compact update model schema does not match its signed contract");
    }
    let burn_p2p::UpdateCodec::SeededFitness {
        population,
        rank,
        seed,
        ..
    } = &contract.training.update_codec
    else {
        bail!("browser compact update requires the signed seeded-fitness codec");
    };
    let payload = CompactUpdatePayload {
        version: COMPACT_UPDATE_PAYLOAD_VERSION,
        training_contract_id: contract.training_contract_id.clone(),
        model_schema_hash: model_schema_hash.clone(),
        parameter_catalog_hash: trace.parameter_catalog_hash.clone(),
        parameter_count: trace.parameter_count,
        body: CompactUpdateBody::SeededFitness {
            population: *population,
            rank: *rank,
            seed: *seed,
            perturbation_generator_hash: trace.perturbation_generator_hash.clone(),
            optimizer_update_hash: trace.optimizer_update_hash.clone(),
            generations: trace.generations.clone(),
        },
    };
    let bytes = burn_p2p_workload::encode_compact_update(
        &payload,
        &contract.training_contract_id,
        &contract.training,
    )
    .context("encode browser compact update")?;
    let descriptor = build_artifact_descriptor_from_bytes(
        &ArtifactBuildSpec::new(
            ArtifactKind::DeltaPack,
            Precision::Custom("seeded-fitness".into()),
            model_schema_hash,
            "burn-p2p-compact-update-cbor-v1",
        )
        .with_base_head(base_head_id.clone()),
        &bytes,
        ChunkingScheme::new(256 * 1024)?,
    )
    .map_err(|error| anyhow!("failed to materialize browser compact update: {error}"))?;
    let workload_update = WorkloadUpdateEnvelope {
        training_contract_id: contract.training_contract_id.clone(),
        revision_id: contract.revision.revision_id.clone(),
        base_head_id,
        window_id: lease.window_id,
        lease_id: lease.lease_id.clone(),
        codec: contract.training.update_codec.clone(),
        artifact: descriptor.clone(),
        decoded_tensor_digest: None,
        claimed_norm_stats: None,
        claimed_feature_sketch: None,
    };
    workload_update
        .validate_against(&contract.training_contract_id, &contract.training)
        .context("validate browser compact update envelope")?;
    Ok(BrowserPublishedUpdate {
        artifact: materialize_browser_training_artifact(descriptor, bytes)?,
        workload_update: Some(workload_update),
    })
}

async fn browser_training_mutable_subset_update<B: Backend>(
    context: &BrowserTrainingRunContext<'_>,
    live: &LiveBrowserParticipantHandle,
    model: &DragonModel<B>,
    model_schema_hash: ContentId,
) -> Result<BrowserPublishedUpdate>
where
    DragonModel<B>: Module<B>,
{
    let contract = live.revision_contract.as_ref().ok_or_else(|| {
        anyhow!("browser mutable-subset publication requires a signed revision contract")
    })?;
    let lease = context.config.training_lease.as_ref().ok_or_else(|| {
        anyhow!("browser mutable-subset publication requires an active training lease")
    })?;
    let base_head_id = live
        .session_runtime
        .runtime
        .storage
        .last_head_id
        .clone()
        .ok_or_else(|| anyhow!("browser mutable-subset publication requires a synced base head"))?;
    ensure!(
        model_schema_hash == contract.training.model_schema_hash,
        "browser mutable-subset model schema does not match signed contract"
    );
    let burn_p2p::UpdateCodec::MutableSubsetParameters {
        parameter_catalog_hash,
        parameter_count,
        encoding,
    } = &contract.training.update_codec
    else {
        bail!("browser random-scaffold AdamW requires the mutable-subset update codec");
    };
    let scaffold = browser_random_scaffold_contract(model, model_schema_hash.clone())?
        .ok_or_else(|| anyhow!("browser mutable-subset update requires random-scaffold mode"))?;
    ensure!(
        parameter_catalog_hash == &scaffold.catalog.catalog_id()?
            && *parameter_count == scaffold.catalog.parameter_count()?,
        "browser mutable-subset catalog does not match signed contract"
    );
    let values = flatten_browser_random_scaffold_mutable(model, &scaffold).await?;
    let encoded = CompactScalarVector::encode(&values, *encoding)?;
    let decoded = encoded.decode()?;
    let decoded_tensor_digest = browser_random_scaffold_tensor_digest_from_mutable(
        &context.config.model_config,
        model_schema_hash.clone(),
        &decoded,
    )?;
    let payload = CompactUpdatePayload {
        version: COMPACT_UPDATE_PAYLOAD_VERSION,
        training_contract_id: contract.training_contract_id.clone(),
        model_schema_hash: model_schema_hash.clone(),
        parameter_catalog_hash: scaffold.catalog.catalog_id()?,
        parameter_count: scaffold.catalog.parameter_count()?,
        body: CompactUpdateBody::MutableSubsetParameters { values: encoded },
    };
    let bytes = burn_p2p_workload::encode_compact_update(
        &payload,
        &contract.training_contract_id,
        &contract.training,
    )
    .context("encode browser mutable-subset update")?;
    let precision = match encoding {
        CompactScalarEncoding::Fp32 => "mutable-subset-fp32",
        CompactScalarEncoding::SymmetricInt8 => "mutable-subset-int8",
        CompactScalarEncoding::SymmetricInt16 => "mutable-subset-int16",
    };
    let descriptor = build_artifact_descriptor_from_bytes(
        &ArtifactBuildSpec::new(
            ArtifactKind::DeltaPack,
            Precision::Custom(precision.into()),
            model_schema_hash,
            "burn-p2p-compact-update-cbor-v1",
        )
        .with_base_head(base_head_id.clone()),
        &bytes,
        ChunkingScheme::new(256 * 1024)?,
    )
    .map_err(|error| anyhow!("failed to materialize browser mutable-subset update: {error}"))?;
    let workload_update = WorkloadUpdateEnvelope {
        training_contract_id: contract.training_contract_id.clone(),
        revision_id: contract.revision.revision_id.clone(),
        base_head_id,
        window_id: lease.window_id,
        lease_id: lease.lease_id.clone(),
        codec: contract.training.update_codec.clone(),
        artifact: descriptor.clone(),
        decoded_tensor_digest: Some(decoded_tensor_digest),
        claimed_norm_stats: None,
        claimed_feature_sketch: None,
    };
    workload_update
        .validate_against(&contract.training_contract_id, &contract.training)
        .context("validate browser mutable-subset update envelope")?;
    Ok(BrowserPublishedUpdate {
        artifact: materialize_browser_training_artifact(descriptor, bytes)?,
        workload_update: Some(workload_update),
    })
}

fn browser_training_contribution(
    context: &BrowserTrainingRunContext<'_>,
    stats: BrowserTrainingContributionStats,
    published_update: Option<BrowserPublishedUpdate>,
) -> WorkloadTrainingContribution {
    let now = Utc::now();
    let fallback_artifact_id = ArtifactId::new(format!(
        "browser-dragon-artifact-{}-{}-{}-{}",
        context.config.experiment_kind.workload_slug(),
        context.config.block_size,
        stats.train_token_count,
        now.timestamp_micros()
    ));
    let requested_canonical_update = context
        .config
        .live_participant
        .as_ref()
        .is_some_and(|live| live.publish_canonical_update);
    let artifact_publication_decision = browser_canonical_artifact_publication_decision(
        requested_canonical_update,
        context.backend_kind,
        browser_uses_compact_update(context.config),
    );
    let mut metadata = BTreeMap::from([
        ("contribution_kind".into(), "browser-local-window".into()),
        ("backend".into(), context.backend_label.into()),
        (
            "experiment_kind".into(),
            context.config.experiment_kind.workload_slug().into(),
        ),
        (
            "publish_canonical_update_requested".into(),
            artifact_publication_decision.requested.to_string(),
        ),
        (
            "publish_canonical_update".into(),
            artifact_publication_decision.should_publish.to_string(),
        ),
        (
            "load_active_head_artifact".into(),
            context
                .config
                .live_participant
                .as_ref()
                .is_none_or(|live| live.load_active_head_artifact)
                .to_string(),
        ),
        ("block_size".into(), context.config.block_size.to_string()),
        ("receipt_payload_version".into(), "browser-window-v1".into()),
    ]);
    metadata.insert(
        "train_loss_observed".into(),
        stats.train_loss_observed.to_string(),
    );
    if stats.train_loss_observed {
        metadata.insert(
            "train_loss_mean".into(),
            format!("{:.8}", stats.train_loss_mean),
        );
    }
    if let Some(eval_loss) = stats.eval_loss {
        metadata.insert("eval_loss".into(), format!("{eval_loss:.8}"));
    }
    if let Some(reason) = artifact_publication_decision.disabled_reason {
        metadata.insert("artifact_publication_disabled_reason".into(), reason.into());
    }
    let artifact_id = published_update
        .as_ref()
        .map(|update| update.artifact.descriptor.artifact_id.clone())
        .unwrap_or(fallback_artifact_id);
    let base_head_id = published_update
        .as_ref()
        .and_then(|update| update.artifact.descriptor.base_head_id.clone());
    let (published_artifact, workload_update) = published_update
        .map(|update| (Some(update.artifact), update.workload_update))
        .unwrap_or((None, None));

    WorkloadTrainingContribution {
        artifact_id,
        completed_batches: stats.train_batch_count as u64,
        completed_examples: stats.train_example_count as u64,
        completed_tokens: stats.train_token_count as u64,
        training_time_ms: stats.training_time_ms,
        eval_time_ms: stats.eval_time_ms,
        total_time_ms: stats.total_time_ms,
        artifact_published: false,
        base_head_id,
        published_artifact,
        workload_update,
        metadata,
    }
}

fn live_browser_training_requested_scopes(
    live: &DragonBrowserLiveParticipantConfig,
) -> BTreeSet<ExperimentScope> {
    let experiment_id = ExperimentId::new(live.experiment_id.clone());
    BTreeSet::from([
        ExperimentScope::Connect,
        ExperimentScope::Discover,
        ExperimentScope::Train {
            experiment_id: experiment_id.clone(),
        },
        ExperimentScope::Archive { experiment_id },
    ])
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
    let directory_entry = snapshot
        .directory
        .entries
        .iter()
        .find(|entry| {
            entry.experiment_id.as_str() == live.experiment_id
                && entry.current_revision_id.as_str() == live.revision_id
        })
        .ok_or_else(|| anyhow!("browser training experiment revision is absent from the edge"))?;
    ensure!(
        matches!(
            &directory_entry.training_protocol,
            TrainingProtocol::ArtifactWindows
        ),
        "browser training does not implement the selected DiLoCo revision; participate as an observer or verifier"
    );
    let revision_contract = resolve_browser_revision_contract(&snapshot, config, live)?;
    let bootstrap_head = revision_contract
        .as_ref()
        .map(|contract| browser_bootstrap_head(&snapshot, contract))
        .transpose()?;
    let requested_scopes = live_browser_training_requested_scopes(live);
    let _ = browser_github_enrollment_config(
        &snapshot,
        release_manifest,
        requested_scopes.clone(),
        900,
    )?;
    let session = match preloaded_session {
        Some(session) => session.clone(),
        None => {
            load_or_enroll_browser_session(edge_base_url, release_manifest, requested_scopes, 900)
                .await?
        }
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
            browser_runtime_role_label(&capability.recommended_role)
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
            transport: browser_trainer_transport_policy(),
            selected_experiment: Some(ExperimentId::new(live.experiment_id.clone())),
            selected_revision: Some(RevisionId::new(live.revision_id.clone())),
            capability,
            include_leaderboard: true,
            enable_direct_swarm: true,
            sync_active_head_artifact: live.load_active_head_artifact
                || live.publish_canonical_update,
            bootstrap_head,
        },
        session,
    )
    .await
    .map_err(map_browser_session_runtime_error)?;

    Ok(Some(LiveBrowserParticipantHandle {
        session_runtime,
        revision_contract,
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

fn browser_bootstrap_head(
    snapshot: &burn_p2p::BrowserEdgeSnapshot,
    contract: &burn_p2p::RevisionContractBundle,
) -> Result<BrowserBootstrapHead> {
    let genesis = &contract.genesis.payload.payload;
    let artifact = genesis.artifact.clone();
    let head_id = artifact
        .head_id
        .clone()
        .ok_or_else(|| anyhow!("signed browser genesis artifact has no head id"))?;
    let directory_entry = snapshot
        .directory
        .entries
        .iter()
        .find(|entry| {
            entry.experiment_id == contract.revision.experiment_id
                && entry.current_revision_id == contract.revision.revision_id
        })
        .ok_or_else(|| {
            anyhow!("signed browser genesis has no matching browser directory experiment revision")
        })?;

    let head = snapshot
        .heads
        .iter()
        .find(|head| head.head_id == head_id)
        .map(|head| {
            if head.study_id != directory_entry.study_id
                || head.experiment_id != contract.revision.experiment_id
                || head.revision_id != contract.revision.revision_id
                || head.artifact_id != artifact.artifact_id
            {
                bail!("edge genesis head metadata disagrees with the signed genesis artifact");
            }
            Ok(head.clone())
        })
        .transpose()?
        .unwrap_or_else(|| HeadDescriptor {
            head_id,
            study_id: directory_entry.study_id.clone(),
            experiment_id: contract.revision.experiment_id.clone(),
            revision_id: contract.revision.revision_id.clone(),
            artifact_id: artifact.artifact_id.clone(),
            parent_head_id: None,
            global_step: 0,
            created_at: genesis.created_at,
            metrics: BTreeMap::new(),
        });

    Ok(BrowserBootstrapHead { head, artifact })
}

fn validate_browser_active_head_descriptor(
    config: &DragonBrowserTrainingConfig,
    contract: Option<&burn_p2p::RevisionContractBundle>,
    head_id: &HeadId,
    descriptor: &burn_p2p::ArtifactDescriptor,
) -> Result<()> {
    ensure!(
        descriptor.kind == ArtifactKind::FullHead,
        "browser active training head {} must use a full-head artifact, found {:?}",
        head_id.as_str(),
        descriptor.kind,
    );
    ensure!(
        descriptor.head_id.as_ref() == Some(head_id),
        "browser active artifact {} is not bound to head {}",
        descriptor.artifact_id.as_str(),
        head_id.as_str(),
    );
    let expected_model_schema = dragon_model_schema_hash(&config.model_config);
    ensure!(
        descriptor.model_schema_hash == expected_model_schema,
        "browser active artifact model schema {} does not match configured Dragon schema {}",
        descriptor.model_schema_hash.as_str(),
        expected_model_schema.as_str(),
    );
    if let Some(contract) = contract {
        ensure!(
            descriptor.model_schema_hash == contract.training.model_schema_hash,
            "browser active artifact does not match the signed revision model schema",
        );
        let genesis = &contract.genesis.payload.payload.artifact;
        if genesis.head_id.as_ref() == Some(head_id) {
            ensure!(
                descriptor == genesis,
                "browser active head {} claims the signed genesis identity but its descriptor differs",
                head_id.as_str(),
            );
        }
    }
    Ok(())
}

fn resolve_browser_revision_contract(
    snapshot: &burn_p2p::BrowserEdgeSnapshot,
    config: &DragonBrowserTrainingConfig,
    live: &DragonBrowserLiveParticipantConfig,
) -> Result<Option<burn_p2p::RevisionContractBundle>> {
    let embedded = live.revision_contract.as_ref();
    let published = snapshot
        .revision_contracts
        .iter()
        .find(|contract| contract.revision.revision_id.as_str() == live.revision_id);
    if let (Some(embedded), Some(published)) = (embedded, published)
        && embedded != published
    {
        bail!("embedded and edge-published browser revision contracts disagree");
    }
    let contract = embedded.or(published).cloned();
    let Some(contract) = contract else {
        if live.publish_canonical_update {
            bail!(
                "browser canonical training requires an authority-signed revision contract from the edge"
            );
        }
        return Ok(None);
    };
    let trust_bundle = snapshot
        .trust_bundle
        .as_ref()
        .ok_or_else(|| anyhow!("browser canonical training requires the edge trust bundle"))?;
    burn_p2p::verify_revision_contract_with_trust_bundle(trust_bundle, &contract).map_err(
        |error| anyhow!("browser revision contract signature verification failed: {error}"),
    )?;
    if contract.revision.experiment_id.as_str() != live.experiment_id
        || contract.revision.revision_id.as_str() != live.revision_id
        || contract.revision.workload_id.as_str() != live.workload_id
    {
        bail!("browser revision contract does not match the selected experiment revision");
    }
    if contract.training.model_schema_hash != dragon_model_schema_hash(&config.model_config) {
        bail!("browser model config does not match the signed model schema");
    }
    validate_browser_optimizer_contract(config, &contract)?;
    let authorized_execution = contract
        .training
        .extensions
        .get(DRAGON_BROWSER_EXECUTION_CONTRACT_EXTENSION)
        .ok_or_else(|| {
            anyhow!("signed revision contract does not authorize a browser execution contract")
        })?;
    let runtime_execution = browser_runtime_execution_contract_hash(config)?;
    if authorized_execution != &runtime_execution {
        bail!(
            "browser runtime execution contract {} does not match signed authorization {}",
            runtime_execution.as_str(),
            authorized_execution.as_str()
        );
    }
    Ok(Some(contract))
}

fn validate_browser_optimizer_contract(
    config: &DragonBrowserTrainingConfig,
    contract: &burn_p2p::RevisionContractBundle,
) -> Result<()> {
    if matches!(config.optimizer, DragonBrowserOptimizerConfig::Adamw)
        && config.model_config.random_scaffold.enabled
    {
        let burn_p2p::UpdateCodec::MutableSubsetParameters {
            parameter_catalog_hash,
            parameter_count,
            ..
        } = &contract.training.update_codec
        else {
            bail!("browser random-scaffold AdamW requires the mutable-subset update codec");
        };
        let device = burn::tensor::Device::<BrowserCpuEvalBackend>::default();
        let model = DragonModel::<BrowserCpuEvalBackend>::new(config.model_config.clone(), &device);
        let scaffold =
            browser_random_scaffold_contract(&model, contract.training.model_schema_hash.clone())?
                .ok_or_else(|| {
                    anyhow!("browser random-scaffold model did not expose its contract")
                })?;
        ensure!(
            parameter_catalog_hash == &scaffold.catalog.catalog_id()?
                && *parameter_count == scaffold.catalog.parameter_count()?,
            "browser random-scaffold mutable catalog does not match signed revision"
        );
        return Ok(());
    }

    let update_codec = config
        .optimizer
        .update_codec()
        .map_err(anyhow::Error::msg)?;
    ensure!(
        contract.training.update_codec == update_codec,
        "browser optimizer codec {:?} does not match signed revision codec {:?}",
        update_codec,
        contract.training.update_codec
    );
    Ok(())
}

async fn finish_live_browser_participant(
    edge_base_url: &str,
    config: &DragonBrowserTrainingConfig,
    handle: Option<&mut LiveBrowserParticipantHandle>,
    contribution: WorkloadTrainingContribution,
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
    if handle
        .session_runtime
        .refresh_session_if_expiring_before(
            Utc::now() + Duration::seconds(BROWSER_LIVE_SESSION_REFRESH_GRACE_SECS),
        )
        .await?
    {
        info!("browser live participant session refreshed before receipt flush");
    }
    let outcome = handle
        .session_runtime
        .run_training_plan(BrowserTrainingPlan {
            study_id: StudyId::new(assignment.study_id.as_str().to_owned()),
            experiment_id: ExperimentId::new(assignment.experiment_id.as_str().to_owned()),
            revision_id: RevisionId::new(assignment.revision_id.as_str().to_owned()),
            workload_id: WorkloadId::new("browser-dragon-training"),
            budget: handle.training_budget.clone(),
            lease: config.training_lease.clone(),
            contribution: Some(contribution),
        })
        .await
        .map_err(|error| match error {
            BrowserSessionRuntimeError::Worker(message) => {
                if is_probable_trainer_fit_failure(&message) {
                    let capability_decision = apply_browser_downgrade_state(
                        edge_base_url,
                        config,
                        config.execution_backend.backend_label(),
                        decide_browser_capability(
                            Some(config),
                            &detect_browser_host_capabilities(),
                        ),
                    );
                    let _ = persist_browser_downgrade(
                        edge_base_url,
                        config,
                        config.execution_backend.backend_label(),
                        &capability_decision,
                        &message,
                        "browser-worker-runtime",
                    );
                }
                anyhow!("browser worker training failed: {message}")
            }
            other => anyhow!(other),
        })?;
    Ok(Some(DragonBrowserLiveParticipantResult {
        receipt_submission_accepted: outcome.receipt_submission_accepted,
        receipt_submission_deferred: outcome.receipt_submission_deferred,
        pending_receipt_count: outcome.pending_receipt_count,
        receipt_submission_error: outcome.receipt_submission_error,
        accepted_receipt_ids: outcome.accepted_receipt_ids,
        emitted_receipt_id: outcome.emitted_receipt_id,
        artifact_published: outcome.artifact_published,
        update_announced: outcome.update_announced,
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
        BrowserSessionRuntimeError::InvalidBootstrapHead(message) => {
            anyhow!("browser signed genesis bootstrap is invalid: {message}")
        }
    }
}

fn elapsed_ms(started_at: Instant) -> u64 {
    started_at.elapsed().as_millis() as u64
}

#[cfg(all(test, target_arch = "wasm32", feature = "wasm-peer"))]
mod tests {
    use super::*;
    use crate::config::{DragonBrowserLiveParticipantConfig, DragonBrowserTrainingObjectiveConfig};
    use burn_dragon_core::{DragonConfig, LanguageHeadConfig};
    use burn_dragon_universality::{
        NcaCorpusConfig, NcaFamilyConfig, NcaFamilyKind, NcaSerializationConfig,
        NcaTokenizationConfig, RuliadCorpusConfig, RuliadFormalTaskMixConfig,
        RuliadSerializationConfig, RuliadSourceSelectionConfig, RuliadTokenizationConfig,
        UsizeRangeConfig,
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
    fn browser_batches_preserve_native_shard_loss_masks() {
        let device = burn::tensor::Device::<NdArray<f32>>::default();
        let masked = TokenWindowRecord {
            inputs: vec![1, 2, 3, 4],
            targets: vec![2, 3, 4, 5],
            loss_mask: Some(vec![1, 1, 0, 0]),
            reset_stream_state: true,
            ..TokenWindowRecord::default()
        };
        let legacy = TokenWindowRecord {
            inputs: vec![5, 6, 7, 8],
            targets: vec![6, 7, 8, 9],
            reset_stream_state: true,
            ..TokenWindowRecord::default()
        };
        let batch = build_batch_from_records::<NdArray<f32>>(&[&masked, &legacy], true, 4, &device)
            .expect("browser token batch");

        assert_eq!(
            batch
                .loss_mask
                .expect("mixed masked and legacy browser rows should emit a mask")
                .into_data()
                .into_vec::<i64>()
                .expect("browser mask values"),
            vec![1, 1, 0, 0, 1, 1, 1, 1]
        );
    }

    #[wasm_bindgen_test]
    fn browser_records_reject_misaligned_loss_masks() {
        let records = [TokenWindowRecord {
            inputs: vec![1, 2, 3, 4],
            targets: vec![2, 3, 4, 5],
            loss_mask: Some(vec![1, 0]),
            reset_stream_state: true,
            ..TokenWindowRecord::default()
        }];

        assert!(
            validate_token_records(&records, 4)
                .expect_err("misaligned browser loss mask must fail")
                .to_string()
                .contains("loss-mask length")
        );
    }

    #[wasm_bindgen_test]
    fn canonical_artifact_publication_is_disabled_when_not_requested() {
        let decision = browser_canonical_artifact_publication_decision_for_platform(
            false,
            BrowserTrainingBackendKind::Cpu,
            false,
            true,
        );

        assert!(!decision.requested);
        assert!(!decision.should_publish);
        assert_eq!(decision.disabled_reason, None);
    }

    #[wasm_bindgen_test]
    fn canonical_artifact_publication_allows_cpu_on_wasm() {
        let decision = browser_canonical_artifact_publication_decision_for_platform(
            true,
            BrowserTrainingBackendKind::Cpu,
            false,
            true,
        );

        assert!(decision.requested);
        assert!(decision.should_publish);
        assert_eq!(decision.disabled_reason, None);
    }

    #[cfg(feature = "wgpu")]
    #[wasm_bindgen_test]
    fn canonical_artifact_publication_skips_wasm_webgpu_sync_recording() {
        let decision = browser_canonical_artifact_publication_decision_for_platform(
            true,
            BrowserTrainingBackendKind::Wgpu,
            false,
            true,
        );

        assert!(decision.requested);
        assert!(!decision.should_publish);
        assert!(
            decision
                .disabled_reason
                .expect("disabled reason")
                .contains("synchronous tensor reads")
        );
    }

    #[cfg(feature = "wgpu")]
    #[wasm_bindgen_test]
    fn canonical_artifact_publication_allows_native_webgpu() {
        let decision = browser_canonical_artifact_publication_decision_for_platform(
            true,
            BrowserTrainingBackendKind::Wgpu,
            false,
            false,
        );

        assert!(decision.requested);
        assert!(decision.should_publish);
        assert_eq!(decision.disabled_reason, None);
    }

    #[cfg(feature = "wgpu")]
    #[wasm_bindgen_test]
    fn browser_training_contribution_marks_wasm_webgpu_publication_skipped() {
        let mut config = sample_browser_training_config();
        config.live_participant = Some(DragonBrowserLiveParticipantConfig {
            principal_id: Some("browser-principal".into()),
            study_id: "dragon-study".into(),
            experiment_id: "dragon-experiment".into(),
            revision_id: "dragon-revision".into(),
            workload_id: "dragon-workload".into(),
            publish_canonical_update: true,
            load_active_head_artifact: true,
            revision_contract: None,
        });
        let context = BrowserTrainingRunContext {
            edge_base_url: "https://edge.example.invalid",
            config: &config,
            backend_label: "burn-webgpu-wasm",
            backend_kind: BrowserTrainingBackendKind::Wgpu,
            setup_time_ms: 0,
            live_session_principal_id: Some("browser-principal".into()),
        };

        let contribution = browser_training_contribution(
            &context,
            BrowserTrainingContributionStats {
                train_batch_count: 1,
                train_example_count: 1,
                train_token_count: 8,
                train_loss_observed: false,
                train_loss_mean: 0.0,
                eval_loss: None,
                training_time_ms: 10,
                eval_time_ms: 0,
                total_time_ms: 10,
            },
            None,
        );

        assert_eq!(
            contribution
                .metadata
                .get("publish_canonical_update_requested"),
            Some(&"true".into())
        );
        assert_eq!(
            contribution.metadata.get("publish_canonical_update"),
            Some(&"false".into())
        );
        assert!(
            contribution
                .metadata
                .get("artifact_publication_disabled_reason")
                .expect("disabled reason")
                .contains("synchronous tensor reads")
        );
        assert!(contribution.published_artifact.is_none());
    }

    #[wasm_bindgen_test]
    fn browser_live_shard_selection_prefers_authenticated_session_principal() {
        let mut config = sample_browser_training_config();
        config.live_participant = Some(DragonBrowserLiveParticipantConfig {
            principal_id: Some("configured-live-principal".into()),
            study_id: "dragon-study".into(),
            experiment_id: "dragon-experiment".into(),
            revision_id: "dragon-revision".into(),
            workload_id: "dragon-workload".into(),
            publish_canonical_update: true,
            load_active_head_artifact: true,
            revision_contract: None,
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
            publish_canonical_update: true,
            load_active_head_artifact: true,
            revision_contract: None,
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
            training_objective: DragonBrowserTrainingObjectiveConfig::default(),
            optimizer: Default::default(),
            execution_backend,
            block_size: 8,
            tbptt_chunk_size: None,
            tbptt_persist_across_steps: false,
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
    async fn browser_training_smoke_generated_ruliad() {
        use burn_dragon_universality::ruliad::{
            RuliadTokenSupervisionConfig, RuliadTokenSupervisionMode,
        };

        #[cfg(feature = "wgpu")]
        let execution_backend = DragonBrowserExecutionBackend::Wgpu;
        #[cfg(not(feature = "wgpu"))]
        let execution_backend = DragonBrowserExecutionBackend::Cpu;
        let supervision = RuliadTokenSupervisionConfig {
            mode: RuliadTokenSupervisionMode::TraceAndAnswer,
            mask_high_entropy_spans: true,
            ..RuliadTokenSupervisionConfig::default()
        };
        let config = DragonBrowserTrainingConfig {
            experiment_kind: crate::config::DragonExperimentKind::RuliadPretraining,
            model_config: tiny_model_config(272),
            training_objective: DragonBrowserTrainingObjectiveConfig::default(),
            optimizer: Default::default(),
            execution_backend,
            block_size: 64,
            tbptt_chunk_size: Some(64),
            tbptt_persist_across_steps: true,
            learning_rate: 1.0e-3,
            weight_decay: 0.0,
            batch_size: 2,
            max_train_batches: Some(2),
            max_eval_batches: Some(2),
            capability_policy: Default::default(),
            training_lease: None,
            train_source: DragonBrowserTokenSource::GeneratedRuliad {
                corpus: Box::new(tiny_ruliad_corpus_config()),
                split: DragonBrowserDatasetSplit::Train,
                max_documents: Some(1),
                supervision,
            },
            eval_source: Some(DragonBrowserTokenSource::GeneratedRuliad {
                corpus: Box::new(tiny_ruliad_corpus_config()),
                split: DragonBrowserDatasetSplit::Validation,
                max_documents: Some(1),
                supervision,
            }),
            live_participant: None,
        };
        let result = run_browser_training_with_release_manifest(
            "https://example.invalid",
            &config,
            &dummy_release_manifest(),
        )
        .await
        .expect("generated Ruliad browser training should succeed");
        let expected_backend = match execution_backend.backend_label() {
            "wgpu" => "burn-webgpu-wasm",
            _ => "burn-ndarray-wasm",
        };
        assert_eq!(result.backend, expected_backend);
        assert_eq!(result.experiment_kind_label, "Ruliad pre-training");
        assert_eq!(result.train_batches, 2);
        assert_eq!(result.train_examples, 4);
        assert!(result.train_loss_observed);
        assert!(result.train_loss_mean.is_finite());
        assert_eq!(result.eval_examples, 4);
        assert!(result.eval_loss.is_some_and(f64::is_finite));
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
            training_objective: DragonBrowserTrainingObjectiveConfig::default(),
            optimizer: Default::default(),
            execution_backend: DragonBrowserExecutionBackend::Cpu,
            block_size: 8,
            tbptt_chunk_size: None,
            tbptt_persist_across_steps: false,
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
    async fn browser_training_guards_composite_self_distillation_objective() {
        let config = DragonBrowserTrainingConfig {
            experiment_kind: crate::config::DragonExperimentKind::ClimbMixPretraining,
            model_config: tiny_model_config(256),
            training_objective: DragonBrowserTrainingObjectiveConfig::SdftSdpo(Default::default()),
            optimizer: Default::default(),
            execution_backend: DragonBrowserExecutionBackend::Cpu,
            block_size: 8,
            tbptt_chunk_size: None,
            tbptt_persist_across_steps: false,
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
            eval_source: None,
            live_participant: None,
        };
        let err = run_browser_training_with_release_manifest(
            "https://example.invalid",
            &config,
            &dummy_release_manifest(),
        )
        .await
        .expect_err("composite browser objective should be guarded");
        assert!(
            err.to_string()
                .contains("browser training is only wired for next_token execution"),
            "unexpected error: {err}"
        );
    }

    #[wasm_bindgen_test(async)]
    async fn browser_training_smoke_http_json() {
        let records = vec![
            TokenWindowRecord {
                inputs: vec![1, 2, 3, 4, 5, 6, 7, 8],
                targets: vec![2, 3, 4, 5, 6, 7, 8, 9],
                reset_stream_state: true,
                ..TokenWindowRecord::default()
            },
            TokenWindowRecord {
                inputs: vec![2, 3, 4, 5, 6, 7, 8, 9],
                targets: vec![3, 4, 5, 6, 7, 8, 9, 10],
                reset_stream_state: false,
                ..TokenWindowRecord::default()
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
            training_objective: DragonBrowserTrainingObjectiveConfig::default(),
            optimizer: Default::default(),
            execution_backend: DragonBrowserExecutionBackend::Cpu,
            block_size: 8,
            tbptt_chunk_size: None,
            tbptt_persist_across_steps: false,
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
                ..TokenWindowRecord::default()
            },
            TokenWindowRecord {
                inputs: vec![2, 3, 4, 5, 6, 7, 8, 9],
                targets: vec![3, 4, 5, 6, 7, 8, 9, 10],
                reset_stream_state: false,
                ..TokenWindowRecord::default()
            },
        ];
        let shard_b = vec![
            TokenWindowRecord {
                inputs: vec![10, 11, 12, 13, 14, 15, 16, 17],
                targets: vec![11, 12, 13, 14, 15, 16, 17, 18],
                reset_stream_state: false,
                ..TokenWindowRecord::default()
            },
            TokenWindowRecord {
                inputs: vec![11, 12, 13, 14, 15, 16, 17, 18],
                targets: vec![12, 13, 14, 15, 16, 17, 18, 19],
                reset_stream_state: false,
                ..TokenWindowRecord::default()
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
            training_objective: DragonBrowserTrainingObjectiveConfig::default(),
            optimizer: Default::default(),
            execution_backend: DragonBrowserExecutionBackend::Cpu,
            block_size: 8,
            tbptt_chunk_size: None,
            tbptt_persist_across_steps: false,
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
                ..TokenWindowRecord::default()
            },
            TokenWindowRecord {
                inputs: vec![2, 3, 4, 5, 6, 7, 8, 9],
                targets: vec![3, 4, 5, 6, 7, 8, 9, 10],
                reset_stream_state: false,
                ..TokenWindowRecord::default()
            },
        ];
        let shard_b = vec![
            TokenWindowRecord {
                inputs: vec![10, 11, 12, 13, 14, 15, 16, 17],
                targets: vec![11, 12, 13, 14, 15, 16, 17, 18],
                reset_stream_state: false,
                ..TokenWindowRecord::default()
            },
            TokenWindowRecord {
                inputs: vec![11, 12, 13, 14, 15, 16, 17, 18],
                targets: vec![12, 13, 14, 15, 16, 17, 18, 19],
                reset_stream_state: false,
                ..TokenWindowRecord::default()
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
            training_objective: DragonBrowserTrainingObjectiveConfig::default(),
            optimizer: Default::default(),
            execution_backend: DragonBrowserExecutionBackend::Cpu,
            block_size: 8,
            tbptt_chunk_size: None,
            tbptt_persist_across_steps: false,
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
                ..TokenWindowRecord::default()
            },
            TokenWindowRecord {
                inputs: vec![2, 3, 4, 5, 6, 7, 8, 9],
                targets: vec![3, 4, 5, 6, 7, 8, 9, 10],
                reset_stream_state: false,
                ..TokenWindowRecord::default()
            },
        ];
        let shard_b = vec![
            TokenWindowRecord {
                inputs: vec![10, 11, 12, 13, 14, 15, 16, 17],
                targets: vec![11, 12, 13, 14, 15, 16, 17, 18],
                reset_stream_state: true,
                ..TokenWindowRecord::default()
            },
            TokenWindowRecord {
                inputs: vec![11, 12, 13, 14, 15, 16, 17, 18],
                targets: vec![12, 13, 14, 15, 16, 17, 18, 19],
                reset_stream_state: false,
                ..TokenWindowRecord::default()
            },
        ];
        let shard_c = vec![
            TokenWindowRecord {
                inputs: vec![20, 21, 22, 23, 24, 25, 26, 27],
                targets: vec![21, 22, 23, 24, 25, 26, 27, 28],
                reset_stream_state: true,
                ..TokenWindowRecord::default()
            },
            TokenWindowRecord {
                inputs: vec![21, 22, 23, 24, 25, 26, 27, 28],
                targets: vec![22, 23, 24, 25, 26, 27, 28, 29],
                reset_stream_state: false,
                ..TokenWindowRecord::default()
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
            training_objective: DragonBrowserTrainingObjectiveConfig::default(),
            optimizer: Default::default(),
            execution_backend: DragonBrowserExecutionBackend::Cpu,
            block_size: 8,
            tbptt_chunk_size: None,
            tbptt_persist_across_steps: false,
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
    async fn browser_training_downgrades_cleanly_when_wgpu_cannot_train() {
        let config = DragonBrowserTrainingConfig {
            experiment_kind: crate::config::DragonExperimentKind::NcaPrepretraining,
            model_config: tiny_model_config(256),
            training_objective: DragonBrowserTrainingObjectiveConfig::default(),
            optimizer: Default::default(),
            execution_backend: DragonBrowserExecutionBackend::Wgpu,
            block_size: 8,
            tbptt_chunk_size: None,
            tbptt_persist_across_steps: false,
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
        .expect_err("browser preflight should downgrade before training starts");
        let error = error.to_string();
        assert!(
            error.contains("downgrading to verifier")
                || error.contains("downgrading browser peer to verifier/observer"),
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
            training_objective: DragonBrowserTrainingObjectiveConfig::default(),
            optimizer: Default::default(),
            execution_backend: DragonBrowserExecutionBackend::Cpu,
            block_size: 8,
            tbptt_chunk_size: None,
            tbptt_persist_across_steps: false,
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

    fn tiny_ruliad_corpus_config() -> RuliadCorpusConfig {
        RuliadCorpusConfig {
            output_dir: PathBuf::from("wasm-browser-ruliad-smoke"),
            seed: 17,
            name: "wasm-browser-ruliad-smoke".into(),
            train_samples: 2,
            validation_samples: 1,
            chunk_token_capacity: 4096,
            serialization: RuliadSerializationConfig {
                document_tokens: 2048,
                preview_samples: 1,
                ..RuliadSerializationConfig::default()
            },
            tokenization: RuliadTokenizationConfig::StructuredSymbolic {
                vocab_size: 272,
                eos_id: Some(271),
            },
            formal_generalization: Default::default(),
            source_selection: RuliadSourceSelectionConfig {
                difficulty_levels: UsizeRangeConfig { min: 0, max: 1 },
                formal_task_mix: RuliadFormalTaskMixConfig {
                    advance_proof_weight: 1,
                    select_proof_action_weight: 0,
                    construct_proof_weight: 0,
                    check_proof_weight: 0,
                },
                ..RuliadSourceSelectionConfig::default()
            },
            families: burn_dragon_universality::ruliad::formal_ruliad_families(),
            proof_tasks: None,
            lean_task_limit: None,
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
            "app_semver": env!("CARGO_PKG_VERSION"),
            "git_commit": "smoke",
            "cargo_lock_hash": "browser-smoke-lock",
            "burn_version_string": "0.21.0",
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
            app_semver: semver::Version::parse(env!("CARGO_PKG_VERSION"))
                .expect("valid burn_dragon version"),
            git_commit: "smoke".into(),
            cargo_lock_hash: ContentId::new("browser-smoke-lock"),
            burn_version_string: "0.21.0".into(),
            enabled_features_hash: ContentId::new("browser-smoke-features"),
            protocol_major: 0,
            supported_workloads: Vec::new(),
            built_at: chrono::Utc::now(),
        })
    }
}
