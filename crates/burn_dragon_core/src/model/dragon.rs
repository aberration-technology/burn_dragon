mod auxiliary_memory;
mod connector;
mod continual_backprop;
mod diagnostics;
#[cfg(any(feature = "probe", test))]
mod interpretability;
mod language_head;
mod language_pipeline;
mod predictive_coding;
mod sequence_dispatch;
pub use continual_backprop::{
    SharedLowrankActivationBatchStats, SharedLowrankContinualBackpropRuntime,
    SharedLowrankFeatureMetrics, SharedLowrankParamIds,
};
#[cfg(any(feature = "probe", test))]
pub use interpretability::{
    HeadTensorComparisonDiagnostics, HeadTensorGeometryDiagnostics,
    LanguageLayerStateDeltaDiagnostics, LanguageLayerStateSummaryDiagnostics,
    LanguageLowRankLayerComparisonDiagnostics, LanguageLowRankLayerGeometryDiagnostics,
    TensorComparisonDiagnostics, TensorDistributionDiagnostics, TensorStateDeltaDiagnostics,
    TensorStateSummaryDiagnostics, compare_model_states, summarize_model_state,
};
pub use predictive_coding::{
    DragonPredictiveCodingHeadActivityVjp, DragonPredictiveCodingHeadVjp,
    DragonPredictiveCodingInitialVjp, DragonPredictiveCodingLayerTrace,
    DragonPredictiveCodingLayerVjp, DragonPredictiveCodingParameterIds,
    DragonPredictiveCodingSequenceScoreHeadParameterIds,
    DragonPredictiveCodingSequenceScoreHeadVjp, DragonPredictiveCodingStateVjp,
    DragonPredictiveCodingSupport, DragonPredictiveCodingVjpProfileSnapshot,
    dragon_predictive_coding_vjp_profile_reset, dragon_predictive_coding_vjp_profile_snapshot,
};

use burn::module::{AutodiffModule, Module, Param};
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData, activation};
use burn_dragon_kernel::api::attention::{
    supports_dense_causal_attention_backend, try_fused_dense_causal_attention_wgpu,
};
use burn_dragon_kernel::api::recurrent::{
    CompiledRecurrentAttentionPlan, RecurrentAttentionOutput, supports_recurrent_backend,
    try_fused_recurrent_attention_input_vjp, try_fused_recurrent_attention_wgpu,
    try_fused_recurrent_attention_wgpu_with_plan,
};
use burn_dragon_kernel::kernels::sequence::mamba3::forward::{
    Mamba3TensorizedState, tensorized_mamba3_forward, use_tensorized_mamba3_forward_experimental,
};
use burn_dragon_time::Instant;
use burn_gdn::{GatedDeltaNet2Executor, GatedDeltaNet2Memory, try_gdn2_chunk_wy};
use rand::distributions::{Distribution, WeightedIndex};
use rand::prelude::*;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::ops::Range;
use std::sync::Once;

use super::attention::Attention;
use super::attention_residual::{
    AttentionResidual, BlockAttentionResidual, ResidualConnectorKind, ResidualHistory,
};
use super::config::{
    ClockedSlowMemoryConfig, DragonConfig, DragonRandomScaffoldConfig, FusedKernelConfig,
    HierarchicalDragonConfig, HierarchicalDragonSharing, LanguageHeadConfig, LatentReasoningConfig,
    NextLatentTransitionConfig, SummaryMemoryConfig, YNeuronRecurrenceConfig,
};
#[cfg(any(feature = "probe", test))]
use super::dragon_support::{
    LanguageDragonInitLayerDiagnostics, average_language_dragon_init_diagnostics,
    positive_fraction, rms_from_values, tensor_values_f32, values_are_finite,
};
use super::dragon_support::{
    LanguageMhcLayerDiagnostics, LanguageMhcMergeBindings, LanguageMhcSplitBindings,
    LanguagePipelineState, RecurrentPositionMode, ResidualConnectorRef, RolloutExecutorMode,
    average_language_mhc_diagnostics, logits_projection_profile_enabled,
    logits_projection_profile_record, shannon_entropy,
};
use super::init::{DragonFiringTargetKind, DragonInitializer, DragonProjectionRole};
use super::norm::DragonNorm;
#[cfg(any(feature = "probe", test))]
use super::residual_stream::LowRankResidualOutput;
use super::residual_stream::lowrank_residual_step_next_branch_thresholds;
#[cfg(any(feature = "probe", test))]
use super::residual_stream::lowrank_residual_step_with_metrics_branch_thresholds;
#[cfg(any(feature = "viz", feature = "probe"))]
use super::residual_stream::{
    decode_y_neuron_tail, lowrank_residual_step_branch_thresholds_relu_native,
};
#[cfg(not(any(feature = "viz", feature = "probe")))]
use super::residual_stream::{
    decode_y_neuron_tail, lowrank_residual_step_next_branch_thresholds_relu_native,
};
use super::scaffold::{
    DragonRandomScaffoldAdapters, DragonRandomScaffoldReport, build_report, fast_scaffold_paths,
    initialize_scaffold_2d, initialize_scaffold_3d, slow_scaffold_paths,
};
use super::sequence::gdn2::{
    GatedDeltaNet2Implementation, GatedDeltaNet2Parameters, ResolvedGatedDeltaNet2Config,
    gated_deltanet2_reference, l2_normalize_last,
};
use super::sequence::linear::{
    expand_attention_values_to_heads, recurrent_attention_dense_score_context_reference,
    recurrent_attention_dense_score_final_rho_reference,
    recurrent_attention_dense_score_initial_context_reference,
    recurrent_attention_dense_score_reference, recurrent_attention_reference,
};
use super::sequence::mamba::{
    MambaReferenceState, MambaSequenceParameters, ResolvedMambaSequenceConfig, mamba_reference,
};
use super::sequence::state::{
    gated_deltanet2_state, mamba3_state, write_gated_deltanet2_state, write_mamba3_state,
};
use super::sequence::{SequenceKernelConfig, SequenceMemorySystem, SequenceTrainingExecutor};
#[cfg(any(feature = "viz", feature = "probe"))]
use super::state::LayerVizState;
use super::state::{LayerState, ModelState};
use super::widen::{
    widen_1d_headed_last_dim_prefix_zero_tail, widen_2d_headed_last_dim_prefix_zero_tail,
    widen_2d_headed_row_prefix, widen_2d_last_dim_prefix, widen_3d_last_dim_prefix,
    widen_3d_last_dim_prefix_zero_tail,
};
use super::{ManifoldHyperConnections, mhc_merge_with_coefficients, mhc_split_with_coefficients};

#[derive(Module, Debug)]
struct SequenceScoreHead<B: Backend> {
    query: Linear<B>,
    candidate: Linear<B>,
    score: Linear<B>,
}

impl<B: Backend> SequenceScoreHead<B> {
    fn deterministic_linear(
        input_dim: usize,
        output_dim: usize,
        seed: u64,
        device: &B::Device,
    ) -> Linear<B> {
        let bound = (1.0 / input_dim.max(1) as f32).sqrt();
        let mut rng = StdRng::seed_from_u64(seed);
        let weights = (0..input_dim.saturating_mul(output_dim))
            .map(|_| rng.gen_range(-bound..bound))
            .collect::<Vec<_>>();
        let biases = (0..output_dim)
            .map(|_| rng.gen_range(-bound..bound))
            .collect::<Vec<_>>();
        Linear {
            weight: Param::from_tensor(Tensor::from_data(
                TensorData::new(weights, [input_dim, output_dim]),
                device,
            )),
            bias: Some(Param::from_tensor(Tensor::from_data(
                TensorData::new(biases, [output_dim]),
                device,
            ))),
        }
    }

    fn new(input_dim: usize, projection_dim: usize, device: &B::Device) -> Self {
        assert!(
            projection_dim > 0,
            "sequence score projection_dim must be positive"
        );
        Self {
            // Optional heads must not advance the backend-global RNG or alter initialization and
            // dropout streams in matched baseline runs. Independent host RNGs preserve ordinary
            // fan-in initialization while keeping the shared Dragon parameter contract invariant.
            query: Self::deterministic_linear(
                input_dim,
                projection_dim,
                0x7175_6572_795f_6b31,
                device,
            ),
            candidate: Self::deterministic_linear(
                input_dim,
                projection_dim,
                0x6361_6e64_5f6b_6579,
                device,
            ),
            score: Self::deterministic_linear(projection_dim, 1, 0x7363_6f72_655f_7631, device),
        }
    }

    fn forward_candidate(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        self.score.forward(self.candidate.forward(hidden))
    }

    fn forward_pair(
        &self,
        prompt_hidden: Tensor<B, 3>,
        terminal_hidden: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let query = self.query.forward(prompt_hidden);
        let candidate = self.candidate.forward(terminal_hidden);
        self.score.forward(query * candidate)
    }

    fn value_clone(&self) -> Self {
        Self {
            query: clone_linear_value(&self.query),
            candidate: clone_linear_value(&self.candidate),
            score: clone_linear_value(&self.score),
        }
    }
}

#[derive(Module, Debug)]
pub struct DragonModel<B: Backend> {
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    mlp_internal_dim_multiplier: usize,
    vocab_size: usize,
    #[module(skip)]
    language_head: LanguageHeadRuntimeKind,
    #[module(skip)]
    tie_input_output_embeddings: bool,
    sequence_kernel: SequenceKernelConfig,
    rollout_fast_steps_per_slow_step: usize,
    kernel: FusedKernelConfig,
    x_relu_threshold: f32,
    y_relu_threshold: f32,
    y_neuron_recurrence: YNeuronRecurrenceConfig,
    clocked_slow_memory: ClockedSlowMemoryConfig,
    summary_memory: SummaryMemoryConfig,
    hierarchical_dragon: HierarchicalDragonConfig,
    latent_reasoning: LatentReasoningConfig,
    #[module(skip)]
    random_scaffold_config: DragonRandomScaffoldConfig,
    #[module(skip)]
    layer_latent_totals: Vec<usize>,
    #[module(skip)]
    shared_lowrank_continual_backprop: Option<SharedLowrankContinualBackpropRuntime>,
    embed: Embedding<B>,
    dropout: Dropout,
    norm: DragonNorm<B>,
    attention: Attention<B>,
    residual_connector: ResidualConnectorKind,
    mhc_first_layer: usize,
    mhc_shared: Option<ManifoldHyperConnections<B>>,
    attention_residual_first_layer: usize,
    attention_residual_shared: Option<AttentionResidual<B>>,
    block_attention_residual_first_layer: usize,
    block_attention_residual_shared: Option<BlockAttentionResidual<B>>,
    encoder: Param<Tensor<B, 3>>,
    encoder_v: Param<Tensor<B, 3>>,
    decoder: Param<Tensor<B, 2>>,
    slow_encoder: Option<Param<Tensor<B, 3>>>,
    slow_encoder_v: Option<Param<Tensor<B, 3>>>,
    slow_decoder: Option<Param<Tensor<B, 2>>>,
    random_scaffold_adapters: Option<DragonRandomScaffoldAdapters<B>>,
    #[module(skip)]
    mamba_config: ResolvedMambaSequenceConfig,
    mamba: Option<MambaSequenceParameters<B>>,
    #[module(skip)]
    gated_deltanet2_config: ResolvedGatedDeltaNet2Config,
    gated_deltanet2: Option<GatedDeltaNet2Parameters<B>>,
    gated_deltanet2_upstream: Option<GatedDeltaNet2Memory<B>>,
    lm_head: Option<Param<Tensor<B, 2>>>,
    nca_factorized_lm_head: Option<Param<Tensor<B, 2>>>,
    nca_special_lm_head: Option<Param<Tensor<B, 2>>>,
    sequence_score_head: Option<SequenceScoreHead<B>>,
    latent_refiner_in: Option<Linear<B>>,
    latent_refiner_out: Option<Linear<B>>,
    latent_refiner_gate: Option<Param<Tensor<B, 1>>>,
    latent_energy_head: Option<Linear<B>>,
    latent_stop_head: Option<Linear<B>>,
    latent_jepa_predictor: Option<Linear<B>>,
    latent_step_decoder_embedding: Option<Param<Tensor<B, 2>>>,
    next_latent_transition: NextLatentTransitionConfig,
    next_latent_transition_in: Option<Linear<B>>,
    next_latent_transition_mid: Option<Linear<B>>,
    next_latent_transition_out: Option<Linear<B>>,
    #[module(skip)]
    nca_factorized_head_tables: Option<NcaFactorizedHeadTables>,
}

#[derive(Clone, Debug)]
pub struct LatentReasoningOutput<B: Backend> {
    pub raw_hidden: Tensor<B, 3>,
    pub final_hidden: Tensor<B, 3>,
    pub step_hiddens: Vec<Tensor<B, 3>>,
    pub energies: Vec<Tensor<B, 3>>,
    pub stop_logits: Vec<Tensor<B, 3>>,
    pub stop_probs: Vec<Tensor<B, 3>>,
    pub steps_used: usize,
}

#[derive(Clone, Debug)]
pub struct SharedLowrankWeights<B: Backend> {
    pub encoder: Tensor<B, 3>,
    pub encoder_v: Tensor<B, 3>,
    pub decoder: Tensor<B, 2>,
}

#[derive(Clone, Debug)]
pub struct SharedLowrankPopulationWeights<B: Backend> {
    pub encoder: Tensor<B, 4>,
    pub encoder_v: Tensor<B, 4>,
    pub decoder: Tensor<B, 3>,
}

impl<B: Backend> SharedLowrankPopulationWeights<B> {
    pub fn population_size(&self) -> usize {
        self.encoder.shape().dims::<4>()[0]
    }
}

#[derive(Clone, Debug)]
pub struct SharedLowrankPopulationFactors<B: Backend> {
    pub encoder_a: Tensor<B, 4>,
    pub encoder_b: Tensor<B, 4>,
    pub encoder_v_a: Tensor<B, 4>,
    pub encoder_v_b: Tensor<B, 4>,
    pub decoder_a: Tensor<B, 3>,
    pub decoder_b: Tensor<B, 3>,
    pub signs: Tensor<B, 1>,
    pub encoder_scale: f64,
    pub encoder_v_scale: f64,
    pub decoder_scale: f64,
    pub sigma: f32,
}

impl<B: Backend> SharedLowrankPopulationFactors<B> {
    pub fn population_size(&self) -> usize {
        self.signs.shape().dims::<1>()[0]
    }
}

struct PopulationLayerLowrankFactors<B: Backend> {
    encoder_a: Tensor<B, 4>,
    encoder_b: Tensor<B, 4>,
    encoder_v_a: Tensor<B, 4>,
    encoder_v_b: Tensor<B, 4>,
    decoder_a: Tensor<B, 3>,
    decoder_b: Tensor<B, 3>,
    signs: Tensor<B, 1>,
    latent_per_head: usize,
}

struct SharedLowrankPopulationProjection<'a, B: Backend> {
    dense: Tensor<B, 4>,
    projector: Tensor<B, 4>,
    population: usize,
    relu_threshold: f32,
    use_fused: bool,
    latent_pattern: &'a crate::kernel::BlockPattern1d,
    sparse_mask: Option<Tensor<B, 4>>,
}

struct FactorizedPopulationProjection<'a, B: Backend> {
    dense: Tensor<B, 4>,
    base_projector: Tensor<B, 4>,
    factor_a: Tensor<B, 4>,
    factor_b: Tensor<B, 4>,
    signs: Tensor<B, 1>,
    sigma_scale: f64,
    population: usize,
    relu_threshold: f32,
    latent_pattern: &'a crate::kernel::BlockPattern1d,
    sparse_mask: Option<Tensor<B, 4>>,
}

struct FactorizedPopulationDecode<B: Backend> {
    y_neuron: Tensor<B, 4>,
    base_decoder: Tensor<B, 2>,
    factor_a: Tensor<B, 3>,
    factor_b: Tensor<B, 3>,
    signs: Tensor<B, 1>,
    sigma_scale: f64,
    population: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HierarchicalDragonBranch {
    Fast,
    Slow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HierarchicalSequenceSlot {
    Fast,
    Slow,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct DragonLatentWidenReport {
    pub old_latent_total: usize,
    pub new_latent_total: usize,
    pub old_latent_per_head: usize,
    pub new_latent_per_head: usize,
    pub appended_latent_total: usize,
}

#[derive(Clone)]
pub(crate) struct NcaFactorizedHeadTables {
    patch_cells: usize,
    state_count: usize,
    special_token_ids: Vec<u32>,
    patch_digit_tables: Vec<Vec<i64>>,
    patch_mask_table: Vec<f32>,
    special_index_table: Vec<i64>,
    special_mask_table: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum LanguageHeadRuntimeKind {
    StandardTokenClassification,
    NcaFactorizedPatch,
}

impl LanguageHeadRuntimeKind {
    fn from_config(config: &LanguageHeadConfig) -> Self {
        match config {
            LanguageHeadConfig::StandardTokenClassification => Self::StandardTokenClassification,
            LanguageHeadConfig::NcaFactorizedPatch { .. } => Self::NcaFactorizedPatch,
        }
    }

    fn uses_flat_token_logits(&self) -> bool {
        matches!(self, Self::StandardTokenClassification)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum LanguageModuleLrScaleTarget {
    Embedding,
    Normalization,
    OutputHead,
    SharedLowrankEncoder,
    SharedLowrankDecoder,
    Attention,
    Mamba,
    GatedDeltaNet2,
    ResidualModules,
    LatentReasoning,
    OtherBackbone,
}

impl core::fmt::Debug for NcaFactorizedHeadTables {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NcaFactorizedHeadTables")
            .field("patch_cells", &self.patch_cells)
            .field("state_count", &self.state_count)
            .field("special_token_ids", &self.special_token_ids)
            .field(
                "patch_digit_tables",
                &format_args!("{} tables", self.patch_digit_tables.len()),
            )
            .field(
                "patch_mask_table",
                &format_args!("len={}", self.patch_mask_table.len()),
            )
            .field(
                "special_index_table",
                &format_args!("len={}", self.special_index_table.len()),
            )
            .field(
                "special_mask_table",
                &format_args!("len={}", self.special_mask_table.len()),
            )
            .finish()
    }
}

impl NcaFactorizedHeadTables {
    fn from_language_head_config(
        config: &LanguageHeadConfig,
        vocab_size: usize,
    ) -> Result<Option<Self>, String> {
        let LanguageHeadConfig::NcaFactorizedPatch {
            state_count,
            patch_size,
            frame_special_tokens,
            eos_id,
        } = config
        else {
            return Ok(None);
        };
        config.validate_for_vocab_size(vocab_size)?;
        let patch_cells = patch_size.saturating_mul(*patch_size);
        let patch_vocab_size = state_count
            .checked_pow(patch_cells as u32)
            .ok_or_else(|| "NCA factorized head patch vocabulary overflow".to_string())?;
        let mut special_token_ids = Vec::new();
        if *frame_special_tokens {
            special_token_ids.push(patch_vocab_size as u32);
            special_token_ids.push((patch_vocab_size + 1) as u32);
        }
        if let Some(eos_id) = eos_id
            && !special_token_ids.contains(eos_id)
        {
            special_token_ids.push(*eos_id);
        }

        let mut patch_digit_tables = vec![vec![0i64; vocab_size]; patch_cells];
        let mut patch_mask_table = vec![0.0f32; vocab_size];
        for token_id in 0..patch_vocab_size.min(vocab_size) {
            patch_mask_table[token_id] = 1.0;
            let mut remainder = token_id;
            for cell_idx in (0..patch_cells).rev() {
                let digit = remainder % state_count;
                patch_digit_tables[cell_idx][token_id] = digit as i64;
                remainder /= state_count;
            }
        }

        let mut special_index_table = vec![0i64; vocab_size];
        let mut special_mask_table = vec![0.0f32; vocab_size];
        for (special_idx, token_id) in special_token_ids.iter().enumerate() {
            let token_id = *token_id as usize;
            if token_id < vocab_size {
                special_index_table[token_id] = special_idx as i64;
                special_mask_table[token_id] = 1.0;
            }
        }

        Ok(Some(Self {
            patch_cells,
            state_count: *state_count,
            special_token_ids,
            patch_digit_tables,
            patch_mask_table,
            special_index_table,
            special_mask_table,
        }))
    }

    fn special_count(&self) -> usize {
        self.special_token_ids.len()
    }
}

/// Named inputs for a single low-rank positive projection.
///
/// This keeps projection call sites declarative for the remaining dense/fused float path.
struct LowrankProjectionRequest<'a, B: Backend> {
    dense: Tensor<B, 4>,
    projector: Tensor<B, 4>,
    relu_threshold: f32,
    use_fused: bool,
    latent_pattern: &'a crate::kernel::BlockPattern1d,
    sparse_mask: Option<Tensor<B, 4>>,
}

fn widen_headed_linear_output_prefix_zero_tail<B: Backend>(
    current: &Linear<B>,
    fresh: &Linear<B>,
    heads: usize,
    old_per_head: usize,
    new_per_head: usize,
) -> Result<Linear<B>, String> {
    let mut widened = fresh.clone();
    widened.weight = Param::from_tensor(widen_2d_headed_last_dim_prefix_zero_tail(
        current.weight.val(),
        fresh.weight.val(),
        heads,
        old_per_head,
        new_per_head,
    )?);
    widened.bias = match (&current.bias, &fresh.bias) {
        (Some(current_bias), Some(fresh_bias)) => Some(Param::from_tensor(
            widen_1d_headed_last_dim_prefix_zero_tail(
                current_bias.val(),
                fresh_bias.val(),
                heads,
                old_per_head,
                new_per_head,
            )?,
        )),
        (None, None) => None,
        _ => return Err("cannot widen linear output with incompatible bias presence".to_string()),
    };
    Ok(widened)
}

fn clone_linear_value<B: Backend>(current: &Linear<B>) -> Linear<B> {
    let mut cloned = current.clone();
    cloned.weight = Param::from_tensor(current.weight.val());
    cloned.bias = current
        .bias
        .as_ref()
        .map(|bias| Param::from_tensor(bias.val()));
    cloned
}

fn scale_linear_output<B: Backend>(current: &Linear<B>, scale: f32) -> Linear<B> {
    let mut scaled = current.clone();
    scaled.weight = Param::from_tensor(current.weight.val().mul_scalar(scale));
    scaled.bias = current
        .bias
        .as_ref()
        .map(|bias| Param::from_tensor(bias.val().mul_scalar(scale)));
    scaled
}

fn widen_upstream_gated_deltanet2_prefix<B: Backend>(
    current: &GatedDeltaNet2Memory<B>,
    fresh: &GatedDeltaNet2Memory<B>,
    old_latent_per_head: usize,
    new_latent_per_head: usize,
) -> Result<GatedDeltaNet2Memory<B>, String> {
    if current.config.heads != fresh.config.heads
        || current.config.head_dim != fresh.config.head_dim
        || current.config.chunk_size != fresh.config.chunk_size
        || current.config.qk_l2_norm != fresh.config.qk_l2_norm
        || current.config.allow_neg_eigval != fresh.config.allow_neg_eigval
        || current.config.erase_gate != fresh.config.erase_gate
        || current.config.write_gate != fresh.config.write_gate
        || current.config.decay_gate != fresh.config.decay_gate
        || current.config.executor != fresh.config.executor
    {
        return Err(format!(
            "cannot widen upstream gated_deltanet2 with incompatible config (current={:?}, fresh={:?})",
            current.config, fresh.config
        ));
    }
    if current.config.latent_per_head != old_latent_per_head
        || fresh.config.latent_per_head != new_latent_per_head
        || old_latent_per_head > new_latent_per_head
    {
        return Err(format!(
            "cannot widen upstream gated_deltanet2 with incompatible latent shape (current={} fresh={} old={} new={})",
            current.config.latent_per_head,
            fresh.config.latent_per_head,
            old_latent_per_head,
            new_latent_per_head
        ));
    }

    let mut widened = fresh.clone();
    widened.query = widen_headed_linear_output_prefix_zero_tail(
        &current.query,
        &fresh.query,
        current.config.heads,
        old_latent_per_head,
        new_latent_per_head,
    )?;
    widened.key = widen_headed_linear_output_prefix_zero_tail(
        &current.key,
        &fresh.key,
        current.config.heads,
        old_latent_per_head,
        new_latent_per_head,
    )?;
    widened.erase = widen_headed_linear_output_prefix_zero_tail(
        &current.erase,
        &fresh.erase,
        current.config.heads,
        old_latent_per_head,
        new_latent_per_head,
    )?;
    widened.decay = widen_headed_linear_output_prefix_zero_tail(
        &current.decay,
        &fresh.decay,
        current.config.heads,
        old_latent_per_head,
        new_latent_per_head,
    )?;
    widened.decay_log = Param::from_tensor(widen_2d_last_dim_prefix(
        current.decay_log.val(),
        fresh.decay_log.val(),
        old_latent_per_head,
        new_latent_per_head,
    )?);
    widened.value = clone_linear_value(&current.value);
    widened.write = clone_linear_value(&current.write);
    let scale = (new_latent_per_head as f32 / old_latent_per_head.max(1) as f32).sqrt();
    widened.out = scale_linear_output(&current.out, scale);
    Ok(widened)
}

impl<B: Backend> DragonModel<B> {
    fn replace_param_value<const D: usize>(
        param: Param<Tensor<B, D>>,
        value: Tensor<B, D>,
    ) -> Param<Tensor<B, D>> {
        let (id, _old, mapper) = param.consume();
        Param::from_mapped_value(id, value, mapper)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn new(config: DragonConfig, device: &B::Device) -> Self {
        config
            .random_scaffold
            .validate_for_model(config.n_embd, config.n_head, config.latent_total())
            .unwrap_or_else(|message| panic!("invalid model.random_scaffold: {message}"));
        let initializer = DragonInitializer::new(&config.initialization);
        let embed = EmbeddingConfig::new(config.vocab_size, config.n_embd)
            .with_initializer(initializer.embedding_initializer(config.n_embd))
            .init(device);
        let dropout = DropoutConfig::new(config.dropout).init();
        let norm = DragonNorm::new(&config.normalization, config.n_embd, device);

        let latent_per_head = config.latent_per_head();
        let latent_total = config.latent_total();
        let attention = Attention::new(
            latent_per_head,
            config.n_head,
            device,
            &config.fused_kernels,
        );
        let residual_depth = config.n_layer.max(1) * config.rollout_fast_steps_per_slow_step.max(1);
        let activation_thresholds =
            initializer.activation_thresholds(config.n_embd, latent_per_head, residual_depth);
        let use_shared_relu_threshold = matches!(
            config.initialization.firing_targets.kind,
            DragonFiringTargetKind::Disabled
        );
        let shared_relu_threshold = config.fused_kernels.relu_threshold;
        let (encoder_path, encoder_v_path, decoder_path) = fast_scaffold_paths();
        let encoder = Param::from_tensor(if config.random_scaffold.enabled {
            initialize_scaffold_3d(
                &config,
                encoder_path,
                [config.n_head, config.n_embd, latent_per_head],
                device,
            )
        } else {
            initializer.headwise_projection_tensor::<B>(
                DragonProjectionRole::Encoder,
                config.n_head,
                config.n_embd,
                latent_per_head,
                residual_depth,
                device,
            )
        });

        let encoder_v = Param::from_tensor(if config.random_scaffold.enabled {
            initialize_scaffold_3d(
                &config,
                encoder_v_path,
                [config.n_head, config.n_embd, latent_per_head],
                device,
            )
        } else {
            initializer.headwise_projection_tensor::<B>(
                DragonProjectionRole::EncoderValue,
                config.n_head,
                config.n_embd,
                latent_per_head,
                residual_depth,
                device,
            )
        });

        let decoder = Param::from_tensor(if config.random_scaffold.enabled {
            initialize_scaffold_2d(&config, decoder_path, [latent_total, config.n_embd], device)
        } else {
            initializer.projection_tensor::<B>(
                DragonProjectionRole::Decoder,
                latent_total,
                config.n_embd,
                residual_depth,
                device,
            )
        });
        let hierarchical_dragon = config.hierarchical_dragon.clone();
        let (slow_encoder, slow_encoder_v, slow_decoder) = if hierarchical_dragon.enabled
            && matches!(
                hierarchical_dragon.weight_sharing,
                HierarchicalDragonSharing::Split
            ) {
            let (slow_encoder_path, slow_encoder_v_path, slow_decoder_path) = slow_scaffold_paths();
            (
                Some(Param::from_tensor(if config.random_scaffold.enabled {
                    initialize_scaffold_3d(
                        &config,
                        slow_encoder_path,
                        [config.n_head, config.n_embd, latent_per_head],
                        device,
                    )
                } else {
                    initializer.headwise_projection_tensor::<B>(
                        DragonProjectionRole::Encoder,
                        config.n_head,
                        config.n_embd,
                        latent_per_head,
                        residual_depth,
                        device,
                    )
                })),
                Some(Param::from_tensor(if config.random_scaffold.enabled {
                    initialize_scaffold_3d(
                        &config,
                        slow_encoder_v_path,
                        [config.n_head, config.n_embd, latent_per_head],
                        device,
                    )
                } else {
                    initializer.headwise_projection_tensor::<B>(
                        DragonProjectionRole::EncoderValue,
                        config.n_head,
                        config.n_embd,
                        latent_per_head,
                        residual_depth,
                        device,
                    )
                })),
                Some(Param::from_tensor(if config.random_scaffold.enabled {
                    initialize_scaffold_2d(
                        &config,
                        slow_decoder_path,
                        [latent_total, config.n_embd],
                        device,
                    )
                } else {
                    Tensor::<B, 2>::zeros([latent_total, config.n_embd], device)
                })),
            )
        } else {
            (None, None, None)
        };
        let random_scaffold_adapters = config
            .random_scaffold
            .enabled
            .then(|| DragonRandomScaffoldAdapters::new(&config, &config.random_scaffold, device));
        let residual_connector = config.resolved_residual_connector_kind();
        let mhc_first_layer = config
            .mhc
            .last_layers
            .map(|last_layers| config.n_layer.max(1).saturating_sub(last_layers))
            .unwrap_or(0);
        let mhc_shared = if residual_connector == ResidualConnectorKind::Mhc
            && config.mhc.enabled
            && (config.mhc.resolved_num_streams() > 1 || config.mhc.resolved_num_views() > 1)
        {
            Some(ManifoldHyperConnections::new_with_dense_dim(
                &config.mhc,
                mhc_first_layer,
                Some(config.n_embd),
                device,
            ))
        } else {
            None
        };
        let attention_residual_first_layer = config
            .attention_residual
            .last_layers
            .map(|last_layers| config.n_layer.max(1).saturating_sub(last_layers))
            .unwrap_or(0);
        let attention_residual_shared = (residual_connector
            == ResidualConnectorKind::AttentionResidual
            && config.attention_residual.enabled)
            .then(|| AttentionResidual::new(&config.attention_residual, config.n_embd, device));
        let block_attention_residual_first_layer = config
            .block_attention_residual
            .last_layers
            .map(|last_layers| config.n_layer.max(1).saturating_sub(last_layers))
            .unwrap_or(0);
        let block_attention_residual_shared = (residual_connector
            == ResidualConnectorKind::BlockAttentionResidual
            && config.block_attention_residual.enabled)
            .then(|| {
                BlockAttentionResidual::new(&config.block_attention_residual, config.n_embd, device)
            });
        let sequence_kernel = config.sequence_kernel;
        let mamba_config = config
            .mamba
            .resolve(config.n_embd, sequence_kernel.memory_system);
        let mamba = matches!(
            sequence_kernel.memory_system,
            SequenceMemorySystem::Mamba3StateSpaceDuality
        )
        .then(|| MambaSequenceParameters::new(mamba_config, sequence_kernel.memory_system, device));
        let gated_deltanet2_config =
            config
                .gated_deltanet2
                .resolve(config.n_head, config.n_embd, config.latent_per_head());
        let gated_deltanet2_executor = match sequence_kernel.executor {
            SequenceTrainingExecutor::GatedDeltaChunkWy => GatedDeltaNet2Executor::ChunkWy,
            _ => GatedDeltaNet2Executor::Reference,
        };
        let use_gdn2 = matches!(
            sequence_kernel.memory_system,
            SequenceMemorySystem::GatedDeltaNet2
        );
        let gated_deltanet2 = (use_gdn2
            && gated_deltanet2_config.implementation
                == GatedDeltaNet2Implementation::BdhAdapterLegacy)
            .then(|| GatedDeltaNet2Parameters::new(gated_deltanet2_config, device));
        let gated_deltanet2_upstream = (use_gdn2
            && gated_deltanet2_config.implementation == GatedDeltaNet2Implementation::UpstreamFull)
            .then(|| {
                GatedDeltaNet2Memory::new(
                    config.n_embd,
                    gated_deltanet2_config.upstream_config(gated_deltanet2_executor),
                    device,
                )
                .unwrap_or_else(|error| panic!("invalid upstream gated_deltanet2 config: {error}"))
            });
        let language_head = LanguageHeadRuntimeKind::from_config(&config.language_head);
        let nca_factorized_head_tables = NcaFactorizedHeadTables::from_language_head_config(
            &config.language_head,
            config.vocab_size,
        )
        .unwrap_or_else(|message| panic!("invalid language head config: {message}"));
        let lm_head = if nca_factorized_head_tables.is_none() {
            Some(Param::from_tensor(initializer.projection_tensor::<B>(
                DragonProjectionRole::LmHead,
                config.n_embd,
                config.vocab_size,
                residual_depth,
                device,
            )))
        } else {
            None
        };
        let nca_factorized_lm_head = nca_factorized_head_tables.as_ref().map(|tables| {
            Param::from_tensor(initializer.projection_tensor::<B>(
                DragonProjectionRole::LmHead,
                config.n_embd,
                tables.patch_cells * tables.state_count,
                residual_depth,
                device,
            ))
        });
        let nca_special_lm_head = nca_factorized_head_tables.as_ref().and_then(|tables| {
            (tables.special_count() > 0).then(|| {
                Param::from_tensor(initializer.projection_tensor::<B>(
                    DragonProjectionRole::LmHead,
                    config.n_embd,
                    tables.special_count(),
                    residual_depth,
                    device,
                ))
            })
        });
        let sequence_score_head = config.sequence_score_head.enabled.then(|| {
            SequenceScoreHead::new(
                config.n_embd,
                config.sequence_score_head.projection_dim,
                device,
            )
        });
        let latent_reasoning = config.latent_reasoning.clone();
        let latent_refiner_hidden =
            config.n_embd * latent_reasoning.refiner_hidden_multiplier.max(1);
        let latent_refiner_in = latent_reasoning
            .enabled
            .then(|| LinearConfig::new(config.n_embd, latent_refiner_hidden).init(device));
        let latent_refiner_out = latent_reasoning.enabled.then(|| {
            let mut out = LinearConfig::new(latent_refiner_hidden, config.n_embd).init(device);
            let [rows, cols] = out.weight.val().shape().dims();
            out.weight = Param::from_tensor(Tensor::<B, 2>::zeros([rows, cols], device));
            if let Some(bias) = out.bias.as_mut() {
                let [dim] = bias.val().shape().dims();
                *bias = Param::from_tensor(Tensor::<B, 1>::zeros([dim], device));
            }
            out
        });
        let latent_refiner_gate =
            (latent_reasoning.enabled && latent_reasoning.residual_refinement_gate).then(|| {
                let init = latent_reasoning.residual_refinement_gate_init;
                let logit = (init / (1.0 - init)).ln();
                Param::from_tensor(Tensor::<B, 1>::ones([config.n_embd], device).mul_scalar(logit))
            });
        let latent_energy_head = (latent_reasoning.enabled && latent_reasoning.energy_head)
            .then(|| LinearConfig::new(config.n_embd, 1).init(device));
        let latent_stop_head = (latent_reasoning.enabled && latent_reasoning.adaptive_halting)
            .then(|| {
                let mut head = LinearConfig::new(config.n_embd, 1).init(device);
                if let Some(bias) = head.bias.as_mut() {
                    *bias = Param::from_tensor(
                        Tensor::<B, 1>::ones([1], device)
                            .mul_scalar(latent_reasoning.stop_bias_init),
                    );
                }
                head
            });
        let latent_jepa_predictor = latent_reasoning
            .enabled
            .then(|| LinearConfig::new(config.n_embd, config.n_embd).init(device));
        let latent_step_decoder_embedding =
            (latent_reasoning.enabled && latent_reasoning.step_conditioned_decoder).then(|| {
                Param::from_tensor(Tensor::<B, 2>::zeros(
                    [latent_reasoning.max_steps.saturating_add(1), config.n_embd],
                    device,
                ))
            });
        let next_latent_transition = config.next_latent_transition.clone();
        let next_latent_transition_hidden =
            config.n_embd * next_latent_transition.hidden_multiplier.max(1);
        let next_latent_transition_in = next_latent_transition.enabled.then(|| {
            LinearConfig::new(config.n_embd * 2, next_latent_transition_hidden).init(device)
        });
        let next_latent_transition_mid = next_latent_transition.enabled.then(|| {
            LinearConfig::new(next_latent_transition_hidden, next_latent_transition_hidden)
                .init(device)
        });
        let next_latent_transition_out = next_latent_transition.enabled.then(|| {
            let mut out =
                LinearConfig::new(next_latent_transition_hidden, config.n_embd).init(device);
            if next_latent_transition.zero_init_output {
                let [rows, cols] = out.weight.val().shape().dims();
                out.weight = Param::from_tensor(Tensor::<B, 2>::zeros([rows, cols], device));
                if let Some(bias) = out.bias.as_mut() {
                    let [dim] = bias.val().shape().dims();
                    *bias = Param::from_tensor(Tensor::<B, 1>::zeros([dim], device));
                }
            }
            out
        });
        let layer_latent_totals = (0..config.n_layer)
            .map(|layer_idx| config.latent_total_for_layer(layer_idx))
            .collect();

        Self {
            n_layer: config.n_layer,
            n_embd: config.n_embd,
            n_head: config.n_head,
            mlp_internal_dim_multiplier: config.mlp_internal_dim_multiplier,
            vocab_size: config.vocab_size,
            language_head,
            tie_input_output_embeddings: config.tie_input_output_embeddings,
            sequence_kernel,
            rollout_fast_steps_per_slow_step: config.rollout_fast_steps_per_slow_step,
            kernel: config.fused_kernels,
            x_relu_threshold: if use_shared_relu_threshold {
                shared_relu_threshold
            } else {
                activation_thresholds.x
            },
            y_relu_threshold: if use_shared_relu_threshold {
                shared_relu_threshold
            } else {
                activation_thresholds.y
            },
            y_neuron_recurrence: config.y_neuron_recurrence,
            clocked_slow_memory: config.clocked_slow_memory,
            summary_memory: config.summary_memory,
            hierarchical_dragon,
            latent_reasoning,
            random_scaffold_config: config.random_scaffold.clone(),
            layer_latent_totals,
            shared_lowrank_continual_backprop: None,
            embed,
            dropout,
            norm,
            attention,
            residual_connector,
            mhc_first_layer,
            mhc_shared,
            attention_residual_first_layer,
            attention_residual_shared,
            block_attention_residual_first_layer,
            block_attention_residual_shared,
            encoder,
            encoder_v,
            decoder,
            slow_encoder,
            slow_encoder_v,
            slow_decoder,
            random_scaffold_adapters,
            mamba_config,
            mamba,
            gated_deltanet2_config,
            gated_deltanet2,
            gated_deltanet2_upstream,
            lm_head,
            nca_factorized_lm_head,
            nca_special_lm_head,
            sequence_score_head,
            latent_refiner_in,
            latent_refiner_out,
            latent_refiner_gate,
            latent_energy_head,
            latent_stop_head,
            latent_jepa_predictor,
            latent_step_decoder_embedding,
            next_latent_transition,
            next_latent_transition_in,
            next_latent_transition_mid,
            next_latent_transition_out,
            nca_factorized_head_tables,
        }
    }

    pub fn latent_total_capacity(&self) -> usize {
        self.decoder.val().shape().dims::<2>()[0]
    }

    pub fn latent_per_head_capacity(&self) -> usize {
        self.encoder.val().shape().dims::<3>()[2]
    }

    pub fn shared_lowrank_weights(&self) -> SharedLowrankWeights<B> {
        SharedLowrankWeights {
            encoder: self.encoder.val(),
            encoder_v: self.encoder_v.val(),
            decoder: self.decoder.val(),
        }
    }

    pub fn shared_lowrank_effective_weights(&self) -> SharedLowrankWeights<B> {
        let scaffold = self.shared_lowrank_weights();
        self.random_scaffold_adapters
            .as_ref()
            .map(|adapters| adapters.effective_fast(scaffold.clone()))
            .unwrap_or(scaffold)
    }

    pub fn uses_random_scaffold(&self) -> bool {
        self.random_scaffold_adapters.is_some()
    }

    pub fn random_scaffold_report(&self) -> Option<DragonRandomScaffoldReport> {
        self.random_scaffold_adapters.as_ref().map(|adapters| {
            let mut config = DragonConfig {
                n_layer: self.n_layer,
                n_embd: self.n_embd,
                n_head: self.n_head,
                mlp_internal_dim_multiplier: self.mlp_internal_dim_multiplier,
                vocab_size: self.vocab_size,
                random_scaffold: self.random_scaffold_config.clone(),
                hierarchical_dragon: self.hierarchical_dragon.clone(),
                ..DragonConfig::default()
            };
            config.n_expert = 1;
            build_report(&config, adapters)
        })
    }

    pub fn random_scaffold_trainable_param_ids(&self) -> Vec<burn::module::ParamId> {
        self.random_scaffold_adapters
            .as_ref()
            .map(|adapters| adapters.trainable_ids(self.random_scaffold_config.trainable_gain))
            .unwrap_or_default()
    }

    /// Folds the immutable scaffold and trained adapters into dense projection
    /// tensors for an ephemeral inference model.
    ///
    /// Training and distributed artifacts must retain the scaffold contract.
    /// Folding is intended only after `valid()` conversion, where it avoids
    /// rematerializing `A * B` for every autoregressive token.
    pub fn materialize_random_scaffold_for_inference(mut self) -> Self {
        let Some(adapters) = self.random_scaffold_adapters.as_ref() else {
            return self;
        };
        let fast = adapters.effective_fast(self.shared_lowrank_weights());
        let slow = if self.hierarchical_dragon.enabled
            && self.hierarchical_dragon.weight_sharing == HierarchicalDragonSharing::Split
        {
            let scaffold = SharedLowrankWeights {
                encoder: self
                    .slow_encoder
                    .as_ref()
                    .expect("split hierarchical slow encoder missing")
                    .val(),
                encoder_v: self
                    .slow_encoder_v
                    .as_ref()
                    .expect("split hierarchical slow encoder_v missing")
                    .val(),
                decoder: self
                    .slow_decoder
                    .as_ref()
                    .expect("split hierarchical slow decoder missing")
                    .val(),
            };
            Some(adapters.effective_slow(scaffold))
        } else {
            None
        };

        self.encoder = Self::replace_param_value(self.encoder, fast.encoder);
        self.encoder_v = Self::replace_param_value(self.encoder_v, fast.encoder_v);
        self.decoder = Self::replace_param_value(self.decoder, fast.decoder);
        if let Some(slow) = slow {
            self.slow_encoder = self
                .slow_encoder
                .map(|parameter| Self::replace_param_value(parameter, slow.encoder));
            self.slow_encoder_v = self
                .slow_encoder_v
                .map(|parameter| Self::replace_param_value(parameter, slow.encoder_v));
            self.slow_decoder = self
                .slow_decoder
                .map(|parameter| Self::replace_param_value(parameter, slow.decoder));
        }
        self.random_scaffold_adapters = None;
        self.random_scaffold_config.enabled = false;
        self
    }

    /// Returns all deterministic immutable random-scaffold parameters.
    ///
    /// Downstream checkpoint and distributed-training integrations use these
    /// IDs to derive a backend-independent catalog of mutable tensor paths.
    pub fn random_scaffold_frozen_param_ids(&self) -> Vec<burn::module::ParamId> {
        if !self.uses_random_scaffold() {
            return Vec::new();
        }
        let mut ids = vec![self.encoder.id, self.encoder_v.id, self.decoder.id];
        ids.extend(self.slow_encoder.as_ref().map(|parameter| parameter.id));
        ids.extend(self.slow_encoder_v.as_ref().map(|parameter| parameter.id));
        ids.extend(self.slow_decoder.as_ref().map(|parameter| parameter.id));
        if !self.random_scaffold_config.trainable_gain {
            ids.extend(
                self.random_scaffold_adapters
                    .as_ref()
                    .into_iter()
                    .flat_map(DragonRandomScaffoldAdapters::gain_ids),
            );
        }
        ids
    }

    pub(crate) fn random_scaffold_encoder_param_ids(&self) -> Vec<burn::module::ParamId> {
        self.random_scaffold_adapters
            .as_ref()
            .map(|adapters| adapters.encoder_ids(self.random_scaffold_config.trainable_gain))
            .unwrap_or_default()
    }

    pub(crate) fn random_scaffold_decoder_param_ids(&self) -> Vec<burn::module::ParamId> {
        self.random_scaffold_adapters
            .as_ref()
            .map(|adapters| adapters.decoder_ids(self.random_scaffold_config.trainable_gain))
            .unwrap_or_default()
    }

    pub fn with_shared_lowrank_weights(mut self, weights: SharedLowrankWeights<B>) -> Self {
        assert!(
            !self.uses_random_scaffold(),
            "with_shared_lowrank_weights cannot replace an immutable random scaffold"
        );
        assert_eq!(
            weights.encoder.shape().dims::<3>(),
            self.encoder.val().shape().dims::<3>(),
            "shared lowrank encoder shape mismatch"
        );
        assert_eq!(
            weights.encoder_v.shape().dims::<3>(),
            self.encoder_v.val().shape().dims::<3>(),
            "shared lowrank encoder_v shape mismatch"
        );
        assert_eq!(
            weights.decoder.shape().dims::<2>(),
            self.decoder.val().shape().dims::<2>(),
            "shared lowrank decoder shape mismatch"
        );
        self.encoder = Self::replace_param_value(self.encoder, weights.encoder);
        self.encoder_v = Self::replace_param_value(self.encoder_v, weights.encoder_v);
        self.decoder = Self::replace_param_value(self.decoder, weights.decoder);
        self
    }

    pub fn supports_shared_lowrank_population_forward(&self) -> bool {
        !self.uses_random_scaffold()
            && !self.y_neuron_recurrence.enabled
            && !self.hierarchical_dragon.enabled
            && self.rollout_fast_steps_per_slow_step == 1
            && self.language_head.uses_flat_token_logits()
    }

    /// Reports whether one sequence invocation can omit terminal recurrent-state materialization.
    pub fn supports_terminal_sequence_state_elision(&self) -> bool {
        self.sequence_kernel.memory_system == SequenceMemorySystem::LinearAttention
            && self.sequence_kernel.executor == SequenceTrainingExecutor::DenseScoreShortContext
            && self.rollout_fast_steps_per_slow_step == 1
            && !self.y_neuron_recurrence.enabled
            && !self.hierarchical_dragon.enabled
            && !self.clocked_slow_memory.enabled
            && !self.summary_memory.enabled
    }

    pub fn widen_latent_total(
        &self,
        target_config: DragonConfig,
        device: &B::Device,
    ) -> Result<(Self, DragonLatentWidenReport), String> {
        let fresh = DragonModel::<B>::new(target_config, device);
        self.widen_to_fresh_target(fresh)
    }

    pub fn widen_to_fresh_target(
        &self,
        fresh: Self,
    ) -> Result<(Self, DragonLatentWidenReport), String> {
        if self.uses_random_scaffold() || fresh.uses_random_scaffold() {
            return Err(
                "random-scaffold models do not support in-process latent widening; the scaffold growth history must be an explicit model contract"
                    .to_string(),
            );
        }
        let old_latent_total = self.latent_total_capacity();
        let old_latent_per_head = self.latent_per_head_capacity();
        let new_latent_total = fresh.latent_total_capacity();
        let new_latent_per_head = fresh.latent_per_head_capacity();

        if new_latent_total <= old_latent_total {
            return Err(format!(
                "target latent_total must exceed current latent_total (current={old_latent_total}, target={new_latent_total})"
            ));
        }
        if self.n_layer != fresh.n_layer {
            return Err(format!(
                "widening cannot change n_layer (current={} target={})",
                self.n_layer, fresh.n_layer
            ));
        }
        if self.n_embd != fresh.n_embd {
            return Err(format!(
                "widening cannot change n_embd (current={} target={})",
                self.n_embd, fresh.n_embd
            ));
        }
        if self.n_head != fresh.n_head {
            return Err(format!(
                "widening cannot change n_head (current={} target={})",
                self.n_head, fresh.n_head
            ));
        }
        if self.vocab_size != fresh.vocab_size {
            return Err(format!(
                "widening cannot change vocab_size (current={} target={})",
                self.vocab_size, fresh.vocab_size
            ));
        }
        if self.language_head != fresh.language_head {
            return Err("widening cannot change language_head".to_string());
        }
        if self.tie_input_output_embeddings != fresh.tie_input_output_embeddings {
            return Err("widening cannot change tie_input_output_embeddings".to_string());
        }
        if self.sequence_kernel != fresh.sequence_kernel {
            return Err(format!(
                "widening cannot change sequence_kernel (current={:?} target={:?})",
                self.sequence_kernel, fresh.sequence_kernel
            ));
        }
        if self.rollout_fast_steps_per_slow_step != fresh.rollout_fast_steps_per_slow_step {
            return Err(format!(
                "widening cannot change rollout_fast_steps_per_slow_step (current={} target={})",
                self.rollout_fast_steps_per_slow_step, fresh.rollout_fast_steps_per_slow_step
            ));
        }
        if self.residual_connector != fresh.residual_connector {
            return Err(format!(
                "widening cannot change residual_connector (current={:?} target={:?})",
                self.residual_connector, fresh.residual_connector
            ));
        }
        if self.mamba_config != fresh.mamba_config {
            return Err(format!(
                "widening cannot change mamba_config (current={:?} target={:?})",
                self.mamba_config, fresh.mamba_config
            ));
        }
        if self.latent_reasoning != fresh.latent_reasoning {
            return Err(format!(
                "widening cannot change latent_reasoning (current={:?} target={:?})",
                self.latent_reasoning, fresh.latent_reasoning
            ));
        }
        if self.next_latent_transition != fresh.next_latent_transition {
            return Err(format!(
                "widening cannot change next_latent_transition (current={:?} target={:?})",
                self.next_latent_transition, fresh.next_latent_transition
            ));
        }
        if self.hierarchical_dragon != fresh.hierarchical_dragon {
            return Err(format!(
                "widening cannot change hierarchical_dragon (current={:?} target={:?})",
                self.hierarchical_dragon, fresh.hierarchical_dragon
            ));
        }
        if !new_latent_total.is_multiple_of(self.n_head) {
            return Err(format!(
                "target latent_total must be divisible by n_head (target={new_latent_total}, n_head={})",
                self.n_head
            ));
        }
        let mut widened = fresh.clone();
        widened.embed = self.embed.clone();
        widened.embed.weight = Param::from_tensor(self.embed.weight.val());
        widened.dropout = self.dropout.clone();
        widened.norm = self.norm.value_clone();
        widened.x_relu_threshold = self.x_relu_threshold;
        widened.y_relu_threshold = self.y_relu_threshold;
        widened.attention = self.attention.widened_from_prefix(
            &fresh.attention,
            old_latent_per_head,
            new_latent_per_head,
        )?;
        widened.residual_connector = self.residual_connector;
        widened.mhc_shared = self.mhc_shared.clone();
        widened.attention_residual_shared = self.attention_residual_shared.clone();
        widened.block_attention_residual_shared = self.block_attention_residual_shared.clone();
        widened.mamba = self
            .mamba
            .as_ref()
            .map(MambaSequenceParameters::value_clone);
        widened.lm_head = self
            .lm_head
            .as_ref()
            .map(|head| Param::from_tensor(head.val()));
        widened.nca_factorized_lm_head = self
            .nca_factorized_lm_head
            .as_ref()
            .map(|head| Param::from_tensor(head.val()));
        widened.nca_special_lm_head = self
            .nca_special_lm_head
            .as_ref()
            .map(|head| Param::from_tensor(head.val()));
        widened.sequence_score_head = self
            .sequence_score_head
            .as_ref()
            .map(SequenceScoreHead::value_clone);
        widened.latent_refiner_in = self.latent_refiner_in.as_ref().map(clone_linear_value);
        widened.latent_refiner_out = self.latent_refiner_out.as_ref().map(clone_linear_value);
        widened.latent_refiner_gate = self
            .latent_refiner_gate
            .as_ref()
            .map(|gate| Param::from_tensor(gate.val()));
        widened.latent_energy_head = self.latent_energy_head.as_ref().map(clone_linear_value);
        widened.latent_stop_head = self.latent_stop_head.as_ref().map(clone_linear_value);
        widened.latent_jepa_predictor = self.latent_jepa_predictor.as_ref().map(clone_linear_value);
        widened.latent_step_decoder_embedding = self
            .latent_step_decoder_embedding
            .as_ref()
            .map(|embedding| Param::from_tensor(embedding.val()));
        widened.hierarchical_dragon = self.hierarchical_dragon.clone();
        widened.next_latent_transition_in = self
            .next_latent_transition_in
            .as_ref()
            .map(clone_linear_value);
        widened.next_latent_transition_mid = self
            .next_latent_transition_mid
            .as_ref()
            .map(clone_linear_value);
        widened.next_latent_transition_out = self
            .next_latent_transition_out
            .as_ref()
            .map(clone_linear_value);

        widened.encoder = Param::from_tensor(widen_3d_last_dim_prefix_zero_tail(
            self.encoder.val(),
            fresh.encoder.val(),
            old_latent_per_head,
            new_latent_per_head,
        )?);
        widened.encoder_v = Param::from_tensor(widen_3d_last_dim_prefix(
            self.encoder_v.val(),
            fresh.encoder_v.val(),
            old_latent_per_head,
            new_latent_per_head,
        )?);
        widened.decoder = Param::from_tensor(widen_2d_headed_row_prefix(
            self.decoder.val(),
            fresh.decoder.val(),
            self.n_head,
            old_latent_per_head,
            new_latent_per_head,
        )?);
        widened.slow_encoder = match (&self.slow_encoder, &fresh.slow_encoder) {
            (Some(current), Some(fresh)) => {
                Some(Param::from_tensor(widen_3d_last_dim_prefix_zero_tail(
                    current.val(),
                    fresh.val(),
                    old_latent_per_head,
                    new_latent_per_head,
                )?))
            }
            (None, None) => None,
            _ => {
                return Err("widening cannot change slow_encoder parameter presence".to_string());
            }
        };
        widened.slow_encoder_v = match (&self.slow_encoder_v, &fresh.slow_encoder_v) {
            (Some(current), Some(fresh)) => Some(Param::from_tensor(widen_3d_last_dim_prefix(
                current.val(),
                fresh.val(),
                old_latent_per_head,
                new_latent_per_head,
            )?)),
            (None, None) => None,
            _ => {
                return Err("widening cannot change slow_encoder_v parameter presence".to_string());
            }
        };
        widened.slow_decoder = match (&self.slow_decoder, &fresh.slow_decoder) {
            (Some(current), Some(fresh)) => Some(Param::from_tensor(widen_2d_headed_row_prefix(
                current.val(),
                fresh.val(),
                self.n_head,
                old_latent_per_head,
                new_latent_per_head,
            )?)),
            (None, None) => None,
            _ => {
                return Err("widening cannot change slow_decoder parameter presence".to_string());
            }
        };
        widened.gated_deltanet2 = match (&self.gated_deltanet2, &fresh.gated_deltanet2) {
            (Some(current), Some(fresh)) => Some(current.widened_from_prefix(
                fresh,
                old_latent_per_head,
                new_latent_per_head,
            )?),
            (None, None) => None,
            _ => {
                return Err("widening cannot change gated_deltanet2 parameter presence".to_string());
            }
        };
        widened.gated_deltanet2_upstream = match (
            &self.gated_deltanet2_upstream,
            &fresh.gated_deltanet2_upstream,
        ) {
            (Some(current), Some(fresh)) => Some(widen_upstream_gated_deltanet2_prefix(
                current,
                fresh,
                old_latent_per_head,
                new_latent_per_head,
            )?),
            (None, None) => None,
            _ => {
                return Err(
                    "widening cannot change upstream gated_deltanet2 parameter presence"
                        .to_string(),
                );
            }
        };

        let report = DragonLatentWidenReport {
            old_latent_total,
            new_latent_total,
            old_latent_per_head,
            new_latent_per_head,
            appended_latent_total: new_latent_total.saturating_sub(old_latent_total),
        };
        Ok((widened, report))
    }
}

mod autodiff;
mod forward;
mod latent;
mod population;

#[cfg(test)]
mod tests;
