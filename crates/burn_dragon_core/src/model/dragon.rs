mod auxiliary_memory;
mod connector;
mod continual_backprop;
mod diagnostics;
#[cfg(any(feature = "probe", test))]
mod interpretability;
mod language_head;
mod language_pipeline;
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

use burn::module::{Module, Param};
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData, activation};
use burn_dragon_kernel::api::attention::{
    supports_dense_causal_attention_backend, try_fused_dense_causal_attention_wgpu,
};
use burn_dragon_kernel::api::recurrent::{
    CompiledRecurrentAttentionPlan, supports_recurrent_backend, try_fused_recurrent_attention_wgpu,
    try_fused_recurrent_attention_wgpu_with_plan,
};
use burn_dragon_kernel::kernels::sequence::mamba3::forward::{
    Mamba3TensorizedState, tensorized_mamba3_forward, use_tensorized_mamba3_forward_experimental,
};
use burn_dragon_time::Instant;
use burn_gdn::{GatedDeltaNet2Executor, GatedDeltaNet2Memory, try_gdn2_chunk_wy};
use rand::distributions::{Distribution, WeightedIndex};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::ops::Range;
use std::sync::Once;

use super::attention::Attention;
use super::attention_residual::{
    AttentionResidual, BlockAttentionResidual, ResidualConnectorKind, ResidualHistory,
};
use super::config::{
    ClockedSlowMemoryConfig, DragonConfig, FusedKernelConfig, HierarchicalDragonConfig,
    HierarchicalDragonSharing, LanguageHeadConfig, LatentReasoningConfig,
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
#[cfg(any(feature = "viz", feature = "probe"))]
use super::residual_stream::lowrank_residual_step_branch_thresholds_relu_native;
use super::residual_stream::lowrank_residual_step_next_branch_thresholds;
#[cfg(any(feature = "probe", test))]
use super::residual_stream::lowrank_residual_step_with_metrics_branch_thresholds;
#[cfg(any(feature = "viz", feature = "probe"))]
use super::residual_stream::{decode_y_neuron_tail, decode_y_neuron_tail_uses_legacy_flat};
#[cfg(not(any(feature = "viz", feature = "probe")))]
use super::residual_stream::{
    decode_y_neuron_tail, decode_y_neuron_tail_uses_legacy_flat,
    lowrank_residual_step_next_branch_thresholds_relu_native,
};
use super::sequence::gdn2::{
    GatedDeltaNet2Implementation, GatedDeltaNet2Parameters, ResolvedGatedDeltaNet2Config,
    gated_deltanet2_reference, l2_normalize_last,
};
use super::sequence::linear::{
    expand_attention_values_to_heads, recurrent_attention_dense_score_final_rho_reference,
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
        let encoder = Param::from_tensor(initializer.headwise_projection_tensor::<B>(
            DragonProjectionRole::Encoder,
            config.n_head,
            config.n_embd,
            latent_per_head,
            residual_depth,
            device,
        ));

        let encoder_v = Param::from_tensor(initializer.headwise_projection_tensor::<B>(
            DragonProjectionRole::EncoderValue,
            config.n_head,
            config.n_embd,
            latent_per_head,
            residual_depth,
            device,
        ));

        let decoder = Param::from_tensor(initializer.projection_tensor::<B>(
            DragonProjectionRole::Decoder,
            latent_total,
            config.n_embd,
            residual_depth,
            device,
        ));
        let hierarchical_dragon = config.hierarchical_dragon.clone();
        let (slow_encoder, slow_encoder_v, slow_decoder) = if hierarchical_dragon.enabled
            && matches!(
                hierarchical_dragon.weight_sharing,
                HierarchicalDragonSharing::Split
            ) {
            (
                Some(Param::from_tensor(
                    initializer.headwise_projection_tensor::<B>(
                        DragonProjectionRole::Encoder,
                        config.n_head,
                        config.n_embd,
                        latent_per_head,
                        residual_depth,
                        device,
                    ),
                )),
                Some(Param::from_tensor(
                    initializer.headwise_projection_tensor::<B>(
                        DragonProjectionRole::EncoderValue,
                        config.n_head,
                        config.n_embd,
                        latent_per_head,
                        residual_depth,
                        device,
                    ),
                )),
                Some(Param::from_tensor(Tensor::<B, 2>::zeros(
                    [latent_total, config.n_embd],
                    device,
                ))),
            )
        } else {
            (None, None, None)
        };
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
            mamba_config,
            mamba,
            gated_deltanet2_config,
            gated_deltanet2,
            gated_deltanet2_upstream,
            lm_head,
            nca_factorized_lm_head,
            nca_special_lm_head,
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

    pub fn with_shared_lowrank_weights(mut self, weights: SharedLowrankWeights<B>) -> Self {
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
        !self.y_neuron_recurrence.enabled
            && !self.hierarchical_dragon.enabled
            && self.rollout_fast_steps_per_slow_step == 1
            && self.language_head.uses_flat_token_logits()
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
        if new_latent_total % self.n_head != 0 {
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

    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut state = self.init_state();
        self.forward_with_state(tokens, &mut state)
    }

    pub fn forward_with_summary_event_mask(
        &self,
        tokens: Tensor<B, 2, Int>,
        summary_event_mask: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let mut state = self.init_state();
        self.forward_with_state_and_summary_event_mask(tokens, summary_event_mask, &mut state)
    }

    pub fn forward_with_hidden(&self, tokens: Tensor<B, 2, Int>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let mut state = self.init_state();
        self.forward_with_hidden_and_state(tokens, &mut state)
    }

    pub fn forward_with_shared_lowrank_population(
        &self,
        tokens: Tensor<B, 2, Int>,
        lowrank: SharedLowrankPopulationWeights<B>,
    ) -> Tensor<B, 3>
    where
        B::Device: 'static,
        B::FloatTensorPrimitive: 'static,
    {
        assert!(
            !self.y_neuron_recurrence.enabled,
            "shared lowrank population forward does not support y-neuron recurrence"
        );
        assert!(
            !self.hierarchical_dragon.enabled,
            "shared lowrank population forward does not support hierarchical Dragon"
        );
        assert_eq!(
            self.rollout_fast_steps_per_slow_step, 1,
            "shared lowrank population forward requires rollout_fast_steps_per_slow_step = 1"
        );
        assert!(
            self.language_head.uses_flat_token_logits(),
            "shared lowrank population forward requires flat token logits"
        );
        let population = lowrank.population_size();
        assert!(population > 0, "population size must be > 0");
        self.assert_shared_lowrank_population_shapes(&lowrank);

        let embedded = self.embed.forward(tokens);
        let embedded_population = Tensor::cat(
            (0..population)
                .map(|_| embedded.clone())
                .collect::<Vec<_>>(),
            0,
        );
        let mut state = self.init_state();
        let hidden = self.forward_hidden_with_shared_lowrank_population_from_embedded_single_pass(
            embedded_population,
            &mut state,
            &lowrank,
            population,
        );
        self.project_hidden_to_logits(hidden)
    }

    pub fn forward_with_shared_lowrank_population_factors(
        &self,
        tokens: Tensor<B, 2, Int>,
        factors: SharedLowrankPopulationFactors<B>,
    ) -> Tensor<B, 3>
    where
        B::Device: 'static,
        B::FloatTensorPrimitive: 'static,
    {
        assert!(
            !self.y_neuron_recurrence.enabled,
            "shared lowrank population factor forward does not support y-neuron recurrence"
        );
        assert!(
            !self.hierarchical_dragon.enabled,
            "shared lowrank population factor forward does not support hierarchical Dragon"
        );
        assert_eq!(
            self.rollout_fast_steps_per_slow_step, 1,
            "shared lowrank population factor forward requires rollout_fast_steps_per_slow_step = 1"
        );
        assert!(
            self.language_head.uses_flat_token_logits(),
            "shared lowrank population factor forward requires flat token logits"
        );
        let population = factors.population_size();
        assert!(population > 0, "population size must be > 0");
        self.assert_shared_lowrank_population_factor_shapes(&factors);

        let embedded = self.embed.forward(tokens);
        let embedded_population = Tensor::cat(
            (0..population)
                .map(|_| embedded.clone())
                .collect::<Vec<_>>(),
            0,
        );
        let mut state = self.init_state();
        let hidden = self
            .forward_hidden_with_shared_lowrank_population_factors_from_embedded_single_pass(
                embedded_population,
                &mut state,
                &factors,
                population,
            );
        self.project_hidden_to_logits(hidden)
    }

    pub fn embed_tokens(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.embed.forward(tokens)
    }

    pub fn begin_language_pipeline_from_embedded(
        &self,
        embedded: Tensor<B, 3>,
    ) -> LanguagePipelineState<B> {
        assert_eq!(
            self.rollout_fast_steps_per_slow_step, 1,
            "language pipeline execution currently requires rollout_fast_steps_per_slow_step = 1"
        );
        assert!(
            !self.y_neuron_recurrence.enabled,
            "language pipeline execution is not supported with y-neuron recurrence enabled"
        );
        assert!(
            !self.hierarchical_dragon.enabled,
            "language pipeline execution is not supported with hierarchical Dragon enabled"
        );
        self.initialize_language_pipeline_state(embedded)
    }

    pub fn begin_language_pipeline(&self, tokens: Tensor<B, 2, Int>) -> LanguagePipelineState<B> {
        self.begin_language_pipeline_from_embedded(self.embed.forward(tokens))
    }

    pub fn forward_language_pipeline_stage_with_state(
        &self,
        pipeline_state: LanguagePipelineState<B>,
        state: &mut ModelState<B>,
        layer_range: Range<usize>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> LanguagePipelineState<B> {
        self.forward_language_pipeline_state_layer_range(
            pipeline_state,
            state,
            state.position,
            RecurrentPositionMode::Sequential,
            summary_event_mask,
            layer_range,
        )
    }

    pub fn finish_language_pipeline_hidden_with_state(
        &self,
        pipeline_state: LanguagePipelineState<B>,
        state: &mut ModelState<B>,
    ) -> Tensor<B, 3> {
        let hidden = self.collapse_language_streams(pipeline_state.current);
        let [_batch, time, _dim] = hidden.shape().dims::<3>();
        state.position = state.position.saturating_add(time);
        hidden
    }

    pub fn finish_language_pipeline_with_state(
        &self,
        pipeline_state: LanguagePipelineState<B>,
        state: &mut ModelState<B>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let hidden = self.finish_language_pipeline_hidden_with_state(pipeline_state, state);
        let logits = self.logits_from_hidden(hidden.clone());
        (hidden, logits)
    }

    pub fn rollout_fast_steps_per_slow_step(&self) -> usize {
        self.rollout_fast_steps_per_slow_step
    }

    pub fn forward_fast(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.forward(tokens)
    }

    pub fn forward_fast_with_summary_event_mask(
        &self,
        tokens: Tensor<B, 2, Int>,
        summary_event_mask: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        self.forward_with_summary_event_mask(tokens, summary_event_mask)
    }

    pub fn generate(
        &self,
        mut indices: Tensor<B, 2, Int>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
    ) -> Tensor<B, 2, Int> {
        let [batch, _] = indices.shape().dims();
        assert_eq!(batch, 1, "generation currently supports batch size 1");

        let mut state = self.init_state();
        let mut logits = self.forward_with_state(indices.clone(), &mut state);
        let [_, mut time, vocab] = logits.shape().dims();
        assert_eq!(time, indices.shape().dims::<2>()[1]);

        let mut last_logits = logits
            .slice_dim(1, (time - 1)..time)
            .reshape([vocab])
            .div_scalar(temperature);

        for _ in 0..max_new_tokens {
            let mut logits_values = last_logits
                .clone()
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("logits to vec");

            if let Some(k) = top_k
                && k > 0
                && k < vocab
            {
                let mut sorted = logits_values.clone();
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
                let threshold = sorted[k - 1];
                for value in logits_values.iter_mut() {
                    if *value < threshold {
                        *value = f32::NEG_INFINITY;
                    }
                }
            }

            let max_logit = logits_values
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut probs: Vec<f32> = logits_values
                .iter()
                .map(|value| (value - max_logit).exp())
                .collect();
            let sum: f32 = probs.iter().sum();
            if sum == 0.0 || sum.is_nan() {
                let uniform = 1.0 / vocab as f32;
                for p in probs.iter_mut() {
                    *p = uniform;
                }
            } else {
                for p in probs.iter_mut() {
                    *p /= sum;
                }
            }

            let dist = WeightedIndex::new(&probs).expect("valid probability distribution");
            let mut rng = thread_rng();
            let next = dist.sample(&mut rng) as i64;

            let next_token = Tensor::<B, 2, Int>::from_data(
                TensorData::new(vec![next], [1, 1]),
                &indices.device(),
            );
            indices = Tensor::cat(vec![indices, next_token.clone()], 1);

            logits = self.forward_with_state(next_token, &mut state);
            let [_, new_time, _] = logits.shape().dims();
            time = new_time;
            last_logits = logits
                .slice_dim(1, (time - 1)..time)
                .reshape([vocab])
                .div_scalar(temperature);
        }

        indices
    }

    pub fn init_state(&self) -> ModelState<B> {
        ModelState::new(self.n_layer)
    }

    pub fn init_state_ephemeral(&self) -> ModelState<B> {
        ModelState::new_ephemeral(self.n_layer)
    }

    fn layer_latent_total(&self, layer_idx: usize) -> usize {
        self.layer_latent_totals
            .get(layer_idx)
            .copied()
            .unwrap_or(self.mlp_internal_dim_multiplier * self.n_embd)
    }

    fn resolve_linear_attention_rho_state(
        &self,
        layer_state: &LayerState<B>,
        _device: &B::Device,
    ) -> Option<Tensor<B, 4>> {
        layer_state.rho.as_ref().cloned()
    }

    fn write_linear_attention_rho_state(&self, layer_state: &mut LayerState<B>, rho: Tensor<B, 4>) {
        layer_state.rho = Some(rho);
        layer_state.rho_norm = None;
        layer_state.sequence_aux = None;
    }

    fn layer_latent_per_head(&self, layer_idx: usize) -> usize {
        let total = self.layer_latent_total(layer_idx);
        assert_eq!(
            total % self.n_head,
            0,
            "layer latent total must divide evenly across heads"
        );
        total / self.n_head
    }

    fn layer_lowrank_weights_from_params(
        &self,
        layer_idx: usize,
        encoder_param: &Param<Tensor<B, 3>>,
        encoder_v_param: &Param<Tensor<B, 3>>,
        decoder_param: &Param<Tensor<B, 2>>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 2>, usize) {
        let latent_per_head = self.layer_latent_per_head(layer_idx);
        let capacity_per_head = self.latent_per_head_capacity();
        let encoder = encoder_param
            .val()
            .slice([0..self.n_head, 0..self.n_embd, 0..latent_per_head])
            .reshape([1, self.n_head, self.n_embd, latent_per_head]);
        let encoder_v = encoder_v_param
            .val()
            .slice([0..self.n_head, 0..self.n_embd, 0..latent_per_head])
            .reshape([1, self.n_head, self.n_embd, latent_per_head]);
        let decoder_capacity = decoder_param.val();
        let decoder = Tensor::cat(
            (0..self.n_head)
                .map(|head| {
                    let start = head * capacity_per_head;
                    decoder_capacity
                        .clone()
                        .slice([start..start + latent_per_head, 0..self.n_embd])
                })
                .collect(),
            0,
        );
        (encoder, encoder_v, decoder, latent_per_head)
    }

    fn layer_lowrank_weights(
        &self,
        layer_idx: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 2>, usize) {
        self.layer_lowrank_weights_from_params(
            layer_idx,
            &self.encoder,
            &self.encoder_v,
            &self.decoder,
        )
    }

    fn layer_lowrank_weights_for_hierarchical_branch(
        &self,
        layer_idx: usize,
        branch: HierarchicalDragonBranch,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 2>, usize) {
        if branch == HierarchicalDragonBranch::Slow
            && matches!(
                self.hierarchical_dragon.weight_sharing,
                HierarchicalDragonSharing::Split
            )
        {
            return self.layer_lowrank_weights_from_params(
                layer_idx,
                self.slow_encoder
                    .as_ref()
                    .expect("split hierarchical slow encoder missing"),
                self.slow_encoder_v
                    .as_ref()
                    .expect("split hierarchical slow encoder_v missing"),
                self.slow_decoder
                    .as_ref()
                    .expect("split hierarchical slow decoder missing"),
            );
        }
        self.layer_lowrank_weights(layer_idx)
    }

    fn hierarchical_dragon_applies_to_layer(&self, layer_idx: usize) -> bool {
        if !self.hierarchical_dragon.enabled {
            return false;
        }
        let first_layer = self
            .hierarchical_dragon
            .last_layers
            .map(|last_layers| self.n_layer.max(1).saturating_sub(last_layers))
            .unwrap_or(0);
        layer_idx >= first_layer
    }

    fn hierarchical_slow_sequence_slot(&self) -> HierarchicalSequenceSlot {
        if matches!(
            self.hierarchical_dragon.rho_sharing,
            HierarchicalDragonSharing::Split
        ) {
            HierarchicalSequenceSlot::Slow
        } else {
            HierarchicalSequenceSlot::Fast
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn recurrent_attention_with_plan_in_hierarchical_slot(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        layer_state: &mut LayerState<B>,
        position: usize,
        position_mode: RecurrentPositionMode,
        fused_plan: Option<&CompiledRecurrentAttentionPlan<B>>,
        slot: HierarchicalSequenceSlot,
    ) -> Tensor<B, 4> {
        match slot {
            HierarchicalSequenceSlot::Fast => self.recurrent_attention_with_plan(
                query,
                value,
                layer_state,
                position,
                position_mode,
                fused_plan,
            ),
            HierarchicalSequenceSlot::Slow => {
                layer_state.swap_fast_slow_sequence_state();
                let context = self.recurrent_attention_with_plan(
                    query,
                    value,
                    layer_state,
                    position,
                    position_mode,
                    fused_plan,
                );
                layer_state.swap_fast_slow_sequence_state();
                context
            }
        }
    }

    fn hierarchical_slow_hidden(
        &self,
        layer_state: &LayerState<B>,
        flat_batch: usize,
        branch_dim: usize,
    ) -> Option<Tensor<B, 4>> {
        let hidden = layer_state.hierarchical_slow_hidden.as_ref()?;
        (hidden.shape().dims::<4>() == [flat_batch, 1, 1, branch_dim]).then(|| hidden.clone())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_hierarchical_lowrank_step(
        &self,
        branch_flat: Tensor<B, 4>,
        layer_state: &mut LayerState<B>,
        layer_idx: usize,
        start_pos: usize,
        position_mode: RecurrentPositionMode,
        branch: HierarchicalDragonBranch,
        slot: HierarchicalSequenceSlot,
    ) -> Tensor<B, 4> {
        let [flat_batch, views, branch_time, branch_dim] = branch_flat.shape().dims::<4>();
        debug_assert_eq!(views, 1, "hierarchical Dragon expects a single branch view");
        if branch_time == 0 {
            return branch_flat;
        }

        let (encoder, encoder_v, decoder, latent) =
            self.layer_lowrank_weights_for_hierarchical_branch(layer_idx, branch);
        let fused = self.kernel.enabled;
        let latent_pattern = &self.kernel.block_sparse.latent;
        let sparse_mask = if fused && latent_pattern.is_sparse() {
            Some(latent_pattern.mask::<B>(latent, &branch_flat.device()))
        } else {
            None
        };
        let fused_recurrent_plan = if matches!(
            (
                self.sequence_kernel.memory_system,
                self.sequence_kernel.executor,
            ),
            (
                SequenceMemorySystem::LinearAttention,
                SequenceTrainingExecutor::Reference,
            )
        ) && self.kernel.enabled
            && self.kernel.wgpu_recurrent_kernel
            && supports_recurrent_backend::<B>()
        {
            Some(CompiledRecurrentAttentionPlan::new(
                flat_batch,
                self.n_head,
                1,
                branch_time,
                latent,
                branch_dim,
                &branch_flat.device(),
            ))
        } else {
            None
        };

        let x_neuron = self.project_lowrank_positive(LowrankProjectionRequest {
            dense: branch_flat.clone(),
            projector: encoder,
            relu_threshold: self.x_relu_threshold,
            use_fused: fused && self.kernel.projection_executor.use_x(),
            latent_pattern,
            sparse_mask: sparse_mask.clone(),
        });
        let context = self.recurrent_attention_with_plan_in_hierarchical_slot(
            x_neuron.clone(),
            branch_flat.clone(),
            layer_state,
            start_pos,
            position_mode,
            fused_recurrent_plan.as_ref(),
            slot,
        );
        let y_gate = self.project_lowrank_positive(LowrankProjectionRequest {
            dense: self.norm.forward(context),
            projector: encoder_v,
            relu_threshold: self.y_relu_threshold,
            use_fused: fused && self.kernel.projection_executor.use_y(),
            latent_pattern,
            sparse_mask,
        });
        let y_neuron = self.dropout.forward(x_neuron * y_gate);
        if branch == HierarchicalDragonBranch::Fast
            && let Some(runtime) = self.shared_lowrank_continual_backprop_runtime()
            && runtime.should_sample_step()
        {
            runtime.record_y_neuron_stats(y_neuron.clone());
        }
        let mixed = y_neuron.swap_dims(1, 2);
        let mixed_flat = mixed.reshape([flat_batch * branch_time, self.n_head * latent]);
        let mlp_flat = mixed_flat.matmul(decoder);
        let mlp_out = mlp_flat.reshape([flat_batch, 1, branch_time, branch_dim]);
        let mlp_out = self.norm.forward(mlp_out);
        self.norm.forward(branch_flat + mlp_out)
    }

    fn forward_hierarchical_branch_layer(
        &self,
        branch_flat: Tensor<B, 4>,
        layer_state: &mut LayerState<B>,
        layer_idx: usize,
        start_pos: usize,
        position_mode: RecurrentPositionMode,
    ) -> Tensor<B, 4> {
        let [flat_batch, _views, branch_time, branch_dim] = branch_flat.shape().dims::<4>();
        if branch_time == 0 {
            return branch_flat;
        }

        let mut fast = branch_flat;
        let mut slow_hidden = self.hierarchical_slow_hidden(layer_state, flat_batch, branch_dim);
        let fast_cycles = self.hierarchical_dragon.fast_cycles.max(1);
        let slow_cycles = self.hierarchical_dragon.slow_cycles.max(1);
        let slow_to_fast_scale = self.hierarchical_dragon.slow_to_fast_scale.max(0.0);
        let fast_to_slow_scale = self.hierarchical_dragon.fast_to_slow_scale.max(0.0);
        let slow_slot = self.hierarchical_slow_sequence_slot();

        for _ in 0..slow_cycles {
            if slow_to_fast_scale > 0.0
                && let Some(slow) = slow_hidden.as_ref()
            {
                fast = self.norm.forward(
                    fast + slow
                        .clone()
                        .repeat_dim(2, branch_time)
                        .mul_scalar(slow_to_fast_scale),
                );
            }
            for _ in 0..fast_cycles {
                fast = self.forward_hierarchical_lowrank_step(
                    fast,
                    layer_state,
                    layer_idx,
                    start_pos,
                    position_mode,
                    HierarchicalDragonBranch::Fast,
                    HierarchicalSequenceSlot::Fast,
                );
            }

            let mut slow_input = fast
                .clone()
                .mean_dim(2)
                .reshape([flat_batch, 1, 1, branch_dim]);
            if fast_to_slow_scale > 0.0
                && let Some(previous_slow) = slow_hidden.as_ref()
            {
                slow_input = self
                    .norm
                    .forward(slow_input + previous_slow.clone().mul_scalar(fast_to_slow_scale));
            }
            let next_slow = self.forward_hierarchical_lowrank_step(
                slow_input,
                layer_state,
                layer_idx,
                start_pos,
                position_mode,
                HierarchicalDragonBranch::Slow,
                slow_slot,
            );
            slow_hidden = Some(next_slow.clone());
            if slow_to_fast_scale > 0.0 {
                fast = self.norm.forward(
                    fast + next_slow
                        .repeat_dim(2, branch_time)
                        .mul_scalar(slow_to_fast_scale),
                );
            }
        }

        layer_state.hierarchical_slow_hidden = slow_hidden;
        fast
    }

    fn assert_shared_lowrank_population_shapes(&self, lowrank: &SharedLowrankPopulationWeights<B>) {
        let [population, heads, embd, latent_capacity] = lowrank.encoder.shape().dims::<4>();
        assert!(population > 0, "population size must be > 0");
        assert_eq!(heads, self.n_head, "population encoder heads mismatch");
        assert_eq!(
            embd, self.n_embd,
            "population encoder embedding dim mismatch"
        );
        assert_eq!(
            latent_capacity,
            self.latent_per_head_capacity(),
            "population encoder latent capacity mismatch"
        );
        assert_eq!(
            lowrank.encoder_v.shape().dims::<4>(),
            [population, self.n_head, self.n_embd, latent_capacity],
            "population encoder_v shape mismatch"
        );
        assert_eq!(
            lowrank.decoder.shape().dims::<3>(),
            [population, self.latent_total_capacity(), self.n_embd],
            "population decoder shape mismatch"
        );
    }

    fn assert_shared_lowrank_population_factor_shapes(
        &self,
        factors: &SharedLowrankPopulationFactors<B>,
    ) {
        let population = factors.population_size();
        let [encoder_population, heads, embd, encoder_rank] = factors.encoder_a.shape().dims::<4>();
        let [
            encoder_b_population,
            encoder_b_heads,
            latent_capacity,
            encoder_b_rank,
        ] = factors.encoder_b.shape().dims::<4>();
        assert!(population > 0, "population size must be > 0");
        assert_eq!(
            encoder_population, population,
            "population encoder factor count mismatch"
        );
        assert_eq!(
            heads, self.n_head,
            "population encoder factor heads mismatch"
        );
        assert_eq!(
            embd, self.n_embd,
            "population encoder factor embedding dim mismatch"
        );
        assert_eq!(
            encoder_b_population, population,
            "population encoder factor-b count mismatch"
        );
        assert_eq!(
            encoder_b_heads, self.n_head,
            "population encoder factor-b heads mismatch"
        );
        assert_eq!(
            latent_capacity,
            self.latent_per_head_capacity(),
            "population encoder factor latent capacity mismatch"
        );
        assert_eq!(
            encoder_b_rank, encoder_rank,
            "population encoder factor rank mismatch"
        );
        assert_eq!(
            factors.encoder_v_a.shape().dims::<4>(),
            [population, self.n_head, self.n_embd, encoder_rank],
            "population encoder_v factor-a shape mismatch"
        );
        assert_eq!(
            factors.encoder_v_b.shape().dims::<4>(),
            [
                population,
                self.n_head,
                self.latent_per_head_capacity(),
                encoder_rank,
            ],
            "population encoder_v factor-b shape mismatch"
        );

        let [decoder_population, decoder_rows, decoder_rank] =
            factors.decoder_a.shape().dims::<3>();
        let [decoder_b_population, decoder_cols, decoder_b_rank] =
            factors.decoder_b.shape().dims::<3>();
        assert_eq!(
            decoder_population, population,
            "population decoder factor-a count mismatch"
        );
        assert_eq!(
            decoder_rows,
            self.latent_total_capacity(),
            "population decoder factor-a rows mismatch"
        );
        assert_eq!(
            decoder_b_population, population,
            "population decoder factor-b count mismatch"
        );
        assert_eq!(
            decoder_cols, self.n_embd,
            "population decoder factor-b cols mismatch"
        );
        assert_eq!(
            decoder_b_rank, decoder_rank,
            "population decoder factor rank mismatch"
        );
    }

    fn population_layer_lowrank_weights(
        &self,
        layer_idx: usize,
        lowrank: &SharedLowrankPopulationWeights<B>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 3>, usize) {
        let latent_per_head = self.layer_latent_per_head(layer_idx);
        let capacity_per_head = self.latent_per_head_capacity();
        let population = lowrank.population_size();
        let encoder = lowrank.encoder.clone().slice([
            0..population,
            0..self.n_head,
            0..self.n_embd,
            0..latent_per_head,
        ]);
        let encoder_v = lowrank.encoder_v.clone().slice([
            0..population,
            0..self.n_head,
            0..self.n_embd,
            0..latent_per_head,
        ]);
        let decoder = Tensor::cat(
            (0..self.n_head)
                .map(|head| {
                    let start = head * capacity_per_head;
                    lowrank.decoder.clone().slice([
                        0..population,
                        start..start + latent_per_head,
                        0..self.n_embd,
                    ])
                })
                .collect(),
            1,
        );
        (encoder, encoder_v, decoder, latent_per_head)
    }

    fn population_layer_lowrank_factors(
        &self,
        layer_idx: usize,
        factors: &SharedLowrankPopulationFactors<B>,
    ) -> (
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 4>,
        Tensor<B, 3>,
        Tensor<B, 3>,
        Tensor<B, 1>,
        usize,
    ) {
        let latent_per_head = self.layer_latent_per_head(layer_idx);
        let capacity_per_head = self.latent_per_head_capacity();
        let population = factors.population_size();
        let encoder_rank = factors.encoder_a.shape().dims::<4>()[3];
        let decoder_rank = factors.decoder_a.shape().dims::<3>()[2];
        let encoder_a = factors.encoder_a.clone();
        let encoder_b = factors.encoder_b.clone().slice([
            0..population,
            0..self.n_head,
            0..latent_per_head,
            0..encoder_rank,
        ]);
        let encoder_v_a = factors.encoder_v_a.clone();
        let encoder_v_b = factors.encoder_v_b.clone().slice([
            0..population,
            0..self.n_head,
            0..latent_per_head,
            0..encoder_rank,
        ]);
        let decoder_a = Tensor::cat(
            (0..self.n_head)
                .map(|head| {
                    let start = head * capacity_per_head;
                    factors.decoder_a.clone().slice([
                        0..population,
                        start..start + latent_per_head,
                        0..decoder_rank,
                    ])
                })
                .collect(),
            1,
        );
        (
            encoder_a,
            encoder_b,
            encoder_v_a,
            encoder_v_b,
            decoder_a,
            factors.decoder_b.clone(),
            factors.signs.clone(),
            latent_per_head,
        )
    }

    fn project_shared_lowrank_population_positive(
        &self,
        dense: Tensor<B, 4>,
        projector: Tensor<B, 4>,
        population: usize,
        relu_threshold: f32,
        use_fused: bool,
        latent_pattern: &crate::kernel::BlockPattern1d,
        sparse_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 4>
    where
        B::FloatTensorPrimitive: 'static,
    {
        let [flat_batch, streams, time, embd] = dense.shape().dims::<4>();
        assert_eq!(
            flat_batch % population,
            0,
            "population flat batch must divide evenly"
        );
        let per_population_batch = flat_batch / population;
        if population == 1 {
            return self.project_lowrank_positive(LowrankProjectionRequest {
                dense,
                projector,
                relu_threshold,
                use_fused,
                latent_pattern,
                sparse_mask,
            });
        }

        let [projector_population, heads, projector_embd, latent] = projector.shape().dims::<4>();
        assert_eq!(
            projector_population, population,
            "population projector count mismatch"
        );
        assert_eq!(projector_embd, embd, "population projector dim mismatch");
        if use_fused {
            return crate::kernel::relu_lowrank::fused_forward_with_executor(
                dense,
                projector,
                None,
                relu_threshold,
                latent_pattern,
                sparse_mask,
                self.kernel.lowrank_grad_input_executor,
            );
        }

        let mut projected = if streams == 1 {
            let dense_grouped = dense.reshape([population, per_population_batch * time, embd]);
            let projector_grouped =
                projector
                    .swap_dims(1, 2)
                    .reshape([population, embd, heads * latent]);
            dense_grouped
                .matmul(projector_grouped)
                .reshape([population, per_population_batch, time, heads, latent])
                .swap_dims(2, 3)
                .reshape([flat_batch, heads, time, latent])
        } else if streams == heads {
            let dense_grouped = dense
                .reshape([population, per_population_batch, heads, time, embd])
                .swap_dims(1, 2)
                .reshape([population, heads, per_population_batch * time, embd]);
            dense_grouped
                .matmul(projector)
                .reshape([population, heads, per_population_batch, time, latent])
                .swap_dims(1, 2)
                .reshape([flat_batch, heads, time, latent])
        } else {
            return Tensor::cat(
                (0..population)
                    .map(|population_idx| {
                        let start = population_idx * per_population_batch;
                        let end = start + per_population_batch;
                        let dense_slice = dense.clone().slice_dim(0, start..end);
                        let projector_slice = projector
                            .clone()
                            .slice_dim(0, population_idx..population_idx + 1);
                        self.project_lowrank_positive(LowrankProjectionRequest {
                            dense: dense_slice,
                            projector: projector_slice,
                            relu_threshold,
                            use_fused,
                            latent_pattern,
                            sparse_mask: sparse_mask.clone(),
                        })
                    })
                    .collect(),
                0,
            );
        };

        if relu_threshold != 0.0 {
            projected = projected.sub_scalar(relu_threshold);
        }
        let mut activated = activation::relu(projected);
        if latent_pattern.is_sparse() {
            let mask = sparse_mask
                .unwrap_or_else(|| latent_pattern.mask::<B>(latent, &activated.device()));
            activated = activated * mask;
        }
        activated
    }

    fn project_shared_lowrank_population_factorized_positive(
        &self,
        dense: Tensor<B, 4>,
        base_projector: Tensor<B, 4>,
        factor_a: Tensor<B, 4>,
        factor_b: Tensor<B, 4>,
        signs: Tensor<B, 1>,
        sigma_scale: f64,
        population: usize,
        relu_threshold: f32,
        latent_pattern: &crate::kernel::BlockPattern1d,
        sparse_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 4> {
        let [flat_batch, streams, time, embd] = dense.shape().dims::<4>();
        assert_eq!(
            flat_batch % population,
            0,
            "population flat batch must divide evenly"
        );
        let per_population_batch = flat_batch / population;
        let [factor_population, heads, factor_embd, rank] = factor_a.shape().dims::<4>();
        let [factor_b_population, factor_b_heads, latent, factor_b_rank] =
            factor_b.shape().dims::<4>();
        assert_eq!(
            factor_population, population,
            "population factor count mismatch"
        );
        assert_eq!(
            factor_b_population, population,
            "population factor-b count mismatch"
        );
        assert_eq!(factor_b_heads, heads, "population factor head mismatch");
        assert_eq!(factor_embd, embd, "population factor embedding mismatch");
        assert_eq!(factor_b_rank, rank, "population factor rank mismatch");

        let mut projected = if streams == 1 || streams == heads {
            let base_projected = dense.clone().matmul(base_projector);
            let steps = per_population_batch * time;
            let correction = if streams == 1 {
                let dense_grouped = dense.reshape([population, steps, embd]);
                dense_grouped
                    .reshape([population, 1, steps, embd])
                    .repeat_dim(1, heads)
                    .matmul(factor_a)
                    .matmul(factor_b.swap_dims(2, 3))
            } else {
                let dense_grouped = dense
                    .reshape([population, per_population_batch, heads, time, embd])
                    .swap_dims(1, 2)
                    .reshape([population, heads, steps, embd]);
                dense_grouped
                    .matmul(factor_a)
                    .matmul(factor_b.swap_dims(2, 3))
            };
            let correction = correction
                * signs
                    .clone()
                    .reshape([population, 1, 1, 1])
                    .mul_scalar(sigma_scale);
            let correction = correction
                .reshape([population, heads, per_population_batch, time, latent])
                .swap_dims(1, 2)
                .reshape([flat_batch, heads, time, latent]);
            base_projected + correction
        } else {
            Tensor::cat(
                (0..population)
                    .map(|population_idx| {
                        let start = population_idx * per_population_batch;
                        let end = start + per_population_batch;
                        let dense_slice = dense.clone().slice_dim(0, start..end);
                        let delta = factor_a
                            .clone()
                            .slice_dim(0, population_idx..population_idx + 1)
                            .matmul(
                                factor_b
                                    .clone()
                                    .slice_dim(0, population_idx..population_idx + 1)
                                    .swap_dims(2, 3),
                            )
                            * signs
                                .clone()
                                .slice_dim(0, population_idx..population_idx + 1)
                                .reshape([1, 1, 1, 1])
                                .mul_scalar(sigma_scale);
                        dense_slice.matmul(base_projector.clone() + delta)
                    })
                    .collect(),
                0,
            )
        };

        if relu_threshold != 0.0 {
            projected = projected.sub_scalar(relu_threshold);
        }
        let mut activated = activation::relu(projected);
        if latent_pattern.is_sparse() {
            let mask = sparse_mask
                .unwrap_or_else(|| latent_pattern.mask::<B>(latent, &activated.device()));
            activated = activated * mask;
        }
        activated
    }

    fn decode_shared_lowrank_population_tail(
        &self,
        y_neuron: Tensor<B, 4>,
        decoder: Tensor<B, 3>,
        population: usize,
    ) -> Tensor<B, 4> {
        let [flat_batch, heads, time, latent] = y_neuron.shape().dims::<4>();
        assert_eq!(
            flat_batch % population,
            0,
            "population y-neuron batch must divide evenly"
        );
        let per_population_batch = flat_batch / population;
        if population == 1 {
            let decoder_rows = decoder.shape().dims::<3>()[1];
            return decode_y_neuron_tail(y_neuron, decoder.reshape([decoder_rows, self.n_embd]));
        }

        let [decoder_population, decoder_rows, decoder_dim] = decoder.shape().dims::<3>();
        assert_eq!(
            decoder_population, population,
            "population decoder count mismatch"
        );
        assert_eq!(
            decoder_rows,
            heads * latent,
            "population decoder latent rows mismatch"
        );
        assert_eq!(decoder_dim, self.n_embd, "population decoder dim mismatch");

        if decode_y_neuron_tail_uses_legacy_flat() {
            let mixed = y_neuron
                .reshape([population, per_population_batch, heads, time, latent])
                .swap_dims(2, 3)
                .reshape([population, per_population_batch * time, heads * latent]);
            return mixed
                .matmul(decoder)
                .reshape([population, per_population_batch, time, self.n_embd])
                .reshape([flat_batch, 1, time, self.n_embd]);
        }

        let y_by_head = y_neuron
            .reshape([population, per_population_batch, heads, time, latent])
            .swap_dims(1, 2)
            .reshape([population, heads, per_population_batch * time, latent]);
        let decoder_by_head = decoder.reshape([population, heads, latent, self.n_embd]);
        y_by_head
            .matmul(decoder_by_head)
            .sum_dim(1)
            .reshape([population, per_population_batch, time, self.n_embd])
            .reshape([flat_batch, 1, time, self.n_embd])
    }

    fn decode_shared_lowrank_population_factors_tail(
        &self,
        y_neuron: Tensor<B, 4>,
        base_decoder: Tensor<B, 2>,
        factor_a: Tensor<B, 3>,
        factor_b: Tensor<B, 3>,
        signs: Tensor<B, 1>,
        sigma_scale: f64,
        population: usize,
    ) -> Tensor<B, 4> {
        let [flat_batch, heads, time, latent] = y_neuron.shape().dims::<4>();
        assert_eq!(
            flat_batch % population,
            0,
            "population y-neuron batch must divide evenly"
        );
        let per_population_batch = flat_batch / population;
        let [factor_population, factor_rows, rank] = factor_a.shape().dims::<3>();
        let [factor_b_population, decoder_dim, factor_b_rank] = factor_b.shape().dims::<3>();
        assert_eq!(
            factor_population, population,
            "population decoder factor count mismatch"
        );
        assert_eq!(
            factor_b_population, population,
            "population decoder factor-b count mismatch"
        );
        assert_eq!(
            factor_rows,
            heads * latent,
            "population decoder factor row mismatch"
        );
        assert_eq!(factor_b_rank, rank, "population decoder rank mismatch");

        let base = decode_y_neuron_tail(y_neuron.clone(), base_decoder);
        let steps = per_population_batch * time;
        let y_flat = y_neuron
            .reshape([population, per_population_batch, heads, time, latent])
            .swap_dims(2, 3)
            .reshape([population, steps, heads * latent]);
        let correction = y_flat.matmul(factor_a).matmul(factor_b.swap_dims(1, 2))
            * signs.reshape([population, 1, 1]).mul_scalar(sigma_scale);
        let correction = correction.reshape([flat_batch, 1, time, decoder_dim]);
        base + correction
    }

    fn project_lowrank_positive(&self, request: LowrankProjectionRequest<'_, B>) -> Tensor<B, 4>
    where
        B::FloatTensorPrimitive: 'static,
    {
        let LowrankProjectionRequest {
            dense,
            projector,
            relu_threshold,
            use_fused,
            latent_pattern,
            sparse_mask,
        } = request;
        if use_fused {
            crate::kernel::relu_lowrank::fused_forward_with_executor(
                dense,
                projector,
                None,
                relu_threshold,
                latent_pattern,
                sparse_mask,
                self.kernel.lowrank_grad_input_executor,
            )
        } else {
            let mut latent = dense.matmul(projector);
            if relu_threshold != 0.0 {
                latent = latent.sub_scalar(relu_threshold);
            }
            activation::relu(latent)
        }
    }

    fn forward_with_state_impl(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let embedded = self.embed.forward(tokens);
        self.forward_with_state_from_embedded(embedded, state, summary_event_mask)
    }

    fn forward_hidden_with_state_impl(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        let embedded = self.embed.forward(tokens);
        self.forward_hidden_with_state_from_embedded(embedded, state, summary_event_mask)
    }

    fn forward_with_state_from_embedded(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        if self.rollout_fast_steps_per_slow_step <= 1 {
            let start_pos = state.position;
            return self.forward_with_state_from_embedded_single_pass(
                embedded,
                state,
                start_pos,
                true,
                RecurrentPositionMode::Sequential,
                summary_event_mask,
            );
        }

        match self.rollout_executor_mode() {
            RolloutExecutorMode::HostLoop => self
                .forward_with_state_from_embedded_rollout_host_loop(
                    embedded,
                    state,
                    summary_event_mask,
                ),
            RolloutExecutorMode::WgpuFused => self.forward_with_state_from_embedded_rollout_fused(
                embedded,
                state,
                summary_event_mask,
            ),
        }
    }

    fn forward_hidden_raw_with_state_from_embedded(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        if self.rollout_fast_steps_per_slow_step <= 1 {
            let start_pos = state.position;
            return self.forward_hidden_with_state_from_embedded_single_pass(
                embedded,
                state,
                start_pos,
                true,
                RecurrentPositionMode::Sequential,
                summary_event_mask,
            );
        }

        match self.rollout_executor_mode() {
            RolloutExecutorMode::HostLoop => self
                .forward_hidden_with_state_from_embedded_rollout_host_loop(
                    embedded,
                    state,
                    summary_event_mask,
                ),
            RolloutExecutorMode::WgpuFused => self
                .forward_hidden_with_state_from_embedded_rollout_fused(
                    embedded,
                    state,
                    summary_event_mask,
                ),
        }
    }

    fn forward_hidden_with_state_from_embedded(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        let hidden =
            self.forward_hidden_raw_with_state_from_embedded(embedded, state, summary_event_mask);
        self.reason_hidden_final(hidden)
    }

    fn latent_decoder_step(&self) -> usize {
        if self.latent_reasoning_enabled() {
            self.latent_reasoning.max_steps
        } else {
            0
        }
    }

    fn forward_with_state_from_embedded_rollout_host_loop(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        assert_eq!(
            state.layers.len(),
            self.n_layer,
            "model state layers mismatch"
        );
        let [batch, slow_steps, _embd] = embedded.shape().dims::<3>();

        if slow_steps == 0 {
            let device = embedded.device();
            let hidden = Tensor::<B, 3>::zeros([batch, 0, self.n_embd], &device);
            let logits = Tensor::<B, 3>::zeros([batch, 0, self.vocab_size], &device);
            return (hidden, logits);
        }

        let mut hidden_slow = Vec::with_capacity(slow_steps);
        let mut logits_slow = Vec::with_capacity(slow_steps);
        for slow_idx in 0..slow_steps {
            let token_embedded = embedded.clone().slice_dim(1, slow_idx..slow_idx + 1);
            let token_summary_event_mask = summary_event_mask
                .as_ref()
                .map(|mask| mask.clone().slice_dim(1, slow_idx..slow_idx + 1));
            let start_pos = state.position;
            let mut hidden_last = None;
            let mut logits_last = None;
            for _ in 0..self.rollout_fast_steps_per_slow_step {
                let (hidden, logits) = self.forward_with_state_from_embedded_single_pass(
                    token_embedded.clone(),
                    state,
                    start_pos,
                    false,
                    RecurrentPositionMode::Sequential,
                    token_summary_event_mask.clone(),
                );
                hidden_last = Some(hidden);
                logits_last = Some(logits);
            }
            hidden_slow.push(hidden_last.expect("rollout hidden output"));
            logits_slow.push(logits_last.expect("rollout logits output"));
            state.position = state.position.saturating_add(1);
        }

        (Tensor::cat(hidden_slow, 1), Tensor::cat(logits_slow, 1))
    }

    fn forward_hidden_with_state_from_embedded_rollout_host_loop(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        assert_eq!(
            state.layers.len(),
            self.n_layer,
            "model state layers mismatch"
        );
        let [batch, slow_steps, _embd] = embedded.shape().dims::<3>();

        if slow_steps == 0 {
            let device = embedded.device();
            return Tensor::<B, 3>::zeros([batch, 0, self.n_embd], &device);
        }

        let mut hidden_slow = Vec::with_capacity(slow_steps);
        for slow_idx in 0..slow_steps {
            let token_embedded = embedded.clone().slice_dim(1, slow_idx..slow_idx + 1);
            let token_summary_event_mask = summary_event_mask
                .as_ref()
                .map(|mask| mask.clone().slice_dim(1, slow_idx..slow_idx + 1));
            let start_pos = state.position;
            let mut hidden_last = None;
            for _ in 0..self.rollout_fast_steps_per_slow_step {
                let hidden = self.forward_hidden_with_state_from_embedded_single_pass(
                    token_embedded.clone(),
                    state,
                    start_pos,
                    false,
                    RecurrentPositionMode::Sequential,
                    token_summary_event_mask.clone(),
                );
                hidden_last = Some(hidden);
            }
            hidden_slow.push(hidden_last.expect("rollout hidden output"));
            state.position = state.position.saturating_add(1);
        }

        Tensor::cat(hidden_slow, 1)
    }

    fn forward_with_state_from_embedded_rollout_fused(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        assert_eq!(
            state.layers.len(),
            self.n_layer,
            "model state layers mismatch"
        );
        let [batch, slow_steps, _embd] = embedded.shape().dims::<3>();

        if slow_steps == 0 {
            let device = embedded.device();
            let hidden = Tensor::<B, 3>::zeros([batch, 0, self.n_embd], &device);
            let logits = Tensor::<B, 3>::zeros([batch, 0, self.vocab_size], &device);
            return (hidden, logits);
        }

        let fast_steps = self.rollout_fast_steps_per_slow_step;
        let mut hidden_slow = Vec::with_capacity(slow_steps);
        let mut logits_slow = Vec::with_capacity(slow_steps);

        for slow_idx in 0..slow_steps {
            let token_embedded = embedded.clone().slice_dim(1, slow_idx..slow_idx + 1);
            let rollout_embedded = token_embedded.repeat_dim(1, fast_steps);
            let token_summary_event_mask = summary_event_mask
                .as_ref()
                .map(|mask| mask.clone().slice_dim(1, slow_idx..slow_idx + 1));
            let start_pos = state.position;
            let hidden_rollout = self.forward_hidden_with_state_from_embedded_single_pass(
                rollout_embedded,
                state,
                start_pos,
                false,
                RecurrentPositionMode::Fixed,
                token_summary_event_mask,
            );
            let last = fast_steps - 1;
            let hidden_last =
                self.reason_hidden_final(hidden_rollout.slice_dim(1, last..fast_steps));
            let logits_last = self.logits_from_hidden(hidden_last.clone());
            hidden_slow.push(hidden_last);
            logits_slow.push(logits_last);
            state.position = state.position.saturating_add(1);
        }

        (Tensor::cat(hidden_slow, 1), Tensor::cat(logits_slow, 1))
    }

    fn forward_hidden_with_state_from_embedded_rollout_fused(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        assert_eq!(
            state.layers.len(),
            self.n_layer,
            "model state layers mismatch"
        );
        let [batch, slow_steps, _embd] = embedded.shape().dims::<3>();

        if slow_steps == 0 {
            let device = embedded.device();
            return Tensor::<B, 3>::zeros([batch, 0, self.n_embd], &device);
        }

        let fast_steps = self.rollout_fast_steps_per_slow_step;
        let mut hidden_slow = Vec::with_capacity(slow_steps);

        for slow_idx in 0..slow_steps {
            let token_embedded = embedded.clone().slice_dim(1, slow_idx..slow_idx + 1);
            let rollout_embedded = token_embedded.repeat_dim(1, fast_steps);
            let token_summary_event_mask = summary_event_mask
                .as_ref()
                .map(|mask| mask.clone().slice_dim(1, slow_idx..slow_idx + 1));
            let start_pos = state.position;
            let hidden_rollout = self.forward_hidden_with_state_from_embedded_single_pass(
                rollout_embedded,
                state,
                start_pos,
                false,
                RecurrentPositionMode::Fixed,
                token_summary_event_mask,
            );
            let last = fast_steps - 1;
            let hidden_last = hidden_rollout.slice_dim(1, last..fast_steps);
            hidden_slow.push(hidden_last);
            state.position = state.position.saturating_add(1);
        }

        Tensor::cat(hidden_slow, 1)
    }

    fn forward_hidden_with_shared_lowrank_population_from_embedded_single_pass(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
        lowrank: &SharedLowrankPopulationWeights<B>,
        population: usize,
    ) -> Tensor<B, 3>
    where
        B::Device: 'static,
        B::FloatTensorPrimitive: 'static,
    {
        assert_eq!(
            state.layers.len(),
            self.n_layer,
            "model state layers mismatch"
        );
        assert!(
            !self.y_neuron_recurrence.enabled,
            "population lowrank forward does not support y-neuron recurrence"
        );
        let [batch, time, embd] = embedded.shape().dims::<3>();
        assert_eq!(batch % population, 0, "population batch must divide evenly");
        let start_pos = state.position;
        let mut current = self.norm.forward(embedded.reshape([batch, 1, time, embd]));
        let fused = self.kernel.enabled;
        let static_mhc_coefficients = self.mhc_shared.as_ref().and_then(|mhc| {
            (!mhc.coefficient_policy().uses_dynamic_stream_controller()).then(|| mhc.coefficients())
        });
        let mut residual_history = self.initialize_language_residual_history(&current);

        for (layer_idx, layer_state) in state.layers.iter_mut().enumerate() {
            let connector = self.residual_connector_for_layer(layer_idx);
            let current_before = residual_history.capture_previous(&current);
            let mhc_coefficients = match connector {
                ResidualConnectorRef::Mhc(_) => static_mhc_coefficients.as_ref(),
                ResidualConnectorRef::Vanilla
                | ResidualConnectorRef::AttentionResidual(_)
                | ResidualConnectorRef::BlockAttentionResidual(_) => None,
            };
            let bindings = self.split_language_residuals_for_layer(
                current,
                &connector,
                residual_history.as_slice(),
                mhc_coefficients,
            );
            let LanguageMhcSplitBindings {
                branch_input,
                merge: merge_bindings,
            } = bindings;
            layer_state.clocked_slow_hidden = None;
            layer_state.summary_memory_hidden = None;

            let [branch_batch, branch_views, branch_time, branch_dim] =
                branch_input.shape().dims::<4>();
            let flat_batch = branch_batch * branch_views;
            assert_eq!(
                flat_batch % population,
                0,
                "population branch batch must divide evenly"
            );
            let branch_flat = branch_input.reshape([flat_batch, 1, branch_time, branch_dim]);
            let (encoder, encoder_v, decoder, latent) =
                self.population_layer_lowrank_weights(layer_idx, lowrank);
            let heads = self.n_head;
            let latent_pattern = &self.kernel.block_sparse.latent;
            let sparse_mask = if fused && latent_pattern.is_sparse() {
                Some(latent_pattern.mask::<B>(latent, &branch_flat.device()))
            } else {
                None
            };
            let fused_recurrent_plan = if matches!(
                (
                    self.sequence_kernel.memory_system,
                    self.sequence_kernel.executor,
                ),
                (
                    SequenceMemorySystem::LinearAttention,
                    SequenceTrainingExecutor::Reference,
                )
            ) && self.kernel.enabled
                && self.kernel.wgpu_recurrent_kernel
                && supports_recurrent_backend::<B>()
            {
                Some(CompiledRecurrentAttentionPlan::new(
                    flat_batch,
                    heads,
                    1,
                    branch_time,
                    latent,
                    branch_dim,
                    &branch_flat.device(),
                ))
            } else {
                None
            };

            let x_neuron = self.project_shared_lowrank_population_positive(
                branch_flat.clone(),
                encoder,
                population,
                self.x_relu_threshold,
                fused && self.kernel.projection_executor.use_x(),
                latent_pattern,
                sparse_mask.clone(),
            );
            let attn = self.recurrent_attention_with_plan(
                x_neuron.clone(),
                branch_flat.clone(),
                layer_state,
                start_pos,
                RecurrentPositionMode::Sequential,
                fused_recurrent_plan.as_ref(),
            );
            let attn = self.norm.forward(attn);
            let y_gate = self.project_shared_lowrank_population_positive(
                attn,
                encoder_v,
                population,
                self.y_relu_threshold,
                fused && self.kernel.projection_executor.use_y(),
                latent_pattern,
                sparse_mask,
            );
            let y_neuron = self.dropout.forward(x_neuron * y_gate);
            let mlp_out = self.decode_shared_lowrank_population_tail(y_neuron, decoder, population);
            let mlp_out = self.norm.forward(mlp_out);
            let branch_out = self.norm.forward(branch_flat + mlp_out).reshape([
                branch_batch,
                branch_views,
                branch_time,
                branch_dim,
            ]);
            let next = self.merge_language_residuals_for_layer(
                branch_out,
                merge_bindings,
                &connector,
                mhc_coefficients,
            );
            current = if self.residual_connector_needs_post_merge_norm(&connector) {
                self.norm.forward(next)
            } else {
                next
            };
            self.update_language_residual_history(&mut residual_history, current_before, &current);
        }

        let hidden = self.collapse_language_streams(current);
        let [_batch, time, _dim] = hidden.shape().dims::<3>();
        state.position = state.position.saturating_add(time);
        hidden
    }

    fn forward_hidden_with_shared_lowrank_population_factors_from_embedded_single_pass(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
        factors: &SharedLowrankPopulationFactors<B>,
        population: usize,
    ) -> Tensor<B, 3>
    where
        B::Device: 'static,
        B::FloatTensorPrimitive: 'static,
    {
        assert_eq!(
            state.layers.len(),
            self.n_layer,
            "model state layers mismatch"
        );
        assert!(
            !self.y_neuron_recurrence.enabled,
            "population lowrank factor forward does not support y-neuron recurrence"
        );
        let [batch, time, embd] = embedded.shape().dims::<3>();
        assert_eq!(batch % population, 0, "population batch must divide evenly");
        let start_pos = state.position;
        let mut current = self.norm.forward(embedded.reshape([batch, 1, time, embd]));
        let fused = self.kernel.enabled;
        let static_mhc_coefficients = self.mhc_shared.as_ref().and_then(|mhc| {
            (!mhc.coefficient_policy().uses_dynamic_stream_controller()).then(|| mhc.coefficients())
        });
        let mut residual_history = self.initialize_language_residual_history(&current);

        for (layer_idx, layer_state) in state.layers.iter_mut().enumerate() {
            let connector = self.residual_connector_for_layer(layer_idx);
            let current_before = residual_history.capture_previous(&current);
            let mhc_coefficients = match connector {
                ResidualConnectorRef::Mhc(_) => static_mhc_coefficients.as_ref(),
                ResidualConnectorRef::Vanilla
                | ResidualConnectorRef::AttentionResidual(_)
                | ResidualConnectorRef::BlockAttentionResidual(_) => None,
            };
            let bindings = self.split_language_residuals_for_layer(
                current,
                &connector,
                residual_history.as_slice(),
                mhc_coefficients,
            );
            let LanguageMhcSplitBindings {
                branch_input,
                merge: merge_bindings,
            } = bindings;
            layer_state.clocked_slow_hidden = None;
            layer_state.summary_memory_hidden = None;

            let [branch_batch, branch_views, branch_time, branch_dim] =
                branch_input.shape().dims::<4>();
            let flat_batch = branch_batch * branch_views;
            assert_eq!(
                flat_batch % population,
                0,
                "population branch batch must divide evenly"
            );
            let branch_flat = branch_input.reshape([flat_batch, 1, branch_time, branch_dim]);
            let (base_encoder, base_encoder_v, base_decoder, latent) =
                self.layer_lowrank_weights(layer_idx);
            let (
                encoder_a,
                encoder_b,
                encoder_v_a,
                encoder_v_b,
                decoder_a,
                decoder_b,
                signs,
                factor_latent,
            ) = self.population_layer_lowrank_factors(layer_idx, factors);
            assert_eq!(
                latent, factor_latent,
                "population factor latent slice mismatch"
            );
            let heads = self.n_head;
            let latent_pattern = &self.kernel.block_sparse.latent;
            let sparse_mask = if fused && latent_pattern.is_sparse() {
                Some(latent_pattern.mask::<B>(latent, &branch_flat.device()))
            } else {
                None
            };
            let fused_recurrent_plan = if matches!(
                (
                    self.sequence_kernel.memory_system,
                    self.sequence_kernel.executor,
                ),
                (
                    SequenceMemorySystem::LinearAttention,
                    SequenceTrainingExecutor::Reference,
                )
            ) && self.kernel.enabled
                && self.kernel.wgpu_recurrent_kernel
                && supports_recurrent_backend::<B>()
            {
                Some(CompiledRecurrentAttentionPlan::new(
                    flat_batch,
                    heads,
                    1,
                    branch_time,
                    latent,
                    branch_dim,
                    &branch_flat.device(),
                ))
            } else {
                None
            };

            let x_neuron = self.project_shared_lowrank_population_factorized_positive(
                branch_flat.clone(),
                base_encoder,
                encoder_a,
                encoder_b,
                signs.clone(),
                factors.sigma as f64 * factors.encoder_scale,
                population,
                self.x_relu_threshold,
                latent_pattern,
                sparse_mask.clone(),
            );
            let attn = self.recurrent_attention_with_plan(
                x_neuron.clone(),
                branch_flat.clone(),
                layer_state,
                start_pos,
                RecurrentPositionMode::Sequential,
                fused_recurrent_plan.as_ref(),
            );
            let attn = self.norm.forward(attn);
            let y_gate = self.project_shared_lowrank_population_factorized_positive(
                attn,
                base_encoder_v,
                encoder_v_a,
                encoder_v_b,
                signs.clone(),
                factors.sigma as f64 * factors.encoder_v_scale,
                population,
                self.y_relu_threshold,
                latent_pattern,
                sparse_mask,
            );
            let y_neuron = self.dropout.forward(x_neuron * y_gate);
            let mlp_out = self.decode_shared_lowrank_population_factors_tail(
                y_neuron,
                base_decoder,
                decoder_a,
                decoder_b,
                signs,
                factors.sigma as f64 * factors.decoder_scale,
                population,
            );
            let mlp_out = self.norm.forward(mlp_out);
            let branch_out = self.norm.forward(branch_flat + mlp_out).reshape([
                branch_batch,
                branch_views,
                branch_time,
                branch_dim,
            ]);
            let next = self.merge_language_residuals_for_layer(
                branch_out,
                merge_bindings,
                &connector,
                mhc_coefficients,
            );
            current = if self.residual_connector_needs_post_merge_norm(&connector) {
                self.norm.forward(next)
            } else {
                next
            };
            self.update_language_residual_history(&mut residual_history, current_before, &current);
        }

        let hidden = self.collapse_language_streams(current);
        let [_batch, time, _dim] = hidden.shape().dims::<3>();
        state.position = state.position.saturating_add(time);
        hidden
    }

    fn forward_hidden_with_state_from_embedded_single_pass_y_neuron_recurrence(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
        start_pos: usize,
        advance_position: bool,
        position_mode: RecurrentPositionMode,
    ) -> Tensor<B, 3> {
        assert_eq!(
            state.layers.len(),
            self.n_layer,
            "model state layers mismatch"
        );
        let [batch, time, embd] = embedded.shape().dims::<3>();
        let mut current = self.norm.forward(embedded.reshape([batch, 1, time, embd]));
        let fused = self.kernel.enabled;
        let static_mhc_coefficients = self.mhc_shared.as_ref().and_then(|mhc| {
            (!mhc.coefficient_policy().uses_dynamic_stream_controller()).then(|| mhc.coefficients())
        });
        let mut residual_history = self.initialize_language_residual_history(&current);

        for (layer_idx, layer_state) in state.layers.iter_mut().enumerate() {
            let connector = self.residual_connector_for_layer(layer_idx);
            let current_before = residual_history.capture_previous(&current);
            let mhc_coefficients = match connector {
                ResidualConnectorRef::Mhc(_) => static_mhc_coefficients.as_ref(),
                ResidualConnectorRef::Vanilla
                | ResidualConnectorRef::AttentionResidual(_)
                | ResidualConnectorRef::BlockAttentionResidual(_) => None,
            };
            let bindings = self.split_language_residuals_for_layer(
                current,
                &connector,
                residual_history.as_slice(),
                mhc_coefficients,
            );
            let LanguageMhcSplitBindings {
                branch_input,
                merge: merge_bindings,
            } = bindings;
            layer_state.clocked_slow_hidden = None;
            layer_state.summary_memory_hidden = None;

            let [branch_batch, branch_views, branch_time, branch_dim] =
                branch_input.shape().dims::<4>();
            let flat_batch = branch_batch * branch_views;
            let branch_flat = branch_input.reshape([flat_batch, 1, branch_time, branch_dim]);
            let (encoder, encoder_v, decoder, latent) = self.layer_lowrank_weights(layer_idx);
            let heads = self.n_head;
            let latent_pattern = &self.kernel.block_sparse.latent;
            let sparse_mask = if fused && latent_pattern.is_sparse() {
                Some(latent_pattern.mask::<B>(latent, &branch_flat.device()))
            } else {
                None
            };
            if !self.y_neuron_recurrence_applies_to_layer(layer_idx) {
                layer_state.y_neuron_state = None;
                let fused_recurrent_plan = if matches!(
                    (
                        self.sequence_kernel.memory_system,
                        self.sequence_kernel.executor,
                    ),
                    (
                        SequenceMemorySystem::LinearAttention,
                        SequenceTrainingExecutor::Reference,
                    )
                ) && self.kernel.enabled
                    && self.kernel.wgpu_recurrent_kernel
                    && supports_recurrent_backend::<B>()
                {
                    Some(CompiledRecurrentAttentionPlan::new(
                        flat_batch,
                        heads,
                        1,
                        branch_time,
                        latent,
                        branch_dim,
                        &branch_flat.device(),
                    ))
                } else {
                    None
                };
                #[cfg(any(feature = "viz", feature = "probe"))]
                let output = lowrank_residual_step_branch_thresholds_relu_native(
                    branch_flat,
                    encoder.clone(),
                    encoder_v.clone(),
                    decoder.clone(),
                    &self.dropout,
                    fused && self.kernel.projection_executor.use_x(),
                    fused && self.kernel.projection_executor.use_y(),
                    self.x_relu_threshold,
                    self.y_relu_threshold,
                    true,
                    latent_pattern,
                    self.kernel.lowrank_grad_input_executor,
                    sparse_mask.clone(),
                    |query, value| {
                        self.recurrent_attention_with_plan(
                            query,
                            value,
                            layer_state,
                            start_pos,
                            position_mode,
                            fused_recurrent_plan.as_ref(),
                        )
                    },
                    |values| activation::relu(values),
                    |values| self.norm.forward(values),
                );
                #[cfg(not(any(feature = "viz", feature = "probe")))]
                let branch_out = lowrank_residual_step_next_branch_thresholds_relu_native(
                    branch_flat,
                    encoder.clone(),
                    encoder_v.clone(),
                    decoder.clone(),
                    &self.dropout,
                    fused && self.kernel.projection_executor.use_x(),
                    fused && self.kernel.projection_executor.use_y(),
                    self.x_relu_threshold,
                    self.y_relu_threshold,
                    true,
                    latent_pattern,
                    self.kernel.lowrank_grad_input_executor,
                    sparse_mask.clone(),
                    |query, value| {
                        self.recurrent_attention_with_plan(
                            query,
                            value,
                            layer_state,
                            start_pos,
                            position_mode,
                            fused_recurrent_plan.as_ref(),
                        )
                    },
                    |values| activation::relu(values),
                    |values| self.norm.forward(values),
                );

                #[cfg(any(feature = "viz", feature = "probe"))]
                if branch_time > 0 {
                    let last = branch_time - 1;
                    let viz_batch = branch_batch.max(1);
                    let viz_views = branch_views.max(1);
                    let x_neuron_last = output
                        .x_neuron
                        .clone()
                        .slice_dim(2, last..branch_time)
                        .reshape([viz_batch, viz_views, heads, latent])
                        .mean_dim(1)
                        .slice_dim(0, 0..1)
                        .reshape([heads, latent]);
                    let y_gate_last = output
                        .y_gate
                        .clone()
                        .slice_dim(2, last..branch_time)
                        .reshape([viz_batch, viz_views, heads, latent])
                        .mean_dim(1)
                        .slice_dim(0, 0..1)
                        .reshape([heads, latent]);
                    let y_neuron_last = output
                        .y_neuron
                        .clone()
                        .slice_dim(2, last..branch_time)
                        .reshape([viz_batch, viz_views, heads, latent])
                        .mean_dim(1)
                        .slice_dim(0, 0..1)
                        .reshape([heads, latent]);
                    let device = x_neuron_last.device();
                    let rho_last =
                        match self.resolve_linear_attention_rho_state(layer_state, &device) {
                            Some(rho) => {
                                let dims = rho.shape().dims::<4>();
                                if dims == [flat_batch, heads, latent, self.n_embd] {
                                    let rho_energy =
                                        rho.clone().abs().sum_dim(3).div_scalar(self.n_embd as f32);
                                    let rho_energy = rho_energy
                                        .reshape([viz_batch, viz_views, heads, latent])
                                        .mean_dim(1)
                                        .sum_dim(0)
                                        .div_scalar(viz_batch as f32);
                                    rho_energy.reshape([heads, latent])
                                } else {
                                    Tensor::<B, 2>::zeros([heads, latent], &device)
                                }
                            }
                            None => Tensor::<B, 2>::zeros([heads, latent], &device),
                        };

                    layer_state.viz = Some(LayerVizState {
                        x_neuron_last,
                        y_gate_last,
                        y_neuron_last,
                        rho_last,
                    });
                }

                #[cfg(any(feature = "viz", feature = "probe"))]
                let branch_out =
                    output
                        .next
                        .reshape([branch_batch, branch_views, branch_time, branch_dim]);
                #[cfg(not(any(feature = "viz", feature = "probe")))]
                let branch_out =
                    branch_out.reshape([branch_batch, branch_views, branch_time, branch_dim]);
                let next = self.merge_language_residuals_for_layer(
                    branch_out,
                    merge_bindings,
                    &connector,
                    mhc_coefficients,
                );
                current = if self.residual_connector_needs_post_merge_norm(&connector) {
                    self.norm.forward(next)
                } else {
                    next
                };
                self.update_language_residual_history(
                    &mut residual_history,
                    current_before,
                    &current,
                );
                continue;
            }
            let x_base = self.project_lowrank_positive(LowrankProjectionRequest {
                dense: branch_flat.clone(),
                projector: encoder.clone(),
                relu_threshold: self.x_relu_threshold,
                use_fused: fused,
                latent_pattern,
                sparse_mask: sparse_mask.clone(),
            });
            let mut next_tokens = Vec::with_capacity(branch_time);
            let mut y_neuron_state = self.resolve_y_neuron_state(
                layer_state,
                flat_batch,
                heads,
                latent,
                &branch_flat.device(),
            );
            let chunk_tokens = self
                .y_neuron_recurrence
                .chunk_tokens
                .max(1)
                .min(branch_time.max(1));
            let fused_recurrent_plan = if matches!(
                (
                    self.sequence_kernel.memory_system,
                    self.sequence_kernel.executor,
                ),
                (
                    SequenceMemorySystem::LinearAttention,
                    SequenceTrainingExecutor::Reference,
                )
            ) && self.kernel.enabled
                && self.kernel.wgpu_recurrent_kernel
                && supports_recurrent_backend::<B>()
            {
                Some(CompiledRecurrentAttentionPlan::new(
                    flat_batch,
                    heads,
                    1,
                    chunk_tokens,
                    latent,
                    branch_dim,
                    &branch_flat.device(),
                ))
            } else {
                None
            };
            let tail_plan = if matches!(
                (
                    self.sequence_kernel.memory_system,
                    self.sequence_kernel.executor,
                ),
                (
                    SequenceMemorySystem::LinearAttention,
                    SequenceTrainingExecutor::Reference,
                )
            ) && self.kernel.enabled
                && self.kernel.wgpu_recurrent_kernel
                && supports_recurrent_backend::<B>()
                && branch_time % chunk_tokens != 0
            {
                let tail_tokens = branch_time % chunk_tokens;
                Some(CompiledRecurrentAttentionPlan::new(
                    flat_batch,
                    heads,
                    1,
                    tail_tokens,
                    latent,
                    branch_dim,
                    &branch_flat.device(),
                ))
            } else {
                None
            };

            #[cfg(any(feature = "viz", feature = "probe"))]
            let mut viz_last: Option<(Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>)> = None;

            for chunk_start in (0..branch_time).step_by(chunk_tokens) {
                let chunk_end = (chunk_start + chunk_tokens).min(branch_time);
                let chunk_len = chunk_end - chunk_start;
                let x_neuron_base = x_base.clone().slice_dim(2, chunk_start..chunk_end);
                let x_neuron = self.inject_y_neuron_state(x_neuron_base, y_neuron_state.clone());
                let current_token = branch_flat.clone().slice_dim(2, chunk_start..chunk_end);
                let token_position = match position_mode {
                    RecurrentPositionMode::Sequential => start_pos + chunk_start,
                    RecurrentPositionMode::Fixed => start_pos,
                };
                let a_dense = self.recurrent_attention_with_plan(
                    x_neuron.clone(),
                    current_token.clone(),
                    layer_state,
                    token_position,
                    position_mode,
                    if chunk_len == chunk_tokens {
                        fused_recurrent_plan.as_ref()
                    } else {
                        tail_plan.as_ref()
                    },
                );
                let a_dense = self.norm.forward(a_dense);
                let y_gate = self.project_lowrank_positive(LowrankProjectionRequest {
                    dense: a_dense,
                    projector: encoder_v.clone(),
                    relu_threshold: self.y_relu_threshold,
                    use_fused: fused,
                    latent_pattern,
                    sparse_mask: sparse_mask.clone(),
                });
                let y_neuron = self.dropout.forward(x_neuron.clone() * y_gate.clone());
                let mixed = y_neuron.clone().swap_dims(1, 2);
                let mixed_flat = mixed.reshape([flat_batch * chunk_len, heads * latent]);
                let mlp_flat = mixed_flat.matmul(decoder.clone());
                let mlp_out = mlp_flat.reshape([flat_batch, 1, chunk_len, branch_dim]);
                let mlp_out = self.norm.forward(mlp_out);
                next_tokens.push(self.norm.forward(current_token + mlp_out));
                let y_neuron_last = y_neuron.clone().slice_dim(2, (chunk_len - 1)..chunk_len);
                y_neuron_state = self.update_y_neuron_state(y_neuron_state, y_neuron_last);

                #[cfg(any(feature = "viz", feature = "probe"))]
                if chunk_end == branch_time {
                    let last_start = chunk_len - 1;
                    viz_last = Some((
                        x_neuron.slice_dim(2, last_start..chunk_len),
                        y_gate.slice_dim(2, last_start..chunk_len),
                        y_neuron.slice_dim(2, last_start..chunk_len),
                    ));
                }
            }

            layer_state.y_neuron_state = Some(y_neuron_state);

            #[cfg(any(feature = "viz", feature = "probe"))]
            if let Some((x_neuron_last_raw, y_gate_last_raw, y_neuron_last_raw)) = viz_last {
                let viz_batch = branch_batch.max(1);
                let viz_views = branch_views.max(1);
                let x_neuron_last = x_neuron_last_raw
                    .reshape([viz_batch, viz_views, heads, latent])
                    .mean_dim(1)
                    .slice_dim(0, 0..1)
                    .reshape([heads, latent]);
                let y_gate_last = y_gate_last_raw
                    .reshape([viz_batch, viz_views, heads, latent])
                    .mean_dim(1)
                    .slice_dim(0, 0..1)
                    .reshape([heads, latent]);
                let y_neuron_last = y_neuron_last_raw
                    .reshape([viz_batch, viz_views, heads, latent])
                    .mean_dim(1)
                    .slice_dim(0, 0..1)
                    .reshape([heads, latent]);
                let device = x_neuron_last.device();
                let rho_last = match self.resolve_linear_attention_rho_state(layer_state, &device) {
                    Some(rho) => {
                        let dims = rho.shape().dims::<4>();
                        if dims == [flat_batch, heads, latent, self.n_embd] {
                            let rho_energy =
                                rho.clone().abs().sum_dim(3).div_scalar(self.n_embd as f32);
                            let rho_energy = rho_energy
                                .reshape([viz_batch, viz_views, heads, latent])
                                .mean_dim(1)
                                .sum_dim(0)
                                .div_scalar(viz_batch as f32);
                            rho_energy.reshape([heads, latent])
                        } else {
                            Tensor::<B, 2>::zeros([heads, latent], &device)
                        }
                    }
                    None => Tensor::<B, 2>::zeros([heads, latent], &device),
                };

                layer_state.viz = Some(LayerVizState {
                    x_neuron_last,
                    y_gate_last,
                    y_neuron_last,
                    rho_last,
                });
            }

            let branch_out = Tensor::cat(next_tokens, 2).reshape([
                branch_batch,
                branch_views,
                branch_time,
                branch_dim,
            ]);
            let next = self.merge_language_residuals_for_layer(
                branch_out,
                merge_bindings,
                &connector,
                mhc_coefficients,
            );
            current = if self.residual_connector_needs_post_merge_norm(&connector) {
                self.norm.forward(next)
            } else {
                next
            };
            self.update_language_residual_history(&mut residual_history, current_before, &current);
        }

        let hidden = self.collapse_language_streams(current);
        let [_batch, time, _dim] = hidden.shape().dims::<3>();
        if advance_position {
            state.position = state.position.saturating_add(time);
        }

        hidden
    }

    pub fn latent_reasoning_config(&self) -> &LatentReasoningConfig {
        &self.latent_reasoning
    }

    pub fn with_fixed_latent_reasoning_steps(mut self, steps: usize) -> Self {
        assert!(steps > 0, "fixed latent reasoning steps must be > 0");
        self.latent_reasoning.enabled = true;
        self.latent_reasoning.max_steps = steps;
        self.latent_reasoning.min_steps = steps;
        self.latent_reasoning.adaptive_halting = false;
        self
    }

    pub fn latent_reasoning_enabled(&self) -> bool {
        self.latent_reasoning.enabled
            && self.latent_reasoning.max_steps > 0
            && self.latent_refiner_in.is_some()
            && self.latent_refiner_out.is_some()
            && (!self.latent_reasoning.adaptive_halting || self.latent_stop_head.is_some())
    }

    fn latent_refine_step(
        &self,
        current: Tensor<B, 3>,
        refiner_in: &Linear<B>,
        refiner_out: &Linear<B>,
    ) -> Tensor<B, 3> {
        let update = refiner_out.forward(activation::gelu(refiner_in.forward(current.clone())));
        let update = if let Some(gate) = self.latent_refiner_gate.as_ref() {
            let [batch, time, dim] = update.shape().dims();
            let [gate_dim] = gate.val().shape().dims();
            if gate_dim == dim {
                let gate = activation::sigmoid(gate.val())
                    .reshape([1, 1, dim])
                    .repeat_dim(0, batch)
                    .repeat_dim(1, time);
                update * gate
            } else {
                update
            }
        } else {
            update
        };
        if self.latent_reasoning.normalize_steps {
            self.norm.forward(current + update)
        } else {
            current + update
        }
    }

    pub fn reason_hidden_final(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        if !self.latent_reasoning_enabled() {
            return hidden;
        }

        let refiner_in = self
            .latent_refiner_in
            .as_ref()
            .expect("latent refiner input missing");
        let refiner_out = self
            .latent_refiner_out
            .as_ref()
            .expect("latent refiner output missing");
        if !self.latent_reasoning.adaptive_halting {
            let mut current = hidden;
            for _ in 0..self.latent_reasoning.max_steps {
                current = self.latent_refine_step(current, refiner_in, refiner_out);
            }
            return current;
        }
        let energy_head = self.latent_energy_head.as_ref();
        let stop_head = self
            .latent_stop_head
            .as_ref()
            .expect("latent stop head missing");

        let [batch, time, dim] = hidden.shape().dims();
        let device = hidden.device();
        let mut current = hidden.clone();
        let mut final_hidden = Tensor::<B, 3>::zeros([batch, time, dim], &device);
        let mut remaining = Tensor::<B, 3>::ones([batch, time, 1], &device);
        for step in 0..self.latent_reasoning.max_steps {
            current = self.latent_refine_step(current, refiner_in, refiner_out);
            let _energy = energy_head.map(|head| head.forward(current.clone()));
            let logits = stop_head.forward(current.clone());
            let probs = if step + 1 < self.latent_reasoning.min_steps {
                Tensor::<B, 3>::zeros(logits.shape().dims::<3>(), &logits.device())
            } else {
                activation::sigmoid(logits.clone())
            };
            let halt = probs
                .clone()
                .greater_equal_elem(self.latent_reasoning.halt_threshold)
                .float()
                * remaining.clone();
            final_hidden = final_hidden + current.clone() * halt.clone().repeat_dim(2, dim);
            remaining = remaining * halt.mul_scalar(-1.0).add_scalar(1.0);
        }
        final_hidden + current * remaining.repeat_dim(2, dim)
    }

    pub fn reason_hidden(&self, hidden: Tensor<B, 3>) -> LatentReasoningOutput<B> {
        if !self.latent_reasoning_enabled() {
            return LatentReasoningOutput {
                raw_hidden: hidden.clone(),
                final_hidden: hidden,
                step_hiddens: Vec::new(),
                energies: Vec::new(),
                stop_logits: Vec::new(),
                stop_probs: Vec::new(),
                steps_used: 0,
            };
        }

        let refiner_in = self
            .latent_refiner_in
            .as_ref()
            .expect("latent refiner input missing");
        let refiner_out = self
            .latent_refiner_out
            .as_ref()
            .expect("latent refiner output missing");
        let energy_head = self.latent_energy_head.as_ref();
        let stop_head = self.latent_stop_head.as_ref();

        let [batch, time, dim] = hidden.shape().dims();
        let device = hidden.device();
        let mut current = hidden.clone();
        let mut final_hidden = Tensor::<B, 3>::zeros([batch, time, dim], &device);
        let mut remaining = Tensor::<B, 3>::ones([batch, time, 1], &device);
        let mut step_hiddens = Vec::with_capacity(self.latent_reasoning.max_steps);
        let mut energies = Vec::with_capacity(self.latent_reasoning.max_steps);
        let mut stop_logits = Vec::with_capacity(self.latent_reasoning.max_steps);
        let mut stop_probs = Vec::with_capacity(self.latent_reasoning.max_steps);
        for step in 0..self.latent_reasoning.max_steps {
            current = self.latent_refine_step(current, refiner_in, refiner_out);
            if let Some(head) = energy_head {
                energies.push(head.forward(current.clone()));
            }
            if let Some(head) = stop_head {
                let logits = head.forward(current.clone());
                let probs = if step + 1 < self.latent_reasoning.min_steps {
                    Tensor::<B, 3>::zeros(logits.shape().dims::<3>(), &logits.device())
                } else {
                    activation::sigmoid(logits.clone())
                };
                let halt = probs
                    .clone()
                    .greater_equal_elem(self.latent_reasoning.halt_threshold)
                    .float()
                    * remaining.clone();
                final_hidden = final_hidden + current.clone() * halt.clone().repeat_dim(2, dim);
                remaining = remaining * halt.mul_scalar(-1.0).add_scalar(1.0);
                stop_logits.push(logits);
                stop_probs.push(probs);
            }
            step_hiddens.push(current.clone());
        }
        if self.latent_reasoning.adaptive_halting {
            final_hidden = final_hidden + current.clone() * remaining.repeat_dim(2, dim);
        } else {
            final_hidden = current.clone();
        }

        LatentReasoningOutput {
            raw_hidden: hidden,
            final_hidden,
            step_hiddens,
            energies,
            stop_logits,
            stop_probs,
            steps_used: self.latent_reasoning.max_steps,
        }
    }

    pub fn latent_jepa_prediction_from_hidden(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        self.latent_jepa_predictor
            .as_ref()
            .map(|predictor| predictor.forward(hidden.clone()))
            .unwrap_or(hidden)
    }

    pub fn next_latent_transition_enabled(&self) -> bool {
        self.next_latent_transition.enabled
            && self.next_latent_transition_in.is_some()
            && self.next_latent_transition_mid.is_some()
            && self.next_latent_transition_out.is_some()
    }

    fn normalize_next_latent_transition_input(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        if !self.next_latent_transition.normalize_input {
            return input;
        }
        let [_batch, _time, dim] = input.shape().dims();
        if dim == 0 {
            return input;
        }
        let mean = input.clone().mean_dim(2);
        let centered = input - mean.repeat_dim(2, dim);
        let variance = centered.clone().powf_scalar(2.0).mean_dim(2);
        centered / variance.add_scalar(1.0e-5).sqrt().repeat_dim(2, dim)
    }

    pub fn next_latent_prediction_from_hidden_action(
        &self,
        hidden: Tensor<B, 3>,
        action_embedding: Tensor<B, 3>,
    ) -> Option<Tensor<B, 3>> {
        if !self.next_latent_transition_enabled() {
            return None;
        }
        let transition_in = self.next_latent_transition_in.as_ref()?;
        let transition_mid = self.next_latent_transition_mid.as_ref()?;
        let transition_out = self.next_latent_transition_out.as_ref()?;
        let [batch, time, dim] = hidden.shape().dims();
        if action_embedding.shape().dims() != [batch, time, dim] {
            return None;
        }
        let input = Tensor::cat(vec![hidden.clone(), action_embedding], 2);
        let input = self.normalize_next_latent_transition_input(input);
        let update = transition_out.forward(activation::gelu(
            transition_mid.forward(activation::gelu(transition_in.forward(input))),
        ));
        Some(hidden + update)
    }

    pub fn latent_energy_from_hidden(&self, hidden: Tensor<B, 3>) -> Option<Tensor<B, 3>> {
        self.latent_energy_head
            .as_ref()
            .map(|head| head.forward(hidden))
    }

    fn project_hidden_to_logits(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        assert!(
            self.language_head.uses_flat_token_logits(),
            "flat token logits are not available for the configured NCA factorized language head; use hidden-state loss helpers instead"
        );
        let prof_enabled = logits_projection_profile_enabled();
        let start = prof_enabled.then(Instant::now);
        let [batch, time, dim] = hidden.shape().dims();
        let head = if self.tie_input_output_embeddings {
            self.embed.weight.val().transpose()
        } else {
            self.lm_head
                .as_ref()
                .expect("flat language-model head weights missing")
                .val()
        };
        let logits = hidden.reshape([batch * time, dim]).matmul(head).reshape([
            batch,
            time,
            self.vocab_size,
        ]);
        if let Some(start) = start {
            logits_projection_profile_record(start.elapsed().as_nanos());
        }
        logits
    }

    fn apply_latent_decoder_step_conditioning(
        &self,
        hidden: Tensor<B, 3>,
        step: usize,
    ) -> Tensor<B, 3> {
        if !self.latent_reasoning.step_conditioned_decoder
            || self.latent_reasoning.step_conditioned_decoder_scale <= f32::EPSILON
        {
            return hidden;
        }
        let Some(embedding) = self.latent_step_decoder_embedding.as_ref() else {
            return hidden;
        };
        let [batch, time, dim] = hidden.shape().dims();
        let [steps, width] = embedding.val().shape().dims();
        if steps == 0 || width != dim {
            return hidden;
        }
        let step = step.min(steps - 1);
        let bias = embedding
            .val()
            .slice([step..step + 1, 0..dim])
            .reshape([1, 1, dim])
            .repeat_dim(0, batch)
            .repeat_dim(1, time)
            .mul_scalar(self.latent_reasoning.step_conditioned_decoder_scale);
        hidden + bias
    }

    fn project_hidden_to_logits_for_latent_step(
        &self,
        hidden: Tensor<B, 3>,
        step: usize,
    ) -> Tensor<B, 3> {
        let hidden = self.apply_latent_decoder_step_conditioning(hidden, step);
        self.project_hidden_to_logits(hidden)
    }

    pub fn logits_from_hidden(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        self.project_hidden_to_logits_for_latent_step(hidden, self.latent_decoder_step())
    }

    pub fn logits_from_hidden_for_latent_step(
        &self,
        hidden: Tensor<B, 3>,
        step: usize,
    ) -> Tensor<B, 3> {
        self.project_hidden_to_logits_for_latent_step(hidden, step)
    }

    pub fn uses_factorized_language_head(&self) -> bool {
        !self.language_head.uses_flat_token_logits()
    }

    pub fn forward_with_state(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
    ) -> Tensor<B, 3> {
        let (_hidden, logits) = self.forward_with_state_impl(tokens, state, None);
        logits
    }

    pub fn forward_hidden(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut state = ModelState::new(self.n_layer);
        self.forward_hidden_with_state(tokens, &mut state)
    }

    pub fn forward_hidden_raw(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut state = ModelState::new(self.n_layer);
        self.forward_hidden_raw_with_state(tokens, &mut state)
    }

    pub fn forward_with_state_and_summary_event_mask(
        &self,
        tokens: Tensor<B, 2, Int>,
        summary_event_mask: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
    ) -> Tensor<B, 3> {
        let (_hidden, logits) =
            self.forward_with_state_impl(tokens, state, Some(summary_event_mask));
        logits
    }

    pub fn forward_hidden_with_state(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
    ) -> Tensor<B, 3> {
        self.forward_hidden_with_state_impl(tokens, state, None)
    }

    pub fn forward_hidden_raw_with_state(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
    ) -> Tensor<B, 3> {
        let embedded = self.embed.forward(tokens);
        self.forward_hidden_raw_with_state_from_embedded(embedded, state, None)
    }

    pub fn forward_hidden_with_state_and_summary_event_mask(
        &self,
        tokens: Tensor<B, 2, Int>,
        summary_event_mask: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
    ) -> Tensor<B, 3> {
        self.forward_hidden_with_state_impl(tokens, state, Some(summary_event_mask))
    }

    pub fn forward_with_hidden_and_state(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        self.forward_with_state_impl(tokens, state, None)
    }

    pub fn forward_with_hidden_and_state_and_summary_event_mask(
        &self,
        tokens: Tensor<B, 2, Int>,
        summary_event_mask: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        self.forward_with_state_impl(tokens, state, Some(summary_event_mask))
    }

    pub fn forward_with_state_embedded(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
    ) -> Tensor<B, 3> {
        let (_hidden, logits) = self.forward_with_state_from_embedded(embedded, state, None);
        logits
    }

    pub fn forward_hidden_with_state_embedded(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
    ) -> Tensor<B, 3> {
        self.forward_hidden_with_state_from_embedded(embedded, state, None)
    }

    pub fn forward_with_hidden_and_state_embedded(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        self.forward_with_state_from_embedded(embedded, state, None)
    }

    #[doc(hidden)]
    pub fn forward_hidden_prefix_layers_from_embedded_for_profile(
        &self,
        embedded: Tensor<B, 3>,
        layer_limit: usize,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        let mut state = ModelState::new(self.n_layer);
        self.forward_hidden_with_state_from_embedded_single_pass_layer_limit(
            embedded,
            &mut state,
            0,
            true,
            RecurrentPositionMode::Sequential,
            summary_event_mask,
            layer_limit.min(self.n_layer),
        )
    }

    pub fn summary_memory_write_trigger_token_ids(&self) -> Option<&[u32]> {
        self.summary_memory.write_trigger_token_ids.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::init::{
        DragonInitializationConfig, DragonInitializationKind, DragonReservoirInitializationConfig,
    };
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn tensor_values<const D: usize>(tensor: Tensor<TestBackend, D>) -> Vec<f32> {
        tensor
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("tensor values")
    }

    fn tiny_scaling_source_config(sequence_kernel: SequenceKernelConfig) -> DragonConfig {
        DragonConfig {
            n_layer: 1,
            n_embd: 16,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 32,
            dropout: 0.0,
            sequence_kernel,
            ..Default::default()
        }
    }

    fn assert_widened_forward_is_finite(model: &DragonModel<TestBackend>) {
        let device = burn::tensor::Device::<TestBackend>::default();
        let tokens = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3], [1, 3]),
            &device,
        );
        let logits = model.forward(tokens);
        assert_eq!(logits.shape().dims(), [1, 3, 32]);
        assert!(tensor_values(logits).iter().all(|value| value.is_finite()));
    }

    fn max_abs_diff(lhs: Vec<f32>, rhs: Vec<f32>) -> f32 {
        assert_eq!(lhs.len(), rhs.len(), "tensor length mismatch");
        lhs.into_iter()
            .zip(rhs)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn tied_language_head_projects_with_input_embeddings() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let mut config = tiny_scaling_source_config(SequenceKernelConfig::default());
        config.tie_input_output_embeddings = true;
        let model = DragonModel::<TestBackend>::new(config, &device);
        let hidden = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                (0..16).map(|value| value as f32 / 16.0).collect(),
                [1, 1, 16],
            ),
            &device,
        );
        let logits = model.logits_from_hidden(hidden.clone());
        let expected = hidden
            .reshape([1, 16])
            .matmul(model.embed.weight.val().transpose())
            .reshape([1, 1, 32]);
        let diff = max_abs_diff(tensor_values(logits), tensor_values(expected));
        assert!(diff <= 1e-6, "tied logits drifted by {diff}");
    }

    #[test]
    fn latent_reasoning_forward_returns_finite_reasoned_hidden() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let mut config = tiny_scaling_source_config(SequenceKernelConfig::default());
        config.latent_reasoning.enabled = true;
        config.latent_reasoning.max_steps = 2;
        config.latent_reasoning.min_steps = 1;
        config.latent_reasoning.adaptive_halting = true;
        config.latent_reasoning.energy_head = true;
        let model = DragonModel::<TestBackend>::new(config, &device);
        let tokens = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]),
            &device,
        );

        let raw = model.forward_hidden_raw(tokens.clone());
        let reasoned = model.forward_hidden(tokens.clone());
        let logits = model.forward(tokens.clone());
        let output = model.reason_hidden(raw.clone());

        assert_eq!(raw.shape().dims(), [1, 4, 16]);
        assert_eq!(reasoned.shape().dims(), [1, 4, 16]);
        assert_eq!(logits.shape().dims(), [1, 4, 32]);
        assert_eq!(output.steps_used, 2);
        assert_eq!(output.energies.len(), 2);
        assert_eq!(output.stop_probs.len(), 2);
        assert!(tensor_values(logits).iter().all(|value| value.is_finite()));
        assert!(
            tensor_values(reasoned)
                .iter()
                .all(|value| value.is_finite()),
            "reasoned hidden contains non-finite values"
        );
    }

    #[test]
    fn latent_reasoning_default_refiner_starts_as_identity_residual() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let mut config = tiny_scaling_source_config(SequenceKernelConfig::default());
        config.latent_reasoning.enabled = true;
        let model = DragonModel::<TestBackend>::new(config, &device);
        let tokens = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]),
            &device,
        );

        let raw = model.forward_hidden_raw(tokens);
        let output = model.reason_hidden(raw.clone());
        let diff = max_abs_diff(tensor_values(raw), tensor_values(output.final_hidden));

        assert!(
            diff <= 1e-6,
            "zero-initialized latent residual should preserve hidden at init; diff={diff}"
        );
        assert_eq!(output.steps_used, 1);
        assert_eq!(output.energies.len(), 0);
        assert_eq!(output.stop_probs.len(), 0);
    }

    #[test]
    fn latent_residual_refinement_gate_scales_learned_updates() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let hidden = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                (0..32).map(|value| value as f32 / 16.0 - 1.0).collect(),
                [1, 2, 16],
            ),
            &device,
        );

        TestBackend::seed(&device, 13);
        let mut gated_config = tiny_scaling_source_config(SequenceKernelConfig::default());
        gated_config.latent_reasoning.enabled = true;
        gated_config.latent_reasoning.max_steps = 1;
        gated_config.latent_reasoning.min_steps = 1;
        gated_config.latent_reasoning.residual_refinement_gate = true;
        gated_config.latent_reasoning.residual_refinement_gate_init = 0.25;
        let mut gated_model = DragonModel::<TestBackend>::new(gated_config, &device);

        let out = gated_model
            .latent_refiner_out
            .as_mut()
            .expect("latent refiner output");
        let [rows, cols] = out.weight.val().shape().dims();
        out.weight = Param::from_tensor(
            Tensor::<TestBackend, 2>::ones([rows, cols], &device).mul_scalar(0.01),
        );
        if let Some(bias) = out.bias.as_mut() {
            let [dim] = bias.val().shape().dims();
            *bias = Param::from_tensor(Tensor::<TestBackend, 1>::zeros([dim], &device));
        }

        let gate = gated_model.latent_refiner_gate.take();
        let open_delta = gated_model.reason_hidden(hidden.clone()).final_hidden - hidden.clone();
        gated_model.latent_refiner_gate = gate;
        let gated_delta = gated_model.reason_hidden(hidden.clone()).final_hidden - hidden;
        let expected = open_delta.mul_scalar(0.25);
        let diff = max_abs_diff(tensor_values(gated_delta), tensor_values(expected));

        assert!(
            diff <= 1.0e-5,
            "residual refinement gate should scale update by init multiplier; diff={diff}"
        );
    }

    #[test]
    fn latent_step_conditioned_decoder_starts_neutral_and_can_shift_step_logits() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let mut config = tiny_scaling_source_config(SequenceKernelConfig::default());
        config.latent_reasoning.enabled = true;
        config.latent_reasoning.max_steps = 2;
        config.latent_reasoning.min_steps = 2;
        config.latent_reasoning.step_conditioned_decoder = true;
        let mut model = DragonModel::<TestBackend>::new(config, &device);
        let hidden = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                (0..32).map(|value| value as f32 / 32.0).collect(),
                [1, 2, 16],
            ),
            &device,
        );

        let neutral_step0 = model.logits_from_hidden_for_latent_step(hidden.clone(), 0);
        let neutral_step2 = model.logits_from_hidden_for_latent_step(hidden.clone(), 2);
        let neutral_diff = max_abs_diff(tensor_values(neutral_step0), tensor_values(neutral_step2));
        assert!(
            neutral_diff <= 1.0e-6,
            "zero-initialized step decoder should preserve logits; diff={neutral_diff}"
        );

        let mut values = vec![0.0f32; 3 * 16];
        for index in 0..16 {
            values[2 * 16 + index] = 0.05;
        }
        model.latent_step_decoder_embedding = Some(Param::from_tensor(
            Tensor::<TestBackend, 2>::from_data(TensorData::new(values, [3, 16]), &device),
        ));

        let shifted_step0 = model.logits_from_hidden_for_latent_step(hidden.clone(), 0);
        let shifted_step2 = model.logits_from_hidden_for_latent_step(hidden, 2);
        let shifted_diff = max_abs_diff(tensor_values(shifted_step0), tensor_values(shifted_step2));
        assert!(
            shifted_diff > 1.0e-5,
            "nonzero step decoder embedding should change step-specific logits"
        );
    }

    #[test]
    fn next_latent_transition_starts_as_identity_delta() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let mut config = tiny_scaling_source_config(SequenceKernelConfig::default());
        config.next_latent_transition.enabled = true;
        let model = DragonModel::<TestBackend>::new(config, &device);
        let tokens = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]),
            &device,
        );

        let raw = model.forward_hidden_raw(tokens.clone());
        let hidden = model.forward_hidden(tokens.clone());
        let context = hidden.clone().slice([0..1, 0..3, 0..16]);
        let action_tokens = tokens.slice([0..1, 1..4]);
        let action_embedding = model.embed_tokens(action_tokens);
        let prediction = model
            .next_latent_prediction_from_hidden_action(context.clone(), action_embedding)
            .expect("next latent transition enabled");
        let diff = max_abs_diff(tensor_values(context), tensor_values(prediction));
        let forward_diff = max_abs_diff(tensor_values(raw), tensor_values(hidden));

        assert!(
            diff <= 1.0e-6,
            "zero-initialized transition drifted by {diff}"
        );
        assert!(
            forward_diff <= 1.0e-6,
            "NextLat transition should not alter forward_hidden; diff={forward_diff}"
        );
    }

    #[test]
    fn hierarchical_dragon_split_rho_shared_weights_forward_persists_slow_state() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let mut config = tiny_scaling_source_config(SequenceKernelConfig::default());
        config.hierarchical_dragon.enabled = true;
        config.hierarchical_dragon.last_layers = Some(1);
        config.hierarchical_dragon.fast_cycles = 1;
        config.hierarchical_dragon.slow_cycles = 1;
        config.hierarchical_dragon.rho_sharing = HierarchicalDragonSharing::Split;
        config.hierarchical_dragon.weight_sharing = HierarchicalDragonSharing::Shared;
        config.hierarchical_dragon.slow_to_fast_scale = 0.1;
        config.hierarchical_dragon.fast_to_slow_scale = 0.1;
        let model = DragonModel::<TestBackend>::new(config, &device);
        let mut state = model.init_state();
        let tokens = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]),
            &device,
        );

        let logits = model.forward_with_state(tokens, &mut state);

        assert_eq!(logits.shape().dims(), [1, 4, 32]);
        assert!(tensor_values(logits).iter().all(|value| value.is_finite()));
        assert!(state.layers[0].rho.is_some(), "fast rho should be written");
        assert!(
            state.layers[0].slow_rho.is_some(),
            "split slow rho should be written"
        );
        assert!(
            state.layers[0].hierarchical_slow_hidden.is_some(),
            "slow hidden summary should be retained"
        );
        assert!(model.slow_encoder.is_none());
        assert!(model.slow_encoder_v.is_none());
        assert!(model.slow_decoder.is_none());
        assert!(!model.supports_shared_lowrank_population_forward());
        assert!(!model.supports_shared_lowrank_continual_backprop());
    }

    #[test]
    fn hierarchical_dragon_split_weights_forward_and_record_round_trip() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let mut config = tiny_scaling_source_config(SequenceKernelConfig::default());
        config.hierarchical_dragon.enabled = true;
        config.hierarchical_dragon.last_layers = Some(1);
        config.hierarchical_dragon.fast_cycles = 1;
        config.hierarchical_dragon.slow_cycles = 1;
        config.hierarchical_dragon.rho_sharing = HierarchicalDragonSharing::Split;
        config.hierarchical_dragon.weight_sharing = HierarchicalDragonSharing::Split;
        let model = DragonModel::<TestBackend>::new(config.clone(), &device);
        let tokens = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]),
            &device,
        );

        let logits = model.forward(tokens.clone());
        let record = model.clone().into_record();
        let reloaded = DragonModel::<TestBackend>::new(config, &device).load_record(record);
        let reloaded_logits = reloaded.forward(tokens);
        let diff = max_abs_diff(
            tensor_values(logits.clone()),
            tensor_values(reloaded_logits),
        );

        assert_eq!(logits.shape().dims(), [1, 4, 32]);
        assert!(tensor_values(logits).iter().all(|value| value.is_finite()));
        assert!(model.slow_encoder.is_some());
        assert!(model.slow_encoder_v.is_some());
        assert!(model.slow_decoder.is_some());
        assert!(
            diff <= 1.0e-6,
            "split hierarchical record round-trip drifted by {diff}"
        );
    }

    #[test]
    fn shared_lowrank_population_forward_matches_base_for_single_member() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let config = tiny_scaling_source_config(SequenceKernelConfig::default());
        let model = DragonModel::<TestBackend>::new(config, &device);
        let tokens = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4, 5, 6], [2, 3]),
            &device,
        );
        let base = model.shared_lowrank_weights();
        let population = SharedLowrankPopulationWeights {
            encoder: base.encoder.reshape([
                1,
                model.n_head,
                model.n_embd,
                model.latent_per_head_capacity(),
            ]),
            encoder_v: base.encoder_v.reshape([
                1,
                model.n_head,
                model.n_embd,
                model.latent_per_head_capacity(),
            ]),
            decoder: base
                .decoder
                .reshape([1, model.latent_total_capacity(), model.n_embd]),
        };

        let expected = model.forward(tokens.clone());
        let actual = model.forward_with_shared_lowrank_population(tokens, population);
        let diff = max_abs_diff(tensor_values(expected), tensor_values(actual));
        assert!(diff <= 1e-5, "population forward drifted by {diff}");
    }

    #[test]
    fn shared_lowrank_population_forward_keeps_members_independent() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let config = tiny_scaling_source_config(SequenceKernelConfig::default());
        let model = DragonModel::<TestBackend>::new(config, &device);
        let tokens = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4, 5, 6], [2, 3]),
            &device,
        );
        let base = model.shared_lowrank_weights();
        let population = SharedLowrankPopulationWeights {
            encoder: Tensor::cat(
                vec![
                    base.encoder.clone().reshape([
                        1,
                        model.n_head,
                        model.n_embd,
                        model.latent_per_head_capacity(),
                    ]),
                    base.encoder.reshape([
                        1,
                        model.n_head,
                        model.n_embd,
                        model.latent_per_head_capacity(),
                    ]),
                ],
                0,
            ),
            encoder_v: Tensor::cat(
                vec![
                    base.encoder_v.clone().reshape([
                        1,
                        model.n_head,
                        model.n_embd,
                        model.latent_per_head_capacity(),
                    ]),
                    base.encoder_v.reshape([
                        1,
                        model.n_head,
                        model.n_embd,
                        model.latent_per_head_capacity(),
                    ]),
                ],
                0,
            ),
            decoder: Tensor::cat(
                vec![
                    base.decoder
                        .clone()
                        .reshape([1, model.latent_total_capacity(), model.n_embd]),
                    base.decoder
                        .reshape([1, model.latent_total_capacity(), model.n_embd]),
                ],
                0,
            ),
        };

        let expected = model.forward(tokens.clone());
        let stacked = model.forward_with_shared_lowrank_population(tokens, population);
        let first = stacked.clone().slice_dim(0, 0..2);
        let second = stacked.slice_dim(0, 2..4);
        let first_diff = max_abs_diff(tensor_values(expected.clone()), tensor_values(first));
        let second_diff = max_abs_diff(tensor_values(expected), tensor_values(second));
        assert!(
            first_diff <= 1e-5,
            "first population drifted by {first_diff}"
        );
        assert!(
            second_diff <= 1e-5,
            "second population drifted by {second_diff}"
        );
    }

    #[test]
    fn shared_lowrank_population_forward_does_not_couple_different_members() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let config = tiny_scaling_source_config(SequenceKernelConfig::default());
        let model = DragonModel::<TestBackend>::new(config, &device);
        let tokens = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4, 5, 6], [2, 3]),
            &device,
        );
        let base = model.shared_lowrank_weights();
        let shifted_encoder = base.encoder.clone().add_scalar(1.0e-3);
        let shifted_encoder_v = base.encoder_v.clone().sub_scalar(1.0e-3);
        let shifted_decoder = base.decoder.clone().add_scalar(1.0e-3);
        let population = SharedLowrankPopulationWeights {
            encoder: Tensor::cat(
                vec![
                    base.encoder.clone().reshape([
                        1,
                        model.n_head,
                        model.n_embd,
                        model.latent_per_head_capacity(),
                    ]),
                    shifted_encoder.reshape([
                        1,
                        model.n_head,
                        model.n_embd,
                        model.latent_per_head_capacity(),
                    ]),
                ],
                0,
            ),
            encoder_v: Tensor::cat(
                vec![
                    base.encoder_v.clone().reshape([
                        1,
                        model.n_head,
                        model.n_embd,
                        model.latent_per_head_capacity(),
                    ]),
                    shifted_encoder_v.reshape([
                        1,
                        model.n_head,
                        model.n_embd,
                        model.latent_per_head_capacity(),
                    ]),
                ],
                0,
            ),
            decoder: Tensor::cat(
                vec![
                    base.decoder
                        .clone()
                        .reshape([1, model.latent_total_capacity(), model.n_embd]),
                    shifted_decoder.reshape([1, model.latent_total_capacity(), model.n_embd]),
                ],
                0,
            ),
        };

        let expected = model.forward(tokens.clone());
        let stacked = model.forward_with_shared_lowrank_population(tokens, population);
        let first = stacked.slice_dim(0, 0..2);
        let diff = max_abs_diff(tensor_values(expected), tensor_values(first));
        assert!(diff <= 1e-5, "base population was coupled by {diff}");
    }

    #[test]
    fn shared_lowrank_population_forward_single_head_keeps_members_independent() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let mut config = tiny_scaling_source_config(SequenceKernelConfig::default());
        config.n_embd = 8;
        config.n_head = 1;
        config.mlp_internal_dim_multiplier = 1;
        config.vocab_size = 16;
        let model = DragonModel::<TestBackend>::new(config, &device);
        let tokens = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4, 5, 6, 7, 8], [2, 4]),
            &device,
        );
        let base = model.shared_lowrank_weights();
        let shifted_encoder = base.encoder.clone().add_scalar(1.0e-3);
        let shifted_encoder_v = base.encoder_v.clone().sub_scalar(1.0e-3);
        let shifted_decoder = base.decoder.clone().add_scalar(1.0e-3);
        let population = SharedLowrankPopulationWeights {
            encoder: Tensor::cat(
                vec![
                    base.encoder.clone().reshape([
                        1,
                        model.n_head,
                        model.n_embd,
                        model.latent_per_head_capacity(),
                    ]),
                    shifted_encoder.reshape([
                        1,
                        model.n_head,
                        model.n_embd,
                        model.latent_per_head_capacity(),
                    ]),
                ],
                0,
            ),
            encoder_v: Tensor::cat(
                vec![
                    base.encoder_v.clone().reshape([
                        1,
                        model.n_head,
                        model.n_embd,
                        model.latent_per_head_capacity(),
                    ]),
                    shifted_encoder_v.reshape([
                        1,
                        model.n_head,
                        model.n_embd,
                        model.latent_per_head_capacity(),
                    ]),
                ],
                0,
            ),
            decoder: Tensor::cat(
                vec![
                    base.decoder
                        .clone()
                        .reshape([1, model.latent_total_capacity(), model.n_embd]),
                    shifted_decoder.reshape([1, model.latent_total_capacity(), model.n_embd]),
                ],
                0,
            ),
        };

        let expected = model.forward(tokens.clone());
        let stacked = model.forward_with_shared_lowrank_population(tokens, population);
        let first = stacked.slice_dim(0, 0..2);
        let diff = max_abs_diff(tensor_values(expected), tensor_values(first));
        assert!(
            diff <= 1e-5,
            "single-head base population was coupled by {diff}"
        );
    }

    #[test]
    fn linear_attention_incremental_forward_matches_full_sequence() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let config = tiny_scaling_source_config(SequenceKernelConfig::reference(
            SequenceMemorySystem::LinearAttention,
        ));
        let model = DragonModel::<TestBackend>::new(config, &device);
        let tokens = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4, 5, 6], [1, 6]),
            &device,
        );

        let full_logits = model.forward(tokens.clone());
        let mut state = model.init_state();
        let mut pieces = Vec::new();
        for index in 0..6 {
            let token = tokens.clone().slice([0..1, index..index + 1]);
            pieces.push(model.forward_with_state(token, &mut state));
        }
        let incremental_logits = Tensor::cat(pieces, 1);
        let diff = max_abs_diff(
            tensor_values(full_logits),
            tensor_values(incremental_logits),
        );
        assert!(
            diff <= 1.0e-4,
            "linear-attention incremental logits drifted from full sequence by {diff}"
        );
        assert_eq!(state.position, 6);
    }

    fn assert_widened_forward_matches_source(
        source: &DragonModel<TestBackend>,
        widened: &DragonModel<TestBackend>,
        tolerance: f32,
    ) {
        let device = burn::tensor::Device::<TestBackend>::default();
        let tokens = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]),
            &device,
        );
        let embedding_weight_diff = max_abs_diff(
            tensor_values(source.embed.weight.val()),
            tensor_values(widened.embed.weight.val()),
        );
        assert!(
            embedding_weight_diff <= tolerance,
            "widened model changed embedding weights before training: max_abs_diff={embedding_weight_diff} tolerance={tolerance}"
        );
        let source_embedded = tensor_values(source.embed_tokens(tokens.clone()));
        let widened_embedded = tensor_values(widened.embed_tokens(tokens.clone()));
        let embedded_diff = max_abs_diff(source_embedded, widened_embedded);
        assert!(
            embedded_diff <= tolerance,
            "widened model changed embeddings before training: max_abs_diff={embedded_diff} tolerance={tolerance}"
        );
        let source_hidden = tensor_values(source.forward_hidden(tokens.clone()));
        let widened_hidden = tensor_values(widened.forward_hidden(tokens.clone()));
        let hidden_diff = max_abs_diff(source_hidden, widened_hidden);
        assert!(
            hidden_diff <= tolerance,
            "widened model changed hidden states before training: max_abs_diff={hidden_diff} tolerance={tolerance}"
        );
        let source_logits = tensor_values(source.forward(tokens.clone()));
        let widened_logits = tensor_values(widened.forward(tokens));
        let diff = max_abs_diff(source_logits, widened_logits);
        assert!(
            diff <= tolerance,
            "widened model changed logits before training: max_abs_diff={diff} tolerance={tolerance}"
        );
    }

    fn assert_widened_record_round_trip_matches_source(
        source: &DragonModel<TestBackend>,
        widened: &DragonModel<TestBackend>,
        target_config: DragonConfig,
        tolerance: f32,
    ) {
        let device = burn::tensor::Device::<TestBackend>::default();
        let record = widened.clone().into_record();
        let reloaded = DragonModel::<TestBackend>::new(target_config, &device).load_record(record);
        assert_widened_forward_matches_source(source, &reloaded, tolerance);
    }

    fn assert_shared_lowrank_prefix_preserved(
        source: &DragonModel<TestBackend>,
        widened: &DragonModel<TestBackend>,
    ) {
        let old_latent_per_head = source.latent_per_head_capacity();
        assert_eq!(
            tensor_values(source.encoder.val()),
            tensor_values(widened.encoder.val().slice([
                0..source.n_head,
                0..source.n_embd,
                0..old_latent_per_head
            ]))
        );
        assert_eq!(
            tensor_values(source.encoder_v.val()),
            tensor_values(widened.encoder_v.val().slice([
                0..source.n_head,
                0..source.n_embd,
                0..old_latent_per_head
            ]))
        );
        for head in 0..source.n_head {
            let source_start = head * old_latent_per_head;
            let widened_start = head * widened.latent_per_head_capacity();
            assert_eq!(
                tensor_values(source.decoder.val().slice([
                    source_start..source_start + old_latent_per_head,
                    0..source.n_embd
                ])),
                tensor_values(widened.decoder.val().slice([
                    widened_start..widened_start + old_latent_per_head,
                    0..source.n_embd
                ]))
            );
        }
        assert!(
            tensor_values(widened.encoder.val().slice([
                0..source.n_head,
                0..source.n_embd,
                old_latent_per_head..widened.latent_per_head_capacity()
            ]))
            .iter()
            .all(|value| *value == 0.0),
            "widened query encoder tail should start as a no-op"
        );
    }

    fn assert_slow_lowrank_prefix_preserved(
        source: &DragonModel<TestBackend>,
        widened: &DragonModel<TestBackend>,
    ) {
        let old_latent_per_head = source.latent_per_head_capacity();
        let source_encoder = source.slow_encoder.as_ref().expect("source slow encoder");
        let source_encoder_v = source
            .slow_encoder_v
            .as_ref()
            .expect("source slow encoder_v");
        let source_decoder = source.slow_decoder.as_ref().expect("source slow decoder");
        let widened_encoder = widened.slow_encoder.as_ref().expect("widened slow encoder");
        let widened_encoder_v = widened
            .slow_encoder_v
            .as_ref()
            .expect("widened slow encoder_v");
        let widened_decoder = widened.slow_decoder.as_ref().expect("widened slow decoder");

        assert_eq!(
            tensor_values(source_encoder.val()),
            tensor_values(widened_encoder.val().slice([
                0..source.n_head,
                0..source.n_embd,
                0..old_latent_per_head
            ]))
        );
        assert_eq!(
            tensor_values(source_encoder_v.val()),
            tensor_values(widened_encoder_v.val().slice([
                0..source.n_head,
                0..source.n_embd,
                0..old_latent_per_head
            ]))
        );
        for head in 0..source.n_head {
            let source_start = head * old_latent_per_head;
            let widened_start = head * widened.latent_per_head_capacity();
            assert_eq!(
                tensor_values(source_decoder.val().slice([
                    source_start..source_start + old_latent_per_head,
                    0..source.n_embd
                ])),
                tensor_values(widened_decoder.val().slice([
                    widened_start..widened_start + old_latent_per_head,
                    0..source.n_embd
                ]))
            );
        }
        assert!(
            tensor_values(widened_encoder.val().slice([
                0..source.n_head,
                0..source.n_embd,
                old_latent_per_head..widened.latent_per_head_capacity()
            ]))
            .iter()
            .all(|value| *value == 0.0),
            "widened slow query encoder tail should start as a no-op"
        );
    }

    #[test]
    fn tiny_reservoir_model_constructs_and_runs_forward() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let config = DragonConfig {
            n_layer: 1,
            n_embd: 16,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 32,
            dropout: 0.0,
            initialization: DragonInitializationConfig {
                kind: DragonInitializationKind::Reservoir,
                reservoir: DragonReservoirInitializationConfig {
                    seed: 7,
                    density: 0.2,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let model = DragonModel::<TestBackend>::new(config, &device);
        let tokens = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3], [1, 3]),
            &device,
        );
        let logits = model.forward(tokens);
        assert_eq!(logits.shape().dims(), [1, 3, 32]);
        let values = logits
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("logits");
        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn tiny_gated_deltanet2_model_constructs_and_runs_forward() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let config = DragonConfig {
            n_layer: 1,
            n_embd: 16,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 32,
            dropout: 0.0,
            sequence_kernel: SequenceKernelConfig::reference(SequenceMemorySystem::GatedDeltaNet2),
            ..Default::default()
        };
        let model = DragonModel::<TestBackend>::new(config, &device);
        let tokens = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3], [1, 3]),
            &device,
        );
        let logits = model.forward(tokens);
        assert_eq!(logits.shape().dims(), [1, 3, 32]);
        let values = logits
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("logits");
        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn hierarchical_dragon_split_rho_mamba3_forward_persists_slow_state() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let mut config = tiny_scaling_source_config(SequenceKernelConfig::reference(
            SequenceMemorySystem::Mamba3StateSpaceDuality,
        ));
        config.mamba = super::super::sequence::mamba::MambaSequenceConfig {
            headdim: 8,
            chunk_size: 4,
            ..Default::default()
        };
        config.hierarchical_dragon.enabled = true;
        config.hierarchical_dragon.last_layers = Some(1);
        config.hierarchical_dragon.fast_cycles = 1;
        config.hierarchical_dragon.slow_cycles = 1;
        config.hierarchical_dragon.rho_sharing = HierarchicalDragonSharing::Split;
        let model = DragonModel::<TestBackend>::new(config, &device);
        let mut state = model.init_state();
        let tokens = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3], [1, 3]),
            &device,
        );

        let logits = model.forward_with_state(tokens, &mut state);

        assert_eq!(logits.shape().dims(), [1, 3, 32]);
        assert!(tensor_values(logits).iter().all(|value| value.is_finite()));
        assert!(state.layers[0].rho.is_some(), "fast Mamba3 state");
        assert!(state.layers[0].slow_rho.is_some(), "slow Mamba3 state");
        assert!(
            state.layers[0].slow_mamba_angle_state.is_some(),
            "slow Mamba3 angle state"
        );
    }

    #[test]
    fn hierarchical_dragon_split_rho_gdn2_forward_persists_slow_state() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let mut config = tiny_scaling_source_config(SequenceKernelConfig::reference(
            SequenceMemorySystem::GatedDeltaNet2,
        ));
        config.hierarchical_dragon.enabled = true;
        config.hierarchical_dragon.last_layers = Some(1);
        config.hierarchical_dragon.fast_cycles = 1;
        config.hierarchical_dragon.slow_cycles = 1;
        config.hierarchical_dragon.rho_sharing = HierarchicalDragonSharing::Split;
        let model = DragonModel::<TestBackend>::new(config, &device);
        let mut state = model.init_state();
        let tokens = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3], [1, 3]),
            &device,
        );

        let logits = model.forward_with_state(tokens, &mut state);

        assert_eq!(logits.shape().dims(), [1, 3, 32]);
        assert!(tensor_values(logits).iter().all(|value| value.is_finite()));
        assert!(state.layers[0].rho.is_some(), "fast GDN2 state");
        assert!(state.layers[0].slow_rho.is_some(), "slow GDN2 state");
    }

    #[test]
    fn widen_latent_total_supports_linear_attention() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let source_config = tiny_scaling_source_config(SequenceKernelConfig::reference(
            SequenceMemorySystem::LinearAttention,
        ));
        let target_config = DragonConfig {
            mlp_internal_dim_multiplier: 4,
            ..source_config.clone()
        };
        let source = DragonModel::<TestBackend>::new(source_config, &device);
        let (widened, report) = source
            .widen_latent_total(target_config.clone(), &device)
            .expect("widen");
        assert_eq!(report.old_latent_total, 32);
        assert_eq!(report.new_latent_total, 64);
        assert_eq!(widened.latent_total_capacity(), 64);
        assert_shared_lowrank_prefix_preserved(&source, &widened);
        assert_widened_forward_matches_source(&source, &widened, 1.0e-5);
        assert_widened_record_round_trip_matches_source(&source, &widened, target_config, 1.0e-5);
        assert_widened_forward_is_finite(&widened);
    }

    #[test]
    fn widen_latent_total_supports_split_hierarchical_dragon_weights() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let mut source_config = tiny_scaling_source_config(SequenceKernelConfig::reference(
            SequenceMemorySystem::LinearAttention,
        ));
        source_config.hierarchical_dragon.enabled = true;
        source_config.hierarchical_dragon.last_layers = Some(1);
        source_config.hierarchical_dragon.fast_cycles = 1;
        source_config.hierarchical_dragon.slow_cycles = 1;
        source_config.hierarchical_dragon.rho_sharing = HierarchicalDragonSharing::Split;
        source_config.hierarchical_dragon.weight_sharing = HierarchicalDragonSharing::Split;
        let target_config = DragonConfig {
            mlp_internal_dim_multiplier: 4,
            ..source_config.clone()
        };
        let source = DragonModel::<TestBackend>::new(source_config, &device);
        let (widened, report) = source
            .widen_latent_total(target_config.clone(), &device)
            .expect("widen split hierarchy");

        assert_eq!(report.old_latent_total, 32);
        assert_eq!(report.new_latent_total, 64);
        assert_shared_lowrank_prefix_preserved(&source, &widened);
        assert_slow_lowrank_prefix_preserved(&source, &widened);
        assert_widened_forward_matches_source(&source, &widened, 1.0e-5);
        assert_widened_record_round_trip_matches_source(&source, &widened, target_config, 1.0e-5);
        assert_widened_forward_is_finite(&widened);
    }

    #[test]
    fn widen_latent_total_supports_dense_score_short_context() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let source_config =
            tiny_scaling_source_config(SequenceKernelConfig::dense_score_short_context());
        let target_config = DragonConfig {
            mlp_internal_dim_multiplier: 4,
            ..source_config.clone()
        };
        let source = DragonModel::<TestBackend>::new(source_config, &device);
        let (widened, report) = source
            .widen_latent_total(target_config.clone(), &device)
            .expect("widen");
        assert_eq!(report.new_latent_total, 64);
        assert_shared_lowrank_prefix_preserved(&source, &widened);
        assert_widened_forward_matches_source(&source, &widened, 1.0e-5);
        assert_widened_record_round_trip_matches_source(&source, &widened, target_config, 1.0e-5);
        assert_widened_forward_is_finite(&widened);
    }

    #[test]
    fn widen_latent_total_supports_mamba3_and_preserves_mamba_params() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let source_config = DragonConfig {
            sequence_kernel: SequenceKernelConfig::reference(
                SequenceMemorySystem::Mamba3StateSpaceDuality,
            ),
            mamba: super::super::sequence::mamba::MambaSequenceConfig {
                headdim: 8,
                chunk_size: 4,
                ..Default::default()
            },
            ..tiny_scaling_source_config(SequenceKernelConfig::reference(
                SequenceMemorySystem::Mamba3StateSpaceDuality,
            ))
        };
        let target_config = DragonConfig {
            mlp_internal_dim_multiplier: 4,
            ..source_config.clone()
        };
        let source = DragonModel::<TestBackend>::new(source_config, &device);
        let source_mamba = source.mamba.as_ref().expect("source mamba").mamba3();
        let source_in_proj = tensor_values(source_mamba.in_proj_tensor());
        let source_dt_bias = tensor_values(source_mamba.dt_bias_tensor());
        let source_out_proj = tensor_values(source_mamba.out_proj_tensor());

        let (widened, report) = source
            .widen_latent_total(target_config.clone(), &device)
            .expect("widen");
        assert_eq!(report.new_latent_total, 64);
        assert_shared_lowrank_prefix_preserved(&source, &widened);
        let widened_mamba = widened.mamba.as_ref().expect("widened mamba").mamba3();
        assert_eq!(
            source_in_proj,
            tensor_values(widened_mamba.in_proj_tensor())
        );
        assert_eq!(
            source_dt_bias,
            tensor_values(widened_mamba.dt_bias_tensor())
        );
        assert_eq!(
            source_out_proj,
            tensor_values(widened_mamba.out_proj_tensor())
        );
        assert_widened_forward_matches_source(&source, &widened, 1.0e-5);
        assert_widened_record_round_trip_matches_source(&source, &widened, target_config, 1.0e-5);
        assert_widened_forward_is_finite(&widened);
    }

    #[test]
    fn widen_latent_total_supports_gdn2_adapter_and_preserves_latent_prefix() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let source_config = tiny_scaling_source_config(SequenceKernelConfig::reference(
            SequenceMemorySystem::GatedDeltaNet2,
        ));
        let target_config = DragonConfig {
            mlp_internal_dim_multiplier: 4,
            ..source_config.clone()
        };
        let source = DragonModel::<TestBackend>::new(source_config, &device);
        let source_gdn2 = source.gated_deltanet2.as_ref().expect("source gdn2");
        let source_key = tensor_values(source_gdn2.key_proj_tensor());

        let (widened, report) = source
            .widen_latent_total(target_config.clone(), &device)
            .expect("widen");
        assert_eq!(report.new_latent_total, 64);
        assert_shared_lowrank_prefix_preserved(&source, &widened);
        let widened_key_prefix = widened
            .gated_deltanet2
            .as_ref()
            .expect("widened gdn2")
            .key_proj_tensor()
            .slice([0..source.n_head, 0..source.n_embd, 0..16]);
        assert_eq!(source_key, tensor_values(widened_key_prefix));
        assert_widened_forward_matches_source(&source, &widened, 5.0e-4);
        assert_widened_record_round_trip_matches_source(&source, &widened, target_config, 5.0e-4);
        assert_widened_forward_is_finite(&widened);
    }

    #[test]
    fn widen_latent_total_supports_upstream_gdn2_and_preserves_headed_prefix() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let source_config = DragonConfig {
            sequence_kernel: SequenceKernelConfig::gated_delta_chunk_wy(),
            gated_deltanet2: super::super::sequence::gdn2::GatedDeltaNet2Config {
                implementation: GatedDeltaNet2Implementation::UpstreamFull,
                chunk_size: 4,
                ..Default::default()
            },
            ..tiny_scaling_source_config(SequenceKernelConfig::gated_delta_chunk_wy())
        };
        let target_config = DragonConfig {
            mlp_internal_dim_multiplier: 4,
            ..source_config.clone()
        };
        let source = DragonModel::<TestBackend>::new(source_config, &device);
        let source_upstream = source
            .gated_deltanet2_upstream
            .as_ref()
            .expect("source upstream gdn2");

        let (widened, report) = source
            .widen_latent_total(target_config.clone(), &device)
            .expect("widen");
        assert_eq!(report.new_latent_total, 64);
        assert_shared_lowrank_prefix_preserved(&source, &widened);
        let widened_upstream = widened
            .gated_deltanet2_upstream
            .as_ref()
            .expect("widened upstream gdn2");
        for head in 0..source.n_head {
            let source_start = head * 16;
            let widened_start = head * 32;
            assert_eq!(
                tensor_values(
                    source_upstream
                        .query
                        .weight
                        .val()
                        .slice([0..source.n_embd, source_start..source_start + 16])
                ),
                tensor_values(
                    widened_upstream
                        .query
                        .weight
                        .val()
                        .slice([0..source.n_embd, widened_start..widened_start + 16])
                )
            );
        }
        assert_widened_forward_matches_source(&source, &widened, 1.0e-4);
        assert_widened_record_round_trip_matches_source(&source, &widened, target_config, 1.0e-4);
        assert_widened_forward_is_finite(&widened);
    }

    #[test]
    fn tiny_upstream_gated_deltanet2_model_constructs_and_runs_forward() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let config = DragonConfig {
            n_layer: 1,
            n_embd: 16,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 32,
            dropout: 0.0,
            sequence_kernel: SequenceKernelConfig::gated_delta_chunk_wy(),
            gated_deltanet2: super::super::sequence::gdn2::GatedDeltaNet2Config {
                implementation: GatedDeltaNet2Implementation::UpstreamFull,
                chunk_size: 4,
                ..Default::default()
            },
            ..Default::default()
        };
        let model = DragonModel::<TestBackend>::new(config, &device);
        let tokens = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3], [1, 3]),
            &device,
        );
        let logits = model.forward(tokens);
        assert_eq!(logits.shape().dims(), [1, 3, 32]);
        let values = logits
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("logits");
        assert!(values.iter().all(|value| value.is_finite()));
    }
}
