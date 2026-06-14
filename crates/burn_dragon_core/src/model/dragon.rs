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
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear};
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
    ClockedSlowMemoryConfig, DragonConfig, FusedKernelConfig, LanguageHeadConfig,
    SummaryMemoryConfig, YNeuronRecurrenceConfig,
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
#[cfg(not(any(feature = "viz", feature = "probe")))]
use super::residual_stream::lowrank_residual_step_next_branch_thresholds_relu_native;
#[cfg(any(feature = "probe", test))]
use super::residual_stream::lowrank_residual_step_with_metrics_branch_thresholds;
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
    sequence_kernel: SequenceKernelConfig,
    rollout_fast_steps_per_slow_step: usize,
    kernel: FusedKernelConfig,
    x_relu_threshold: f32,
    y_relu_threshold: f32,
    y_neuron_recurrence: YNeuronRecurrenceConfig,
    clocked_slow_memory: ClockedSlowMemoryConfig,
    summary_memory: SummaryMemoryConfig,
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
    #[module(skip)]
    nca_factorized_head_tables: Option<NcaFactorizedHeadTables>,
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
            mamba_config,
            mamba,
            gated_deltanet2_config,
            gated_deltanet2,
            gated_deltanet2_upstream,
            lm_head,
            nca_factorized_lm_head,
            nca_special_lm_head,
            nca_factorized_head_tables,
        }
    }

    pub fn latent_total_capacity(&self) -> usize {
        self.decoder.val().shape().dims::<2>()[0]
    }

    pub fn latent_per_head_capacity(&self) -> usize {
        self.encoder.val().shape().dims::<3>()[2]
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
        let logits = self.project_hidden_to_logits(hidden.clone());
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

    fn layer_lowrank_weights(
        &self,
        layer_idx: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 2>, usize) {
        let latent_per_head = self.layer_latent_per_head(layer_idx);
        let capacity_per_head = self.latent_per_head_capacity();
        let encoder = self
            .encoder
            .val()
            .slice([0..self.n_head, 0..self.n_embd, 0..latent_per_head])
            .reshape([1, self.n_head, self.n_embd, latent_per_head]);
        let encoder_v = self
            .encoder_v
            .val()
            .slice([0..self.n_head, 0..self.n_embd, 0..latent_per_head])
            .reshape([1, self.n_head, self.n_embd, latent_per_head]);
        let decoder_capacity = self.decoder.val();
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

    fn forward_hidden_with_state_from_embedded(
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
            let hidden_last = hidden_rollout.slice_dim(1, last..fast_steps);
            let logits_last = self.project_hidden_to_logits(hidden_last.clone());
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

    fn project_hidden_to_logits(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        assert!(
            self.language_head.uses_flat_token_logits(),
            "flat token logits are not available for the configured NCA factorized language head; use hidden-state loss helpers instead"
        );
        let prof_enabled = logits_projection_profile_enabled();
        let start = prof_enabled.then(Instant::now);
        let [batch, time, dim] = hidden.shape().dims();
        let logits = hidden
            .reshape([batch * time, dim])
            .matmul(
                self.lm_head
                    .as_ref()
                    .expect("flat language-model head weights missing")
                    .val(),
            )
            .reshape([batch, time, self.vocab_size]);
        if let Some(start) = start {
            logits_projection_profile_record(start.elapsed().as_nanos());
        }
        logits
    }

    pub fn logits_from_hidden(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        self.project_hidden_to_logits(hidden)
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
