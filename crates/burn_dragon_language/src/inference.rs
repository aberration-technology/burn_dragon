use burn_dragon_core::DragonConfig;
#[cfg(feature = "train")]
use burn_dragon_train::wgpu as shared_wgpu;

use crate::ModelOverrides;
use crate::summary_events::resolve_summary_memory_write_triggers;
use crate::tokenizer::Tokenizer;

/// Optional WGPU fused-core overrides applied during model-config construction.
///
/// `rollout` falls back to `recurrent` when omitted so callers can override both execution
/// surfaces with one field.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WgpuFusedCoreOverride {
    pub recurrent: Option<bool>,
    pub rollout: Option<bool>,
}

/// Build a model configuration by applying training overrides.
pub fn build_model_config(overrides: &ModelOverrides, training_block_size: usize) -> DragonConfig {
    let mut model_config = DragonConfig::default();

    if let Some(n_layer) = overrides.n_layer {
        model_config.n_layer = n_layer;
    }
    if let Some(n_embd) = overrides.n_embd {
        model_config.n_embd = n_embd;
    }
    if let Some(n_head) = overrides.n_head {
        model_config.n_head = n_head;
    }
    if let Some(multiplier) = overrides.mlp_internal_dim_multiplier {
        model_config.mlp_internal_dim_multiplier = multiplier;
    }
    if let Some(language_head) = &overrides.language_head {
        model_config.language_head = language_head.clone();
    }
    if let Some(sequence_score_head) = overrides.sequence_score_head {
        model_config.sequence_score_head = sequence_score_head;
    }
    if let Some(tie_input_output_embeddings) = overrides.tie_input_output_embeddings {
        model_config.tie_input_output_embeddings = tie_input_output_embeddings;
    }
    if let Some(latent_total) = overrides.latent_total {
        assert!(
            latent_total % model_config.n_embd == 0,
            "model.latent_total must be divisible by n_embd (got latent_total={} n_embd={})",
            latent_total,
            model_config.n_embd
        );
        model_config.mlp_internal_dim_multiplier = latent_total / model_config.n_embd;
    }
    if let Some(initialization) = &overrides.initialization {
        model_config.initialization = initialization.clone();
    }
    if let Some(random_scaffold) = &overrides.random_scaffold {
        random_scaffold
            .validate_for_model(
                model_config.n_embd,
                model_config.n_head,
                model_config.latent_total(),
            )
            .unwrap_or_else(|message| panic!("invalid model.random_scaffold override: {message}"));
        model_config.random_scaffold = random_scaffold.clone();
    }
    if let Some(sequence_kernel) = overrides.sequence_kernel {
        sequence_kernel
            .validate()
            .unwrap_or_else(|message| panic!("invalid model.sequence_kernel override: {message}"));
        model_config.sequence_kernel = sequence_kernel;
    }
    if let Some(mamba) = &overrides.mamba {
        let memory_system = overrides
            .sequence_kernel
            .unwrap_or(model_config.sequence_kernel)
            .memory_system;
        mamba
            .validate(memory_system, model_config.n_embd)
            .unwrap_or_else(|message| panic!("invalid model.mamba override: {message}"));
        model_config.mamba = mamba.clone();
    }
    if let Some(gated_deltanet2) = &overrides.gated_deltanet2 {
        gated_deltanet2
            .validate(
                model_config.n_head,
                model_config.n_embd,
                model_config.latent_per_head(),
            )
            .unwrap_or_else(|message| panic!("invalid model.gated_deltanet2 override: {message}"));
        model_config.gated_deltanet2 = gated_deltanet2.clone();
    }
    if let Some(residual_connector) = overrides.residual_connector {
        model_config.residual_connector = residual_connector;
    }
    if let Some(attention_residual) = &overrides.attention_residual {
        model_config.attention_residual = attention_residual.clone();
    }
    if let Some(block_attention_residual) = &overrides.block_attention_residual {
        model_config.block_attention_residual = block_attention_residual.clone();
    }
    if let Some(schedule) = &overrides.latent_fanout_schedule {
        if let Err(message) = model_config.validate_latent_fanout_schedule(schedule) {
            panic!("{message}");
        }
        model_config.latent_fanout_schedule = Some(schedule.clone());
    }
    if let Some(relu_threshold) = overrides.relu_threshold {
        model_config.fused_kernels.relu_threshold = relu_threshold;
    }
    if let Some(dropout) = overrides.dropout {
        model_config.dropout = dropout;
    }
    if let Some(normalization) = &overrides.normalization {
        model_config.normalization = normalization.clone();
    }
    if let Some(enabled) = overrides.fused_kernels {
        model_config.fused_kernels.enabled = enabled;
    }
    if let Some(enabled) = overrides.fused_recurrent_kernel {
        model_config
            .fused_kernels
            .set_wgpu_recurrent_kernel(enabled);
    }
    if let Some(enabled) = overrides.fused_rollout_kernel {
        model_config.fused_kernels.set_wgpu_rollout_fused(enabled);
    }
    if let Some(executor) = overrides.fused_projection_executor {
        model_config.fused_kernels.set_projection_executor(executor);
    }
    if let Some(executor) = overrides.fused_attention_executor {
        model_config.fused_kernels.set_attention_executor(executor);
    }
    if let Some(executor) = overrides.fused_lowrank_grad_input_executor {
        model_config
            .fused_kernels
            .set_lowrank_grad_input_executor(executor);
    }
    let block = overrides.block_size.unwrap_or(training_block_size).max(1);
    model_config.fused_kernels.set_block_sizes(block, block);
    if let Some(rollout_fast_steps) = overrides.rollout_fast_steps_per_slow_step {
        model_config.set_rollout_fast_steps_per_slow_step(rollout_fast_steps);
    }
    if let Some(rotary_embedding) = overrides.rotary_embedding {
        model_config
            .fused_kernels
            .set_rotary_embedding(rotary_embedding);
    }
    if let Some(y_neuron_recurrence) = &overrides.y_neuron_recurrence {
        model_config.y_neuron_recurrence = y_neuron_recurrence.clone();
    }
    if let Some(clocked_slow_memory) = &overrides.clocked_slow_memory {
        model_config.clocked_slow_memory = clocked_slow_memory.clone();
    }
    if let Some(summary_memory) = &overrides.summary_memory {
        model_config.summary_memory = summary_memory.clone();
    }
    if let Some(hierarchical_dragon) = &overrides.hierarchical_dragon {
        model_config.hierarchical_dragon = hierarchical_dragon.clone();
    }
    if let Some(latent_reasoning) = &overrides.latent_reasoning {
        model_config.latent_reasoning = latent_reasoning.clone();
    }
    if let Some(next_latent_transition) = &overrides.next_latent_transition {
        model_config.next_latent_transition = next_latent_transition.clone();
    }
    if let Some(mhc) = &overrides.mhc {
        model_config.mhc = mhc.clone();
    }
    if matches!(
        model_config.sequence_kernel.memory_system,
        burn_dragon_core::SequenceMemorySystem::Mamba3StateSpaceDuality
    ) {
        model_config
            .mamba
            .validate(
                model_config.sequence_kernel.memory_system,
                model_config.n_embd,
            )
            .unwrap_or_else(|message| panic!("invalid Mamba sequence kernel config: {message}"));
    }
    if matches!(
        model_config.sequence_kernel.memory_system,
        burn_dragon_core::SequenceMemorySystem::GatedDeltaNet2
    ) {
        model_config
            .gated_deltanet2
            .validate(
                model_config.n_head,
                model_config.n_embd,
                model_config.latent_per_head(),
            )
            .unwrap_or_else(|message| {
                panic!("invalid GatedDeltaNet2 sequence kernel config: {message}")
            });
    }

    match overrides.residual_connector {
        Some(burn_dragon_core::ResidualConnectorKind::Vanilla) => {
            model_config.mhc.enabled = false;
            model_config.attention_residual.enabled = false;
            model_config.block_attention_residual.enabled = false;
        }
        Some(burn_dragon_core::ResidualConnectorKind::Mhc) => {
            model_config.mhc.enabled = true;
            model_config.attention_residual.enabled = false;
            model_config.block_attention_residual.enabled = false;
        }
        Some(burn_dragon_core::ResidualConnectorKind::AttentionResidual) => {
            model_config.mhc.enabled = false;
            model_config.attention_residual.enabled = true;
            model_config.block_attention_residual.enabled = false;
        }
        Some(burn_dragon_core::ResidualConnectorKind::BlockAttentionResidual) => {
            model_config.mhc.enabled = false;
            model_config.attention_residual.enabled = false;
            model_config.block_attention_residual.enabled = true;
        }
        None => {
            model_config.residual_connector = burn_dragon_core::ResidualConnectorKind::Vanilla;
            model_config.mhc.enabled = false;
            model_config.attention_residual.enabled = false;
            model_config.block_attention_residual.enabled = false;
        }
    }

    model_config
}

pub fn build_model_config_with_tokenizer(
    overrides: &ModelOverrides,
    training_block_size: usize,
    tokenizer: &dyn Tokenizer,
) -> anyhow::Result<DragonConfig> {
    let mut model_config = build_model_config(overrides, training_block_size);
    resolve_summary_memory_write_triggers(&mut model_config, tokenizer)?;
    model_config.vocab_size = tokenizer.len();
    model_config
        .language_head
        .validate_for_vocab_size(model_config.vocab_size)
        .unwrap_or_else(|message| panic!("invalid language_head config: {message}"));
    if model_config.tie_input_output_embeddings
        && !model_config.language_head.uses_flat_token_logits()
    {
        anyhow::bail!(
            "model.tie_input_output_embeddings requires language_head.type=\"standard_token_classification\""
        );
    }
    Ok(model_config)
}

pub fn is_wgpu_backend_name(backend_name: &str) -> bool {
    #[cfg(feature = "train")]
    {
        shared_wgpu::is_wgpu_backend_name(backend_name)
    }
    #[cfg(not(feature = "train"))]
    {
        backend_name.eq_ignore_ascii_case("wgpu")
            || backend_name
                .get(..5)
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case("wgpu-"))
    }
}

pub fn apply_wgpu_fused_core_override(
    model_config: &mut DragonConfig,
    backend_name: &str,
    override_config: WgpuFusedCoreOverride,
) {
    #[cfg(feature = "train")]
    {
        shared_wgpu::apply_wgpu_fused_core_override(
            model_config,
            backend_name,
            shared_wgpu::WgpuFusedCoreOverride {
                recurrent: override_config.recurrent,
                rollout: override_config.rollout,
            },
        );
    }

    #[cfg(not(feature = "train"))]
    {
        if !is_wgpu_backend_name(backend_name) {
            return;
        }

        if let Some(enabled) = override_config.recurrent {
            model_config
                .fused_kernels
                .set_wgpu_recurrent_kernel(enabled);
            if enabled {
                model_config.fused_kernels.enabled = true;
            }
        }

        let rollout_override = override_config.rollout.or(override_config.recurrent);
        if let Some(enabled) = rollout_override {
            model_config.fused_kernels.set_wgpu_rollout_fused(enabled);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        WgpuFusedCoreOverride, apply_wgpu_fused_core_override, build_model_config,
        is_wgpu_backend_name,
    };
    use crate::ModelOverrides;
    use burn_dragon_core::{
        DragonConfig, DragonInitializationConfig, DragonInitializationKind, ResidualConnectorKind,
    };

    #[test]
    fn backend_name_detection_accepts_wgpu_variants() {
        assert!(is_wgpu_backend_name("wgpu"));
        assert!(is_wgpu_backend_name("WGPU"));
        assert!(is_wgpu_backend_name("wgpu-fused-core"));
        assert!(is_wgpu_backend_name("wgpu-nofusion"));
        assert!(!is_wgpu_backend_name("cuda"));
    }

    #[test]
    fn wgpu_override_wrapper_delegates_to_shared_behavior() {
        let mut model_config = DragonConfig::default();
        model_config.fused_kernels.enabled = false;
        model_config.fused_kernels.set_wgpu_recurrent_kernel(false);
        model_config.fused_kernels.set_wgpu_rollout_fused(false);

        apply_wgpu_fused_core_override(
            &mut model_config,
            "wgpu",
            WgpuFusedCoreOverride {
                recurrent: Some(true),
                rollout: None,
            },
        );

        assert!(
            model_config.fused_kernels.enabled,
            "wgpu backend override should enable fused kernels for recurrent path selection"
        );
        assert!(
            model_config.fused_kernels.wgpu_recurrent_kernel,
            "wgpu recurrent kernel should be enabled by override"
        );
        assert!(
            model_config.fused_kernels.wgpu_rollout_fused,
            "wgpu rollout fused path should default to recurrent override when unspecified"
        );
    }

    #[test]
    fn model_override_applies_rollout_fast_steps() {
        let overrides = ModelOverrides {
            rollout_fast_steps_per_slow_step: Some(8),
            ..ModelOverrides::default()
        };

        let config = build_model_config(&overrides, 32);
        assert_eq!(config.rollout_fast_steps_per_slow_step, 8);
    }

    #[test]
    fn model_override_applies_backend_neutral_fused_kernel_controls() {
        let overrides = ModelOverrides {
            fused_kernels: Some(true),
            fused_recurrent_kernel: Some(false),
            fused_rollout_kernel: Some(true),
            fused_projection_executor: Some(burn_dragon_core::FusedProjectionExecutor::None),
            fused_attention_executor: Some(
                burn_dragon_core::FusedAttentionExecutor::AttentionContext,
            ),
            fused_lowrank_grad_input_executor: Some(
                burn_dragon_core::LowrankGradInputExecutor::Direct,
            ),
            ..ModelOverrides::default()
        };

        let config = build_model_config(&overrides, 512);
        assert!(config.fused_kernels.enabled);
        assert!(!config.fused_kernels.wgpu_recurrent_kernel);
        assert!(config.fused_kernels.wgpu_rollout_fused);
        assert_eq!(
            config.fused_kernels.projection_executor,
            burn_dragon_core::FusedProjectionExecutor::None
        );
        assert_eq!(
            config.fused_kernels.attention_executor,
            burn_dragon_core::FusedAttentionExecutor::AttentionContext
        );
        assert_eq!(
            config.fused_kernels.lowrank_grad_input_executor,
            burn_dragon_core::LowrankGradInputExecutor::Direct
        );
    }

    #[test]
    fn model_override_applies_explicit_latent_total() {
        let overrides = ModelOverrides {
            n_embd: Some(256),
            latent_total: Some(32768),
            ..ModelOverrides::default()
        };

        let config = build_model_config(&overrides, 32);
        assert_eq!(config.latent_total(), 32768);
        assert_eq!(config.mlp_internal_dim_multiplier, 128);
    }

    #[test]
    fn model_override_applies_initialization_family() {
        let overrides = ModelOverrides {
            initialization: Some(DragonInitializationConfig {
                kind: DragonInitializationKind::HeadwiseSemiOrthogonal,
                ..Default::default()
            }),
            ..ModelOverrides::default()
        };

        let config = build_model_config(&overrides, 32);
        assert_eq!(
            config.initialization.kind,
            DragonInitializationKind::HeadwiseSemiOrthogonal
        );
    }

    #[test]
    fn model_override_applies_latent_fanout_schedule() {
        let overrides = ModelOverrides {
            n_layer: Some(8),
            n_embd: Some(256),
            n_head: Some(4),
            latent_total: Some(32768),
            latent_fanout_schedule: Some(burn_dragon_core::LatentFanoutScheduleConfig::LateLayer {
                base_latent_total: 8192,
                last_layers: 4,
            }),
            ..ModelOverrides::default()
        };

        let config = build_model_config(&overrides, 32);
        assert_eq!(config.latent_total_for_layer(0), 8192);
        assert_eq!(config.latent_total_for_layer(7), 32768);
    }

    #[test]
    fn model_override_applies_sequence_kernel() {
        let overrides = ModelOverrides {
            sequence_kernel: Some(burn_dragon_core::SequenceKernelConfig::reference(
                burn_dragon_core::SequenceMemorySystem::LinearAttention,
            )),
            ..ModelOverrides::default()
        };

        let config = build_model_config(&overrides, 32);
        assert_eq!(
            config.sequence_kernel,
            burn_dragon_core::SequenceKernelConfig::reference(
                burn_dragon_core::SequenceMemorySystem::LinearAttention,
            )
        );
    }

    #[test]
    fn model_override_applies_gated_deltanet2_config() {
        let overrides = ModelOverrides {
            n_head: Some(8),
            n_embd: Some(512),
            latent_total: Some(1024),
            sequence_kernel: Some(burn_dragon_core::SequenceKernelConfig::reference(
                burn_dragon_core::SequenceMemorySystem::GatedDeltaNet2,
            )),
            gated_deltanet2: Some(burn_dragon_core::GatedDeltaNet2Config {
                qk_l2_norm: false,
                write_gate: burn_dragon_core::GatedDeltaNet2GateMode::Disabled,
                ..Default::default()
            }),
            ..ModelOverrides::default()
        };

        let config = build_model_config(&overrides, 512);
        assert_eq!(
            config.sequence_kernel.memory_system,
            burn_dragon_core::SequenceMemorySystem::GatedDeltaNet2
        );
        assert!(!config.gated_deltanet2.qk_l2_norm);
        assert_eq!(
            config.gated_deltanet2.write_gate,
            burn_dragon_core::GatedDeltaNet2GateMode::Disabled
        );
    }

    #[test]
    fn model_override_without_explicit_connector_defaults_to_vanilla() {
        let overrides = ModelOverrides {
            mhc: Some(burn_dragon_core::ManifoldHyperConnectionsConfig {
                enabled: true,
                num_streams: 2,
                num_views: 1,
                mhc_iters: 4,
                mhc_tau: 0.1,
                add_branch_out_to_residual: true,
                dropout: 0.0,
                ..Default::default()
            }),
            ..ModelOverrides::default()
        };

        let config = build_model_config(&overrides, 32);
        assert_eq!(config.residual_connector, ResidualConnectorKind::Vanilla);
        assert!(!config.mhc.enabled);
        assert!(!config.attention_residual.enabled);
        assert!(!config.block_attention_residual.enabled);
    }

    #[test]
    fn model_override_explicit_vanilla_disables_other_connectors() {
        let overrides = ModelOverrides {
            residual_connector: Some(ResidualConnectorKind::Vanilla),
            mhc: Some(burn_dragon_core::ManifoldHyperConnectionsConfig {
                enabled: true,
                num_streams: 2,
                num_views: 1,
                mhc_iters: 4,
                mhc_tau: 0.1,
                add_branch_out_to_residual: true,
                dropout: 0.0,
                ..Default::default()
            }),
            attention_residual: Some(burn_dragon_core::AttentionResidualConfig {
                enabled: true,
                num_heads: 4,
                ..Default::default()
            }),
            block_attention_residual: Some(burn_dragon_core::BlockAttentionResidualConfig {
                enabled: true,
                num_heads: 4,
                layers_per_block: 2,
                ..Default::default()
            }),
            ..ModelOverrides::default()
        };

        let config = build_model_config(&overrides, 32);
        assert_eq!(config.residual_connector, ResidualConnectorKind::Vanilla);
        assert!(!config.mhc.enabled);
        assert!(!config.attention_residual.enabled);
        assert!(!config.block_attention_residual.enabled);
    }

    #[test]
    fn model_override_explicit_block_attention_residual_enables_block_connector() {
        let overrides = ModelOverrides {
            residual_connector: Some(ResidualConnectorKind::BlockAttentionResidual),
            block_attention_residual: Some(burn_dragon_core::BlockAttentionResidualConfig {
                enabled: true,
                num_heads: 4,
                layers_per_block: 2,
                block_history_window: Some(3),
                intra_block_history_window: Some(1),
                ..Default::default()
            }),
            attention_residual: Some(burn_dragon_core::AttentionResidualConfig {
                enabled: true,
                num_heads: 4,
                ..Default::default()
            }),
            ..ModelOverrides::default()
        };

        let config = build_model_config(&overrides, 32);
        assert_eq!(
            config.residual_connector,
            ResidualConnectorKind::BlockAttentionResidual
        );
        assert!(!config.mhc.enabled);
        assert!(!config.attention_residual.enabled);
        assert!(config.block_attention_residual.enabled);
        assert_eq!(config.block_attention_residual.layers_per_block, 2);
    }
}
