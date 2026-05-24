use crate::train::prelude::*;
use crate::train::startup_autotune::StartupAutotuneReport;

pub(crate) const LANGUAGE_ARCH_VERSION: &str = "dragon_dragon_v1";
pub(crate) const SHARD_LAYOUT_VERSION_UNSHARDED: u32 = 1;

#[derive(Clone)]
pub struct PreparedDatasets {
    pub train: Arc<Dataset>,
    pub valid: Arc<Dataset>,
}

pub fn build_vocab_only(config: &TrainingConfig) -> Result<()> {
    let datasets = prepare_datasets(&config.dataset, &config.training)?;
    let tokenizer = datasets.train.tokenizer();
    info!(
        "Tokenizer `{}` ready with {} tokens",
        config.dataset.tokenizer.kind_name(),
        tokenizer.len()
    );
    Ok(())
}

pub fn prepare_dataset(
    dataset_cfg: &DatasetConfig,
    training: &TrainingHyperparameters,
) -> Result<Arc<Dataset>> {
    Ok(prepare_datasets(dataset_cfg, training)?.train)
}

pub fn prepare_datasets(
    dataset_cfg: &DatasetConfig,
    training: &TrainingHyperparameters,
) -> Result<PreparedDatasets> {
    let tokenizer_path = dataset_cfg.tokenizer.storage_path(&dataset_cfg.cache_dir);
    let tokenizer_preexists = tokenizer_path
        .as_ref()
        .map(|path| path.is_file())
        .unwrap_or(false);

    let (dataset_enum, dataset_summary) = build_dataset(dataset_cfg, training)?;
    let dataset = Arc::new(dataset_enum);

    let tokenizer = dataset.tokenizer();
    match tokenizer_path {
        Some(path) if tokenizer_preexists => info!(
            "Loaded {} tokenizer with {} tokens from {}",
            dataset_cfg.tokenizer.kind_name(),
            tokenizer.len(),
            path.display()
        ),
        Some(path) => info!(
            "Built {} tokenizer with {} tokens at {}",
            dataset_cfg.tokenizer.kind_name(),
            tokenizer.len(),
            path.display()
        ),
        None => info!(
            "Initialized {} tokenizer with {} tokens (no persistence required)",
            dataset_cfg.tokenizer.kind_name(),
            tokenizer.len()
        ),
    };

    info!("{dataset_summary}");

    let valid = if let Some(validation_cfg) = &dataset_cfg.validation {
        let effective_cfg = build_validation_dataset_config(dataset_cfg, validation_cfg);
        let (dataset_enum, dataset_summary) = build_dataset(&effective_cfg, training)?;
        let dataset = Arc::new(dataset_enum);
        ensure_validation_tokenizer_compatible(
            tokenizer.as_ref(),
            dataset.tokenizer().as_ref(),
            dataset_cfg.tokenizer.kind_name(),
        )?;
        info!("Prepared validation override dataset: {dataset_summary}");
        dataset
    } else {
        Arc::clone(&dataset)
    };

    Ok(PreparedDatasets {
        train: dataset,
        valid,
    })
}

fn build_validation_dataset_config(
    dataset_cfg: &DatasetConfig,
    validation_cfg: &ValidationDatasetConfig,
) -> DatasetConfig {
    DatasetConfig {
        cache_dir: validation_cfg
            .cache_dir
            .clone()
            .unwrap_or_else(|| dataset_cfg.cache_dir.join("validation")),
        train_split_ratio: validation_cfg
            .train_split_ratio
            .unwrap_or(dataset_cfg.train_split_ratio),
        validation: None,
        source: validation_cfg.source.clone(),
        tokenizer: dataset_cfg.tokenizer.clone(),
    }
}

fn ensure_validation_tokenizer_compatible(
    train_tokenizer: &dyn crate::tokenizer::Tokenizer,
    valid_tokenizer: &dyn crate::tokenizer::Tokenizer,
    tokenizer_label: &str,
) -> Result<()> {
    if train_tokenizer.len() != valid_tokenizer.len() {
        return Err(anyhow!(
            "validation dataset tokenizer is incompatible with the training tokenizer: vocab sizes differ (train={}, valid={}, tokenizer={tokenizer_label})",
            train_tokenizer.len(),
            valid_tokenizer.len(),
        ));
    }
    if train_tokenizer.bos_id() != valid_tokenizer.bos_id()
        || train_tokenizer.eos_id() != valid_tokenizer.eos_id()
        || train_tokenizer.pad_id() != valid_tokenizer.pad_id()
        || train_tokenizer.unk_id() != valid_tokenizer.unk_id()
    {
        return Err(anyhow!(
            "validation dataset tokenizer is incompatible with the training tokenizer: special token ids differ (tokenizer={tokenizer_label})"
        ));
    }
    Ok(())
}

pub fn log_theoretical_profile(config: &DragonConfig, batch: usize, block: usize, backend: &str) {
    let batch = batch as u64;
    let time = block as u64;
    let embed = config.n_embd as u64;
    let latent_per_head = config.latent_per_head() as u64;
    let latent_total = config.latent_total() as u64;
    let heads = config.n_head as u64;
    let bt = batch * time;

    let encoder_matmul = 2 * bt * embed * latent_total;
    let attn_scores = 2 * batch * heads * time * time * latent_per_head;
    let attn_value = 2 * batch * heads * time * time * embed;
    let decoder_matmul = 2 * bt * latent_total * embed;
    let total = encoder_matmul + attn_scores + attn_value + decoder_matmul;

    info!(
        "[train:{backend}] approx forward GFLOPs: total={total_gflops:.2}, encoder={enc:.2}, \
         attn_scores={scores:.2}, attn_value={value:.2}, decoder={dec:.2} (backward ~2x forward)",
        total_gflops = total as f64 / 1e9,
        enc = encoder_matmul as f64 / 1e9,
        scores = attn_scores as f64 / 1e9,
        value = attn_value as f64 / 1e9,
        dec = decoder_matmul as f64 / 1e9,
    );
}

#[derive(Serialize)]
pub struct RunConfigOutput {
    run_name: String,
    backend_name: String,
    arch_version: String,
    shard_layout_version: u32,
    block_size: usize,
    seed: u64,
    training_batch_size: usize,
    training_gradient_accumulation_steps: usize,
    training_effective_batch_size: usize,
    training_checkpoint_interval_iters: usize,
    training_execution_form: String,
    training_launch_mode_requested: burn_dragon_train::train::pipeline::TrainingLaunchMode,
    #[serde(skip_serializing_if = "Option::is_none")]
    training_sequence_kernel_override: Option<SequenceKernelConfig>,
    optimizer_spec: OptimizerSpec,
    overrides: ModelOverrides,
    model_spec: ModelSpec,
    parallel_spec: ParallelSpec,
    kernel_spec: KernelSpec,
    state_layout: StateLayout,
    metrics_sink: MetricsSinkSpec,
    #[serde(skip_serializing_if = "Option::is_none")]
    startup_autotune: Option<StartupAutotuneReport>,
}

pub(crate) fn build_training_execution_form(config: &TrainingConfig) -> String {
    if config.parallel.pipeline.enabled {
        "pipeline".to_string()
    } else if config.training.tbptt_chunk_size.is_some() {
        "tbptt".to_string()
    } else {
        "default_stateful".to_string()
    }
}

pub(crate) fn effective_training_kernel_block_size(training: &TrainingHyperparameters) -> usize {
    training
        .tbptt_chunk_size
        .filter(|chunk| *chunk > 0 && *chunk < training.block_size)
        .unwrap_or(training.block_size)
        .max(1)
}

pub(crate) fn build_model_spec(model_config: &DragonConfig) -> ModelSpec {
    ModelSpec {
        arch: "dragon_dragon".to_string(),
        n_embd: model_config.n_embd,
        n_head: model_config.n_head,
        n_layer: model_config.n_layer,
        latent_total: model_config.latent_total(),
        latent_per_head: model_config.latent_per_head(),
        shared_layer_weights: true,
        sequence_kernel: model_config.sequence_kernel,
        dragon_initialization_kind: model_config.initialization.kind,
        dragon_residual_scaling_kind: model_config.initialization.residual_scaling.kind,
        dragon_neuron_gain_kind: model_config.initialization.neuron_gains.kind,
        dragon_topology_prior_kind: model_config.initialization.topology_prior.kind,
        dragon_firing_target_kind: model_config.initialization.firing_targets.kind,
        dragon_reservoir_initialization: matches!(
            model_config.initialization.kind,
            DragonInitializationKind::Reservoir
        )
        .then(|| ReservoirInitializationSpec::from(&model_config.initialization.reservoir)),
        gated_deltanet2: matches!(
            model_config.sequence_kernel.memory_system,
            burn_dragon_core::SequenceMemorySystem::GatedDeltaNet2
        )
        .then(|| GatedDeltaNet2Spec {
            chunk_size: model_config.gated_deltanet2.chunk_size,
            qk_l2_norm: model_config.gated_deltanet2.qk_l2_norm,
            allow_neg_eigval: model_config.gated_deltanet2.allow_neg_eigval,
            erase_gate: model_config.gated_deltanet2.erase_gate,
            write_gate: model_config.gated_deltanet2.write_gate,
            decay_gate: model_config.gated_deltanet2.decay_gate,
            state_precision: model_config.gated_deltanet2.state_precision,
            state_epsilon: model_config.gated_deltanet2.state_epsilon,
        }),
    }
}

pub(crate) fn build_parallel_spec(config: &TrainingConfig) -> ParallelSpec {
    ParallelSpec {
        mode: config.parallel.mode,
        world_size: config.parallel.world_size,
        data_parallel_size: config.parallel.data.size,
        tensor_parallel_size: config.parallel.tensor.size,
        tensor_parallel_axis: config.parallel.tensor.axis,
        tensor_parallel_partition: config.parallel.tensor.partition,
        fsdp_enabled: config.parallel.fsdp.enabled,
        checkpoint_format: config.parallel.checkpoint.format,
        collective_num_nodes: config.parallel.data.collective_num_nodes,
        collective_global_address: config.parallel.data.collective_global_address.clone(),
        collective_node_address: config.parallel.data.collective_node_address.clone(),
        collective_data_service_port: config.parallel.data.collective_data_service_port,
        pipeline_enabled: config.parallel.pipeline.enabled,
        pipeline_stage_count: config.parallel.pipeline.stage_count,
        pipeline_virtual_stages_per_rank: config.parallel.pipeline.virtual_stages_per_rank,
        pipeline_schedule: config.parallel.pipeline.schedule,
        pipeline_microbatches: config.parallel.pipeline.microbatches,
        pipeline_partition: config.parallel.pipeline.partition,
        pipeline_activation_checkpointing: config.parallel.pipeline.activation_checkpointing,
        pipeline_shared_weight_sync: config.parallel.pipeline.shared_weight_sync,
        pipeline_communication: config.parallel.pipeline.communication,
        pipeline_cache_enabled: config.parallel.pipeline.cache.enabled,
        pipeline_cache_policy: config.parallel.pipeline.cache.policy,
        pipeline_cache_reuse_across_backward: config.parallel.pipeline.cache.reuse_across_backward,
        pipeline_cache_max_inflight_microbatches: config
            .parallel
            .pipeline
            .cache
            .max_inflight_microbatches,
        pipeline_cache_eviction: config.parallel.pipeline.cache.eviction,
        pipeline_cache_transport_dtype: config.parallel.pipeline.cache.transport_dtype,
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use burn_dragon_core::{
        DragonInitializationConfig, DragonReservoirInitializationConfig, DragonTopologyPriorConfig,
        DragonTopologyPriorKind,
    };

    #[test]
    fn model_spec_records_reservoir_initialization_details_only_for_reservoir_runs() {
        let baseline = DragonConfig::default();
        assert!(
            build_model_spec(&baseline)
                .dragon_reservoir_initialization
                .is_none()
        );

        let reservoir = DragonConfig {
            initialization: DragonInitializationConfig {
                kind: DragonInitializationKind::Reservoir,
                topology_prior: DragonTopologyPriorConfig {
                    kind: DragonTopologyPriorKind::ModularBridges,
                    ..Default::default()
                },
                reservoir: DragonReservoirInitializationConfig {
                    seed: 1337,
                    density: 0.12,
                    encoder_value_scale: 0.5,
                    decoder_scale: 1.25,
                },
                ..Default::default()
            },
            ..Default::default()
        };

        let spec = build_model_spec(&reservoir);
        let reservoir_spec = spec
            .dragon_reservoir_initialization
            .expect("reservoir metadata");
        assert_eq!(reservoir_spec.seed, 1337);
        assert_eq!(reservoir_spec.density, 0.12);
        assert_eq!(reservoir_spec.encoder_value_scale, 0.5);
        assert_eq!(reservoir_spec.decoder_scale, 1.25);
    }
}

pub(crate) fn build_optimizer_spec(config: &TrainingConfig) -> OptimizerSpec {
    OptimizerSpec {
        name: config.optimizer.name,
        learning_rate: config.optimizer.learning_rate,
        weight_decay: config.optimizer.weight_decay,
        weight_decay_final: config.optimizer.weight_decay_final,
        schedule_mode: config.optimizer.schedule_mode,
    }
}

pub(crate) fn build_kernel_spec(
    _config: &TrainingConfig,
    model_config: &DragonConfig,
    _backend_name: &str,
) -> KernelSpec {
    KernelSpec {
        sequence_kernel: model_config.sequence_kernel,
        fused_kernels_enabled: model_config.fused_kernels.enabled,
        rollout_fast_steps_per_slow_step: model_config.rollout_fast_steps_per_slow_step,
        wgpu_fused_core_recurrent: None,
        wgpu_fused_core_rollout: None,
    }
}

pub(crate) fn build_state_layout(model_config: &DragonConfig) -> StateLayout {
    let stream_count = model_config.mhc.resolved_num_streams();
    let layers = (0..model_config.n_layer)
        .map(|layer_index| {
            let latent_total = model_config.latent_total_for_layer(layer_index);
            let latent_per_head = model_config.latent_per_head_for_layer(layer_index);
            let mut tensors = match model_config.sequence_kernel.memory_system {
                burn_dragon_core::SequenceMemorySystem::Mamba3StateSpaceDuality => {
                    let mamba = model_config.mamba.resolve(
                        model_config.n_embd,
                        burn_dragon_core::SequenceMemorySystem::Mamba3StateSpaceDuality,
                    );
                    vec![
                        StateTensorSpec {
                            name: "rho".to_string(),
                            axes: vec![
                                StateAxisSpec {
                                    name: "batch_views".to_string(),
                                    size: None,
                                },
                                StateAxisSpec {
                                    name: "mamba_heads".to_string(),
                                    size: Some(mamba.nheads),
                                },
                                StateAxisSpec {
                                    name: "mamba_head_dim".to_string(),
                                    size: Some(mamba.headdim),
                                },
                                StateAxisSpec {
                                    name: "mamba_state".to_string(),
                                    size: Some(mamba.d_state),
                                },
                            ],
                        },
                        StateTensorSpec {
                            name: "mamba_angle_state".to_string(),
                            axes: vec![
                                StateAxisSpec {
                                    name: "batch_views".to_string(),
                                    size: None,
                                },
                                StateAxisSpec {
                                    name: "mamba_heads".to_string(),
                                    size: Some(mamba.nheads),
                                },
                                StateAxisSpec {
                                    name: "mamba_rope_angles".to_string(),
                                    size: Some(mamba.num_rope_angles),
                                },
                            ],
                        },
                        StateTensorSpec {
                            name: "mamba_k_state".to_string(),
                            axes: vec![
                                StateAxisSpec {
                                    name: "batch_views".to_string(),
                                    size: None,
                                },
                                StateAxisSpec {
                                    name: "mamba_heads".to_string(),
                                    size: Some(mamba.nheads),
                                },
                                StateAxisSpec {
                                    name: "mamba_state".to_string(),
                                    size: Some(mamba.d_state),
                                },
                            ],
                        },
                        StateTensorSpec {
                            name: "mamba_v_state".to_string(),
                            axes: vec![
                                StateAxisSpec {
                                    name: "batch_views".to_string(),
                                    size: None,
                                },
                                StateAxisSpec {
                                    name: "mamba_heads".to_string(),
                                    size: Some(mamba.nheads),
                                },
                                StateAxisSpec {
                                    name: "mamba_head_dim".to_string(),
                                    size: Some(mamba.headdim),
                                },
                            ],
                        },
                    ]
                }
                burn_dragon_core::SequenceMemorySystem::GatedDeltaNet2 => {
                    vec![StateTensorSpec {
                        name: "rho".to_string(),
                        axes: vec![
                            StateAxisSpec {
                                name: "batch_views".to_string(),
                                size: None,
                            },
                            StateAxisSpec {
                                name: "gdn2_heads".to_string(),
                                size: Some(model_config.n_head),
                            },
                            StateAxisSpec {
                                name: "gdn2_latent_per_head".to_string(),
                                size: Some(latent_per_head),
                            },
                            StateAxisSpec {
                                name: "dense_dim".to_string(),
                                size: Some(model_config.n_embd),
                            },
                        ],
                    }]
                }
                _ => {
                    vec![StateTensorSpec {
                        name: "rho".to_string(),
                        axes: vec![
                            StateAxisSpec {
                                name: "batch_views".to_string(),
                                size: None,
                            },
                            StateAxisSpec {
                                name: "heads".to_string(),
                                size: Some(model_config.n_head),
                            },
                            StateAxisSpec {
                                name: "latent_per_head".to_string(),
                                size: Some(latent_per_head),
                            },
                            StateAxisSpec {
                                name: "dense_dim".to_string(),
                                size: Some(model_config.n_embd),
                            },
                        ],
                    }]
                }
            };
            if model_config.y_neuron_recurrence.enabled {
                tensors.push(StateTensorSpec {
                    name: "y_neuron_state".to_string(),
                    axes: vec![
                        StateAxisSpec {
                            name: "batch_views".to_string(),
                            size: None,
                        },
                        StateAxisSpec {
                            name: "heads".to_string(),
                            size: Some(model_config.n_head),
                        },
                        StateAxisSpec {
                            name: "latent_per_head".to_string(),
                            size: Some(latent_per_head),
                        },
                    ],
                });
            }
            if model_config.clocked_slow_memory.enabled {
                tensors.push(StateTensorSpec {
                    name: "clocked_slow_hidden".to_string(),
                    axes: vec![
                        StateAxisSpec {
                            name: "batch".to_string(),
                            size: None,
                        },
                        StateAxisSpec {
                            name: "streams".to_string(),
                            size: Some(stream_count),
                        },
                        StateAxisSpec {
                            name: "time".to_string(),
                            size: Some(1),
                        },
                        StateAxisSpec {
                            name: "dense_dim".to_string(),
                            size: Some(model_config.n_embd),
                        },
                    ],
                });
            }
            if model_config.summary_memory.enabled {
                tensors.push(StateTensorSpec {
                    name: "summary_memory_hidden".to_string(),
                    axes: vec![
                        StateAxisSpec {
                            name: "batch".to_string(),
                            size: None,
                        },
                        StateAxisSpec {
                            name: "streams".to_string(),
                            size: Some(stream_count),
                        },
                        StateAxisSpec {
                            name: "time".to_string(),
                            size: Some(1),
                        },
                        StateAxisSpec {
                            name: "dense_dim".to_string(),
                            size: Some(model_config.n_embd),
                        },
                    ],
                });
            }
            LayerStateSpec {
                layer_index,
                latent_total,
                latent_per_head,
                tensors,
            }
        })
        .collect();

    StateLayout {
        state_family: "dragon_model_state".to_string(),
        position_tracked: true,
        layers,
    }
}

pub(crate) fn build_language_metrics_sink(metric_every: usize) -> MetricsSinkSpec {
    MetricsSinkSpec::new(
        "language_dragon_burn_train_v1",
        vec![
            MetricSinkEntry::new(
                "Loss",
                MetricSinkSplit::Train,
                MetricSinkValueKind::Numeric,
                metric_every,
            ),
            MetricSinkEntry::new(
                "Loss",
                MetricSinkSplit::Valid,
                MetricSinkValueKind::Numeric,
                metric_every,
            ),
            MetricSinkEntry::new(
                "Learning Rate",
                MetricSinkSplit::Train,
                MetricSinkValueKind::Numeric,
                metric_every,
            ),
            MetricSinkEntry::new(
                "device",
                MetricSinkSplit::Train,
                MetricSinkValueKind::Text,
                metric_every,
            ),
            MetricSinkEntry::new(
                "device",
                MetricSinkSplit::Valid,
                MetricSinkValueKind::Text,
                metric_every,
            ),
        ],
    )
}

pub fn write_run_config(
    config: &TrainingConfig,
    model_config: &DragonConfig,
    run_dir: &Path,
    run_name: &str,
    backend_name: &str,
    effective_training_sequence_kernel_override: Option<SequenceKernelConfig>,
    startup_autotune: Option<&StartupAutotuneReport>,
) -> Result<()> {
    fs::create_dir_all(run_dir)
        .with_context(|| format!("failed to create run directory {}", run_dir.display()))?;

    let block_size = config
        .model
        .block_size
        .unwrap_or(config.training.block_size)
        .max(1);
    let output = RunConfigOutput {
        run_name: run_name.to_string(),
        backend_name: backend_name.to_string(),
        arch_version: LANGUAGE_ARCH_VERSION.to_string(),
        shard_layout_version: SHARD_LAYOUT_VERSION_UNSHARDED,
        block_size,
        seed: config.training.seed,
        training_batch_size: config.training.batch_size,
        training_gradient_accumulation_steps: config.training.gradient_accumulation_steps,
        training_effective_batch_size: config
            .training
            .batch_size
            .saturating_mul(config.training.gradient_accumulation_steps),
        training_checkpoint_interval_iters: config.training.checkpoint_interval_iters,
        training_execution_form: build_training_execution_form(config),
        training_launch_mode_requested: config.training.launch_mode,
        training_sequence_kernel_override: effective_training_sequence_kernel_override,
        optimizer_spec: build_optimizer_spec(config),
        overrides: config.model.clone(),
        model_spec: build_model_spec(model_config),
        parallel_spec: build_parallel_spec(config),
        kernel_spec: build_kernel_spec(config, model_config, backend_name),
        state_layout: build_state_layout(model_config),
        metrics_sink: build_language_metrics_sink(config.training.log_frequency),
        startup_autotune: startup_autotune.cloned(),
    };
    let payload =
        serde_json::to_string_pretty(&output).context("failed to serialize web config")?;
    let path = run_dir.join("config.json");
    fs::write(&path, payload).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}
