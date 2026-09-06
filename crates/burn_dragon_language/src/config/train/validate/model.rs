//! Dataset, generation, Dragon architecture, and learning-rate contracts.

use super::*;

impl TrainingConfig {
    pub(super) fn validate_dataset_model_contracts(&self) -> Result<()> {
        if !(0.0 < self.dataset.train_split_ratio && self.dataset.train_split_ratio <= 1.0) {
            return Err(anyhow!(
                "dataset.train_split_ratio must be in (0, 1] (got {})",
                self.dataset.train_split_ratio
            ));
        }
        if let Some(validation) = &self.dataset.validation
            && let Some(train_split_ratio) = validation.train_split_ratio
            && !(0.0 < train_split_ratio && train_split_ratio <= 1.0)
        {
            return Err(anyhow!(
                "dataset.validation.train_split_ratio must be in (0, 1] when set (got {})",
                train_split_ratio
            ));
        }
        if let Some(max_tokens) = self.generation.max_tokens
            && max_tokens <= 0
        {
            return Err(anyhow!("generation.max_tokens must be > 0"));
        }
        if self.generation.temperature <= 0.0 {
            return Err(anyhow!("generation.temperature must be > 0"));
        }
        if let Some(top_k) = self.generation.top_k
            && top_k == 0
        {
            return Err(anyhow!("generation.top_k must be > 0"));
        }

        validate_dataset_source(
            &self.dataset.source,
            &self.dataset.tokenizer.kind,
            false,
            "dataset",
        )?;
        if let Some(validation) = &self.dataset.validation {
            validate_dataset_source(
                &validation.source,
                &self.dataset.tokenizer.kind,
                true,
                "dataset.validation",
            )?;
        }

        if let Some(gdpo) = &self.training.gdpo
            && gdpo.enabled
        {
            return Err(anyhow!(
                "training.gdpo.enabled is a legacy objective flag and cannot be combined with training.objective; use training.objective.type = \"sdpo\" for self-distilled policy optimization"
            ));
        }

        validate_training_objective_config(&self.training.objective)?;

        if let Some(n_layer) = self.model.n_layer
            && n_layer == 0
        {
            return Err(anyhow!("model.n_layer must be > 0 when set"));
        }
        if let Some(n_embd) = self.model.n_embd
            && n_embd == 0
        {
            return Err(anyhow!("model.n_embd must be > 0 when set"));
        }
        if let Some(n_head) = self.model.n_head
            && n_head == 0
        {
            return Err(anyhow!("model.n_head must be > 0 when set"));
        }
        let mut resolved_model = DragonConfig::default();
        if let Some(language_head) = &self.model.language_head {
            resolved_model.language_head = language_head.clone();
        }
        if let Some(tie_input_output_embeddings) = self.model.tie_input_output_embeddings {
            resolved_model.tie_input_output_embeddings = tie_input_output_embeddings;
        }
        if resolved_model.tie_input_output_embeddings
            && !matches!(
                resolved_model.language_head,
                LanguageHeadConfig::StandardTokenClassification
            )
        {
            return Err(anyhow!(
                "model.tie_input_output_embeddings requires model.language_head.type = \"standard_token_classification\""
            ));
        }
        if let Some(n_layer) = self.model.n_layer {
            resolved_model.n_layer = n_layer;
        }
        if let Some(n_embd) = self.model.n_embd {
            resolved_model.n_embd = n_embd;
        }
        if let Some(n_head) = self.model.n_head {
            resolved_model.n_head = n_head;
        }
        if let Some(slopes) = &self.model.alibi_slopes {
            let rotary = self
                .model
                .rotary_embedding
                .unwrap_or(resolved_model.fused_kernels.rotary_embedding);
            let memory = self
                .model
                .sequence_kernel
                .unwrap_or(resolved_model.sequence_kernel)
                .memory_system;
            if rotary != RotaryEmbedding::Alibi || memory != SequenceMemorySystem::LinearAttention {
                return Err(anyhow!(
                    "model.alibi_slopes requires ALiBi linear attention"
                ));
            }
            burn_dragon_core::kernel::linear_attention::validate_alibi_slopes(
                slopes,
                resolved_model.n_head,
            )
            .map_err(|error| anyhow!("model.{error}"))?;
        }
        if let Some(multiplier) = self.model.mlp_internal_dim_multiplier
            && multiplier == 0
        {
            return Err(anyhow!(
                "model.mlp_internal_dim_multiplier must be > 0 when set"
            ));
        }
        if let Some(multiplier) = self.model.mlp_internal_dim_multiplier {
            resolved_model.mlp_internal_dim_multiplier = multiplier;
        }
        if let Some(latent_total) = self.model.latent_total {
            if latent_total == 0 {
                return Err(anyhow!("model.latent_total must be > 0 when set"));
            }
            let resolved_n_embd = resolved_model.n_embd;
            if latent_total % resolved_n_embd != 0 {
                return Err(anyhow!(
                    "model.latent_total must be divisible by model.n_embd (got latent_total={} n_embd={})",
                    latent_total,
                    resolved_n_embd
                ));
            }
            if let Some(multiplier) = self.model.mlp_internal_dim_multiplier
                && multiplier * resolved_n_embd != latent_total
            {
                return Err(anyhow!(
                    "model.latent_total and model.mlp_internal_dim_multiplier disagree (latent_total={} n_embd={} multiplier={})",
                    latent_total,
                    resolved_n_embd,
                    multiplier
                ));
            }
            resolved_model.mlp_internal_dim_multiplier = latent_total / resolved_model.n_embd;
        }
        if let Some(initialization) = &self.model.initialization {
            initialization.validate().map_err(anyhow::Error::msg)?;
            resolved_model.initialization = initialization.clone();
        }
        if let Some(random_scaffold) = &self.model.random_scaffold {
            random_scaffold
                .validate_for_model(
                    resolved_model.n_embd,
                    resolved_model.n_head,
                    resolved_model.latent_total(),
                )
                .map_err(|message| anyhow!("model.random_scaffold {message}"))?;
            resolved_model.random_scaffold = random_scaffold.clone();
        }
        if let Some(sequence_kernel) = self.model.sequence_kernel {
            sequence_kernel
                .validate()
                .map_err(|message| anyhow!("model.sequence_kernel {message}"))?;
            resolved_model.sequence_kernel = sequence_kernel;
        }
        if let Some(sequence_kernel) = self.training.sequence_kernel_override {
            sequence_kernel
                .validate()
                .map_err(|message| anyhow!("training.sequence_kernel_override {message}"))?;
        }
        if let Some(mamba) = &self.model.mamba {
            let memory_system = self
                .training
                .sequence_kernel_override
                .unwrap_or(
                    self.model
                        .sequence_kernel
                        .unwrap_or(resolved_model.sequence_kernel),
                )
                .memory_system;
            mamba
                .validate(memory_system, resolved_model.n_embd)
                .map_err(|message| anyhow!("model.mamba {message}"))?;
            resolved_model.mamba = mamba.clone();
        }
        if let Some(gated_deltanet2) = &self.model.gated_deltanet2 {
            gated_deltanet2
                .validate(
                    resolved_model.n_head,
                    resolved_model.n_embd,
                    resolved_model.latent_per_head(),
                )
                .map_err(|message| anyhow!("model.gated_deltanet2 {message}"))?;
            resolved_model.gated_deltanet2 = gated_deltanet2.clone();
        }
        if matches!(
            self.training
                .sequence_kernel_override
                .unwrap_or(resolved_model.sequence_kernel)
                .memory_system,
            burn_dragon_core::SequenceMemorySystem::Mamba3StateSpaceDuality
        ) {
            resolved_model
                .mamba
                .validate(
                    resolved_model.sequence_kernel.memory_system,
                    resolved_model.n_embd,
                )
                .map_err(|message| anyhow!("resolved model.mamba {message}"))?;
        }
        if matches!(
            self.training
                .sequence_kernel_override
                .unwrap_or(resolved_model.sequence_kernel)
                .memory_system,
            burn_dragon_core::SequenceMemorySystem::GatedDeltaNet2
        ) {
            resolved_model
                .gated_deltanet2
                .validate(
                    resolved_model.n_head,
                    resolved_model.n_embd,
                    resolved_model.latent_per_head(),
                )
                .map_err(|message| anyhow!("resolved model.gated_deltanet2 {message}"))?;
        }
        if resolved_model.latent_total() % self.parallel.tensor.size != 0 {
            return Err(anyhow!(
                "resolved model.latent_total must be divisible by parallel.tensor.size (got latent_total={} tensor_size={})",
                resolved_model.latent_total(),
                self.parallel.tensor.size
            ));
        }
        if self.training.dynamics_anchor.enabled
            && !self.training.auto_batch_size.enabled
            && self.training.batch_size > 1
            && resolved_model.latent_total() >= 65_536
        {
            return Err(anyhow!(
                "training.dynamics_anchor with fixed training.batch_size > 1 is not allowed for resolved model.latent_total >= 65536; use training.batch_size=1 or enable training.auto_batch_size"
            ));
        }
        if self.training.predictive_coding.enabled
            && !self.training.auto_batch_size.enabled
            && self.training.batch_size > 1
            && resolved_model.latent_total() >= 16_384
        {
            return Err(anyhow!(
                "training.predictive_coding with fixed training.batch_size > 1 is not allowed for resolved model.latent_total >= 16384; use training.batch_size=1 or enable training.auto_batch_size"
            ));
        }
        if self.training.neuron_scaling.enabled {
            if resolved_model.random_scaffold.enabled {
                return Err(anyhow!(
                    "training.neuron_scaling is not compatible with model.random_scaffold.enabled; scaffold growth history must be represented by a new model revision"
                ));
            }
            let max_latent_total = self.training.neuron_scaling.max_latent_total;
            if max_latent_total < resolved_model.latent_total() {
                return Err(anyhow!(
                    "training.neuron_scaling.max_latent_total must be >= resolved model.latent_total (got max={} current={})",
                    max_latent_total,
                    resolved_model.latent_total()
                ));
            }
            if !max_latent_total.is_multiple_of(resolved_model.n_embd) {
                return Err(anyhow!(
                    "training.neuron_scaling.max_latent_total must be divisible by model.n_embd (got max={} n_embd={})",
                    max_latent_total,
                    resolved_model.n_embd
                ));
            }
            if !max_latent_total.is_multiple_of(resolved_model.n_head) {
                return Err(anyhow!(
                    "training.neuron_scaling.max_latent_total must be divisible by model.n_head (got max={} n_head={})",
                    max_latent_total,
                    resolved_model.n_head
                ));
            }
        }
        if resolved_model.random_scaffold.enabled
            && matches!(
                self.optimizer.name,
                burn_dragon_train::OptimizerKind::Eggroll
            )
        {
            return Err(anyhow!(
                "optimizer.name=eggroll does not yet support adapter-parameter population evaluation for model.random_scaffold; use optimizer.name=adamw for the paper-faithful local baseline"
            ));
        }
        if resolved_model.random_scaffold.enabled && self.training.continual_backprop.enabled {
            return Err(anyhow!(
                "training.continual_backprop is not compatible with model.random_scaffold because feature replacement mutates the immutable scaffold"
            ));
        }
        if resolved_model.random_scaffold.enabled && self.training.init_checkpoint_path.is_some() {
            return Err(anyhow!(
                "training.init_checkpoint_path transfer is not supported for model.random_scaffold; resume an exact scaffold checkpoint or start a fresh revision"
            ));
        }
        if matches!(
            self.parallel.tensor.partition,
            TensorParallelPartitionKind::HeadAligned
        ) && self.parallel.tensor.size > resolved_model.n_head
        {
            return Err(anyhow!(
                "parallel.tensor.partition=head_aligned requires parallel.tensor.size <= model.n_head (got tensor_size={} n_head={})",
                self.parallel.tensor.size,
                resolved_model.n_head
            ));
        }
        if let Some(schedule) = &self.model.latent_fanout_schedule
            && let Err(message) = resolved_model.validate_latent_fanout_schedule(schedule)
        {
            return Err(anyhow!(message));
        }
        if let Some(latent_reasoning) = &self.model.latent_reasoning {
            latent_reasoning.validate().map_err(anyhow::Error::msg)?;
            resolved_model.latent_reasoning = latent_reasoning.clone();
        }
        if let Some(next_latent_transition) = &self.model.next_latent_transition {
            next_latent_transition
                .validate()
                .map_err(anyhow::Error::msg)?;
            resolved_model.next_latent_transition = next_latent_transition.clone();
        }
        if let Some(hierarchical_dragon) = &self.model.hierarchical_dragon {
            resolved_model.hierarchical_dragon = hierarchical_dragon.clone();
            if hierarchical_dragon.enabled {
                if matches!(hierarchical_dragon.last_layers, Some(0)) {
                    return Err(anyhow!(
                        "model.hierarchical_dragon.last_layers must be > 0 when set"
                    ));
                }
                if hierarchical_dragon.fast_cycles == 0 {
                    return Err(anyhow!(
                        "model.hierarchical_dragon.fast_cycles must be > 0 when enabled"
                    ));
                }
                if hierarchical_dragon.slow_cycles == 0 {
                    return Err(anyhow!(
                        "model.hierarchical_dragon.slow_cycles must be > 0 when enabled"
                    ));
                }
                if !hierarchical_dragon.slow_to_fast_scale.is_finite()
                    || hierarchical_dragon.slow_to_fast_scale < 0.0
                {
                    return Err(anyhow!(
                        "model.hierarchical_dragon.slow_to_fast_scale must be finite and >= 0 when enabled"
                    ));
                }
                if !hierarchical_dragon.fast_to_slow_scale.is_finite()
                    || hierarchical_dragon.fast_to_slow_scale < 0.0
                {
                    return Err(anyhow!(
                        "model.hierarchical_dragon.fast_to_slow_scale must be finite and >= 0 when enabled"
                    ));
                }
                if resolved_model.y_neuron_recurrence.enabled {
                    return Err(anyhow!(
                        "model.hierarchical_dragon is not yet supported together with model.y_neuron_recurrence"
                    ));
                }
                if resolved_model.clocked_slow_memory.enabled {
                    return Err(anyhow!(
                        "model.hierarchical_dragon is not supported together with model.clocked_slow_memory"
                    ));
                }
                if self.parallel.pipeline.enabled {
                    return Err(anyhow!(
                        "model.hierarchical_dragon is not yet supported together with parallel.pipeline.enabled"
                    ));
                }
            }
        }
        let latent_jepa_can_run = self.training.latent_reasoning.enabled
            && self
                .training
                .latent_reasoning
                .jepa_future_offsets
                .iter()
                .any(|offset| *offset < self.training.block_size);
        if latent_jepa_can_run && !resolved_model.latent_reasoning.enabled {
            return Err(anyhow!(
                "training.latent_reasoning JEPA offsets within training.block_size require model.latent_reasoning.enabled"
            ));
        }
        if !self.training.latent_reasoning.eval_step_sweep.is_empty()
            && !resolved_model.latent_reasoning.enabled
        {
            return Err(anyhow!(
                "training.latent_reasoning.eval_step_sweep requires model.latent_reasoning.enabled"
            ));
        }
        if self.training.latent_reasoning.enabled
            && self.training.latent_reasoning.next_latent.enabled
            && !resolved_model.next_latent_transition.enabled
        {
            return Err(anyhow!(
                "training.latent_reasoning.next_latent.enabled requires model.next_latent_transition.enabled"
            ));
        }
        if self.training.latent_reasoning.enabled
            && self.training.latent_reasoning.energy_model.enabled
        {
            if !resolved_model.latent_reasoning.enabled {
                return Err(anyhow!(
                    "training.latent_reasoning.energy_model.enabled requires model.latent_reasoning.enabled"
                ));
            }
            if !resolved_model.latent_reasoning.energy_head {
                return Err(anyhow!(
                    "training.latent_reasoning.energy_model.enabled requires model.latent_reasoning.energy_head=true"
                ));
            }
        }
        if self.training.latent_reasoning.enabled
            && self.training.latent_reasoning.step_contract.enabled
            && !resolved_model.latent_reasoning.enabled
        {
            return Err(anyhow!(
                "training.latent_reasoning.step_contract.enabled requires model.latent_reasoning.enabled"
            ));
        }
        if self.training.latent_reasoning.enabled
            && self.training.latent_reasoning.next_latent.enabled
            && self.training.latent_reasoning.next_latent.token_kl_weight > f32::EPSILON
            && !resolved_model.language_head.uses_flat_token_logits()
        {
            return Err(anyhow!(
                "training.latent_reasoning.next_latent.token_kl_weight requires a flat token language head"
            ));
        }
        if self.training.latent_reasoning.enabled
            && self.training.latent_reasoning.step_contract.enabled
            && self.training.latent_reasoning.step_contract.token_kl_weight > f32::EPSILON
            && !resolved_model.language_head.uses_flat_token_logits()
        {
            return Err(anyhow!(
                "training.latent_reasoning.step_contract.token_kl_weight requires a flat token language head"
            ));
        }
        if let Some(dropout) = self.model.dropout
            && dropout < 0.0
        {
            return Err(anyhow!("model.dropout must be >= 0"));
        }
        if let Some(block_size) = self.model.block_size
            && block_size == 0
        {
            return Err(anyhow!("model.block_size must be > 0 when set"));
        }
        if let Some(rollout_fast_steps) = self.model.rollout_fast_steps_per_slow_step
            && !DragonConfig::is_valid_rollout_fast_steps(rollout_fast_steps)
        {
            return Err(anyhow!(
                "model.rollout_fast_steps_per_slow_step must be one of {:?} when set (got {})",
                DragonConfig::SUPPORTED_ROLLOUT_FAST_STEPS,
                rollout_fast_steps
            ));
        }
        if let Some(y_neuron_recurrence) = &self.model.y_neuron_recurrence
            && y_neuron_recurrence.enabled
        {
            if y_neuron_recurrence.carry_in_scale < 0.0 {
                return Err(anyhow!(
                    "model.y_neuron_recurrence.carry_in_scale must be >= 0 when enabled"
                ));
            }
            if matches!(y_neuron_recurrence.last_layers, Some(0)) {
                return Err(anyhow!(
                    "model.y_neuron_recurrence.last_layers must be > 0 when set"
                ));
            }
            if y_neuron_recurrence.chunk_tokens == 0 {
                return Err(anyhow!(
                    "model.y_neuron_recurrence.chunk_tokens must be > 0 when enabled"
                ));
            }
            if !(0.0..=1.0).contains(&y_neuron_recurrence.state_decay) {
                return Err(anyhow!(
                    "model.y_neuron_recurrence.state_decay must be in [0, 1] when enabled"
                ));
            }
            if y_neuron_recurrence.state_update_scale <= 0.0 {
                return Err(anyhow!(
                    "model.y_neuron_recurrence.state_update_scale must be > 0 when enabled"
                ));
            }
            if matches!(y_neuron_recurrence.state_rms_cap, Some(value) if value <= 0.0) {
                return Err(anyhow!(
                    "model.y_neuron_recurrence.state_rms_cap must be > 0 when set"
                ));
            }
        }
        if let Some(clocked_slow_memory) = &self.model.clocked_slow_memory
            && clocked_slow_memory.enabled
        {
            if matches!(clocked_slow_memory.last_layers, Some(0)) {
                return Err(anyhow!(
                    "model.clocked_slow_memory.last_layers must be > 0 when set"
                ));
            }
            if clocked_slow_memory.chunk_tokens == 0 {
                return Err(anyhow!(
                    "model.clocked_slow_memory.chunk_tokens must be > 0 when enabled"
                ));
            }
            if clocked_slow_memory.residual_scale <= 0.0 {
                return Err(anyhow!(
                    "model.clocked_slow_memory.residual_scale must be > 0 when enabled"
                ));
            }
            if matches!(self.model.y_neuron_recurrence.as_ref(), Some(value) if value.enabled) {
                return Err(anyhow!(
                    "model.clocked_slow_memory is not yet supported together with model.y_neuron_recurrence"
                ));
            }
        }
        if let Some(summary_memory) = &self.model.summary_memory
            && summary_memory.enabled
        {
            if matches!(summary_memory.last_layers, Some(0)) {
                return Err(anyhow!(
                    "model.summary_memory.last_layers must be > 0 when set"
                ));
            }
            if summary_memory.chunk_tokens == 0 {
                return Err(anyhow!(
                    "model.summary_memory.chunk_tokens must be > 0 when enabled"
                ));
            }
            if summary_memory.residual_scale <= 0.0 {
                return Err(anyhow!(
                    "model.summary_memory.residual_scale must be > 0 when enabled"
                ));
            }
            if !(0.0..=1.0).contains(&summary_memory.state_decay) {
                return Err(anyhow!(
                    "model.summary_memory.state_decay must be in [0, 1] when enabled"
                ));
            }
            if summary_memory.state_update_scale <= 0.0 {
                return Err(anyhow!(
                    "model.summary_memory.state_update_scale must be > 0 when enabled"
                ));
            }
            if summary_memory.surprise_gate_threshold < 0.0 {
                return Err(anyhow!(
                    "model.summary_memory.surprise_gate_threshold must be >= 0 when enabled"
                ));
            }
            if summary_memory.surprise_gate_sharpness <= 0.0 {
                return Err(anyhow!(
                    "model.summary_memory.surprise_gate_sharpness must be > 0 when enabled"
                ));
            }
            if matches!(
                summary_memory.write_trigger_text.as_ref(),
                Some(value) if value.trim().is_empty()
            ) {
                return Err(anyhow!(
                    "model.summary_memory.write_trigger_text must not be empty when set"
                ));
            }
            if matches!(
                summary_memory.write_trigger_token_ids.as_ref(),
                Some(value) if value.is_empty()
            ) {
                return Err(anyhow!(
                    "model.summary_memory.write_trigger_token_ids must not be empty when set"
                ));
            }
            if matches!(self.model.y_neuron_recurrence.as_ref(), Some(value) if value.enabled) {
                return Err(anyhow!(
                    "model.summary_memory is not yet supported together with model.y_neuron_recurrence"
                ));
            }
        }
        if let Some(mhc) = &self.model.mhc
            && mhc.enabled
        {
            if mhc.num_streams == 0 {
                return Err(anyhow!("model.mhc.num_streams must be > 0 when enabled"));
            }
            if mhc.num_views == 0 {
                return Err(anyhow!("model.mhc.num_views must be > 0 when enabled"));
            }
            if matches!(mhc.last_layers, Some(0)) {
                return Err(anyhow!("model.mhc.last_layers must be > 0 when set"));
            }
            if mhc.mhc_tau <= 0.0 {
                return Err(anyhow!("model.mhc.mhc_tau must be > 0 when enabled"));
            }
        }
        if let Some(attention_residual) = &self.model.attention_residual
            && attention_residual.enabled
        {
            if attention_residual.num_heads == 0 {
                return Err(anyhow!(
                    "model.attention_residual.num_heads must be > 0 when enabled"
                ));
            }
            if matches!(attention_residual.last_layers, Some(0)) {
                return Err(anyhow!(
                    "model.attention_residual.last_layers must be > 0 when set"
                ));
            }
            if matches!(attention_residual.history_window, Some(0)) {
                return Err(anyhow!(
                    "model.attention_residual.history_window must be > 0 when set"
                ));
            }
        }
        if let Some(block_attention_residual) = &self.model.block_attention_residual
            && block_attention_residual.enabled
        {
            if block_attention_residual.num_heads == 0 {
                return Err(anyhow!(
                    "model.block_attention_residual.num_heads must be > 0 when enabled"
                ));
            }
            if matches!(block_attention_residual.last_layers, Some(0)) {
                return Err(anyhow!(
                    "model.block_attention_residual.last_layers must be > 0 when set"
                ));
            }
            if block_attention_residual.layers_per_block == 0 {
                return Err(anyhow!(
                    "model.block_attention_residual.layers_per_block must be > 0 when enabled"
                ));
            }
            if matches!(block_attention_residual.block_history_window, Some(0)) {
                return Err(anyhow!(
                    "model.block_attention_residual.block_history_window must be > 0 when set"
                ));
            }
            if matches!(block_attention_residual.intra_block_history_window, Some(0)) {
                return Err(anyhow!(
                    "model.block_attention_residual.intra_block_history_window must be > 0 when set"
                ));
            }
        }
        if let Some(mhc) = self.model.mhc.as_ref()
            && mhc.enabled
            && self.model.residual_connector != Some(ResidualConnectorKind::Mhc)
        {
            return Err(anyhow!(
                "model.residual_connector = \"mhc\" is required when model.mhc.enabled = true"
            ));
        }
        if let Some(attention_residual) = self.model.attention_residual.as_ref()
            && attention_residual.enabled
            && self.model.residual_connector != Some(ResidualConnectorKind::AttentionResidual)
        {
            return Err(anyhow!(
                "model.residual_connector = \"attention_residual\" is required when model.attention_residual.enabled = true"
            ));
        }
        if let Some(block_attention_residual) = self.model.block_attention_residual.as_ref()
            && block_attention_residual.enabled
            && self.model.residual_connector != Some(ResidualConnectorKind::BlockAttentionResidual)
        {
            return Err(anyhow!(
                "model.residual_connector = \"block_attention_residual\" is required when model.block_attention_residual.enabled = true"
            ));
        }
        if let Some(residual_connector) = self.model.residual_connector {
            match residual_connector {
                ResidualConnectorKind::Vanilla => {}
                ResidualConnectorKind::Mhc => {
                    let mhc = self.model.mhc.as_ref().ok_or_else(|| {
                        anyhow!("model.mhc must be set when model.residual_connector = \"mhc\"")
                    })?;
                    if !mhc.enabled {
                        return Err(anyhow!(
                            "model.mhc.enabled must be true when model.residual_connector = \"mhc\""
                        ));
                    }
                }
                ResidualConnectorKind::AttentionResidual => {
                    let attention_residual = self
                        .model
                        .attention_residual
                        .as_ref()
                        .ok_or_else(|| anyhow!("model.attention_residual must be set when model.residual_connector = \"attention_residual\""))?;
                    if !attention_residual.enabled {
                        return Err(anyhow!(
                            "model.attention_residual.enabled must be true when model.residual_connector = \"attention_residual\""
                        ));
                    }
                }
                ResidualConnectorKind::BlockAttentionResidual => {
                    let block_attention_residual = self
                        .model
                        .block_attention_residual
                        .as_ref()
                        .ok_or_else(|| anyhow!("model.block_attention_residual must be set when model.residual_connector = \"block_attention_residual\""))?;
                    if !block_attention_residual.enabled {
                        return Err(anyhow!(
                            "model.block_attention_residual.enabled must be true when model.residual_connector = \"block_attention_residual\""
                        ));
                    }
                }
            }
        }

        if let Some(schedule) = &self.optimizer.lr_schedule {
            match schedule {
                LearningRateScheduleConfig::Constant { initial_lr }
                | LearningRateScheduleConfig::Cosine { initial_lr, .. }
                | LearningRateScheduleConfig::Linear { initial_lr, .. }
                | LearningRateScheduleConfig::Exponential { initial_lr, .. }
                | LearningRateScheduleConfig::Step { initial_lr, .. }
                | LearningRateScheduleConfig::Noam { initial_lr, .. } => {
                    if matches!(initial_lr.as_ref(), Some(value) if *value <= 0.0) {
                        return Err(anyhow!("optimizer.lr_schedule.initial_lr must be > 0"));
                    }
                }
            }

            match schedule {
                LearningRateScheduleConfig::Cosine {
                    min_lr,
                    warmup_steps,
                    num_iters,
                    ..
                } => {
                    if matches!(min_lr.as_ref(), Some(value) if *value < 0.0) {
                        return Err(anyhow!("optimizer.lr_schedule.min_lr must be >= 0"));
                    }
                    if matches!(warmup_steps, Some(0)) {
                        return Err(anyhow!("optimizer.lr_schedule.warmup_steps must be > 0"));
                    }
                    if matches!(num_iters, Some(0)) {
                        return Err(anyhow!("optimizer.lr_schedule.num_iters must be > 0"));
                    }
                }
                LearningRateScheduleConfig::Linear {
                    final_lr,
                    num_iters,
                    ..
                } => {
                    if *final_lr < 0.0 {
                        return Err(anyhow!("optimizer.lr_schedule.final_lr must be >= 0"));
                    }
                    if matches!(num_iters, Some(0)) {
                        return Err(anyhow!("optimizer.lr_schedule.num_iters must be > 0"));
                    }
                }
                LearningRateScheduleConfig::Exponential { gamma, .. } => {
                    if *gamma <= 0.0 {
                        return Err(anyhow!("optimizer.lr_schedule.gamma must be > 0"));
                    }
                }
                LearningRateScheduleConfig::Step {
                    gamma, step_size, ..
                } => {
                    if *gamma <= 0.0 {
                        return Err(anyhow!("optimizer.lr_schedule.gamma must be > 0"));
                    }
                    if matches!(step_size, Some(0)) {
                        return Err(anyhow!("optimizer.lr_schedule.step_size must be > 0"));
                    }
                }
                LearningRateScheduleConfig::Noam {
                    warmup_steps,
                    model_size,
                    ..
                } => {
                    if matches!(warmup_steps, Some(0)) {
                        return Err(anyhow!("optimizer.lr_schedule.warmup_steps must be > 0"));
                    }
                    if matches!(model_size, Some(0)) {
                        return Err(anyhow!("optimizer.lr_schedule.model_size must be > 0"));
                    }
                }
                LearningRateScheduleConfig::Constant { .. } => {}
            }
        }
        Ok(())
    }
}

fn validate_dataset_source(
    source: &DatasetSourceConfig,
    tokenizer_kind: &TokenizerKind,
    _allow_validation_only_hf: bool,
    label: &str,
) -> Result<()> {
    match source {
        DatasetSourceConfig::NemotronClimbMix { max_records, .. } => {
            if matches!(max_records, Some(0)) {
                return Err(anyhow!("{label}.max_records must be > 0 when set"));
            }
            if !matches!(tokenizer_kind, TokenizerKind::Pretokenized(_)) {
                return Err(anyhow!(
                    "{label}.tokenizer.type must be `pretokenized` for climbmix datasets"
                ));
            }
        }
        DatasetSourceConfig::UniversalityManifest { manifest } => {
            if manifest.as_os_str().is_empty() {
                return Err(anyhow!("{label}.manifest must not be empty"));
            }
            if !matches!(tokenizer_kind, TokenizerKind::Pretokenized(_)) {
                return Err(anyhow!(
                    "{label}.tokenizer.type must be `pretokenized` for universality manifests"
                ));
            }
        }
        DatasetSourceConfig::UniversalityNca { config } => {
            if config.as_os_str().is_empty() {
                return Err(anyhow!("{label}.config must not be empty"));
            }
            if !matches!(tokenizer_kind, TokenizerKind::Pretokenized(_)) {
                return Err(anyhow!(
                    "{label}.tokenizer.type must be `pretokenized` for on-the-fly universality NCA datasets"
                ));
            }
        }
        DatasetSourceConfig::UniversalityRuliad { config } => {
            if config.as_os_str().is_empty() {
                return Err(anyhow!("{label}.config must not be empty"));
            }
            if !matches!(tokenizer_kind, TokenizerKind::Pretokenized(_)) {
                return Err(anyhow!(
                    "{label}.tokenizer.type must be `pretokenized` for on-the-fly universality ruliad datasets"
                ));
            }
        }
    }
    Ok(())
}
