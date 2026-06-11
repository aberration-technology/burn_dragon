use anyhow::{Result, anyhow};
use std::collections::HashSet;

use burn_dragon_core::{
    DragonConfig, ResidualConnectorKind, objective::validate_training_objective_config,
};
use burn_dragon_train::{
    LearningRateScheduleConfig, ParallelismKind, PipelineCommunicationKind, PipelineScheduleKind,
    TensorParallelPartitionKind, train::pipeline::TrainingLaunchMode,
};

use super::{DatasetSourceConfig, TrainingConfig};
use crate::tokenizer::TokenizerKind;

impl TrainingConfig {
    pub fn validate(&self) -> Result<()> {
        if self.training.block_size == 0 {
            return Err(anyhow!("training.block_size must be > 0"));
        }
        if let Some(tbptt_chunk_size) = self.training.tbptt_chunk_size {
            if tbptt_chunk_size == 0 {
                return Err(anyhow!("training.tbptt_chunk_size must be > 0 when set"));
            }
            if tbptt_chunk_size > self.training.block_size {
                return Err(anyhow!(
                    "training.tbptt_chunk_size must be <= training.block_size (got {} > {})",
                    tbptt_chunk_size,
                    self.training.block_size
                ));
            }
        }
        if let Some(min_logical_block_size) = self.training.min_logical_block_size
            && min_logical_block_size == 0
        {
            return Err(anyhow!(
                "training.min_logical_block_size must be > 0 when set"
            ));
        }
        if self.training.tbptt_persist_across_steps && self.training.tbptt_chunk_size.is_none() {
            return Err(anyhow!(
                "training.tbptt_persist_across_steps requires training.tbptt_chunk_size"
            ));
        }
        if self.training.batch_size == 0 {
            return Err(anyhow!("training.batch_size must be > 0"));
        }
        if self.training.gradient_accumulation_steps == 0 {
            return Err(anyhow!("training.gradient_accumulation_steps must be > 0"));
        }
        if self.parallel.world_size == 0 {
            return Err(anyhow!("parallel.world_size must be > 0"));
        }
        if self.parallel.data.size == 0 {
            return Err(anyhow!("parallel.data.size must be > 0"));
        }
        let collective_globals = (
            self.parallel.data.collective_num_nodes,
            self.parallel.data.collective_global_address.as_ref(),
            self.parallel.data.collective_node_address.as_ref(),
            self.parallel.data.collective_data_service_port,
        );
        match collective_globals {
            (None, None, None, None) => {}
            (Some(num_nodes), Some(global_address), Some(node_address), Some(port)) => {
                if num_nodes == 0 {
                    return Err(anyhow!(
                        "parallel.data.collective_num_nodes must be > 0 when set"
                    ));
                }
                if global_address.trim().is_empty() {
                    return Err(anyhow!(
                        "parallel.data.collective_global_address must not be empty when set"
                    ));
                }
                if node_address.trim().is_empty() {
                    return Err(anyhow!(
                        "parallel.data.collective_node_address must not be empty when set"
                    ));
                }
                if port == 0 {
                    return Err(anyhow!(
                        "parallel.data.collective_data_service_port must be > 0 when set"
                    ));
                }
            }
            _ => {
                return Err(anyhow!(
                    "parallel.data collective global settings must either all be set or all be omitted"
                ));
            }
        }
        if self.parallel.tensor.size == 0 {
            return Err(anyhow!("parallel.tensor.size must be > 0"));
        }
        let pipeline_stage_multiplier = if self.parallel.pipeline.enabled {
            self.parallel.pipeline.stage_count.max(1)
        } else {
            1
        };
        let expected_world_size = self
            .parallel
            .data
            .size
            .checked_mul(self.parallel.tensor.size)
            .and_then(|value| value.checked_mul(pipeline_stage_multiplier))
            .ok_or_else(|| anyhow!("parallel size configuration overflow"))?;
        if self.parallel.mode != ParallelismKind::Single
            && expected_world_size != self.parallel.world_size
        {
            return Err(anyhow!(
                "parallel.data.size * parallel.tensor.size * pipeline_stage_multiplier must equal parallel.world_size (got {} * {} * {} != {})",
                self.parallel.data.size,
                self.parallel.tensor.size,
                pipeline_stage_multiplier,
                self.parallel.world_size
            ));
        }
        match self.parallel.mode {
            ParallelismKind::Single => {
                if self.parallel.world_size != 1
                    || self.parallel.data.size != 1
                    || self.parallel.tensor.size != 1
                {
                    return Err(anyhow!(
                        "parallel.mode=single requires parallel.world_size=1, parallel.data.size=1, and parallel.tensor.size=1"
                    ));
                }
                if self.parallel.fsdp.enabled {
                    return Err(anyhow!(
                        "parallel.fsdp.enabled must be false when parallel.mode=single"
                    ));
                }
            }
            ParallelismKind::Ddp => {
                if self.parallel.world_size < 2 {
                    return Err(anyhow!(
                        "parallel.mode=ddp requires parallel.world_size >= 2"
                    ));
                }
                if self.parallel.tensor.size != 1 {
                    return Err(anyhow!(
                        "parallel.mode=ddp requires parallel.tensor.size = 1"
                    ));
                }
                if self.parallel.data.size * pipeline_stage_multiplier != self.parallel.world_size {
                    return Err(anyhow!(
                        "parallel.mode=ddp requires parallel.data.size * pipeline_stage_multiplier = parallel.world_size"
                    ));
                }
                if self.parallel.fsdp.enabled {
                    return Err(anyhow!(
                        "parallel.fsdp.enabled must be false when parallel.mode=ddp"
                    ));
                }
            }
            ParallelismKind::Fsdp => {
                if self.parallel.world_size < 2 {
                    return Err(anyhow!(
                        "parallel.mode=fsdp requires parallel.world_size >= 2"
                    ));
                }
                if self.parallel.tensor.size != 1 {
                    return Err(anyhow!(
                        "parallel.mode=fsdp requires parallel.tensor.size = 1"
                    ));
                }
                if self.parallel.data.size * pipeline_stage_multiplier != self.parallel.world_size {
                    return Err(anyhow!(
                        "parallel.mode=fsdp requires parallel.data.size * pipeline_stage_multiplier = parallel.world_size"
                    ));
                }
                if !self.parallel.fsdp.enabled {
                    return Err(anyhow!(
                        "parallel.fsdp.enabled must be true when parallel.mode=fsdp"
                    ));
                }
            }
            ParallelismKind::TensorParallelNeuron => {
                if self.parallel.world_size < 2 {
                    return Err(anyhow!(
                        "parallel.mode=tensor_parallel_neuron requires parallel.world_size >= 2"
                    ));
                }
                if self.parallel.data.size != 1 {
                    return Err(anyhow!(
                        "parallel.mode=tensor_parallel_neuron requires parallel.data.size = 1"
                    ));
                }
                if self.parallel.tensor.size * pipeline_stage_multiplier != self.parallel.world_size
                {
                    return Err(anyhow!(
                        "parallel.mode=tensor_parallel_neuron requires parallel.tensor.size * pipeline_stage_multiplier = parallel.world_size"
                    ));
                }
                if self.parallel.fsdp.enabled {
                    return Err(anyhow!(
                        "parallel.fsdp.enabled must be false when parallel.mode=tensor_parallel_neuron"
                    ));
                }
            }
            ParallelismKind::Hybrid2D => {
                if self.parallel.world_size < 4 {
                    return Err(anyhow!(
                        "parallel.mode=hybrid_2d requires parallel.world_size >= 4"
                    ));
                }
                if self.parallel.data.size < 2 || self.parallel.tensor.size < 2 {
                    return Err(anyhow!(
                        "parallel.mode=hybrid_2d requires parallel.data.size >= 2 and parallel.tensor.size >= 2"
                    ));
                }
            }
        }
        if self.parallel.pipeline.enabled {
            if self.parallel.pipeline.stage_count == 0 {
                return Err(anyhow!(
                    "parallel.pipeline.stage_count must be > 0 when pipeline is enabled"
                ));
            }
            if self.parallel.pipeline.virtual_stages_per_rank == 0 {
                return Err(anyhow!(
                    "parallel.pipeline.virtual_stages_per_rank must be > 0 when pipeline is enabled"
                ));
            }
            if self.parallel.pipeline.microbatches == 0 {
                return Err(anyhow!(
                    "parallel.pipeline.microbatches must be > 0 when pipeline is enabled"
                ));
            }
            if self.parallel.pipeline.microbatches > self.training.batch_size {
                return Err(anyhow!(
                    "parallel.pipeline.microbatches must be <= training.batch_size (got {} > {})",
                    self.parallel.pipeline.microbatches,
                    self.training.batch_size
                ));
            }
            if self.parallel.mode != ParallelismKind::Single
                && self.parallel.pipeline.stage_count > self.parallel.world_size
            {
                return Err(anyhow!(
                    "parallel.pipeline.stage_count must be <= parallel.world_size (got {} > {})",
                    self.parallel.pipeline.stage_count,
                    self.parallel.world_size
                ));
            }
            if self.parallel.pipeline.virtual_stages_per_rank > self.parallel.pipeline.stage_count {
                return Err(anyhow!(
                    "parallel.pipeline.virtual_stages_per_rank must be <= parallel.pipeline.stage_count (got {} > {})",
                    self.parallel.pipeline.virtual_stages_per_rank,
                    self.parallel.pipeline.stage_count
                ));
            }
            if matches!(
                self.parallel.pipeline.schedule,
                PipelineScheduleKind::Interleaved1f1b
            ) && self.parallel.pipeline.microbatches < self.parallel.pipeline.stage_count
            {
                return Err(anyhow!(
                    "parallel.pipeline.microbatches must be >= parallel.pipeline.stage_count for interleaved_1f1b (got {} < {})",
                    self.parallel.pipeline.microbatches,
                    self.parallel.pipeline.stage_count
                ));
            }
            if self.parallel.pipeline.cache.max_inflight_microbatches == 0 {
                return Err(anyhow!(
                    "parallel.pipeline.cache.max_inflight_microbatches must be > 0 when pipeline is enabled"
                ));
            }
        } else if self.parallel.pipeline.cache.enabled {
            return Err(anyhow!(
                "parallel.pipeline.cache.enabled requires parallel.pipeline.enabled"
            ));
        }
        if self.parallel.pipeline.cache.enabled
            && self.parallel.pipeline.communication != PipelineCommunicationKind::BlockResidualCache
        {
            return Err(anyhow!(
                "parallel.pipeline.cache.enabled requires parallel.pipeline.communication = \"block_residual_cache\""
            ));
        }
        if self.parallel.pipeline.enabled
            && self.parallel.pipeline.communication == PipelineCommunicationKind::BlockResidualCache
            && self.model.residual_connector != Some(ResidualConnectorKind::BlockAttentionResidual)
        {
            return Err(anyhow!(
                "parallel.pipeline.communication = \"block_residual_cache\" requires model.residual_connector = \"block_attention_residual\""
            ));
        }
        if matches!(self.training.target_effective_batch_size, Some(0)) {
            return Err(anyhow!(
                "training.target_effective_batch_size must be > 0 when set"
            ));
        }
        if self.training.max_iters == 0 {
            return Err(anyhow!("training.max_iters must be > 0"));
        }
        if self.training.checkpoint_interval_iters == 0 {
            return Err(anyhow!("training.checkpoint_interval_iters must be > 0"));
        }
        if self.training.log_frequency == 0 {
            return Err(anyhow!("training.log_frequency must be > 0"));
        }
        if self.training.init_checkpoint_epoch.is_some()
            && self.training.init_checkpoint_path.is_none()
        {
            return Err(anyhow!(
                "training.init_checkpoint_epoch requires training.init_checkpoint_path"
            ));
        }
        if self.training.init_transfer.backbone_blend_alpha.is_some()
            && self.training.init_checkpoint_path.is_none()
        {
            return Err(anyhow!(
                "training.init_transfer.backbone_blend_alpha requires training.init_checkpoint_path"
            ));
        }
        if self
            .training
            .init_transfer
            .interface_checkpoint_path
            .is_some()
            && self.training.init_checkpoint_path.is_none()
        {
            return Err(anyhow!(
                "training.init_transfer.interface_checkpoint_path requires training.init_checkpoint_path"
            ));
        }
        if self
            .training
            .init_transfer
            .interface_checkpoint_epoch
            .is_some()
            && self
                .training
                .init_transfer
                .interface_checkpoint_path
                .is_none()
        {
            return Err(anyhow!(
                "training.init_transfer.interface_checkpoint_epoch requires training.init_transfer.interface_checkpoint_path"
            ));
        }
        if (self
            .training
            .init_transfer
            .preserve_interface_input_embedding
            || self.training.init_transfer.preserve_interface_output_head
            || self
                .training
                .init_transfer
                .interface_output_head_blend_alpha
                .is_some())
            && self
                .training
                .init_transfer
                .interface_checkpoint_path
                .is_none()
        {
            return Err(anyhow!(
                "training.init_transfer.preserve_interface_input_embedding, training.init_transfer.preserve_interface_output_head, and training.init_transfer.interface_output_head_blend_alpha require training.init_transfer.interface_checkpoint_path"
            ));
        }
        if self
            .training
            .init_transfer
            .interface_output_head_blend_alpha
            .is_some()
            && self.training.init_transfer.preserve_interface_output_head
        {
            return Err(anyhow!(
                "training.init_transfer.interface_output_head_blend_alpha cannot be combined with training.init_transfer.preserve_interface_output_head"
            ));
        }
        if self.training.init_transfer.decoder_blend_alpha.is_some()
            && self.training.init_checkpoint_path.is_none()
        {
            return Err(anyhow!(
                "training.init_transfer.decoder_blend_alpha requires training.init_checkpoint_path"
            ));
        }
        if self.training.init_transfer.norm_blend_alpha.is_some()
            && self.training.init_checkpoint_path.is_none()
        {
            return Err(anyhow!(
                "training.init_transfer.norm_blend_alpha requires training.init_checkpoint_path"
            ));
        }
        if (self.training.init_transfer.backbone_grad_scale.is_some()
            || self
                .training
                .init_transfer
                .backbone_grad_scale_steps
                .is_some())
            && self.training.init_checkpoint_path.is_none()
        {
            return Err(anyhow!(
                "training.init_transfer.backbone_grad_scale and training.init_transfer.backbone_grad_scale_steps require training.init_checkpoint_path"
            ));
        }
        if self.training.init_transfer.fresh_top_layers.is_some()
            && self.training.init_checkpoint_path.is_none()
        {
            return Err(anyhow!(
                "training.init_transfer.fresh_top_layers requires training.init_checkpoint_path"
            ));
        }
        if self.training.init_transfer.preserve_fresh_decoder
            && self.training.init_checkpoint_path.is_none()
        {
            return Err(anyhow!(
                "training.init_transfer.preserve_fresh_decoder requires training.init_checkpoint_path"
            ));
        }
        if self.training.init_transfer.preserve_fresh_norm
            && self.training.init_checkpoint_path.is_none()
        {
            return Err(anyhow!(
                "training.init_transfer.preserve_fresh_norm requires training.init_checkpoint_path"
            ));
        }
        if self.training.init_transfer.match_fresh_rms
            && self.training.init_checkpoint_path.is_none()
        {
            return Err(anyhow!(
                "training.init_transfer.match_fresh_rms requires training.init_checkpoint_path"
            ));
        }
        if let Some(alpha) = self.training.init_transfer.backbone_blend_alpha
            && !(0.0..=1.0).contains(&alpha)
        {
            return Err(anyhow!(
                "training.init_transfer.backbone_blend_alpha must be in [0, 1]"
            ));
        }
        if let Some(alpha) = self.training.init_transfer.decoder_blend_alpha
            && !(0.0..=1.0).contains(&alpha)
        {
            return Err(anyhow!(
                "training.init_transfer.decoder_blend_alpha must be in [0, 1]"
            ));
        }
        if let Some(alpha) = self.training.init_transfer.norm_blend_alpha
            && !(0.0..=1.0).contains(&alpha)
        {
            return Err(anyhow!(
                "training.init_transfer.norm_blend_alpha must be in [0, 1]"
            ));
        }
        if let Some(alpha) = self
            .training
            .init_transfer
            .interface_output_head_blend_alpha
            && !(0.0..=1.0).contains(&alpha)
        {
            return Err(anyhow!(
                "training.init_transfer.interface_output_head_blend_alpha must be in [0, 1]"
            ));
        }
        if self.training.continual_backprop.enabled {
            if !(0.0..1.0).contains(&self.training.continual_backprop.utility_decay) {
                return Err(anyhow!(
                    "training.continual_backprop.utility_decay must be in [0, 1)"
                ));
            }
            if self.training.continual_backprop.replacement_rate <= 0.0
                || !self
                    .training
                    .continual_backprop
                    .replacement_rate
                    .is_finite()
            {
                return Err(anyhow!(
                    "training.continual_backprop.replacement_rate must be finite and > 0"
                ));
            }
            if self.training.continual_backprop.maturity_steps == 0 {
                return Err(anyhow!(
                    "training.continual_backprop.maturity_steps must be > 0"
                ));
            }
            if self.training.continual_backprop.sample_interval_steps == 0 {
                return Err(anyhow!(
                    "training.continual_backprop.sample_interval_steps must be > 0"
                ));
            }
            if self.training.continual_backprop.replace_interval_steps == 0 {
                return Err(anyhow!(
                    "training.continual_backprop.replace_interval_steps must be > 0"
                ));
            }
            if self.training.continual_backprop.utility_epsilon <= 0.0
                || !self.training.continual_backprop.utility_epsilon.is_finite()
            {
                return Err(anyhow!(
                    "training.continual_backprop.utility_epsilon must be finite and > 0"
                ));
            }
            if self.training.continual_backprop.lr_coupling_power < 0.0
                || !self
                    .training
                    .continual_backprop
                    .lr_coupling_power
                    .is_finite()
            {
                return Err(anyhow!(
                    "training.continual_backprop.lr_coupling_power must be finite and >= 0"
                ));
            }
        }
        let mut seen_module_lr_targets = HashSet::new();
        for entry in &self.training.module_lr_scales {
            if entry.scale <= 0.0 || !entry.scale.is_finite() {
                return Err(anyhow!(
                    "training.module_lr_scales[{:#?}] scale must be finite and > 0",
                    entry.target
                ));
            }
            if let Some(schedule) = &entry.schedule {
                if schedule.final_scale <= 0.0 || !schedule.final_scale.is_finite() {
                    return Err(anyhow!(
                        "training.module_lr_scales[{:#?}].schedule.final_scale must be finite and > 0",
                        entry.target
                    ));
                }
                if !schedule.start_fraction.is_finite()
                    || !(0.0..=1.0).contains(&schedule.start_fraction)
                {
                    return Err(anyhow!(
                        "training.module_lr_scales[{:#?}].schedule.start_fraction must be finite and in [0, 1]",
                        entry.target
                    ));
                }
                if !schedule.end_fraction.is_finite()
                    || !(0.0..=1.0).contains(&schedule.end_fraction)
                {
                    return Err(anyhow!(
                        "training.module_lr_scales[{:#?}].schedule.end_fraction must be finite and in [0, 1]",
                        entry.target
                    ));
                }
                if schedule.end_fraction < schedule.start_fraction {
                    return Err(anyhow!(
                        "training.module_lr_scales[{:#?}].schedule.end_fraction must be >= start_fraction",
                        entry.target
                    ));
                }
            }
            if !seen_module_lr_targets.insert(entry.target) {
                return Err(anyhow!(
                    "training.module_lr_scales contains duplicate target {:?}",
                    entry.target
                ));
            }
        }
        if self.training.init_transfer.backbone_grad_scale.is_some()
            ^ self
                .training
                .init_transfer
                .backbone_grad_scale_steps
                .is_some()
        {
            return Err(anyhow!(
                "training.init_transfer.backbone_grad_scale and training.init_transfer.backbone_grad_scale_steps must be set together"
            ));
        }
        if let Some(scale) = self.training.init_transfer.backbone_grad_scale
            && !(0.0..=1.0).contains(&scale)
        {
            return Err(anyhow!(
                "training.init_transfer.backbone_grad_scale must be in [0, 1]"
            ));
        }
        if matches!(
            self.training.init_transfer.backbone_grad_scale_steps,
            Some(0)
        ) {
            return Err(anyhow!(
                "training.init_transfer.backbone_grad_scale_steps must be > 0 when set"
            ));
        }
        if matches!(self.training.init_transfer.fresh_top_layers, Some(0)) {
            return Err(anyhow!(
                "training.init_transfer.fresh_top_layers must be > 0 when set"
            ));
        }
        match self.training.launch_mode {
            TrainingLaunchMode::Fresh => {
                if self.training.resume_run_dir.is_some()
                    || self.training.resume_checkpoint_epoch.is_some()
                    || self.training.init_checkpoint_path.is_some()
                    || self.training.init_checkpoint_epoch.is_some()
                    || self.training.init_transfer != Default::default()
                {
                    return Err(anyhow!(
                        "training.launch_mode = \"fresh\" requires resume and init checkpoint settings to all be unset"
                    ));
                }
            }
            TrainingLaunchMode::ResumeExactRun => {
                if self.training.resume_run_dir.is_none() {
                    return Err(anyhow!(
                        "training.launch_mode = \"resume_exact_run\" requires training.resume_run_dir"
                    ));
                }
                if self.training.init_checkpoint_path.is_some()
                    || self.training.init_checkpoint_epoch.is_some()
                    || self.training.init_transfer != Default::default()
                {
                    return Err(anyhow!(
                        "training.launch_mode = \"resume_exact_run\" cannot be combined with init checkpoint or init transfer settings"
                    ));
                }
            }
            TrainingLaunchMode::ResumeLatestCheckpointIfPresent => {
                if self.training.resume_run_dir.is_some() {
                    return Err(anyhow!(
                        "training.launch_mode = \"resume_latest_checkpoint_if_present\" cannot be combined with training.resume_run_dir"
                    ));
                }
                if self.training.resume_checkpoint_epoch.is_some() {
                    return Err(anyhow!(
                        "training.launch_mode = \"resume_latest_checkpoint_if_present\" cannot be combined with training.resume_checkpoint_epoch"
                    ));
                }
            }
            TrainingLaunchMode::InitFromCheckpoint => {
                if self.training.init_checkpoint_path.is_none() {
                    return Err(anyhow!(
                        "training.launch_mode = \"init_from_checkpoint\" requires training.init_checkpoint_path"
                    ));
                }
                if self.training.resume_run_dir.is_some()
                    || self.training.resume_checkpoint_epoch.is_some()
                {
                    return Err(anyhow!(
                        "training.launch_mode = \"init_from_checkpoint\" cannot be combined with training.resume_run_dir or training.resume_checkpoint_epoch"
                    ));
                }
            }
        }
        if self.wgpu.training.startup_autotune.enabled {
            let autotune = &self.wgpu.training.startup_autotune;
            if autotune.target_device_memory_mb == 0 {
                return Err(anyhow!(
                    "wgpu.training.startup_autotune.target_device_memory_mb must be > 0 when enabled"
                ));
            }
            if autotune.min_batch_size == 0 {
                return Err(anyhow!(
                    "wgpu.training.startup_autotune.min_batch_size must be > 0 when enabled"
                ));
            }
            if matches!(autotune.max_batch_size, Some(0)) {
                return Err(anyhow!(
                    "wgpu.training.startup_autotune.max_batch_size must be > 0 when set"
                ));
            }
            if autotune.probe_steps == 0 {
                return Err(anyhow!(
                    "wgpu.training.startup_autotune.probe_steps must be > 0 when enabled"
                ));
            }
            if let Some(max_batch_size) = autotune.max_batch_size
                && max_batch_size < autotune.min_batch_size
            {
                return Err(anyhow!(
                    "wgpu.training.startup_autotune.max_batch_size must be >= min_batch_size"
                ));
            }
        }
        if let Some(epochs) = self.training.epochs
            && epochs == 0
        {
            return Err(anyhow!("training.epochs must be > 0"));
        }
        self.optimizer.validate()?;
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
        if let Some(n_layer) = self.model.n_layer {
            resolved_model.n_layer = n_layer;
        }
        if let Some(n_embd) = self.model.n_embd {
            resolved_model.n_embd = n_embd;
        }
        if let Some(n_head) = self.model.n_head {
            resolved_model.n_head = n_head;
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

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use crate::config::TrainingObjectiveConfig;

    fn parse_config(extra_training: &str) -> TrainingConfig {
        let toml = format!(
            r#"
[dataset]
cache_dir = "target/test-cache"
type = "nemotron_climb_mix"
max_records = 4

[training]
block_size = 8
batch_size = 2
max_iters = 1
log_frequency = 1
{extra_training}

[optimizer]
learning_rate = 0.001
weight_decay = 0.0

[generation]
prompt = ""
"#
        );
        toml::from_str(&toml).expect("training config should parse")
    }

    #[test]
    fn default_objective_is_next_token() {
        let config = parse_config("");
        assert!(config.training.objective.is_next_token());
        config.validate().expect("default objective validates");
    }

    #[test]
    fn sdft_objective_config_validates() {
        let config = parse_config(
            r#"
[training.objective]
type = "sdft"
max_completion_tokens = 4
teacher_update_rate = 0.25
"#,
        );
        assert!(matches!(
            config.training.objective,
            TrainingObjectiveConfig::Sdft(_)
        ));
        config.validate().expect("sdft objective validates");
    }

    #[test]
    fn sdpo_rejects_invalid_alpha() {
        let config = parse_config(
            r#"
[training.objective]
type = "sdpo"
alpha = 1.25
"#,
        );
        let err = config
            .validate()
            .expect_err("invalid sdpo alpha should fail");
        assert!(
            err.to_string().contains("training.objective.alpha"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn sdft_rejects_unwired_top_entropy_quantile() {
        let config = parse_config(
            r#"
[training.objective]
type = "sdft"
top_entropy_quantile = 0.25
"#,
        );
        let err = config
            .validate()
            .expect_err("unwired SDFT entropy mask should fail");
        assert!(
            err.to_string().contains("top_entropy_quantile"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn sdpo_rejects_unwired_reward_feedback_fields() {
        let config = parse_config(
            r#"
[training.objective]
type = "sdpo"
success_reward_threshold = 1.0
include_environment_feedback = true
"#,
        );
        let err = config
            .validate()
            .expect_err("unwired SDPO reward/feedback fields should fail");
        assert!(
            err.to_string().contains("success_reward_threshold"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn sdpo_rejects_unwired_topk_fields() {
        let config = parse_config(
            r#"
[training.objective]
type = "sdpo"
distillation_topk = 100
"#,
        );
        let err = config
            .validate()
            .expect_err("unwired SDPO top-k distillation should fail");
        assert!(
            err.to_string().contains("distillation_topk"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn sdft_sdpo_composite_objective_config_validates() {
        let config = parse_config(
            r#"
[training.objective]
type = "sdft_sdpo"
sdft_weight = 0.25
sdpo_weight = 0.75

[training.objective.sdft]
max_completion_tokens = 2
generate_from_teacher = true

[training.objective.sdpo]
group_size = 2
max_completion_tokens = 2
alpha = 0.25
"#,
        );
        assert!(matches!(
            config.training.objective,
            TrainingObjectiveConfig::SdftSdpo(_)
        ));
        config
            .validate()
            .expect("composite SDFT/SDPO objective validates");
    }

    #[test]
    fn reservoir_model_initialization_config_validates() {
        let config = parse_config(
            r#"
[model]
n_layer = 1
n_embd = 32
n_head = 4
latent_total = 64

[model.initialization]
kind = "reservoir"

[model.initialization.reservoir]
seed = 1337
density = 0.08
encoder_value_scale = 0.70
decoder_scale = 1.00

[model.initialization.topology_prior]
kind = "modular_bridges"
community_count = 4
bridge_fraction = 0.03
intra_community_gain = 1.5
inter_community_gain = 0.5
bridge_gain = 1.0
"#,
        );
        config
            .validate()
            .expect("reservoir model initialization validates");
    }

    #[test]
    fn legacy_gdpo_flag_is_mutually_exclusive_with_objective_switch() {
        let config = parse_config(
            r#"
[training.gdpo]
enabled = true
"#,
        );
        let err = config
            .validate()
            .expect_err("legacy gdpo objective flag should fail");
        assert!(
            err.to_string().contains("training.gdpo.enabled"),
            "unexpected error: {err}"
        );
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
