//! Parallel execution, checkpoint transfer, optimizer, and launch contracts.

use super::*;

impl TrainingConfig {
    pub(super) fn validate_execution_contracts(&self) -> Result<()> {
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
            if self
                .training
                .continual_backprop
                .max_replacements_per_interval
                == 0
            {
                return Err(anyhow!(
                    "training.continual_backprop.max_replacements_per_interval must be > 0"
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
        if self.training.resume_horizon_extension.enabled {
            if !matches!(
                self.training.launch_mode,
                TrainingLaunchMode::ResumeExactRun
            ) {
                return Err(anyhow!(
                    "training.resume_horizon_extension.enabled requires training.launch_mode = \"resume_exact_run\""
                ));
            }
            if self.training.epochs.is_some() {
                return Err(anyhow!(
                    "training.resume_horizon_extension only supports max_iters-based runs; training.epochs must be unset"
                ));
            }
            if self
                .training
                .module_lr_scales
                .iter()
                .any(|entry| entry.schedule.is_some())
            {
                return Err(anyhow!(
                    "training.resume_horizon_extension cannot change a run with fraction-of-total module_lr_scales schedules"
                ));
            }
            let schedule_is_horizon_independent = match &self.optimizer.lr_schedule {
                None | Some(LearningRateScheduleConfig::Constant { .. }) => true,
                Some(LearningRateScheduleConfig::Exponential { .. }) => true,
                Some(LearningRateScheduleConfig::Cosine { num_iters, .. })
                | Some(LearningRateScheduleConfig::Linear { num_iters, .. }) => num_iters.is_some(),
                Some(LearningRateScheduleConfig::Step { step_size, .. }) => step_size.is_some(),
                Some(LearningRateScheduleConfig::Noam { warmup_steps, .. }) => {
                    warmup_steps.is_some()
                }
            };
            if !schedule_is_horizon_independent {
                return Err(anyhow!(
                    "training.resume_horizon_extension requires an optimizer LR schedule whose timing is explicit and independent of training.max_iters"
                ));
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
        if matches!(self.optimizer.name, OptimizerKind::PredictiveCoding) {
            return Err(anyhow!(
                "optimizer.name=predictive_coding is retired because it used global backpropagation; set training.algorithm=predictive_coding with optimizer.name=adamw for analytic local VJPs"
            ));
        }
        if matches!(self.optimizer.name, OptimizerKind::Eggroll) {
            if self.optimizer.eggroll.gradient_learning_rate.is_some() {
                return Err(anyhow!(
                    "optimizer.name=eggroll does not support optimizer.eggroll.gradient_learning_rate; choose pure EGGROLL or optimizer.name=adamw"
                ));
            }
            if self.optimizer.eggroll.gradient_weight_decay.is_some() {
                return Err(anyhow!(
                    "optimizer.name=eggroll does not support optimizer.eggroll.gradient_weight_decay; choose pure EGGROLL or optimizer.name=adamw"
                ));
            }
            if self.parallel.mode != ParallelismKind::Single {
                return Err(anyhow!(
                    "optimizer.name=eggroll currently requires parallel.mode=single"
                ));
            }
            if self.parallel.pipeline.enabled {
                return Err(anyhow!(
                    "optimizer.name=eggroll does not yet support parallel.pipeline.enabled"
                ));
            }
            if self.training.gradient_accumulation_steps != 1 {
                return Err(anyhow!(
                    "optimizer.name=eggroll currently requires training.gradient_accumulation_steps = 1"
                ));
            }
            if self.training.continual_backprop.enabled {
                return Err(anyhow!(
                    "optimizer.name=eggroll does not yet support training.continual_backprop.enabled"
                ));
            }
            if self.training.neuron_scaling.enabled {
                return Err(anyhow!(
                    "optimizer.name=eggroll does not yet support training.neuron_scaling.enabled"
                ));
            }
            if self.training.predictive_coding.enabled {
                return Err(anyhow!(
                    "optimizer.name=eggroll does not support the historical training.predictive_coding recurrent-state replay auxiliary"
                ));
            }
            if self.training.tbptt_persist_across_steps {
                return Err(anyhow!(
                    "optimizer.name=eggroll does not yet support training.tbptt_persist_across_steps"
                ));
            }
            if !self.training.objective.is_next_token() {
                return Err(anyhow!(
                    "optimizer.name=eggroll currently supports only the next-token training objective"
                ));
            }
            if self.training.ruliad_supervision.uses_target_loss_mask() {
                return Err(anyhow!(
                    "optimizer.name=eggroll does not yet support masked training.ruliad_supervision={:?}; use optimizer.name=adamw or disable ruliad target masks",
                    self.training.ruliad_supervision.mode
                ));
            }
            if self.training.ruliad_supervision.verifier_reward.enabled {
                return Err(anyhow!(
                    "optimizer.name=eggroll does not yet support training.ruliad_supervision.verifier_reward.enabled"
                ));
            }
            if self.training.ruliad_supervision.proof_policy.enabled {
                return Err(anyhow!(
                    "optimizer.name=eggroll does not yet support training.ruliad_supervision.proof_policy.enabled"
                ));
            }
        }
        Ok(())
    }
}
