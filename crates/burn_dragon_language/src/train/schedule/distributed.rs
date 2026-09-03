//! Dynamic width scaling and distributed execution helpers.

use super::*;

pub(super) struct DynamicNeuronScaleState<'a, B: AutodiffBackend> {
    pub(super) model: &'a mut LanguageTrainModel<B>,
    pub(super) optimizer: &'a mut crate::train::continual_backprop::LanguageOptimizer<B>,
    pub(super) model_config: &'a mut DragonConfig,
    pub(super) scale_generation: &'a mut usize,
    pub(super) batch_size: usize,
    pub(super) gradient_accumulation_steps: usize,
}

pub(super) fn apply_dynamic_neuron_scale<B>(
    env: &TrainEnvironment<'_, B>,
    state: DynamicNeuronScaleState<'_, B>,
    request: ModelScaleRequest,
    event: TrainingEventContext<'_>,
) -> Result<Option<(usize, usize)>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let DynamicNeuronScaleState {
        model,
        optimizer,
        model_config: current_model_config,
        scale_generation,
        batch_size,
        gradient_accumulation_steps,
    } = state;
    let TrainingEventContext {
        epoch,
        absolute_step,
        bus,
    } = event;
    let current_latent_total = model.model.latent_total_capacity();
    let skip = |reason: String, bus: &TrainingEventBus| {
        let _ = bus.send_model_scale_skipped(ModelScaleSkipped {
            run_id: env.run_name.to_string().into(),
            epoch: Some(epoch),
            absolute_step: Some(absolute_step),
            from_capacity_units: current_latent_total,
            requested_capacity_units: Some(request.to_capacity_units),
            reason,
        });
    };

    if request.run_id.as_str() != env.run_name {
        skip(
            format!(
                "scale request run_id {} does not match active run {}",
                request.run_id, env.run_name
            ),
            bus,
        );
        return Ok(None);
    }
    if request.from_capacity_units != current_latent_total {
        skip(
            format!(
                "scale request source capacity {} does not match active latent_total {}",
                request.from_capacity_units, current_latent_total
            ),
            bus,
        );
        return Ok(None);
    }
    if request.to_capacity_units <= current_latent_total {
        skip(
            format!(
                "scale request target {} must exceed current latent_total {}",
                request.to_capacity_units, current_latent_total
            ),
            bus,
        );
        return Ok(None);
    }
    if request.to_capacity_units > env.training.neuron_scaling.max_latent_total {
        skip(
            format!(
                "scale request target {} exceeds configured max_latent_total {}",
                request.to_capacity_units, env.training.neuron_scaling.max_latent_total
            ),
            bus,
        );
        return Ok(None);
    }
    if !request
        .to_capacity_units
        .is_multiple_of(current_model_config.n_embd)
    {
        skip(
            format!(
                "scale request target {} is not divisible by n_embd {}",
                request.to_capacity_units, current_model_config.n_embd
            ),
            bus,
        );
        return Ok(None);
    }
    if !request
        .to_capacity_units
        .is_multiple_of(current_model_config.n_head)
    {
        skip(
            format!(
                "scale request target {} is not divisible by n_head {}",
                request.to_capacity_units, current_model_config.n_head
            ),
            bus,
        );
        return Ok(None);
    }
    if !request
        .to_capacity_units
        .is_multiple_of(env.parallel_config.tensor.size)
    {
        skip(
            format!(
                "scale request target {} is not divisible by tensor parallel size {}",
                request.to_capacity_units, env.parallel_config.tensor.size
            ),
            bus,
        );
        return Ok(None);
    }

    let mut target_config = current_model_config.clone();
    target_config.mlp_internal_dim_multiplier = request.to_capacity_units / target_config.n_embd;
    let (widened, report) = model
        .model
        .widen_latent_total(target_config.clone(), env.device)
        .map_err(|message| anyhow!("failed to widen Dragon latent_total in-process: {message}"))?;
    model.model = widened;
    *model = model
        .clone()
        .with_continual_backprop(&env.training.continual_backprop)
        .with_gradient_scale_schedule(
            env.training,
            env.epochs
                .saturating_mul(env.train_loader.num_items().max(1)),
        )
        .with_neuron_scale_stabilization(
            report.old_latent_total,
            report.new_latent_total,
            &env.training.neuron_scaling.stabilization,
        );
    optimizer.prepare_after_neuron_scale(model);
    optimizer.refresh_continual_backprop_fresh_model(DragonModel::<B>::new(
        target_config.clone(),
        env.device,
    ));
    *current_model_config = target_config;
    *scale_generation = scale_generation.saturating_add(1);

    let _ = bus.send_model_scale_applied(ModelScaleApplied {
        run_id: env.run_name.to_string().into(),
        epoch: Some(epoch),
        absolute_step: Some(absolute_step),
        from_capacity_units: report.old_latent_total,
        to_capacity_units: report.new_latent_total,
        scale_generation: *scale_generation,
        batch_size: Some(batch_size),
        gradient_accumulation_steps: Some(gradient_accumulation_steps),
        message: format!(
            "applied Dragon neuron scaling {} -> {} in-process; optimizer_state_policy=drop_widened_param_moments; stabilization_freeze_base_steps={} stabilization_unfreeze_ramp_steps={}",
            report.old_latent_total,
            report.new_latent_total,
            env.training.neuron_scaling.stabilization.freeze_base_steps,
            env.training.neuron_scaling.stabilization.unfreeze_ramp_steps,
        ),
    });
    info!(
        "applied in-process Dragon neuron scaling {} -> {} at epoch={} absolute_step={}",
        report.old_latent_total, report.new_latent_total, epoch, absolute_step
    );
    Ok(Some((report.old_latent_total, report.new_latent_total)))
}

#[cfg(feature = "ddp")]
pub(super) struct CollectiveSessionGuard<B: BackendTrait> {
    pub(super) peer_id: PeerId,
    pub(super) _marker: PhantomData<B>,
}

#[cfg(feature = "ddp")]
impl<B: BackendTrait> CollectiveSessionGuard<B> {
    fn register(
        peer_id: PeerId,
        device: B::Device,
        config: burn_collective::CollectiveConfig,
    ) -> Result<Self> {
        info!("registering DDP collective session for peer_id={peer_id}");
        register::<B>(peer_id, device, config)
            .map_err(|err| anyhow!("failed to register DDP collective session: {err:?}"))?;
        info!("registered DDP collective session for peer_id={peer_id}");
        Ok(Self {
            peer_id,
            _marker: PhantomData,
        })
    }
}

#[cfg(feature = "ddp")]
impl<B: BackendTrait> Drop for CollectiveSessionGuard<B> {
    fn drop(&mut self) {
        let _ = finish_collective::<B>(self.peer_id);
    }
}

#[cfg(feature = "ddp")]
pub(super) fn shard_bounds(
    total_items: usize,
    shard_index: usize,
    shard_count: usize,
) -> Result<(usize, usize)> {
    if shard_count == 0 {
        return Err(anyhow!("cannot shard a dataloader across zero ranks"));
    }
    if shard_index >= shard_count {
        return Err(anyhow!(
            "rank-local dataloader shard {shard_index} is out of range for shard_count={shard_count}"
        ));
    }
    if total_items < shard_count {
        return Err(anyhow!(
            "rank-local dataloader sharding requires at least one step per rank (steps={total_items}, world_size={shard_count})"
        ));
    }

    let base = total_items / shard_count;
    let remainder = total_items % shard_count;
    let start = shard_index * base + shard_index.min(remainder);
    let width = base + usize::from(shard_index < remainder);
    Ok((start, start + width))
}

#[cfg(feature = "ddp")]
pub(super) fn shard_dataloader<B, I>(
    loader: Arc<dyn DataLoader<B, I>>,
    shard_index: usize,
    shard_count: usize,
    label: &str,
) -> Result<Arc<dyn DataLoader<B, I>>>
where
    B: BackendTrait + 'static,
    I: 'static,
{
    if shard_count <= 1 {
        return Ok(loader);
    }

    let total_items = loader.num_items();
    let (start, end) = shard_bounds(total_items, shard_index, shard_count)
        .with_context(|| format!("failed to shard {label} dataloader"))?;
    Ok(loader.slice(start, end))
}

#[cfg(feature = "ddp")]
pub(super) fn mean_scalar_from_tensor<B: BackendTrait>(tensor: Tensor<B, 1>) -> f64 {
    tensor
        .mean()
        .into_data()
        .iter::<f64>()
        .next()
        .unwrap_or(0.0)
}

#[cfg(feature = "ddp")]
pub(super) fn reduce_mean_scalar<B: BackendTrait>(
    peer_id: PeerId,
    tensor: Tensor<B, 1>,
) -> Result<f64> {
    let reduced = all_reduce::<B>(
        peer_id,
        tensor.into_primitive().tensor(),
        ReduceOperation::Mean,
    )
    .map_err(|err| anyhow!("failed to all-reduce scalar metric: {err:?}"))?;
    Ok(mean_scalar_from_tensor(Tensor::<B, 1>::from_primitive(
        TensorPrimitive::Float(reduced),
    )))
}

#[cfg(feature = "ddp")]
pub(super) fn process_group_peer_id(runtime: &ParallelRuntime) -> PeerId {
    runtime.global_rank.into()
}

#[cfg(feature = "ddp")]
pub(super) fn process_group_data_shard(
    runtime: &ParallelRuntime,
    config: &ParallelConfig,
) -> Result<(
    usize,
    usize,
    Option<PipelineRankAssignment>,
    Option<PipelineParallelLayout>,
)> {
    let layout = resolve_pipeline_parallel_layout(runtime, config)?;
    if let Some(layout) = layout {
        let assignment = layout.assignment(runtime.global_rank).clone();
        return Ok((
            assignment.data_parallel_rank,
            layout.data_parallel_size,
            Some(assignment),
            Some(layout),
        ));
    }

    Ok((runtime.global_rank, runtime.world_size, None, None))
}

#[cfg(feature = "ddp")]
pub(super) fn all_reduce_gradients_in_module_order<B, M>(
    module: &M,
    grads: &mut GradientsParams,
    peer_id: PeerId,
    op: ReduceOperation,
) -> Result<()>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    struct GradientAllReduceVisitor<'a, B: AutodiffBackend> {
        grads: &'a mut GradientsParams,
        peer_id: PeerId,
        op: ReduceOperation,
        trace_grads: bool,
        index: usize,
        error: Option<anyhow::Error>,
        _marker: PhantomData<B>,
    }

    impl<B: AutodiffBackend> burn::module::ModuleVisitor<B> for GradientAllReduceVisitor<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            if self.error.is_some() {
                return;
            }

            self.index += 1;
            let grad_index = self.index;

            let grad = match self.grads.remove::<B::InnerBackend, D>(param.id) {
                Some(grad) => grad,
                None => {
                    if self.trace_grads && grad_index <= 12 {
                        info!(
                            "process-group DDP peer_id={} gradient[{grad_index}] missing, zero-filling shape={:?}",
                            self.peer_id,
                            param.val().shape().dims::<D>()
                        );
                    }
                    param.val().inner().zeros_like()
                }
            };

            if self.trace_grads && grad_index <= 12 {
                info!(
                    "process-group DDP peer_id={} gradient[{grad_index}] entering all-reduce shape={:?}",
                    self.peer_id,
                    grad.shape().dims::<D>()
                );
            }

            match all_reduce::<B::InnerBackend>(
                self.peer_id,
                grad.into_primitive().tensor(),
                self.op,
            ) {
                Ok(reduced) => {
                    if self.trace_grads && grad_index <= 12 {
                        info!(
                            "process-group DDP peer_id={} gradient[{grad_index}] completed all-reduce",
                            self.peer_id
                        );
                    }
                    self.grads.register(
                        param.id,
                        Tensor::<B::InnerBackend, D>::from_primitive(TensorPrimitive::Float(
                            reduced,
                        )),
                    )
                }
                Err(err) => {
                    self.error = Some(anyhow!(
                        "failed to all-reduce process-group DDP gradients: {err:?}"
                    ));
                }
            }
        }
    }

    let trace_grads = true;
    let mut visitor = GradientAllReduceVisitor::<B> {
        grads,
        peer_id,
        op,
        trace_grads,
        index: 0,
        error: None,
        _marker: PhantomData,
    };
    module.visit(&mut visitor);

    if let Some(err) = visitor.error {
        return Err(err);
    }

    Ok(())
}

#[cfg(feature = "ddp")]
pub(super) fn scale_gradients_in_module_order<B, M>(
    module: &M,
    grads: &mut GradientsParams,
    scale: f32,
) where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    if (scale - 1.0).abs() <= f32::EPSILON {
        return;
    }

    struct GradientScaleVisitor<'a, B: AutodiffBackend> {
        grads: &'a mut GradientsParams,
        scale: f32,
        _marker: PhantomData<B>,
    }

    impl<B: AutodiffBackend> burn::module::ModuleVisitor<B> for GradientScaleVisitor<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            if let Some(grad) = self.grads.remove::<B::InnerBackend, D>(param.id) {
                self.grads.register(param.id, grad.mul_scalar(self.scale));
            }
        }
    }

    let mut visitor = GradientScaleVisitor::<B> {
        grads,
        scale,
        _marker: PhantomData,
    };
    module.visit(&mut visitor);
}

#[cfg(feature = "ddp")]
pub(super) fn reduce_sum_scalar<B: BackendTrait>(
    peer_id: PeerId,
    tensor: Tensor<B, 1>,
) -> Result<f64> {
    let reduced = all_reduce::<B>(
        peer_id,
        tensor.into_primitive().tensor(),
        ReduceOperation::Sum,
    )
    .map_err(|err| anyhow!("failed to all-reduce scalar sum: {err:?}"))?;
    Ok(mean_scalar_from_tensor(Tensor::<B, 1>::from_primitive(
        TensorPrimitive::Float(reduced),
    )))
}

#[cfg(feature = "ddp")]
pub(super) fn scalar_tensor<B: BackendTrait>(device: &B::Device, value: f32) -> Tensor<B, 1> {
    Tensor::<B, 1>::from_floats([value], device)
}

#[cfg(feature = "ddp")]
pub(super) fn scalar_flag<B: BackendTrait>(device: &B::Device, enabled: bool) -> Tensor<B, 1> {
    scalar_tensor::<B>(device, if enabled { 1.0 } else { 0.0 })
}

#[cfg(feature = "ddp")]
pub(super) fn broadcast_float_tensor_rooted<B: BackendTrait, const D: usize>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    tensor: Option<Tensor<B, D>>,
) -> Result<Tensor<B, D>> {
    let root_tensor = if global_rank == root_rank {
        Some(
            tensor
                .ok_or_else(|| anyhow!("collective root rank {root_rank} is missing a tensor"))?
                .into_primitive()
                .tensor(),
        )
    } else {
        None
    };
    let broadcasted = broadcast::<B>(peer_id, root_tensor).map_err(|err| {
        anyhow!("failed to broadcast rooted tensor from rank {root_rank}: {err:?}")
    })?;
    Ok(Tensor::<B, D>::from_primitive(TensorPrimitive::Float(
        broadcasted,
    )))
}

#[cfg(feature = "ddp")]
pub(super) fn broadcast_usize_rooted<B: BackendTrait>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    device: &B::Device,
    value: Option<usize>,
) -> Result<usize> {
    let tensor = broadcast_float_tensor_rooted::<B, 1>(
        peer_id,
        global_rank,
        root_rank,
        value.map(|value| {
            let value = value as u64;
            Tensor::<B, 1>::from_floats(
                [
                    (value & 0xffff) as f32,
                    ((value >> 16) & 0xffff) as f32,
                    ((value >> 32) & 0xffff) as f32,
                    ((value >> 48) & 0xffff) as f32,
                ],
                device,
            )
        }),
    )?;
    let chunks = tensor
        .into_data()
        .iter::<f64>()
        .map(|value| value.round().clamp(0.0, 65_535.0) as u64)
        .collect::<Vec<_>>();
    ensure!(
        chunks.len() == 4,
        "rooted usize broadcast produced {} chunks instead of 4",
        chunks.len()
    );
    usize::try_from(chunks[0] | (chunks[1] << 16) | (chunks[2] << 32) | (chunks[3] << 48))
        .context("rooted usize broadcast exceeded the local pointer width")
}

#[cfg(feature = "ddp")]
pub(super) fn broadcast_bool_rooted<B: BackendTrait>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    device: &B::Device,
    value: Option<bool>,
) -> Result<bool> {
    let tensor = broadcast_float_tensor_rooted::<B, 1>(
        peer_id,
        global_rank,
        root_rank,
        value.map(|value| scalar_flag::<B>(device, value)),
    )?;
    Ok(mean_scalar_from_tensor(tensor) >= 0.5)
}

#[cfg(feature = "ddp")]
pub(super) fn broadcast_int_tensor_rooted<B: AutodiffBackend, const D: usize>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    tensor: Option<Tensor<B, D, Int>>,
) -> Result<Tensor<B, D, Int>> {
    let broadcasted = broadcast_float_tensor_rooted::<B::InnerBackend, D>(
        peer_id,
        global_rank,
        root_rank,
        tensor.map(|tensor| tensor.float().inner()),
    )?;
    Ok(Tensor::<B, D>::from_inner(broadcasted).int())
}

#[cfg(feature = "ddp")]
pub(super) fn broadcast_optional_int_tensor_rooted<B: AutodiffBackend, const D: usize>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    device: &B::Device,
    tensor: Option<Tensor<B, D, Int>>,
) -> Result<Option<Tensor<B, D, Int>>> {
    let has_tensor = broadcast_bool_rooted::<B::InnerBackend>(
        peer_id,
        global_rank,
        root_rank,
        device,
        Some(tensor.is_some()),
    )?;
    if !has_tensor {
        return Ok(None);
    }
    broadcast_int_tensor_rooted(peer_id, global_rank, root_rank, tensor).map(Some)
}

#[cfg(feature = "ddp")]
pub(super) fn broadcast_sequence_batch_rooted<B: AutodiffBackend>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    device: &B::Device,
    batch: Option<SequenceBatch<B>>,
) -> Result<SequenceBatch<B>> {
    let inputs = broadcast_int_tensor_rooted(
        peer_id,
        global_rank,
        root_rank,
        batch.as_ref().map(|batch| batch.inputs.clone()),
    )?;
    let targets = broadcast_int_tensor_rooted(
        peer_id,
        global_rank,
        root_rank,
        batch.as_ref().map(|batch| batch.targets.clone()),
    )?;
    let summary_event_mask = broadcast_optional_int_tensor_rooted(
        peer_id,
        global_rank,
        root_rank,
        device,
        batch
            .as_ref()
            .and_then(|batch| batch.summary_event_mask.clone()),
    )?;
    let loss_mask = broadcast_optional_int_tensor_rooted(
        peer_id,
        global_rank,
        root_rank,
        device,
        batch.as_ref().and_then(|batch| batch.loss_mask.clone()),
    )?;
    let reset_stream_state = broadcast_bool_rooted::<B::InnerBackend>(
        peer_id,
        global_rank,
        root_rank,
        device,
        Some(batch.as_ref().is_some_and(|batch| batch.reset_stream_state)),
    )?;
    let has_absolute_step = broadcast_bool_rooted::<B::InnerBackend>(
        peer_id,
        global_rank,
        root_rank,
        device,
        Some(
            batch
                .as_ref()
                .is_some_and(|batch| batch.absolute_step.is_some()),
        ),
    )?;
    let absolute_step = if has_absolute_step {
        Some(broadcast_usize_rooted::<B::InnerBackend>(
            peer_id,
            global_rank,
            root_rank,
            device,
            batch.as_ref().and_then(|batch| batch.absolute_step),
        )?)
    } else {
        None
    };

    Ok(SequenceBatch {
        inputs,
        targets,
        loss_mask,
        summary_event_mask,
        ruliad_policy_batch: None,
        absolute_step,
        reset_stream_state,
    })
}

#[cfg(feature = "ddp")]
pub(super) fn detach_pipeline_state_to_inner<B: AutodiffBackend>(
    state: &LanguagePipelineState<B>,
) -> LanguagePipelineState<B::InnerBackend> {
    LanguagePipelineState::from_parts(
        state.current().clone().detach().inner(),
        state
            .residual_history()
            .iter()
            .cloned()
            .map(|tensor| tensor.detach().inner())
            .collect(),
    )
}

#[cfg(feature = "ddp")]
pub(super) fn attach_pipeline_state_require_grad<B: AutodiffBackend>(
    state: LanguagePipelineState<B::InnerBackend>,
) -> LanguagePipelineState<B> {
    let (current, residual_history) = state.into_parts();
    LanguagePipelineState::from_parts(
        Tensor::<B, 4>::from_inner(current).require_grad(),
        residual_history
            .into_iter()
            .map(|tensor| Tensor::<B, 4>::from_inner(tensor).require_grad())
            .collect(),
    )
}

#[cfg(feature = "ddp")]
pub(super) fn broadcast_pipeline_state_rooted<B: AutodiffBackend>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    device: &B::Device,
    state: Option<&LanguagePipelineState<B>>,
) -> Result<LanguagePipelineState<B::InnerBackend>> {
    let history_len = broadcast_usize_rooted::<B::InnerBackend>(
        peer_id,
        global_rank,
        root_rank,
        device,
        state.map(|state| state.residual_history().len()),
    )?;
    let current = broadcast_float_tensor_rooted::<B::InnerBackend, 4>(
        peer_id,
        global_rank,
        root_rank,
        state.map(|state| state.current().clone().detach().inner()),
    )?;
    let residual_history = (0..history_len)
        .map(|index| {
            broadcast_float_tensor_rooted::<B::InnerBackend, 4>(
                peer_id,
                global_rank,
                root_rank,
                state.map(|state| state.residual_history()[index].clone().detach().inner()),
            )
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(LanguagePipelineState::from_parts(current, residual_history))
}

#[cfg(feature = "ddp")]
pub(super) fn broadcast_pipeline_state_inner_rooted<B: BackendTrait>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    device: &B::Device,
    state: Option<&LanguagePipelineState<B>>,
) -> Result<LanguagePipelineState<B>> {
    let history_len = broadcast_usize_rooted::<B>(
        peer_id,
        global_rank,
        root_rank,
        device,
        state.map(|state| state.residual_history().len()),
    )?;
    let current = broadcast_float_tensor_rooted::<B, 4>(
        peer_id,
        global_rank,
        root_rank,
        state.map(|state| state.current().clone()),
    )?;
    let residual_history = (0..history_len)
        .map(|index| {
            broadcast_float_tensor_rooted::<B, 4>(
                peer_id,
                global_rank,
                root_rank,
                state.map(|state| state.residual_history()[index].clone()),
            )
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(LanguagePipelineState::from_parts(current, residual_history))
}

#[cfg(feature = "ddp")]
pub(super) fn pipeline_surrogate_loss<B: AutodiffBackend>(
    output_state: &LanguagePipelineState<B>,
    grad_state: LanguagePipelineState<B::InnerBackend>,
) -> Tensor<B, 1> {
    let (grad_current, grad_history) = grad_state.into_parts();
    assert_eq!(
        output_state.residual_history().len(),
        grad_history.len(),
        "pipeline residual history length mismatch"
    );

    let mut surrogate = output_state
        .current()
        .clone()
        .mul(Tensor::<B, 4>::from_inner(grad_current))
        .sum();
    for (residual, grad) in output_state
        .residual_history()
        .iter()
        .zip(grad_history.into_iter())
    {
        surrogate = surrogate + residual.clone().mul(Tensor::<B, 4>::from_inner(grad)).sum();
    }
    surrogate
}

#[cfg(feature = "ddp")]
pub(super) fn pipeline_input_grad_state<B: AutodiffBackend>(
    input_state: &LanguagePipelineState<B>,
    grads: &mut B::Gradients,
) -> LanguagePipelineState<B::InnerBackend> {
    LanguagePipelineState::from_parts(
        input_state
            .current()
            .grad_remove(grads)
            .unwrap_or_else(|| input_state.current().clone().inner().zeros_like()),
        input_state
            .residual_history()
            .iter()
            .map(|tensor| {
                tensor
                    .grad_remove(grads)
                    .unwrap_or_else(|| tensor.clone().inner().zeros_like())
            })
            .collect(),
    )
}

#[cfg(feature = "ddp")]
pub(super) fn slice_batch_int<B: BackendTrait>(
    tensor: Tensor<B, 2, Int>,
    range: std::ops::Range<usize>,
) -> Tensor<B, 2, Int> {
    let [_batch, block_size] = tensor.shape().dims();
    tensor.slice([range.start..range.end, 0..block_size])
}

#[cfg(feature = "ddp")]
pub(super) fn pipeline_replica_root_rank(
    layout: &PipelineParallelLayout,
    data_parallel_rank: usize,
) -> usize {
    data_parallel_rank * layout.stage_count
}

#[cfg(feature = "ddp")]
pub(super) fn global_rank_for_virtual_stage(
    plan: &PipelinePlan,
    layout: &PipelineParallelLayout,
    data_parallel_rank: usize,
    virtual_stage_id: usize,
) -> usize {
    let physical_stage_id = plan.assignment(virtual_stage_id).physical_stage_id;
    data_parallel_rank * layout.stage_count + physical_stage_id
}

#[cfg(feature = "ddp")]
pub(super) struct DistributedPipelineForwardCache<B: AutodiffBackend> {
    pub(super) input_state: Option<LanguagePipelineState<B>>,
    pub(super) output_state: Option<LanguagePipelineState<B>>,
    pub(super) loss: Option<Tensor<B, 1>>,
}

#[cfg(feature = "ddp")]
pub(super) fn save_process_group_checkpoint<B, O, S>(
    run_dir: &Path,
    epoch: usize,
    learner: &burn_train::Learner<
        burn_train::LearningComponentsMarker<B, S, LanguageTrainModel<B>, O>,
    >,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    O: Optimizer<LanguageTrainModel<B>, B> + 'static,
    S: LrScheduler + 'static,
{
    let checkpoint_dir = run_dir.join("checkpoint");
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    FileCheckpointer::new(recorder, &checkpoint_dir, "model")
        .save(epoch, learner.model().model.into_record())
        .with_context(|| {
            format!(
                "failed to save process-group model checkpoint {epoch} in {}",
                checkpoint_dir.display()
            )
        })?;
    Ok(())
}

#[cfg(feature = "ddp")]
pub(super) fn load_process_group_checkpoint<B, O, S>(
    run_dir: &Path,
    epoch: usize,
    device: &B::Device,
    mut learner: burn_train::Learner<
        burn_train::LearningComponentsMarker<B, S, LanguageTrainModel<B>, O>,
    >,
) -> Result<burn_train::Learner<burn_train::LearningComponentsMarker<B, S, LanguageTrainModel<B>, O>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    O: Optimizer<LanguageTrainModel<B>, B> + 'static,
    S: LrScheduler + 'static,
{
    let checkpoint_dir = run_dir.join("checkpoint");
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let model_record = FileCheckpointer::new(recorder.clone(), &checkpoint_dir, "model")
        .restore(epoch, device)
        .with_context(|| {
            format!(
                "failed to restore process-group model checkpoint {epoch} from {}",
                checkpoint_dir.display()
            )
        })?;
    learner.load_model(model_record);

    let optim_path = checkpoint_dir.join(format!("optim-{epoch}.bin"));
    if optim_path.is_file() {
        let optim_record = FileCheckpointer::new(recorder.clone(), &checkpoint_dir, "optim")
            .restore(epoch, device)
            .with_context(|| {
                format!(
                    "failed to restore process-group optimizer checkpoint {epoch} from {}",
                    checkpoint_dir.display()
                )
            })?;
        learner.load_optim(optim_record);
    }

    let scheduler_path = checkpoint_dir.join(format!("scheduler-{epoch}.bin"));
    if scheduler_path.is_file() {
        let scheduler_record = FileCheckpointer::new(recorder, &checkpoint_dir, "scheduler")
            .restore(epoch, device)
            .with_context(|| {
                format!(
                    "failed to restore process-group scheduler checkpoint {epoch} from {}",
                    checkpoint_dir.display()
                )
            })?;
        learner.load_scheduler(scheduler_record);
    }

    Ok(learner)
}

#[cfg(feature = "ddp")]
pub(super) fn run_process_group_validation<B, O, S>(
    env: &TrainEnvironment<'_, B>,
    learner: &burn_train::Learner<
        burn_train::LearningComponentsMarker<B, S, LanguageTrainModel<B>, O>,
    >,
) -> Option<f64>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    O: Optimizer<LanguageTrainModel<B>, B> + 'static,
    S: LrScheduler + 'static,
{
    if !env.parallel_runtime.is_primary() {
        return None;
    }

    let model = learner.model().valid();
    let mut iterator = env.valid_loader.iter();
    let mut total = 0.0;
    let mut count = 0usize;

    while let Some(item) = iterator.next() {
        let output = model.step(item);
        let loss_value: LossValue<ValidBackend<B>> = output.adapt();
        total += mean_scalar_from_tensor(loss_value.value());
        count += 1;
    }

    (count > 0).then_some(total / count as f64)
}

#[cfg(feature = "ddp")]
pub(super) struct DistributedPipelineTrainStepResult {
    pub(super) grads: GradientsParams,
    pub(super) mean_train_loss: Option<f64>,
}

#[cfg(feature = "ddp")]
pub(super) fn distributed_pipeline_train_step<B>(
    peer_id: PeerId,
    model: &LanguageTrainModel<B>,
    batch: SequenceBatch<B>,
    layout: &PipelineParallelLayout,
    assignment: &PipelineRankAssignment,
    device: &B::Device,
    reduce_metric_loss: bool,
) -> Result<DistributedPipelineTrainStepResult>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let plan = model
        .pipeline_plan
        .as_ref()
        .ok_or_else(|| anyhow!("distributed pipeline step requires a pipeline plan"))?;
    let [batch_size, _block_size] = batch.inputs.shape().dims();
    let ranges = split_microbatch_ranges(batch_size, plan.microbatches)?;
    let chunk_inputs = ranges
        .iter()
        .cloned()
        .map(|range| slice_batch_int(batch.inputs.clone(), range))
        .collect::<Vec<_>>();
    let chunk_targets = ranges
        .iter()
        .cloned()
        .map(|range| slice_batch_int(batch.targets.clone(), range))
        .collect::<Vec<_>>();
    let chunk_masks = ranges
        .iter()
        .cloned()
        .map(|range| {
            batch
                .summary_event_mask
                .clone()
                .map(|mask| slice_batch_int(mask, range))
        })
        .collect::<Vec<_>>();
    let mut chunk_states = (0..plan.microbatches)
        .map(|_| model.model.init_state())
        .collect::<Vec<ModelState<B>>>();
    let mut forward_cache = HashMap::<(usize, usize), DistributedPipelineForwardCache<B>>::new();
    let mut incoming_forward =
        HashMap::<(usize, usize), LanguagePipelineState<B::InnerBackend>>::new();
    let mut incoming_backward =
        HashMap::<(usize, usize), LanguagePipelineState<B::InnerBackend>>::new();
    let mut local_accumulator = GradientsAccumulator::new();
    let mut local_loss: Option<Tensor<B::InnerBackend, 1>> = None;
    let last_virtual_stage_id = plan.total_virtual_stages.saturating_sub(1);

    for event in &plan.events {
        let microbatch_id = event.microbatch_id;
        let local_forward_output = if event.kind
            == burn_dragon_train::train::pipeline::PipelineEventKind::Forward
            && event.physical_stage_id == assignment.pipeline_stage_id
        {
            let input_state = if event.virtual_stage_id == 0 {
                model
                    .model
                    .begin_language_pipeline(chunk_inputs[microbatch_id].clone())
            } else {
                let input_state = incoming_forward
                    .remove(&(event.virtual_stage_id, microbatch_id))
                    .ok_or_else(|| {
                        anyhow!(
                            "missing forward pipeline state for virtual_stage={} microbatch={microbatch_id}",
                            event.virtual_stage_id
                        )
                    })?;
                attach_pipeline_state_require_grad::<B>(input_state)
            };
            let cached_input = (event.virtual_stage_id > 0).then_some(input_state.clone());
            let output_state = model.model.forward_language_pipeline_stage_with_state(
                input_state,
                &mut chunk_states[microbatch_id],
                plan.assignment(event.virtual_stage_id).layer_range.clone(),
                chunk_masks[microbatch_id].clone(),
            );

            if event.virtual_stage_id == last_virtual_stage_id {
                let hidden = model.model.finish_language_pipeline_hidden_with_state(
                    output_state,
                    &mut chunk_states[microbatch_id],
                );
                let weight = ranges[microbatch_id].len() as f32 / batch_size as f32;
                let loss = model
                    .model
                    .language_loss_from_hidden(hidden, chunk_targets[microbatch_id].clone())
                    .mul_scalar(weight);
                if reduce_metric_loss {
                    local_loss = Some(match local_loss {
                        Some(accumulated) => accumulated + loss.clone().detach().inner(),
                        None => loss.clone().detach().inner(),
                    });
                }
                forward_cache.insert(
                    (event.virtual_stage_id, microbatch_id),
                    DistributedPipelineForwardCache {
                        input_state: cached_input,
                        output_state: None,
                        loss: Some(loss),
                    },
                );
                None
            } else {
                forward_cache.insert(
                    (event.virtual_stage_id, microbatch_id),
                    DistributedPipelineForwardCache {
                        input_state: cached_input,
                        output_state: Some(output_state.clone()),
                        loss: None,
                    },
                );
                Some(output_state)
            }
        } else {
            None
        };

        if event.kind == burn_dragon_train::train::pipeline::PipelineEventKind::Forward
            && event.virtual_stage_id < last_virtual_stage_id
        {
            for replica_id in 0..layout.data_parallel_size {
                let sender_rank =
                    global_rank_for_virtual_stage(plan, layout, replica_id, event.virtual_stage_id);
                let receiver_rank = global_rank_for_virtual_stage(
                    plan,
                    layout,
                    replica_id,
                    event.virtual_stage_id + 1,
                );

                if sender_rank == receiver_rank {
                    if assignment.data_parallel_rank == replica_id
                        && assignment.global_rank == receiver_rank
                    {
                        let forwarded = detach_pipeline_state_to_inner(
                            local_forward_output.as_ref().ok_or_else(|| {
                                anyhow!(
                                    "missing local forward state for virtual_stage={} microbatch={microbatch_id}",
                                    event.virtual_stage_id
                                )
                            })?,
                        );
                        incoming_forward
                            .insert((event.virtual_stage_id + 1, microbatch_id), forwarded);
                    }
                    continue;
                }

                let broadcasted = broadcast_pipeline_state_rooted(
                    peer_id,
                    assignment.global_rank,
                    sender_rank,
                    device,
                    (assignment.data_parallel_rank == replica_id
                        && assignment.global_rank == sender_rank)
                        .then_some(local_forward_output.as_ref())
                        .flatten(),
                )?;
                if assignment.data_parallel_rank == replica_id
                    && assignment.global_rank == receiver_rank
                {
                    incoming_forward
                        .insert((event.virtual_stage_id + 1, microbatch_id), broadcasted);
                }
            }
        }

        let local_backward_grad = if event.kind
            == burn_dragon_train::train::pipeline::PipelineEventKind::Backward
            && event.physical_stage_id == assignment.pipeline_stage_id
        {
            let cached = forward_cache
                .remove(&(event.virtual_stage_id, microbatch_id))
                .ok_or_else(|| {
                    anyhow!(
                        "missing backward cache for virtual_stage={} microbatch={microbatch_id}",
                        event.virtual_stage_id
                    )
                })?;

            let mut grads = if event.virtual_stage_id == last_virtual_stage_id {
                cached
                    .loss
                    .ok_or_else(|| {
                        anyhow!(
                            "missing terminal loss for virtual_stage={} microbatch={microbatch_id}",
                            event.virtual_stage_id
                        )
                    })?
                    .backward()
            } else {
                let output_state = cached.output_state.as_ref().ok_or_else(|| {
                    anyhow!(
                        "missing stage output for virtual_stage={} microbatch={microbatch_id}",
                        event.virtual_stage_id
                    )
                })?;
                let grad_state = incoming_backward
                        .remove(&(event.virtual_stage_id, microbatch_id))
                        .ok_or_else(|| {
                            anyhow!(
                                "missing backward pipeline gradient for virtual_stage={} microbatch={microbatch_id}",
                                event.virtual_stage_id
                            )
                        })?;
                pipeline_surrogate_loss(output_state, grad_state).backward()
            };

            let input_grad = cached
                .input_state
                .as_ref()
                .map(|input_state| pipeline_input_grad_state(input_state, &mut grads));
            local_accumulator.accumulate(model, GradientsParams::from_grads(grads, model));
            input_grad
        } else {
            None
        };

        if event.kind == burn_dragon_train::train::pipeline::PipelineEventKind::Backward
            && event.virtual_stage_id > 0
        {
            for replica_id in 0..layout.data_parallel_size {
                let sender_rank =
                    global_rank_for_virtual_stage(plan, layout, replica_id, event.virtual_stage_id);
                let receiver_rank = global_rank_for_virtual_stage(
                    plan,
                    layout,
                    replica_id,
                    event.virtual_stage_id - 1,
                );

                if sender_rank == receiver_rank {
                    if assignment.data_parallel_rank == replica_id
                        && assignment.global_rank == receiver_rank
                    {
                        let grad_state = local_backward_grad.clone().ok_or_else(|| {
                            anyhow!(
                                "missing local backward gradient for virtual_stage={} microbatch={microbatch_id}",
                                event.virtual_stage_id
                            )
                        })?;
                        incoming_backward
                            .insert((event.virtual_stage_id - 1, microbatch_id), grad_state);
                    }
                    continue;
                }

                let broadcasted = broadcast_pipeline_state_inner_rooted::<B::InnerBackend>(
                    peer_id,
                    assignment.global_rank,
                    sender_rank,
                    device,
                    (assignment.data_parallel_rank == replica_id
                        && assignment.global_rank == sender_rank)
                        .then_some(local_backward_grad.as_ref())
                        .flatten(),
                )?;
                if assignment.data_parallel_rank == replica_id
                    && assignment.global_rank == receiver_rank
                {
                    incoming_backward
                        .insert((event.virtual_stage_id - 1, microbatch_id), broadcasted);
                }
            }
        }
    }

    let mean_train_loss = if reduce_metric_loss {
        let reduced_loss = reduce_sum_scalar::<B::InnerBackend>(
            peer_id,
            if assignment.is_last_stage() {
                local_loss.unwrap_or_else(|| Tensor::<B::InnerBackend, 1>::zeros([1], device))
            } else {
                Tensor::<B::InnerBackend, 1>::zeros([1], device)
            },
        )?;
        Some(reduced_loss / layout.data_parallel_size as f64)
    } else {
        None
    };

    Ok(DistributedPipelineTrainStepResult {
        grads: local_accumulator.grads(),
        mean_train_loss,
    })
}

#[cfg(feature = "ddp")]
pub(super) fn train_with_collective_pipeline_scheduler<B, O, S>(
    env: &TrainEnvironment<'_, B>,
    mut learner: burn_train::Learner<
        burn_train::LearningComponentsMarker<B, S, LanguageTrainModel<B>, O>,
    >,
    local_train_loader: Arc<dyn DataLoader<B, SequenceBatch<B>>>,
    peer_id: PeerId,
    layout: PipelineParallelLayout,
    assignment: PipelineRankAssignment,
) -> Result<DragonModel<ValidBackend<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    O: Optimizer<LanguageTrainModel<B>, B> + 'static,
    S: LrScheduler + 'static,
{
    let global_train_steps = env.train_loader.num_items();
    if global_train_steps % layout.data_parallel_size != 0 {
        return Err(anyhow!(
            "parallel.pipeline.enabled process-group execution requires env.train_loader.num_items() divisible by parallel.data.size so every replica executes the same number of collectives (got {} steps across {} replicas)",
            global_train_steps,
            layout.data_parallel_size
        ));
    }

    let local_train_steps = local_train_loader.num_items();
    let expected_local_train_steps = global_train_steps / layout.data_parallel_size;
    if local_train_steps != expected_local_train_steps {
        return Err(anyhow!(
            "parallel.pipeline.enabled process-group execution expected {} local steps for dp_rank={} but resolved {}",
            expected_local_train_steps,
            assignment.data_parallel_rank,
            local_train_steps
        ));
    }
    let metric_every = env.training.log_frequency.max(1);
    let grad_accumulation = env.training.gradient_accumulation_steps.max(1);
    let logical_replica_count = layout.data_parallel_size;
    let start_epoch = env
        .resume_checkpoint_epoch
        .map(|epoch| epoch + 1)
        .unwrap_or(1);

    for epoch in start_epoch..=env.epochs {
        info!(
            "Executing process-group pipeline epoch {} on global_rank={} stage={} dp_rank={}",
            epoch,
            assignment.global_rank,
            assignment.pipeline_stage_id,
            assignment.data_parallel_rank
        );

        let mut iterator = local_train_loader.iter();
        let mut iteration = 0usize;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0usize;

        while iteration < local_train_steps {
            let mut batch = None;
            for replica_id in 0..layout.data_parallel_size {
                let batch_root_rank = pipeline_replica_root_rank(&layout, replica_id);
                let replica_root_batch = if assignment.data_parallel_rank == replica_id
                    && assignment.global_rank == batch_root_rank
                {
                    iterator.next()
                } else {
                    None
                };
                let replica_batch = broadcast_sequence_batch_rooted(
                    peer_id,
                    assignment.global_rank,
                    batch_root_rank,
                    env.device,
                    replica_root_batch,
                )?;
                if assignment.data_parallel_rank == replica_id {
                    batch = Some(replica_batch);
                }
            }
            let batch = batch.ok_or_else(|| {
                anyhow!(
                    "missing local replica batch for dp_rank={} at iteration={iteration}",
                    assignment.data_parallel_rank
                )
            })?;

            iteration += 1;
            for _ in 0..logical_replica_count {
                learner.lr_step();
            }

            let absolute_step = epoch
                .saturating_sub(1)
                .saturating_mul(local_train_steps)
                .saturating_add(iteration.saturating_sub(1));
            let source_selection_due = source_selection_telemetry_due(env, absolute_step);
            let log_train_metrics = iteration % metric_every == 0 || iteration == local_train_steps;
            let step = distributed_pipeline_train_step(
                peer_id,
                &learner.model(),
                batch,
                &layout,
                &assignment,
                env.device,
                source_selection_due || log_train_metrics,
            )?;
            if source_selection_due
                && let (Some(dataset), Some(mean_train_loss)) = (
                    env.source_selection_dataset
                        .as_ref()
                        .filter(|dataset| dataset.uses_live_source_selection()),
                    step.mean_train_loss,
                )
            {
                let _ = dataset.record_source_selection_loss(absolute_step, mean_train_loss as f32);
            }

            accumulator.accumulate(&learner.model(), step.grads);
            accumulation_current += 1;

            if grad_accumulation <= accumulation_current {
                let mut grads = accumulator.grads();
                all_reduce_gradients_in_module_order(
                    &learner.model(),
                    &mut grads,
                    peer_id,
                    ReduceOperation::Sum,
                )?;
                scale_gradients_in_module_order::<B, _>(
                    &learner.model(),
                    &mut grads,
                    1.0 / layout.data_parallel_size as f32,
                );
                learner.optimizer_step(grads);
                accumulation_current = 0;
            }

            if env.parallel_runtime.is_primary()
                && log_train_metrics
                && let Some(mean_train_loss) = step.mean_train_loss
            {
                let progress = iterator.progress();
                let global_iteration = epoch
                    .saturating_sub(1)
                    .saturating_mul(logical_replica_count.saturating_mul(local_train_steps))
                    .saturating_add(iteration.saturating_mul(logical_replica_count));
                info!(
                    "train epoch={} local_step={}/{} global_iteration={} loss={:.4} lr={:.6} global_progress={}/{}",
                    epoch,
                    progress.items_processed,
                    progress.items_total,
                    global_iteration,
                    mean_train_loss,
                    learner.lr_current(),
                    epoch,
                    env.epochs
                );
            }
        }

        if env.parallel_runtime.is_primary() {
            if let Some(valid_loss) = run_process_group_validation::<B, O, S>(env, &learner) {
                info!("valid epoch={} loss={valid_loss:.4}", epoch);
            }
            save_process_group_checkpoint::<B, O, S>(env.run_dir, epoch, &learner)?;
        }
    }

    Ok(learner.model().valid().model)
}

#[cfg(feature = "ddp")]
pub(super) fn train_with_collective_scheduler<B, O, S>(
    env: &TrainEnvironment<'_, B>,
    model: LanguageTrainModel<B>,
    optimizer: O,
    scheduler: S,
    collective: burn_collective::CollectiveConfig,
    peer_id: PeerId,
) -> Result<DragonModel<ValidBackend<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    O: Optimizer<LanguageTrainModel<B>, B> + 'static,
    S: LrScheduler + 'static,
{
    let _session = CollectiveSessionGuard::<B::InnerBackend>::register(
        peer_id,
        env.device.clone(),
        collective,
    )?;

    let (data_shard_index, data_shard_count, pipeline_assignment, pipeline_layout) =
        process_group_data_shard(env.parallel_runtime, env.parallel_config)?;

    let local_train_loader = shard_dataloader(
        Arc::clone(&env.train_loader),
        data_shard_index,
        data_shard_count,
        "train",
    )?;

    let metric_every = env.training.log_frequency.max(1);
    let grad_accumulation = env.training.gradient_accumulation_steps.max(1);
    let local_train_steps = local_train_loader.num_items();
    let mut learner = burn_train::Learner::new(model, optimizer, scheduler);
    if let Some(checkpoint) = env.resume_checkpoint_epoch {
        learner =
            load_process_group_checkpoint::<B, O, S>(env.run_dir, checkpoint, env.device, learner)?;
    }
    let start_epoch = env
        .resume_checkpoint_epoch
        .map(|epoch| epoch + 1)
        .unwrap_or(1);

    info!(
        "training strategy: mode={:?} replicas={} local_rank={} global_rank={} local_train_steps={} start_epoch={}",
        env.parallel_runtime.mode,
        env.parallel_runtime.world_size,
        env.parallel_runtime.local_rank,
        env.parallel_runtime.global_rank,
        local_train_steps,
        start_epoch
    );
    if let (Some(layout), Some(assignment)) = (&pipeline_layout, &pipeline_assignment) {
        info!(
            "process-group pipeline topology: {} rank={} stage={} dp_rank={} predecessor={:?} successor={:?} pipeline_group={:?} dp_group={:?}",
            layout.summary(),
            assignment.global_rank,
            assignment.pipeline_stage_id,
            assignment.data_parallel_rank,
            assignment.predecessor_global_rank,
            assignment.successor_global_rank,
            assignment.pipeline_group_ranks,
            assignment.data_parallel_group_ranks,
        );
    }

    if let (Some(layout), Some(assignment)) = (pipeline_layout.clone(), pipeline_assignment.clone())
    {
        return train_with_collective_pipeline_scheduler(
            env,
            learner,
            local_train_loader,
            peer_id,
            layout,
            assignment,
        );
    }

    for epoch in start_epoch..=env.epochs {
        info!(
            "Executing process-group DDP epoch {} on global_rank={}",
            epoch, env.parallel_runtime.global_rank
        );

        let mut iterator = local_train_loader.iter();
        let mut iteration = 0usize;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0usize;
        let logical_replica_count = env.parallel_runtime.world_size;
        while let Some(item) = iterator.next() {
            iteration += 1;
            for _ in 0..logical_replica_count {
                learner.lr_step();
            }

            let item = learner.train_step(item);
            let absolute_step = epoch
                .saturating_sub(1)
                .saturating_mul(local_train_steps)
                .saturating_add(iteration.saturating_sub(1));
            let source_selection_due = source_selection_telemetry_due(env, absolute_step);
            let log_train_metrics = iteration % metric_every == 0 || iteration == local_train_steps;
            let mean_train_loss = if source_selection_due || log_train_metrics {
                let train_output = item.item.sync();
                let loss_value: LossValue<ValidBackend<B>> = train_output.adapt();
                Some(reduce_mean_scalar::<ValidBackend<B>>(
                    peer_id,
                    loss_value.value(),
                )?)
            } else {
                None
            };
            if source_selection_due
                && let (Some(dataset), Some(mean_train_loss)) = (
                    env.source_selection_dataset
                        .as_ref()
                        .filter(|dataset| dataset.uses_live_source_selection()),
                    mean_train_loss,
                )
            {
                let _ = dataset.record_source_selection_loss(absolute_step, mean_train_loss as f32);
            }

            accumulator.accumulate(&learner.model(), item.grads);
            accumulation_current += 1;

            if grad_accumulation <= accumulation_current {
                info!(
                    "process-group DDP rank={} iteration={} entering gradient all-reduce",
                    env.parallel_runtime.global_rank, iteration
                );
                let mut grads = accumulator.grads();
                // Fresh multi-process launches instantiate random ParamIds per rank, so
                // cross-rank gradient sync must follow deterministic module traversal order.
                all_reduce_gradients_in_module_order(
                    &learner.model(),
                    &mut grads,
                    peer_id,
                    ReduceOperation::Mean,
                )?;
                info!(
                    "process-group DDP rank={} iteration={} completed gradient all-reduce",
                    env.parallel_runtime.global_rank, iteration
                );
                learner.optimizer_step(grads);
                accumulation_current = 0;
            }

            if env.parallel_runtime.is_primary()
                && log_train_metrics
                && let Some(mean_train_loss) = mean_train_loss
            {
                let progress = iterator.progress();
                let global_iteration = epoch
                    .saturating_sub(1)
                    .saturating_mul(logical_replica_count.saturating_mul(local_train_steps))
                    .saturating_add(iteration.saturating_mul(logical_replica_count));
                info!(
                    "train epoch={} local_step={}/{} global_iteration={} loss={:.4} lr={:.6} global_progress={}/{}",
                    epoch,
                    progress.items_processed,
                    progress.items_total,
                    global_iteration,
                    mean_train_loss,
                    learner.lr_current(),
                    epoch,
                    env.epochs
                );
            }
        }

        if env.parallel_runtime.is_primary() {
            if let Some(valid_loss) = run_process_group_validation::<B, O, S>(env, &learner) {
                info!("valid epoch={} loss={valid_loss:.4}", epoch);
            }
            save_process_group_checkpoint::<B, O, S>(env.run_dir, epoch, &learner)?;
        }
    }

    Ok(learner.model().valid().model)
}

#[cfg(feature = "ddp")]
pub(super) fn train_with_process_group_scheduler<B, O, S>(
    env: &TrainEnvironment<'_, B>,
    model: LanguageTrainModel<B>,
    optimizer: O,
    scheduler: S,
) -> Result<DragonModel<ValidBackend<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    O: Optimizer<LanguageTrainModel<B>, B> + 'static,
    S: LrScheduler + 'static,
{
    let collective = resolve_collective_config(env.parallel_runtime, env.parallel_config)?;
    train_with_collective_scheduler::<B, O, S>(
        env,
        model,
        optimizer,
        scheduler,
        collective,
        process_group_peer_id(env.parallel_runtime),
    )
}

pub fn resolve_lr_scheduler(
    optimizer_cfg: &OptimizerConfig,
    total_steps: usize,
    override_num_iters: Option<usize>,
    model_config: &DragonConfig,
) -> Result<ResolvedLrScheduler> {
    burn_dragon_train::train::pipeline::resolve_lr_scheduler(
        optimizer_cfg,
        total_steps,
        override_num_iters,
        model_config.n_embd,
    )
}

pub fn resolve_train_schedule(
    training: &TrainingHyperparameters,
    steps_per_epoch: usize,
) -> Result<TrainSchedule> {
    burn_dragon_train::train::pipeline::resolve_train_schedule(
        training.epochs,
        training.max_iters,
        steps_per_epoch,
        "training",
    )
}
