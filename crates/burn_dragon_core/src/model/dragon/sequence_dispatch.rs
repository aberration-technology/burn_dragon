use super::*;

impl<B: Backend> DragonModel<B> {
    pub(super) fn rollout_executor_mode(&self) -> RolloutExecutorMode {
        if self.sequence_kernel.memory_system == SequenceMemorySystem::LinearAttention
            && self.sequence_kernel.executor == SequenceTrainingExecutor::Reference
            && self.kernel.enabled
            && self.kernel.wgpu_recurrent_kernel
            && self.kernel.wgpu_rollout_fused
            && supports_recurrent_backend::<B>()
        {
            return RolloutExecutorMode::WgpuFused;
        }
        RolloutExecutorMode::HostLoop
    }

    pub(super) fn recurrent_attention_reference(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        rho_state: Option<Tensor<B, 4>>,
        decay: Option<Tensor<B, 1>>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        recurrent_attention_reference(query, value, rho_state, decay)
    }

    pub(super) fn recurrent_attention_dense_score_reference(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        rho_state: Option<Tensor<B, 4>>,
        decay: Option<Tensor<B, 1>>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        recurrent_attention_dense_score_reference(query, value, rho_state, decay)
    }

    pub(super) fn recurrent_attention_dense_score_final_rho_reference(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        rho_state: Option<Tensor<B, 4>>,
        decay: Option<Tensor<B, 1>>,
    ) -> Tensor<B, 4> {
        recurrent_attention_dense_score_final_rho_reference(query, value, rho_state, decay)
    }

    pub(super) fn recurrent_attention_dense_score_initial_context_reference(
        &self,
        query: Tensor<B, 4>,
        rho_state: Option<Tensor<B, 4>>,
        decay: Option<Tensor<B, 1>>,
        n_embd: usize,
    ) -> Tensor<B, 4> {
        recurrent_attention_dense_score_initial_context_reference(query, rho_state, decay, n_embd)
    }

    pub(super) fn recurrent_attention_with_plan(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        layer_state: &mut LayerState<B>,
        position: usize,
        position_mode: RecurrentPositionMode,
        fused_plan: Option<&CompiledRecurrentAttentionPlan<B>>,
    ) -> Tensor<B, 4> {
        match (
            self.sequence_kernel.memory_system,
            self.sequence_kernel.executor,
        ) {
            (SequenceMemorySystem::LinearAttention, SequenceTrainingExecutor::Reference) => {
                let query = match position_mode {
                    RecurrentPositionMode::Sequential => {
                        self.attention.rotate_positions(query, position)
                    }
                    RecurrentPositionMode::Fixed => {
                        self.attention.rotate_positions_fixed(query, position)
                    }
                };
                let decay = self.attention.alibi_decay();
                let device = query.device();
                let initial_rho = self.resolve_linear_attention_rho_state(layer_state, &device);

                if self.kernel.enabled && self.kernel.wgpu_recurrent_kernel {
                    let fused = if let Some(plan) = fused_plan {
                        try_fused_recurrent_attention_wgpu_with_plan(
                            &query,
                            &value,
                            initial_rho.as_ref(),
                            decay.as_ref(),
                            plan,
                        )
                    } else {
                        try_fused_recurrent_attention_wgpu(
                            &query,
                            &value,
                            initial_rho.as_ref(),
                            decay.as_ref(),
                        )
                    };
                    if let Some(output) = fused {
                        self.write_linear_attention_rho_state(layer_state, output.rho);
                        return output.context;
                    }
                }

                let (context, rho) =
                    self.recurrent_attention_reference(query, value, initial_rho, decay);
                self.write_linear_attention_rho_state(layer_state, rho);
                context
            }
            (
                SequenceMemorySystem::LinearAttention,
                SequenceTrainingExecutor::DenseScoreShortContext,
            ) => {
                let query = match position_mode {
                    RecurrentPositionMode::Sequential => {
                        self.attention.rotate_positions(query, position)
                    }
                    RecurrentPositionMode::Fixed => {
                        self.attention.rotate_positions_fixed(query, position)
                    }
                };
                let decay = self.attention.alibi_decay();
                let device = query.device();
                let initial_rho = self.resolve_linear_attention_rho_state(layer_state, &device);
                if self.kernel.enabled
                    && self.kernel.wgpu_rollout_fused
                    && supports_dense_causal_attention_backend::<B>()
                {
                    let decay_tensor = decay
                        .clone()
                        .unwrap_or_else(|| Tensor::<B, 1>::ones([self.n_head], &device));
                    if let Some(fused_context) =
                        try_fused_dense_causal_attention_wgpu(&query, &value, &decay_tensor)
                    {
                        let initial_context = self
                            .recurrent_attention_dense_score_initial_context_reference(
                                query.clone(),
                                initial_rho.clone(),
                                decay.clone(),
                                value.shape().dims::<4>()[3],
                            );
                        let rho = self.recurrent_attention_dense_score_final_rho_reference(
                            query.clone(),
                            value.clone(),
                            initial_rho.clone(),
                            decay.clone(),
                        );
                        self.write_linear_attention_rho_state(layer_state, rho);
                        return initial_context + fused_context;
                    }
                }
                let (context, rho) = self.recurrent_attention_dense_score_reference(
                    query,
                    value,
                    initial_rho,
                    decay,
                );
                self.write_linear_attention_rho_state(layer_state, rho);
                context
            }
            (
                SequenceMemorySystem::Mamba3StateSpaceDuality,
                SequenceTrainingExecutor::Reference,
            ) => {
                let params = self
                    .mamba
                    .as_ref()
                    .expect("mamba3 sequence family requires initialized mamba params");
                let [batch, views, _time, dim] = value.shape().dims::<4>();
                assert_eq!(views, 1, "Mamba3 expects a single dense stream view");
                assert_eq!(
                    dim, self.n_embd,
                    "Mamba3 dense stream dim {} must match model dim {}",
                    dim, self.n_embd
                );
                let config = self.mamba_config;
                let device = value.device();
                let initial_state = mamba3_state(
                    layer_state,
                    batch,
                    config.nheads,
                    config.headdim,
                    config.d_state,
                    config.num_rope_angles,
                    &device,
                );
                if self.kernel.enabled
                    && config.use_fast_path
                    && use_tensorized_mamba3_forward_experimental()
                {
                    let params = params.mamba3();
                    let output = tensorized_mamba3_forward(
                        value,
                        config.d_inner,
                        config.d_state,
                        config.headdim,
                        config.ngroups,
                        config.num_rope_angles,
                        config.norm_eps,
                        config.a_floor,
                        config.chunk_size,
                        params.in_proj_tensor(),
                        params.dt_bias_tensor(),
                        params.b_bias_tensor(),
                        params.c_bias_tensor(),
                        params.b_norm_weight_tensor(),
                        params.c_norm_weight_tensor(),
                        params.d_skip_tensor(),
                        params.out_proj_tensor(),
                        Some(Mamba3TensorizedState {
                            ssm: initial_state.ssm,
                            angle: initial_state.angle,
                            k: initial_state.k,
                            v: initial_state.v,
                        }),
                    );
                    write_mamba3_state(
                        layer_state,
                        output.state.ssm,
                        output.state.angle,
                        output.state.k,
                        output.state.v,
                    );
                    return output.context;
                }
                let (context, next_state) = mamba_reference(
                    value,
                    params,
                    Some(MambaReferenceState {
                        ssm: initial_state.ssm,
                        angle: Some(initial_state.angle),
                        k: Some(initial_state.k),
                        v: Some(initial_state.v),
                    }),
                );
                write_mamba3_state(
                    layer_state,
                    next_state.ssm,
                    next_state.angle.expect("mamba3 next angle state"),
                    next_state.k.expect("mamba3 next k state"),
                    next_state.v.expect("mamba3 next v state"),
                );
                context
            }
            (
                SequenceMemorySystem::GatedDeltaNet2,
                SequenceTrainingExecutor::Reference | SequenceTrainingExecutor::GatedDeltaChunkWy,
            ) => {
                let [batch, value_views, time, dense_dim] = value.shape().dims::<4>();
                assert_eq!(
                    value_views, 1,
                    "GatedDeltaNet2 expects one dense value view"
                );
                assert_eq!(
                    dense_dim, self.n_embd,
                    "GatedDeltaNet2 dense stream dim {} must match model dim {}",
                    dense_dim, self.n_embd
                );
                if self.gated_deltanet2_config.implementation
                    == GatedDeltaNet2Implementation::UpstreamFull
                {
                    let block = self.gated_deltanet2_upstream.as_ref().expect(
                        "upstream gated_deltanet2 sequence family requires initialized block",
                    );
                    let mut upstream_state = layer_state.rho.take();
                    let context = block.forward(
                        value.reshape([batch, time, dense_dim]),
                        &mut upstream_state,
                        true,
                    );
                    if let Some(state) = upstream_state {
                        write_gated_deltanet2_state(layer_state, state);
                    }
                    return context.reshape([batch, 1, time, dense_dim]);
                }
                let params = self
                    .gated_deltanet2
                    .as_ref()
                    .expect("gated_deltanet2 sequence family requires initialized GD2 params");
                let [query_batch, query_heads, _query_time, latent] = query.shape().dims::<4>();
                assert_eq!(
                    query_batch, batch,
                    "GatedDeltaNet2 query/value batch mismatch"
                );
                assert_eq!(
                    query_heads, self.n_head,
                    "GatedDeltaNet2 query heads {} must match model heads {}",
                    query_heads, self.n_head
                );
                let projected =
                    params.project_inputs(value.clone(), latent, self.gated_deltanet2_config);
                let mut query = match position_mode {
                    RecurrentPositionMode::Sequential => {
                        self.attention.rotate_positions(query, position)
                    }
                    RecurrentPositionMode::Fixed => {
                        self.attention.rotate_positions_fixed(query, position)
                    }
                };
                let mut key = match position_mode {
                    RecurrentPositionMode::Sequential => {
                        self.attention.rotate_positions(projected.key, position)
                    }
                    RecurrentPositionMode::Fixed => self
                        .attention
                        .rotate_positions_fixed(projected.key, position),
                };
                let device = value.device();
                let initial_state = gated_deltanet2_state(
                    layer_state,
                    batch,
                    self.n_head,
                    latent,
                    dense_dim,
                    &device,
                );
                let value = expand_attention_values_to_heads(value, self.n_head);
                if self.gated_deltanet2_config.qk_l2_norm {
                    query = l2_normalize_last(query, self.gated_deltanet2_config.state_epsilon);
                    key = l2_normalize_last(key, self.gated_deltanet2_config.state_epsilon);
                }
                if matches!(
                    self.sequence_kernel.executor,
                    SequenceTrainingExecutor::GatedDeltaChunkWy
                ) {
                    if let Some(output) = try_gdn2_chunk_wy(
                        query.clone(),
                        key.clone(),
                        value.clone(),
                        projected.erase.clone(),
                        projected.write.clone(),
                        projected.log_decay.clone(),
                        initial_state.clone(),
                        self.gated_deltanet2_config.chunk_size,
                    ) {
                        write_gated_deltanet2_state(layer_state, output.state);
                        return output.context;
                    }
                    static GDN2_CHUNK_WY_FALLBACK_WARN: Once = Once::new();
                    GDN2_CHUNK_WY_FALLBACK_WARN.call_once(|| {
                        eprintln!(
                            "notice: gated_deltanet2 chunk-WY custom backward unavailable or not needed for this call; using the direct recurrence"
                        );
                    });
                }
                let (context, next_state) = gated_deltanet2_reference(
                    query,
                    key,
                    value,
                    projected.erase,
                    projected.write,
                    projected.log_decay,
                    Some(initial_state),
                    false,
                    self.gated_deltanet2_config.state_epsilon,
                );
                write_gated_deltanet2_state(layer_state, next_state);
                context
            }
            (family, executor) => panic!(
                "sequence kernel family {:?} with executor {:?} is not implemented in DragonModel yet",
                family, executor
            ),
        }
    }
}
