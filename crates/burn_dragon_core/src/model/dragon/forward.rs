//! Stateful recurrent execution, hierarchy routing, and sequence-memory updates.

use super::*;

impl<B: Backend> DragonModel<B> {
    pub(super) fn forward_with_state_impl(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let embedded = self.embed.forward(tokens);
        self.forward_with_state_from_embedded(embedded, state, summary_event_mask)
    }

    pub(super) fn forward_hidden_with_state_impl(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        let embedded = self.embed.forward(tokens);
        self.forward_hidden_with_state_from_embedded(embedded, state, summary_event_mask)
    }

    pub(super) fn forward_with_state_from_embedded(
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

    pub(super) fn forward_hidden_raw_with_state_from_embedded(
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

    pub(super) fn forward_hidden_with_state_from_embedded(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        let hidden =
            self.forward_hidden_raw_with_state_from_embedded(embedded, state, summary_event_mask);
        self.reason_hidden_final(hidden)
    }

    pub(super) fn latent_decoder_step(&self) -> usize {
        if self.latent_reasoning_enabled() {
            self.latent_reasoning.max_steps
        } else {
            0
        }
    }

    pub(super) fn forward_with_state_from_embedded_rollout_host_loop(
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

    pub(super) fn forward_hidden_with_state_from_embedded_rollout_host_loop(
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

    pub(super) fn forward_with_state_from_embedded_rollout_fused(
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

    pub(super) fn forward_hidden_with_state_from_embedded_rollout_fused(
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

    pub(super) fn forward_hidden_with_shared_lowrank_population_from_embedded_single_pass(
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
                SharedLowrankPopulationProjection {
                    dense: branch_flat.clone(),
                    projector: encoder,
                    population,
                    relu_threshold: self.x_relu_threshold,
                    use_fused: fused && self.kernel.projection_executor.use_x(),
                    latent_pattern,
                    sparse_mask: sparse_mask.clone(),
                },
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
                SharedLowrankPopulationProjection {
                    dense: attn,
                    projector: encoder_v,
                    population,
                    relu_threshold: self.y_relu_threshold,
                    use_fused: fused && self.kernel.projection_executor.use_y(),
                    latent_pattern,
                    sparse_mask,
                },
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

    pub(super) fn forward_hidden_with_shared_lowrank_population_factors_from_embedded_single_pass(
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
            let PopulationLayerLowrankFactors {
                encoder_a,
                encoder_b,
                encoder_v_a,
                encoder_v_b,
                decoder_a,
                decoder_b,
                signs,
                latent_per_head: factor_latent,
            } = self.population_layer_lowrank_factors(layer_idx, factors);
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
                FactorizedPopulationProjection {
                    dense: branch_flat.clone(),
                    base_projector: base_encoder,
                    factor_a: encoder_a,
                    factor_b: encoder_b,
                    signs: signs.clone(),
                    sigma_scale: factors.sigma as f64 * factors.encoder_scale,
                    population,
                    relu_threshold: self.x_relu_threshold,
                    latent_pattern,
                    sparse_mask: sparse_mask.clone(),
                },
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
                FactorizedPopulationProjection {
                    dense: attn,
                    base_projector: base_encoder_v,
                    factor_a: encoder_v_a,
                    factor_b: encoder_v_b,
                    signs: signs.clone(),
                    sigma_scale: factors.sigma as f64 * factors.encoder_v_scale,
                    population,
                    relu_threshold: self.y_relu_threshold,
                    latent_pattern,
                    sparse_mask,
                },
            );
            let y_neuron = self.dropout.forward(x_neuron * y_gate);
            let mlp_out =
                self.decode_shared_lowrank_population_factors_tail(FactorizedPopulationDecode {
                    y_neuron,
                    base_decoder,
                    factor_a: decoder_a,
                    factor_b: decoder_b,
                    signs,
                    sigma_scale: factors.sigma as f64 * factors.decoder_scale,
                    population,
                });
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

    pub(super) fn forward_hidden_with_state_from_embedded_single_pass_y_neuron_recurrence(
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
        let shared_lowrank_weights = self.shared_lowrank_effective_weights();

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
            let (encoder, encoder_v, decoder, latent) =
                self.layer_lowrank_weights_from_tensors(layer_idx, shared_lowrank_weights.clone());
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
}
