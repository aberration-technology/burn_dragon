//! Shared low-rank population forwarding and projection kernels.

use super::*;

impl<B: Backend> DragonModel<B> {
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

    pub fn init_state_stateless(&self) -> ModelState<B> {
        ModelState::new_stateless(self.n_layer)
    }

    pub(super) fn layer_latent_total(&self, layer_idx: usize) -> usize {
        self.layer_latent_totals
            .get(layer_idx)
            .copied()
            .unwrap_or(self.mlp_internal_dim_multiplier * self.n_embd)
    }

    pub(super) fn resolve_linear_attention_rho_state(
        &self,
        layer_state: &LayerState<B>,
        _device: &B::Device,
    ) -> Option<Tensor<B, 4>> {
        layer_state.rho.as_ref().cloned()
    }

    pub(super) fn write_linear_attention_rho_state(
        &self,
        layer_state: &mut LayerState<B>,
        rho: Tensor<B, 4>,
    ) {
        layer_state.rho = layer_state.retain_terminal_sequence_state.then_some(rho);
        layer_state.rho_norm = None;
        layer_state.sequence_aux = None;
    }

    pub(super) fn layer_latent_per_head(&self, layer_idx: usize) -> usize {
        let total = self.layer_latent_total(layer_idx);
        assert_eq!(
            total % self.n_head,
            0,
            "layer latent total must divide evenly across heads"
        );
        total / self.n_head
    }

    pub(super) fn layer_lowrank_weights_from_tensors(
        &self,
        layer_idx: usize,
        weights: SharedLowrankWeights<B>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 2>, usize) {
        let latent_per_head = self.layer_latent_per_head(layer_idx);
        let capacity_per_head = self.latent_per_head_capacity();
        let encoder = weights
            .encoder
            .slice([0..self.n_head, 0..self.n_embd, 0..latent_per_head])
            .reshape([1, self.n_head, self.n_embd, latent_per_head]);
        let encoder_v = weights
            .encoder_v
            .slice([0..self.n_head, 0..self.n_embd, 0..latent_per_head])
            .reshape([1, self.n_head, self.n_embd, latent_per_head]);
        let decoder_capacity = weights.decoder;
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

    pub(super) fn layer_lowrank_weights(
        &self,
        layer_idx: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 2>, usize) {
        self.layer_lowrank_weights_from_tensors(layer_idx, self.shared_lowrank_effective_weights())
    }

    pub(super) fn layer_lowrank_weights_for_hierarchical_branch(
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
            let scaffold = SharedLowrankWeights {
                encoder: self
                    .slow_encoder
                    .as_ref()
                    .expect("split hierarchical slow encoder missing")
                    .val(),
                encoder_v: self
                    .slow_encoder_v
                    .as_ref()
                    .expect("split hierarchical slow encoder_v missing")
                    .val(),
                decoder: self
                    .slow_decoder
                    .as_ref()
                    .expect("split hierarchical slow decoder missing")
                    .val(),
            };
            let weights = self
                .random_scaffold_adapters
                .as_ref()
                .map(|adapters| adapters.effective_slow(scaffold.clone()))
                .unwrap_or(scaffold);
            return self.layer_lowrank_weights_from_tensors(layer_idx, weights);
        }
        self.layer_lowrank_weights(layer_idx)
    }

    pub(super) fn hierarchical_dragon_applies_to_layer(&self, layer_idx: usize) -> bool {
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

    pub(super) fn hierarchical_slow_sequence_slot(&self) -> HierarchicalSequenceSlot {
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
    pub(super) fn recurrent_attention_with_plan_in_hierarchical_slot(
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

    pub(super) fn hierarchical_slow_hidden(
        &self,
        layer_state: &LayerState<B>,
        flat_batch: usize,
        branch_dim: usize,
    ) -> Option<Tensor<B, 4>> {
        let hidden = layer_state.hierarchical_slow_hidden.as_ref()?;
        (hidden.shape().dims::<4>() == [flat_batch, 1, 1, branch_dim]).then(|| hidden.clone())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_hierarchical_lowrank_step(
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

    pub(super) fn forward_hierarchical_branch_layer(
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

    pub(super) fn assert_shared_lowrank_population_shapes(
        &self,
        lowrank: &SharedLowrankPopulationWeights<B>,
    ) {
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

    pub(super) fn assert_shared_lowrank_population_factor_shapes(
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

    pub(super) fn population_layer_lowrank_weights(
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

    pub(super) fn population_layer_lowrank_factors(
        &self,
        layer_idx: usize,
        factors: &SharedLowrankPopulationFactors<B>,
    ) -> PopulationLayerLowrankFactors<B> {
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
        PopulationLayerLowrankFactors {
            encoder_a,
            encoder_b,
            encoder_v_a,
            encoder_v_b,
            decoder_a,
            decoder_b: factors.decoder_b.clone(),
            signs: factors.signs.clone(),
            latent_per_head,
        }
    }

    pub(super) fn project_shared_lowrank_population_positive(
        &self,
        request: SharedLowrankPopulationProjection<'_, B>,
    ) -> Tensor<B, 4>
    where
        B::FloatTensorPrimitive: 'static,
    {
        let SharedLowrankPopulationProjection {
            dense,
            projector,
            population,
            relu_threshold,
            use_fused,
            latent_pattern,
            sparse_mask,
        } = request;
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

    pub(super) fn project_shared_lowrank_population_factorized_positive(
        &self,
        request: FactorizedPopulationProjection<'_, B>,
    ) -> Tensor<B, 4> {
        let FactorizedPopulationProjection {
            dense,
            base_projector,
            factor_a,
            factor_b,
            signs,
            sigma_scale,
            population,
            relu_threshold,
            latent_pattern,
            sparse_mask,
        } = request;
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

    pub(super) fn decode_shared_lowrank_population_tail(
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

        let mixed = y_neuron
            .reshape([population, per_population_batch, heads, time, latent])
            .swap_dims(2, 3)
            .reshape([population, per_population_batch * time, heads * latent]);
        mixed
            .matmul(decoder)
            .reshape([population, per_population_batch, time, self.n_embd])
            .reshape([flat_batch, 1, time, self.n_embd])
    }

    pub(super) fn decode_shared_lowrank_population_factors_tail(
        &self,
        request: FactorizedPopulationDecode<B>,
    ) -> Tensor<B, 4> {
        let FactorizedPopulationDecode {
            y_neuron,
            base_decoder,
            factor_a,
            factor_b,
            signs,
            sigma_scale,
            population,
        } = request;
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

    pub(super) fn project_lowrank_positive(
        &self,
        request: LowrankProjectionRequest<'_, B>,
    ) -> Tensor<B, 4>
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
}
