//! Latent recurrent reasoning, energy heads, and language decoding.

use super::*;

impl<B: Backend> DragonModel<B> {
    pub fn latent_reasoning_config(&self) -> &LatentReasoningConfig {
        &self.latent_reasoning
    }

    pub fn with_fixed_latent_reasoning_steps(mut self, steps: usize) -> Self {
        assert!(steps > 0, "fixed latent reasoning steps must be > 0");
        self.latent_reasoning.enabled = true;
        self.latent_reasoning.max_steps = steps;
        self.latent_reasoning.min_steps = steps;
        self.latent_reasoning.adaptive_halting = false;
        self
    }

    pub fn latent_reasoning_enabled(&self) -> bool {
        self.latent_reasoning.enabled
            && self.latent_reasoning.max_steps > 0
            && self.latent_refiner_in.is_some()
            && self.latent_refiner_out.is_some()
            && (!self.latent_reasoning.adaptive_halting || self.latent_stop_head.is_some())
    }

    pub(super) fn latent_refine_step(
        &self,
        current: Tensor<B, 3>,
        refiner_in: &Linear<B>,
        refiner_out: &Linear<B>,
    ) -> Tensor<B, 3> {
        let update = refiner_out.forward(activation::gelu(refiner_in.forward(current.clone())));
        let update = if let Some(gate) = self.latent_refiner_gate.as_ref() {
            let [batch, time, dim] = update.shape().dims();
            let [gate_dim] = gate.val().shape().dims();
            if gate_dim == dim {
                let gate = activation::sigmoid(gate.val())
                    .reshape([1, 1, dim])
                    .repeat_dim(0, batch)
                    .repeat_dim(1, time);
                update * gate
            } else {
                update
            }
        } else {
            update
        };
        if self.latent_reasoning.normalize_steps {
            self.norm.forward(current + update)
        } else {
            current + update
        }
    }

    pub fn reason_hidden_final(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        if !self.latent_reasoning_enabled() {
            return hidden;
        }

        let refiner_in = self
            .latent_refiner_in
            .as_ref()
            .expect("latent refiner input missing");
        let refiner_out = self
            .latent_refiner_out
            .as_ref()
            .expect("latent refiner output missing");
        if !self.latent_reasoning.adaptive_halting {
            let mut current = hidden;
            for _ in 0..self.latent_reasoning.max_steps {
                current = self.latent_refine_step(current, refiner_in, refiner_out);
            }
            return current;
        }
        let energy_head = self.latent_energy_head.as_ref();
        let stop_head = self
            .latent_stop_head
            .as_ref()
            .expect("latent stop head missing");

        let [batch, time, dim] = hidden.shape().dims();
        let device = hidden.device();
        let mut current = hidden.clone();
        let mut final_hidden = Tensor::<B, 3>::zeros([batch, time, dim], &device);
        let mut remaining = Tensor::<B, 3>::ones([batch, time, 1], &device);
        for step in 0..self.latent_reasoning.max_steps {
            current = self.latent_refine_step(current, refiner_in, refiner_out);
            let _energy = energy_head.map(|head| head.forward(current.clone()));
            let logits = stop_head.forward(current.clone());
            let probs = if step + 1 < self.latent_reasoning.min_steps {
                Tensor::<B, 3>::zeros(logits.shape().dims::<3>(), &logits.device())
            } else {
                activation::sigmoid(logits.clone())
            };
            let halt = probs
                .clone()
                .greater_equal_elem(self.latent_reasoning.halt_threshold)
                .float()
                * remaining.clone();
            final_hidden = final_hidden + current.clone() * halt.clone().repeat_dim(2, dim);
            remaining = remaining * halt.mul_scalar(-1.0).add_scalar(1.0);
        }
        final_hidden + current * remaining.repeat_dim(2, dim)
    }

    pub fn reason_hidden(&self, hidden: Tensor<B, 3>) -> LatentReasoningOutput<B> {
        if !self.latent_reasoning_enabled() {
            return LatentReasoningOutput {
                raw_hidden: hidden.clone(),
                final_hidden: hidden,
                step_hiddens: Vec::new(),
                energies: Vec::new(),
                stop_logits: Vec::new(),
                stop_probs: Vec::new(),
                steps_used: 0,
            };
        }

        let refiner_in = self
            .latent_refiner_in
            .as_ref()
            .expect("latent refiner input missing");
        let refiner_out = self
            .latent_refiner_out
            .as_ref()
            .expect("latent refiner output missing");
        let energy_head = self.latent_energy_head.as_ref();
        let stop_head = self.latent_stop_head.as_ref();

        let [batch, time, dim] = hidden.shape().dims();
        let device = hidden.device();
        let mut current = hidden.clone();
        let mut final_hidden = Tensor::<B, 3>::zeros([batch, time, dim], &device);
        let mut remaining = Tensor::<B, 3>::ones([batch, time, 1], &device);
        let mut step_hiddens = Vec::with_capacity(self.latent_reasoning.max_steps);
        let mut energies = Vec::with_capacity(self.latent_reasoning.max_steps);
        let mut stop_logits = Vec::with_capacity(self.latent_reasoning.max_steps);
        let mut stop_probs = Vec::with_capacity(self.latent_reasoning.max_steps);
        for step in 0..self.latent_reasoning.max_steps {
            current = self.latent_refine_step(current, refiner_in, refiner_out);
            if let Some(head) = energy_head {
                energies.push(head.forward(current.clone()));
            }
            if let Some(head) = stop_head {
                let logits = head.forward(current.clone());
                let probs = if step + 1 < self.latent_reasoning.min_steps {
                    Tensor::<B, 3>::zeros(logits.shape().dims::<3>(), &logits.device())
                } else {
                    activation::sigmoid(logits.clone())
                };
                let halt = probs
                    .clone()
                    .greater_equal_elem(self.latent_reasoning.halt_threshold)
                    .float()
                    * remaining.clone();
                final_hidden = final_hidden + current.clone() * halt.clone().repeat_dim(2, dim);
                remaining = remaining * halt.mul_scalar(-1.0).add_scalar(1.0);
                stop_logits.push(logits);
                stop_probs.push(probs);
            }
            step_hiddens.push(current.clone());
        }
        if self.latent_reasoning.adaptive_halting {
            final_hidden = final_hidden + current.clone() * remaining.repeat_dim(2, dim);
        } else {
            final_hidden = current.clone();
        }

        LatentReasoningOutput {
            raw_hidden: hidden,
            final_hidden,
            step_hiddens,
            energies,
            stop_logits,
            stop_probs,
            steps_used: self.latent_reasoning.max_steps,
        }
    }

    pub fn latent_jepa_prediction_from_hidden(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        self.latent_jepa_predictor
            .as_ref()
            .map(|predictor| predictor.forward(hidden.clone()))
            .unwrap_or(hidden)
    }

    pub fn next_latent_transition_enabled(&self) -> bool {
        self.next_latent_transition.enabled
            && self.next_latent_transition_in.is_some()
            && self.next_latent_transition_mid.is_some()
            && self.next_latent_transition_out.is_some()
    }

    pub(super) fn normalize_next_latent_transition_input(
        &self,
        input: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        if !self.next_latent_transition.normalize_input {
            return input;
        }
        let [_batch, _time, dim] = input.shape().dims();
        if dim == 0 {
            return input;
        }
        let mean = input.clone().mean_dim(2);
        let centered = input - mean.repeat_dim(2, dim);
        let variance = centered.clone().powf_scalar(2.0).mean_dim(2);
        centered / variance.add_scalar(1.0e-5).sqrt().repeat_dim(2, dim)
    }

    pub fn next_latent_prediction_from_hidden_action(
        &self,
        hidden: Tensor<B, 3>,
        action_embedding: Tensor<B, 3>,
    ) -> Option<Tensor<B, 3>> {
        if !self.next_latent_transition_enabled() {
            return None;
        }
        let transition_in = self.next_latent_transition_in.as_ref()?;
        let transition_mid = self.next_latent_transition_mid.as_ref()?;
        let transition_out = self.next_latent_transition_out.as_ref()?;
        let [batch, time, dim] = hidden.shape().dims();
        if action_embedding.shape().dims() != [batch, time, dim] {
            return None;
        }
        let input = Tensor::cat(vec![hidden.clone(), action_embedding], 2);
        let input = self.normalize_next_latent_transition_input(input);
        let update = transition_out.forward(activation::gelu(
            transition_mid.forward(activation::gelu(transition_in.forward(input))),
        ));
        Some(hidden + update)
    }

    pub fn latent_energy_from_hidden(&self, hidden: Tensor<B, 3>) -> Option<Tensor<B, 3>> {
        self.latent_energy_head
            .as_ref()
            .map(|head| head.forward(hidden))
    }

    pub fn sequence_score_head_enabled(&self) -> bool {
        self.sequence_score_head.is_some()
    }

    /// Score complete sequence representations without projecting through the vocabulary head.
    pub fn sequence_scores_from_hidden(&self, hidden: Tensor<B, 3>) -> Option<Tensor<B, 3>> {
        self.sequence_score_head
            .as_ref()
            .map(|head| head.forward_candidate(hidden))
    }

    /// Score prompt-candidate compatibility in a learned low-rank query-key space.
    ///
    /// A terminal-only linear score can rank candidate surface forms but cannot represent a
    /// changed preference when the candidate set is fixed and only the requested target changes.
    /// Independent prompt and candidate projections form a general low-rank bilinear map. This
    /// keeps prompt conditioning in the score contract without adding task-specific outputs or
    /// changing the Dragon backbone width.
    pub fn sequence_scores_from_hidden_pair(
        &self,
        prompt_hidden: Tensor<B, 3>,
        terminal_hidden: Tensor<B, 3>,
    ) -> Option<Tensor<B, 3>> {
        if prompt_hidden.shape() != terminal_hidden.shape() {
            return None;
        }
        self.sequence_score_head
            .as_ref()
            .map(|head| head.forward_pair(prompt_hidden, terminal_hidden))
    }

    pub(super) fn project_hidden_to_logits(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        self.project_hidden_to_logits_with_parameter_gradients(hidden, true)
    }

    fn project_hidden_to_logits_with_parameter_gradients(
        &self,
        hidden: Tensor<B, 3>,
        parameter_gradients: bool,
    ) -> Tensor<B, 3> {
        assert!(
            self.language_head.uses_flat_token_logits(),
            "flat token logits are not available for the configured NCA factorized language head; use hidden-state loss helpers instead"
        );
        let prof_enabled = logits_projection_profile_enabled();
        let start = prof_enabled.then(Instant::now);
        let [batch, time, dim] = hidden.shape().dims();
        let head = if self.tie_input_output_embeddings {
            self.embed.weight.val().transpose()
        } else {
            self.lm_head
                .as_ref()
                .expect("flat language-model head weights missing")
                .val()
        };
        let head = if parameter_gradients {
            head
        } else {
            head.detach()
        };
        let logits = hidden.reshape([batch * time, dim]).matmul(head).reshape([
            batch,
            time,
            self.vocab_size,
        ]);
        if let Some(start) = start {
            logits_projection_profile_record(start.elapsed().as_nanos());
        }
        logits
    }

    pub(super) fn apply_latent_decoder_step_conditioning(
        &self,
        hidden: Tensor<B, 3>,
        step: usize,
    ) -> Tensor<B, 3> {
        self.apply_latent_decoder_step_conditioning_with_parameter_gradients(hidden, step, true)
    }

    fn apply_latent_decoder_step_conditioning_with_parameter_gradients(
        &self,
        hidden: Tensor<B, 3>,
        step: usize,
        parameter_gradients: bool,
    ) -> Tensor<B, 3> {
        if !self.latent_reasoning.step_conditioned_decoder
            || self.latent_reasoning.step_conditioned_decoder_scale <= f32::EPSILON
        {
            return hidden;
        }
        let Some(embedding) = self.latent_step_decoder_embedding.as_ref() else {
            return hidden;
        };
        let [batch, time, dim] = hidden.shape().dims();
        let [steps, width] = embedding.val().shape().dims();
        if steps == 0 || width != dim {
            return hidden;
        }
        let step = step.min(steps - 1);
        let weight = embedding.val();
        let weight = if parameter_gradients {
            weight
        } else {
            weight.detach()
        };
        let bias = weight
            .slice([step..step + 1, 0..dim])
            .reshape([1, 1, dim])
            .repeat_dim(0, batch)
            .repeat_dim(1, time)
            .mul_scalar(self.latent_reasoning.step_conditioned_decoder_scale);
        hidden + bias
    }

    pub(super) fn project_hidden_to_logits_for_latent_step(
        &self,
        hidden: Tensor<B, 3>,
        step: usize,
    ) -> Tensor<B, 3> {
        let hidden = self.apply_latent_decoder_step_conditioning(hidden, step);
        self.project_hidden_to_logits(hidden)
    }

    pub fn logits_from_hidden(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        self.project_hidden_to_logits_for_latent_step(hidden, self.latent_decoder_step())
    }

    /// Preserve the hidden-input derivative without training the decoder or its
    /// step-conditioning parameters through an auxiliary distillation branch.
    pub fn logits_from_hidden_with_frozen_head(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden = self.apply_latent_decoder_step_conditioning_with_parameter_gradients(
            hidden,
            self.latent_decoder_step(),
            false,
        );
        self.project_hidden_to_logits_with_parameter_gradients(hidden, false)
    }

    pub fn logits_from_hidden_for_latent_step(
        &self,
        hidden: Tensor<B, 3>,
        step: usize,
    ) -> Tensor<B, 3> {
        self.project_hidden_to_logits_for_latent_step(hidden, step)
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

    pub fn forward_hidden_raw(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut state = ModelState::new(self.n_layer);
        self.forward_hidden_raw_with_state(tokens, &mut state)
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

    pub fn forward_hidden_raw_with_state(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
    ) -> Tensor<B, 3> {
        let embedded = self.embed.forward(tokens);
        self.forward_hidden_raw_with_state_from_embedded(embedded, state, None)
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
