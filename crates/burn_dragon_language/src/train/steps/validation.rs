//! Validation and explicit recurrent stream-state execution.

use super::*;

impl<B: BackendTrait> ValidStep for LanguageTrainModel<B> {
    type Input = SequenceBatch<B>;
    type Output = LanguageModelOutput<B>;

    fn step(&self, batch: SequenceBatch<B>) -> LanguageModelOutput<B> {
        let loss_mask = batch.loss_mask;
        if self.pipeline_enabled() {
            let (loss, _hidden, _logits) = self.forward_loss_with_pipeline(
                batch.inputs,
                batch.targets,
                loss_mask,
                batch.summary_event_mask,
            );
            return LanguageModelOutput::new(loss);
        }
        if let Some(summary_event_mask) = batch.summary_event_mask {
            if let Some(chunk_size) =
                self.effective_tbptt_chunk_size(batch.inputs.shape().dims::<2>()[1])
            {
                let [batch_size, block_size] = batch.inputs.shape().dims();
                let mut state = self.model.init_state();
                let mut loss: Option<Tensor<B, 1>> = None;
                for start in (0..block_size).step_by(chunk_size) {
                    let end = (start + chunk_size).min(block_size);
                    let chunk_inputs =
                        Self::slice_tokens(batch.inputs.clone(), batch_size, start, end);
                    let chunk_targets =
                        Self::slice_tokens(batch.targets.clone(), batch_size, start, end);
                    let chunk_loss_mask = loss_mask
                        .clone()
                        .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
                    let chunk_mask =
                        Self::slice_tokens(summary_event_mask.clone(), batch_size, start, end);
                    let hidden = self.model.forward_hidden_with_state_and_summary_event_mask(
                        chunk_inputs,
                        chunk_mask,
                        &mut state,
                    );
                    let chunk_weight = (end - start) as f32 / block_size as f32;
                    let chunk_loss = self
                        .language_loss_from_hidden(hidden, chunk_targets, chunk_loss_mask)
                        .mul_scalar(chunk_weight);
                    loss = Some(match loss {
                        Some(accumulated) => accumulated + chunk_loss,
                        None => chunk_loss,
                    });
                }
                LanguageModelOutput::new(
                    loss.expect("tbptt valid step should produce at least one loss chunk"),
                )
            } else {
                let mut state = self.model.init_state();
                let hidden = self.model.forward_hidden_with_state_and_summary_event_mask(
                    batch.inputs,
                    summary_event_mask,
                    &mut state,
                );
                let loss = self.language_loss_from_hidden(hidden, batch.targets, loss_mask);
                LanguageModelOutput::new(loss)
            }
        } else if let Some(chunk_size) =
            self.effective_tbptt_chunk_size(batch.inputs.shape().dims::<2>()[1])
        {
            let [batch_size, block_size] = batch.inputs.shape().dims();
            let mut state = self.model.init_state();
            let mut loss: Option<Tensor<B, 1>> = None;
            for start in (0..block_size).step_by(chunk_size) {
                let end = (start + chunk_size).min(block_size);
                let chunk_inputs = Self::slice_tokens(batch.inputs.clone(), batch_size, start, end);
                let chunk_targets =
                    Self::slice_tokens(batch.targets.clone(), batch_size, start, end);
                let chunk_loss_mask = loss_mask
                    .clone()
                    .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
                let hidden = self
                    .model
                    .forward_hidden_with_state(chunk_inputs, &mut state);
                let chunk_weight = (end - start) as f32 / block_size as f32;
                let chunk_loss = self
                    .language_loss_from_hidden(hidden, chunk_targets, chunk_loss_mask)
                    .mul_scalar(chunk_weight);
                loss = Some(match loss {
                    Some(accumulated) => accumulated + chunk_loss,
                    None => chunk_loss,
                });
            }
            LanguageModelOutput::new(
                loss.expect("tbptt valid step should produce at least one loss chunk"),
            )
        } else {
            let hidden = self.model.forward_hidden(batch.inputs);
            let loss = self.language_loss_from_hidden(hidden, batch.targets, loss_mask);
            LanguageModelOutput::new(loss)
        }
    }
}

impl<B: BackendTrait> LanguageTrainModel<B> {
    pub(crate) fn sequence_state_diagnostics(
        state: &ModelState<B>,
        max_rho_slots: usize,
    ) -> Option<SequenceStateDiagnostics> {
        let mut rho_rms: Option<Tensor<B, 1>> = None;
        let mut slot_variance_ratio: Option<Tensor<B, 1>> = None;
        let mut slot_redundancy: Option<Tensor<B, 1>> = None;
        let mut layers = 0usize;

        for rho in state.layers.iter().filter_map(|layer| layer.rho.as_ref()) {
            let [batch, heads, original_slots, dim] = rho.shape().dims::<4>();
            if batch == 0 || heads == 0 || original_slots < 2 || dim == 0 {
                continue;
            }
            let rho = Self::sample_rho_slots_with_limit(
                rho.clone(),
                original_slots,
                max_rho_slots.max(2),
            );
            let [batch, heads, slots, dim] = rho.shape().dims::<4>();
            let groups = batch.saturating_mul(heads);
            let rows = rho.reshape([groups, slots, dim]);
            let layer_energy = rows.clone().powf_scalar(2.0).mean().reshape([1]);
            let layer_rms = layer_energy.clone().clamp_min(1.0e-12).sqrt();

            let slot_mean = rows.clone().mean_dim(1);
            let slot_variance = (rows.clone() - slot_mean.repeat_dim(1, slots))
                .powf_scalar(2.0)
                .mean()
                .reshape([1]);
            let layer_variance_ratio = slot_variance / layer_energy.clamp_min(1.0e-12);

            let row_mean = rows.clone().mean_dim(2);
            let centered = rows - row_mean.repeat_dim(2, dim);
            let row_energy = centered
                .clone()
                .powf_scalar(2.0)
                .sum_dim(2)
                .clamp_min(1.0e-8);
            let normalized = centered / row_energy.clone().sqrt().repeat_dim(2, dim);
            let gram = normalized
                .clone()
                .matmul(normalized.clone().swap_dims(1, 2));
            let total_sq = gram.powf_scalar(2.0).sum().reshape([1]);
            let diag_sq = normalized
                .powf_scalar(2.0)
                .sum_dim(2)
                .powf_scalar(2.0)
                .sum()
                .reshape([1]);
            let off_diagonal = (groups * slots * slots.saturating_sub(1)).max(1) as f32;
            let layer_redundancy = (total_sq - diag_sq)
                .clamp_min(0.0)
                .div_scalar(off_diagonal)
                .sqrt();

            rho_rms = Some(match rho_rms {
                Some(total) => total + layer_rms,
                None => layer_rms,
            });
            slot_variance_ratio = Some(match slot_variance_ratio {
                Some(total) => total + layer_variance_ratio,
                None => layer_variance_ratio,
            });
            slot_redundancy = Some(match slot_redundancy {
                Some(total) => total + layer_redundancy,
                None => layer_redundancy,
            });
            layers = layers.saturating_add(1);
        }

        let scalar = |tensor: Tensor<B, 1>| {
            tensor
                .div_scalar(layers.max(1) as f32)
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("sequence-state diagnostic tensor")[0] as f64
        };
        Some(SequenceStateDiagnostics {
            rho_layers: layers,
            rho_rms: scalar(rho_rms?),
            rho_slot_variance_ratio: scalar(slot_variance_ratio?),
            rho_slot_redundancy: scalar(slot_redundancy?),
        })
    }

    pub(crate) fn step_with_stream_state(
        &self,
        batch: SequenceBatch<B>,
        state: &mut ModelState<B>,
    ) -> LanguageModelOutput<B> {
        if batch.reset_stream_state {
            *state = self.model.init_state();
        }
        if self.pipeline_enabled() {
            return <Self as ValidStep>::step(self, batch);
        }
        let loss_mask = batch.loss_mask;
        if let Some(summary_event_mask) = batch.summary_event_mask {
            if let Some(chunk_size) =
                self.effective_tbptt_chunk_size(batch.inputs.shape().dims::<2>()[1])
            {
                let [batch_size, block_size] = batch.inputs.shape().dims();
                let mut loss: Option<Tensor<B, 1>> = None;
                for start in (0..block_size).step_by(chunk_size) {
                    let end = (start + chunk_size).min(block_size);
                    let chunk_inputs =
                        Self::slice_tokens(batch.inputs.clone(), batch_size, start, end);
                    let chunk_targets =
                        Self::slice_tokens(batch.targets.clone(), batch_size, start, end);
                    let chunk_loss_mask = loss_mask
                        .clone()
                        .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
                    let chunk_mask =
                        Self::slice_tokens(summary_event_mask.clone(), batch_size, start, end);
                    let hidden = self.model.forward_hidden_with_state_and_summary_event_mask(
                        chunk_inputs,
                        chunk_mask,
                        state,
                    );
                    let chunk_weight = (end - start) as f32 / block_size as f32;
                    let chunk_loss = self
                        .language_loss_from_hidden(hidden, chunk_targets, chunk_loss_mask)
                        .mul_scalar(chunk_weight);
                    loss = Some(match loss {
                        Some(accumulated) => accumulated + chunk_loss,
                        None => chunk_loss,
                    });
                }
                return LanguageModelOutput::new(
                    loss.expect("streaming valid step should produce at least one loss chunk"),
                );
            }
            let hidden = self.model.forward_hidden_with_state_and_summary_event_mask(
                batch.inputs,
                summary_event_mask,
                state,
            );
            let loss = self.language_loss_from_hidden(hidden, batch.targets, loss_mask);
            return LanguageModelOutput::new(loss);
        }
        if let Some(chunk_size) =
            self.effective_tbptt_chunk_size(batch.inputs.shape().dims::<2>()[1])
        {
            let [batch_size, block_size] = batch.inputs.shape().dims();
            let mut loss: Option<Tensor<B, 1>> = None;
            for start in (0..block_size).step_by(chunk_size) {
                let end = (start + chunk_size).min(block_size);
                let chunk_inputs = Self::slice_tokens(batch.inputs.clone(), batch_size, start, end);
                let chunk_targets =
                    Self::slice_tokens(batch.targets.clone(), batch_size, start, end);
                let chunk_loss_mask = loss_mask
                    .clone()
                    .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
                let hidden = self.model.forward_hidden_with_state(chunk_inputs, state);
                let chunk_weight = (end - start) as f32 / block_size as f32;
                let chunk_loss = self
                    .language_loss_from_hidden(hidden, chunk_targets, chunk_loss_mask)
                    .mul_scalar(chunk_weight);
                loss = Some(match loss {
                    Some(accumulated) => accumulated + chunk_loss,
                    None => chunk_loss,
                });
            }
            return LanguageModelOutput::new(
                loss.expect("streaming valid step should produce at least one loss chunk"),
            );
        }
        let hidden = self.model.forward_hidden_with_state(batch.inputs, state);
        let loss = self.language_loss_from_hidden(hidden, batch.targets, loss_mask);
        LanguageModelOutput::new(loss)
    }

    pub(crate) fn step_with_predictive_context_stream_state(
        &self,
        batch: SequenceBatch<B>,
        neuron_mask: Tensor<B, 4>,
        activity_mask: Tensor<B, 4>,
        state: &mut ModelState<B>,
    ) -> LanguageModelOutput<B>
    where
        B::Device: 'static,
        B::FloatTensorPrimitive: 'static,
    {
        if batch.reset_stream_state {
            *state = self.model.init_state();
        }
        debug_assert!(
            batch.summary_event_mask.is_none(),
            "analytic predictive coding rejects summary memory"
        );
        let [batch_size, block_size] = batch.inputs.shape().dims::<2>();
        let chunk_size = self
            .effective_tbptt_chunk_size(block_size)
            .unwrap_or(block_size)
            .max(1);
        let mut loss: Option<Tensor<B, 1>> = None;
        for start in (0..block_size).step_by(chunk_size) {
            let end = (start + chunk_size).min(block_size);
            let inputs = Self::slice_tokens(batch.inputs.clone(), batch_size, start, end);
            let targets = Self::slice_tokens(batch.targets.clone(), batch_size, start, end);
            let loss_mask = batch
                .loss_mask
                .clone()
                .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
            let logits = self
                .model
                .predictive_coding_forward_with_subnetwork_masks_and_state(
                    inputs,
                    neuron_mask.clone(),
                    activity_mask.clone(),
                    state,
                )
                .expect("validated predictive context masks");
            let chunk_loss = masked_token_mean(
                self.model
                    .language_token_losses_from_logits(logits, targets),
                loss_mask,
            )
            .mul_scalar((end - start) as f32 / block_size.max(1) as f32);
            loss = Some(match loss {
                Some(total) => total + chunk_loss,
                None => chunk_loss,
            });
        }
        LanguageModelOutput::new(loss.expect("streaming context batch must contain tokens"))
    }
}
