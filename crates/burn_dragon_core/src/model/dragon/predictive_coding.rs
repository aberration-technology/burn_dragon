use super::*;
use crate::model::norm::DragonNormVjp;
use crate::model::residual_stream::lowrank_residual_step_with_metrics_branch_thresholds_relu_native;
use burn::module::ParamId;
use burn::tensor::TensorPrimitive;
use burn_dragon_kernel::api::attention::dense_causal_attention_vjp_with_initial_rho;
use burn_dragon_kernel::api::projection::{relu_lowrank_input_vjp, relu_lowrank_vjp};

/// Exact subset of Dragon currently covered by the plain-backend local VJPs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DragonPredictiveCodingSupport {
    pub layers: usize,
    pub shared_parameter_tensors: usize,
    pub head_parameter_tensors: usize,
    pub normalization_parameter_tensors: usize,
    pub embedding_parameter_tensors: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DragonPredictiveCodingParameterIds {
    pub embedding: ParamId,
    pub encoder: ParamId,
    pub encoder_v: ParamId,
    pub decoder: ParamId,
    pub norm_gamma: ParamId,
    pub norm_beta: ParamId,
    pub norm_alpha: ParamId,
    pub norm_shift: ParamId,
    pub lm_head: ParamId,
}

#[derive(Debug, Clone)]
pub struct DragonPredictiveCodingLayerTrace<B: Backend> {
    pub input: Tensor<B, 4>,
    /// Clamped rho entering this layer factor. `None` is the all-zero initial
    /// state used by a stateless block.
    pub initial_rho: Option<Tensor<B, 4>>,
    pub attention_pre_norm: Tensor<B, 4>,
    pub attention_readout: Tensor<B, 4>,
    pub residual_pre_norm: Tensor<B, 4>,
    pub residual_delta: Tensor<B, 4>,
    pub x_neuron: Tensor<B, 4>,
    pub y_gate: Tensor<B, 4>,
    pub y_neuron: Tensor<B, 4>,
    pub next: Tensor<B, 4>,
}

#[derive(Debug, Clone)]
pub struct DragonPredictiveCodingLayerVjp<B: Backend> {
    pub grad_input: Tensor<B, 4>,
    pub grad_encoder: Tensor<B, 3>,
    pub grad_encoder_v: Tensor<B, 3>,
    pub grad_decoder: Tensor<B, 2>,
    pub grad_norm_gamma: Tensor<B, 1>,
    pub grad_norm_beta: Tensor<B, 1>,
    pub grad_norm_alpha: Tensor<B, 1>,
    pub grad_norm_shift: Tensor<B, 1>,
}

#[derive(Debug, Clone)]
pub struct DragonPredictiveCodingInitialVjp<B: Backend> {
    pub grad_embedding: Tensor<B, 2>,
    pub grad_norm_gamma: Tensor<B, 1>,
    pub grad_norm_beta: Tensor<B, 1>,
    pub grad_norm_alpha: Tensor<B, 1>,
    pub grad_norm_shift: Tensor<B, 1>,
}

#[derive(Debug, Clone)]
pub struct DragonPredictiveCodingHeadVjp<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub grad_hidden: Tensor<B, 3>,
    pub grad_lm_head: Tensor<B, 2>,
    /// Raw number of supervised tokens before denominator clamping. This lets
    /// truncated factors aggregate masked document losses exactly without a
    /// device-to-host synchronization.
    pub supervised_tokens: Tensor<B, 1>,
}

#[derive(Debug, Clone)]
pub struct DragonPredictiveCodingHeadActivityVjp<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub grad_hidden: Tensor<B, 3>,
    /// Number of supervised token observations used to normalize the loss.
    /// Multiplying `grad_hidden` by this value recovers independent per-token
    /// output errors for predictive-coding activity inference.
    pub normalization: Tensor<B, 1>,
}

impl<B: Backend> DragonModel<B>
where
    B::Device: 'static,
    B::FloatTensorPrimitive: 'static,
{
    /// Validates the exact architecture contract of the analytic PC executor.
    /// Unsupported combinations fail closed instead of silently using global
    /// backpropagation.
    pub fn predictive_coding_support(&self) -> Result<DragonPredictiveCodingSupport, String> {
        if self.uses_random_scaffold() {
            return Err("random scaffold adapters are not covered by local PC VJPs".into());
        }
        if self.dropout.prob != 0.0 {
            return Err(
                "local PC requires dropout=0 so plain and training forwards coincide".into(),
            );
        }
        if self.y_neuron_recurrence.enabled {
            return Err("y-neuron recurrence is not covered by local PC VJPs".into());
        }
        if self.hierarchical_dragon.enabled {
            return Err("hierarchical Dragon is not covered by local PC VJPs".into());
        }
        if self.clocked_slow_memory.enabled {
            return Err("clocked slow memory is not covered by local PC VJPs".into());
        }
        if self.summary_memory.enabled {
            return Err("summary memory is not covered by local PC VJPs".into());
        }
        if self.latent_reasoning.enabled {
            return Err("latent reasoning is not covered by the local PC output factor".into());
        }
        if self.rollout_fast_steps_per_slow_step != 1 {
            return Err("local PC requires rollout_fast_steps_per_slow_step=1".into());
        }
        if self.tie_input_output_embeddings {
            return Err("tied input/output embeddings are not covered by local PC VJPs".into());
        }
        if !self.language_head.uses_flat_token_logits() || self.lm_head.is_none() {
            return Err("local PC requires the flat language-model head".into());
        }
        if self.sequence_kernel.memory_system != SequenceMemorySystem::LinearAttention
            || self.sequence_kernel.executor != SequenceTrainingExecutor::DenseScoreShortContext
        {
            return Err(
                "local PC currently requires linear_attention+dense_score_short_context".into(),
            );
        }
        if self.kernel.rotary_embedding != crate::RotaryEmbedding::Alibi {
            return Err("local PC dense-attention VJP currently requires ALiBi".into());
        }
        if (0..self.n_layer).any(|layer| {
            !matches!(
                self.residual_connector_for_layer(layer),
                ResidualConnectorRef::Vanilla
            )
        }) {
            return Err("local PC currently requires vanilla residual connectors".into());
        }
        let capacity = self.latent_total_capacity();
        if self
            .layer_latent_totals
            .iter()
            .any(|layer_latent| *layer_latent != capacity)
        {
            return Err("local PC currently requires uniform full latent fanout".into());
        }

        Ok(DragonPredictiveCodingSupport {
            layers: self.n_layer,
            shared_parameter_tensors: 3,
            head_parameter_tensors: 1,
            normalization_parameter_tensors: 4,
            embedding_parameter_tensors: 1,
        })
    }

    pub fn predictive_coding_parameter_ids(
        &self,
    ) -> Result<DragonPredictiveCodingParameterIds, String> {
        self.predictive_coding_support()?;
        let (norm_gamma, norm_beta, norm_alpha, norm_shift) = self.norm.parameter_ids();
        Ok(DragonPredictiveCodingParameterIds {
            embedding: self.embed.weight.id,
            encoder: self.encoder.id,
            encoder_v: self.encoder_v.id,
            decoder: self.decoder.id,
            norm_gamma,
            norm_beta,
            norm_alpha,
            norm_shift,
            lm_head: self.lm_head.as_ref().expect("validated flat LM head").id,
        })
    }

    pub fn predictive_coding_layer_count(&self) -> usize {
        self.n_layer
    }

    pub fn predictive_coding_initial_activity(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 4> {
        self.predictive_coding_support()
            .expect("unsupported Dragon predictive-coding architecture");
        let (current, history) = self.begin_language_pipeline(tokens).into_parts();
        assert!(
            history.is_empty(),
            "vanilla PC pipeline must not retain residual history"
        );
        current
    }

    pub fn predictive_coding_initial_activity_with_activity_mask(
        &self,
        tokens: Tensor<B, 2, Int>,
        activity_mask: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        self.predictive_coding_support()
            .expect("unsupported Dragon predictive-coding architecture");
        self.predictive_coding_validate_activity_mask(&activity_mask)
            .expect("invalid Dragon predictive-coding activity mask");
        let [batch, time] = tokens.shape().dims::<2>();
        let embedded = self
            .embed
            .forward(tokens)
            .reshape([batch, 1, time, self.n_embd]);
        self.predictive_coding_masked_norm(embedded, Some(&activity_mask))
    }

    pub fn predictive_coding_initial_vjp(
        &self,
        tokens: Tensor<B, 2, Int>,
        grad_activity: Tensor<B, 4>,
    ) -> DragonPredictiveCodingInitialVjp<B> {
        self.predictive_coding_initial_vjp_impl(tokens, grad_activity, None)
    }

    pub fn predictive_coding_initial_vjp_with_activity_mask(
        &self,
        tokens: Tensor<B, 2, Int>,
        grad_activity: Tensor<B, 4>,
        activity_mask: Tensor<B, 4>,
    ) -> DragonPredictiveCodingInitialVjp<B> {
        self.predictive_coding_validate_activity_mask(&activity_mask)
            .expect("invalid Dragon predictive-coding activity mask");
        self.predictive_coding_initial_vjp_impl(tokens, grad_activity, Some(activity_mask))
    }

    fn predictive_coding_initial_vjp_impl(
        &self,
        tokens: Tensor<B, 2, Int>,
        grad_activity: Tensor<B, 4>,
        activity_mask: Option<Tensor<B, 4>>,
    ) -> DragonPredictiveCodingInitialVjp<B> {
        self.predictive_coding_support()
            .expect("unsupported Dragon predictive-coding architecture");
        let [batch, time] = tokens.shape().dims::<2>();
        let embedded = self.embed.forward(tokens.clone());
        let norm_vjp = self.predictive_coding_masked_norm_vjp(
            embedded.reshape([batch, 1, time, self.n_embd]),
            grad_activity,
            activity_mask.as_ref(),
        );
        let grad_embedded = norm_vjp.grad_input.reshape([batch, time, self.n_embd]);
        let grad_embedding =
            Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(B::embedding_backward(
                self.embed.weight.val().into_primitive().tensor(),
                grad_embedded.into_primitive().tensor(),
                tokens.into_primitive(),
            )));
        DragonPredictiveCodingInitialVjp {
            grad_embedding,
            grad_norm_gamma: norm_vjp.grad_gamma,
            grad_norm_beta: norm_vjp.grad_beta,
            grad_norm_alpha: norm_vjp.grad_alpha,
            grad_norm_shift: norm_vjp.grad_shift,
        }
    }

    fn predictive_coding_dense_attention(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        initial_rho: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 4> {
        let decay = self
            .attention
            .alibi_decay()
            .expect("validated local PC ALiBi decay");
        if self.kernel.enabled
            && self.kernel.wgpu_rollout_fused
            && supports_dense_causal_attention_backend::<B>()
            && let Some(output) = try_fused_dense_causal_attention_wgpu(&query, &value, &decay)
        {
            return match initial_rho {
                Some(rho) => {
                    let value_dim = value.shape().dims::<4>()[3];
                    output
                        + self.recurrent_attention_dense_score_initial_context_reference(
                            query,
                            Some(rho),
                            Some(decay),
                            value_dim,
                        )
                }
                None => output,
            };
        }
        self.recurrent_attention_dense_score_context_reference(
            query,
            value,
            initial_rho,
            Some(decay),
        )
    }

    pub fn predictive_coding_terminal_rho(
        &self,
        trace: &DragonPredictiveCodingLayerTrace<B>,
    ) -> Tensor<B, 4> {
        let decay = self
            .attention
            .alibi_decay()
            .expect("validated local PC ALiBi decay");
        self.recurrent_attention_dense_score_final_rho_reference(
            trace.x_neuron.clone(),
            trace.input.clone(),
            trace.initial_rho.clone(),
            Some(decay),
        )
    }

    pub fn predictive_coding_neuron_dim_per_head(&self) -> Result<usize, String> {
        self.predictive_coding_support()?;
        Ok(self.latent_per_head_capacity())
    }

    pub fn predictive_coding_validate_neuron_mask(
        &self,
        neuron_mask: &Tensor<B, 4>,
    ) -> Result<(), String> {
        let [batch, heads, time, latent] = neuron_mask.shape().dims::<4>();
        let expected_latent = self.latent_per_head_capacity();
        if batch != 1
            || (heads != 1 && heads != self.n_head)
            || time != 1
            || latent != expected_latent
        {
            return Err(format!(
                "predictive-coding neuron mask must have shape [1, 1|{}, 1, {expected_latent}], got {:?}",
                self.n_head,
                neuron_mask.shape()
            ));
        }
        Ok(())
    }

    pub fn predictive_coding_validate_activity_mask(
        &self,
        activity_mask: &Tensor<B, 4>,
    ) -> Result<(), String> {
        let [batch, streams, time, dim] = activity_mask.shape().dims::<4>();
        if batch != 1 || streams != 1 || time != 1 || dim != self.n_embd {
            return Err(format!(
                "predictive-coding activity mask must have shape [1, 1, 1, {}], got {:?}",
                self.n_embd,
                activity_mask.shape()
            ));
        }
        Ok(())
    }

    pub fn predictive_coding_validate_rho_state(
        &self,
        rho: &Tensor<B, 4>,
        batch: usize,
    ) -> Result<(), String> {
        let expected = [
            batch,
            self.n_head,
            self.latent_per_head_capacity(),
            self.n_embd,
        ];
        if rho.shape().dims::<4>() != expected {
            return Err(format!(
                "predictive-coding rho must have shape {expected:?}, got {:?}",
                rho.shape()
            ));
        }
        Ok(())
    }

    fn predictive_coding_neuron_mask(
        &self,
        latent: usize,
        device: &B::Device,
        context_mask: Option<Tensor<B, 4>>,
    ) -> Option<Tensor<B, 4>> {
        let latent_pattern = &self.kernel.block_sparse.latent;
        let configured = (self.kernel.enabled && latent_pattern.is_sparse())
            .then(|| latent_pattern.mask::<B>(latent, device));
        match (configured, context_mask) {
            (Some(configured), Some(context)) => Some(configured * context),
            (Some(configured), None) => Some(configured),
            (None, context) => context,
        }
    }

    fn predictive_coding_masked_norm(
        &self,
        input: Tensor<B, 4>,
        activity_mask: Option<&Tensor<B, 4>>,
    ) -> Tensor<B, 4> {
        match activity_mask {
            Some(mask) => self.norm.forward(input * mask.clone()) * mask.clone(),
            None => self.norm.forward(input),
        }
    }

    fn predictive_coding_masked_norm_vjp(
        &self,
        input: Tensor<B, 4>,
        grad_output: Tensor<B, 4>,
        activity_mask: Option<&Tensor<B, 4>>,
    ) -> DragonNormVjp<B, 4> {
        match activity_mask {
            Some(mask) => {
                let vjp = self
                    .norm
                    .vjp_with_parameters(input * mask.clone(), grad_output * mask.clone());
                DragonNormVjp {
                    grad_input: vjp.grad_input * mask.clone(),
                    ..vjp
                }
            }
            None => self.norm.vjp_with_parameters(input, grad_output),
        }
    }

    fn predictive_coding_masked_norm_input_vjp(
        &self,
        input: Tensor<B, 4>,
        grad_output: Tensor<B, 4>,
        activity_mask: Option<&Tensor<B, 4>>,
    ) -> Tensor<B, 4> {
        match activity_mask {
            Some(mask) => {
                self.norm
                    .vjp_input(input * mask.clone(), grad_output * mask.clone())
                    * mask.clone()
            }
            None => self.norm.vjp_input(input, grad_output),
        }
    }

    pub fn predictive_coding_forward_layer(
        &self,
        input: Tensor<B, 4>,
        layer_index: usize,
    ) -> DragonPredictiveCodingLayerTrace<B> {
        self.predictive_coding_forward_layer_impl(input, layer_index, None, None, None)
    }

    pub fn predictive_coding_forward_layer_with_neuron_mask(
        &self,
        input: Tensor<B, 4>,
        layer_index: usize,
        neuron_mask: Tensor<B, 4>,
    ) -> DragonPredictiveCodingLayerTrace<B> {
        self.predictive_coding_validate_neuron_mask(&neuron_mask)
            .expect("invalid Dragon predictive-coding neuron mask");
        self.predictive_coding_forward_layer_impl(input, layer_index, None, Some(neuron_mask), None)
    }

    pub fn predictive_coding_forward_layer_with_subnetwork_masks(
        &self,
        input: Tensor<B, 4>,
        layer_index: usize,
        neuron_mask: Tensor<B, 4>,
        activity_mask: Tensor<B, 4>,
    ) -> DragonPredictiveCodingLayerTrace<B> {
        self.predictive_coding_validate_neuron_mask(&neuron_mask)
            .expect("invalid Dragon predictive-coding neuron mask");
        self.predictive_coding_validate_activity_mask(&activity_mask)
            .expect("invalid Dragon predictive-coding activity mask");
        self.predictive_coding_forward_layer_impl(
            input,
            layer_index,
            None,
            Some(neuron_mask),
            Some(activity_mask),
        )
    }

    /// Evaluate one local Dragon factor with a detached incoming rho state and
    /// optional context masks. This is the recurrent/TBPTT integration point;
    /// callers intentionally decide whether to propagate or truncate the
    /// returned state's derivative.
    pub fn predictive_coding_forward_layer_with_recurrent_state(
        &self,
        input: Tensor<B, 4>,
        layer_index: usize,
        initial_rho: Option<Tensor<B, 4>>,
        neuron_mask: Option<Tensor<B, 4>>,
        activity_mask: Option<Tensor<B, 4>>,
    ) -> Result<DragonPredictiveCodingLayerTrace<B>, String> {
        if let Some(mask) = neuron_mask.as_ref() {
            self.predictive_coding_validate_neuron_mask(mask)?;
        }
        if let Some(mask) = activity_mask.as_ref() {
            self.predictive_coding_validate_activity_mask(mask)?;
            if neuron_mask.is_none() {
                return Err("activity mask requires a neuron mask".to_string());
            }
        }
        if let Some(rho) = initial_rho.as_ref() {
            let input_batch = input.shape().dims::<4>()[0];
            self.predictive_coding_validate_rho_state(rho, input_batch)?;
        }
        Ok(self.predictive_coding_forward_layer_impl(
            input,
            layer_index,
            initial_rho,
            neuron_mask,
            activity_mask,
        ))
    }

    fn predictive_coding_forward_layer_impl(
        &self,
        input: Tensor<B, 4>,
        layer_index: usize,
        initial_rho: Option<Tensor<B, 4>>,
        context_mask: Option<Tensor<B, 4>>,
        activity_mask: Option<Tensor<B, 4>>,
    ) -> DragonPredictiveCodingLayerTrace<B> {
        self.predictive_coding_support()
            .expect("unsupported Dragon predictive-coding architecture");
        assert!(
            layer_index < self.n_layer,
            "local PC layer index out of range"
        );
        let (encoder, encoder_v, decoder, latent) = self.layer_lowrank_weights(layer_index);
        let latent_pattern = &self.kernel.block_sparse.latent;
        let sparse_mask = self.predictive_coding_neuron_mask(latent, &input.device(), context_mask);
        let output = lowrank_residual_step_with_metrics_branch_thresholds_relu_native(
            input.clone(),
            encoder,
            encoder_v,
            decoder,
            &self.dropout,
            self.kernel.enabled && self.kernel.projection_executor.use_x(),
            self.kernel.enabled && self.kernel.projection_executor.use_y(),
            self.x_relu_threshold,
            self.y_relu_threshold,
            true,
            latent_pattern,
            self.kernel.lowrank_grad_input_executor,
            sparse_mask,
            |query, value| {
                self.predictive_coding_dense_attention(query, value, initial_rho.clone())
            },
            activation::relu,
            |values| self.predictive_coding_masked_norm(values, activity_mask.as_ref()),
        );
        DragonPredictiveCodingLayerTrace {
            input,
            initial_rho,
            attention_pre_norm: output
                .attention_pre_norm
                .expect("full low-rank output retains pre-normalization attention"),
            attention_readout: output
                .attention_readout
                .expect("full low-rank output retains attention readout"),
            residual_pre_norm: output
                .residual_pre_norm
                .expect("full low-rank output retains pre-normalization residual"),
            residual_delta: output
                .residual_delta
                .expect("full low-rank output retains residual delta"),
            x_neuron: output.x_neuron,
            y_gate: output.y_gate,
            y_neuron: output.y_neuron,
            next: output.next,
        }
    }

    pub fn predictive_coding_layer_vjp(
        &self,
        layer_index: usize,
        trace: &DragonPredictiveCodingLayerTrace<B>,
        grad_next: Tensor<B, 4>,
    ) -> DragonPredictiveCodingLayerVjp<B> {
        self.predictive_coding_layer_vjp_impl(layer_index, trace, grad_next, None, None)
    }

    pub fn predictive_coding_layer_vjp_with_neuron_mask(
        &self,
        layer_index: usize,
        trace: &DragonPredictiveCodingLayerTrace<B>,
        grad_next: Tensor<B, 4>,
        neuron_mask: Tensor<B, 4>,
    ) -> DragonPredictiveCodingLayerVjp<B> {
        self.predictive_coding_validate_neuron_mask(&neuron_mask)
            .expect("invalid Dragon predictive-coding neuron mask");
        self.predictive_coding_layer_vjp_impl(
            layer_index,
            trace,
            grad_next,
            Some(neuron_mask),
            None,
        )
    }

    pub fn predictive_coding_layer_vjp_with_subnetwork_masks(
        &self,
        layer_index: usize,
        trace: &DragonPredictiveCodingLayerTrace<B>,
        grad_next: Tensor<B, 4>,
        neuron_mask: Tensor<B, 4>,
        activity_mask: Tensor<B, 4>,
    ) -> DragonPredictiveCodingLayerVjp<B> {
        self.predictive_coding_validate_neuron_mask(&neuron_mask)
            .expect("invalid Dragon predictive-coding neuron mask");
        self.predictive_coding_validate_activity_mask(&activity_mask)
            .expect("invalid Dragon predictive-coding activity mask");
        self.predictive_coding_layer_vjp_impl(
            layer_index,
            trace,
            grad_next,
            Some(neuron_mask),
            Some(activity_mask),
        )
    }

    fn predictive_coding_layer_vjp_impl(
        &self,
        layer_index: usize,
        trace: &DragonPredictiveCodingLayerTrace<B>,
        grad_next: Tensor<B, 4>,
        context_mask: Option<Tensor<B, 4>>,
        activity_mask: Option<Tensor<B, 4>>,
    ) -> DragonPredictiveCodingLayerVjp<B> {
        self.predictive_coding_support()
            .expect("unsupported Dragon predictive-coding architecture");
        let (encoder, encoder_v, decoder, latent) = self.layer_lowrank_weights(layer_index);
        let sparse_mask =
            self.predictive_coding_neuron_mask(latent, &trace.input.device(), context_mask);

        let residual_sum = trace.input.clone() + trace.residual_delta.clone();
        let DragonNormVjp {
            grad_input: grad_residual_sum,
            grad_gamma: residual_gamma,
            grad_beta: residual_beta,
            grad_alpha: residual_alpha,
            grad_shift: residual_shift,
        } = self.predictive_coding_masked_norm_vjp(residual_sum, grad_next, activity_mask.as_ref());
        let DragonNormVjp {
            grad_input: grad_mlp_raw,
            grad_gamma: mlp_gamma,
            grad_beta: mlp_beta,
            grad_alpha: mlp_alpha,
            grad_shift: mlp_shift,
        } = self.predictive_coding_masked_norm_vjp(
            trace.residual_pre_norm.clone(),
            grad_residual_sum.clone(),
            activity_mask.as_ref(),
        );

        let [batch, heads, time, latent] = trace.y_neuron.shape().dims::<4>();
        let dim = trace.input.shape().dims::<4>()[3];
        let y_flat = trace
            .y_neuron
            .clone()
            .swap_dims(1, 2)
            .reshape([batch * time, heads * latent]);
        let grad_mlp_flat = grad_mlp_raw.reshape([batch * time, dim]);
        let grad_decoder = y_flat.clone().transpose().matmul(grad_mlp_flat.clone());
        let grad_y = grad_mlp_flat
            .matmul(decoder.transpose())
            .reshape([batch, time, heads, latent])
            .swap_dims(1, 2);

        let grad_x_from_product = grad_y.clone() * trace.y_gate.clone();
        let grad_y_gate = grad_y * trace.x_neuron.clone();
        let y_vjp = relu_lowrank_vjp(
            trace.attention_readout.clone(),
            encoder_v,
            grad_y_gate,
            self.y_relu_threshold,
            sparse_mask.clone(),
            self.kernel.lowrank_grad_input_executor,
        )
        .expect("validated local PC y-projection VJP");

        let DragonNormVjp {
            grad_input: grad_raw_attention,
            grad_gamma: attention_gamma,
            grad_beta: attention_beta,
            grad_alpha: attention_alpha,
            grad_shift: attention_shift,
        } = self.predictive_coding_masked_norm_vjp(
            trace.attention_pre_norm.clone(),
            y_vjp.grad_input,
            activity_mask.as_ref(),
        );
        let decay = self
            .attention
            .alibi_decay()
            .expect("validated local PC ALiBi decay");
        let attention_vjp = dense_causal_attention_vjp_with_initial_rho(
            grad_raw_attention,
            trace.x_neuron.clone(),
            trace.input.clone(),
            decay,
            trace.initial_rho.clone(),
        );
        let grad_x = grad_x_from_product + attention_vjp.grad_query;
        let x_vjp = relu_lowrank_vjp(
            trace.input.clone(),
            encoder,
            grad_x,
            self.x_relu_threshold,
            sparse_mask,
            self.kernel.lowrank_grad_input_executor,
        )
        .expect("validated local PC x-projection VJP");

        DragonPredictiveCodingLayerVjp {
            grad_input: grad_residual_sum + attention_vjp.grad_value + x_vjp.grad_input,
            grad_encoder: x_vjp.grad_weight.reshape([
                self.n_head,
                self.n_embd,
                self.latent_per_head_capacity(),
            ]),
            grad_encoder_v: y_vjp.grad_weight.reshape([
                self.n_head,
                self.n_embd,
                self.latent_per_head_capacity(),
            ]),
            grad_decoder,
            grad_norm_gamma: residual_gamma + mlp_gamma + attention_gamma,
            grad_norm_beta: residual_beta + mlp_beta + attention_beta,
            grad_norm_alpha: residual_alpha + mlp_alpha + attention_alpha,
            grad_norm_shift: residual_shift + mlp_shift + attention_shift,
        }
    }

    /// Input-only local VJP used while settling adjacent activities.
    ///
    /// Unlike [`Self::predictive_coding_layer_vjp`], this does not construct
    /// decoder, encoder, or normalization parameter derivatives that activity
    /// inference would immediately discard.
    pub fn predictive_coding_layer_activity_vjp(
        &self,
        layer_index: usize,
        trace: &DragonPredictiveCodingLayerTrace<B>,
        grad_next: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        self.predictive_coding_layer_activity_vjp_impl(layer_index, trace, grad_next, None, None)
    }

    pub fn predictive_coding_layer_activity_vjp_with_neuron_mask(
        &self,
        layer_index: usize,
        trace: &DragonPredictiveCodingLayerTrace<B>,
        grad_next: Tensor<B, 4>,
        neuron_mask: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        self.predictive_coding_validate_neuron_mask(&neuron_mask)
            .expect("invalid Dragon predictive-coding neuron mask");
        self.predictive_coding_layer_activity_vjp_impl(
            layer_index,
            trace,
            grad_next,
            Some(neuron_mask),
            None,
        )
    }

    pub fn predictive_coding_layer_activity_vjp_with_subnetwork_masks(
        &self,
        layer_index: usize,
        trace: &DragonPredictiveCodingLayerTrace<B>,
        grad_next: Tensor<B, 4>,
        neuron_mask: Tensor<B, 4>,
        activity_mask: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        self.predictive_coding_validate_neuron_mask(&neuron_mask)
            .expect("invalid Dragon predictive-coding neuron mask");
        self.predictive_coding_validate_activity_mask(&activity_mask)
            .expect("invalid Dragon predictive-coding activity mask");
        self.predictive_coding_layer_activity_vjp_impl(
            layer_index,
            trace,
            grad_next,
            Some(neuron_mask),
            Some(activity_mask),
        )
    }

    fn predictive_coding_layer_activity_vjp_impl(
        &self,
        layer_index: usize,
        trace: &DragonPredictiveCodingLayerTrace<B>,
        grad_next: Tensor<B, 4>,
        context_mask: Option<Tensor<B, 4>>,
        activity_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 4> {
        self.predictive_coding_support()
            .expect("unsupported Dragon predictive-coding architecture");
        let (encoder, encoder_v, decoder, latent) = self.layer_lowrank_weights(layer_index);
        let sparse_mask =
            self.predictive_coding_neuron_mask(latent, &trace.input.device(), context_mask);

        let residual_sum = trace.input.clone() + trace.residual_delta.clone();
        let grad_residual_sum = self.predictive_coding_masked_norm_input_vjp(
            residual_sum,
            grad_next,
            activity_mask.as_ref(),
        );
        let grad_mlp_raw = self.predictive_coding_masked_norm_input_vjp(
            trace.residual_pre_norm.clone(),
            grad_residual_sum.clone(),
            activity_mask.as_ref(),
        );

        let [batch, heads, time, latent] = trace.y_neuron.shape().dims::<4>();
        let dim = trace.input.shape().dims::<4>()[3];
        let grad_y = grad_mlp_raw
            .reshape([batch * time, dim])
            .matmul(decoder.transpose())
            .reshape([batch, time, heads, latent])
            .swap_dims(1, 2);
        let grad_x_from_product = grad_y.clone() * trace.y_gate.clone();
        let grad_y_gate = grad_y * trace.x_neuron.clone();
        let grad_attention_readout = relu_lowrank_input_vjp(
            trace.attention_readout.clone(),
            encoder_v,
            grad_y_gate,
            self.y_relu_threshold,
            sparse_mask.clone(),
            self.kernel.lowrank_grad_input_executor,
        )
        .expect("validated local PC y-projection input VJP");

        let grad_raw_attention = self.predictive_coding_masked_norm_input_vjp(
            trace.attention_pre_norm.clone(),
            grad_attention_readout,
            activity_mask.as_ref(),
        );
        let decay = self
            .attention
            .alibi_decay()
            .expect("validated local PC ALiBi decay");
        let attention_vjp = dense_causal_attention_vjp_with_initial_rho(
            grad_raw_attention,
            trace.x_neuron.clone(),
            trace.input.clone(),
            decay,
            trace.initial_rho.clone(),
        );
        let grad_x = grad_x_from_product + attention_vjp.grad_query;
        let grad_projection_input = relu_lowrank_input_vjp(
            trace.input.clone(),
            encoder,
            grad_x,
            self.x_relu_threshold,
            sparse_mask,
            self.kernel.lowrank_grad_input_executor,
        )
        .expect("validated local PC x-projection input VJP");

        grad_residual_sum + attention_vjp.grad_value + grad_projection_input
    }

    pub fn predictive_coding_hidden_from_activity(&self, activity: Tensor<B, 4>) -> Tensor<B, 3> {
        self.collapse_language_streams(activity)
    }

    /// Forward the analytic-PC-compatible Dragon architecture while applying
    /// a fixed context competition mask to every layer's neuron channels.
    ///
    /// A `[1, 1|n_head, 1, latent_per_head]` binary mask is the fixed context-
    /// competition upper bound used by continual-learning experiments. The operation
    /// remains ordinarily differentiable when `B` is an autodiff backend.
    pub fn predictive_coding_forward_with_neuron_mask(
        &self,
        tokens: Tensor<B, 2, Int>,
        neuron_mask: Tensor<B, 4>,
    ) -> Result<Tensor<B, 3>, String> {
        self.predictive_coding_forward_with_context_masks(tokens, Some(neuron_mask), None)
    }

    /// Forward an oracle context-selected Dragon subnetwork. Neuron masks
    /// select rho channels inside each low-rank factor; activity masks select
    /// residual-state channels, including the embedding and language-head
    /// interfaces.
    pub fn predictive_coding_forward_with_subnetwork_masks(
        &self,
        tokens: Tensor<B, 2, Int>,
        neuron_mask: Tensor<B, 4>,
        activity_mask: Tensor<B, 4>,
    ) -> Result<Tensor<B, 3>, String> {
        self.predictive_coding_forward_with_context_masks(
            tokens,
            Some(neuron_mask),
            Some(activity_mask),
        )
    }

    fn predictive_coding_forward_with_context_masks(
        &self,
        tokens: Tensor<B, 2, Int>,
        neuron_mask: Option<Tensor<B, 4>>,
        activity_mask: Option<Tensor<B, 4>>,
    ) -> Result<Tensor<B, 3>, String> {
        self.predictive_coding_support()?;
        if let Some(mask) = neuron_mask.as_ref() {
            self.predictive_coding_validate_neuron_mask(mask)?;
        }
        if let Some(mask) = activity_mask.as_ref() {
            self.predictive_coding_validate_activity_mask(mask)?;
        }
        let [batch, time] = tokens.shape().dims::<2>();
        let mut activity = match activity_mask.as_ref() {
            Some(mask) => {
                self.predictive_coding_initial_activity_with_activity_mask(tokens, mask.clone())
            }
            None => self.predictive_coding_initial_activity(tokens),
        };
        for layer in 0..self.predictive_coding_layer_count() {
            activity = match (neuron_mask.as_ref(), activity_mask.as_ref()) {
                (Some(neuron_mask), Some(activity_mask)) => {
                    self.predictive_coding_forward_layer_with_subnetwork_masks(
                        activity,
                        layer,
                        neuron_mask.clone(),
                        activity_mask.clone(),
                    )
                    .next
                }
                (Some(mask), None) => {
                    self.predictive_coding_forward_layer_with_neuron_mask(
                        activity,
                        layer,
                        mask.clone(),
                    )
                    .next
                }
                (None, None) => self.predictive_coding_forward_layer(activity, layer).next,
                (None, Some(_)) => unreachable!("activity mask requires a neuron mask"),
            };
            if let Some(mask) = activity_mask.as_ref() {
                activity = activity * mask.clone();
            }
        }
        let hidden = self.predictive_coding_hidden_from_activity(activity);
        let head = self.predictive_coding_head_weight()?;
        let vocab = head.shape().dims::<2>()[1];
        Ok(hidden
            .reshape([batch * time, self.n_embd])
            .matmul(head)
            .reshape([batch, time, vocab]))
    }

    pub fn predictive_coding_head_weight(&self) -> Result<Tensor<B, 2>, String> {
        self.predictive_coding_support()?;
        Ok(self.lm_head.as_ref().expect("validated flat LM head").val())
    }

    pub fn predictive_coding_head_vjp(
        &self,
        hidden: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
    ) -> DragonPredictiveCodingHeadVjp<B> {
        let head = self
            .predictive_coding_head_weight()
            .expect("unsupported Dragon predictive-coding head");
        let [batch, time, dim] = hidden.shape().dims::<3>();
        let vocab = head.shape().dims::<2>()[1];
        assert_eq!(targets.shape().dims::<2>(), [batch, time]);

        let logits = hidden
            .clone()
            .reshape([batch * time, dim])
            .matmul(head.clone())
            .reshape([batch, time, vocab]);
        let log_probs = activation::log_softmax(logits.clone(), 2);
        let selected = log_probs
            .clone()
            .gather(2, targets.clone().reshape([batch, time, 1]))
            .reshape([batch, time]);
        let mask = loss_mask.map_or_else(
            || Tensor::<B, 2>::ones([batch, time], &hidden.device()),
            |mask| mask.float(),
        );
        let supervised_tokens = mask.clone().sum();
        let denominator = supervised_tokens.clone().clamp_min(1.0);
        let loss = (selected.mul_scalar(-1.0) * mask.clone())
            .sum()
            .div(denominator.clone())
            .reshape([1]);

        let one_hot = targets.one_hot::<3>(vocab).float();
        let grad_logits = (activation::softmax(logits, 2) - one_hot)
            * mask.reshape([batch, time, 1])
            / denominator.reshape([1, 1, 1]);
        let hidden_flat = hidden.reshape([batch * time, dim]);
        let grad_logits_flat = grad_logits.reshape([batch * time, vocab]);
        let grad_hidden = grad_logits_flat
            .clone()
            .matmul(head.transpose())
            .reshape([batch, time, dim]);
        let grad_lm_head = hidden_flat.transpose().matmul(grad_logits_flat);

        DragonPredictiveCodingHeadVjp {
            loss,
            grad_hidden,
            grad_lm_head,
            supervised_tokens: supervised_tokens.reshape([1]),
        }
    }

    /// Terminal-factor VJP used during activity inference. This omits the
    /// expensive head-weight outer product, which is needed only once during
    /// the parameter-update phase.
    pub fn predictive_coding_head_activity_vjp(
        &self,
        hidden: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        loss_mask: Option<Tensor<B, 2, Int>>,
    ) -> DragonPredictiveCodingHeadActivityVjp<B> {
        let head = self
            .predictive_coding_head_weight()
            .expect("unsupported Dragon predictive-coding head");
        let [batch, time, dim] = hidden.shape().dims::<3>();
        let vocab = head.shape().dims::<2>()[1];
        assert_eq!(targets.shape().dims::<2>(), [batch, time]);

        let logits = hidden
            .reshape([batch * time, dim])
            .matmul(head.clone())
            .reshape([batch, time, vocab]);
        let log_probs = activation::log_softmax(logits.clone(), 2);
        let selected = log_probs
            .gather(2, targets.clone().reshape([batch, time, 1]))
            .reshape([batch, time]);
        let mask = loss_mask.map_or_else(
            || Tensor::<B, 2>::ones([batch, time], &head.device()),
            |mask| mask.float(),
        );
        let denominator = mask.clone().sum().clamp_min(1.0);
        let loss = (selected.mul_scalar(-1.0) * mask.clone())
            .sum()
            .div(denominator.clone())
            .reshape([1]);
        let grad_logits = (activation::softmax(logits, 2) - targets.one_hot::<3>(vocab).float())
            * mask.reshape([batch, time, 1])
            / denominator.clone().reshape([1, 1, 1]);
        let grad_hidden = grad_logits
            .reshape([batch * time, vocab])
            .matmul(head.transpose())
            .reshape([batch, time, dim]);
        DragonPredictiveCodingHeadActivityVjp {
            loss,
            grad_hidden,
            normalization: denominator,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::AutodiffModule;
    use burn::optim::GradientsParams;
    use burn::tensor::TensorData;
    use burn_autodiff::Autodiff;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;
    type TestAutodiffBackend = Autodiff<TestBackend>;

    fn config() -> DragonConfig {
        let mut config = DragonConfig {
            n_layer: 1,
            n_embd: 8,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 16,
            dropout: 0.0,
            ..DragonConfig::default()
        };
        config.sequence_kernel.executor = SequenceTrainingExecutor::DenseScoreShortContext;
        config.fused_kernels.rotary_embedding = crate::RotaryEmbedding::Alibi;
        config
    }

    fn max_abs_diff<const D: usize>(
        left: Tensor<TestBackend, D>,
        right: Tensor<TestBackend, D>,
    ) -> f32 {
        (left - right)
            .abs()
            .max()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("max difference")[0]
    }

    #[test]
    fn pc_layer_forward_matches_language_pipeline_stage() {
        let device = Default::default();
        let model = DragonModel::<TestBackend>::new(config(), &device);
        model.predictive_coding_support().expect("supported model");
        let tokens = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]),
            &device,
        );
        let activity = model.predictive_coding_initial_activity(tokens.clone());
        let trace = model.predictive_coding_forward_layer(activity, 0);
        let mut state = model.init_state_ephemeral();
        let expected = model.forward_language_pipeline_stage_with_state(
            model.begin_language_pipeline(tokens),
            &mut state,
            0..1,
            None,
        );
        let diff = max_abs_diff(
            trace.next.reshape([1, 4, 8]),
            expected.current().clone().reshape([1, 4, 8]),
        );
        assert!(diff < 1.0e-5, "PC factor forward mismatch: {diff}");
    }

    #[test]
    fn pc_layer_vjp_matches_local_autodiff() {
        let device = Default::default();
        let model = DragonModel::<TestAutodiffBackend>::new(config(), &device);
        let tokens = Tensor::<TestAutodiffBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]),
            &device,
        );
        let input = model
            .predictive_coding_initial_activity(tokens)
            .detach()
            .require_grad();
        let trace = model.predictive_coding_forward_layer(input.clone(), 0);
        let grad_next =
            Tensor::<TestAutodiffBackend, 4>::ones([1, 1, 4, 8], &device).mul_scalar(0.125);
        let mut raw_grads = (trace.next * grad_next).sum().backward();
        let input_grad = input.grad_remove(&mut raw_grads).expect("input gradient");
        let parameter_grads = GradientsParams::from_grads(raw_grads, &model);
        let ids = model.shared_lowrank_param_ids();
        let pc_ids = model.predictive_coding_parameter_ids().expect("PC ids");

        let plain = model.valid();
        let plain_input = input.detach().inner();
        let plain_trace = plain.predictive_coding_forward_layer(plain_input, 0);
        let analytic = plain.predictive_coding_layer_vjp(
            0,
            &plain_trace,
            Tensor::<TestBackend, 4>::ones([1, 1, 4, 8], &device).mul_scalar(0.125),
        );
        let activity_only = plain.predictive_coding_layer_activity_vjp(
            0,
            &plain_trace,
            Tensor::<TestBackend, 4>::ones([1, 1, 4, 8], &device).mul_scalar(0.125),
        );

        let input_diff = (input_grad - analytic.grad_input.clone())
            .abs()
            .max()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("input diff")[0];
        let encoder_diff = max_abs_diff(
            parameter_grads
                .get::<TestBackend, 3>(ids.encoder)
                .expect("encoder gradient"),
            analytic.grad_encoder,
        );
        let encoder_v_diff = max_abs_diff(
            parameter_grads
                .get::<TestBackend, 3>(ids.encoder_v)
                .expect("encoder-v gradient"),
            analytic.grad_encoder_v,
        );
        let decoder_diff = (parameter_grads
            .get::<TestBackend, 2>(ids.decoder)
            .expect("decoder gradient")
            - analytic.grad_decoder)
            .abs()
            .max()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("decoder diff")[0];
        let gamma_diff = max_abs_diff(
            parameter_grads
                .get::<TestBackend, 1>(pc_ids.norm_gamma)
                .expect("norm gamma gradient"),
            analytic.grad_norm_gamma,
        );
        let beta_diff = max_abs_diff(
            parameter_grads
                .get::<TestBackend, 1>(pc_ids.norm_beta)
                .expect("norm beta gradient"),
            analytic.grad_norm_beta,
        );
        let activity_only_diff = max_abs_diff(analytic.grad_input.clone(), activity_only);
        assert!(input_diff < 2.0e-4, "input VJP mismatch: {input_diff}");
        assert!(
            encoder_diff < 2.0e-4,
            "encoder VJP mismatch: {encoder_diff}"
        );
        assert!(
            encoder_v_diff < 2.0e-4,
            "encoder-v VJP mismatch: {encoder_v_diff}"
        );
        assert!(
            decoder_diff < 2.0e-4,
            "decoder VJP mismatch: {decoder_diff}"
        );
        assert!(gamma_diff < 2.0e-4, "norm gamma VJP mismatch: {gamma_diff}");
        assert!(beta_diff < 2.0e-4, "norm beta VJP mismatch: {beta_diff}");
        assert!(
            activity_only_diff < 2.0e-4,
            "activity-only VJP mismatch: {activity_only_diff}"
        );
    }

    #[test]
    fn pc_initial_embedding_vjp_matches_local_autodiff() {
        let device = Default::default();
        let model = DragonModel::<TestAutodiffBackend>::new(config(), &device);
        let tokens = Tensor::<TestAutodiffBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 1, 4], [1, 4]),
            &device,
        );
        let grad_activity =
            Tensor::<TestAutodiffBackend, 4>::ones([1, 1, 4, 8], &device).mul_scalar(0.125);
        let activity = model.predictive_coding_initial_activity(tokens.clone());
        let raw_grads = (activity * grad_activity).sum().backward();
        let parameter_grads = GradientsParams::from_grads(raw_grads, &model);
        let ids = model.predictive_coding_parameter_ids().expect("PC ids");

        let plain = model.valid();
        let analytic = plain.predictive_coding_initial_vjp(
            tokens.inner(),
            Tensor::<TestBackend, 4>::ones([1, 1, 4, 8], &device).mul_scalar(0.125),
        );
        let embedding_diff = max_abs_diff(
            parameter_grads
                .get::<TestBackend, 2>(ids.embedding)
                .expect("embedding gradient"),
            analytic.grad_embedding,
        );
        let gamma_diff = max_abs_diff(
            parameter_grads
                .get::<TestBackend, 1>(ids.norm_gamma)
                .expect("norm gamma gradient"),
            analytic.grad_norm_gamma,
        );
        let beta_diff = max_abs_diff(
            parameter_grads
                .get::<TestBackend, 1>(ids.norm_beta)
                .expect("norm beta gradient"),
            analytic.grad_norm_beta,
        );
        assert!(
            embedding_diff < 2.0e-4,
            "embedding VJP mismatch: {embedding_diff}"
        );
        assert!(
            gamma_diff < 2.0e-4,
            "initial norm gamma mismatch: {gamma_diff}"
        );
        assert!(
            beta_diff < 2.0e-4,
            "initial norm beta mismatch: {beta_diff}"
        );
    }

    #[test]
    fn pc_head_vjp_matches_local_autodiff() {
        let device = Default::default();
        let model = DragonModel::<TestAutodiffBackend>::new(config(), &device);
        let hidden = Tensor::<TestAutodiffBackend, 3>::random(
            [1, 4, 8],
            burn::tensor::Distribution::Normal(0.0, 0.5),
            &device,
        )
        .require_grad();
        let hidden_plain = hidden.clone().detach().inner();
        let targets = Tensor::<TestAutodiffBackend, 2, Int>::from_data(
            TensorData::new(vec![1_i64, 2, 3, 4], [1, 4]),
            &device,
        );
        let head = model.predictive_coding_head_weight().expect("head");
        let logits = hidden
            .clone()
            .reshape([4, 8])
            .matmul(head)
            .reshape([1, 4, 16]);
        let loss = activation::log_softmax(logits, 2)
            .gather(2, targets.clone().reshape([1, 4, 1]))
            .mean()
            .mul_scalar(-1.0)
            .reshape([1]);
        let mut raw_grads = loss.backward();
        let hidden_grad = hidden.grad_remove(&mut raw_grads).expect("hidden gradient");
        let parameter_grads = GradientsParams::from_grads(raw_grads, &model);
        let ids = model.predictive_coding_parameter_ids().expect("PC ids");

        let analytic =
            model
                .valid()
                .predictive_coding_head_vjp(hidden_plain, targets.inner(), None);
        let hidden_diff = max_abs_diff(hidden_grad, analytic.grad_hidden);
        let head_diff = max_abs_diff(
            parameter_grads
                .get::<TestBackend, 2>(ids.lm_head)
                .expect("head gradient"),
            analytic.grad_lm_head,
        );
        assert!(
            hidden_diff < 2.0e-4,
            "head input VJP mismatch: {hidden_diff}"
        );
        assert!(head_diff < 2.0e-4, "head weight VJP mismatch: {head_diff}");
    }

    #[test]
    fn pc_support_rejects_dropout_that_plain_backend_would_skip() {
        let device = Default::default();
        let mut config = config();
        config.dropout = 0.1;
        let model = DragonModel::<TestBackend>::new(config, &device);
        let error = model
            .predictive_coding_support()
            .expect_err("dropout mismatch must fail closed");
        assert!(error.contains("dropout=0"), "unexpected error: {error}");
    }
}
