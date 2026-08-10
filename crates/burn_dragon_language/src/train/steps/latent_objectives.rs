//! JEPA, NextLat, energy, step-contract, and latent-state objectives.

use super::*;

impl<B: BackendTrait> LanguageTrainModel<B> {
    pub(super) fn latent_reasoning_target_hidden(
        &self,
        hidden: Tensor<B, 3>,
        clean_inputs: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        if !self.latent_reasoning.enabled
            || self.pipeline_enabled()
            || !matches!(
                self.latent_reasoning.target_encoder,
                crate::config::LatentReasoningTargetEncoder::EmaTeacher
            )
        {
            return hidden.detach();
        }
        self.current_teacher_model()
            .forward_hidden(clean_inputs)
            .detach()
    }

    pub(super) fn shifted_latent_negative(target: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, time, dim] = target.shape().dims();
        if batch > 1 {
            let head = target.clone().slice([0..1, 0..time, 0..dim]);
            let tail = target.slice([1..batch, 0..time, 0..dim]);
            Tensor::cat(vec![tail, head], 0)
        } else if time > 1 {
            let head = target.clone().slice([0..batch, 0..1, 0..dim]);
            let tail = target.slice([0..batch, 1..time, 0..dim]);
            Tensor::cat(vec![tail, head], 1)
        } else {
            target
        }
    }

    pub(super) fn sigreg_loss_from_hidden(&self, hidden: Tensor<B, 3>) -> Option<Tensor<B, 1>> {
        if !self.latent_reasoning.sigreg.enabled
            || !matches!(
                self.latent_reasoning.sigreg.target,
                crate::config::LatentReasoningSigRegTarget::Hidden
                    | crate::config::LatentReasoningSigRegTarget::HiddenAndRhoMemorySlots
            )
        {
            return None;
        }
        let [batch, time, dim] = hidden.shape().dims();
        if batch == 0 || time == 0 || dim == 0 {
            return None;
        }
        let mean = hidden.clone().mean_dim(0).mean_dim(1);
        let centered = hidden - mean.clone().repeat_dim(0, batch).repeat_dim(1, time);
        let variance = centered.powf_scalar(2.0).mean_dim(0).mean_dim(1);
        let variance_floor = self.latent_reasoning.sigreg.min_variance.max(0.0);
        let variance_loss = variance
            .mul_scalar(-1.0)
            .add_scalar(variance_floor)
            .clamp_min(0.0)
            .powf_scalar(2.0)
            .mean();
        let mean_tolerance = self.latent_reasoning.sigreg.mean_tolerance.max(0.0);
        let mean_loss = mean
            .abs()
            .add_scalar(-mean_tolerance)
            .clamp_min(0.0)
            .powf_scalar(2.0)
            .mean();
        Some((variance_loss + mean_loss).reshape([1]))
    }

    pub(super) fn sigreg_loss_from_rho_memory_state(
        &self,
        state: &ModelState<B>,
    ) -> Option<Tensor<B, 1>> {
        if !self.latent_reasoning.sigreg.enabled
            || !matches!(
                self.latent_reasoning.sigreg.target,
                crate::config::LatentReasoningSigRegTarget::RhoMemorySlots
                    | crate::config::LatentReasoningSigRegTarget::HiddenAndRhoMemorySlots
            )
        {
            return None;
        }
        let mut total: Option<Tensor<B, 1>> = None;
        let mut components = 0usize;
        for rho in state.layers.iter().filter_map(|layer| layer.rho.as_ref()) {
            let [batch, heads, original_slots, dim] = rho.shape().dims::<4>();
            if batch == 0 || heads == 0 || original_slots < 2 || dim == 0 {
                continue;
            }
            let rho = self.sigreg_sample_rho_slots(rho.clone(), original_slots);
            let [batch, heads, slots, dim] = rho.shape().dims::<4>();
            let groups = batch * heads;
            let rows = rho.reshape([groups, slots, dim]);
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
            let denom = (groups * slots * slots.saturating_sub(1)).max(1) as f32;
            let loss = (total_sq - diag_sq).clamp_min(0.0).div_scalar(denom);
            total = Some(match total {
                Some(accumulated) => accumulated + loss,
                None => loss,
            });
            components = components.saturating_add(1);
        }
        total.map(|loss| loss.div_scalar(components.max(1) as f32))
    }

    pub(super) fn sigreg_sample_rho_slots(&self, rho: Tensor<B, 4>, slots: usize) -> Tensor<B, 4> {
        Self::sample_rho_slots_with_limit(rho, slots, self.latent_reasoning.sigreg.max_rho_slots)
    }

    pub(super) fn sample_rho_slots_with_limit(
        rho: Tensor<B, 4>,
        slots: usize,
        max_slots: usize,
    ) -> Tensor<B, 4> {
        let max_slots = max_slots.max(2);
        if slots <= max_slots {
            return rho;
        }
        let sample_slots = max_slots.min(slots);
        let denominator = sample_slots.saturating_sub(1).max(1);
        let source_span = slots.saturating_sub(1);
        let indices = (0..sample_slots)
            .map(|idx| ((idx * source_span + denominator / 2) / denominator) as i64)
            .collect::<Vec<_>>();
        let device = rho.device();
        let indices =
            Tensor::<B, 1, Int>::from_data(TensorData::new(indices, [sample_slots]), &device);
        rho.select(2, indices)
    }

    pub(super) fn normalized_rho_rows(rho: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [_batch, _heads, _slots, dim] = rho.shape().dims::<4>();
        let energy = rho
            .clone()
            .powf_scalar(2.0)
            .mean_dim(3)
            .clamp_min(1.0e-8)
            .sqrt();
        let normalized = rho / energy.clone().repeat_dim(3, dim);
        (normalized, energy)
    }

    pub(super) fn dragon_state_consistency_loss(
        &self,
        student_state: &ModelState<B>,
        teacher_state: &ModelState<B>,
    ) -> (Option<Tensor<B, 1>>, usize) {
        let config = &self.latent_reasoning.dragon_state;
        if !config.enabled {
            return (None, 0);
        }
        let rho_weight = config.rho_weight.max(0.0);
        let rho_energy_weight = config.rho_energy_weight.max(0.0);
        if rho_weight <= f32::EPSILON && rho_energy_weight <= f32::EPSILON {
            return (None, 0);
        }
        let mut total: Option<Tensor<B, 1>> = None;
        let mut components = 0usize;
        for (student_layer, teacher_layer) in student_state.layers.iter().zip(&teacher_state.layers)
        {
            let (Some(student_rho), Some(teacher_rho)) =
                (student_layer.rho.as_ref(), teacher_layer.rho.as_ref())
            else {
                continue;
            };
            let student_dims = student_rho.shape().dims::<4>();
            if student_dims != teacher_rho.shape().dims::<4>() {
                continue;
            }
            let [_batch, _heads, slots, _dim] = student_dims;
            if slots < 2 {
                continue;
            }
            let student_rho =
                Self::sample_rho_slots_with_limit(student_rho.clone(), slots, config.max_rho_slots);
            let teacher_rho =
                Self::sample_rho_slots_with_limit(teacher_rho.clone(), slots, config.max_rho_slots)
                    .detach();
            let (student_rows, student_energy) = Self::normalized_rho_rows(student_rho);
            let (teacher_rows, teacher_energy) = Self::normalized_rho_rows(teacher_rho);
            if rho_weight > f32::EPSILON {
                let row_loss = crate::train::next_latent::smooth_l1_mean(
                    student_rows,
                    teacher_rows.detach(),
                    config.smooth_l1_beta,
                )
                .mul_scalar(rho_weight);
                total = Some(match total {
                    Some(accumulated) => accumulated + row_loss,
                    None => row_loss,
                });
                components = components.saturating_add(1);
            }
            if rho_energy_weight > f32::EPSILON {
                let energy_loss = crate::train::next_latent::smooth_l1_mean(
                    student_energy,
                    teacher_energy.detach(),
                    config.smooth_l1_beta,
                )
                .mul_scalar(rho_energy_weight);
                total = Some(match total {
                    Some(accumulated) => accumulated + energy_loss,
                    None => energy_loss,
                });
                components = components.saturating_add(1);
            }
        }
        (
            total.map(|loss| loss.div_scalar(components.max(1) as f32)),
            components,
        )
    }

    pub(super) fn next_latent_auxiliary_loss(
        &self,
        hidden: Tensor<B, 3>,
        target_hidden: Tensor<B, 3>,
        clean_inputs: Tensor<B, 2, Int>,
    ) -> (Option<Tensor<B, 1>>, usize) {
        let config = &self.latent_reasoning.next_latent;
        if !config.enabled || !self.model.next_latent_transition_enabled() {
            return (None, 0);
        }
        let regression_weight = config.regression_weight.max(0.0);
        let token_kl_weight = config.token_kl_weight.max(0.0);
        if regression_weight <= f32::EPSILON && token_kl_weight <= f32::EPSILON {
            return (None, 0);
        }
        let [batch, time, dim] = hidden.shape().dims();
        if batch == 0 || time < 2 || dim == 0 {
            return (None, 0);
        }
        let max_horizon = config.horizon.min(time.saturating_sub(1));
        let mut rollout_state = hidden;
        let mut total: Option<Tensor<B, 1>> = None;
        let mut loss_components = 0usize;
        let mut transition_components = 0usize;
        for horizon_index in 0..max_horizon {
            let rollout_time = time.saturating_sub(horizon_index + 1);
            if rollout_time == 0 {
                break;
            }
            let current = rollout_state.slice([0..batch, 0..rollout_time, 0..dim]);
            let action_tokens = clean_inputs
                .clone()
                .slice([0..batch, horizon_index + 1..time]);
            let mut action_embedding = self.model.embed_tokens(action_tokens);
            if config.detach_action_embedding {
                action_embedding = action_embedding.detach();
            }
            let Some(prediction) = self
                .model
                .next_latent_prediction_from_hidden_action(current, action_embedding)
            else {
                break;
            };
            let target = target_hidden
                .clone()
                .slice([0..batch, horizon_index + 1..time, 0..dim])
                .detach();
            if regression_weight > f32::EPSILON {
                let regression = crate::train::next_latent::smooth_l1_mean(
                    prediction.clone(),
                    target.clone(),
                    config.smooth_l1_beta,
                )
                .mul_scalar(regression_weight);
                total = Some(match total {
                    Some(accumulated) => accumulated + regression,
                    None => regression,
                });
                loss_components = loss_components.saturating_add(1);
            }
            if token_kl_weight > f32::EPSILON && !self.model.uses_factorized_language_head() {
                let student_logits = self.model.logits_from_hidden(prediction.clone());
                let teacher_logits = self.model.logits_from_hidden(target).detach();
                let token_kl = crate::train::next_latent::token_kl_mean_from_logits(
                    student_logits,
                    teacher_logits,
                )
                .mul_scalar(token_kl_weight);
                total = Some(match total {
                    Some(accumulated) => accumulated + token_kl,
                    None => token_kl,
                });
                loss_components = loss_components.saturating_add(1);
            }
            rollout_state = prediction;
            transition_components = transition_components.saturating_add(1);
        }
        (
            total.map(|loss| loss.div_scalar(loss_components.max(1) as f32)),
            transition_components,
        )
    }

    pub(super) fn latent_energy_model_auxiliary_loss(
        &self,
        hidden: Tensor<B, 3>,
        target_hidden: Tensor<B, 3>,
    ) -> (Option<Tensor<B, 1>>, usize) {
        let config = &self.latent_reasoning.energy_model;
        if !config.enabled || !self.model.latent_reasoning_enabled() {
            return (None, 0);
        }
        let contrastive_weight = config.contrastive_weight.max(0.0);
        let monotonic_weight = config.monotonic_weight.max(0.0);
        let contractive_weight = config.contractive_weight.max(0.0);
        if contrastive_weight <= f32::EPSILON
            && monotonic_weight <= f32::EPSILON
            && contractive_weight <= f32::EPSILON
        {
            return (None, 0);
        }
        let Some(mut previous_energy) = self.model.latent_energy_from_hidden(hidden.clone()) else {
            return (None, 0);
        };
        let output = self.model.reason_hidden(hidden);
        if output.step_hiddens.is_empty() || output.energies.is_empty() {
            return (None, 0);
        }
        let target = target_hidden.detach();
        let negative = match self.latent_reasoning.negative_source {
            crate::config::LatentReasoningNegativeSource::InBatchAndCorruptAnswer
            | crate::config::LatentReasoningNegativeSource::TemporalShift => {
                Self::shifted_latent_negative(target.clone()).detach()
            }
        };
        let negative_energy = self.model.latent_energy_from_hidden(negative);
        let step_limit = config
            .max_rollout_steps_for_loss
            .min(output.step_hiddens.len())
            .min(output.energies.len());
        let mut total: Option<Tensor<B, 1>> = None;
        let mut components = 0usize;
        for step_index in 0..step_limit {
            let state = output
                .step_hiddens
                .get(step_index)
                .expect("step hidden")
                .clone();
            let energy = output
                .energies
                .get(step_index)
                .expect("step energy")
                .clone();
            if contrastive_weight > f32::EPSILON
                && let Some(negative_energy) = negative_energy.as_ref()
            {
                let contrastive = latent_energy_contrastive_margin_loss(
                    energy.clone(),
                    negative_energy.clone(),
                    config.margin,
                )
                .mul_scalar(contrastive_weight);
                total = Some(match total {
                    Some(accumulated) => accumulated + contrastive,
                    None => contrastive,
                });
                components = components.saturating_add(1);
            }
            if monotonic_weight > f32::EPSILON {
                let monotonic = latent_energy_monotonic_penalty(
                    previous_energy.clone(),
                    energy.clone(),
                    config.monotonic_tolerance,
                )
                .mul_scalar(monotonic_weight);
                total = Some(match total {
                    Some(accumulated) => accumulated + monotonic,
                    None => monotonic,
                });
                components = components.saturating_add(1);
            }
            if contractive_weight > f32::EPSILON {
                let contractive =
                    latent_energy_contractivity_penalty(state, target.clone(), config.trust_radius)
                        .mul_scalar(contractive_weight);
                total = Some(match total {
                    Some(accumulated) => accumulated + contractive,
                    None => contractive,
                });
                components = components.saturating_add(1);
            }
            previous_energy = energy;
        }
        (
            total.map(|loss| loss.div_scalar(components.max(1) as f32)),
            components,
        )
    }

    pub(super) fn latent_step_contract_auxiliary_loss(
        &self,
        hidden: Tensor<B, 3>,
        targets: Option<Tensor<B, 2, Int>>,
        loss_mask: Option<Tensor<B, 2, Int>>,
    ) -> (Option<Tensor<B, 1>>, usize) {
        let config = &self.latent_reasoning.step_contract;
        if !config.enabled || !self.model.latent_reasoning_enabled() {
            return (None, 0);
        }
        let ce_weight = config.ce_weight.max(0.0);
        let token_kl_weight = config.token_kl_weight.max(0.0);
        let monotonic_ce_weight = config.monotonic_ce_weight.max(0.0);
        let contractive_weight = config.contractive_weight.max(0.0);
        if ce_weight <= f32::EPSILON
            && token_kl_weight <= f32::EPSILON
            && monotonic_ce_weight <= f32::EPSILON
            && contractive_weight <= f32::EPSILON
        {
            return (None, 0);
        }

        let output = self.model.reason_hidden(hidden.clone());
        if output.step_hiddens.is_empty() {
            return (None, 0);
        }

        let step_limit = config
            .max_rollout_steps_for_loss
            .max(1)
            .min(output.step_hiddens.len());
        let mut total: Option<Tensor<B, 1>> = None;
        let mut components = 0usize;
        let mut previous_hidden = hidden.clone().detach();
        let mut previous_ce = targets.as_ref().map(|targets| {
            self.language_loss_from_hidden_for_latent_step(
                hidden.clone(),
                targets.clone(),
                loss_mask.clone(),
                0,
            )
            .detach()
        });
        let reference_logits = (token_kl_weight > f32::EPSILON
            && !self.model.uses_factorized_language_head())
        .then(|| {
            self.model
                .logits_from_hidden_for_latent_step(output.final_hidden, output.steps_used)
                .detach()
        });

        for (index, state) in output.step_hiddens.into_iter().take(step_limit).enumerate() {
            let step = index.saturating_add(1);
            let step_ce = targets.as_ref().map(|targets| {
                self.language_loss_from_hidden_for_latent_step(
                    state.clone(),
                    targets.clone(),
                    loss_mask.clone(),
                    step,
                )
            });
            if ce_weight > f32::EPSILON
                && let Some(step_ce) = step_ce.as_ref()
            {
                let component = step_ce.clone().mul_scalar(ce_weight);
                total = Some(match total {
                    Some(accumulated) => accumulated + component,
                    None => component,
                });
                components = components.saturating_add(1);
            }
            if monotonic_ce_weight > f32::EPSILON
                && let (Some(step_ce), Some(previous_ce_value)) =
                    (step_ce.as_ref(), previous_ce.as_ref())
            {
                let penalty = (step_ce.clone()
                    - previous_ce_value
                        .clone()
                        .add_scalar(config.ce_tolerance.max(0.0)))
                .clamp_min(0.0)
                .mul_scalar(monotonic_ce_weight);
                total = Some(match total {
                    Some(accumulated) => accumulated + penalty,
                    None => penalty,
                });
                components = components.saturating_add(1);
            }
            if token_kl_weight > f32::EPSILON
                && let Some(reference_logits) = reference_logits.as_ref()
            {
                let step_logits = self
                    .model
                    .logits_from_hidden_for_latent_step(state.clone(), step);
                let token_kl = crate::train::next_latent::token_kl_mean_from_logits(
                    step_logits,
                    reference_logits.clone(),
                )
                .mul_scalar(token_kl_weight);
                total = Some(match total {
                    Some(accumulated) => accumulated + token_kl,
                    None => token_kl,
                });
                components = components.saturating_add(1);
            }
            if contractive_weight > f32::EPSILON {
                let contractive = latent_energy_contractivity_penalty(
                    state.clone(),
                    previous_hidden.clone(),
                    config.trust_radius,
                )
                .mul_scalar(contractive_weight);
                total = Some(match total {
                    Some(accumulated) => accumulated + contractive,
                    None => contractive,
                });
                components = components.saturating_add(1);
            }
            previous_hidden = state.detach();
            if let Some(step_ce) = step_ce {
                previous_ce = Some(step_ce.detach());
            }
        }

        (
            total.map(|loss| loss.div_scalar(components.max(1) as f32)),
            components,
        )
    }

    pub(super) fn latent_reasoning_fallback_every_steps(&self) -> usize {
        self.latent_reasoning.every_steps.max(1)
    }

    pub(super) fn latent_reasoning_fallback_start_after_steps(&self) -> usize {
        self.latent_reasoning.constraint_balancer.start_after_steps
    }

    pub(super) fn latent_reasoning_jepa_every_steps(&self) -> usize {
        self.latent_reasoning
            .jepa_every_steps
            .unwrap_or_else(|| self.latent_reasoning_fallback_every_steps())
            .max(1)
    }

    pub(super) fn latent_reasoning_jepa_start_after_steps(&self) -> usize {
        self.latent_reasoning
            .jepa_start_after_steps
            .unwrap_or_else(|| self.latent_reasoning_fallback_start_after_steps())
    }

    pub(super) fn latent_reasoning_default_start_policy(
        &self,
    ) -> LatentReasoningAuxiliaryStartPolicy {
        if self.latent_reasoning.start_after_capability_gate_passed {
            LatentReasoningAuxiliaryStartPolicy::FixedStepAndCapabilityGate
        } else {
            LatentReasoningAuxiliaryStartPolicy::FixedStep
        }
    }

    pub(super) fn latent_reasoning_jepa_start_policy(&self) -> LatentReasoningAuxiliaryStartPolicy {
        self.latent_reasoning
            .jepa_start_policy
            .unwrap_or_else(|| self.latent_reasoning_default_start_policy())
    }

    pub(super) fn latent_reasoning_next_latent_every_steps(&self) -> usize {
        self.latent_reasoning
            .next_latent
            .every_steps
            .unwrap_or_else(|| self.latent_reasoning_fallback_every_steps())
            .max(1)
    }

    pub(super) fn latent_reasoning_next_latent_start_after_steps(&self) -> usize {
        self.latent_reasoning
            .next_latent
            .start_after_steps
            .unwrap_or_else(|| self.latent_reasoning_fallback_start_after_steps())
    }

    pub(super) fn latent_reasoning_next_latent_start_policy(
        &self,
    ) -> LatentReasoningAuxiliaryStartPolicy {
        self.latent_reasoning
            .next_latent
            .start_policy
            .unwrap_or_else(|| self.latent_reasoning_default_start_policy())
    }

    pub(super) fn latent_reasoning_dragon_state_every_steps(&self) -> usize {
        self.latent_reasoning
            .dragon_state
            .every_steps
            .unwrap_or_else(|| self.latent_reasoning_fallback_every_steps())
            .max(1)
    }

    pub(super) fn latent_reasoning_dragon_state_start_after_steps(&self) -> usize {
        self.latent_reasoning
            .dragon_state
            .start_after_steps
            .unwrap_or_else(|| self.latent_reasoning_fallback_start_after_steps())
    }

    pub(super) fn latent_reasoning_dragon_state_start_policy(
        &self,
    ) -> LatentReasoningAuxiliaryStartPolicy {
        self.latent_reasoning
            .dragon_state
            .start_policy
            .unwrap_or_else(|| self.latent_reasoning_default_start_policy())
    }

    pub(super) fn latent_reasoning_energy_model_every_steps(&self) -> usize {
        self.latent_reasoning
            .energy_model
            .every_steps
            .unwrap_or_else(|| self.latent_reasoning_fallback_every_steps())
            .max(1)
    }

    pub(super) fn latent_reasoning_energy_model_start_after_steps(&self) -> usize {
        self.latent_reasoning
            .energy_model
            .start_after_steps
            .unwrap_or_else(|| self.latent_reasoning_fallback_start_after_steps())
    }

    pub(super) fn latent_reasoning_energy_model_start_policy(
        &self,
    ) -> LatentReasoningAuxiliaryStartPolicy {
        self.latent_reasoning
            .energy_model
            .start_policy
            .unwrap_or_else(|| self.latent_reasoning_default_start_policy())
    }

    pub(super) fn latent_reasoning_step_contract_every_steps(&self) -> usize {
        self.latent_reasoning
            .step_contract
            .every_steps
            .unwrap_or_else(|| self.latent_reasoning_fallback_every_steps())
            .max(1)
    }

    pub(super) fn latent_reasoning_step_contract_start_after_steps(&self) -> usize {
        self.latent_reasoning
            .step_contract
            .start_after_steps
            .unwrap_or_else(|| self.latent_reasoning_fallback_start_after_steps())
    }

    pub(super) fn latent_reasoning_step_contract_start_policy(
        &self,
    ) -> LatentReasoningAuxiliaryStartPolicy {
        self.latent_reasoning
            .step_contract
            .start_policy
            .unwrap_or_else(|| self.latent_reasoning_default_start_policy())
    }

    pub(super) fn latent_reasoning_sigreg_every_steps(&self) -> usize {
        self.latent_reasoning
            .sigreg
            .every_steps
            .unwrap_or_else(|| self.latent_reasoning_fallback_every_steps())
            .max(1)
    }

    pub(super) fn latent_reasoning_sigreg_start_after_steps(&self) -> usize {
        self.latent_reasoning
            .sigreg
            .start_after_steps
            .unwrap_or_else(|| self.latent_reasoning_fallback_start_after_steps())
    }

    pub(super) fn latent_reasoning_sigreg_start_policy(
        &self,
    ) -> LatentReasoningAuxiliaryStartPolicy {
        self.latent_reasoning
            .sigreg
            .start_policy
            .unwrap_or_else(|| self.latent_reasoning_default_start_policy())
    }

    pub(super) fn latent_reasoning_auxiliary_scale(&self) -> Option<f32> {
        self.latent_reasoning_auxiliary_scale_for_schedule(
            self.latent_reasoning_fallback_every_steps(),
            self.latent_reasoning_fallback_start_after_steps(),
            self.latent_reasoning_default_start_policy(),
        )
    }

    pub(super) fn latent_reasoning_auxiliary_scale_for_every_steps(
        &self,
        every_steps: usize,
    ) -> Option<f32> {
        self.latent_reasoning_auxiliary_scale_for_schedule(
            every_steps,
            self.latent_reasoning_fallback_start_after_steps(),
            self.latent_reasoning_default_start_policy(),
        )
    }

    pub(super) fn latent_reasoning_auxiliary_scale_for_schedule(
        &self,
        every_steps: usize,
        start_after_steps: usize,
        start_policy: LatentReasoningAuxiliaryStartPolicy,
    ) -> Option<f32> {
        if !self.latent_reasoning.enabled {
            return None;
        }
        let requires_capability = matches!(
            start_policy,
            LatentReasoningAuxiliaryStartPolicy::CapabilityGate
                | LatentReasoningAuxiliaryStartPolicy::FixedStepAndCapabilityGate
        );
        let requires_fixed_step = matches!(
            start_policy,
            LatentReasoningAuxiliaryStartPolicy::FixedStep
                | LatentReasoningAuxiliaryStartPolicy::FixedStepAndCapabilityGate
        );
        if requires_capability
            && !self
                .latent_reasoning_capability_gate_open
                .load(Ordering::Relaxed)
        {
            return None;
        }
        let step = self.gradient_scale_step.load(Ordering::Relaxed);
        let current_step = step.saturating_add(1);
        if requires_fixed_step && start_after_steps > 0 && current_step <= start_after_steps {
            return None;
        }
        let every_steps = every_steps.max(1);
        if every_steps > 1 && !current_step.is_multiple_of(every_steps) {
            return None;
        }
        let mut aux_scale = self
            .latent_reasoning
            .constraint_balancer
            .normalized_aux_scale
            .max(0.0);
        let warmup_steps = self.latent_reasoning.constraint_balancer.warmup_steps;
        if warmup_steps > 0 {
            let warmup_start = if requires_fixed_step {
                start_after_steps
            } else {
                0
            };
            let active_step = current_step.saturating_sub(warmup_start).max(1);
            let progress = (active_step as f32 / warmup_steps as f32).min(1.0);
            aux_scale *= progress;
        }
        (aux_scale > f32::EPSILON).then_some(aux_scale)
    }

    pub(super) fn latent_rho_memory_auxiliary_loss(
        &self,
        state: &ModelState<B>,
    ) -> Option<Tensor<B, 1>> {
        let aux_scale = self.latent_reasoning_auxiliary_scale_for_schedule(
            self.latent_reasoning_sigreg_every_steps(),
            self.latent_reasoning_sigreg_start_after_steps(),
            self.latent_reasoning_sigreg_start_policy(),
        )?;
        let loss = self.sigreg_loss_from_rho_memory_state(state)?;
        crate::train::profile::record_latent_reasoning(
            0,
            0,
            0,
            0,
            0,
            1,
            self.model.latent_reasoning_config().max_steps,
        );
        Some(loss.mul_scalar(aux_scale))
    }

    pub(super) fn add_latent_rho_memory_auxiliary_loss(
        &self,
        loss: Tensor<B, 1>,
        state: &ModelState<B>,
    ) -> Tensor<B, 1> {
        self.latent_rho_memory_auxiliary_loss(state)
            .map(|aux| loss.clone() + aux)
            .unwrap_or(loss)
    }

    pub(super) fn latent_dragon_state_auxiliary_loss(
        &self,
        student_state: &ModelState<B>,
        teacher_state: Option<&ModelState<B>>,
    ) -> Option<Tensor<B, 1>> {
        let aux_scale = self.latent_reasoning_auxiliary_scale_for_schedule(
            self.latent_reasoning_dragon_state_every_steps(),
            self.latent_reasoning_dragon_state_start_after_steps(),
            self.latent_reasoning_dragon_state_start_policy(),
        )?;
        let teacher_state = teacher_state?;
        let (loss, components) = self.dragon_state_consistency_loss(student_state, teacher_state);
        if components > 0 {
            crate::train::profile::record_latent_reasoning(
                0,
                components,
                0,
                0,
                0,
                0,
                self.model.latent_reasoning_config().max_steps,
            );
        }
        loss.map(|loss| loss.mul_scalar(aux_scale))
    }

    pub(super) fn add_latent_dragon_state_auxiliary_loss(
        &self,
        loss: Tensor<B, 1>,
        student_state: &ModelState<B>,
        teacher_state: Option<&ModelState<B>>,
    ) -> Tensor<B, 1> {
        self.latent_dragon_state_auxiliary_loss(student_state, teacher_state)
            .map(|aux| loss.clone() + aux)
            .unwrap_or(loss)
    }

    pub(super) fn latent_reasoning_auxiliary_loss(
        &self,
        hidden: Tensor<B, 3>,
        clean_inputs: Tensor<B, 2, Int>,
        targets: Option<Tensor<B, 2, Int>>,
        loss_mask: Option<Tensor<B, 2, Int>>,
    ) -> Option<Tensor<B, 1>> {
        let next_latent_aux_scale = self.latent_reasoning_auxiliary_scale_for_schedule(
            self.latent_reasoning_next_latent_every_steps(),
            self.latent_reasoning_next_latent_start_after_steps(),
            self.latent_reasoning_next_latent_start_policy(),
        );
        let jepa_aux_scale = self.latent_reasoning_auxiliary_scale_for_schedule(
            self.latent_reasoning_jepa_every_steps(),
            self.latent_reasoning_jepa_start_after_steps(),
            self.latent_reasoning_jepa_start_policy(),
        );
        let energy_model_aux_scale = self.latent_reasoning_auxiliary_scale_for_schedule(
            self.latent_reasoning_energy_model_every_steps(),
            self.latent_reasoning_energy_model_start_after_steps(),
            self.latent_reasoning_energy_model_start_policy(),
        );
        let step_contract_aux_scale = self.latent_reasoning_auxiliary_scale_for_schedule(
            self.latent_reasoning_step_contract_every_steps(),
            self.latent_reasoning_step_contract_start_after_steps(),
            self.latent_reasoning_step_contract_start_policy(),
        );
        let sigreg_aux_scale = self.latent_reasoning_auxiliary_scale_for_schedule(
            self.latent_reasoning_sigreg_every_steps(),
            self.latent_reasoning_sigreg_start_after_steps(),
            self.latent_reasoning_sigreg_start_policy(),
        );
        if next_latent_aux_scale.is_none()
            && jepa_aux_scale.is_none()
            && energy_model_aux_scale.is_none()
            && step_contract_aux_scale.is_none()
            && sigreg_aux_scale.is_none()
        {
            return None;
        }
        let [batch, time, dim] = hidden.shape().dims();
        if batch == 0 || time == 0 || dim == 0 {
            let aux_scale = sigreg_aux_scale?;
            let loss = self.sigreg_loss_from_hidden(hidden);
            if loss.is_some() {
                crate::train::profile::record_latent_reasoning(
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    self.model.latent_reasoning_config().max_steps,
                );
            }
            return loss.map(|loss| loss.mul_scalar(aux_scale));
        }

        let target_hidden =
            self.latent_reasoning_target_hidden(hidden.clone(), clean_inputs.clone());
        let mut total: Option<Tensor<B, 1>> = None;
        let mut components = 0usize;
        let mut next_latent_components = 0usize;
        if let Some(next_latent_aux_scale) = next_latent_aux_scale {
            let (next_latent_loss, active_components) = self.next_latent_auxiliary_loss(
                hidden.clone(),
                target_hidden.clone(),
                clean_inputs.clone(),
            );
            next_latent_components = active_components;
            if let Some(next_latent_loss) = next_latent_loss {
                let next_latent_loss = next_latent_loss.mul_scalar(next_latent_aux_scale);
                total = Some(match total {
                    Some(accumulated) => accumulated + next_latent_loss,
                    None => next_latent_loss,
                });
                components = components.saturating_add(1);
            }
        }
        let mut jepa_components = 0usize;
        if let Some(jepa_aux_scale) = jepa_aux_scale {
            for offset in self
                .latent_reasoning
                .jepa_future_offsets
                .iter()
                .copied()
                .filter(|offset| *offset > 0 && *offset < time)
            {
                let context = hidden.clone().slice([0..batch, 0..time - offset, 0..dim]);
                let target = target_hidden
                    .clone()
                    .slice([0..batch, offset..time, 0..dim])
                    .detach();
                let prediction = self.model.latent_jepa_prediction_from_hidden(context);
                let positive_energy = (prediction.clone() - target.clone())
                    .powf_scalar(2.0)
                    .mean()
                    .reshape([1]);
                let negative = Self::shifted_latent_negative(target).detach();
                let negative_energy = (prediction - negative).powf_scalar(2.0).mean().reshape([1]);
                let margin = self.model.latent_reasoning_config().energy_margin;
                let margin_loss =
                    activation::softplus(positive_energy.clone() - negative_energy + margin, 1.0);
                let component = (positive_energy + margin_loss).mul_scalar(jepa_aux_scale);
                total = Some(match total {
                    Some(accumulated) => accumulated + component,
                    None => component,
                });
                components = components.saturating_add(1);
                jepa_components = jepa_components.saturating_add(1);
            }
        }
        let mut energy_model_components = 0usize;
        if let Some(energy_model_aux_scale) = energy_model_aux_scale {
            let (energy_model_loss, active_components) =
                self.latent_energy_model_auxiliary_loss(hidden.clone(), target_hidden.clone());
            energy_model_components = active_components;
            if let Some(energy_model_loss) = energy_model_loss {
                let energy_model_loss = energy_model_loss.mul_scalar(energy_model_aux_scale);
                total = Some(match total {
                    Some(accumulated) => accumulated + energy_model_loss,
                    None => energy_model_loss,
                });
                components = components.saturating_add(1);
            }
        }
        let mut step_contract_components = 0usize;
        if let Some(step_contract_aux_scale) = step_contract_aux_scale {
            let (step_contract_loss, active_components) =
                self.latent_step_contract_auxiliary_loss(hidden.clone(), targets, loss_mask);
            step_contract_components = active_components;
            if let Some(step_contract_loss) = step_contract_loss {
                let step_contract_loss = step_contract_loss.mul_scalar(step_contract_aux_scale);
                total = Some(match total {
                    Some(accumulated) => accumulated + step_contract_loss,
                    None => step_contract_loss,
                });
                components = components.saturating_add(1);
            }
        }
        let mut sigreg_components = 0usize;
        if let Some(sigreg_aux_scale) = sigreg_aux_scale
            && let Some(sigreg) = self.sigreg_loss_from_hidden(hidden)
        {
            let sigreg = sigreg.mul_scalar(sigreg_aux_scale);
            total = Some(match total {
                Some(accumulated) => accumulated + sigreg,
                None => sigreg,
            });
            components = components.saturating_add(1);
            sigreg_components = sigreg_components.saturating_add(1);
        }
        if components > 0 {
            crate::train::profile::record_latent_reasoning(
                next_latent_components,
                0,
                jepa_components,
                energy_model_components,
                step_contract_components,
                sigreg_components,
                self.model.latent_reasoning_config().max_steps,
            );
        }
        total.map(|loss| loss.div_scalar(components.max(1) as f32))
    }
}
