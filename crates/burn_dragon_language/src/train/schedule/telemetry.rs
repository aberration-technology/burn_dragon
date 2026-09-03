//! Dataset and source-selection telemetry emission.

use super::*;

pub(super) fn dataset_eos_id(dataset: Option<&Arc<Dataset>>) -> Option<i64> {
    dataset
        .and_then(|dataset| dataset.tokenizer().eos_id())
        .map(i64::from)
}

pub(super) fn source_selection_telemetry_due<B>(
    env: &TrainEnvironment<'_, B>,
    absolute_step: usize,
) -> bool
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    source_selection_telemetry_due_for(
        env.training,
        env.source_selection_dataset.as_ref(),
        absolute_step,
    )
}

pub(super) fn source_selection_telemetry_due_for(
    training: &TrainingHyperparameters,
    source_selection_dataset: Option<&Arc<Dataset>>,
    absolute_step: usize,
) -> bool {
    let every = training.events.source_selection_every_steps.max(1);
    if !absolute_step.is_multiple_of(every) {
        return false;
    }
    source_selection_dataset.is_some_and(|dataset| dataset.uses_live_source_selection())
}

pub(super) fn train_loss_metric_name(training: &TrainingHyperparameters) -> &'static str {
    if training.tbptt_persist_across_steps {
        METRIC_STREAM_WARM_LOSS
    } else {
        METRIC_LOSS
    }
}

pub(super) fn emit_source_selection_telemetry<B>(
    env: &TrainEnvironment<'_, B>,
    absolute_step: usize,
    loss: f64,
    bus: &TrainingEventBus,
) where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    emit_source_selection_telemetry_sample(
        env.run_name,
        env.source_selection_dataset.as_ref(),
        absolute_step,
        loss,
        bus,
    );
}

pub(super) fn emit_source_selection_telemetry_sample(
    run_name: &str,
    source_selection_dataset: Option<&Arc<Dataset>>,
    absolute_step: usize,
    loss: f64,
    bus: &TrainingEventBus,
) {
    let Some(dataset) = source_selection_dataset else {
        return;
    };
    let recorded_snapshot = dataset.record_source_selection_loss(absolute_step, loss as f32);
    let loss = recorded_snapshot.as_ref().map(|_| loss as f32);
    let snapshot =
        recorded_snapshot.or_else(|| dataset.source_selection_snapshot_at_step(absolute_step));
    let Some(snapshot) = snapshot else {
        return;
    };
    let _ = bus.send_source_selection_sample(
        crate::train::events::source_selection_sample_from_snapshot(
            run_name.to_string(),
            absolute_step,
            loss,
            &snapshot,
        ),
    );
}

pub(super) fn emit_source_selection_capability_feedback_batch(
    run_name: &str,
    source_selection_dataset: Option<&Arc<Dataset>>,
    absolute_step: usize,
    feedback: &[burn_dragon_universality::RuliadCapabilityFeedback],
    bus: &TrainingEventBus,
) {
    let Some(dataset) = source_selection_dataset else {
        return;
    };
    let Some(snapshot) =
        dataset.record_ruliad_capability_feedback_batch_at_step(feedback, Some(absolute_step))
    else {
        return;
    };
    let _ = bus.send_source_selection_sample(
        crate::train::events::source_selection_sample_from_snapshot(
            run_name.to_string(),
            absolute_step,
            None,
            &snapshot,
        ),
    );
}

pub(super) fn emit_output_degeneracy<B>(
    env: &TrainEnvironment<'_, B>,
    epoch: usize,
    absolute_step: usize,
    stats: &crate::train::steps::OutputDegeneracyStats,
    bus: &TrainingEventBus,
) where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    emit_output_degeneracy_sample(
        env.run_name,
        env.source_selection_dataset.as_ref(),
        epoch,
        absolute_step,
        stats,
        bus,
    );
}

pub(super) fn emit_output_degeneracy_sample(
    run_name: &str,
    source_selection_dataset: Option<&Arc<Dataset>>,
    epoch: usize,
    absolute_step: usize,
    stats: &crate::train::steps::OutputDegeneracyStats,
    bus: &TrainingEventBus,
) {
    emit_output_degeneracy_sample_with_prefix(
        run_name,
        source_selection_dataset,
        epoch,
        absolute_step,
        stats,
        bus,
        None,
    );
}

pub(super) fn emit_output_degeneracy_sample_with_prefix(
    run_name: &str,
    source_selection_dataset: Option<&Arc<Dataset>>,
    epoch: usize,
    absolute_step: usize,
    stats: &crate::train::steps::OutputDegeneracyStats,
    bus: &TrainingEventBus,
    metric_prefix: Option<&str>,
) {
    if metric_prefix.is_none() {
        let _ = bus.send_output_degeneracy_sample(OutputDegeneracySample {
            run_id: run_name.to_string().into(),
            split: TrainingMetricSplit::Valid,
            epoch,
            absolute_step,
            token_count: stats.token_count,
            entropy_bits: stats.entropy_bits,
            mean_max_probability: stats.mean_max_probability,
            argmax_unique_fraction: stats.argmax_unique_fraction,
            eos_fraction: stats.eos_fraction,
            repetition_fraction: stats.repetition_fraction,
            distinct_1_fraction: stats.distinct_1_fraction,
            distinct_2_fraction: stats.distinct_2_fraction,
            period_2_fraction: stats.period_2_fraction,
            period_3_fraction: stats.period_3_fraction,
            max_period_2_to_16_fraction: stats.max_period_2_to_16_fraction,
            max_period_2_to_64_fraction: stats.max_period_2_to_64_fraction,
            dominant_period_2_to_64: stats.dominant_period_2_to_64,
            generated_preview: decode_degeneracy_preview(source_selection_dataset, stats),
        });
    }
    for (name, value) in [
        ("Output Entropy Bits", stats.entropy_bits),
        ("Output Mean Max Probability", stats.mean_max_probability),
        (
            "Output Argmax Unique Fraction",
            stats.argmax_unique_fraction,
        ),
        ("Output EOS Fraction", stats.eos_fraction),
        ("Output Repetition Fraction", stats.repetition_fraction),
        ("Output Distinct-1 Fraction", stats.distinct_1_fraction),
        ("Output Distinct-2 Fraction", stats.distinct_2_fraction),
        ("Output Period-2 Fraction", stats.period_2_fraction),
        ("Output Period-3 Fraction", stats.period_3_fraction),
        (
            "Output Max Period-2..16 Fraction",
            stats.max_period_2_to_16_fraction,
        ),
        (
            "Output Max Period-2..64 Fraction",
            stats.max_period_2_to_64_fraction,
        ),
    ] {
        let metric_name = metric_prefix
            .map(|prefix| format!("{prefix} {name}"))
            .unwrap_or_else(|| name.to_string());
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: run_name.to_string().into(),
            split: TrainingMetricSplit::Valid,
            epoch,
            step_in_epoch: 0,
            absolute_step,
            name: metric_name,
            value,
            running_value: value,
        });
    }
}

pub(super) fn decode_degeneracy_preview(
    dataset: Option<&Arc<Dataset>>,
    stats: &crate::train::steps::OutputDegeneracyStats,
) -> Option<String> {
    if stats.generated_tokens.is_empty() {
        return None;
    }
    let prompt_tokens = stats
        .prompt_tokens
        .iter()
        .copied()
        .take(160)
        .collect::<Vec<_>>();
    let generated_tokens = stats
        .generated_tokens
        .iter()
        .copied()
        .take(160)
        .collect::<Vec<_>>();
    let prompt = decode_degeneracy_tokens(dataset, &prompt_tokens);
    let generated = decode_degeneracy_tokens(dataset, &generated_tokens);
    let preview = if prompt.trim().is_empty() {
        generated
    } else {
        format!(
            "prompt(period{}={:.3}): {}\n--- generated(period{}={:.3}) ---\n{}",
            stats.prompt_dominant_period_2_to_64,
            stats.prompt_max_period_2_to_64_fraction,
            prompt,
            stats.dominant_period_2_to_64,
            stats.max_period_2_to_64_fraction,
            generated
        )
    };
    Some(preview.chars().take(2_000).collect())
}

pub(super) fn decode_degeneracy_tokens(dataset: Option<&Arc<Dataset>>, tokens: &[i64]) -> String {
    if tokens.is_empty() {
        return String::new();
    }
    dataset
        .and_then(|dataset| dataset.decode_ruliad_payload_tokens(tokens, true))
        .filter(|preview| !preview.trim().is_empty())
        .or_else(|| dataset.map(|dataset| dataset.decode(tokens)))
        .filter(|preview| !preview.trim().is_empty())
        .unwrap_or_else(|| {
            tokens
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        })
}

pub(super) fn emit_continual_backprop_telemetry<B>(
    env: &TrainEnvironment<'_, B>,
    optimizer: &crate::train::continual_backprop::LanguageOptimizer<B>,
    epoch: usize,
    absolute_step: usize,
    bus: &TrainingEventBus,
    last_emitted_optimizer_step: &mut usize,
) where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let Some(telemetry) = optimizer.continual_backprop_telemetry() else {
        return;
    };
    if telemetry.optimizer_step == 0 || telemetry.optimizer_step == *last_emitted_optimizer_step {
        return;
    }
    if telemetry.replacement_count == 0
        && !absolute_step.is_multiple_of(env.training.events.continual_backprop_every_steps.max(1))
    {
        return;
    }
    *last_emitted_optimizer_step = telemetry.optimizer_step;
    let _ = bus.send_continual_backprop_sample(ContinualBackpropSample {
        run_id: env.run_name.to_string().into(),
        epoch: Some(epoch),
        absolute_step,
        optimizer_step: telemetry.optimizer_step,
        feature_count: telemetry.feature_count,
        eligible_count: telemetry.eligible_count,
        replacement_count: telemetry.replacement_count,
        replacement_budget: telemetry.replacement_budget as f64,
        lr_multiplier: telemetry.lr_multiplier as f64,
        replacement_rate_scale: telemetry.replacement_rate_scale as f64,
        effective_replacement_rate: telemetry.effective_replacement_rate as f64,
        effective_max_replacements_per_interval: telemetry.effective_max_replacements_per_interval,
        paused: telemetry.paused,
        pause_reason: telemetry.pause_reason,
        utility_min: telemetry.utility_min as f64,
        utility_mean: telemetry.utility_mean as f64,
        utility_max: telemetry.utility_max as f64,
        age_mean: telemetry.age_mean as f64,
        age_max: telemetry.age_max as f64,
        batch_stat_samples: telemetry.batch_stat_samples,
        activation_abs_mean: telemetry.activation_abs_mean as f64,
        zero_utility_fraction: telemetry.zero_utility_fraction as f64,
    });
}

pub(super) fn emit_predictive_context_routing_metrics<B>(
    env: &TrainEnvironment<'_, B>,
    epoch: usize,
    step_in_epoch: usize,
    absolute_step: usize,
    known_contexts: usize,
    decision: &crate::train::PredictiveContextRoutingDecision,
    bus: &TrainingEventBus,
) where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    if !decision.probed && !decision.created && decision.replaced.is_none() {
        return;
    }
    let metrics = [
        (
            "Predictive Context Index",
            decision.identity.context_index as f64,
        ),
        (
            "Predictive Context Generation",
            decision.identity.generation as f64,
        ),
        ("Predictive Context Count", known_contexts as f64),
        ("Predictive Context Created", f64::from(decision.created)),
        (
            "Predictive Context Replaced",
            f64::from(decision.replaced.is_some()),
        ),
        (
            "Predictive Context Novelty Deferred",
            f64::from(decision.novelty_deferred),
        ),
        (
            "Predictive Context Probe Tokens",
            decision.probe_tokens as f64,
        ),
    ];
    for (name, value) in metrics {
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: env.run_name.to_string().into(),
            split: TrainingMetricSplit::Train,
            epoch,
            step_in_epoch,
            absolute_step,
            name: name.to_string(),
            value,
            running_value: value,
        });
    }
    if let Some(loss) = decision.selected_loss {
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: env.run_name.to_string().into(),
            split: TrainingMetricSplit::Train,
            epoch,
            step_in_epoch,
            absolute_step,
            name: "Predictive Context Selected Loss".to_string(),
            value: loss,
            running_value: loss,
        });
    }
    if let Some(loss) = decision.reserve_loss {
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: env.run_name.to_string().into(),
            split: TrainingMetricSplit::Train,
            epoch,
            step_in_epoch,
            absolute_step,
            name: "Predictive Context Reserve Loss".to_string(),
            value: loss,
            running_value: loss,
        });
    }
    if let Some(supported) = decision.reserve_supported_novelty {
        let value = f64::from(supported);
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: env.run_name.to_string().into(),
            split: TrainingMetricSplit::Train,
            epoch,
            step_in_epoch,
            absolute_step,
            name: "Predictive Context Reserve Supports Novelty".to_string(),
            value,
            running_value: value,
        });
    }
}

pub(super) fn emit_predictive_coding_telemetry<B>(
    env: &TrainEnvironment<'_, B>,
    epoch: usize,
    step_in_epoch: usize,
    absolute_step: usize,
    optimizer_step: usize,
    bus: &TrainingEventBus,
) where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    if !env.training.predictive_coding.enabled {
        let _ = crate::train::profile::take_predictive_coding();
        return;
    }
    let snapshot = crate::train::profile::take_predictive_coding();
    if !snapshot.has_activity() {
        return;
    }
    let energy_delta = snapshot.energy_delta_mean();
    let grad_norm_mean = snapshot.grad_norm_mean();
    let grad_norm_max = snapshot.grad_norm_max();
    let delta_rms_mean = snapshot.delta_rms_mean();
    let clip_fraction_mean = snapshot.clip_fraction_mean();
    let amortization_loss = snapshot.amortization_loss_mean();
    let (observation_contract, deployment_aligned) = match (
        env.training.predictive_coding.observation_contract,
        env.training.predictive_coding.parameter_update,
    ) {
        (
            PredictiveCodingObservationContract::ObservedPrefix,
            PredictiveCodingParameterUpdate::Optimizer,
        ) => ("observed_prefix_amortized", true),
        (PredictiveCodingObservationContract::ObservedPrefix, _) => {
            ("observed_prefix_online_state_control", false)
        }
        (PredictiveCodingObservationContract::OracleNextTokenNegativeControl, _) => {
            ("oracle_next_token_negative_control", false)
        }
    };
    let _ = bus.send_predictive_coding_sample(PredictiveCodingSample {
        run_id: env.run_name.to_string().into(),
        epoch: Some(epoch),
        absolute_step,
        optimizer_step,
        learning_contract: "recurrent_state_replay_auxiliary".to_string(),
        execution_contract_version: burn_pc::PcExecutionContract::CURRENT_VERSION,
        activity_derivative_contract: "global_autodiff".to_string(),
        parameter_derivative_contract: "global_autodiff".to_string(),
        global_autodiff_graph: true,
        observation_contract: observation_contract.to_string(),
        deployment_aligned,
        chunks_seen: snapshot.chunks_seen,
        chunks_corrected: snapshot.chunks_corrected,
        inference_steps: snapshot.inference_steps,
        dual_steps: 0,
        skipped_empty_state: snapshot.skipped_empty_state,
        factors: 0,
        local_vjp_calls: 0,
        temporal_state_vjp_calls: 0,
        fused_temporal_vjp_calls: 0,
        temporal_credit_mode: "recurrent_replay_auxiliary".to_string(),
        temporal_window_chunks: 1,
        global_backward_calls: usize::from(snapshot.chunks_corrected > 0),
        gradient_tensors: 0,
        direct_forward_updates: 0,
        feedback_parameter_updates: 0,
        adjoint_teacher_updates: 0,
        adjoint_local_updates: 0,
        adjoint_calibration_samples: 0,
        adjoint_calibration_loss: None,
        adjoint_cosine_alignment: None,
        adjoint_prediction_teacher_norm_ratio: None,
        adjoint_update_rms: None,
        local_parameter_update_intents: 0,
        parameter_updates: usize::from(matches!(
            env.training.predictive_coding.parameter_update,
            PredictiveCodingParameterUpdate::Optimizer
        )),
        terminal_factor_kind: "next_token_replay".to_string(),
        structured_terminal_steps: 0,
        structured_terminal_skipped_steps: 0,
        structured_terminal_groups: 0,
        structured_terminal_rows: 0,
        energy_before: snapshot.energy_before_mean(),
        energy_after: snapshot.energy_after_mean(),
        energy_delta,
        grad_norm_mean,
        grad_norm_max,
        delta_rms_mean,
        clip_fraction_mean,
        constraint_rms: None,
        dual_rms: None,
        composite_signal_rms: None,
        amortization_components: snapshot.amortization_components,
        amortization_loss,
        elapsed_ms: snapshot.elapsed_ms(),
    });
    for (name, value) in [
        ("Predictive Coding Energy Delta", energy_delta),
        ("Predictive Coding Grad Norm Mean", grad_norm_mean),
        ("Predictive Coding Grad Norm Max", grad_norm_max),
        ("Predictive Coding Delta RMS", delta_rms_mean),
        ("Predictive Coding Clip Fraction", clip_fraction_mean),
        ("Predictive Coding Amortization Loss", amortization_loss),
        (
            "Predictive Coding Amortization Components",
            Some(snapshot.amortization_components as f64),
        ),
        (
            "Predictive Coding Corrected Fraction",
            (snapshot.chunks_seen > 0)
                .then(|| snapshot.chunks_corrected as f64 / snapshot.chunks_seen as f64),
        ),
        ("Predictive Coding Elapsed MS", Some(snapshot.elapsed_ms())),
    ] {
        let Some(value) = value.filter(|value| value.is_finite()) else {
            continue;
        };
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: env.run_name.to_string().into(),
            split: TrainingMetricSplit::Train,
            epoch,
            step_in_epoch,
            absolute_step,
            name: name.to_string(),
            value,
            running_value: value,
        });
    }
}

pub(super) fn emit_latent_reasoning_telemetry<B>(
    env: &TrainEnvironment<'_, B>,
    epoch: usize,
    step_in_epoch: usize,
    absolute_step: usize,
    bus: &TrainingEventBus,
) where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    if !env.training.latent_reasoning.enabled {
        let _ = crate::train::profile::take_latent_reasoning();
        return;
    }
    let snapshot = crate::train::profile::take_latent_reasoning();
    if !snapshot.has_activity() {
        return;
    }
    for (name, value) in [
        (
            "Latent Reasoning Loss Calls",
            Some(snapshot.loss_calls as f64),
        ),
        (
            "Latent Reasoning NextLat Components",
            Some(snapshot.next_latent_components as f64),
        ),
        (
            "Latent Reasoning Dragon State Components",
            Some(snapshot.dragon_state_components as f64),
        ),
        (
            "Latent Reasoning JEPA Components",
            Some(snapshot.jepa_components as f64),
        ),
        (
            "Latent Reasoning Energy Model Components",
            Some(snapshot.energy_model_components as f64),
        ),
        (
            "Latent Reasoning Step Contract Components",
            Some(snapshot.step_contract_components as f64),
        ),
        (
            "Latent Reasoning SIGReg Components",
            Some(snapshot.sigreg_components as f64),
        ),
        (
            "Latent Reasoning Configured Steps",
            snapshot.configured_steps_mean(),
        ),
    ] {
        let Some(value) = value.filter(|value| value.is_finite()) else {
            continue;
        };
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: env.run_name.to_string().into(),
            split: TrainingMetricSplit::Train,
            epoch,
            step_in_epoch,
            absolute_step,
            name: name.to_string(),
            value,
            running_value: value,
        });
    }
}
