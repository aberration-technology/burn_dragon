//! Latent-reasoning step sweeps and diagnostic emission.

use super::*;

pub(super) fn latent_eval_step_sweep(training: &TrainingHyperparameters) -> Vec<usize> {
    let mut steps = training
        .latent_reasoning
        .eval_step_sweep
        .iter()
        .copied()
        .filter(|steps| *steps > 0)
        .collect::<BTreeSet<_>>();
    steps.retain(|steps| *steps > 0);
    steps.into_iter().collect()
}

pub(super) fn latent_eval_step_sweep_excluding(
    training: &TrainingHyperparameters,
    fixed_steps: Option<usize>,
) -> Vec<usize> {
    latent_eval_step_sweep(training)
        .into_iter()
        .filter(|steps| Some(*steps) != fixed_steps)
        .collect()
}

pub(super) fn fixed_latent_eval_steps<B>(model: &LanguageTrainModel<B>) -> Option<usize>
where
    B: BackendTrait,
{
    let config = model.model.latent_reasoning_config();
    (model.model.latent_reasoning_enabled()
        && !config.adaptive_halting
        && config.min_steps == config.max_steps)
        .then_some(config.max_steps)
}

pub(super) fn latent_eval_step_sweep_for_model<B>(
    training: &TrainingHyperparameters,
    model: &LanguageTrainModel<B>,
) -> Vec<usize>
where
    B: BackendTrait,
{
    latent_eval_step_sweep_excluding(training, fixed_latent_eval_steps(model))
}

pub(super) fn model_with_fixed_latent_eval_steps<B>(
    model: &LanguageTrainModel<B>,
    steps: usize,
) -> LanguageTrainModel<B>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    model
        .clone()
        .map_model(|model| model.with_fixed_latent_reasoning_steps(steps))
}

pub(super) struct LatentEvalSweep<'a, B: BackendTrait> {
    pub(super) run_name: &'a str,
    pub(super) training: &'a TrainingHyperparameters,
    pub(super) source_selection_dataset: Option<&'a Arc<Dataset>>,
    pub(super) model: &'a LanguageTrainModel<B>,
    pub(super) batch: SequenceBatch<B>,
    pub(super) eos_id: Option<i64>,
    pub(super) include_degeneracy: bool,
    pub(super) event: TrainingEventContext<'a>,
}

pub(super) fn emit_latent_eval_step_validation_sweep<B>(request: LatentEvalSweep<'_, B>)
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    let LatentEvalSweep {
        run_name,
        training,
        source_selection_dataset,
        model,
        batch,
        eos_id,
        include_degeneracy,
        event:
            TrainingEventContext {
                epoch,
                absolute_step,
                bus,
            },
    } = request;
    if !model.model.latent_reasoning_enabled() {
        return;
    }
    for steps in latent_eval_step_sweep_for_model(training, model) {
        let eval_model = model_with_fixed_latent_eval_steps(model, steps);
        let probe_tokens = if include_degeneracy {
            training.events.degeneracy_probe_tokens
        } else {
            0
        };
        let (loss, degeneracy) =
            eval_model.validation_loss_and_output_degeneracy(batch.clone(), probe_tokens, eos_id);
        let loss = mean_scalar_from_loss(loss);
        let prefix = format!("Latent Eval Steps {steps}");
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: run_name.to_string().into(),
            split: TrainingMetricSplit::Valid,
            epoch,
            step_in_epoch: 0,
            absolute_step,
            name: format!("{prefix} Teacher Forced CE"),
            value: loss,
            running_value: loss,
        });
        if let Some(degeneracy) = degeneracy {
            emit_output_degeneracy_sample_with_prefix(
                run_name,
                source_selection_dataset,
                epoch,
                absolute_step,
                &degeneracy,
                bus,
                Some(&prefix),
            );
        }
        if let Some(diagnostics) = eval_model.latent_reasoning_step_diagnostics(batch.clone()) {
            emit_latent_reasoning_step_diagnostics(
                run_name,
                epoch,
                absolute_step,
                steps,
                &diagnostics,
                bus,
            );
        }
    }
}

pub(super) fn emit_latent_reasoning_step_diagnostics(
    run_name: &str,
    epoch: usize,
    absolute_step: usize,
    steps: usize,
    diagnostics: &crate::train::steps::LatentReasoningStepDiagnostics,
    bus: &TrainingEventBus,
) {
    let prefix = format!("Latent Eval Steps {steps}");
    let event = TrainingEventContext {
        epoch,
        absolute_step,
        bus,
    };
    for (name, value) in [
        ("Raw Hidden CE", diagnostics.raw_loss),
        ("Final Hidden CE", diagnostics.final_loss),
        ("Raw Hidden Entropy Bits", diagnostics.raw_entropy_bits),
        ("Final Hidden Entropy Bits", diagnostics.final_entropy_bits),
        ("Final Delta RMS", diagnostics.final_delta_rms),
        ("Final Raw Cosine", diagnostics.final_raw_cosine),
        (
            "Best Energy Step",
            diagnostics.best_energy_step.unwrap_or(0) as f64,
        ),
    ] {
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: run_name.to_string().into(),
            split: TrainingMetricSplit::Valid,
            epoch,
            step_in_epoch: 0,
            absolute_step,
            name: format!("{prefix} {name}"),
            value,
            running_value: value,
        });
    }
    for (index, value) in diagnostics.step_loss.iter().copied().enumerate() {
        emit_latent_step_metric(run_name, &prefix, index, "CE", value, event);
    }
    for (index, value) in diagnostics.step_ce_delta.iter().copied().enumerate() {
        emit_latent_step_metric(run_name, &prefix, index, "CE Delta", value, event);
    }
    for (index, value) in diagnostics
        .step_ce_monotonic_violation_rate
        .iter()
        .copied()
        .enumerate()
    {
        emit_latent_step_metric(
            run_name,
            &prefix,
            index,
            "CE Monotonic Violation Rate",
            value,
            event,
        );
    }
    for (index, value) in diagnostics.step_entropy_bits.iter().copied().enumerate() {
        emit_latent_step_metric(run_name, &prefix, index, "Entropy Bits", value, event);
    }
    for (index, value) in diagnostics.step_delta_rms.iter().copied().enumerate() {
        emit_latent_step_metric(run_name, &prefix, index, "Delta RMS", value, event);
    }
    for (index, value) in diagnostics.step_raw_cosine.iter().copied().enumerate() {
        emit_latent_step_metric(run_name, &prefix, index, "Raw Cosine", value, event);
    }
    for (index, value) in diagnostics.step_energy_mean.iter().copied().enumerate() {
        emit_latent_step_metric(run_name, &prefix, index, "Energy Mean", value, event);
    }
    for (index, value) in diagnostics.step_energy_delta.iter().copied().enumerate() {
        emit_latent_step_metric(run_name, &prefix, index, "Energy Delta", value, event);
    }
    for (index, value) in diagnostics
        .step_energy_monotonic_violation_rate
        .iter()
        .copied()
        .enumerate()
    {
        emit_latent_step_metric(
            run_name,
            &prefix,
            index,
            "Energy Monotonic Violation Rate",
            value,
            event,
        );
    }
}

pub(super) fn emit_latent_step_metric(
    run_name: &str,
    prefix: &str,
    index: usize,
    suffix: &str,
    value: f64,
    event: TrainingEventContext<'_>,
) {
    let TrainingEventContext {
        epoch,
        absolute_step,
        bus,
    } = event;
    let _ = bus.send_metric_sample(TrainingMetricSample {
        run_id: run_name.to_string().into(),
        split: TrainingMetricSplit::Valid,
        epoch,
        step_in_epoch: 0,
        absolute_step,
        name: format!("{prefix} Step {} {suffix}", index.saturating_add(1)),
        value,
        running_value: value,
    });
}
