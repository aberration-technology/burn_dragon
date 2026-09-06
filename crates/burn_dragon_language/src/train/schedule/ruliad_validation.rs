//! Ruliad source-weighted validation and policy scoring.

use super::*;

pub(super) fn run_source_weighted_validation<B>(
    env: &TrainEnvironment<'_, B>,
    valid_model: &LanguageTrainModel<ValidBackend<B>>,
    steps_per_epoch: usize,
    batch_size: usize,
    context_routing: Option<&crate::train::PredictiveContextRoutingRuntime<B>>,
    event: TrainingEventContext<'_>,
) -> Result<Option<f64>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let TrainingEventContext {
        epoch,
        absolute_step: training_absolute_step,
        bus,
    } = event;
    let requested_batches = env.training.events.source_weighted_validation_batches;
    if requested_batches == 0 {
        return Ok(None);
    }
    let Some(dataset) = env.source_selection_dataset.as_ref() else {
        return Ok(None);
    };
    if !dataset.uses_live_source_selection() {
        return Ok(None);
    }

    let base_absolute_step = epoch.saturating_sub(1).saturating_mul(steps_per_epoch);
    let mut total = 0.0;
    let mut count = 0usize;
    for batch_index in 0..requested_batches {
        let absolute_step = base_absolute_step.saturating_add(batch_index);
        let Some(batch) = dataset.sample_source_weighted_validation_batch::<ValidBackend<B>>(
            epoch,
            absolute_step,
            batch_size,
            env.summary_event_token_ids.as_deref(),
            env.device,
        ) else {
            break;
        };
        let loss = if let Some(routing) = context_routing {
            let (loss, _, _) = routing.validation_loss(valid_model, batch, 0, None)?;
            mean_scalar_from_loss(loss)
        } else {
            let output = valid_model.step(batch);
            let loss_value: LossValue<ValidBackend<B>> = output.adapt();
            mean_scalar_from_loss(loss_value.value())
        };
        count += 1;
        total += loss;
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: env.run_name.to_string().into(),
            split: TrainingMetricSplit::Valid,
            epoch,
            step_in_epoch: count,
            absolute_step: training_absolute_step,
            name: "Source Weighted Loss".to_string(),
            value: loss,
            running_value: total / count as f64,
        });
    }

    Ok((count > 0).then_some(total / count as f64))
}

pub(super) struct RuliadCorrectnessValidation<'a, B: BackendTrait> {
    pub(super) run_name: &'a str,
    pub(super) run_dir: &'a Path,
    pub(super) training: &'a TrainingHyperparameters,
    pub(super) dataset: Option<&'a Arc<Dataset>>,
    pub(super) model: &'a LanguageTrainModel<B>,
    pub(super) training_batch_size: usize,
    pub(super) device: &'a B::Device,
    pub(super) output_degeneracy: Option<&'a crate::train::steps::OutputDegeneracyStats>,
    pub(super) event: TrainingEventContext<'a>,
}

#[derive(Clone, Debug)]
pub(super) struct RuliadCorrectnessValidationResult {
    pub(super) free_run: burn_dragon_universality::RuliadEvalReport,
    pub(super) policy_context_free_run: Option<burn_dragon_universality::RuliadEvalReport>,
    pub(super) closed_loop_policy: Option<RuliadPolicyRolloutProbeResult>,
    pub(super) constrained_policy: Option<RuliadCorrectnessConstrainedPolicyResult>,
}

pub(super) fn ruliad_constrained_policy_probe_due(
    training: &TrainingHyperparameters,
    epoch: usize,
) -> bool {
    training.ruliad_policy_probe.enabled
        && epoch.is_multiple_of(training.ruliad_policy_probe.every_epochs.max(1))
}

pub(super) fn ruliad_closed_loop_policy_probe_due(
    training: &TrainingHyperparameters,
    epoch: usize,
) -> bool {
    training.ruliad_policy_probe.enabled
        && epoch.is_multiple_of(
            training
                .ruliad_policy_probe
                .effective_closed_loop_every_epochs()
                .max(1),
        )
}

pub(super) fn run_ruliad_correctness_validation<B>(
    request: RuliadCorrectnessValidation<'_, B>,
) -> Result<Option<RuliadCorrectnessValidationResult>>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    let RuliadCorrectnessValidation {
        run_name,
        run_dir,
        training,
        dataset,
        model,
        training_batch_size,
        device,
        output_degeneracy,
        event:
            TrainingEventContext {
                epoch,
                absolute_step,
                bus,
            },
    } = request;
    let requested_items = training.events.ruliad_correctness_probe_items;
    let max_new_tokens = training.events.ruliad_correctness_probe_tokens;
    if requested_items == 0 || max_new_tokens == 0 {
        return Ok(None);
    }
    let every = training.events.ruliad_correctness_probe_every_epochs.max(1);
    if !epoch.is_multiple_of(every) {
        return Ok(None);
    }
    if model.model.uses_factorized_language_head() {
        return Ok(None);
    }
    let Some(dataset) = dataset else {
        return Ok(None);
    };

    let base_absolute_step = absolute_step;
    let panel = crate::dataset::resolve_ruliad_validation_panel(
        dataset,
        training,
        epoch,
        base_absolute_step,
    )?;
    if let Some(fingerprint) = panel.fingerprint_sha256.as_deref() {
        info!(
            "ruliad validation panel: sha256={fingerprint} base_items={} policy_items={}",
            panel.base_items.len(),
            panel.policy_items.len()
        );
    }
    let probe_items = panel.base_items;
    let policy_items = panel.policy_items;
    if probe_items.is_empty() {
        return Ok(None);
    }

    let training_serialization_items = panel.training_serialization_items;
    let reuse_base_for_training_serialization =
        !training_serialization_items.is_empty() && training_serialization_items == probe_items;
    if !training_serialization_items.is_empty() && !reuse_base_for_training_serialization {
        let _ = run_ruliad_correctness_validation_for_items(
            run_name,
            run_dir,
            dataset,
            model,
            epoch,
            base_absolute_step,
            device,
            training,
            &training_serialization_items,
            training_batch_size,
            "ruliad_training_serialization_probe",
            "ruliad_correctness_training_serialization",
            Some("Ruliad Training Serialization"),
            None,
            bus,
            RuliadProbeDecodeMode::FreeRun,
        )?;
    }

    let closed_loop_policy_probe_due = ruliad_closed_loop_policy_probe_due(training, epoch);
    let policy_probe_result = if closed_loop_policy_probe_due {
        Some(run_ruliad_policy_rollout_probe(
            run_name,
            dataset,
            model,
            epoch,
            base_absolute_step,
            device,
            training,
            &policy_items,
            bus,
        )?)
    } else {
        None
    };
    let base_report = run_ruliad_correctness_validation_for_items(
        run_name,
        run_dir,
        dataset,
        model,
        epoch,
        base_absolute_step,
        device,
        training,
        &probe_items,
        training_batch_size,
        "ruliad_validation_probe",
        "ruliad_correctness",
        None,
        output_degeneracy,
        bus,
        RuliadProbeDecodeMode::FreeRun,
    )?;
    if reuse_base_for_training_serialization {
        emit_reused_ruliad_correctness_validation(
            run_name,
            epoch,
            base_absolute_step,
            &base_report,
            output_degeneracy,
            bus,
        );
    }
    if training.events.source_selection_capability_feedback {
        let feedback = merge_ruliad_policy_capability_feedback(
            crate::dataset::ruliad_capability_feedback_from_report(&base_report),
            training.ruliad_policy_probe.enabled,
            policy_probe_result.as_ref(),
        );
        emit_source_selection_capability_feedback_batch(
            run_name,
            Some(dataset),
            base_absolute_step,
            &feedback,
            bus,
        );
    }
    let constrained_policy_due = ruliad_constrained_policy_probe_due(training, epoch);
    let policy_context_free_run = if constrained_policy_due && !policy_items.is_empty() {
        let policy_context_items = ruliad_policy_context_probe_items(
            dataset,
            &policy_items,
            &training.ruliad_policy_probe,
        )?;
        Some(run_ruliad_correctness_validation_for_items(
            run_name,
            run_dir,
            dataset,
            model,
            epoch,
            base_absolute_step,
            device,
            training,
            &policy_context_items,
            training_batch_size,
            "ruliad_policy_context_probe",
            "ruliad_correctness_policy_context",
            Some("Ruliad Policy Context"),
            None,
            bus,
            RuliadProbeDecodeMode::FreeRun,
        )?)
    } else {
        None
    };
    let constrained_policy = if constrained_policy_due {
        Some(run_ruliad_correctness_constrained_policy_probe(
            run_name,
            dataset,
            model,
            epoch,
            base_absolute_step,
            device,
            training,
            &policy_items,
            bus,
            RuliadPolicyControlMode::Disabled,
        )?)
    } else {
        None
    };
    if training.events.ruliad_contract_probe_enabled {
        let _ = run_ruliad_correctness_validation_for_items(
            run_name,
            run_dir,
            dataset,
            model,
            epoch,
            base_absolute_step,
            device,
            training,
            &probe_items,
            training_batch_size,
            "ruliad_validation_prompt_schema_probe",
            "ruliad_correctness_prompt_schema",
            Some("Ruliad Prompt Schema"),
            None,
            bus,
            RuliadProbeDecodeMode::PromptSchemaContract,
        )?;
        let _ = run_ruliad_correctness_validation_for_items(
            run_name,
            run_dir,
            dataset,
            model,
            epoch,
            base_absolute_step,
            device,
            training,
            &probe_items,
            training_batch_size,
            "ruliad_validation_contract_probe",
            "ruliad_correctness_contract",
            Some("Ruliad Contract"),
            None,
            bus,
            RuliadProbeDecodeMode::FixedContract,
        )?;
    }
    if model.model.latent_reasoning_enabled() {
        for steps in latent_eval_step_sweep_for_model(training, model) {
            let eval_model = model_with_fixed_latent_eval_steps(model, steps);
            let metric_prefix = format!("Ruliad Eval Steps {steps}");
            let probe_name = format!("ruliad_correctness_eval_steps_{steps}");
            let _ = run_ruliad_correctness_validation_for_items(
                run_name,
                run_dir,
                dataset,
                &eval_model,
                epoch,
                base_absolute_step,
                device,
                training,
                &probe_items,
                training_batch_size,
                "ruliad_validation_probe",
                &probe_name,
                Some(&metric_prefix),
                None,
                bus,
                RuliadProbeDecodeMode::FreeRun,
            )?;
        }
    }
    Ok(Some(RuliadCorrectnessValidationResult {
        free_run: base_report,
        policy_context_free_run,
        closed_loop_policy: policy_probe_result,
        constrained_policy,
    }))
}

pub(super) fn run_routed_ruliad_correctness_validation<B>(
    request: RuliadCorrectnessValidation<'_, B>,
    router: &crate::train::PredictiveContextValidationRouter<B>,
) -> Result<Option<RuliadCorrectnessValidationResult>>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    let RuliadCorrectnessValidation {
        run_name,
        run_dir,
        training,
        dataset,
        model,
        training_batch_size,
        device,
        output_degeneracy,
        event:
            TrainingEventContext {
                epoch,
                absolute_step,
                bus,
            },
    } = request;
    let requested_items = training.events.ruliad_correctness_probe_items;
    if requested_items == 0 || training.events.ruliad_correctness_probe_tokens == 0 {
        return Ok(None);
    }
    let every = training.events.ruliad_correctness_probe_every_epochs.max(1);
    if !epoch.is_multiple_of(every) || model.model.uses_factorized_language_head() {
        return Ok(None);
    }
    let Some(dataset) = dataset else {
        return Ok(None);
    };
    let panel =
        crate::dataset::resolve_ruliad_validation_panel(dataset, training, epoch, absolute_step)?;
    if let Some(fingerprint) = panel.fingerprint_sha256.as_deref() {
        info!(
            "ruliad validation panel: sha256={fingerprint} base_items={} policy_items={}",
            panel.base_items.len(),
            panel.policy_items.len()
        );
    }
    let probe_items = panel.base_items;
    if probe_items.is_empty() {
        return Ok(None);
    }

    let serialization_items = panel.training_serialization_items;
    let reuse_training_serialization = training.sequence_state_probe.enabled
        && !serialization_items.is_empty()
        && serialization_items == probe_items;
    if training.sequence_state_probe.enabled
        && !serialization_items.is_empty()
        && serialization_items != probe_items
    {
        let _ = evaluate_ruliad_correctness_validation_for_items_core(
            Some(run_name),
            Some(run_dir),
            dataset,
            model,
            epoch,
            absolute_step,
            device,
            training,
            &serialization_items,
            training_batch_size,
            "ruliad_training_serialization_probe",
            Some("ruliad_correctness_training_serialization"),
            Some("Ruliad Training Serialization"),
            None,
            Some(bus),
            RuliadProbeDecodeMode::FreeRun,
            Some(router),
        )?;
    }

    let base = evaluate_ruliad_correctness_validation_for_items_core(
        Some(run_name),
        Some(run_dir),
        dataset,
        model,
        epoch,
        absolute_step,
        device,
        training,
        &probe_items,
        training_batch_size,
        "ruliad_validation_probe",
        Some("ruliad_correctness"),
        None,
        output_degeneracy,
        Some(bus),
        RuliadProbeDecodeMode::FreeRun,
        Some(router),
    )?;
    if reuse_training_serialization {
        emit_reused_ruliad_correctness_validation(
            run_name,
            epoch,
            absolute_step,
            &base.report,
            output_degeneracy,
            bus,
        );
    }
    if training.events.source_selection_capability_feedback {
        emit_source_selection_capability_feedback_batch(
            run_name,
            Some(dataset),
            absolute_step,
            &crate::dataset::ruliad_capability_feedback_from_report(&base.report),
            bus,
        );
    }
    let _ = bus.send_metric_sample(TrainingMetricSample {
        run_id: run_name.to_string().into(),
        split: TrainingMetricSplit::Valid,
        epoch,
        step_in_epoch: 0,
        absolute_step,
        name: "Ruliad Correctness Routed Subnetwork".to_string(),
        value: 1.0,
        running_value: 1.0,
    });
    Ok(Some(RuliadCorrectnessValidationResult {
        free_run: base.report,
        policy_context_free_run: None,
        closed_loop_policy: None,
        constrained_policy: None,
    }))
}

pub(super) fn emit_reused_ruliad_correctness_validation(
    run_name: &str,
    epoch: usize,
    absolute_step: usize,
    report: &burn_dragon_universality::RuliadEvalReport,
    output_degeneracy: Option<&crate::train::steps::OutputDegeneracyStats>,
    bus: &TrainingEventBus,
) {
    const PROBE_NAME: &str = "ruliad_correctness_training_serialization";
    const METRIC_PREFIX: &str = "Ruliad Training Serialization";
    emit_ruliad_correctness_metrics_with_labels(RuliadCorrectnessMetrics {
        identity: RuliadProbeIdentity {
            run_name,
            epoch,
            absolute_step,
            probe_name: PROBE_NAME,
        },
        report,
        bus,
        metric_prefix: Some(METRIC_PREFIX),
        output_degeneracy,
        examples: &[],
        schema_alignment: RuliadAnswerSchemaAlignmentSummary::default(),
        completion_degeneracy: None,
        generation_budget: None,
    });
    let _ = bus.send_metric_sample(TrainingMetricSample {
        run_id: run_name.to_string().into(),
        split: TrainingMetricSplit::Valid,
        epoch,
        step_in_epoch: 0,
        absolute_step,
        name: format!("{METRIC_PREFIX} Probe Reused Canonical Evaluation"),
        value: 1.0,
        running_value: 1.0,
    });
    eprintln!(
        "ruliad correctness probe reused run={run_name} epoch={epoch} probe={PROBE_NAME} source=ruliad_correctness items={}",
        report.item_count,
    );
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct RuliadPolicyRolloutProbeSummary {
    pub(super) items: usize,
    pub(super) solved: usize,
    pub(super) steps: usize,
    pub(super) valid_actions: usize,
    pub(super) invalid_actions: usize,
    pub(super) repeated_states: usize,
    pub(super) backtracks: usize,
    pub(super) scored_states: usize,
    pub(super) scored_actions: usize,
    pub(super) top1_expert_actions: usize,
    pub(super) frontier_exhaustions: usize,
    pub(super) solved_goals: usize,
    pub(super) total_goals: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct RuliadPolicyRolloutProbeResult {
    pub(super) summary: RuliadPolicyRolloutProbeSummary,
    pub(super) difficulty_summaries: BTreeMap<usize, RuliadPolicyRolloutProbeSummary>,
    pub(super) source_summaries: BTreeMap<String, RuliadPolicyRolloutProbeSummary>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct RuliadCorrectnessConstrainedPolicySummary {
    pub(super) items: usize,
    pub(super) equivalent_top1: usize,
    pub(super) preferred_top1: usize,
    pub(super) equivalent_nll_sum: f64,
    pub(super) valid_invalid_margin_sum: f64,
    pub(super) valid_invalid_margin_items: usize,
    pub(super) canonical_items: usize,
    pub(super) canonical_equivalent_top1: usize,
    pub(super) canonical_preferred_top1: usize,
    pub(super) canonical_equivalent_nll_sum: f64,
    pub(super) canonical_valid_invalid_margin_sum: f64,
    pub(super) canonical_valid_invalid_margin_items: usize,
    pub(super) worst_presentation_items: usize,
    pub(super) worst_presentation_equivalent_top1: usize,
    pub(super) worst_presentation_equivalent_nll_sum: f64,
    pub(super) worst_presentation_valid_invalid_margin_sum: f64,
    pub(super) worst_presentation_valid_invalid_margin_items: usize,
    pub(super) complete_orbit_items: usize,
    pub(super) presentation_rows: usize,
    pub(super) presentation_equivalent_top1: usize,
    pub(super) presentation_preferred_top1: usize,
    pub(super) orbit_js_divergence_sum: f64,
    pub(super) orbit_top1_consensus_fraction_sum: f64,
    pub(super) context_swap_items: usize,
    pub(super) context_swap_equivalent_top1: usize,
    pub(super) context_swap_equivalent_nll_sum: f64,
    pub(super) context_swap_top1_changes: usize,
    pub(super) context_swap_equivalent_probability_drop_sum: f64,
    pub(super) context_swap_js_divergence_sum: f64,
    pub(super) counterfactual_target_items: usize,
    pub(super) counterfactual_target_equivalent_top1: usize,
    pub(super) counterfactual_target_equivalent_nll_sum: f64,
    pub(super) counterfactual_target_top1_changes: usize,
    pub(super) counterfactual_target_equivalent_probability_gain_sum: f64,
    pub(super) counterfactual_target_js_divergence_sum: f64,
    pub(super) elapsed_ms: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct RuliadCorrectnessConstrainedPolicyResult {
    pub(super) summary: RuliadCorrectnessConstrainedPolicySummary,
    pub(super) difficulty_summaries: BTreeMap<usize, RuliadCorrectnessConstrainedPolicySummary>,
    pub(super) source_summaries: BTreeMap<String, RuliadCorrectnessConstrainedPolicySummary>,
    pub(super) structured_decode: Option<RuliadStructuredPolicyEvaluation>,
    pub(super) controls: Option<RuliadPolicyControlEvaluation>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct RuliadConstrainedActionScore {
    pub(super) equivalent_top1: bool,
    pub(super) preferred_top1: bool,
    pub(super) equivalent_nll: f64,
    pub(super) valid_invalid_margin: Option<f64>,
}

pub(super) struct RuliadCorrectnessConstrainedPolicyJob {
    pub(super) difficulty_level: usize,
    pub(super) source_label: String,
    pub(super) presentations: Vec<RuliadPolicyActionPresentation>,
    pub(super) prompt_contexts: Vec<RuliadPolicyActionPromptContext>,
    pub(super) base_context: Option<RuliadPolicyActionPromptContext>,
    pub(super) selected_index: usize,
    pub(super) equivalent_indices: Vec<usize>,
}

pub(super) fn update_ruliad_correctness_constrained_summaries(
    result: &mut RuliadCorrectnessConstrainedPolicyResult,
    job: &RuliadCorrectnessConstrainedPolicyJob,
    mut update: impl FnMut(&mut RuliadCorrectnessConstrainedPolicySummary),
) {
    update(&mut result.summary);
    update(
        result
            .difficulty_summaries
            .entry(job.difficulty_level)
            .or_default(),
    );
    update(
        result
            .source_summaries
            .entry(job.source_label.clone())
            .or_default(),
    );
}

#[derive(Clone)]
pub(super) struct RuliadPolicyActionPromptContext {
    pub(super) problem: burn_dragon_universality::ruliad::RuliadProofProblem,
    pub(super) actions: burn_dragon_universality::ruliad::RuliadProofActionSet,
}

#[derive(Clone)]
pub(super) struct RuliadPolicyActionPresentation {
    pub(super) rotation: usize,
    pub(super) prompt_tokens: Vec<i64>,
    pub(super) candidate_tokens: Vec<Vec<i64>>,
    pub(super) answer_contract: burn_dragon_universality::ruliad::RuliadProofActionAnswerContract,
}

pub(super) fn ruliad_policy_action_presentations(
    actions: &burn_dragon_universality::ruliad::RuliadProofActionSet,
    symmetry: crate::config::RuliadProofPolicyCandidateSymmetry,
    presentation_index: usize,
) -> Result<
    Vec<(
        usize,
        burn_dragon_universality::ruliad::RuliadProofActionSet,
    )>,
> {
    crate::train::ruliad_policy::candidate_presentation_rotations(
        symmetry,
        actions.selected_index,
        actions.candidates.len(),
        presentation_index,
    )?
    .into_iter()
    .map(|rotation| Ok((rotation, actions.rotate_left(rotation)?)))
    .collect()
}

pub(super) fn apply_ruliad_policy_probe_candidate_symmetry(
    actions: burn_dragon_universality::ruliad::RuliadProofActionSet,
    symmetry: crate::config::RuliadProofPolicyCandidateSymmetry,
    presentation_index: usize,
) -> Result<burn_dragon_universality::ruliad::RuliadProofActionSet> {
    ruliad_policy_action_presentations(&actions, symmetry, presentation_index)?
        .into_iter()
        .next()
        .map(|(_, actions)| actions)
        .ok_or_else(|| anyhow!("proof-action symmetry produced no presentation"))
}

pub(super) fn record_ruliad_correctness_constrained_scores(
    summary: &mut RuliadCorrectnessConstrainedPolicySummary,
    job: &RuliadCorrectnessConstrainedPolicyJob,
    scores: &[f32],
) {
    let Some(score) = ruliad_correctness_constrained_score(job, scores) else {
        return;
    };
    summary.items = summary.items.saturating_add(1);
    summary.equivalent_top1 = summary
        .equivalent_top1
        .saturating_add(usize::from(score.equivalent_top1));
    summary.preferred_top1 = summary
        .preferred_top1
        .saturating_add(usize::from(score.preferred_top1));
    summary.equivalent_nll_sum += score.equivalent_nll;
    if let Some(margin) = score.valid_invalid_margin {
        summary.valid_invalid_margin_sum += margin;
        summary.valid_invalid_margin_items = summary.valid_invalid_margin_items.saturating_add(1);
    }
}

pub(super) fn categorical_log_probability_js_divergence(
    left: &[f32],
    right: &[f32],
) -> Option<f64> {
    if left.is_empty()
        || left.len() != right.len()
        || left.iter().chain(right).any(|value| !value.is_finite())
    {
        return None;
    }
    let mut left_probabilities = left
        .iter()
        .map(|value| f64::from(*value).exp())
        .collect::<Vec<_>>();
    let mut right_probabilities = right
        .iter()
        .map(|value| f64::from(*value).exp())
        .collect::<Vec<_>>();
    let left_sum = left_probabilities.iter().sum::<f64>();
    let right_sum = right_probabilities.iter().sum::<f64>();
    if !left_sum.is_finite() || !right_sum.is_finite() || left_sum <= 0.0 || right_sum <= 0.0 {
        return None;
    }
    for probability in &mut left_probabilities {
        *probability /= left_sum;
    }
    for probability in &mut right_probabilities {
        *probability /= right_sum;
    }
    Some(
        left_probabilities
            .iter()
            .zip(&right_probabilities)
            .map(|(left, right)| {
                let midpoint = 0.5 * (left + right);
                let left_kl = if *left > 0.0 {
                    left * (left / midpoint).ln()
                } else {
                    0.0
                };
                let right_kl = if *right > 0.0 {
                    right * (right / midpoint).ln()
                } else {
                    0.0
                };
                0.5 * (left_kl + right_kl)
            })
            .sum(),
    )
}

pub(super) fn record_ruliad_correctness_context_swap(
    summary: &mut RuliadCorrectnessConstrainedPolicySummary,
    job: &RuliadCorrectnessConstrainedPolicyJob,
    original_scores: &[f32],
    swapped_scores: &[f32],
) {
    let Some(original) = ruliad_correctness_constrained_score(job, original_scores) else {
        return;
    };
    let Some(swapped) = ruliad_correctness_constrained_score(job, swapped_scores) else {
        return;
    };
    let Some(js_divergence) =
        categorical_log_probability_js_divergence(original_scores, swapped_scores)
    else {
        return;
    };
    let Some(original_top1) = crate::train::ruliad_policy::best_candidate_index(original_scores)
    else {
        return;
    };
    let Some(swapped_top1) = crate::train::ruliad_policy::best_candidate_index(swapped_scores)
    else {
        return;
    };
    summary.context_swap_items = summary.context_swap_items.saturating_add(1);
    summary.context_swap_equivalent_top1 = summary
        .context_swap_equivalent_top1
        .saturating_add(usize::from(swapped.equivalent_top1));
    summary.context_swap_equivalent_nll_sum += swapped.equivalent_nll;
    summary.context_swap_top1_changes = summary
        .context_swap_top1_changes
        .saturating_add(usize::from(original_top1 != swapped_top1));
    summary.context_swap_equivalent_probability_drop_sum +=
        (-original.equivalent_nll).exp() - (-swapped.equivalent_nll).exp();
    summary.context_swap_js_divergence_sum += js_divergence;
}

pub(super) fn record_ruliad_correctness_counterfactual_target(
    summary: &mut RuliadCorrectnessConstrainedPolicySummary,
    counterfactual_job: &RuliadCorrectnessConstrainedPolicyJob,
    original_scores: &[f32],
    counterfactual_scores: &[f32],
) {
    let Some(before) = ruliad_correctness_constrained_score(counterfactual_job, original_scores)
    else {
        return;
    };
    let Some(after) =
        ruliad_correctness_constrained_score(counterfactual_job, counterfactual_scores)
    else {
        return;
    };
    let Some(js_divergence) =
        categorical_log_probability_js_divergence(original_scores, counterfactual_scores)
    else {
        return;
    };
    let Some(original_top1) = crate::train::ruliad_policy::best_candidate_index(original_scores)
    else {
        return;
    };
    let Some(counterfactual_top1) =
        crate::train::ruliad_policy::best_candidate_index(counterfactual_scores)
    else {
        return;
    };
    summary.counterfactual_target_items = summary.counterfactual_target_items.saturating_add(1);
    summary.counterfactual_target_equivalent_top1 = summary
        .counterfactual_target_equivalent_top1
        .saturating_add(usize::from(after.equivalent_top1));
    summary.counterfactual_target_equivalent_nll_sum += after.equivalent_nll;
    summary.counterfactual_target_top1_changes = summary
        .counterfactual_target_top1_changes
        .saturating_add(usize::from(original_top1 != counterfactual_top1));
    summary.counterfactual_target_equivalent_probability_gain_sum +=
        (-after.equivalent_nll).exp() - (-before.equivalent_nll).exp();
    summary.counterfactual_target_js_divergence_sum += js_divergence;
}

pub(super) fn proof_action_set_with_swapped_state(
    original: &burn_dragon_universality::ruliad::RuliadProofActionSet,
    donor: &burn_dragon_universality::ruliad::RuliadProofActionSet,
) -> burn_dragon_universality::ruliad::RuliadProofActionSet {
    let mut swapped = original.clone();
    swapped.current = donor.current.clone();
    swapped.target = donor.target.clone();
    swapped
}

pub(super) fn context_swapped_action_requests(
    dataset: &Dataset,
    jobs: &[RuliadCorrectnessConstrainedPolicyJob],
    prompt_context: crate::config::RuliadProofPolicyPromptContext,
) -> Result<Vec<crate::train::ruliad_policy::EncodedRuliadProofActionRequest>> {
    if jobs.len() < 2 {
        return Ok(Vec::new());
    }
    jobs.iter()
        .enumerate()
        .map(|(job_index, job)| {
            if job.prompt_contexts.len() != job.presentations.len() {
                return Err(anyhow!("proof-action context metadata is incomplete"));
            }
            let donor = (1..jobs.len())
                .map(|offset| &jobs[(job_index + offset) % jobs.len()])
                .find(|donor| {
                    donor.prompt_contexts.len() == job.prompt_contexts.len()
                        && donor.prompt_contexts.iter().zip(&job.prompt_contexts).any(
                            |(donor, original)| {
                                donor.actions.current != original.actions.current
                                    || donor.actions.target != original.actions.target
                            },
                        )
                })
                .ok_or_else(|| anyhow!("proof-action context swap has no distinct state donor"))?;
            let presentations = job
                .presentations
                .iter()
                .zip(&job.prompt_contexts)
                .zip(&donor.prompt_contexts)
                .map(|((presentation, original_context), donor_context)| {
                    let counterfactual_actions = proof_action_set_with_swapped_state(
                        &original_context.actions,
                        &donor_context.actions,
                    );
                    let prompt = crate::train::ruliad_policy::ruliad_proof_policy_prompt(
                        prompt_context,
                        &original_context.problem,
                        &counterfactual_actions,
                    )?;
                    let prompt_tokens: Vec<i64> = dataset
                        .encode_ruliad_payload_tokens(&prompt)
                        .ok_or_else(|| anyhow!("Ruliad dataset cannot encode context-swap prompt"))?
                        .into_iter()
                        .map(i64::from)
                        .collect();
                    Ok(
                        crate::train::ruliad_policy::EncodedRuliadProofActionPresentation {
                            rotation: presentation.rotation,
                            original_prompt_token_count: prompt_tokens.len(),
                            prompt_tokens,
                            candidate_tokens: presentation.candidate_tokens.clone(),
                        },
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(
                crate::train::ruliad_policy::EncodedRuliadProofActionRequest {
                    presentations,
                    answer_contract: job
                        .presentations
                        .first()
                        .map(|presentation| presentation.answer_contract)
                        .ok_or_else(|| anyhow!("proof-action context-swap job is empty"))?,
                },
            )
        })
        .collect()
}

pub(super) fn encoded_ruliad_policy_action_request(
    job: &RuliadCorrectnessConstrainedPolicyJob,
) -> Result<crate::train::ruliad_policy::EncodedRuliadProofActionRequest> {
    let answer_contract = job
        .presentations
        .first()
        .map(|presentation| presentation.answer_contract)
        .ok_or_else(|| anyhow!("proof-action scoring job has no presentations"))?;
    if job
        .presentations
        .iter()
        .any(|presentation| presentation.answer_contract != answer_contract)
    {
        return Err(anyhow!(
            "proof-action scoring job mixes incompatible answer contracts"
        ));
    }
    Ok(
        crate::train::ruliad_policy::EncodedRuliadProofActionRequest {
            answer_contract,
            presentations: job
                .presentations
                .iter()
                .map(|presentation| {
                    crate::train::ruliad_policy::EncodedRuliadProofActionPresentation {
                        rotation: presentation.rotation,
                        original_prompt_token_count: presentation.prompt_tokens.len(),
                        prompt_tokens: presentation.prompt_tokens.clone(),
                        candidate_tokens: presentation.candidate_tokens.clone(),
                    }
                })
                .collect(),
        },
    )
}

fn encode_ruliad_policy_candidate_tokens(
    dataset: &Dataset,
    answer: &str,
    contract: burn_dragon_universality::ruliad::RuliadProofActionAnswerContract,
) -> Option<Vec<i64>> {
    let mut tokens = dataset
        .encode_ruliad_payload_tokens(answer)?
        .into_iter()
        .map(i64::from)
        .collect::<Vec<_>>();
    if contract == burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep
        && let Some(stop_token_id) = dataset.ruliad_document_end_token_id().map(i64::from)
        && tokens.last().copied() != Some(stop_token_id)
    {
        tokens.push(stop_token_id);
    }
    (!tokens.is_empty()).then_some(tokens)
}

pub(super) fn counterfactual_target_action_jobs(
    dataset: &Dataset,
    jobs: &[RuliadCorrectnessConstrainedPolicyJob],
    prompt_context: crate::config::RuliadProofPolicyPromptContext,
) -> Result<Vec<(usize, RuliadCorrectnessConstrainedPolicyJob)>> {
    let mut counterfactual_jobs = Vec::with_capacity(jobs.len());
    for (job_index, job) in jobs.iter().enumerate() {
        let Some(base_context) = job.base_context.as_ref() else {
            continue;
        };
        let Some(candidate_index) = crate::train::ruliad_policy::counterfactual_candidate_indices(
            &base_context.actions,
            1,
            base_context
                .actions
                .selected_index
                .saturating_add(job_index)
                .saturating_add(1),
        )
        .into_iter()
        .next() else {
            continue;
        };
        let (counterfactual_problem, counterfactual_actions) =
            burn_dragon_universality::ruliad::counterfactual_proof_action_target(
                &base_context.problem,
                &base_context.actions,
                candidate_index,
            )?;
        let mut presentations = Vec::with_capacity(job.presentations.len());
        let mut prompt_contexts = Vec::with_capacity(job.presentations.len());
        for presentation in &job.presentations {
            let presented_actions = counterfactual_actions.rotate_left(presentation.rotation)?;
            let prompt = crate::train::ruliad_policy::ruliad_proof_policy_prompt(
                prompt_context,
                &counterfactual_problem,
                &presented_actions,
            )?;
            let prompt_tokens = dataset
                .encode_ruliad_payload_tokens(&prompt)
                .ok_or_else(|| {
                    anyhow!("Ruliad dataset cannot encode counterfactual-target prompt")
                })?
                .into_iter()
                .map(i64::from)
                .collect();
            presentations.push(RuliadPolicyActionPresentation {
                rotation: presentation.rotation,
                prompt_tokens,
                candidate_tokens: presentation.candidate_tokens.clone(),
                answer_contract: presentation.answer_contract,
            });
            prompt_contexts.push(RuliadPolicyActionPromptContext {
                problem: counterfactual_problem.clone(),
                actions: presented_actions,
            });
        }
        let selected_index = counterfactual_actions.selected_index;
        let equivalent_indices = counterfactual_actions.equivalent_indices.clone();
        counterfactual_jobs.push((
            job_index,
            RuliadCorrectnessConstrainedPolicyJob {
                difficulty_level: job.difficulty_level,
                source_label: job.source_label.clone(),
                presentations,
                prompt_contexts,
                base_context: Some(RuliadPolicyActionPromptContext {
                    problem: counterfactual_problem,
                    actions: counterfactual_actions,
                }),
                selected_index,
                equivalent_indices,
            },
        ));
    }
    Ok(counterfactual_jobs)
}

pub(super) fn ruliad_correctness_constrained_score(
    job: &RuliadCorrectnessConstrainedPolicyJob,
    scores: &[f32],
) -> Option<RuliadConstrainedActionScore> {
    let top1 = crate::train::ruliad_policy::best_candidate_index(scores)?;
    if scores.len()
        != job
            .presentations
            .first()
            .map(|presentation| presentation.candidate_tokens.len())
            .unwrap_or_default()
        || job.equivalent_indices.is_empty()
    {
        return None;
    }
    let equivalent_probability = job
        .equivalent_indices
        .iter()
        .filter_map(|index| scores.get(*index))
        .map(|score| score.exp() as f64)
        .sum::<f64>()
        .clamp(1.0e-12, 1.0);
    let best_equivalent = job
        .equivalent_indices
        .iter()
        .filter_map(|index| scores.get(*index))
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let best_invalid = scores
        .iter()
        .copied()
        .enumerate()
        .filter(|(index, _)| !job.equivalent_indices.contains(index))
        .map(|(_, score)| score)
        .fold(f32::NEG_INFINITY, f32::max);

    Some(RuliadConstrainedActionScore {
        equivalent_top1: job.equivalent_indices.contains(&top1),
        preferred_top1: top1 == job.selected_index,
        equivalent_nll: -equivalent_probability.ln(),
        valid_invalid_margin: (best_equivalent.is_finite() && best_invalid.is_finite())
            .then(|| f64::from(best_equivalent - best_invalid)),
    })
}

pub(super) fn record_ruliad_correctness_orbit_diagnostics(
    summary: &mut RuliadCorrectnessConstrainedPolicySummary,
    job: &RuliadCorrectnessConstrainedPolicyJob,
    orbit: &crate::train::ruliad_policy::SemanticActionOrbitSummary,
) {
    let presentation_scores = orbit
        .presentation_log_probs
        .iter()
        .filter_map(|(rotation, scores)| {
            ruliad_correctness_constrained_score(job, scores).map(|score| (*rotation, score))
        })
        .collect::<Vec<_>>();
    if presentation_scores.is_empty() {
        return;
    }
    summary.presentation_rows = summary
        .presentation_rows
        .saturating_add(presentation_scores.len());
    summary.presentation_equivalent_top1 = summary.presentation_equivalent_top1.saturating_add(
        presentation_scores
            .iter()
            .filter(|(_, score)| score.equivalent_top1)
            .count(),
    );
    summary.presentation_preferred_top1 = summary.presentation_preferred_top1.saturating_add(
        presentation_scores
            .iter()
            .filter(|(_, score)| score.preferred_top1)
            .count(),
    );
    summary.complete_orbit_items = summary
        .complete_orbit_items
        .saturating_add(usize::from(orbit.complete_cyclic_orbit));
    summary.orbit_js_divergence_sum += orbit.js_divergence;
    summary.orbit_top1_consensus_fraction_sum += orbit.top1_consensus_fraction;

    if let Some((_, canonical)) = presentation_scores
        .iter()
        .find(|(rotation, _)| *rotation == 0)
    {
        summary.canonical_items = summary.canonical_items.saturating_add(1);
        summary.canonical_equivalent_top1 = summary
            .canonical_equivalent_top1
            .saturating_add(usize::from(canonical.equivalent_top1));
        summary.canonical_preferred_top1 = summary
            .canonical_preferred_top1
            .saturating_add(usize::from(canonical.preferred_top1));
        summary.canonical_equivalent_nll_sum += canonical.equivalent_nll;
        if let Some(margin) = canonical.valid_invalid_margin {
            summary.canonical_valid_invalid_margin_sum += margin;
            summary.canonical_valid_invalid_margin_items = summary
                .canonical_valid_invalid_margin_items
                .saturating_add(1);
        }
    }

    summary.worst_presentation_items = summary.worst_presentation_items.saturating_add(1);
    summary.worst_presentation_equivalent_top1 = summary
        .worst_presentation_equivalent_top1
        .saturating_add(usize::from(
            presentation_scores
                .iter()
                .all(|(_, score)| score.equivalent_top1),
        ));
    summary.worst_presentation_equivalent_nll_sum += presentation_scores
        .iter()
        .map(|(_, score)| score.equivalent_nll)
        .fold(f64::NEG_INFINITY, f64::max);
    if let Some(worst_margin) = presentation_scores
        .iter()
        .filter_map(|(_, score)| score.valid_invalid_margin)
        .min_by(f64::total_cmp)
    {
        summary.worst_presentation_valid_invalid_margin_sum += worst_margin;
        summary.worst_presentation_valid_invalid_margin_items = summary
            .worst_presentation_valid_invalid_margin_items
            .saturating_add(1);
    }
}

/// Materialize free-generation rows from the exact state representation used by the typed proof
/// policy. This keeps policy selection, autoregressive rendering, and verifier scoring on one
/// conditional contract while preserving the separate document-prompt probes.
pub(super) fn ruliad_policy_context_probe_items(
    dataset: &Dataset,
    probe_items: &[crate::dataset::RuliadValidationProbeItem],
    config: &crate::config::RuliadPolicyProbeConfig,
) -> Result<Vec<crate::dataset::RuliadValidationProbeItem>> {
    probe_items
        .iter()
        .map(|probe| {
            let Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
                problem,
                certificate,
                proof_step_index,
                action_answer_contract,
                task: burn_dragon_universality::ruliad::RuliadTaskKind::SelectProofAction,
                ..
            }) = probe.item.spec.as_ref()
            else {
                return Err(anyhow!(
                    "policy-context generation panel contains a non-proof-action item"
                ));
            };
            let actions = burn_dragon_universality::ruliad::oracle_proof_action_set(
                problem,
                certificate,
                proof_step_index.unwrap_or_default(),
                config.candidates,
            )?;
            let prompt = crate::train::ruliad_policy::ruliad_proof_policy_prompt(
                config.prompt_context,
                problem,
                &actions,
            )?;
            let answer_contract = match config.scoring {
                crate::config::RuliadProofPolicyScoring::CompletionLikelihood => {
                    *action_answer_contract
                }
                crate::config::RuliadProofPolicyScoring::SemanticEnergy
                | crate::config::RuliadProofPolicyScoring::ResidualEnergy => {
                    burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep
                }
            };
            let expected_answer = burn_dragon_universality::ruliad::proof_action_answer(
                &actions,
                actions.selected_index,
                answer_contract,
            )?;
            let prompt_tokens = dataset
                .encode_ruliad_payload_tokens(&prompt)
                .ok_or_else(|| anyhow!("failed to encode the proof-policy prompt"))?
                .into_iter()
                .map(i64::from)
                .collect::<Vec<_>>();
            if prompt_tokens.is_empty() {
                return Err(anyhow!("proof-policy prompt encoded to no tokens"));
            }

            let mut item = probe.item.clone();
            item.prompt = prompt;
            item.expected_answer = expected_answer;
            if let Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
                action_presentation_rotation,
                action_candidate_count,
                action_answer_contract,
                ..
            }) = item.spec.as_mut()
            {
                *action_presentation_rotation = Some(0);
                *action_candidate_count = Some(config.candidates);
                *action_answer_contract = answer_contract;
            }
            Ok(crate::dataset::RuliadValidationProbeItem {
                item,
                prompt_tokens,
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_ruliad_correctness_constrained_policy_probe<B>(
    run_name: &str,
    dataset: &Dataset,
    model: &LanguageTrainModel<B>,
    epoch: usize,
    absolute_step: usize,
    device: &B::Device,
    training: &TrainingHyperparameters,
    probe_items: &[crate::dataset::RuliadValidationProbeItem],
    bus: &TrainingEventBus,
    control_mode: RuliadPolicyControlMode,
) -> Result<RuliadCorrectnessConstrainedPolicyResult>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    let config = training.ruliad_policy_probe;
    let started = burn_dragon_time::Instant::now();
    let mut jobs = Vec::<RuliadCorrectnessConstrainedPolicyJob>::new();
    let mut structured_items = Vec::new();
    for (probe_index, probe) in probe_items.iter().enumerate() {
        let Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
            problem,
            certificate,
            proof_step_index,
            action_answer_contract,
            task: burn_dragon_universality::RuliadTaskKind::SelectProofAction,
            ..
        }) = probe.item.spec.as_ref()
        else {
            continue;
        };
        let actions = burn_dragon_universality::ruliad::oracle_proof_action_set(
            problem,
            certificate,
            proof_step_index.unwrap_or_default(),
            config.candidates,
        )?;
        let action_presentations =
            ruliad_policy_action_presentations(&actions, config.candidate_symmetry, probe_index)?;
        let encoded_presentations = action_presentations
            .into_iter()
            .map(|(rotation, presented_actions)| {
                let scoring_contract = match config.scoring {
                    crate::config::RuliadProofPolicyScoring::CompletionLikelihood => {
                        *action_answer_contract
                    }
                    crate::config::RuliadProofPolicyScoring::SemanticEnergy
                    | crate::config::RuliadProofPolicyScoring::ResidualEnergy => {
                        burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep
                    }
                };
                let prompt_text = crate::train::ruliad_policy::ruliad_proof_policy_prompt(
                    config.prompt_context,
                    problem,
                    &presented_actions,
                )
                .ok()?;
                let prompt_tokens = dataset.encode_ruliad_payload_tokens(&prompt_text)?;
                let candidate_tokens = (0..presented_actions.candidates.len())
                    .map(|candidate_index| {
                        let answer = burn_dragon_universality::ruliad::proof_action_answer(
                            &presented_actions,
                            candidate_index,
                            scoring_contract,
                        )
                        .ok()?;
                        encode_ruliad_policy_candidate_tokens(
                            dataset,
                            &answer,
                            scoring_contract,
                        )
                    })
                    .collect::<Option<Vec<_>>>()?;
                Some((
                    RuliadPolicyActionPresentation {
                        rotation,
                        prompt_tokens: prompt_tokens.into_iter().map(i64::from).collect(),
                        candidate_tokens,
                        answer_contract: scoring_contract,
                    },
                    RuliadPolicyActionPromptContext {
                        problem: problem.clone(),
                        actions: presented_actions,
                    },
                ))
            })
            .collect::<Option<Vec<_>>>();
        let Some(encoded_presentations) = encoded_presentations else {
            continue;
        };
        let (presentations, prompt_contexts) = encoded_presentations.into_iter().unzip();
        let answer_contract = match action_answer_contract {
            burn_dragon_universality::RuliadProofActionAnswerContract::PresentationIndex => {
                "action_index"
            }
            burn_dragon_universality::RuliadProofActionAnswerContract::SemanticStep => {
                "proof_action_step"
            }
        };
        jobs.push(RuliadCorrectnessConstrainedPolicyJob {
            difficulty_level: probe.item.difficulty_level.unwrap_or_default(),
            source_label: burn_dragon_universality::ruliad_source_capability_label(
                &probe.item.family,
                &probe.item.task_kind,
                probe.item.difficulty_level.unwrap_or_default(),
                answer_contract,
            ),
            presentations,
            prompt_contexts,
            base_context: Some(RuliadPolicyActionPromptContext {
                problem: problem.clone(),
                actions: actions.clone(),
            }),
            selected_index: actions.selected_index,
            equivalent_indices: actions.equivalent_indices,
        });
        let mut structured_item = probe.item.clone();
        if let Some(burn_dragon_universality::RuliadSampleSpec::FormalProof {
            action_candidate_count,
            ..
        }) = structured_item.spec.as_mut()
        {
            *action_candidate_count = Some(config.candidates);
        }
        structured_items.push(structured_item);
    }

    let mut result = RuliadCorrectnessConstrainedPolicyResult::default();
    let requests = jobs
        .iter()
        .map(encoded_ruliad_policy_action_request)
        .collect::<Result<Vec<_>>>()?;
    let swapped_requests = context_swapped_action_requests(dataset, &jobs, config.prompt_context)?;
    let counterfactual_jobs =
        counterfactual_target_action_jobs(dataset, &jobs, config.prompt_context)?;
    let counterfactual_requests = counterfactual_jobs
        .iter()
        .map(|(_, job)| encoded_ruliad_policy_action_request(job))
        .collect::<Result<Vec<_>>>()?;
    let original_request_count = requests.len();
    let swapped_request_count = swapped_requests.len();
    let counterfactual_request_count = counterfactual_requests.len();
    let controls = match control_mode {
        RuliadPolicyControlMode::Disabled => Vec::new(),
        RuliadPolicyControlMode::Checkpoint => no_context_action_requests(dataset, &requests)?,
    };
    let mut scoring_requests = requests;
    scoring_requests.extend(swapped_requests);
    scoring_requests.extend(counterfactual_requests);
    scoring_requests.extend(controls);
    let mut decisions =
        crate::train::ruliad_policy::select_ruliad_proof_actions_batch_with_contract(
            &model.model,
            &scoring_requests,
            config.scoring_batch_rows.max(1),
            config.scoring,
            config.normalization,
            device,
        )?;
    let mut auxiliary_decisions = if decisions.len() > original_request_count {
        decisions.split_off(original_request_count)
    } else {
        Vec::new()
    };
    let mut counterfactual_decisions = if auxiliary_decisions.len() > swapped_request_count {
        auxiliary_decisions.split_off(swapped_request_count)
    } else {
        Vec::new()
    };
    let control_decisions = if counterfactual_decisions.len() > counterfactual_request_count {
        counterfactual_decisions.split_off(counterfactual_request_count)
    } else {
        Vec::new()
    };
    let swapped_decisions = auxiliary_decisions;
    if control_mode == RuliadPolicyControlMode::Checkpoint {
        result.controls = Some(evaluate_ruliad_policy_controls(
            &structured_items,
            &jobs,
            &decisions,
            &control_decisions,
        )?);
    }
    for (index, job) in jobs.iter().enumerate() {
        let Some(decision) = decisions.get(index) else {
            continue;
        };
        update_ruliad_correctness_constrained_summaries(&mut result, job, |summary| {
            record_ruliad_correctness_constrained_scores(
                summary,
                job,
                &decision.orbit.averaged_log_probs,
            );
            record_ruliad_correctness_orbit_diagnostics(summary, job, &decision.orbit);
        });
        if let Some(swapped) = swapped_decisions.get(index) {
            update_ruliad_correctness_constrained_summaries(&mut result, job, |summary| {
                record_ruliad_correctness_context_swap(
                    summary,
                    job,
                    &decision.orbit.averaged_log_probs,
                    &swapped.orbit.averaged_log_probs,
                );
            });
        }
    }
    for ((original_index, counterfactual_job), counterfactual_decision) in
        counterfactual_jobs.iter().zip(&counterfactual_decisions)
    {
        let Some(original_decision) = decisions.get(*original_index) else {
            continue;
        };
        update_ruliad_correctness_constrained_summaries(
            &mut result,
            counterfactual_job,
            |summary| {
                record_ruliad_correctness_counterfactual_target(
                    summary,
                    counterfactual_job,
                    &original_decision.orbit.averaged_log_probs,
                    &counterfactual_decision.orbit.averaged_log_probs,
                );
            },
        );
    }
    let structured = evaluate_ruliad_structured_policy_decisions(
        "ruliad_correctness",
        &structured_items,
        &jobs,
        &decisions,
    )?;
    if structured.evaluation.report.verifier_match_count != result.summary.equivalent_top1 {
        return Err(anyhow!(
            "structured proof-policy verifier disagrees with semantic top-1: verifier={} constrained={}",
            structured.evaluation.report.verifier_match_count,
            result.summary.equivalent_top1
        ));
    }
    let structured_examples = ruliad_probe_examples(
        &structured.items,
        &structured.completions,
        training.events.capability_probe_example_count,
    );
    emit_ruliad_correctness_metrics_with_labels(RuliadCorrectnessMetrics {
        identity: RuliadProbeIdentity {
            run_name,
            epoch,
            absolute_step,
            probe_name: "ruliad_correctness_structured_policy",
        },
        report: &structured.evaluation.report,
        bus,
        metric_prefix: Some("Ruliad Structured Policy"),
        output_degeneracy: None,
        examples: &structured_examples,
        schema_alignment: ruliad_answer_schema_alignment_summary(
            &structured.items,
            &structured.completions,
        ),
        completion_degeneracy: None,
        generation_budget: None,
    });
    result.structured_decode = Some(structured.evaluation);
    let elapsed_ms = started.elapsed().as_micros() as f64 / 1_000.0;
    result.summary.elapsed_ms = elapsed_ms;
    for summary in result.difficulty_summaries.values_mut() {
        summary.elapsed_ms = elapsed_ms;
    }
    for summary in result.source_summaries.values_mut() {
        summary.elapsed_ms = elapsed_ms;
    }
    let summary = &result.summary;
    for (name, value) in [
        ("Ruliad Correctness Constrained Items", summary.items as f64),
        (
            "Ruliad Correctness Constrained Equivalent Top-1 Rate",
            ratio_usize(summary.equivalent_top1, summary.items),
        ),
        (
            "Ruliad Correctness Constrained Preferred Top-1 Rate",
            ratio_usize(summary.preferred_top1, summary.items),
        ),
        (
            "Ruliad Correctness Constrained Equivalent NLL",
            summary.equivalent_nll_sum / summary.items.max(1) as f64,
        ),
        (
            "Ruliad Correctness Constrained Valid-Invalid Margin",
            summary.valid_invalid_margin_sum / summary.valid_invalid_margin_items.max(1) as f64,
        ),
        (
            "Ruliad Correctness Constrained Canonical Equivalent Top-1 Rate",
            ratio_usize(summary.canonical_equivalent_top1, summary.canonical_items),
        ),
        (
            "Ruliad Correctness Constrained Canonical Preferred Top-1 Rate",
            ratio_usize(summary.canonical_preferred_top1, summary.canonical_items),
        ),
        (
            "Ruliad Correctness Constrained Canonical Equivalent NLL",
            summary.canonical_equivalent_nll_sum / summary.canonical_items.max(1) as f64,
        ),
        (
            "Ruliad Correctness Constrained Canonical Valid-Invalid Margin",
            summary.canonical_valid_invalid_margin_sum
                / summary.canonical_valid_invalid_margin_items.max(1) as f64,
        ),
        (
            "Ruliad Correctness Constrained Worst-Presentation Equivalent Top-1 Rate",
            ratio_usize(
                summary.worst_presentation_equivalent_top1,
                summary.worst_presentation_items,
            ),
        ),
        (
            "Ruliad Correctness Constrained Worst-Presentation Equivalent NLL",
            summary.worst_presentation_equivalent_nll_sum
                / summary.worst_presentation_items.max(1) as f64,
        ),
        (
            "Ruliad Correctness Constrained Worst-Presentation Valid-Invalid Margin",
            summary.worst_presentation_valid_invalid_margin_sum
                / summary.worst_presentation_valid_invalid_margin_items.max(1) as f64,
        ),
        (
            "Ruliad Correctness Constrained Orbit JS Divergence",
            summary.orbit_js_divergence_sum / summary.items.max(1) as f64,
        ),
        (
            "Ruliad Correctness Constrained Orbit Top-1 Consensus Fraction",
            summary.orbit_top1_consensus_fraction_sum / summary.items.max(1) as f64,
        ),
        (
            "Ruliad Correctness Constrained Complete Orbit Items",
            summary.complete_orbit_items as f64,
        ),
        (
            "Ruliad Correctness Constrained Presentation Rows",
            summary.presentation_rows as f64,
        ),
        (
            "Ruliad Correctness Constrained Presentation Equivalent Top-1 Rate",
            ratio_usize(
                summary.presentation_equivalent_top1,
                summary.presentation_rows,
            ),
        ),
        (
            "Ruliad Correctness Constrained Presentation Preferred Top-1 Rate",
            ratio_usize(
                summary.presentation_preferred_top1,
                summary.presentation_rows,
            ),
        ),
        (
            "Ruliad Correctness Constrained Context-Swap Items",
            summary.context_swap_items as f64,
        ),
        (
            "Ruliad Correctness Constrained Context-Swap Equivalent Top-1 Rate",
            ratio_usize(
                summary.context_swap_equivalent_top1,
                summary.context_swap_items,
            ),
        ),
        (
            "Ruliad Correctness Constrained Context-Swap Equivalent NLL",
            summary.context_swap_equivalent_nll_sum / summary.context_swap_items.max(1) as f64,
        ),
        (
            "Ruliad Correctness Constrained Context-Swap Top-1 Change Rate",
            ratio_usize(
                summary.context_swap_top1_changes,
                summary.context_swap_items,
            ),
        ),
        (
            "Ruliad Correctness Constrained Context-Swap Equivalent Probability Drop",
            summary.context_swap_equivalent_probability_drop_sum
                / summary.context_swap_items.max(1) as f64,
        ),
        (
            "Ruliad Correctness Constrained Context-Swap JS Divergence",
            summary.context_swap_js_divergence_sum / summary.context_swap_items.max(1) as f64,
        ),
        (
            "Ruliad Correctness Constrained Counterfactual-Target Items",
            summary.counterfactual_target_items as f64,
        ),
        (
            "Ruliad Correctness Constrained Counterfactual-Target Equivalent Top-1 Rate",
            ratio_usize(
                summary.counterfactual_target_equivalent_top1,
                summary.counterfactual_target_items,
            ),
        ),
        (
            "Ruliad Correctness Constrained Counterfactual-Target Equivalent NLL",
            summary.counterfactual_target_equivalent_nll_sum
                / summary.counterfactual_target_items.max(1) as f64,
        ),
        (
            "Ruliad Correctness Constrained Counterfactual-Target Top-1 Change Rate",
            ratio_usize(
                summary.counterfactual_target_top1_changes,
                summary.counterfactual_target_items,
            ),
        ),
        (
            "Ruliad Correctness Constrained Counterfactual-Target Equivalent Probability Gain",
            summary.counterfactual_target_equivalent_probability_gain_sum
                / summary.counterfactual_target_items.max(1) as f64,
        ),
        (
            "Ruliad Correctness Constrained Counterfactual-Target JS Divergence",
            summary.counterfactual_target_js_divergence_sum
                / summary.counterfactual_target_items.max(1) as f64,
        ),
        (
            "Ruliad Correctness Constrained Elapsed MS",
            summary.elapsed_ms,
        ),
        (
            "Ruliad Correctness Constrained Symmetry Balanced",
            usize::from(!matches!(
                config.candidate_symmetry,
                crate::config::RuliadProofPolicyCandidateSymmetry::Canonical
            )) as f64,
        ),
        (
            "Ruliad Correctness Constrained Symmetry Orbit Averaged",
            usize::from(matches!(
                config.candidate_symmetry,
                crate::config::RuliadProofPolicyCandidateSymmetry::CyclicOrbitAverage
            )) as f64,
        ),
    ] {
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: run_name.to_string().into(),
            split: TrainingMetricSplit::Valid,
            epoch,
            step_in_epoch: 0,
            absolute_step,
            name: name.to_string(),
            value,
            running_value: value,
        });
    }
    Ok(result)
}

impl RuliadPolicyRolloutProbeSummary {
    pub(super) fn accumulate(&mut self, item: Self) {
        self.items = self.items.saturating_add(item.items);
        self.solved = self.solved.saturating_add(item.solved);
        self.steps = self.steps.saturating_add(item.steps);
        self.valid_actions = self.valid_actions.saturating_add(item.valid_actions);
        self.invalid_actions = self.invalid_actions.saturating_add(item.invalid_actions);
        self.repeated_states = self.repeated_states.saturating_add(item.repeated_states);
        self.backtracks = self.backtracks.saturating_add(item.backtracks);
        self.scored_states = self.scored_states.saturating_add(item.scored_states);
        self.scored_actions = self.scored_actions.saturating_add(item.scored_actions);
        self.top1_expert_actions = self
            .top1_expert_actions
            .saturating_add(item.top1_expert_actions);
        self.frontier_exhaustions = self
            .frontier_exhaustions
            .saturating_add(item.frontier_exhaustions);
        self.solved_goals = self.solved_goals.saturating_add(item.solved_goals);
        self.total_goals = self.total_goals.saturating_add(item.total_goals);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct RuliadPolicyScoringSummary {
    pub(super) batches: usize,
    pub(super) rows: usize,
    pub(super) unpadded_tokens: usize,
    pub(super) padded_tokens: usize,
    pub(super) maximum_batch_rows: usize,
    pub(super) maximum_pipeline_depth: usize,
    pub(super) elapsed_ms: f64,
    pub(super) cpu_prepare_ms: f64,
    pub(super) model_scoring_ms: f64,
    pub(super) cpu_transition_ms: f64,
}

impl RuliadPolicyScoringSummary {
    pub(super) fn record_batch(&mut self, sequence_lengths: &[usize]) {
        if sequence_lengths.is_empty() {
            return;
        }
        let rows = sequence_lengths.len();
        let maximum_len = sequence_lengths.iter().copied().max().unwrap_or_default();
        self.batches = self.batches.saturating_add(1);
        self.rows = self.rows.saturating_add(rows);
        self.unpadded_tokens = self
            .unpadded_tokens
            .saturating_add(sequence_lengths.iter().copied().sum::<usize>());
        self.padded_tokens = self
            .padded_tokens
            .saturating_add(rows.saturating_mul(maximum_len));
        self.maximum_batch_rows = self.maximum_batch_rows.max(rows);
    }

    pub(super) fn record_pipeline_depth(&mut self, depth: usize) {
        self.maximum_pipeline_depth = self.maximum_pipeline_depth.max(depth);
    }
}

#[derive(Clone)]
pub(super) struct RuliadPolicyBeamNode {
    pub(super) state: burn_dragon_universality::ruliad::RuliadProofPolicyState,
    pub(super) log_probability: f32,
    pub(super) steps: usize,
}

pub(super) struct RuliadPolicyBeamExpansion {
    pub(super) search_index: usize,
    pub(super) node: RuliadPolicyBeamNode,
    pub(super) actions: burn_dragon_universality::ruliad::RuliadProofActionSet,
    pub(super) presentations: Vec<RuliadPolicyActionPresentation>,
}

pub(super) struct RuliadPolicyScoredExpansion {
    pub(super) expansion: RuliadPolicyBeamExpansion,
    pub(super) scores: Vec<f32>,
}

pub(super) struct RuliadPolicyProbeSearch {
    pub(super) problem: burn_dragon_universality::ruliad::RuliadProofProblem,
    pub(super) certificate_hash: String,
    pub(super) answer_contract: burn_dragon_universality::ruliad::RuliadProofActionAnswerContract,
    pub(super) difficulty_level: usize,
    pub(super) rollout_limit: usize,
    pub(super) beam: Vec<RuliadPolicyBeamNode>,
    pub(super) best_node: RuliadPolicyBeamNode,
    pub(super) best_state_scores: BTreeMap<String, f32>,
    pub(super) summary: RuliadPolicyRolloutProbeSummary,
    pub(super) done: bool,
}

pub(super) fn prepare_ruliad_policy_search_expansions(
    dataset: &Dataset,
    config: crate::config::RuliadPolicyProbeConfig,
    search_index: usize,
    search: &mut RuliadPolicyProbeSearch,
    depth: usize,
) -> Result<Vec<RuliadPolicyBeamExpansion>> {
    if search.done {
        return Ok(Vec::new());
    }
    if depth >= search.rollout_limit {
        search.done = true;
        return Ok(Vec::new());
    }

    let mut expansions = Vec::with_capacity(search.beam.len());
    for (node_index, node) in search.beam.clone().into_iter().enumerate() {
        if node.state.solved() {
            search.best_node = node;
            search.done = true;
            break;
        }
        let actions = match node.state.action_set(&search.problem, config.candidates) {
            Ok(actions) => actions,
            Err(_) => {
                search.summary.frontier_exhaustions =
                    search.summary.frontier_exhaustions.saturating_add(1);
                continue;
            }
        };
        let action_presentations = ruliad_policy_action_presentations(
            &actions,
            config.candidate_symmetry,
            search_index.wrapping_add(depth).wrapping_add(node_index),
        )?;
        let presentations = action_presentations
            .into_iter()
            .map(|(rotation, presented_actions)| {
                let scoring_contract = match config.scoring {
                    crate::config::RuliadProofPolicyScoring::CompletionLikelihood => {
                        search.answer_contract
                    }
                    crate::config::RuliadProofPolicyScoring::SemanticEnergy
                    | crate::config::RuliadProofPolicyScoring::ResidualEnergy => {
                        burn_dragon_universality::ruliad::RuliadProofActionAnswerContract::SemanticStep
                    }
                };
                let prompt = crate::train::ruliad_policy::ruliad_proof_policy_prompt(
                    config.prompt_context,
                    &search.problem,
                    &presented_actions,
                )
                .ok()?;
                let prompt_tokens = dataset.encode_ruliad_payload_tokens(&prompt)?;
                let candidate_tokens = (0..presented_actions.candidates.len())
                    .map(|candidate_index| {
                        let answer = burn_dragon_universality::ruliad::proof_action_answer(
                            &presented_actions,
                            candidate_index,
                            scoring_contract,
                        )
                        .ok()?;
                        encode_ruliad_policy_candidate_tokens(
                            dataset,
                            &answer,
                            scoring_contract,
                        )
                    })
                    .collect::<Option<Vec<_>>>()?;
                Some(RuliadPolicyActionPresentation {
                    rotation,
                    prompt_tokens: prompt_tokens.into_iter().map(i64::from).collect(),
                    candidate_tokens,
                    answer_contract: scoring_contract,
                })
            })
            .collect::<Option<Vec<_>>>();
        let Some(presentations) = presentations else {
            search.summary.invalid_actions = search.summary.invalid_actions.saturating_add(1);
            continue;
        };
        expansions.push(RuliadPolicyBeamExpansion {
            search_index,
            node,
            actions,
            presentations,
        });
    }
    if !search.done && expansions.is_empty() {
        search.summary.frontier_exhaustions = search.summary.frontier_exhaustions.saturating_add(1);
        search.done = true;
    }
    Ok(expansions)
}

pub(super) fn apply_ruliad_policy_scored_expansions(
    search: &mut RuliadPolicyProbeSearch,
    children: &mut BTreeMap<String, RuliadPolicyBeamNode>,
    scored_expansions: Vec<RuliadPolicyScoredExpansion>,
    config: crate::config::RuliadPolicyProbeConfig,
    depth: usize,
) -> Result<()> {
    for scored in scored_expansions {
        if search.done {
            break;
        }
        let expansion = scored.expansion;
        search.summary.scored_states = search.summary.scored_states.saturating_add(1);
        search.summary.scored_actions = search
            .summary
            .scored_actions
            .saturating_add(expansion.actions.candidates.len());
        if crate::train::ruliad_policy::best_candidate_index(&scored.scores)
            .is_some_and(|index| expansion.actions.is_equivalent_index(index))
        {
            search.summary.top1_expert_actions =
                search.summary.top1_expert_actions.saturating_add(1);
        }
        for (candidate_index, log_probability) in scored.scores.into_iter().enumerate() {
            let mut state = expansion.node.state.clone();
            let repeated = match state.apply(&expansion.actions, candidate_index) {
                Ok(repeated) => repeated,
                Err(_) => {
                    search.summary.invalid_actions =
                        search.summary.invalid_actions.saturating_add(1);
                    continue;
                }
            };
            search.summary.repeated_states = search
                .summary
                .repeated_states
                .saturating_add(usize::from(repeated));
            let child = RuliadPolicyBeamNode {
                state,
                log_probability: expansion.node.log_probability + log_probability,
                steps: expansion.node.steps.saturating_add(1),
            };
            if child.state.solved() {
                search.best_node = child;
                search.done = true;
                break;
            }
            let state_key = child.state.canonical_state_key(&search.problem)?;
            if search
                .best_state_scores
                .get(&state_key)
                .is_some_and(|score| *score >= child.log_probability)
            {
                continue;
            }
            search
                .best_state_scores
                .insert(state_key.clone(), child.log_probability);
            if children
                .get(&state_key)
                .is_none_or(|existing| child.log_probability > existing.log_probability)
            {
                children.insert(state_key, child);
            }
        }
    }

    if search.done {
        return Ok(());
    }
    let next = std::mem::take(children);
    if next.is_empty() {
        search.summary.frontier_exhaustions = search.summary.frontier_exhaustions.saturating_add(1);
        search.done = true;
        return Ok(());
    }
    search.beam = next.into_values().collect();
    search.beam.sort_by(|left, right| {
        right
            .log_probability
            .total_cmp(&left.log_probability)
            .then_with(|| right.state.solved_goals().cmp(&left.state.solved_goals()))
    });
    search.beam.truncate(config.beam_width.max(1));
    search.best_node = search.beam[0].clone();
    if depth.saturating_add(1) >= search.rollout_limit {
        search.done = true;
    }
    Ok(())
}

pub(super) fn bounded_padded_batch_end(
    sequence_lengths: &[usize],
    start: usize,
    maximum_rows: usize,
    token_budget: usize,
) -> usize {
    let mut end = start;
    let mut maximum_sequence_len = 0usize;
    while end < sequence_lengths.len() && end.saturating_sub(start) < maximum_rows.max(1) {
        let sequence_len = sequence_lengths[end];
        let next_maximum = maximum_sequence_len.max(sequence_len);
        let next_rows = end.saturating_sub(start).saturating_add(1);
        if end > start && next_rows.saturating_mul(next_maximum) > token_budget.max(1) {
            break;
        }
        maximum_sequence_len = next_maximum;
        end = end.saturating_add(1);
    }
    end.max(start.saturating_add(1).min(sequence_lengths.len()))
}
