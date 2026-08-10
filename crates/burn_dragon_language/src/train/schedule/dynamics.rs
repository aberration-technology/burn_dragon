//! Continual-learning dynamics control and checkpoint recovery.

use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DynamicsControlOutcome {
    Continue,
    Stop,
}

pub(super) fn apply_pending_dynamics_control<B>(
    env: &TrainEnvironment<'_, B>,
    slot: &DragonDynamicsControlSlot,
    active: &mut ActiveDynamicsControl,
    optimizer: &mut crate::train::continual_backprop::LanguageOptimizer<B>,
    model: &LanguageTrainModel<B>,
) -> DynamicsControlOutcome
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let Some(event) = slot.take() else {
        optimizer.set_continual_backprop_runtime_control(
            active.continual_backprop_scale,
            active.max_replacements_per_interval,
        );
        model.set_recovery_auxiliary_active(active.recovery_auxiliary_active());
        return DynamicsControlOutcome::Continue;
    };
    if event.run_id.as_str() != env.run_name {
        return DynamicsControlOutcome::Continue;
    }
    apply_dynamics_control_event(env, &event, active, optimizer, model)
}

pub(super) fn apply_dynamics_control_event<B>(
    env: &TrainEnvironment<'_, B>,
    event: &burn_dragon_train::train::events::DynamicsControlEvent,
    active: &mut ActiveDynamicsControl,
    optimizer: &mut crate::train::continual_backprop::LanguageOptimizer<B>,
    model: &LanguageTrainModel<B>,
) -> DynamicsControlOutcome
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    active.apply_event(event);
    optimizer.set_continual_backprop_runtime_control(
        active.continual_backprop_scale,
        active.max_replacements_per_interval,
    );
    model.set_recovery_auxiliary_active(active.recovery_auxiliary_active());
    if let Some(dataset) = env.source_selection_dataset.as_ref() {
        dataset.apply_source_selection_dynamics_control(
            active.source_difficulty_pressure as f32,
            active.hash_noise_max_probability as f32,
        );
    }
    info!(
        "dynamics control applied: mode={:?} lr_scale={:.3} cbp_scale={:.3} source_pressure={:.3} hash_noise_max={:.3} reason={}",
        active.mode,
        active.lr_scale,
        active.continual_backprop_scale,
        active.source_difficulty_pressure,
        active.hash_noise_max_probability,
        active.last_reason
    );
    if event.stop_if_repeated {
        DynamicsControlOutcome::Stop
    } else {
        DynamicsControlOutcome::Continue
    }
}

pub(super) struct DynamicTrainingState<'a, B: AutodiffBackend, S> {
    pub(super) active: &'a mut ActiveDynamicsControl,
    pub(super) optimizer: &'a mut crate::train::continual_backprop::LanguageOptimizer<B>,
    pub(super) scheduler: &'a mut S,
    pub(super) model: &'a mut LanguageTrainModel<B>,
    pub(super) model_config: &'a mut DragonConfig,
}

pub(super) fn handle_post_validation_dynamics_control<B, S>(
    env: &TrainEnvironment<'_, B>,
    slot: &DragonDynamicsControlSlot,
    state: DynamicTrainingState<'_, B, S>,
    epoch: usize,
) -> Result<DynamicsControlOutcome>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    S: LrScheduler + Clone + 'static,
{
    let DynamicTrainingState {
        active,
        optimizer,
        scheduler,
        model,
        model_config: current_model_config,
    } = state;
    let Some(event) = slot.take() else {
        return Ok(DynamicsControlOutcome::Continue);
    };
    if event.run_id.as_str() != env.run_name {
        return Ok(DynamicsControlOutcome::Continue);
    }
    let rollback_epoch = event.rollback_to_epoch;
    if let Some(rollback_epoch) = rollback_epoch {
        let mut checkpoint_dynamics_control = active.clone();
        let (rollback_model, rollback_config) = load_dragon_training_state_checkpoint(
            env.run_dir,
            rollback_epoch,
            current_model_config,
            env.device,
            optimizer,
            scheduler,
            &mut checkpoint_dynamics_control,
        )
        .with_context(|| {
            format!(
                "failed to apply dynamics rollback from epoch {epoch} to checkpoint epoch {rollback_epoch}"
            )
        })?;
        model.model = rollback_model;
        load_runtime_state_checkpoint(
            env.run_dir,
            rollback_epoch,
            model,
            env.device,
            false,
            false,
        )?;
        if let Some(rollback_config) = rollback_config {
            *current_model_config = rollback_config;
        }
        optimizer.refresh_continual_backprop_fresh_model(DragonModel::<B>::new(
            current_model_config.clone(),
            env.device,
        ));
        let outcome = apply_dynamics_control_event(env, &event, active, optimizer, model);
        info!(
            "dynamics rollback applied: epoch={} rollback_epoch={} latent_total={} checkpoint_mode={:?} active_mode={:?} reason={}",
            epoch,
            rollback_epoch,
            current_model_config.latent_total(),
            checkpoint_dynamics_control.mode,
            active.mode,
            active.last_reason
        );
        return Ok(outcome);
    }
    let outcome = apply_dynamics_control_event(env, &event, active, optimizer, model);
    Ok(outcome)
}

#[derive(Debug, Default)]
struct RuliadCapabilityRegressionEvidence {
    regressed: bool,
    details: Vec<String>,
}

fn update_ruliad_capability_regression_state<B>(
    env: &TrainEnvironment<'_, B>,
    validation: &DynamicValidationReport,
    state: &mut ContinualLearningStabilityState,
) -> RuliadCapabilityRegressionEvidence
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let capability_contract = env
        .training
        .ruliad_policy_probe
        .checkpoint_capability_contract;
    let previous_regressions = state.consecutive_ruliad_correctness_regressions;
    let mut free_run_regressed = false;
    let mut policy_regressed = false;
    let mut details = Vec::new();

    if capability_contract.requires_free_run()
        && let Some(report) = validation.ruliad_eval_report.as_ref()
        && report.scored_count > 0
    {
        let verifier_accuracy = report.verifier_accuracy;
        let partial_progress = report.mean_partial_progress;
        let verifier_best = state
            .best_ruliad_verifier_accuracy
            .unwrap_or(verifier_accuracy);
        let partial_best = state
            .best_ruliad_partial_progress
            .unwrap_or(partial_progress);
        let verifier_improved = verifier_accuracy > verifier_best + f32::EPSILON;
        let partial_improved = partial_progress > partial_best + f32::EPSILON;
        if verifier_improved || state.best_ruliad_verifier_accuracy.is_none() {
            state.best_ruliad_verifier_accuracy = Some(verifier_accuracy);
        }
        if partial_improved || state.best_ruliad_partial_progress.is_none() {
            state.best_ruliad_partial_progress = Some(partial_progress);
        }

        let verifier_regressed = ruliad_metric_materially_regressed(
            verifier_best,
            verifier_accuracy,
            report.scored_count,
            0.125,
        );
        let partial_regressed = ruliad_metric_materially_regressed(
            partial_best,
            partial_progress,
            report.scored_count,
            0.25,
        );
        free_run_regressed = verifier_regressed || partial_regressed;
        if free_run_regressed {
            details.push(format!(
                "free_run verifier {:.3}->{:.3}, partial_progress {:.3}->{:.3}",
                verifier_best, verifier_accuracy, partial_best, partial_progress
            ));
        }
    }

    if capability_contract.requires_closed_loop_policy()
        && let Some(rollout) = validation.ruliad_policy_rollout.as_ref()
        && let Some(current) = ruliad_policy_competence_key(rollout)
    {
        let best = state
            .best_ruliad_policy_observed_competence
            .unwrap_or(current);
        let summary = rollout.summary;
        let (current_solve, current_goal, current_valid_action) =
            ruliad_policy_observations(summary);
        let best_solve = state
            .best_ruliad_policy_solve_observation
            .unwrap_or(current_solve);
        let best_goal = state
            .best_ruliad_policy_goal_observation
            .unwrap_or(current_goal);
        let best_valid_action = state
            .best_ruliad_policy_valid_action_observation
            .unwrap_or(current_valid_action);
        let regression_z = env
            .training
            .ruliad_policy_probe
            .promotion_gate
            .regression_confidence_z;
        let solve_regressed = binomial_observation_materially_regressed(
            best_solve,
            current_solve,
            0.125,
            regression_z,
        );
        let goal_regressed =
            binomial_observation_materially_regressed(best_goal, current_goal, 0.25, regression_z);
        let valid_action_regressed = binomial_observation_materially_regressed(
            best_valid_action,
            current_valid_action,
            0.90,
            regression_z,
        );
        state.best_ruliad_policy_observed_competence = Some(best.componentwise_max(current));
        state.best_ruliad_policy_solve_observation =
            Some(best_solve.prefer_stronger_evidence(current_solve, regression_z));
        state.best_ruliad_policy_goal_observation =
            Some(best_goal.prefer_stronger_evidence(current_goal, regression_z));
        state.best_ruliad_policy_valid_action_observation =
            Some(best_valid_action.prefer_stronger_evidence(current_valid_action, regression_z));
        policy_regressed = solve_regressed || goal_regressed || valid_action_regressed;
        if policy_regressed {
            details.push(format!(
                "closed_loop_policy solve {:.3}->{:.3}, goal_completion {:.3}->{:.3}, valid_action {:.3}->{:.3} (Wilson z={regression_z:.3})",
                best_solve.rate(),
                current_solve.rate(),
                best_goal.rate(),
                current_goal.rate(),
                best_valid_action.rate(),
                current_valid_action.rate(),
            ));
        }
    }

    let regressed = match capability_contract {
        crate::config::RuliadCheckpointCapabilityContract::FreeRunText => free_run_regressed,
        crate::config::RuliadCheckpointCapabilityContract::ClosedLoopPolicy => policy_regressed,
        crate::config::RuliadCheckpointCapabilityContract::Joint => {
            free_run_regressed || policy_regressed
        }
    };
    state.consecutive_ruliad_correctness_regressions = if regressed {
        previous_regressions.saturating_add(1)
    } else {
        0
    };
    RuliadCapabilityRegressionEvidence { regressed, details }
}

pub(super) fn apply_continual_learning_stability_policy<B>(
    env: &TrainEnvironment<'_, B>,
    validation: DynamicValidationReport,
    epoch: usize,
    absolute_step: usize,
    state: &mut ContinualLearningStabilityState,
    bus: &TrainingEventBus,
) where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let valid_loss = validation.primary_loss();
    let policy = &env.training.dynamics;
    let event = TrainingEventContext {
        epoch,
        absolute_step,
        bus,
    };
    let capability_contract = env
        .training
        .ruliad_policy_probe
        .checkpoint_capability_contract;
    let ruliad_correctness_improved =
        validation_ruliad_capability_improved(&validation, state, capability_contract);
    let has_capability_probe =
        validation.ruliad_eval_report.is_some() || validation.ruliad_policy_rollout.is_some();
    let capability_regression = update_ruliad_capability_regression_state(env, &validation, state);
    let capability_quality_collapsed =
        has_capability_probe && validation_capability_quality_collapse(&validation, env.training);
    let closed_loop_policy_stable = matches!(
        capability_contract,
        crate::config::RuliadCheckpointCapabilityContract::ClosedLoopPolicy
    ) && validation.ruliad_policy_rollout.is_some()
        && !capability_regression.regressed
        && !capability_quality_collapsed;
    let output_degeneracy_failed = validation.output_degeneracy.as_ref().is_some_and(|stats| {
        capability_contract.requires_free_run()
            && output_degeneracy_tripped(&env.training.gates, stats)
    });
    let mut recovery_requested = output_degeneracy_failed;
    let improved = state.best_valid_loss.is_none_or(|best| {
        valid_loss < best * (1.0 - env.training.gates.plateau_min_relative_improvement)
    });
    if improved {
        state.best_valid_loss = Some(valid_loss);
        state.consecutive_validation_regressions = 0;
        if !capability_quality_collapsed
            && !capability_regression.regressed
            && !output_degeneracy_failed
        {
            emit_dynamics_control(
                env,
                policy,
                DynamicsMode::Stable,
                None,
                "validation improved; returning stability controls to baseline".to_string(),
                event,
            );
        }
    } else if let Some(best) = state.best_valid_loss {
        if valid_loss > best * (1.0 + env.training.gates.validation_regression_max_relative) {
            if ruliad_correctness_improved || closed_loop_policy_stable {
                state.consecutive_validation_regressions = 0;
                let reason = if ruliad_correctness_improved {
                    "ruliad correctness improved"
                } else {
                    "closed-loop policy has no statistically supported regression"
                };
                emit_policy_gate_with_action(
                    env,
                    "continual_learning_validation_regression_suppressed_by_ruliad_progress",
                    TrainingGateAction::Alert,
                    TrainingGateSeverity::Info,
                    format!(
                        "teacher-forced validation worsened but {reason}: best loss {:.6}, current {:.6}; suppressing rollback",
                        best, valid_loss,
                    ),
                    event,
                );
            } else {
                state.consecutive_validation_regressions =
                    state.consecutive_validation_regressions.saturating_add(1);
            }
        } else {
            state.consecutive_validation_regressions = 0;
        }
        if state.consecutive_validation_regressions
            >= env.training.gates.validation_regression_patience_epochs
        {
            let rollback_epoch = state.best_checkpoint_epoch;
            let mode = if rollback_epoch.is_some() {
                DynamicsMode::RollbackRecovery
            } else {
                DynamicsMode::ValidationRecovery
            };
            let message = format!(
                "validation regression detected: best {:.6}, current {:.6}; requesting {:?}{}",
                best,
                valid_loss,
                mode,
                rollback_epoch
                    .map(|epoch| format!(" to checkpoint epoch {epoch}"))
                    .unwrap_or_default()
            );
            emit_policy_gate_with_action(
                env,
                "continual_learning_validation_regression",
                TrainingGateAction::Alert,
                TrainingGateSeverity::Warning,
                message.clone(),
                event,
            );
            emit_dynamics_control(env, policy, mode, rollback_epoch, message, event);
            recovery_requested = true;
        }
    }

    if has_capability_probe {
        update_capability_run_control_state(
            env,
            &validation,
            state,
            &mut recovery_requested,
            TrainingEventContext {
                epoch,
                absolute_step,
                bus,
            },
        );
    }

    if state.consecutive_ruliad_correctness_regressions
        >= env.training.gates.capability_regression_patience_epochs
        && !recovery_requested
    {
        let rollback_epoch = capability_rollback_checkpoint_epoch(state);
        let mode = if rollback_epoch.is_some() {
            DynamicsMode::RollbackRecovery
        } else {
            DynamicsMode::ValidationRecovery
        };
        let message = format!(
            "ruliad deployment-capability regression detected: {}; requesting {:?}{}",
            capability_regression.details.join("; "),
            mode,
            rollback_epoch
                .map(|epoch| format!(" to checkpoint epoch {epoch}"))
                .unwrap_or_default()
        );
        emit_policy_gate_with_action(
            env,
            "continual_learning_ruliad_capability_regression",
            TrainingGateAction::Alert,
            TrainingGateSeverity::Warning,
            message.clone(),
            event,
        );
        emit_dynamics_control(env, policy, mode, rollback_epoch, message, event);
        recovery_requested = true;
    }

    if !capability_contract.requires_free_run() {
        return;
    }
    let Some(degeneracy) = validation.output_degeneracy else {
        return;
    };
    let degenerate = output_degeneracy_tripped(&env.training.gates, &degeneracy);
    if uncertain_argmax_loop(&env.training.gates, &degeneracy) {
        emit_policy_gate(
            env,
            bus,
            "continual_learning_uncertain_argmax_loop",
            epoch,
            absolute_step,
            format!(
                "low-confidence argmax loop observed without distribution collapse: entropy {:.3}, max_prob {:.3}, unique {:.3}, distinct2 {:.3}, repetition {:.3}, period2 {:.3}, period3 {:.3}, max_period2_16 {:.3}, max_period2_64 {:.3} (period {})",
                degeneracy.entropy_bits,
                degeneracy.mean_max_probability,
                degeneracy.argmax_unique_fraction,
                degeneracy.distinct_2_fraction,
                degeneracy.repetition_fraction,
                degeneracy.period_2_fraction,
                degeneracy.period_3_fraction,
                degeneracy.max_period_2_to_16_fraction,
                degeneracy.max_period_2_to_64_fraction,
                degeneracy.dominant_period_2_to_64
            ),
        );
    }
    if degenerate {
        state.consecutive_output_degeneracy = state.consecutive_output_degeneracy.saturating_add(1);
    } else {
        state.consecutive_output_degeneracy = 0;
    }
    if state.consecutive_output_degeneracy >= env.training.gates.degeneracy_patience {
        let hard_collapse = hard_output_collapse(env, &degeneracy);
        let rollback_epoch = capability_rollback_checkpoint_epoch(state);
        let mode = if hard_collapse && rollback_epoch.is_some() {
            DynamicsMode::RollbackRecovery
        } else if hard_collapse {
            DynamicsMode::HardRecovery
        } else {
            DynamicsMode::PlasticityRecovery
        };
        emit_policy_gate_with_action(
            env,
            "continual_learning_output_degeneracy",
            TrainingGateAction::Alert,
            TrainingGateSeverity::Warning,
            format!(
                "{} output degeneracy detected while leaving continual backprop active and routing through dynamics recovery: entropy {:.3}, max_prob {:.3}, unique {:.3}, distinct2 {:.3}, repetition {:.3}, period2 {:.3}, period3 {:.3}, max_period2_16 {:.3}, max_period2_64 {:.3} (period {})",
                if hard_collapse { "hard" } else { "soft" },
                degeneracy.entropy_bits,
                degeneracy.mean_max_probability,
                degeneracy.argmax_unique_fraction,
                degeneracy.distinct_2_fraction,
                degeneracy.repetition_fraction,
                degeneracy.period_2_fraction,
                degeneracy.period_3_fraction,
                degeneracy.max_period_2_to_16_fraction,
                degeneracy.max_period_2_to_64_fraction,
                degeneracy.dominant_period_2_to_64
            ),
            event,
        );
        let dragon_control_adds_rollback = hard_collapse && rollback_epoch.is_some();
        if dragon_control_adds_rollback || !recovery_requested {
            let message = format!(
                "{} output degeneracy detected: entropy {:.3}, max_prob {:.3}, unique {:.3}, distinct2 {:.3}, repetition {:.3}, period2 {:.3}, period3 {:.3}, max_period2_16 {:.3}, max_period2_64 {:.3} (period {}); requesting {:?}{}",
                if hard_collapse { "hard" } else { "soft" },
                degeneracy.entropy_bits,
                degeneracy.mean_max_probability,
                degeneracy.argmax_unique_fraction,
                degeneracy.distinct_2_fraction,
                degeneracy.repetition_fraction,
                degeneracy.period_2_fraction,
                degeneracy.period_3_fraction,
                degeneracy.max_period_2_to_16_fraction,
                degeneracy.max_period_2_to_64_fraction,
                degeneracy.dominant_period_2_to_64,
                mode,
                (mode == DynamicsMode::RollbackRecovery)
                    .then_some(rollback_epoch)
                    .flatten()
                    .map(|epoch| format!(" to checkpoint epoch {epoch}"))
                    .unwrap_or_default(),
            );
            emit_dynamics_control(
                env,
                policy,
                mode,
                (mode == DynamicsMode::RollbackRecovery)
                    .then_some(rollback_epoch)
                    .flatten(),
                message,
                event,
            );
        }
    }
}

pub(super) fn emit_dynamics_control<B>(
    env: &TrainEnvironment<'_, B>,
    policy: &burn_dragon_train::train::events::DynamicsEquilibriumPolicy,
    mode: DynamicsMode,
    rollback_to_epoch: Option<usize>,
    reason: String,
    event: TrainingEventContext<'_>,
) where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let TrainingEventContext {
        epoch,
        absolute_step,
        bus,
    } = event;
    if !env.training.dynamics.enabled {
        return;
    }
    let (
        lr_scale,
        continual_backprop_scale,
        max_replacements_per_interval,
        source_difficulty_pressure,
        hash_noise_max_probability,
    ) = match mode {
        DynamicsMode::Stable => (
            1.0,
            1.0,
            None,
            policy.stable_source_difficulty_pressure,
            policy.stable_hash_noise_max_probability,
        ),
        DynamicsMode::DifficultyAdvance | DynamicsMode::CapacityLimited => (
            1.0,
            1.0,
            None,
            policy.difficulty_advance_source_pressure,
            policy.stable_hash_noise_max_probability,
        ),
        DynamicsMode::SourceCapabilityRecovery => (
            policy.source_capability_recovery_lr_scale,
            policy.source_capability_recovery_continual_backprop_scale,
            policy.source_capability_recovery_max_replacements_per_interval,
            policy.source_capability_recovery_source_difficulty_pressure,
            policy.source_capability_recovery_hash_noise_max_probability,
        ),
        DynamicsMode::PlasticityRecovery => (
            policy.soft_recovery_lr_scale,
            policy.soft_recovery_continual_backprop_scale,
            policy.soft_recovery_max_replacements_per_interval,
            policy.recovery_source_difficulty_pressure,
            policy.recovery_hash_noise_max_probability,
        ),
        DynamicsMode::ValidationRecovery => (
            policy.validation_recovery_lr_scale,
            policy.validation_recovery_continual_backprop_scale,
            policy.validation_recovery_max_replacements_per_interval,
            policy.recovery_source_difficulty_pressure,
            policy.recovery_hash_noise_max_probability,
        ),
        DynamicsMode::RollbackRecovery | DynamicsMode::HardRecovery => (
            policy.hard_recovery_lr_scale,
            policy.hard_recovery_continual_backprop_scale,
            policy.hard_recovery_max_replacements_per_interval,
            policy.recovery_source_difficulty_pressure,
            policy.recovery_hash_noise_max_probability,
        ),
        DynamicsMode::HardCollapse => (
            0.0,
            policy.minimum_continual_backprop_scale,
            policy.hard_recovery_max_replacements_per_interval,
            policy.recovery_source_difficulty_pressure,
            policy.recovery_hash_noise_max_probability,
        ),
    };
    let _ = bus.send_dynamics_control(DynamicsControlEvent {
        run_id: env.run_name.to_string().into(),
        epoch: Some(epoch),
        absolute_step: Some(absolute_step),
        mode,
        lr_scale,
        continual_backprop_scale: continual_backprop_scale
            .max(policy.minimum_continual_backprop_scale),
        max_replacements_per_interval,
        source_difficulty_pressure,
        hash_noise_max_probability,
        rollback_to_epoch,
        stop_if_repeated: mode == DynamicsMode::HardCollapse,
        reason,
    });
}

pub(super) fn output_degeneracy_tripped(
    gates: &burn_dragon_train::TrainingGatesConfig,
    degeneracy: &crate::train::steps::OutputDegeneracyStats,
) -> bool {
    output_diversity_degeneracy(gates, degeneracy)
}

pub(super) fn output_distribution_collapse(
    gates: &burn_dragon_train::TrainingGatesConfig,
    degeneracy: &crate::train::steps::OutputDegeneracyStats,
) -> bool {
    degeneracy.entropy_bits < gates.degeneracy_entropy_min_bits
        || degeneracy.mean_max_probability > gates.degeneracy_max_probability_max
}

pub(super) fn output_diversity_degeneracy(
    gates: &burn_dragon_train::TrainingGatesConfig,
    degeneracy: &crate::train::steps::OutputDegeneracyStats,
) -> bool {
    degeneracy.argmax_unique_fraction < gates.degeneracy_argmax_unique_min_fraction
        || degeneracy.distinct_2_fraction < gates.degeneracy_distinct_2_min_fraction
        || degeneracy.eos_fraction > gates.degeneracy_eos_max_fraction
        || degeneracy.repetition_fraction > gates.degeneracy_repetition_max_fraction
        || periodic_structure_degeneracy(gates, degeneracy)
}

pub(super) fn output_degeneracy_is_confident(
    _gates: &burn_dragon_train::TrainingGatesConfig,
    degeneracy: &crate::train::steps::OutputDegeneracyStats,
) -> bool {
    degeneracy.mean_max_probability >= 0.25
}

pub(super) fn uncertain_argmax_loop(
    gates: &burn_dragon_train::TrainingGatesConfig,
    degeneracy: &crate::train::steps::OutputDegeneracyStats,
) -> bool {
    output_diversity_degeneracy(gates, degeneracy)
        && !output_distribution_collapse(gates, degeneracy)
        && !output_degeneracy_is_confident(gates, degeneracy)
}

pub(super) fn hard_argmax_loop_collapse(
    gates: &burn_dragon_train::TrainingGatesConfig,
    degeneracy: &crate::train::steps::OutputDegeneracyStats,
) -> bool {
    degeneracy.repetition_fraction > gates.degeneracy_repetition_max_fraction
        || degeneracy.eos_fraction > gates.degeneracy_eos_max_fraction
        || (periodic_structure_high(gates, degeneracy)
            && output_distribution_collapse(gates, degeneracy)
            && (low_diversity_collapse(gates, degeneracy) || short_period_argmax_loop(degeneracy)))
}

pub(super) fn low_diversity_collapse(
    gates: &burn_dragon_train::TrainingGatesConfig,
    degeneracy: &crate::train::steps::OutputDegeneracyStats,
) -> bool {
    degeneracy.argmax_unique_fraction < gates.degeneracy_argmax_unique_min_fraction
        || degeneracy.distinct_2_fraction < gates.degeneracy_distinct_2_min_fraction
}

pub(super) fn periodic_structure_high(
    gates: &burn_dragon_train::TrainingGatesConfig,
    degeneracy: &crate::train::steps::OutputDegeneracyStats,
) -> bool {
    degeneracy.period_2_fraction > gates.degeneracy_period_2_max_fraction
        || degeneracy.period_3_fraction > gates.degeneracy_period_3_max_fraction
        || degeneracy.max_period_2_to_16_fraction > gates.degeneracy_period_2_to_16_max_fraction
        || degeneracy.max_period_2_to_64_fraction > gates.degeneracy_period_2_to_64_max_fraction
}

pub(super) fn periodic_structure_degeneracy(
    gates: &burn_dragon_train::TrainingGatesConfig,
    degeneracy: &crate::train::steps::OutputDegeneracyStats,
) -> bool {
    periodic_structure_high(gates, degeneracy)
        && (low_diversity_collapse(gates, degeneracy)
            || output_distribution_collapse(gates, degeneracy)
            || degeneracy.eos_fraction > gates.degeneracy_eos_max_fraction
            || degeneracy.repetition_fraction > gates.degeneracy_repetition_max_fraction
            || short_period_argmax_loop(degeneracy))
}

pub(super) fn short_period_argmax_loop(
    degeneracy: &crate::train::steps::OutputDegeneracyStats,
) -> bool {
    (2..=8).contains(&degeneracy.dominant_period_2_to_64)
}

pub(super) fn hard_output_collapse<B>(
    env: &TrainEnvironment<'_, B>,
    degeneracy: &crate::train::steps::OutputDegeneracyStats,
) -> bool
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    hard_output_collapse_for_gates(&env.training.gates, degeneracy)
}

pub(super) fn hard_output_collapse_for_gates(
    gates: &burn_dragon_train::TrainingGatesConfig,
    degeneracy: &crate::train::steps::OutputDegeneracyStats,
) -> bool {
    (output_distribution_collapse(gates, degeneracy)
        && (low_diversity_collapse(gates, degeneracy)
            || hard_argmax_loop_collapse(gates, degeneracy)))
        || (output_degeneracy_is_confident(gates, degeneracy)
            && hard_argmax_loop_collapse(gates, degeneracy))
}

pub(super) fn emit_policy_gate<B>(
    env: &TrainEnvironment<'_, B>,
    bus: &TrainingEventBus,
    gate: &str,
    epoch: usize,
    absolute_step: usize,
    message: String,
) where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    emit_policy_gate_with_action(
        env,
        gate,
        TrainingGateAction::Alert,
        TrainingGateSeverity::Warning,
        message,
        TrainingEventContext {
            epoch,
            absolute_step,
            bus,
        },
    );
}

pub(super) fn emit_policy_gate_with_action<B>(
    env: &TrainEnvironment<'_, B>,
    gate: &str,
    action: TrainingGateAction,
    severity: TrainingGateSeverity,
    message: String,
    event: TrainingEventContext<'_>,
) where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let TrainingEventContext {
        epoch,
        absolute_step,
        bus,
    } = event;
    let _ = bus.send_gate_event(TrainingGateEvent {
        run_id: env.run_name.to_string().into(),
        gate: gate.to_string(),
        action,
        severity,
        epoch: Some(epoch),
        absolute_step: Some(absolute_step),
        message,
    });
}

pub(super) fn mean_scalar_from_loss<B: BackendTrait>(tensor: Tensor<B, 1>) -> f64 {
    let values = tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("loss tensor to vec");
    if values.is_empty() {
        0.0
    } else {
        values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64
    }
}

pub(super) fn historical_best_validation(
    run_dir: &Path,
    max_epoch: usize,
) -> HistoricalBestValidation {
    let path = run_dir.join("events").join("training_events.jsonl");
    let Ok(file) = fs::File::open(&path) else {
        return HistoricalBestValidation::default();
    };

    let mut best_loss = None;
    let mut best_checkpoint_epoch = None;
    let mut best_checkpoint_loss = None;

    for line in BufReader::new(file).lines() {
        let Ok(line) = line else {
            continue;
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(event) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if event.get("type").and_then(|value| value.as_str()) != Some("validation_finished") {
            continue;
        }
        let Some(epoch) = event
            .get("epoch")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize)
        else {
            continue;
        };
        if epoch > max_epoch {
            continue;
        }
        let Some(loss) = event.get("loss").and_then(|value| value.as_f64()) else {
            continue;
        };
        if !loss.is_finite() {
            continue;
        }

        if best_loss.is_none_or(|best| loss < best) {
            best_loss = Some(loss);
        }
        if run_dir
            .join("checkpoint")
            .join(format!("model-{epoch}.bin"))
            .is_file()
            && best_checkpoint_loss.is_none_or(|best| loss < best)
        {
            best_checkpoint_loss = Some(loss);
            best_checkpoint_epoch = Some(epoch);
        }
    }

    HistoricalBestValidation {
        best_loss,
        best_checkpoint_epoch,
    }
}

pub(super) fn save_dragon_model_checkpoint<B>(
    run_dir: &Path,
    epoch: usize,
    model: &DragonModel<B>,
) -> Result<()>
where
    B: BackendTrait + Clone + 'static,
{
    let checkpoint_dir = run_dir.join("checkpoint");
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    FileCheckpointer::new(recorder, &checkpoint_dir, "model")
        .save(epoch, model.clone().into_record())
        .with_context(|| {
            format!(
                "failed to save dynamic Dragon model checkpoint {epoch} in {}",
                checkpoint_dir.display()
            )
        })?;
    Ok(())
}

pub(super) struct DragonTrainingCheckpointContext<'a> {
    pub(super) run_dir: &'a Path,
    pub(super) epoch: usize,
    pub(super) completed_steps: usize,
    pub(super) model_config: &'a DragonConfig,
    pub(super) dynamics_control: &'a ActiveDynamicsControl,
}

pub(super) fn save_dragon_training_state_checkpoint<B, S>(
    context: DragonTrainingCheckpointContext<'_>,
    model: &LanguageTrainModel<B>,
    optimizer: &crate::train::continual_backprop::LanguageOptimizer<B>,
    scheduler: &S,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    S: LrScheduler + Clone + 'static,
{
    let DragonTrainingCheckpointContext {
        run_dir,
        epoch,
        completed_steps,
        model_config,
        dynamics_control,
    } = context;
    save_dragon_model_checkpoint(run_dir, epoch, &model.model)?;
    save_runtime_state_checkpoint(run_dir, epoch, model)?;
    crate::train::manifest::save_experiment_checkpoint_progress(run_dir, epoch, completed_steps)?;
    let checkpoint_dir = run_dir.join("checkpoint");
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    FileCheckpointer::new(recorder.clone(), &checkpoint_dir, "optimizer")
        .save(epoch, optimizer.to_record())
        .with_context(|| {
            format!(
                "failed to save dynamic Dragon optimizer checkpoint {epoch} in {}",
                checkpoint_dir.display()
            )
        })?;
    FileCheckpointer::new(recorder, &checkpoint_dir, "scheduler")
        .save(epoch, scheduler.to_record::<B>())
        .with_context(|| {
            format!(
                "failed to save dynamic Dragon scheduler checkpoint {epoch} in {}",
                checkpoint_dir.display()
            )
        })?;
    let dynamics_path = checkpoint_dir.join(format!("dynamics-{epoch}.json"));
    fs::write(
        &dynamics_path,
        serde_json::to_vec_pretty(dynamics_control).context("serialize dynamics control")?,
    )
    .with_context(|| format!("failed to save {}", dynamics_path.display()))?;
    let model_config_path = checkpoint_dir.join(format!("model-config-{epoch}.json"));
    fs::write(
        &model_config_path,
        serde_json::to_vec_pretty(model_config).context("serialize Dragon model config")?,
    )
    .with_context(|| format!("failed to save {}", model_config_path.display()))?;
    Ok(())
}

pub(super) fn source_selection_state_checkpoint_path(run_dir: &Path, epoch: usize) -> PathBuf {
    run_dir
        .join("checkpoint")
        .join(format!("source-selection-state-{epoch}.json"))
}

pub(super) fn continual_learning_stability_checkpoint_path(
    run_dir: &Path,
    epoch: usize,
) -> PathBuf {
    run_dir
        .join("checkpoint")
        .join(format!("stability-{epoch}.json"))
}

pub(super) fn save_continual_learning_stability_checkpoint(
    run_dir: &Path,
    epoch: usize,
    state: &ContinualLearningStabilityState,
) -> Result<()> {
    let path = continual_learning_stability_checkpoint_path(run_dir, epoch);
    let temporary = path.with_extension("json.tmp");
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(state).context("serialize continual-learning stability state")?,
    )
    .with_context(|| format!("write {}", temporary.display()))?;
    fs::rename(&temporary, &path)
        .with_context(|| format!("replace {} from {}", path.display(), temporary.display()))
}

pub(super) fn load_continual_learning_stability_checkpoint(
    run_dir: &Path,
    epoch: usize,
    require_exact: bool,
) -> Result<Option<ContinualLearningStabilityState>> {
    let path = continual_learning_stability_checkpoint_path(run_dir, epoch);
    if !path.is_file() {
        if require_exact {
            return Err(anyhow!(
                "exact resume requires continual-learning stability checkpoint {}",
                path.display()
            ));
        }
        return Ok(None);
    }
    serde_json::from_slice(&fs::read(&path).with_context(|| format!("read {}", path.display()))?)
        .with_context(|| format!("parse {}", path.display()))
        .map(Some)
}

pub(super) fn save_source_selection_state_checkpoint(
    run_dir: &Path,
    epoch: usize,
    absolute_step: usize,
    source_selection_dataset: Option<&Arc<Dataset>>,
) -> Result<()> {
    let Some(dataset) = source_selection_dataset else {
        return Ok(());
    };
    dataset
        .write_source_selection_state(
            &source_selection_state_checkpoint_path(run_dir, epoch),
            absolute_step,
        )
        .with_context(|| {
            format!(
                "failed to save source-selection state checkpoint for epoch {epoch} in {}",
                run_dir.display()
            )
        })?;
    Ok(())
}

pub(super) fn checkpoint_artifact_epoch(name: &str) -> Option<usize> {
    for (prefix, suffix) in [
        ("model-", ".bin"),
        ("optimizer-", ".bin"),
        ("scheduler-", ".bin"),
        ("dynamics-", ".json"),
        ("model-config-", ".json"),
        ("runtime-state-", ".bin"),
        ("teacher-model-", ".bin"),
        ("context-routing-", ".json"),
        ("context-stream-states-", ".bin"),
        ("source-selection-state-", ".json"),
        ("stability-", ".json"),
        ("training-ecs-state-", ".json"),
    ] {
        if let Some(epoch) = name
            .strip_prefix(prefix)
            .and_then(|value| value.strip_suffix(suffix))
            .and_then(|value| value.parse::<usize>().ok())
        {
            return Some(epoch);
        }
    }
    if let Some((prefix, epoch)) = name.strip_suffix(".bin")?.rsplit_once('-')
        && prefix.starts_with("context-optimizer-")
    {
        return epoch.parse().ok();
    }
    None
}

pub(super) fn prune_dragon_model_checkpoints(
    run_dir: &Path,
    current_epoch: usize,
    protected_epochs: &[Option<usize>],
) -> Result<()> {
    let checkpoint_dir = run_dir.join("checkpoint");
    let Ok(entries) = fs::read_dir(&checkpoint_dir) else {
        return Ok(());
    };
    let mut keep_epochs = BTreeSet::new();
    keep_epochs.extend(
        current_epoch
            .saturating_sub(CHECKPOINT_KEEP_LAST - 1)
            .max(1)..=current_epoch,
    );
    for epoch in protected_epochs.iter().flatten().copied() {
        keep_epochs.insert(epoch);
    }

    for entry in entries {
        let path = entry?.path();
        let Some(epoch) = path
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(checkpoint_artifact_epoch)
        else {
            continue;
        };
        if !keep_epochs.contains(&epoch) {
            fs::remove_file(&path)
                .with_context(|| format!("failed to prune checkpoint {}", path.display()))?;
        }
    }
    Ok(())
}

pub(super) fn load_dragon_model_checkpoint<B>(
    run_dir: &Path,
    epoch: usize,
    model_config: &DragonConfig,
    device: &B::Device,
) -> Result<DragonModel<B>>
where
    B: BackendTrait + Clone + 'static,
{
    let checkpoint_dir = run_dir.join("checkpoint");
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let record = FileCheckpointer::new(recorder, &checkpoint_dir, "model")
        .restore(epoch, device)
        .with_context(|| {
            format!(
                "failed to restore dynamic Dragon model checkpoint {epoch} from {}",
                checkpoint_dir.display()
            )
        })?;
    Ok(DragonModel::<B>::new(model_config.clone(), device).load_record(record))
}

pub(super) fn load_dragon_training_state_checkpoint<B, S>(
    run_dir: &Path,
    epoch: usize,
    model_config: &DragonConfig,
    device: &B::Device,
    optimizer: &mut crate::train::continual_backprop::LanguageOptimizer<B>,
    scheduler: &mut S,
    dynamics_control: &mut ActiveDynamicsControl,
) -> Result<(DragonModel<B>, Option<DragonConfig>)>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    S: LrScheduler + Clone + 'static,
{
    let checkpoint_dir = run_dir.join("checkpoint");
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let model_config_path = checkpoint_dir.join(format!("model-config-{epoch}.json"));
    let saved_model_config = if model_config_path.is_file() {
        let bytes = fs::read(&model_config_path)
            .with_context(|| format!("failed to read {}", model_config_path.display()))?;
        Some(
            serde_json::from_slice::<DragonConfig>(&bytes)
                .with_context(|| format!("failed to parse {}", model_config_path.display()))?,
        )
    } else {
        None
    };
    let checkpoint_model_config = saved_model_config.as_ref().unwrap_or(model_config);
    let model = load_dragon_model_checkpoint(run_dir, epoch, checkpoint_model_config, device)?;

    let optimizer_path = checkpoint_dir.join(format!("optimizer-{epoch}.bin"));
    if optimizer_path.is_file() {
        let record = FileCheckpointer::new(recorder.clone(), &checkpoint_dir, "optimizer")
            .restore(epoch, device)
            .with_context(|| {
                format!(
                    "failed to restore dynamic Dragon optimizer checkpoint {epoch} from {}",
                    checkpoint_dir.display()
                )
            })?;
        *optimizer = optimizer.clone().load_record(record);
    }

    let scheduler_path = checkpoint_dir.join(format!("scheduler-{epoch}.bin"));
    if scheduler_path.is_file() {
        let record = FileCheckpointer::new(recorder, &checkpoint_dir, "scheduler")
            .restore(epoch, device)
            .with_context(|| {
                format!(
                    "failed to restore dynamic Dragon scheduler checkpoint {epoch} from {}",
                    checkpoint_dir.display()
                )
            })?;
        *scheduler = scheduler.clone().load_record::<B>(record);
    }

    let dynamics_path = checkpoint_dir.join(format!("dynamics-{epoch}.json"));
    if dynamics_path.is_file() {
        let bytes = fs::read(&dynamics_path)
            .with_context(|| format!("failed to read {}", dynamics_path.display()))?;
        *dynamics_control = serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse {}", dynamics_path.display()))?;
    }

    Ok((model, saved_model_config))
}
