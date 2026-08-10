use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use burn_dragon_train::train::events::{
    BurnInterrupterControl, DynamicsEquilibriumPlugin, ModelCapacityConfig, ModelCapacityState,
    PredictiveCodingSample, TrainingEventBusConfig, TrainingEventMetricLogger, TrainingRunContext,
    TrainingRunOptions, TrainingRuntimeThread,
};
use burn_ecs::bevy_ecs;
use burn_ecs::prelude::{
    App, Component, IntoScheduleConfigs, MessageReader, MessageWriter, Plugin, Query, Res,
    SourceSelectionBucketMetric, SourceSelectionCapabilityCoverageMetric,
    SourceSelectionGroupMetric, SourceSelectionSample, TrainingAppExt, TrainingMetricSample,
    TrainingMetricSplit, TrainingPlugins, TrainingRunCheckpointExt, TrainingRunId,
    TrainingRunRegistry, TrainingRunStateCheckpoint, TrainingSet, Update,
};

use crate::config::{LocalPredictiveCodingSolver, TrainingAlgorithm, TrainingHyperparameters};
use crate::dataset::Dataset;
use crate::train::dynamics::{DragonDynamicsControlPlugin, DragonDynamicsControlSlot};
use crate::train::neuron_scaling::{DragonNeuronScalingPlugin, NeuronScaleRequestSlot};

#[derive(Clone, Component)]
pub struct RuliadSourceSelectionConfig {
    dataset: Arc<Dataset>,
    source_selection_every_steps: usize,
}

impl RuliadSourceSelectionConfig {
    pub fn new(dataset: Arc<Dataset>, source_selection_every_steps: usize) -> Self {
        Self {
            dataset,
            source_selection_every_steps: source_selection_every_steps.max(1),
        }
    }
}

pub struct TrainingEventHandles {
    pub interrupter: burn_train::Interrupter,
    pub metric_logger: TrainingEventMetricLogger,
}

const TRAINING_ECS_STATE_PREFIX: &str = "training-ecs-state";

pub(crate) fn training_event_state_checkpoint_path(run_dir: &Path, epoch: usize) -> PathBuf {
    run_dir
        .join("checkpoint")
        .join(format!("{TRAINING_ECS_STATE_PREFIX}-{epoch}.json"))
}

fn load_training_event_state_checkpoint(
    run_name: &str,
    run_dir: &Path,
    training: &TrainingHyperparameters,
) -> Result<Option<TrainingRunStateCheckpoint>> {
    if !matches!(
        training.launch_mode,
        burn_dragon_train::train::pipeline::TrainingLaunchMode::ResumeExactRun
    ) {
        return Ok(None);
    }
    let (_, epoch) = crate::checkpoint::resolve_checkpoint_base(
        &run_dir.join("checkpoint"),
        training.resume_checkpoint_epoch,
    )?;
    let path = training_event_state_checkpoint_path(run_dir, epoch);
    if !path.is_file() {
        return Err(anyhow!(
            "exact resume requires training ECS state checkpoint {}",
            path.display()
        ));
    }
    let checkpoint: TrainingRunStateCheckpoint = serde_json::from_slice(
        &fs::read(&path).with_context(|| format!("read {}", path.display()))?,
    )
    .with_context(|| format!("parse {}", path.display()))?;
    if checkpoint.run_id.as_str() != run_name {
        return Err(anyhow!(
            "training ECS checkpoint {} belongs to run {}, expected {}",
            path.display(),
            checkpoint.run_id,
            run_name
        ));
    }
    Ok(Some(checkpoint))
}

pub(crate) fn save_training_event_state_checkpoint(
    handles: &TrainingEventHandles,
    run_name: &str,
    run_dir: &Path,
    epoch: usize,
) -> Result<()> {
    let checkpoint = handles.metric_logger.bus().snapshot_run_state(run_name)?;
    let path = training_event_state_checkpoint_path(run_dir, epoch);
    let temporary = path.with_extension("json.tmp");
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(&checkpoint).context("serialize training ECS state")?,
    )
    .with_context(|| format!("write {}", temporary.display()))?;
    fs::rename(&temporary, &path)
        .with_context(|| format!("replace {} from {}", path.display(), temporary.display()))
}

/// Run-scoped observable state for predictive-context routing.
///
/// GPU masks, recurrent tensors, and optimizers remain owned by the training thread. This ECS
/// component is the control-plane projection used by dashboards, gates, and external observers,
/// so multiple runs can coexist without sharing context lifecycle state.
#[derive(Clone, Component, Debug, Default, PartialEq)]
pub struct PredictiveContextRoutingTelemetryState {
    pub current_context: usize,
    pub current_generation: u64,
    pub known_contexts: usize,
    pub probes: u64,
    pub creations: u64,
    pub replacements: u64,
    pub novelty_deferrals: u64,
    pub probe_tokens: u64,
    pub selected_loss: Option<f64>,
    pub last_absolute_step: usize,
}

#[derive(Clone, Component)]
struct LocalPredictiveCodingTelemetryConfig {
    profile: crate::train::local_predictive_coding::LocalPredictiveCodingProfile,
    training_algorithm: TrainingAlgorithm,
    solver: LocalPredictiveCodingSolver,
    terminal_criterion: crate::config::LocalPredictiveCodingTerminalCriterion,
    learning_schedule: burn_pc::PcLearningSchedule,
    temporal_credit: burn_pc::PcTemporalCreditConfig,
    execution_contract: burn_pc::PcExecutionContract,
}

fn local_predictive_coding_event_contract(
    solver: LocalPredictiveCodingSolver,
    learning_schedule: burn_pc::PcLearningSchedule,
) -> (&'static str, &'static str) {
    if matches!(learning_schedule, burn_pc::PcLearningSchedule::Incremental) {
        return match solver {
            LocalPredictiveCodingSolver::SynchronousEquilibrium => (
                "local_incremental_factor_vjp_v1",
                "interleaved_synchronous_activities",
            ),
            LocalPredictiveCodingSolver::ReverseGaussSeidel => (
                "local_incremental_factor_vjp_v1",
                "interleaved_gauss_seidel_activities",
            ),
            LocalPredictiveCodingSolver::ErrorEquilibrium
            | LocalPredictiveCodingSolver::FixedPrediction
            | LocalPredictiveCodingSolver::AugmentedLagrangian
            | LocalPredictiveCodingSolver::LayerLocalPrediction
            | LocalPredictiveCodingSolver::DirectKolenPollack
            | LocalPredictiveCodingSolver::AmortizedAdjoint
            | LocalPredictiveCodingSolver::FirstOrderAdjoint => {
                unreachable!("validated incremental PC solver")
            }
        };
    }
    match solver {
        LocalPredictiveCodingSolver::SynchronousEquilibrium => {
            ("local_factor_vjp_v1", "equilibrium_layer_activities")
        }
        LocalPredictiveCodingSolver::ReverseGaussSeidel => (
            "local_prospective_gauss_seidel_v1",
            "settled_layer_activities",
        ),
        LocalPredictiveCodingSolver::AugmentedLagrangian => (
            "local_augmented_lagrangian_v1",
            "primal_dual_layer_activities",
        ),
        LocalPredictiveCodingSolver::ErrorEquilibrium => {
            ("local_error_equilibrium_v1", "inferred_error_coordinates")
        }
        LocalPredictiveCodingSolver::FixedPrediction => {
            ("local_fixed_prediction_v1", "fixed_feedforward_predictions")
        }
        LocalPredictiveCodingSolver::LayerLocalPrediction => {
            ("local_layer_prediction_v1", "detached_layer_predictions")
        }
        LocalPredictiveCodingSolver::DirectKolenPollack => (
            "local_direct_kolen_pollack_v1",
            "tied_direct_feedback_activities",
        ),
        LocalPredictiveCodingSolver::AmortizedAdjoint => (
            "local_amortized_adjoint_v1",
            "calibrated_layer_output_adjoints",
        ),
        LocalPredictiveCodingSolver::FirstOrderAdjoint => (
            "local_first_order_adjoint_v1",
            "parallel_residual_jacobian_adjoints",
        ),
    }
}

fn effective_predictive_coding_temporal_credit(
    algorithm: TrainingAlgorithm,
    backprop_window_chunks: usize,
    local_credit: burn_pc::PcTemporalCreditConfig,
) -> burn_pc::PcTemporalCreditConfig {
    if matches!(algorithm, TrainingAlgorithm::Backpropagation) && backprop_window_chunks > 1 {
        burn_pc::PcTemporalCreditConfig {
            mode: burn_pc::PcTemporalCreditMode::ExactWindow,
            window_chunks: backprop_window_chunks,
        }
    } else {
        local_credit
    }
}

pub fn train_loss_metric_frequency(
    training: &TrainingHyperparameters,
    source_selection_dataset: Option<&Arc<Dataset>>,
) -> usize {
    if source_selection_dataset
        .as_ref()
        .is_some_and(|dataset| dataset.uses_live_source_selection())
    {
        training.events.source_selection_every_steps.max(1)
    } else {
        training.log_frequency.max(1)
    }
}

pub fn build_training_event_handles(
    run_name: &str,
    run_dir: &std::path::Path,
    steps_per_epoch: usize,
    training: &TrainingHyperparameters,
    source_selection_dataset: Option<Arc<Dataset>>,
    neuron_scaling_slot: Option<(usize, NeuronScaleRequestSlot)>,
    dynamics_control_slot: Option<DragonDynamicsControlSlot>,
) -> Result<TrainingEventHandles> {
    build_training_event_handles_with_local_predictive_coding(
        run_name,
        run_dir,
        steps_per_epoch,
        training,
        source_selection_dataset,
        neuron_scaling_slot,
        dynamics_control_slot,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn build_training_event_handles_with_local_predictive_coding(
    run_name: &str,
    run_dir: &std::path::Path,
    steps_per_epoch: usize,
    training: &TrainingHyperparameters,
    source_selection_dataset: Option<Arc<Dataset>>,
    neuron_scaling_slot: Option<(usize, NeuronScaleRequestSlot)>,
    dynamics_control_slot: Option<DragonDynamicsControlSlot>,
    local_predictive_coding_profile: Option<
        crate::train::local_predictive_coding::LocalPredictiveCodingProfile,
    >,
) -> Result<TrainingEventHandles> {
    let interrupter = burn_train::Interrupter::new();
    let control = BurnInterrupterControl::new(interrupter.clone());
    let run = TrainingRunContext::new(run_name, run_name, run_dir, steps_per_epoch);
    let source_selection = source_selection_dataset
        .filter(|dataset| dataset.uses_live_source_selection())
        .map(|dataset| {
            RuliadSourceSelectionConfig::new(dataset, training.events.source_selection_every_steps)
        });
    let local_predictive_coding = local_predictive_coding_profile
        .filter(|_| {
            matches!(training.algorithm, TrainingAlgorithm::PredictiveCoding)
                || matches!(
                    training.local_predictive_coding.terminal_criterion,
                    crate::config::LocalPredictiveCodingTerminalCriterion::RuliadVerifierSet
                )
        })
        .map(|profile| {
            profile.reset();
            let temporal_credit = effective_predictive_coding_temporal_credit(
                training.algorithm,
                training.tbptt_credit_window_chunks,
                training.local_predictive_coding.temporal_credit,
            );
            LocalPredictiveCodingTelemetryConfig {
                profile,
                training_algorithm: training.algorithm,
                solver: training.local_predictive_coding.solver,
                terminal_criterion: training.local_predictive_coding.terminal_criterion,
                learning_schedule: training.local_predictive_coding.learning_schedule,
                temporal_credit,
                execution_contract: training.local_predictive_coding.execution_contract(),
            }
        });
    let neuron_scaling = (training
        .neuron_scaling
        .enabled
        .then_some(training.neuron_scaling.clone()))
    .zip(neuron_scaling_slot);
    let capacity = neuron_scaling
        .as_ref()
        .map(|(config, (current_latent_total, _))| ModelCapacityConfig {
            policy: crate::train::neuron_scaling::capacity_policy_from_neuron_scaling(config),
            capacity: Some(ModelCapacityState::new(
                *current_latent_total,
                config.max_latent_total.max(*current_latent_total),
            )),
        });
    let options = TrainingRunOptions {
        sinks: training.events.clone(),
        gates: training.gates.clone(),
        dynamics: training
            .dynamics
            .enabled
            .then_some(training.dynamics.clone()),
        capacity,
    };
    let dynamics_enabled = training.dynamics.enabled;
    let context_routing_enabled = training.predictive_context_routing.enabled;
    let restored_event_state = load_training_event_state_checkpoint(run_name, run_dir, training)?;
    let event_thread = TrainingRuntimeThread::spawn(
        move || {
            let mut app = App::new();
            app.add_plugins(TrainingPlugins)
                .insert_training_control(control);
            if source_selection.is_some() {
                app.add_plugins(RuliadSourceSelectionTelemetryPlugin);
            }
            if local_predictive_coding.is_some() {
                app.add_plugins(DragonLocalPredictiveCodingTelemetryPlugin);
            }
            if context_routing_enabled {
                app.add_plugins(DragonPredictiveContextRoutingTelemetryPlugin);
            }
            if neuron_scaling.is_some() {
                app.add_plugins(DragonNeuronScalingPlugin);
            }
            if dynamics_enabled {
                app.add_plugins(DynamicsEquilibriumPlugin);
                if dynamics_control_slot.is_some() {
                    app.add_plugins(DragonDynamicsControlPlugin);
                }
            }
            let run_entity = app.try_add_training_run_with(run, options)?;
            if let Some(checkpoint) = restored_event_state {
                app.restore_training_run_state(checkpoint)?;
            }
            if let Some(source_selection) = source_selection {
                app.world_mut()
                    .entity_mut(run_entity)
                    .insert(source_selection);
            }
            if let Some(local_predictive_coding) = local_predictive_coding {
                app.world_mut()
                    .entity_mut(run_entity)
                    .insert(local_predictive_coding);
            }
            if context_routing_enabled {
                app.world_mut()
                    .entity_mut(run_entity)
                    .insert(PredictiveContextRoutingTelemetryState::default());
            }
            if let Some((config, (_, request_slot))) = neuron_scaling {
                app.world_mut().entity_mut(run_entity).insert(
                    crate::train::neuron_scaling::DragonNeuronScalingState::new(
                        config,
                        request_slot,
                    ),
                );
            }
            if let Some(slot) = dynamics_control_slot {
                app.world_mut().entity_mut(run_entity).insert(slot);
            }
            Ok(app)
        },
        TrainingEventBusConfig::default(),
    )?;
    let metric_logger =
        TrainingEventMetricLogger::with_thread(event_thread, run_name, steps_per_epoch);
    Ok(TrainingEventHandles {
        interrupter,
        metric_logger,
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct DragonPredictiveContextRoutingTelemetryPlugin;

impl Plugin for DragonPredictiveContextRoutingTelemetryPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            record_predictive_context_routing_from_metrics.in_set(TrainingSet::Telemetry),
        );
    }
}

fn record_predictive_context_routing_from_metrics(
    mut metrics: MessageReader<TrainingMetricSample>,
    registry: Res<TrainingRunRegistry>,
    mut runs: Query<&mut PredictiveContextRoutingTelemetryState>,
) {
    for sample in metrics.read() {
        if sample.split != TrainingMetricSplit::Train
            || !sample.name.starts_with("Predictive Context ")
        {
            continue;
        }
        let Some(mut state) = registry.get_query_mut(&sample.run_id, &mut runs) else {
            continue;
        };
        apply_predictive_context_routing_metric(&mut state, sample);
    }
}

fn apply_predictive_context_routing_metric(
    state: &mut PredictiveContextRoutingTelemetryState,
    sample: &TrainingMetricSample,
) {
    state.last_absolute_step = sample.absolute_step;
    match sample.name.as_str() {
        "Predictive Context Index" => state.current_context = sample.value.max(0.0) as usize,
        "Predictive Context Generation" => {
            state.current_generation = sample.value.max(0.0) as u64;
        }
        "Predictive Context Count" => state.known_contexts = sample.value.max(0.0) as usize,
        "Predictive Context Created" if sample.value > 0.5 => {
            state.creations = state.creations.saturating_add(1);
        }
        "Predictive Context Replaced" if sample.value > 0.5 => {
            state.replacements = state.replacements.saturating_add(1);
        }
        "Predictive Context Novelty Deferred" if sample.value > 0.5 => {
            state.novelty_deferrals = state.novelty_deferrals.saturating_add(1);
        }
        "Predictive Context Probe Tokens" => {
            state.probes = state.probes.saturating_add(1);
            state.probe_tokens = state
                .probe_tokens
                .saturating_add(sample.value.max(0.0) as u64);
        }
        "Predictive Context Selected Loss" => state.selected_loss = Some(sample.value),
        _ => {}
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct DragonLocalPredictiveCodingTelemetryPlugin;

impl Plugin for DragonLocalPredictiveCodingTelemetryPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            record_local_predictive_coding_from_loss.in_set(TrainingSet::Telemetry),
        );
    }
}

fn record_local_predictive_coding_from_loss(
    mut metrics: MessageReader<TrainingMetricSample>,
    registry: Res<TrainingRunRegistry>,
    runs: Query<&LocalPredictiveCodingTelemetryConfig>,
    mut output: MessageWriter<PredictiveCodingSample>,
) {
    for sample in metrics.read() {
        if sample.split != TrainingMetricSplit::Train
            || (sample.name != "Loss" && sample.name != "Stream Warm Loss")
        {
            continue;
        }
        let Some(config) = registry.get_query(&sample.run_id, &runs) else {
            continue;
        };
        let snapshot = config.profile.take();
        if snapshot.steps == 0 {
            continue;
        }
        let global_backprop_control = matches!(
            config.training_algorithm,
            TrainingAlgorithm::Backpropagation
        );
        let (learning_contract, observation_contract) = if global_backprop_control {
            (
                "global_backpropagation_control_v1",
                "verifier_terminal_global_autodiff",
            )
        } else {
            local_predictive_coding_event_contract(config.solver, config.learning_schedule)
        };
        output.write(PredictiveCodingSample {
            run_id: sample.run_id.clone(),
            epoch: Some(sample.epoch),
            absolute_step: sample.absolute_step,
            optimizer_step: sample.absolute_step,
            learning_contract: learning_contract.to_string(),
            execution_contract_version: config.execution_contract.version,
            activity_derivative_contract: if global_backprop_control {
                "global_autodiff".to_string()
            } else {
                config
                    .execution_contract
                    .activity_derivatives
                    .as_str()
                    .to_string()
            },
            parameter_derivative_contract: if global_backprop_control {
                "global_autodiff".to_string()
            } else {
                config
                    .execution_contract
                    .parameter_derivatives
                    .as_str()
                    .to_string()
            },
            global_autodiff_graph: global_backprop_control,
            observation_contract: observation_contract.to_string(),
            deployment_aligned: false,
            chunks_seen: snapshot.steps as usize,
            chunks_corrected: snapshot.steps as usize,
            inference_steps: snapshot.inference_steps as usize,
            dual_steps: snapshot.dual_steps as usize,
            skipped_empty_state: 0,
            factors: snapshot.factors as usize,
            local_vjp_calls: snapshot.local_vjp_calls as usize,
            temporal_state_vjp_calls: snapshot.temporal_state_vjp_calls as usize,
            fused_temporal_vjp_calls: snapshot.fused_temporal_vjp_calls as usize,
            temporal_credit_mode: config.temporal_credit.mode.as_str().to_string(),
            temporal_window_chunks: config.temporal_credit.window_chunks,
            global_backward_calls: snapshot.global_backward_calls as usize,
            gradient_tensors: snapshot.gradient_tensors as usize,
            direct_forward_updates: snapshot.direct_forward_updates as usize,
            feedback_parameter_updates: snapshot.feedback_parameter_updates as usize,
            adjoint_teacher_updates: snapshot.adjoint_teacher_updates as usize,
            adjoint_local_updates: snapshot.adjoint_local_updates as usize,
            adjoint_calibration_samples: snapshot.adjoint_calibration_samples as usize,
            adjoint_calibration_loss: snapshot.last_adjoint_calibration_loss,
            adjoint_cosine_alignment: snapshot.last_adjoint_cosine_alignment,
            adjoint_prediction_teacher_norm_ratio: snapshot
                .last_adjoint_prediction_teacher_norm_ratio,
            adjoint_update_rms: snapshot.last_adjoint_update_rms,
            local_parameter_update_intents: snapshot.parameter_updates as usize,
            parameter_updates: snapshot.optimizer_updates as usize,
            terminal_factor_kind: match config.terminal_criterion {
                crate::config::LocalPredictiveCodingTerminalCriterion::NextToken => {
                    "next_token".to_string()
                }
                crate::config::LocalPredictiveCodingTerminalCriterion::RuliadVerifierSet => {
                    "alternating_next_token_ruliad_verifier_set".to_string()
                }
            },
            structured_terminal_steps: snapshot.structured_terminal_steps as usize,
            structured_terminal_skipped_steps: snapshot.structured_terminal_skipped_steps as usize,
            structured_terminal_groups: snapshot.structured_terminal_groups as usize,
            structured_terminal_rows: snapshot.structured_terminal_rows as usize,
            energy_before: snapshot.last_energy_before,
            energy_after: snapshot.last_energy_after,
            energy_delta: snapshot
                .last_energy_before
                .zip(snapshot.last_energy_after)
                .map(|(before, after)| before - after),
            grad_norm_mean: snapshot.last_grad_norm_mean,
            grad_norm_max: snapshot.last_grad_norm_max,
            delta_rms_mean: snapshot.last_delta_rms_mean,
            clip_fraction_mean: snapshot.last_clip_fraction_mean,
            constraint_rms: snapshot.last_constraint_rms,
            dual_rms: snapshot.last_dual_rms,
            composite_signal_rms: snapshot.last_composite_signal_rms,
            amortization_components: 0,
            amortization_loss: None,
            elapsed_ms: snapshot.elapsed_ns as f64 / 1_000_000.0,
        });
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RuliadSourceSelectionTelemetryPlugin;

impl Plugin for RuliadSourceSelectionTelemetryPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            record_ruliad_source_selection_from_loss.in_set(TrainingSet::Telemetry),
        );
    }
}

fn record_ruliad_source_selection_from_loss(
    mut metrics: MessageReader<TrainingMetricSample>,
    registry: Res<TrainingRunRegistry>,
    source_selection_runs: Query<&RuliadSourceSelectionConfig>,
    mut source_selection_events: MessageWriter<SourceSelectionSample>,
) {
    for sample in metrics.read() {
        if sample.split != TrainingMetricSplit::Train
            || (sample.name != "Loss" && sample.name != "Stream Warm Loss")
        {
            continue;
        }
        let Some(source_selection) = registry.get_query(&sample.run_id, &source_selection_runs)
        else {
            continue;
        };
        if sample.absolute_step % source_selection.source_selection_every_steps != 0 {
            continue;
        }
        let recorded_snapshot = source_selection
            .dataset
            .record_source_selection_loss(sample.absolute_step, sample.value as f32);
        let loss = recorded_snapshot.as_ref().map(|_| sample.value as f32);
        let snapshot = recorded_snapshot.or_else(|| {
            source_selection
                .dataset
                .source_selection_snapshot_at_step(sample.absolute_step)
        });
        let Some(snapshot) = snapshot else {
            continue;
        };
        source_selection_events.write(source_selection_sample_from_snapshot(
            sample.run_id.clone(),
            sample.absolute_step,
            loss,
            &snapshot,
        ));
    }
}

pub(crate) fn source_selection_sample_from_snapshot(
    run_id: impl Into<TrainingRunId>,
    absolute_step: usize,
    loss: Option<f32>,
    snapshot: &burn_dragon_universality::RuliadMetricSnapshot,
) -> SourceSelectionSample {
    let run_id = run_id.into();
    SourceSelectionSample {
        run_id,
        absolute_step,
        loss,
        entropy_bits: snapshot.sampler_entropy_bits as f64,
        active_candidate_count: snapshot.active_candidate_count,
        active_max_entropy_bits: snapshot.active_max_entropy_bits as f64,
        normalized_entropy: snapshot.normalized_sampler_entropy as f64,
        hash_noise_probability: snapshot.hash_noise_probability as f64,
        mean_loss: snapshot.mean_loss as f64,
        mean_learning_progress: snapshot.mean_learning_progress as f64,
        frontier_loss: snapshot.frontier_loss as f64,
        target_loss: snapshot.target_loss as f64,
        target_difficulty_score: snapshot.target_difficulty_score as f64,
        max_difficulty_level: snapshot.max_difficulty_level,
        active_max_difficulty_level: snapshot.active_max_difficulty_level,
        curriculum_released_max_difficulty_level: snapshot.curriculum_released_max_difficulty_level,
        materialized_frontier_edge: snapshot.max_difficulty_level,
        mean_difficulty_level: snapshot.mean_difficulty_level as f64,
        normalized_difficulty_score: snapshot.normalized_difficulty_score as f64,
        max_difficulty_probability: snapshot.max_difficulty_probability as f64,
        active_max_difficulty_probability: snapshot.active_max_difficulty_probability as f64,
        mastered_probability: snapshot.mastered_probability as f64,
        capability_feedback_probability: snapshot.capability_feedback_probability as f64,
        capability_verifier_ema: snapshot.capability_verifier_ema as f64,
        capability_completion_health_ema: snapshot.capability_completion_health_ema as f64,
        capability_schema_wrong_ema: snapshot.capability_schema_wrong_ema as f64,
        capability_malformed_ema: snapshot.capability_malformed_ema as f64,
        capability_missing_ema: snapshot.capability_missing_ema as f64,
        capability_lagging_probability: snapshot.capability_lagging_probability as f64,
        capability_frontier_allowed_max_difficulty: snapshot
            .capability_frontier_allowed_max_difficulty,
        capability_frontier_coverage: snapshot
            .capability_frontier_coverage
            .iter()
            .map(source_selection_capability_coverage_metric)
            .collect(),
        frontier_extension_count: snapshot.frontier_extension_count,
        frontier_saturated: snapshot.frontier_saturated,
        unbounded_frontier: snapshot.frontier_unbounded,
        top_buckets: snapshot
            .top_buckets
            .iter()
            .map(|bucket| SourceSelectionBucketMetric {
                label: bucket.label.clone(),
                family: bucket.family.clone(),
                task_kind: bucket.task_kind.clone(),
                difficulty_level: bucket.difficulty_level,
                probability: bucket.probability as f64,
                loss_ema: bucket.loss_ema as f64,
                previous_loss_ema: bucket.previous_loss_ema as f64,
                learning_progress: bucket.learning_progress as f64,
                mastered: bucket.mastered,
                capability_feedback_count: bucket.capability_feedback_count,
                capability_verifier_ema: bucket.capability_verifier_ema as f64,
                capability_completion_health_ema: bucket.capability_completion_health_ema as f64,
                capability_schema_wrong_ema: bucket.capability_schema_wrong_ema as f64,
                capability_malformed_ema: bucket.capability_malformed_ema as f64,
                capability_missing_ema: bucket.capability_missing_ema as f64,
                capability_lagging: bucket.capability_lagging,
            })
            .collect(),
        difficulty_buckets: snapshot
            .difficulty_buckets
            .iter()
            .map(source_selection_group_metric)
            .collect(),
        family_buckets: snapshot
            .family_buckets
            .iter()
            .map(source_selection_group_metric)
            .collect(),
        task_buckets: snapshot
            .task_buckets
            .iter()
            .map(source_selection_group_metric)
            .collect(),
        contract_buckets: snapshot
            .contract_buckets
            .iter()
            .map(source_selection_group_metric)
            .collect(),
        verifier_failures: snapshot.verifier_failures as u64,
    }
}

fn source_selection_capability_coverage_metric(
    coverage: &burn_dragon_universality::RuliadCapabilityCoverageMetric,
) -> SourceSelectionCapabilityCoverageMetric {
    SourceSelectionCapabilityCoverageMetric {
        difficulty_level: coverage.difficulty_level,
        candidate_coverage: coverage.candidate_coverage as f64,
        family_coverage: coverage.family_coverage as f64,
        task_coverage: coverage.task_coverage as f64,
        contract_coverage: coverage.contract_coverage as f64,
        observed_items: coverage.observed_items,
        mastered: coverage.mastered,
    }
}

fn source_selection_group_metric(
    group: &burn_dragon_universality::RuliadGroupMetric,
) -> SourceSelectionGroupMetric {
    SourceSelectionGroupMetric {
        label: group.label.clone(),
        candidate_count: group.candidate_count,
        probability: group.probability as f64,
        mean_loss: group.mean_loss as f64,
        learning_progress: group.learning_progress as f64,
        mastered_probability: group.mastered_probability as f64,
        mean_difficulty_level: group.mean_difficulty_level as f64,
        capability_feedback_probability: group.capability_feedback_probability as f64,
        capability_verifier_ema: group.capability_verifier_ema as f64,
        capability_completion_health_ema: group.capability_completion_health_ema as f64,
        capability_schema_wrong_ema: group.capability_schema_wrong_ema as f64,
        capability_malformed_ema: group.capability_malformed_ema as f64,
        capability_missing_ema: group.capability_missing_ema as f64,
        capability_lagging_probability: group.capability_lagging_probability as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_frontier_coverage_survives_the_ecs_event_boundary() {
        let metric = source_selection_capability_coverage_metric(
            &burn_dragon_universality::RuliadCapabilityCoverageMetric {
                difficulty_level: 7,
                candidate_coverage: 0.75,
                family_coverage: 1.0,
                task_coverage: 0.5,
                contract_coverage: 0.25,
                observed_items: 128,
                mastered: false,
            },
        );

        assert_eq!(metric.difficulty_level, 7);
        assert_eq!(metric.observed_items, 128);
        assert_eq!(metric.candidate_coverage, 0.75);
        assert_eq!(metric.family_coverage, 1.0);
        assert_eq!(metric.task_coverage, 0.5);
        assert_eq!(metric.contract_coverage, 0.25);
        assert!(!metric.mastered);
    }

    #[test]
    fn local_pc_event_contract_distinguishes_error_solvers() {
        assert_eq!(
            local_predictive_coding_event_contract(
                LocalPredictiveCodingSolver::SynchronousEquilibrium,
                burn_pc::PcLearningSchedule::Equilibrium,
            ),
            ("local_factor_vjp_v1", "equilibrium_layer_activities")
        );
        assert_eq!(
            local_predictive_coding_event_contract(
                LocalPredictiveCodingSolver::FixedPrediction,
                burn_pc::PcLearningSchedule::Equilibrium,
            ),
            ("local_fixed_prediction_v1", "fixed_feedforward_predictions")
        );
        assert_eq!(
            local_predictive_coding_event_contract(
                LocalPredictiveCodingSolver::ErrorEquilibrium,
                burn_pc::PcLearningSchedule::Equilibrium,
            ),
            ("local_error_equilibrium_v1", "inferred_error_coordinates")
        );
        assert_eq!(
            local_predictive_coding_event_contract(
                LocalPredictiveCodingSolver::LayerLocalPrediction,
                burn_pc::PcLearningSchedule::Equilibrium,
            ),
            ("local_layer_prediction_v1", "detached_layer_predictions")
        );
        assert_eq!(
            local_predictive_coding_event_contract(
                LocalPredictiveCodingSolver::FirstOrderAdjoint,
                burn_pc::PcLearningSchedule::Equilibrium,
            ),
            (
                "local_first_order_adjoint_v1",
                "parallel_residual_jacobian_adjoints"
            )
        );
        assert_eq!(
            local_predictive_coding_event_contract(
                LocalPredictiveCodingSolver::ReverseGaussSeidel,
                burn_pc::PcLearningSchedule::Incremental,
            ),
            (
                "local_incremental_factor_vjp_v1",
                "interleaved_gauss_seidel_activities"
            )
        );
    }

    #[test]
    fn bounded_backprop_reports_its_effective_temporal_credit_contract() {
        let detached = burn_pc::PcTemporalCreditConfig::default();
        let bounded = effective_predictive_coding_temporal_credit(
            TrainingAlgorithm::Backpropagation,
            2,
            detached,
        );
        assert_eq!(bounded.mode, burn_pc::PcTemporalCreditMode::ExactWindow);
        assert_eq!(bounded.window_chunks, 2);

        let local = burn_pc::PcTemporalCreditConfig {
            mode: burn_pc::PcTemporalCreditMode::ExactWindow,
            window_chunks: 4,
        };
        assert_eq!(
            effective_predictive_coding_temporal_credit(
                TrainingAlgorithm::PredictiveCoding,
                2,
                local,
            ),
            local
        );
    }

    #[test]
    fn predictive_context_metrics_update_entity_scoped_lifecycle_state() {
        let mut state = PredictiveContextRoutingTelemetryState::default();
        let sample = |name: &str, value: f64, absolute_step: usize| TrainingMetricSample {
            run_id: "run-a".into(),
            split: TrainingMetricSplit::Train,
            epoch: 2,
            step_in_epoch: 3,
            absolute_step,
            name: name.to_string(),
            value,
            running_value: value,
        };
        for metric in [
            sample("Predictive Context Index", 3.0, 11),
            sample("Predictive Context Generation", 2.0, 11),
            sample("Predictive Context Count", 4.0, 11),
            sample("Predictive Context Created", 1.0, 11),
            sample("Predictive Context Replaced", 1.0, 11),
            sample("Predictive Context Novelty Deferred", 1.0, 11),
            sample("Predictive Context Probe Tokens", 128.0, 11),
            sample("Predictive Context Selected Loss", 0.75, 11),
        ] {
            apply_predictive_context_routing_metric(&mut state, &metric);
        }
        assert_eq!(state.current_context, 3);
        assert_eq!(state.current_generation, 2);
        assert_eq!(state.known_contexts, 4);
        assert_eq!(state.probes, 1);
        assert_eq!(state.creations, 1);
        assert_eq!(state.replacements, 1);
        assert_eq!(state.novelty_deferrals, 1);
        assert_eq!(state.probe_tokens, 128);
        assert_eq!(state.selected_loss, Some(0.75));
        assert_eq!(state.last_absolute_step, 11);
    }
}
