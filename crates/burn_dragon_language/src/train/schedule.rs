use crate::dataset::scheduler::TokenSequenceDataset;
use crate::train::dynamics::{ActiveDynamicsControl, DragonDynamicsControlSlot};
use crate::train::prelude::*;
use crate::train::utils::log_theoretical_profile;
#[cfg(feature = "ddp")]
use burn::tensor::TensorPrimitive;
use burn_dragon_train::train::events::{
    CapabilityProbeExample, CapabilityProbeGroupMetric, CapabilityProbeSample, CheckpointEvent,
    ContinualBackpropSample, DynamicsControlEvent, DynamicsMode, ModelScaleApplied,
    ModelScaleRequest, ModelScaleSkipped, OutputDegeneracySample, PredictiveCodingSample,
    StepFinished, StepStarted, TrainingEpochSummary, TrainingEventBus, TrainingGateAction,
    TrainingGateEvent, TrainingGateSeverity, TrainingMetricSample, TrainingMetricSplit,
    ValidationFinished,
};
#[cfg(feature = "ddp")]
use std::collections::HashMap;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::io::{BufRead, BufReader};
#[cfg(feature = "ddp")]
use std::marker::PhantomData;

const CHECKPOINT_KEEP_LAST: usize = 2;
const METRIC_LOSS: &str = "Loss";
const METRIC_STREAM_WARM_LOSS: &str = "Stream Warm Loss";
const METRIC_RANDOM_COLD_LOSS: &str = "Random Cold Loss";

#[derive(Clone, Debug, Default)]
struct ContinualLearningStabilityState {
    best_valid_loss: Option<f64>,
    best_checkpoint_epoch: Option<usize>,
    best_ruliad_competence: Option<RuliadCompetenceKey>,
    best_ruliad_verifier_accuracy: Option<f32>,
    best_ruliad_partial_progress: Option<f32>,
    consecutive_validation_regressions: usize,
    consecutive_output_degeneracy: usize,
    consecutive_ruliad_correctness_regressions: usize,
    first_capability_pass_epoch: Option<usize>,
    last_capability_pass_epoch: Option<usize>,
    consecutive_capability_gate_failures: usize,
}

#[derive(Clone, Debug, Default)]
struct DynamicValidationReport {
    loss: f64,
    source_weighted_loss: Option<f64>,
    stream_warm_loss: Option<f64>,
    output_degeneracy: Option<crate::train::steps::OutputDegeneracyStats>,
    ruliad_eval_report: Option<burn_dragon_universality::RuliadEvalReport>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct RuliadCompetenceKey {
    verifier_ppm: u32,
    semantic_ppm: u32,
    partial_ppm: u32,
    certificate_ppm: u32,
    completion_health_ppm: u32,
}

impl RuliadCompetenceKey {
    fn has_free_run_correctness(self) -> bool {
        self.verifier_ppm > 0 || self.semantic_ppm > 0
    }
}

fn ratio_to_ppm(value: f32) -> u32 {
    (value.clamp(0.0, 1.0) * 1_000_000.0).round() as u32
}

fn ruliad_competence_key(
    report: &burn_dragon_universality::RuliadEvalReport,
) -> Option<RuliadCompetenceKey> {
    if report.scored_count == 0 || report.item_count == 0 {
        return None;
    }
    let item_count = report.item_count.max(1);
    let unhealthy_count = report
        .malformed_completion_count
        .saturating_add(report.missing_completion_count)
        .saturating_add(report.schema_valid_wrong_count)
        .min(item_count);
    let completion_health = item_count.saturating_sub(unhealthy_count) as f32 / item_count as f32;
    Some(RuliadCompetenceKey {
        verifier_ppm: ratio_to_ppm(report.verifier_accuracy),
        semantic_ppm: ratio_to_ppm(report.semantic_accuracy),
        partial_ppm: ratio_to_ppm(report.mean_partial_progress),
        certificate_ppm: ratio_to_ppm(report.mean_certificate_prefix_coverage),
        completion_health_ppm: ratio_to_ppm(completion_health),
    })
}

fn ruliad_competence_score(report: &burn_dragon_universality::RuliadEvalReport) -> f64 {
    let Some(key) = ruliad_competence_key(report) else {
        return 0.0;
    };
    // Dashboard-only lexicographic encoding. Checkpoint promotion compares the key directly.
    const SCALE: f64 = 1_000_001.0;
    ((((f64::from(key.verifier_ppm) * SCALE + f64::from(key.semantic_ppm)) * SCALE
        + f64::from(key.partial_ppm))
        * SCALE
        + f64::from(key.certificate_ppm))
        * SCALE)
        + f64::from(key.completion_health_ppm)
}

fn ruliad_capability_rates(
    report: &burn_dragon_universality::RuliadEvalReport,
) -> (f64, f64, f64, f64) {
    let item_count = report.item_count.max(1) as f64;
    let schema_wrong_rate = report.schema_valid_wrong_count as f64 / item_count;
    let malformed_rate = report.malformed_completion_count as f64 / item_count;
    let missing_rate = report.missing_completion_count as f64 / item_count;
    let unhealthy_count = report
        .schema_valid_wrong_count
        .saturating_add(report.malformed_completion_count)
        .saturating_add(report.missing_completion_count)
        .min(report.item_count);
    let completion_health_rate = if report.item_count == 0 {
        0.0
    } else {
        report.item_count.saturating_sub(unhealthy_count) as f64 / report.item_count as f64
    };
    (
        schema_wrong_rate,
        malformed_rate,
        missing_rate,
        completion_health_rate,
    )
}

#[derive(Clone, Debug, Default, PartialEq)]
struct RuliadCapabilityGateStatus {
    passed: bool,
    reasons: Vec<String>,
}

fn ruliad_capability_gate_status(
    report: &burn_dragon_universality::RuliadEvalReport,
    output_degeneracy: Option<&crate::train::steps::OutputDegeneracyStats>,
    gates: &burn_dragon_train::TrainingGatesConfig,
) -> RuliadCapabilityGateStatus {
    if !gates.enabled {
        return RuliadCapabilityGateStatus {
            passed: true,
            reasons: Vec::new(),
        };
    }
    let mut reasons = Vec::new();
    if report.item_count == 0 || report.scored_count == 0 {
        reasons.push("no_scored_ruliad_items".to_string());
    }
    let (schema_wrong_rate, malformed_rate, missing_rate, completion_health_rate) =
        ruliad_capability_rates(report);
    if schema_wrong_rate > gates.capability_schema_wrong_max_rate {
        reasons.push(format!(
            "schema_wrong_rate={schema_wrong_rate:.3}>{:.3}",
            gates.capability_schema_wrong_max_rate
        ));
    }
    if malformed_rate > gates.capability_malformed_max_rate {
        reasons.push(format!(
            "malformed_rate={malformed_rate:.3}>{:.3}",
            gates.capability_malformed_max_rate
        ));
    }
    if missing_rate > gates.capability_missing_max_rate {
        reasons.push(format!(
            "missing_rate={missing_rate:.3}>{:.3}",
            gates.capability_missing_max_rate
        ));
    }
    if completion_health_rate < gates.capability_completion_health_min_rate {
        reasons.push(format!(
            "completion_health={completion_health_rate:.3}<{}",
            gates.capability_completion_health_min_rate
        ));
    }
    if let Some(stats) = output_degeneracy {
        if stats.entropy_bits < gates.capability_output_entropy_min_bits {
            reasons.push(format!(
                "output_entropy_bits={:.3}<{}",
                stats.entropy_bits, gates.capability_output_entropy_min_bits
            ));
        }
        if stats.distinct_2_fraction < gates.capability_distinct_2_min_fraction {
            reasons.push(format!(
                "output_distinct2={:.3}<{}",
                stats.distinct_2_fraction, gates.capability_distinct_2_min_fraction
            ));
        }
    }
    RuliadCapabilityGateStatus {
        passed: reasons.is_empty(),
        reasons,
    }
}

fn validation_ruliad_correctness_improved(
    validation: &DynamicValidationReport,
    state: &ContinualLearningStabilityState,
) -> bool {
    validation
        .ruliad_eval_report
        .as_ref()
        .filter(|report| report.scored_count > 0)
        .is_some_and(|report| {
            let verifier_best = state
                .best_ruliad_verifier_accuracy
                .unwrap_or(report.verifier_accuracy);
            let partial_best = state
                .best_ruliad_partial_progress
                .unwrap_or(report.mean_partial_progress);
            report.verifier_accuracy > verifier_best + f32::EPSILON
                || report.mean_partial_progress > partial_best + f32::EPSILON
        })
}

fn ruliad_regression_tolerance(scored_count: usize) -> f32 {
    (1.0 / scored_count.max(1) as f32).max(0.01)
}

fn ruliad_metric_materially_regressed(
    best: f32,
    current: f32,
    scored_count: usize,
    minimum_best: f32,
) -> bool {
    if best < minimum_best {
        return false;
    }
    let tolerance = ruliad_regression_tolerance(scored_count);
    current + tolerance < best && current < best * 0.90
}

fn update_capability_run_control_state<B>(
    env: &TrainEnvironment<'_, B>,
    report: &burn_dragon_universality::RuliadEvalReport,
    output_degeneracy: Option<&crate::train::steps::OutputDegeneracyStats>,
    epoch: usize,
    absolute_step: usize,
    state: &mut ContinualLearningStabilityState,
    bus: &TrainingEventBus,
    recovery_requested: &mut bool,
) where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let gates = &env.training.gates;
    if !gates.enabled {
        return;
    }
    let status = ruliad_capability_gate_status(report, output_degeneracy, gates);
    if status.passed {
        if state.first_capability_pass_epoch.is_none() {
            state.first_capability_pass_epoch = Some(epoch);
            emit_policy_gate_with_action(
                env,
                bus,
                "continual_learning_capability_gate_first_pass",
                TrainingGateAction::Alert,
                TrainingGateSeverity::Info,
                epoch,
                absolute_step,
                "ruliad capability gate passed for the first time; capability-gated auxiliaries and regression tracking may open".to_string(),
            );
        }
        state.last_capability_pass_epoch = Some(epoch);
        state.consecutive_capability_gate_failures = 0;
        return;
    }

    let reasons = status.reasons.join(", ");
    let before_grace =
        state.first_capability_pass_epoch.is_none() && epoch <= gates.capability_grace_epochs;
    if before_grace {
        emit_policy_gate_with_action(
            env,
            bus,
            "continual_learning_capability_gate_grace",
            TrainingGateAction::Alert,
            TrainingGateSeverity::Info,
            epoch,
            absolute_step,
            format!(
                "ruliad capability gate has not passed during grace window epoch {epoch}/{}: {reasons}",
                gates.capability_grace_epochs
            ),
        );
        return;
    }

    let require_failure_recovery =
        !gates.capability_required_after_first_pass || state.first_capability_pass_epoch.is_some();
    if !require_failure_recovery {
        emit_policy_gate_with_action(
            env,
            bus,
            "continual_learning_capability_gate_not_ready",
            TrainingGateAction::Alert,
            TrainingGateSeverity::Warning,
            epoch,
            absolute_step,
            format!("ruliad capability gate still has not passed: {reasons}"),
        );
        return;
    }

    state.consecutive_capability_gate_failures =
        state.consecutive_capability_gate_failures.saturating_add(1);
    let failures = state.consecutive_capability_gate_failures;
    emit_policy_gate_with_action(
        env,
        bus,
        "continual_learning_capability_gate_regression",
        TrainingGateAction::Alert,
        TrainingGateSeverity::Warning,
        epoch,
        absolute_step,
        format!(
            "ruliad capability gate failed after capability was required ({failures}/{}): {reasons}",
            gates.capability_regression_patience_epochs
        ),
    );
    if failures < gates.capability_regression_patience_epochs || *recovery_requested {
        return;
    }

    let rollback_epoch = state.best_checkpoint_epoch;
    let mode = if rollback_epoch.is_some() {
        DynamicsMode::RollbackRecovery
    } else {
        DynamicsMode::ValidationRecovery
    };
    let message = format!(
        "ruliad capability regression persisted for {failures} validation epochs; requesting {:?}{}",
        mode,
        rollback_epoch
            .map(|epoch| format!(" to checkpoint epoch {epoch}"))
            .unwrap_or_default()
    );
    emit_dynamics_control(
        env,
        bus,
        &env.training.dynamics,
        mode,
        epoch,
        absolute_step,
        rollback_epoch,
        message,
    );
    *recovery_requested = true;
}

fn should_promote_checkpoint(
    validation: &DynamicValidationReport,
    best_loss: Option<f64>,
    best_competence: Option<RuliadCompetenceKey>,
    gates: &burn_dragon_train::TrainingGatesConfig,
) -> bool {
    let loss_improved = best_loss.is_none_or(|best| validation.loss < best);
    let Some(current_competence) = validation
        .ruliad_eval_report
        .as_ref()
        .and_then(ruliad_competence_key)
    else {
        return loss_improved;
    };
    if let Some(report) = validation.ruliad_eval_report.as_ref()
        && !ruliad_capability_gate_status(report, validation.output_degeneracy.as_ref(), gates)
            .passed
    {
        return false;
    }
    let Some(best_competence) = best_competence else {
        return true;
    };
    if current_competence > best_competence {
        return true;
    }
    if current_competence < best_competence {
        return false;
    }
    if !current_competence.has_free_run_correctness() {
        return false;
    }
    if validation
        .output_degeneracy
        .as_ref()
        .is_some_and(|stats| output_degeneracy_tripped(gates, stats))
    {
        return false;
    }
    loss_improved
}

fn teacher_forced_validation_metric_name(
    source_selection_dataset: Option<&Arc<Dataset>>,
) -> Option<&'static str> {
    let dataset = source_selection_dataset?;
    if dataset.uses_target_loss_mask() {
        Some("Teacher Forced Answer CE")
    } else if dataset.uses_live_source_selection() {
        Some("Teacher Forced CE")
    } else {
        None
    }
}

fn emit_teacher_forced_validation_metric(
    run_name: &str,
    source_selection_dataset: Option<&Arc<Dataset>>,
    epoch: usize,
    step_in_epoch: usize,
    absolute_step: usize,
    value: f64,
    running_value: f64,
    bus: &TrainingEventBus,
) {
    let Some(name) = teacher_forced_validation_metric_name(source_selection_dataset) else {
        return;
    };
    let _ = bus.send_metric_sample(TrainingMetricSample {
        run_id: run_name.to_string(),
        split: TrainingMetricSplit::Valid,
        epoch,
        step_in_epoch,
        absolute_step,
        name: name.to_string(),
        value,
        running_value,
    });
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct HistoricalBestValidation {
    best_loss: Option<f64>,
    best_checkpoint_epoch: Option<usize>,
}

#[derive(Clone, Copy, Debug, Default)]
struct EggrollStepTiming {
    total_ms: f64,
    candidate_eval_ms: f64,
    update_ms: f64,
}

struct QuietMetricsRenderer;

impl burn_train::renderer::MetricsRendererTraining for QuietMetricsRenderer {
    fn update_train(&mut self, _state: burn_train::renderer::MetricState) {}

    fn update_valid(&mut self, _state: burn_train::renderer::MetricState) {}

    fn render_train(
        &mut self,
        _item: burn_train::renderer::TrainingProgress,
        _progress_indicators: Vec<burn_train::renderer::ProgressType>,
    ) {
    }

    fn render_valid(
        &mut self,
        _item: burn_train::renderer::TrainingProgress,
        _progress_indicators: Vec<burn_train::renderer::ProgressType>,
    ) {
    }

    fn on_train_end(
        &mut self,
        _summary: Option<burn_train::LearnerSummary>,
    ) -> std::result::Result<(), Box<dyn core::error::Error>> {
        Ok(())
    }
}

impl burn_train::renderer::MetricsRendererEvaluation for QuietMetricsRenderer {
    fn update_test(
        &mut self,
        _name: burn_train::renderer::EvaluationName,
        _state: burn_train::renderer::MetricState,
    ) {
    }

    fn render_test(
        &mut self,
        _item: burn_train::renderer::EvaluationProgress,
        _progress_indicators: Vec<burn_train::renderer::ProgressType>,
    ) {
    }

    fn on_test_end(
        &mut self,
        _summary: Option<burn_train::LearnerSummary>,
    ) -> std::result::Result<(), Box<dyn core::error::Error>> {
        Ok(())
    }
}

impl burn_train::renderer::MetricsRenderer for QuietMetricsRenderer {
    fn manual_close(&mut self) {}

    fn register_metric(&mut self, _definition: burn_train::metric::MetricDefinition) {}
}

fn quiet_progress_renderer_enabled() -> bool {
    std::env::var("DragonModel_TRAINING_PROGRESS_RENDERER")
        .ok()
        .map(|value| quiet_progress_renderer_enabled_for(&value))
        .unwrap_or(true)
}

fn quiet_progress_renderer_enabled_for(value: &str) -> bool {
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "default" | "progress" | "on" | "true" | "1"
    )
}

struct FileMetricBestCheckpointingStrategy {
    run_dir: PathBuf,
    metric_name: String,
    direction: burn_train::metric::store::Direction,
    split: burn_train::metric::store::Split,
    best_epoch: Option<usize>,
    best_value: Option<f64>,
}

impl FileMetricBestCheckpointingStrategy {
    fn new<M>(
        run_dir: &Path,
        metric: &M,
        direction: burn_train::metric::store::Direction,
        split: burn_train::metric::store::Split,
    ) -> Self
    where
        M: burn_train::metric::Metric,
    {
        Self {
            run_dir: run_dir.to_path_buf(),
            metric_name: metric.name().to_string(),
            direction,
            split,
            best_epoch: None,
            best_value: None,
        }
    }

    fn is_better(&self, candidate: f64, current: f64) -> bool {
        match self.direction {
            burn_train::metric::store::Direction::Lowest => candidate < current,
            burn_train::metric::store::Direction::Highest => candidate > current,
        }
    }

    fn checkpoint_path(&self, epoch: usize) -> PathBuf {
        self.run_dir
            .join("checkpoint")
            .join(format!("model-{epoch}.bin"))
    }

    fn metric_log_path(&self, epoch: usize) -> PathBuf {
        let split_dir = match self.split {
            burn_train::metric::store::Split::Train => "train",
            burn_train::metric::store::Split::Valid => "valid",
            burn_train::metric::store::Split::Test(_) => "test",
        };
        self.run_dir
            .join(split_dir)
            .join(format!("epoch-{epoch}"))
            .join(format!("{}.log", self.metric_name))
    }

    fn checkpoint_exists(&self, epoch: usize) -> bool {
        self.checkpoint_path(epoch).is_file()
    }

    fn existing_checkpoint_epochs(&self) -> BTreeSet<usize> {
        let checkpoint_dir = self.run_dir.join("checkpoint");
        let Ok(entries) = fs::read_dir(checkpoint_dir) else {
            return BTreeSet::new();
        };

        entries
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                let name = path.file_name()?.to_str()?;
                name.strip_prefix("model-")?
                    .strip_suffix(".bin")?
                    .parse::<usize>()
                    .ok()
            })
            .collect()
    }

    fn metric_mean_from_log(&self, epoch: usize) -> Option<f64> {
        let path = self.metric_log_path(epoch);
        let content = fs::read_to_string(path).ok()?;
        let mut sum = 0.0;
        let mut count = 0usize;

        for line in content.lines() {
            let field = line.split(',').next()?.trim();
            let value = field.parse::<f64>().ok()?;
            sum += value;
            count += 1;
        }

        (count > 0).then_some(sum / count as f64)
    }

    fn update_best_candidate(&mut self, epoch: usize, value: f64) -> Option<usize> {
        let should_replace = self
            .best_value
            .is_none_or(|current| self.is_better(value, current));

        if !should_replace {
            return None;
        }

        let previous_best = self.best_epoch.replace(epoch);
        self.best_value = Some(value);
        previous_best.filter(|previous_best| *previous_best != epoch)
    }

    fn refresh_best_from_history(&mut self, last_epoch: usize) {
        self.best_epoch = None;
        self.best_value = None;

        for epoch in 1..=last_epoch {
            if let Some(value) = self.metric_mean_from_log(epoch) {
                let _ = self.update_best_candidate(epoch, value);
            }
        }
    }

    fn refresh_best_from_store(
        &mut self,
        store: &burn_train::metric::store::EventStoreClient,
    ) -> bool {
        let Some(best_epoch) = store.find_epoch(
            &self.metric_name,
            burn_train::metric::store::Aggregate::Mean,
            self.direction,
            &self.split,
        ) else {
            return false;
        };

        self.best_epoch = Some(best_epoch);
        self.best_value = store.find_metric(
            &self.metric_name,
            best_epoch,
            burn_train::metric::store::Aggregate::Mean,
            &self.split,
        );
        true
    }

    fn checkpoint_actions_after_refresh(
        &self,
        epoch: usize,
    ) -> Vec<burn_train::checkpoint::CheckpointingAction> {
        let mut keep_epochs = BTreeSet::new();
        keep_epochs.extend(epoch.saturating_sub(CHECKPOINT_KEEP_LAST - 1).max(1)..=epoch);
        if let Some(best_epoch) = self.best_epoch {
            keep_epochs.insert(best_epoch);
        }

        let existing_epochs = self.existing_checkpoint_epochs();
        let mut actions = vec![burn_train::checkpoint::CheckpointingAction::Save];
        actions.extend(
            existing_epochs
                .into_iter()
                .filter(|existing_epoch| !keep_epochs.contains(existing_epoch))
                .map(burn_train::checkpoint::CheckpointingAction::Delete),
        );
        actions
    }

    fn actions_for_epoch(
        &mut self,
        epoch: usize,
    ) -> Vec<burn_train::checkpoint::CheckpointingAction> {
        self.refresh_best_from_history(epoch);
        self.checkpoint_actions_after_refresh(epoch)
    }

    fn actions_for_epoch_with_store(
        &mut self,
        epoch: usize,
        store: &burn_train::metric::store::EventStoreClient,
    ) -> Vec<burn_train::checkpoint::CheckpointingAction> {
        if !self.refresh_best_from_store(store) {
            self.refresh_best_from_history(epoch);
        }
        self.checkpoint_actions_after_refresh(epoch)
    }
}

impl burn_train::checkpoint::CheckpointingStrategy for FileMetricBestCheckpointingStrategy {
    fn checkpointing(
        &mut self,
        epoch: usize,
        store: &burn_train::metric::store::EventStoreClient,
    ) -> Vec<burn_train::checkpoint::CheckpointingAction> {
        self.actions_for_epoch_with_store(epoch, store)
    }
}

pub struct TrainEnvironment<'a, B>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    pub parallel_runtime: &'a ParallelRuntime,
    pub parallel_config: &'a ParallelConfig,
    pub run_dir: &'a Path,
    pub run_name: &'a str,
    pub backend_name: &'a str,
    pub training: &'a TrainingHyperparameters,
    pub resume_checkpoint_epoch: Option<usize>,
    pub model_config: &'a DragonConfig,
    pub device: &'a B::Device,
    pub devices: &'a [B::Device],
    pub train_dataset: Option<Arc<Dataset>>,
    pub valid_dataset: Option<Arc<Dataset>>,
    pub train_loader: Arc<dyn DataLoader<B, SequenceBatch<B>>>,
    pub valid_loader: Arc<dyn DataLoader<ValidBackend<B>, SequenceBatch<ValidBackend<B>>>>,
    pub source_selection_dataset: Option<Arc<Dataset>>,
    pub summary_event_token_ids: Option<Vec<u32>>,
    pub neuron_scaling_slot: Option<crate::train::neuron_scaling::NeuronScaleRequestSlot>,
    pub epochs: usize,
    pub total_steps: usize,
    pub valid_steps: usize,
}

pub struct ForwardEggrollTrainEnvironment<'a, B>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    pub parallel_runtime: &'a ParallelRuntime,
    pub run_dir: &'a Path,
    pub run_name: &'a str,
    pub backend_name: &'a str,
    pub training: &'a TrainingHyperparameters,
    pub resume_checkpoint_epoch: Option<usize>,
    pub model_config: &'a DragonConfig,
    pub device: &'a B::Device,
    pub train_loader: Arc<dyn DataLoader<B, SequenceBatch<B>>>,
    pub valid_loader: Arc<dyn DataLoader<B, SequenceBatch<B>>>,
    pub source_selection_dataset: Option<Arc<Dataset>>,
    pub summary_event_token_ids: Option<Vec<u32>>,
    pub epochs: usize,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct EggrollChunkAutotuneCandidateReport {
    pub population_chunk_size: usize,
    pub evaluated_population_size: usize,
    pub elapsed_ms: f64,
    pub forward_evaluations_per_second: f64,
    pub mean_loss: f64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct EggrollChunkAutotuneReport {
    pub selected_population_chunk_size: usize,
    pub configured_population_chunk_size: usize,
    pub population_size: usize,
    pub max_probe_population_size: usize,
    pub candidates: Vec<EggrollChunkAutotuneCandidateReport>,
}

#[derive(Clone, Debug)]
struct EggrollPopulationExecutionPlan {
    backend: EggrollPopulationExecutionBackend,
    scope: EggrollPerturbationScope,
    population_tile_size: Option<usize>,
}

impl EggrollPopulationExecutionPlan {
    fn executor_name(&self) -> &'static str {
        match self.backend {
            EggrollPopulationExecutionBackend::Factorized => "factorized_tensorized",
            EggrollPopulationExecutionBackend::Auto
            | EggrollPopulationExecutionBackend::Reference
            | EggrollPopulationExecutionBackend::Cuda => "stacked_tensorized",
        }
    }

    fn scope_name(&self) -> &'static str {
        match self.scope {
            EggrollPerturbationScope::DragonCoreProjection => "dragon_core_projection",
        }
    }
}

fn parameter_updates_enabled(training: &TrainingHyperparameters) -> bool {
    !training.predictive_coding.enabled
        || matches!(
            training.predictive_coding.parameter_update,
            PredictiveCodingParameterUpdate::Optimizer
        )
}

pub(crate) fn train_with_scheduler<B, O, S>(
    env: &TrainEnvironment<'_, B>,
    model: LanguageTrainModel<B>,
    optimizer: O,
    scheduler: S,
) -> Result<DragonModel<ValidBackend<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    O: Optimizer<LanguageTrainModel<B>, B> + 'static,
    S: LrScheduler + 'static,
{
    fs::create_dir_all(env.run_dir)?;

    let source_selection_dataset = env.source_selection_dataset.as_ref().cloned();
    let train_loss_metric_every = crate::train::events::train_loss_metric_frequency(
        env.training,
        source_selection_dataset.as_ref(),
    );
    #[cfg(feature = "ddp")]
    if env.parallel_runtime.mode == ParallelismKind::Ddp
        && env.parallel_runtime.is_process_group_launch()
    {
        return train_with_process_group_scheduler(env, model, optimizer, scheduler);
    }
    let training_strategy = match env.parallel_runtime.mode {
        ParallelismKind::Single => {
            LearningStrategy::Default(ExecutionStrategy::single(env.device.clone()))
        }
        ParallelismKind::Ddp => LearningStrategy::Default(ExecutionStrategy::multi(
            env.devices.to_vec(),
            MultiDeviceOptim::OptimMainDevice,
        )),
        mode => {
            return Err(anyhow!(
                "parallel.mode={mode:?} is not wired into language training yet"
            ));
        }
    };
    let event_handles = crate::train::events::build_training_event_handles(
        env.run_name,
        env.run_dir,
        env.train_loader.num_items(),
        env.training,
        source_selection_dataset,
        env.neuron_scaling_slot
            .as_ref()
            .map(|slot| (env.model_config.latent_total(), slot.clone())),
        None,
    )?;

    let builder = SupervisedTraining::new(
        env.run_dir,
        Arc::clone(&env.train_loader),
        Arc::clone(&env.valid_loader),
    )
    .num_epochs(env.epochs)
    .grads_accumulation(env.training.gradient_accumulation_steps.max(1))
    .with_training_strategy(training_strategy)
    .with_application_logger(None)
    .with_interrupter(event_handles.interrupter)
    .with_metric_logger(event_handles.metric_logger)
    .with_file_checkpointer(BinFileRecorder::<FullPrecisionSettings>::new())
    .with_checkpointing_strategy(FileMetricBestCheckpointingStrategy::new(
        env.run_dir,
        &LossMetric::<ValidBackend<B>>::new(),
        burn_train::metric::store::Direction::Lowest,
        burn_train::metric::store::Split::Valid,
    ));
    let builder = builder.metric_train_numeric(ScalarMetric::<
        ValidBackend<B>,
        LossValue<ValidBackend<B>>,
    >::new_every("Loss", train_loss_metric_every));
    let builder = builder
        .metric_valid_numeric(LossMetric::<ValidBackend<B>>::new())
        .metric_train_numeric(LearningRateMetric::new())
        .metric_train(DeviceMetric::new("device", env.backend_name))
        .metric_valid(DeviceMetric::new("device", env.backend_name));
    let builder = if quiet_progress_renderer_enabled() {
        builder.renderer(QuietMetricsRenderer)
    } else {
        builder
    };
    #[cfg(feature = "rerun")]
    let builder = crate::train::rerun::attach_metric_loggers(builder, env.run_dir);
    let builder = builder.summary();
    let builder = match env.resume_checkpoint_epoch {
        Some(checkpoint) => builder.checkpoint(checkpoint),
        None => builder,
    };

    info!("run name: {}", env.run_name);
    info!(
        "training strategy: mode={:?} replicas={}",
        env.parallel_runtime.mode,
        env.devices.len()
    );
    info!(
        "checkpoint policy: logical_epoch_steps={} keep_last={} keep_best_valid_loss=true",
        env.train_loader.num_items(),
        CHECKPOINT_KEEP_LAST
    );

    let learner = burn_train::Learner::new(model, optimizer, scheduler);
    let TrainingResult { model, .. } = builder.launch(learner);

    log_theoretical_profile(
        env.model_config,
        env.training
            .batch_size
            .saturating_mul(env.training.gradient_accumulation_steps.max(1)),
        env.training.block_size,
        env.backend_name,
    );

    Ok(model.model)
}

pub(crate) fn autotune_eggroll_population_chunk_size<B>(
    optimizer_cfg: &OptimizerConfig,
    model: &LanguageTrainModel<B>,
    train_loader: &Arc<dyn DataLoader<B, SequenceBatch<B>>>,
) -> Result<Option<EggrollChunkAutotuneReport>>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    if !matches!(optimizer_cfg.name, OptimizerKind::Eggroll)
        || !optimizer_cfg.eggroll_auto_population.enabled
        || !optimizer_cfg.eggroll_auto_population.chunk_autotune.enabled
    {
        return Ok(None);
    }

    let Some(batch) = train_loader.iter().next() else {
        return Ok(None);
    };
    let eggroll = optimizer_cfg.effective_eggroll_config();
    let candidates = resolve_eggroll_chunk_autotune_candidates(optimizer_cfg);
    if candidates.is_empty() {
        return Ok(None);
    }
    let execution_plan = resolve_eggroll_population_execution_plan(optimizer_cfg, model)?;

    let mut reports = Vec::with_capacity(candidates.len());
    for population_chunk_size in candidates {
        let report = measure_eggroll_chunk_candidate(
            &execution_plan,
            model,
            batch.clone(),
            &eggroll,
            population_chunk_size,
            optimizer_cfg
                .eggroll_auto_population
                .chunk_autotune
                .max_probe_population_size,
        )?;
        reports.push(report);
    }
    let Some(selected) = reports
        .iter()
        .filter(|report| {
            report.forward_evaluations_per_second.is_finite()
                && report.forward_evaluations_per_second > 0.0
                && report.mean_loss.is_finite()
        })
        .max_by(|left, right| {
            left.forward_evaluations_per_second
                .partial_cmp(&right.forward_evaluations_per_second)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    else {
        return Ok(None);
    };

    Ok(Some(EggrollChunkAutotuneReport {
        selected_population_chunk_size: selected.population_chunk_size,
        configured_population_chunk_size: eggroll.population.population_chunk_size,
        population_size: eggroll.population.population_size,
        max_probe_population_size: optimizer_cfg
            .eggroll_auto_population
            .chunk_autotune
            .max_probe_population_size,
        candidates: reports,
    }))
}

pub(crate) fn train_with_eggroll_forward_only<B>(
    env: &ForwardEggrollTrainEnvironment<'_, B>,
    optimizer_cfg: &OptimizerConfig,
    mut model: LanguageTrainModel<B>,
) -> Result<DragonModel<B>>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    fs::create_dir_all(env.run_dir)?;
    let eggroll = optimizer_cfg.effective_eggroll_config();

    let source_selection_dataset = env.source_selection_dataset.as_ref().cloned();
    let event_handles = crate::train::events::build_training_event_handles(
        env.run_name,
        env.run_dir,
        env.train_loader.num_items(),
        env.training,
        source_selection_dataset,
        None,
        None,
    )?;
    let bus = event_handles.metric_logger.bus();
    crate::train::profile::reset_predictive_coding();
    let emit_step_events = env.training.events.persist_step_events;
    let steps_per_epoch = env.train_loader.num_items().max(1);
    let start_epoch = env
        .resume_checkpoint_epoch
        .map(|epoch| epoch + 1)
        .unwrap_or(1);
    let mut best_valid_loss: Option<f64> = None;
    let mut best_valid_epoch: Option<usize> = None;
    let mut best_ruliad_competence: Option<RuliadCompetenceKey> = None;
    if let Some(epoch) = env.resume_checkpoint_epoch {
        model.model =
            load_dragon_model_checkpoint(env.run_dir, epoch, env.model_config, env.device)?;
        let historical_best = historical_best_validation(env.run_dir, epoch);
        best_valid_loss = historical_best.best_loss;
        best_valid_epoch = historical_best.best_checkpoint_epoch;
        info!(
            "EGGROLL forward-only resumed model checkpoint epoch={} with fresh optimizer state historical_best_valid_loss={:?} historical_best_checkpoint_epoch={:?}",
            epoch, best_valid_loss, best_valid_epoch
        );
    }

    let pair_count = eggroll.population.pair_count().max(1);
    let chunk_pair_count = (eggroll.population.population_chunk_size.max(2) / 2)
        .max(1)
        .min(pair_count);
    let eggroll_execution_plan = resolve_eggroll_population_execution_plan(optimizer_cfg, &model)?;
    let mut optimizer_state = burn_dragon_eggroll::EggrollModuleOptimizerState::<B>::new();

    info!(
        "training strategy: mode={:?} optimizer=eggroll backend_mode=forward_only population_size={} pair_count={} population_chunk_size={} chunk_pair_count={} rank={} sigma={} interval_steps={} start_epoch={} eggroll_executor={} eggroll_scope={}",
        env.parallel_runtime.mode,
        eggroll.population.population_size,
        pair_count,
        eggroll.population.population_chunk_size,
        chunk_pair_count,
        eggroll.population.rank,
        eggroll.sigma,
        eggroll.interval_steps,
        start_epoch,
        eggroll_execution_plan.executor_name(),
        eggroll_execution_plan.scope_name()
    );

    for epoch in start_epoch..=env.epochs {
        if event_handles.interrupter.should_stop() {
            let reason = event_handles
                .interrupter
                .get_message()
                .unwrap_or_else(|| "training interrupted".to_string());
            info!("Training interrupted: {reason}");
            break;
        }

        let mut iterator = env.train_loader.iter();
        let mut iteration = 0usize;
        while let Some(batch) = iterator.next() {
            iteration += 1;
            let absolute_step = epoch
                .saturating_sub(1)
                .saturating_mul(steps_per_epoch)
                .saturating_add(iteration.saturating_sub(1));
            if emit_step_events {
                let _ = bus.send_step_started(StepStarted {
                    run_id: env.run_name.to_string(),
                    absolute_step,
                    epoch,
                });
            }

            let eggroll_step_active = absolute_step.is_multiple_of(eggroll.interval_steps.max(1));
            let (mean_train_loss, metrics, eggroll_step_timing) = if eggroll_step_active {
                let eggroll_step_start = burn_dragon_time::Instant::now();
                let candidate_eval_start = burn_dragon_time::Instant::now();
                let mut losses = Vec::with_capacity(pair_count * 2);
                let mut fitness = Vec::with_capacity(pair_count);
                for chunk_start in (0..pair_count).step_by(chunk_pair_count) {
                    let chunk_end = chunk_start.saturating_add(chunk_pair_count).min(pair_count);
                    let pairs_in_chunk = chunk_end.saturating_sub(chunk_start);
                    let chunk_losses = evaluate_eggroll_population_chunk(
                        &eggroll_execution_plan,
                        &model,
                        batch.clone(),
                        &eggroll,
                        absolute_step as u64,
                        chunk_start,
                        pairs_in_chunk,
                    )?;
                    for (offset, pair_index) in (chunk_start..chunk_end).enumerate() {
                        let plus_loss = chunk_losses[offset * 2];
                        let minus_loss = chunk_losses[offset * 2 + 1];
                        losses.push(plus_loss);
                        losses.push(minus_loss);
                        fitness.push(burn_dragon_eggroll::AntitheticFitness {
                            pair_index: pair_index as u64,
                            plus: -(plus_loss as f32),
                            minus: -(minus_loss as f32),
                        });
                    }
                }
                let candidate_eval_ms = candidate_eval_start.elapsed().as_millis() as f64;

                let mean_train_loss = if losses.is_empty() {
                    f64::NAN
                } else {
                    losses.iter().sum::<f64>() / losses.len() as f64
                };
                let update_start = burn_dragon_time::Instant::now();
                let (updated, metrics) = apply_eggroll_population_update(
                    &eggroll_execution_plan,
                    model,
                    &eggroll,
                    absolute_step as u64,
                    &fitness,
                    &mut optimizer_state,
                )?;
                let update_ms = update_start.elapsed().as_millis() as f64;
                model = updated;
                (
                    mean_train_loss,
                    Some(metrics),
                    Some(EggrollStepTiming {
                        total_ms: eggroll_step_start.elapsed().as_millis() as f64,
                        candidate_eval_ms,
                        update_ms,
                    }),
                )
            } else {
                (f64::NAN, None, None)
            };

            if source_selection_telemetry_due_for(
                env.training,
                env.source_selection_dataset.as_ref(),
                absolute_step,
            ) && mean_train_loss.is_finite()
            {
                emit_source_selection_telemetry_sample(
                    env.run_name,
                    env.source_selection_dataset.as_ref(),
                    absolute_step,
                    mean_train_loss,
                    &bus,
                );
            }
            let log_train_metrics =
                iteration % env.training.log_frequency.max(1) == 0 || iteration == steps_per_epoch;
            if log_train_metrics {
                let progress = iterator.progress();
                if let Some(metrics) = &metrics {
                    info!(
                        "train epoch={} step={}/{} loss={:.4} eggroll_fitness_mean={:.4} eggroll_fitness_std={:.4}",
                        epoch,
                        progress.items_processed,
                        progress.items_total,
                        mean_train_loss,
                        metrics.fitness_mean,
                        metrics.fitness_std
                    );
                } else {
                    info!(
                        "train epoch={} step={}/{} loss={:.4} eggroll_interval_skip=true",
                        epoch, progress.items_processed, progress.items_total, mean_train_loss
                    );
                }
            }
            if mean_train_loss.is_finite() {
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "Loss".to_string(),
                    value: mean_train_loss,
                    running_value: mean_train_loss,
                });
            }
            if let Some(metrics) = &metrics {
                let timing = eggroll_step_timing.unwrap_or_default();
                let elapsed_ms = timing.total_ms;
                let forward_evaluations = eggroll.population.population_size as f64;
                let forward_evaluations_per_second = if timing.candidate_eval_ms > 0.0 {
                    forward_evaluations * 1000.0 / timing.candidate_eval_ms
                } else {
                    0.0
                };
                let candidate_eval_fraction = if elapsed_ms > 0.0 {
                    timing.candidate_eval_ms / elapsed_ms
                } else {
                    0.0
                };
                let update_fraction = if elapsed_ms > 0.0 {
                    timing.update_ms / elapsed_ms
                } else {
                    0.0
                };
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "EGGROLL Fitness Std".to_string(),
                    value: metrics.fitness_std as f64,
                    running_value: metrics.fitness_std as f64,
                });
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "EGGROLL Coefficient RMS".to_string(),
                    value: metrics.coefficient_rms as f64,
                    running_value: metrics.coefficient_rms as f64,
                });
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "EGGROLL Coefficient Clip Fraction".to_string(),
                    value: metrics.coefficient_clip_fraction as f64,
                    running_value: metrics.coefficient_clip_fraction as f64,
                });
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "EGGROLL Step Milliseconds".to_string(),
                    value: elapsed_ms,
                    running_value: elapsed_ms,
                });
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "EGGROLL Candidate Eval Milliseconds".to_string(),
                    value: timing.candidate_eval_ms,
                    running_value: timing.candidate_eval_ms,
                });
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "EGGROLL Update Milliseconds".to_string(),
                    value: timing.update_ms,
                    running_value: timing.update_ms,
                });
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "EGGROLL Candidate Eval Fraction".to_string(),
                    value: candidate_eval_fraction,
                    running_value: candidate_eval_fraction,
                });
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "EGGROLL Update Fraction".to_string(),
                    value: update_fraction,
                    running_value: update_fraction,
                });
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "EGGROLL Forward Evaluations Per Second".to_string(),
                    value: forward_evaluations_per_second,
                    running_value: forward_evaluations_per_second,
                });
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "EGGROLL Population Size".to_string(),
                    value: eggroll.population.population_size as f64,
                    running_value: eggroll.population.population_size as f64,
                });
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "EGGROLL Population Chunk Size".to_string(),
                    value: eggroll.population.population_chunk_size as f64,
                    running_value: eggroll.population.population_chunk_size as f64,
                });
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "EGGROLL Stacked Tensorized Executor Active".to_string(),
                    value: 1.0,
                    running_value: 1.0,
                });
                let scope_id = match eggroll_execution_plan.scope {
                    EggrollPerturbationScope::DragonCoreProjection => 1.0,
                };
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "EGGROLL Perturbation Scope ID".to_string(),
                    value: scope_id,
                    running_value: scope_id,
                });
            }
            if emit_step_events {
                let _ = bus.send_step_finished(StepFinished {
                    run_id: env.run_name.to_string(),
                    absolute_step,
                    epoch,
                    loss: mean_train_loss.is_finite().then_some(mean_train_loss),
                });
            }
        }
        drop(iterator);

        let _ = bus.send_epoch_summary(TrainingEpochSummary {
            run_id: env.run_name.to_string(),
            split: TrainingMetricSplit::Train,
            epoch,
        });
        let validation =
            run_dynamic_validation_forward_only(env, &model, epoch, steps_per_epoch, &bus)?;
        let valid_loss = validation.loss;
        info!("valid epoch={} loss={valid_loss:.4}", epoch);
        let checkpoint_promoted = should_promote_checkpoint(
            &validation,
            best_valid_loss,
            best_ruliad_competence,
            &env.training.gates,
        );
        if checkpoint_promoted {
            best_valid_loss = Some(valid_loss);
            best_valid_epoch = Some(epoch);
            if let Some(competence) = validation
                .ruliad_eval_report
                .as_ref()
                .and_then(ruliad_competence_key)
            {
                best_ruliad_competence = Some(competence);
            }
        }
        let absolute_step = epoch.saturating_mul(steps_per_epoch).saturating_sub(1);
        save_dragon_model_checkpoint(env.run_dir, epoch, &model.model)?;
        save_source_selection_state_checkpoint(
            env.run_dir,
            epoch,
            absolute_step,
            env.source_selection_dataset.as_ref(),
        )?;
        prune_dragon_model_checkpoints(env.run_dir, epoch, best_valid_epoch)?;
        let _ = bus.send_checkpoint(CheckpointEvent {
            run_id: env.run_name.to_string(),
            checkpoint_id: format!("model-{epoch}"),
            epoch: Some(epoch),
            absolute_step: Some(absolute_step),
            promoted: checkpoint_promoted,
        });
        let _ = bus.flush();
    }

    log_theoretical_profile(
        env.model_config,
        env.training.batch_size,
        env.training.block_size,
        env.backend_name,
    );

    Ok(model.model)
}

fn eggroll_batch_loss_tensor<B>(
    model: &LanguageTrainModel<B>,
    batch: SequenceBatch<B>,
) -> Tensor<B, 1>
where
    B: BackendTrait,
{
    let output = ValidStep::step(model, batch);
    let loss_value: LossValue<B> = output.adapt();
    loss_value.value()
}

fn scalar_values_from_loss_tensors<B>(loss_tensors: Vec<Tensor<B, 1>>) -> Vec<f64>
where
    B: BackendTrait,
{
    if loss_tensors.is_empty() {
        return Vec::new();
    }
    let values = Tensor::cat(loss_tensors, 0)
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("loss tensor to vec");
    values.into_iter().map(|value| value as f64).collect()
}

fn resolve_eggroll_population_execution_plan<B>(
    optimizer_cfg: &OptimizerConfig,
    model: &LanguageTrainModel<B>,
) -> Result<EggrollPopulationExecutionPlan>
where
    B: BackendTrait,
{
    let execution = &optimizer_cfg.eggroll_population_execution;
    let scope = execution.perturbation_scope;

    if let Some(reason) = eggroll_population_execution_unsupported_reason(model) {
        return Err(anyhow!(
            "optimizer.eggroll_population_execution unsupported: {reason}"
        ));
    }

    Ok(EggrollPopulationExecutionPlan {
        backend: execution.backend,
        scope,
        population_tile_size: execution.population_tile_size,
    })
}

fn eggroll_population_execution_unsupported_reason<B>(
    model: &LanguageTrainModel<B>,
) -> Option<String>
where
    B: BackendTrait,
{
    if !matches!(model.objective, TrainingObjectiveConfig::NextToken) {
        return Some(
            "EGGROLL stacked tensorized executor currently supports objective=next_token"
                .to_string(),
        );
    }
    if model.tbptt_chunk_size.is_some() || model.tbptt_persist_across_steps {
        return Some(
            "EGGROLL stacked tensorized executor currently does not support TBPTT".to_string(),
        );
    }
    if model.pipeline_plan.is_some() {
        return Some(
            "EGGROLL stacked tensorized executor currently does not support pipeline execution"
                .to_string(),
        );
    }
    if !model.model.supports_shared_lowrank_population_forward() {
        return Some(
            "EGGROLL stacked tensorized executor requires flat logits, rollout_fast_steps_per_slow_step=1, and y-neuron recurrence disabled"
                .to_string(),
        );
    }
    None
}

fn evaluate_eggroll_population_chunk<B>(
    plan: &EggrollPopulationExecutionPlan,
    model: &LanguageTrainModel<B>,
    batch: SequenceBatch<B>,
    eggroll: &burn_eggroll::EggrollConfig,
    generation: u64,
    pair_start: usize,
    pair_count: usize,
) -> Result<Vec<f64>>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    evaluate_eggroll_population_chunk_stacked_tensorized(
        plan, model, batch, eggroll, generation, pair_start, pair_count,
    )
}

fn apply_eggroll_population_update<B>(
    _plan: &EggrollPopulationExecutionPlan,
    model: LanguageTrainModel<B>,
    eggroll: &burn_eggroll::EggrollConfig,
    generation: u64,
    fitness: &[burn_dragon_eggroll::AntitheticFitness],
    state: &mut burn_dragon_eggroll::EggrollModuleOptimizerState<B>,
) -> Result<(LanguageTrainModel<B>, burn_eggroll::EggrollMetrics)>
where
    B: BackendTrait + Clone,
{
    apply_shared_lowrank_eggroll_update(model, eggroll, generation, fitness, state)
}

fn apply_shared_lowrank_eggroll_update<B>(
    model: LanguageTrainModel<B>,
    eggroll: &burn_eggroll::EggrollConfig,
    generation: u64,
    fitness: &[burn_dragon_eggroll::AntitheticFitness],
    state: &mut burn_dragon_eggroll::EggrollModuleOptimizerState<B>,
) -> Result<(LanguageTrainModel<B>, burn_eggroll::EggrollMetrics)>
where
    B: BackendTrait,
{
    let coefficients =
        burn_dragon_eggroll::pair_gradient_coefficients(eggroll, generation, fitness)?;
    let population = fitness
        .iter()
        .flat_map(|item| [item.plus, item.minus])
        .collect::<Vec<_>>();
    let metrics = burn_eggroll::eggroll_metrics(
        generation,
        population.len(),
        eggroll.population.rank,
        eggroll.effective_sigma(generation),
        &population,
        &coefficients
            .iter()
            .map(|coefficient| coefficient.coefficient)
            .collect::<Vec<_>>(),
        eggroll.coefficient_clip,
    );
    let ids = model.model.shared_lowrank_param_ids();
    let weights = model.model.shared_lowrank_weights();
    let next = SharedLowrankWeights {
        encoder: burn_dragon_eggroll::apply_antithetic_update_to_tensor_with_coefficients(
            weights.encoder,
            ids.encoder.val(),
            eggroll,
            generation,
            &coefficients,
            state,
        ),
        encoder_v: burn_dragon_eggroll::apply_antithetic_update_to_tensor_with_coefficients(
            weights.encoder_v,
            ids.encoder_v.val(),
            eggroll,
            generation,
            &coefficients,
            state,
        ),
        decoder: burn_dragon_eggroll::apply_antithetic_update_to_tensor_with_coefficients(
            weights.decoder,
            ids.decoder.val(),
            eggroll,
            generation,
            &coefficients,
            state,
        ),
    };
    Ok((
        model.map_model(|dragon| dragon.with_shared_lowrank_weights(next)),
        metrics,
    ))
}

fn evaluate_eggroll_population_chunk_stacked_tensorized<B>(
    plan: &EggrollPopulationExecutionPlan,
    model: &LanguageTrainModel<B>,
    batch: SequenceBatch<B>,
    eggroll: &burn_eggroll::EggrollConfig,
    generation: u64,
    pair_start: usize,
    pair_count: usize,
) -> Result<Vec<f64>>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    if batch.summary_event_mask.is_some() {
        return Err(anyhow!(
            "EGGROLL stacked tensorized population evaluator does not support summary_event_mask batches"
        ));
    }
    if batch.loss_mask.is_some() {
        return Err(anyhow!(
            "EGGROLL stacked tensorized population evaluator does not support target loss masks; use AdamW answer-only warmup or the verifier-ranked EGGROLL path"
        ));
    }
    let tile_pair_count = plan
        .population_tile_size
        .map(|tile| make_even_population_size(tile).saturating_div(2).max(1))
        .unwrap_or(pair_count.max(1))
        .min(pair_count.max(1));
    if tile_pair_count < pair_count {
        let mut losses = Vec::with_capacity(pair_count * 2);
        let mut local_start = pair_start;
        let pair_end = pair_start.saturating_add(pair_count);
        while local_start < pair_end {
            let local_count = (pair_end - local_start).min(tile_pair_count);
            losses.extend(evaluate_eggroll_population_chunk_stacked_tensorized(
                plan,
                model,
                batch.clone(),
                eggroll,
                generation,
                local_start,
                local_count,
            )?);
            local_start = local_start.saturating_add(local_count);
        }
        return Ok(losses);
    }

    let logits = match plan.backend {
        EggrollPopulationExecutionBackend::Factorized => {
            let lowrank = build_shared_lowrank_population_factors(
                model, eggroll, generation, pair_start, pair_count,
            );
            model
                .model
                .forward_with_shared_lowrank_population_factors(batch.inputs.clone(), lowrank)
        }
        EggrollPopulationExecutionBackend::Auto
        | EggrollPopulationExecutionBackend::Reference
        | EggrollPopulationExecutionBackend::Cuda => {
            let lowrank = build_shared_lowrank_population_weights(
                model, eggroll, generation, pair_start, pair_count,
            );
            model
                .model
                .forward_with_shared_lowrank_population(batch.inputs.clone(), lowrank)
        }
    };
    Ok(scalar_values_from_loss_tensors(vec![
        population_next_token_losses(model, logits, batch.targets, pair_count * 2),
    ]))
}

fn build_shared_lowrank_population_factors<B>(
    model: &LanguageTrainModel<B>,
    eggroll: &burn_eggroll::EggrollConfig,
    generation: u64,
    pair_start: usize,
    pair_count: usize,
) -> SharedLowrankPopulationFactors<B>
where
    B: BackendTrait,
{
    let base = model.model.shared_lowrank_weights();
    let ids = model.model.shared_lowrank_param_ids();
    let sigma = eggroll.effective_sigma(generation);
    let encoder_spec = burn_eggroll::MatrixNoisePopulationSpec::new(
        eggroll.population.seed,
        ids.encoder.val(),
        generation,
        pair_start as u64,
        pair_count,
        eggroll.population.rank,
    );
    let encoder_v_spec = burn_eggroll::MatrixNoisePopulationSpec::new(
        eggroll.population.seed,
        ids.encoder_v.val(),
        generation,
        pair_start as u64,
        pair_count,
        eggroll.population.rank,
    );
    let decoder_spec = burn_eggroll::MatrixNoisePopulationSpec::new(
        eggroll.population.seed,
        ids.decoder.val(),
        generation,
        pair_start as u64,
        pair_count,
        eggroll.population.rank,
    );
    let [heads, embd, latent_capacity] = base.encoder.shape().dims::<3>();
    let [decoder_rows, decoder_cols] = base.decoder.shape().dims::<2>();
    let device = base.encoder.device();
    let encoder = burn_eggroll::low_rank_factors_3d_antithetic_population_with_mode(
        heads,
        embd,
        latent_capacity,
        encoder_spec,
        eggroll.population.matrix_noise,
        &device,
    );
    let encoder_v = burn_eggroll::low_rank_factors_3d_antithetic_population_with_mode(
        heads,
        embd,
        latent_capacity,
        encoder_v_spec,
        eggroll.population.matrix_noise,
        &device,
    );
    let decoder = burn_eggroll::low_rank_factors_2d_antithetic_population_with_mode(
        decoder_rows,
        decoder_cols,
        decoder_spec,
        eggroll.population.matrix_noise,
        &device,
    );

    SharedLowrankPopulationFactors {
        encoder_a: encoder.a,
        encoder_b: encoder.b,
        encoder_v_a: encoder_v.a,
        encoder_v_b: encoder_v.b,
        decoder_a: decoder.a,
        decoder_b: decoder.b,
        signs: encoder.signs,
        encoder_scale: encoder.scale,
        encoder_v_scale: encoder_v.scale,
        decoder_scale: decoder.scale,
        sigma,
    }
}

fn build_shared_lowrank_population_weights<B>(
    model: &LanguageTrainModel<B>,
    eggroll: &burn_eggroll::EggrollConfig,
    generation: u64,
    pair_start: usize,
    pair_count: usize,
) -> SharedLowrankPopulationWeights<B>
where
    B: BackendTrait,
{
    let base = model.model.shared_lowrank_weights();
    let ids = model.model.shared_lowrank_param_ids();
    let sigma = eggroll.effective_sigma(generation);
    let encoder_spec = burn_eggroll::MatrixNoisePopulationSpec::new(
        eggroll.population.seed,
        ids.encoder.val(),
        generation,
        pair_start as u64,
        pair_count,
        eggroll.population.rank,
    );
    let encoder_v_spec = burn_eggroll::MatrixNoisePopulationSpec::new(
        eggroll.population.seed,
        ids.encoder_v.val(),
        generation,
        pair_start as u64,
        pair_count,
        eggroll.population.rank,
    );
    let decoder_spec = burn_eggroll::MatrixNoisePopulationSpec::new(
        eggroll.population.seed,
        ids.decoder.val(),
        generation,
        pair_start as u64,
        pair_count,
        eggroll.population.rank,
    );

    SharedLowrankPopulationWeights {
        encoder: burn_eggroll::perturb_matrix_3d_antithetic_population_with_mode(
            base.encoder,
            sigma,
            encoder_spec,
            eggroll.population.matrix_noise,
        ),
        encoder_v: burn_eggroll::perturb_matrix_3d_antithetic_population_with_mode(
            base.encoder_v,
            sigma,
            encoder_v_spec,
            eggroll.population.matrix_noise,
        ),
        decoder: burn_eggroll::perturb_matrix_2d_antithetic_population_with_mode(
            base.decoder,
            sigma,
            decoder_spec,
            eggroll.population.matrix_noise,
        ),
    }
}

fn population_next_token_losses<B>(
    model: &LanguageTrainModel<B>,
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    population: usize,
) -> Tensor<B, 1>
where
    B: BackendTrait,
{
    let [stacked_batch, _time, _vocab] = logits.shape().dims::<3>();
    assert_eq!(
        stacked_batch % population,
        0,
        "stacked population batch must divide evenly"
    );
    let base_batch = stacked_batch / population;
    let targets = Tensor::cat(
        (0..population).map(|_| targets.clone()).collect::<Vec<_>>(),
        0,
    );
    model
        .model
        .language_token_losses_from_logits(logits, targets)
        .mean_dim(1)
        .reshape([population, base_batch])
        .mean_dim(1)
        .reshape([population])
}

fn resolve_eggroll_chunk_autotune_candidates(optimizer_cfg: &OptimizerConfig) -> Vec<usize> {
    let population_size = optimizer_cfg.eggroll.population.population_size.max(2);
    let configured = optimizer_cfg
        .eggroll
        .population
        .population_chunk_size
        .max(2)
        .min(population_size);
    let max_probe = optimizer_cfg
        .eggroll_auto_population
        .chunk_autotune
        .max_probe_population_size
        .max(2)
        .min(population_size);
    let configured = make_even_population_size(configured.min(max_probe));
    let mut candidates = if optimizer_cfg
        .eggroll_auto_population
        .chunk_autotune
        .candidates
        .is_empty()
    {
        vec![16, 32, 64, 128, configured]
    } else {
        let mut candidates = optimizer_cfg
            .eggroll_auto_population
            .chunk_autotune
            .candidates
            .clone();
        candidates.push(configured);
        candidates
    };
    candidates = candidates
        .into_iter()
        .map(|candidate| make_even_population_size(candidate.max(2).min(max_probe)))
        .filter(|candidate| *candidate >= 2 && *candidate <= max_probe)
        .collect();
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

fn make_even_population_size(value: usize) -> usize {
    value.saturating_sub(value % 2).max(2)
}

fn measure_eggroll_chunk_candidate<B>(
    plan: &EggrollPopulationExecutionPlan,
    model: &LanguageTrainModel<B>,
    batch: SequenceBatch<B>,
    eggroll: &burn_eggroll::EggrollConfig,
    population_chunk_size: usize,
    max_probe_population_size: usize,
) -> Result<EggrollChunkAutotuneCandidateReport>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    let evaluated_population_size = make_even_population_size(
        population_chunk_size
            .min(max_probe_population_size)
            .min(eggroll.population.population_size)
            .max(2),
    );
    let pair_count = (evaluated_population_size / 2).max(1);
    let started = burn_dragon_time::Instant::now();
    let losses = evaluate_eggroll_population_chunk(plan, model, batch, eggroll, 0, 0, pair_count)?;
    let elapsed_ms = started.elapsed().as_millis() as f64;
    let mean_loss = if losses.is_empty() {
        f64::NAN
    } else {
        losses.iter().sum::<f64>() / losses.len() as f64
    };
    let forward_evaluations_per_second = if elapsed_ms > 0.0 {
        losses.len() as f64 * 1000.0 / elapsed_ms
    } else {
        0.0
    };
    Ok(EggrollChunkAutotuneCandidateReport {
        population_chunk_size,
        evaluated_population_size: losses.len(),
        elapsed_ms,
        forward_evaluations_per_second,
        mean_loss,
    })
}

fn build_dynamic_train_loader<B>(
    env: &TrainEnvironment<'_, B>,
    batch_size: usize,
    consumed_steps: usize,
) -> Arc<dyn DataLoader<B, SequenceBatch<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let batch_size = batch_size.max(1);
    let Some(train_dataset) = env.train_dataset.as_ref() else {
        return Arc::clone(&env.train_loader);
    };
    if env.training.tbptt_persist_across_steps {
        Arc::new(
            StreamingDataLoader::<B>::new(
                Arc::clone(train_dataset),
                DatasetSplit::Train,
                env.device,
                env.train_loader.num_items().max(1),
                Some(env.total_steps),
                env.training.min_logical_block_size,
                env.training.seed,
            )
            .with_batch_size(batch_size)
            .with_initial_consumed_steps(consumed_steps)
            .with_summary_event_token_ids(env.summary_event_token_ids.clone()),
        )
    } else {
        Arc::new(
            RandomDataLoader::<B>::new(
                Arc::clone(train_dataset),
                DatasetSplit::Train,
                env.device,
                env.train_loader.num_items().max(1),
                Some(env.total_steps),
            )
            .with_batch_size(batch_size)
            .with_initial_consumed_steps(consumed_steps)
            .with_ruliad_policy_batch(env.training.ruliad_supervision.verifier_reward.enabled)
            .with_summary_event_token_ids(env.summary_event_token_ids.clone()),
        )
    }
}

fn build_dynamic_valid_loader<B>(
    env: &TrainEnvironment<'_, B>,
    batch_size: usize,
) -> Arc<dyn DataLoader<ValidBackend<B>, SequenceBatch<ValidBackend<B>>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let Some(valid_dataset) = env.valid_dataset.as_ref() else {
        return Arc::clone(&env.valid_loader);
    };
    Arc::new(
        RandomDataLoader::<ValidBackend<B>>::new(
            Arc::clone(valid_dataset),
            DatasetSplit::Val,
            env.device,
            env.valid_steps.max(1),
            None,
        )
        .with_batch_size(batch_size.max(1))
        .with_ruliad_policy_batch(env.training.ruliad_supervision.verifier_reward.enabled)
        .with_summary_event_token_ids(env.summary_event_token_ids.clone()),
    )
}

pub(crate) fn train_with_dynamic_neuron_scaling_scheduler<B, S>(
    env: &TrainEnvironment<'_, B>,
    mut model: LanguageTrainModel<B>,
    mut optimizer: crate::train::continual_backprop::LanguageOptimizer<B>,
    mut scheduler: S,
) -> Result<DragonModel<ValidBackend<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    S: LrScheduler + Clone + 'static,
{
    fs::create_dir_all(env.run_dir)?;
    let source_selection_dataset = env.source_selection_dataset.clone();
    let dynamics_control_slot = DragonDynamicsControlSlot::default();
    let event_handles = crate::train::events::build_training_event_handles(
        env.run_name,
        env.run_dir,
        env.train_loader.num_items(),
        env.training,
        source_selection_dataset,
        env.neuron_scaling_slot
            .as_ref()
            .map(|slot| (env.model_config.latent_total(), slot.clone())),
        Some(dynamics_control_slot.clone()),
    )?;
    let bus = event_handles.metric_logger.bus();
    let emit_step_events = env.training.events.persist_step_events;
    let steps_per_epoch = env.train_loader.num_items().max(1);
    let start_epoch = env
        .resume_checkpoint_epoch
        .map(|epoch| epoch + 1)
        .unwrap_or(1);
    let mut active_batch_size = env.training.batch_size.max(1);
    let mut active_grad_accumulation = env.training.gradient_accumulation_steps.max(1);
    let mut active_train_loader = Arc::clone(&env.train_loader);
    let mut active_valid_loader = Arc::clone(&env.valid_loader);
    let mut current_model_config = env.model_config.clone();
    let mut scale_generation = 0usize;
    let mut stability = ContinualLearningStabilityState::default();
    let mut dynamics_control = ActiveDynamicsControl::default();
    let mut last_cbp_telemetry_step = 0usize;
    let mut best_valid_loss: Option<f64> = None;
    let mut best_valid_epoch: Option<usize> = None;

    if let Some(epoch) = env.resume_checkpoint_epoch {
        let (loaded_model, loaded_model_config) = load_dragon_training_state_checkpoint(
            env.run_dir,
            epoch,
            env.model_config,
            env.device,
            &mut optimizer,
            &mut scheduler,
            &mut dynamics_control,
        )?;
        model.model = loaded_model;
        if let Some(loaded_model_config) = loaded_model_config {
            current_model_config = loaded_model_config;
        }
        optimizer.refresh_continual_backprop_fresh_model(DragonModel::<B>::new(
            current_model_config.clone(),
            env.device,
        ));
        let historical_best = historical_best_validation(env.run_dir, epoch);
        best_valid_loss = historical_best.best_loss;
        best_valid_epoch = historical_best.best_checkpoint_epoch;
        stability.best_valid_loss = historical_best.best_loss;
        stability.best_checkpoint_epoch = historical_best.best_checkpoint_epoch;
        info!(
            "resumed dynamic training checkpoint epoch={} historical_best_valid_loss={:?} historical_best_checkpoint_epoch={:?}",
            epoch, best_valid_loss, best_valid_epoch
        );
    }

    let dynamic_neuron_scaling = env.neuron_scaling_slot.is_some();
    let update_parameters = parameter_updates_enabled(env.training);
    info!("run name: {}", env.run_name);
    info!(
        "training strategy: mode={:?} replicas={} event_scheduler=true dynamic_neuron_scaling={} parameter_updates={} start_epoch={}",
        env.parallel_runtime.mode,
        env.devices.len(),
        dynamic_neuron_scaling,
        update_parameters,
        start_epoch
    );
    info!(
        "checkpoint policy: logical_epoch_steps={} keep_last={} keep_best_valid_loss=true event_scheduler=true dynamic_neuron_scaling={}",
        env.train_loader.num_items(),
        CHECKPOINT_KEEP_LAST,
        dynamic_neuron_scaling
    );

    for epoch in start_epoch..=env.epochs {
        if event_handles.interrupter.should_stop() {
            let reason = event_handles
                .interrupter
                .get_message()
                .unwrap_or_else(|| "training interrupted".to_string());
            info!("Training interrupted: {reason}");
            break;
        }

        let mut iterator = active_train_loader.iter();
        let mut iteration = 0usize;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0usize;
        let mut last_lr = 0.0;
        let mut stop_requested = false;

        while let Some(item) = iterator.next() {
            iteration += 1;
            let absolute_step = epoch
                .saturating_sub(1)
                .saturating_mul(steps_per_epoch)
                .saturating_add(iteration.saturating_sub(1));
            if emit_step_events {
                let _ = bus.send_step_started(StepStarted {
                    run_id: env.run_name.to_string(),
                    absolute_step,
                    epoch,
                });
            }

            model.set_recovery_auxiliary_active(dynamics_control.recovery_auxiliary_active());
            let item = burn_train::TrainStep::step(&model, item);
            let source_selection_due = source_selection_telemetry_due(env, absolute_step);
            let log_train_metrics =
                iteration % env.training.log_frequency.max(1) == 0 || iteration == steps_per_epoch;
            let mean_train_loss = if source_selection_due || log_train_metrics {
                let train_output = item.item.sync();
                let loss_value: LossValue<ValidBackend<B>> = train_output.adapt();
                Some(mean_scalar_from_loss(loss_value.value()))
            } else {
                None
            };
            if source_selection_due && let Some(mean_train_loss) = mean_train_loss {
                emit_source_selection_telemetry(env, absolute_step, mean_train_loss, &bus);
            }
            if update_parameters {
                accumulator.accumulate(&model, item.grads);
                accumulation_current += 1;
            } else {
                last_lr = 0.0;
            }

            if update_parameters && active_grad_accumulation <= accumulation_current {
                if apply_pending_dynamics_control(
                    env,
                    &dynamics_control_slot,
                    &mut dynamics_control,
                    &mut optimizer,
                    &model,
                ) == DynamicsControlOutcome::Stop
                {
                    stop_requested = true;
                    break;
                }
                let lr = scheduler.step() * dynamics_control.lr_scale;
                let grads = accumulator.grads();
                model = optimizer.step(lr, model, grads);
                accumulation_current = 0;
                last_lr = lr;
                emit_continual_backprop_telemetry(
                    env,
                    &optimizer,
                    epoch,
                    absolute_step,
                    &bus,
                    &mut last_cbp_telemetry_step,
                );
            }

            if log_train_metrics && let Some(mean_train_loss) = mean_train_loss {
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: train_loss_metric_name(env.training).to_string(),
                    value: mean_train_loss,
                    running_value: mean_train_loss,
                });
                let _ = bus.send_metric_sample(TrainingMetricSample {
                    run_id: env.run_name.to_string(),
                    split: TrainingMetricSplit::Train,
                    epoch,
                    step_in_epoch: iteration,
                    absolute_step,
                    name: "Learning Rate".to_string(),
                    value: last_lr,
                    running_value: last_lr,
                });
                emit_predictive_coding_telemetry(
                    env,
                    epoch,
                    iteration,
                    absolute_step,
                    model.gradient_scale_step_index(),
                    &bus,
                );
                emit_latent_reasoning_telemetry(env, epoch, iteration, absolute_step, &bus);
            }
            if emit_step_events {
                let _ = bus.send_step_finished(StepFinished {
                    run_id: env.run_name.to_string(),
                    absolute_step,
                    epoch,
                    loss: mean_train_loss,
                });
            }

            if log_train_metrics {
                let progress = iterator.progress();
                info!(
                    "train epoch={} step={}/{} loss={:.4} lr={:.6} global_progress={}/{}",
                    epoch,
                    progress.items_processed,
                    progress.items_total,
                    mean_train_loss.unwrap_or(f64::NAN),
                    last_lr,
                    epoch,
                    env.epochs
                );
            }
        }

        if stop_requested {
            info!("stopping training after dynamics control request");
            break;
        }

        if update_parameters && accumulation_current > 0 {
            if apply_pending_dynamics_control(
                env,
                &dynamics_control_slot,
                &mut dynamics_control,
                &mut optimizer,
                &model,
            ) == DynamicsControlOutcome::Stop
            {
                info!("stopping training after dynamics control request");
                break;
            }
            let lr = scheduler.step() * dynamics_control.lr_scale;
            let grads = accumulator.grads();
            model = optimizer.step(lr, model, grads);
            let absolute_step = epoch.saturating_mul(steps_per_epoch).saturating_sub(1);
            emit_continual_backprop_telemetry(
                env,
                &optimizer,
                epoch,
                absolute_step,
                &bus,
                &mut last_cbp_telemetry_step,
            );
        }
        drop(iterator);
        let _ = bus.send_epoch_summary(TrainingEpochSummary {
            run_id: env.run_name.to_string(),
            split: TrainingMetricSplit::Train,
            epoch,
        });

        let validation = run_dynamic_validation(
            env,
            &active_valid_loader,
            &model,
            epoch,
            steps_per_epoch,
            active_batch_size,
            &bus,
        )?;
        let valid_loss = validation.loss;
        info!("valid epoch={} loss={valid_loss:.4}", epoch);
        if let Some(source_weighted_loss) = validation.source_weighted_loss {
            info!(
                "valid epoch={} source_weighted_loss={source_weighted_loss:.4}",
                epoch
            );
        }
        if let Some(stream_warm_loss) = validation.stream_warm_loss {
            info!(
                "valid epoch={} stream_warm_loss={stream_warm_loss:.4}",
                epoch
            );
        }
        let checkpoint_promoted = should_promote_checkpoint(
            &validation,
            best_valid_loss,
            stability.best_ruliad_competence,
            &env.training.gates,
        );
        if checkpoint_promoted {
            best_valid_loss = Some(valid_loss);
            best_valid_epoch = Some(epoch);
            stability.best_checkpoint_epoch = Some(epoch);
            if let Some(competence) = validation
                .ruliad_eval_report
                .as_ref()
                .and_then(ruliad_competence_key)
            {
                stability.best_ruliad_competence = Some(competence);
            }
        }
        let absolute_step = epoch.saturating_mul(steps_per_epoch).saturating_sub(1);
        save_dragon_training_state_checkpoint(
            env.run_dir,
            epoch,
            &model,
            &current_model_config,
            &optimizer,
            &scheduler,
            &dynamics_control,
        )?;
        save_source_selection_state_checkpoint(
            env.run_dir,
            epoch,
            absolute_step,
            env.source_selection_dataset.as_ref(),
        )?;
        prune_dragon_model_checkpoints(env.run_dir, epoch, best_valid_epoch)?;
        apply_continual_learning_stability_policy(
            env,
            validation,
            epoch,
            absolute_step,
            &mut stability,
            &bus,
        );
        let _ = bus.send_checkpoint(CheckpointEvent {
            run_id: env.run_name.to_string(),
            checkpoint_id: format!("model-{epoch}"),
            epoch: Some(epoch),
            absolute_step: Some(absolute_step),
            promoted: checkpoint_promoted,
        });
        let _ = bus.flush();
        if handle_post_validation_dynamics_control(
            env,
            &dynamics_control_slot,
            &mut dynamics_control,
            &mut optimizer,
            &mut scheduler,
            &mut model,
            &mut current_model_config,
            epoch,
        )? == DynamicsControlOutcome::Stop
        {
            info!("stopping training after dynamics control request");
            break;
        }

        if let Some(request) = env
            .neuron_scaling_slot
            .as_ref()
            .and_then(|slot| slot.take())
        {
            if let Some((old_capacity_units, new_capacity_units)) = apply_dynamic_neuron_scale(
                env,
                &mut model,
                &mut optimizer,
                &mut current_model_config,
                &mut scale_generation,
                request,
                epoch,
                absolute_step,
                &bus,
                active_batch_size,
                active_grad_accumulation,
            )? {
                let next_batch_size =
                    crate::train::startup_autotune::resolve_scaled_auto_batch_size(
                        &env.training.auto_batch_size,
                        active_batch_size,
                        old_capacity_units,
                        new_capacity_units,
                    );
                let next_grad_accumulation =
                    crate::train::startup_autotune::resolve_gradient_accumulation_steps(
                        next_batch_size,
                        env.training.gradient_accumulation_steps,
                        env.training.target_effective_batch_size,
                    );
                if next_batch_size != active_batch_size
                    || next_grad_accumulation != active_grad_accumulation
                {
                    active_batch_size = next_batch_size;
                    active_grad_accumulation = next_grad_accumulation;
                    let consumed_steps = epoch.saturating_mul(steps_per_epoch);
                    active_train_loader =
                        build_dynamic_train_loader(env, active_batch_size, consumed_steps);
                    active_valid_loader = build_dynamic_valid_loader(env, active_batch_size);
                    info!(
                        "auto batch after neuron scale: batch_size={} gradient_accumulation_steps={} effective_batch_size={} consumed_steps={}",
                        active_batch_size,
                        active_grad_accumulation,
                        active_batch_size.saturating_mul(active_grad_accumulation),
                        consumed_steps
                    );
                    emit_policy_gate(
                        env,
                        &bus,
                        "auto_batch_size_after_neuron_scale",
                        epoch,
                        absolute_step,
                        format!(
                            "auto batch selected batch_size={} gradient_accumulation_steps={} after capacity {} -> {}",
                            active_batch_size,
                            active_grad_accumulation,
                            old_capacity_units,
                            new_capacity_units
                        ),
                    );
                }
            }
            let _ = bus.flush();
        }
    }

    log_theoretical_profile(
        &current_model_config,
        active_batch_size.saturating_mul(active_grad_accumulation),
        env.training.block_size,
        env.backend_name,
    );

    Ok(model.valid().model)
}

fn run_dynamic_validation<B>(
    env: &TrainEnvironment<'_, B>,
    valid_loader: &Arc<dyn DataLoader<ValidBackend<B>, SequenceBatch<ValidBackend<B>>>>,
    model: &LanguageTrainModel<B>,
    epoch: usize,
    steps_per_epoch: usize,
    batch_size: usize,
    bus: &TrainingEventBus,
) -> Result<DynamicValidationReport>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let valid_model = model.valid();
    let mut iterator = valid_loader.iter();
    let mut total = 0.0;
    let mut count = 0usize;
    let mut output_degeneracy = None;
    let mut latent_eval_sweep_emitted = false;
    let probe_enabled = epoch.is_multiple_of(env.training.events.degeneracy_probe_every_epochs);
    let probe_absolute_step = epoch.saturating_mul(steps_per_epoch).saturating_sub(1);
    while let Some(item) = iterator.next() {
        let eval_sweep_enabled =
            !latent_eval_sweep_emitted && !latent_eval_step_sweep(env.training).is_empty();
        let degeneracy_probe_enabled = probe_enabled && output_degeneracy.is_none();
        let item_for_eval_sweep = item.clone();
        let (loss_tensor, degeneracy) = if degeneracy_probe_enabled {
            valid_model.validation_loss_and_output_degeneracy(
                item,
                env.training.events.degeneracy_probe_tokens,
                dataset_eos_id(env.source_selection_dataset.as_ref()),
            )
        } else {
            let output = valid_model.step(item);
            let loss_value: LossValue<ValidBackend<B>> = output.adapt();
            (loss_value.value(), None)
        };
        let loss = mean_scalar_from_loss(loss_tensor);
        count += 1;
        total += loss;
        let absolute_step = epoch
            .saturating_sub(1)
            .saturating_mul(steps_per_epoch)
            .saturating_add(count.saturating_sub(1));
        if let Some(degeneracy) = degeneracy {
            emit_output_degeneracy(env, epoch, probe_absolute_step, &degeneracy, bus);
            output_degeneracy = Some(degeneracy);
        }
        if eval_sweep_enabled {
            emit_latent_eval_step_validation_sweep(
                env.run_name,
                env.training,
                env.source_selection_dataset.as_ref(),
                epoch,
                probe_absolute_step,
                &valid_model,
                item_for_eval_sweep,
                dataset_eos_id(env.source_selection_dataset.as_ref()),
                degeneracy_probe_enabled,
                bus,
            );
            latent_eval_sweep_emitted = true;
        }
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: env.run_name.to_string(),
            split: TrainingMetricSplit::Valid,
            epoch,
            step_in_epoch: count,
            absolute_step,
            name: "Loss".to_string(),
            value: loss,
            running_value: total / count as f64,
        });
        emit_teacher_forced_validation_metric(
            env.run_name,
            env.source_selection_dataset.as_ref(),
            epoch,
            count,
            absolute_step,
            loss,
            total / count as f64,
            bus,
        );
    }
    let mean = if count == 0 {
        0.0
    } else {
        total / count as f64
    };
    if env.training.tbptt_persist_across_steps {
        let absolute_step = epoch
            .saturating_sub(1)
            .saturating_mul(steps_per_epoch)
            .saturating_add(count);
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: env.run_name.to_string(),
            split: TrainingMetricSplit::Valid,
            epoch,
            step_in_epoch: count.saturating_add(1),
            absolute_step,
            name: METRIC_RANDOM_COLD_LOSS.to_string(),
            value: mean,
            running_value: mean,
        });
    }
    let source_weighted_loss =
        run_source_weighted_validation(env, &valid_model, epoch, steps_per_epoch, batch_size, bus)?;
    let ruliad_eval_report = run_ruliad_correctness_validation(
        env.run_name,
        env.training,
        env.source_selection_dataset
            .as_ref()
            .or(env.valid_dataset.as_ref()),
        &valid_model,
        epoch,
        steps_per_epoch,
        env.device,
        output_degeneracy.as_ref(),
        bus,
    )?;
    if let Some(report) = ruliad_eval_report.as_ref() {
        let capability_gate = emit_ruliad_capability_gate_metrics(
            env.run_name,
            epoch,
            epoch.saturating_mul(steps_per_epoch).saturating_sub(1),
            report,
            output_degeneracy.as_ref(),
            &env.training.gates,
            bus,
        );
        if capability_gate.passed {
            model.set_latent_reasoning_capability_gate_open(true);
        }
        if env.training.events.source_selection_capability_feedback {
            emit_source_selection_capability_feedback_sample(
                env.run_name,
                env.source_selection_dataset.as_ref(),
                epoch.saturating_mul(steps_per_epoch).saturating_sub(1),
                report,
                bus,
            );
        }
    }
    let stream_warm_loss = if env.training.tbptt_persist_across_steps {
        run_stream_warm_validation(env, model, epoch, steps_per_epoch, batch_size, bus)?
    } else {
        None
    };
    if let Some(source_weighted_loss) = source_weighted_loss {
        let delta = source_weighted_loss - mean;
        let ratio = if mean.abs() <= f64::EPSILON {
            0.0
        } else {
            source_weighted_loss / mean
        };
        let absolute_step = epoch
            .saturating_sub(1)
            .saturating_mul(steps_per_epoch)
            .saturating_add(count);
        for (name, value) in [
            ("Source Weighted Loss Delta", delta),
            ("Source Weighted Loss Ratio", ratio),
        ] {
            let _ = bus.send_metric_sample(TrainingMetricSample {
                run_id: env.run_name.to_string(),
                split: TrainingMetricSplit::Valid,
                epoch,
                step_in_epoch: count.saturating_add(1),
                absolute_step,
                name: name.to_string(),
                value,
                running_value: value,
            });
        }
    }
    let _ = bus.send_epoch_summary(TrainingEpochSummary {
        run_id: env.run_name.to_string(),
        split: TrainingMetricSplit::Valid,
        epoch,
    });
    let _ = bus.send_validation_finished(ValidationFinished {
        run_id: env.run_name.to_string(),
        epoch,
        absolute_step: Some(epoch.saturating_mul(steps_per_epoch).saturating_sub(1)),
        loss: Some(mean),
    });
    Ok(DynamicValidationReport {
        loss: mean,
        source_weighted_loss,
        stream_warm_loss,
        output_degeneracy,
        ruliad_eval_report,
    })
}

fn run_stream_warm_validation<B>(
    env: &TrainEnvironment<'_, B>,
    model: &LanguageTrainModel<B>,
    epoch: usize,
    steps_per_epoch: usize,
    batch_size: usize,
    bus: &TrainingEventBus,
) -> Result<Option<f64>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let Some(valid_dataset) = env.valid_dataset.as_ref() else {
        return Ok(None);
    };
    let loader = StreamingDataLoader::<ValidBackend<B>>::new(
        Arc::clone(valid_dataset),
        DatasetSplit::Val,
        env.device,
        env.valid_steps.max(1),
        None,
        env.training.min_logical_block_size,
        env.training.seed,
    )
    .with_batch_size(batch_size.max(1))
    .with_summary_event_token_ids(env.summary_event_token_ids.clone());
    let valid_model = model.valid();
    let mut state = valid_model.model.init_state();
    let mut iterator = loader.iter();
    let mut total = 0.0;
    let mut count = 0usize;
    while let Some(item) = iterator.next() {
        let output = valid_model.step_with_stream_state(item, &mut state);
        let loss_value: LossValue<ValidBackend<B>> = output.adapt();
        let loss = mean_scalar_from_loss(loss_value.value());
        count += 1;
        total += loss;
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: env.run_name.to_string(),
            split: TrainingMetricSplit::Valid,
            epoch,
            step_in_epoch: count,
            absolute_step: epoch
                .saturating_sub(1)
                .saturating_mul(steps_per_epoch)
                .saturating_add(count.saturating_sub(1)),
            name: METRIC_STREAM_WARM_LOSS.to_string(),
            value: loss,
            running_value: total / count as f64,
        });
    }
    Ok((count > 0).then_some(total / count as f64))
}

fn run_dynamic_validation_forward_only<B>(
    env: &ForwardEggrollTrainEnvironment<'_, B>,
    model: &LanguageTrainModel<B>,
    epoch: usize,
    steps_per_epoch: usize,
    bus: &TrainingEventBus,
) -> Result<DynamicValidationReport>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    let mut iterator = env.valid_loader.iter();
    let mut total = 0.0;
    let mut count = 0usize;
    let mut output_degeneracy = None;
    let mut latent_eval_sweep_emitted = false;
    let probe_enabled = epoch.is_multiple_of(env.training.events.degeneracy_probe_every_epochs);
    let probe_absolute_step = epoch.saturating_mul(steps_per_epoch).saturating_sub(1);
    while let Some(item) = iterator.next() {
        let eval_sweep_enabled =
            !latent_eval_sweep_emitted && !latent_eval_step_sweep(env.training).is_empty();
        let degeneracy_probe_enabled = probe_enabled && output_degeneracy.is_none();
        let item_for_eval_sweep = item.clone();
        let (loss_tensor, degeneracy) = if degeneracy_probe_enabled {
            model.validation_loss_and_output_degeneracy(
                item,
                env.training.events.degeneracy_probe_tokens,
                dataset_eos_id(env.source_selection_dataset.as_ref()),
            )
        } else {
            let output = ValidStep::step(model, item);
            let loss_value: LossValue<B> = output.adapt();
            (loss_value.value(), None)
        };
        let loss = mean_scalar_from_loss(loss_tensor);
        count += 1;
        total += loss;
        let absolute_step = epoch
            .saturating_sub(1)
            .saturating_mul(steps_per_epoch)
            .saturating_add(count.saturating_sub(1));
        if let Some(degeneracy) = degeneracy {
            emit_output_degeneracy_sample(
                env.run_name,
                env.source_selection_dataset.as_ref(),
                epoch,
                probe_absolute_step,
                &degeneracy,
                bus,
            );
            output_degeneracy = Some(degeneracy);
        }
        if eval_sweep_enabled {
            emit_latent_eval_step_validation_sweep(
                env.run_name,
                env.training,
                env.source_selection_dataset.as_ref(),
                epoch,
                probe_absolute_step,
                model,
                item_for_eval_sweep,
                dataset_eos_id(env.source_selection_dataset.as_ref()),
                degeneracy_probe_enabled,
                bus,
            );
            latent_eval_sweep_emitted = true;
        }
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: env.run_name.to_string(),
            split: TrainingMetricSplit::Valid,
            epoch,
            step_in_epoch: count,
            absolute_step,
            name: "Loss".to_string(),
            value: loss,
            running_value: total / count as f64,
        });
        emit_teacher_forced_validation_metric(
            env.run_name,
            env.source_selection_dataset.as_ref(),
            epoch,
            count,
            absolute_step,
            loss,
            total / count as f64,
            bus,
        );
    }
    let mean = if count == 0 {
        0.0
    } else {
        total / count as f64
    };
    let source_weighted_loss =
        run_source_weighted_validation_forward_only(env, model, epoch, steps_per_epoch, bus)?;
    let ruliad_eval_report = run_ruliad_correctness_validation(
        env.run_name,
        env.training,
        env.source_selection_dataset.as_ref(),
        model,
        epoch,
        steps_per_epoch,
        env.device,
        output_degeneracy.as_ref(),
        bus,
    )?;
    if let Some(report) = ruliad_eval_report.as_ref() {
        let capability_gate = emit_ruliad_capability_gate_metrics(
            env.run_name,
            epoch,
            epoch.saturating_mul(steps_per_epoch).saturating_sub(1),
            report,
            output_degeneracy.as_ref(),
            &env.training.gates,
            bus,
        );
        if capability_gate.passed {
            model.set_latent_reasoning_capability_gate_open(true);
        }
        if env.training.events.source_selection_capability_feedback {
            emit_source_selection_capability_feedback_sample(
                env.run_name,
                env.source_selection_dataset.as_ref(),
                epoch.saturating_mul(steps_per_epoch).saturating_sub(1),
                report,
                bus,
            );
        }
    }
    if let Some(source_weighted_loss) = source_weighted_loss {
        let delta = source_weighted_loss - mean;
        let ratio = if mean.abs() <= f64::EPSILON {
            0.0
        } else {
            source_weighted_loss / mean
        };
        let absolute_step = epoch
            .saturating_sub(1)
            .saturating_mul(steps_per_epoch)
            .saturating_add(count);
        for (name, value) in [
            ("Source Weighted Loss Delta", delta),
            ("Source Weighted Loss Ratio", ratio),
        ] {
            let _ = bus.send_metric_sample(TrainingMetricSample {
                run_id: env.run_name.to_string(),
                split: TrainingMetricSplit::Valid,
                epoch,
                step_in_epoch: count.saturating_add(1),
                absolute_step,
                name: name.to_string(),
                value,
                running_value: value,
            });
        }
    }
    let _ = bus.send_epoch_summary(TrainingEpochSummary {
        run_id: env.run_name.to_string(),
        split: TrainingMetricSplit::Valid,
        epoch,
    });
    let _ = bus.send_validation_finished(ValidationFinished {
        run_id: env.run_name.to_string(),
        epoch,
        absolute_step: Some(epoch.saturating_mul(steps_per_epoch).saturating_sub(1)),
        loss: Some(mean),
    });
    Ok(DynamicValidationReport {
        loss: mean,
        source_weighted_loss,
        stream_warm_loss: None,
        output_degeneracy,
        ruliad_eval_report,
    })
}

fn latent_eval_step_sweep(training: &TrainingHyperparameters) -> Vec<usize> {
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

fn model_with_fixed_latent_eval_steps<B>(
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

fn emit_latent_eval_step_validation_sweep<B>(
    run_name: &str,
    training: &TrainingHyperparameters,
    source_selection_dataset: Option<&Arc<Dataset>>,
    epoch: usize,
    absolute_step: usize,
    model: &LanguageTrainModel<B>,
    batch: SequenceBatch<B>,
    eos_id: Option<i64>,
    include_degeneracy: bool,
    bus: &TrainingEventBus,
) where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    if !model.model.latent_reasoning_enabled() {
        return;
    }
    for steps in latent_eval_step_sweep(training) {
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
            run_id: run_name.to_string(),
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

fn emit_latent_reasoning_step_diagnostics(
    run_name: &str,
    epoch: usize,
    absolute_step: usize,
    steps: usize,
    diagnostics: &crate::train::steps::LatentReasoningStepDiagnostics,
    bus: &TrainingEventBus,
) {
    let prefix = format!("Latent Eval Steps {steps}");
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
            run_id: run_name.to_string(),
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
        emit_latent_step_metric(
            run_name,
            epoch,
            absolute_step,
            &prefix,
            index,
            "CE",
            value,
            bus,
        );
    }
    for (index, value) in diagnostics.step_ce_delta.iter().copied().enumerate() {
        emit_latent_step_metric(
            run_name,
            epoch,
            absolute_step,
            &prefix,
            index,
            "CE Delta",
            value,
            bus,
        );
    }
    for (index, value) in diagnostics
        .step_ce_monotonic_violation_rate
        .iter()
        .copied()
        .enumerate()
    {
        emit_latent_step_metric(
            run_name,
            epoch,
            absolute_step,
            &prefix,
            index,
            "CE Monotonic Violation Rate",
            value,
            bus,
        );
    }
    for (index, value) in diagnostics.step_entropy_bits.iter().copied().enumerate() {
        emit_latent_step_metric(
            run_name,
            epoch,
            absolute_step,
            &prefix,
            index,
            "Entropy Bits",
            value,
            bus,
        );
    }
    for (index, value) in diagnostics.step_delta_rms.iter().copied().enumerate() {
        emit_latent_step_metric(
            run_name,
            epoch,
            absolute_step,
            &prefix,
            index,
            "Delta RMS",
            value,
            bus,
        );
    }
    for (index, value) in diagnostics.step_raw_cosine.iter().copied().enumerate() {
        emit_latent_step_metric(
            run_name,
            epoch,
            absolute_step,
            &prefix,
            index,
            "Raw Cosine",
            value,
            bus,
        );
    }
    for (index, value) in diagnostics.step_energy_mean.iter().copied().enumerate() {
        emit_latent_step_metric(
            run_name,
            epoch,
            absolute_step,
            &prefix,
            index,
            "Energy Mean",
            value,
            bus,
        );
    }
    for (index, value) in diagnostics.step_energy_delta.iter().copied().enumerate() {
        emit_latent_step_metric(
            run_name,
            epoch,
            absolute_step,
            &prefix,
            index,
            "Energy Delta",
            value,
            bus,
        );
    }
    for (index, value) in diagnostics
        .step_energy_monotonic_violation_rate
        .iter()
        .copied()
        .enumerate()
    {
        emit_latent_step_metric(
            run_name,
            epoch,
            absolute_step,
            &prefix,
            index,
            "Energy Monotonic Violation Rate",
            value,
            bus,
        );
    }
}

fn emit_latent_step_metric(
    run_name: &str,
    epoch: usize,
    absolute_step: usize,
    prefix: &str,
    index: usize,
    suffix: &str,
    value: f64,
    bus: &TrainingEventBus,
) {
    let _ = bus.send_metric_sample(TrainingMetricSample {
        run_id: run_name.to_string(),
        split: TrainingMetricSplit::Valid,
        epoch,
        step_in_epoch: 0,
        absolute_step,
        name: format!("{prefix} Step {} {suffix}", index.saturating_add(1)),
        value,
        running_value: value,
    });
}

fn run_source_weighted_validation<B>(
    env: &TrainEnvironment<'_, B>,
    valid_model: &LanguageTrainModel<ValidBackend<B>>,
    epoch: usize,
    steps_per_epoch: usize,
    batch_size: usize,
    bus: &TrainingEventBus,
) -> Result<Option<f64>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
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
        let output = valid_model.step(batch);
        let loss_value: LossValue<ValidBackend<B>> = output.adapt();
        let loss = mean_scalar_from_loss(loss_value.value());
        count += 1;
        total += loss;
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: env.run_name.to_string(),
            split: TrainingMetricSplit::Valid,
            epoch,
            step_in_epoch: count,
            absolute_step,
            name: "Source Weighted Loss".to_string(),
            value: loss,
            running_value: total / count as f64,
        });
    }

    Ok((count > 0).then_some(total / count as f64))
}

fn run_ruliad_correctness_validation<B>(
    run_name: &str,
    training: &TrainingHyperparameters,
    dataset: Option<&Arc<Dataset>>,
    model: &LanguageTrainModel<B>,
    epoch: usize,
    steps_per_epoch: usize,
    device: &B::Device,
    output_degeneracy: Option<&crate::train::steps::OutputDegeneracyStats>,
    bus: &TrainingEventBus,
) -> Result<Option<burn_dragon_universality::RuliadEvalReport>>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
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

    let base_absolute_step = epoch.saturating_sub(1).saturating_mul(steps_per_epoch);
    let probe_items =
        dataset.sample_ruliad_validation_probe_items(epoch, base_absolute_step, requested_items);
    if probe_items.is_empty() {
        return Ok(None);
    }

    let base_report = run_ruliad_correctness_validation_for_items(
        run_name,
        dataset,
        model,
        epoch,
        base_absolute_step,
        device,
        training,
        &probe_items,
        "ruliad_validation_probe",
        "ruliad_correctness",
        None,
        output_degeneracy,
        bus,
    )?;
    if model.model.latent_reasoning_enabled() {
        for steps in latent_eval_step_sweep(training) {
            let eval_model = model_with_fixed_latent_eval_steps(model, steps);
            let metric_prefix = format!("Ruliad Eval Steps {steps}");
            let probe_name = format!("ruliad_correctness_eval_steps_{steps}");
            let _ = run_ruliad_correctness_validation_for_items(
                run_name,
                dataset,
                &eval_model,
                epoch,
                base_absolute_step,
                device,
                training,
                &probe_items,
                "ruliad_validation_probe",
                &probe_name,
                Some(&metric_prefix),
                None,
                bus,
            )?;
        }
    }
    Ok(Some(base_report))
}

#[allow(clippy::too_many_arguments)]
fn run_ruliad_correctness_validation_for_items<B>(
    run_name: &str,
    dataset: &Dataset,
    model: &LanguageTrainModel<B>,
    epoch: usize,
    absolute_step: usize,
    device: &B::Device,
    training: &TrainingHyperparameters,
    probe_items: &[crate::dataset::RuliadValidationProbeItem],
    dataset_name: &str,
    probe_name: &str,
    metric_prefix: Option<&str>,
    output_degeneracy: Option<&crate::train::steps::OutputDegeneracyStats>,
    bus: &TrainingEventBus,
) -> Result<burn_dragon_universality::RuliadEvalReport>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    let max_new_tokens = training.events.ruliad_correctness_probe_tokens;
    let mut items = Vec::with_capacity(probe_items.len());
    let mut completions = Vec::with_capacity(probe_items.len());
    let mut generated_token_rows = Vec::with_capacity(probe_items.len());
    let close_token_id = dataset.ruliad_document_end_token_id().map(i64::from);
    let generation_settings = crate::generation::GenerationSettings {
        max_new_tokens: Some(max_new_tokens),
        temperature: 1.0,
        top_k: Some(1),
        strategy: crate::generation::resolve_context_strategy(
            &training.context_strategy,
            training.block_size,
        ),
        stop_on_token: close_token_id,
    };

    for probe in probe_items.iter().cloned() {
        let prompt_len = probe.prompt_tokens.len();
        let full_tokens = crate::generation::generate_tokens(
            &model.model,
            probe.prompt_tokens,
            device,
            generation_settings,
            None,
        )?;
        let generated_tokens = full_tokens
            .get(prompt_len..)
            .map(|tokens| tokens.to_vec())
            .unwrap_or_default();
        let completion = dataset
            .decode_ruliad_payload_tokens(&generated_tokens, true)
            .unwrap_or_else(|| dataset.decode(&generated_tokens));
        generated_token_rows.push(generated_tokens);
        completions.push(burn_dragon_universality::RuliadCompletionRecord {
            oracle_hash: probe.item.oracle_hash.clone(),
            completion,
        });
        items.push(probe.item);
    }

    let report = burn_dragon_universality::evaluate_completions(dataset_name, &items, &completions);
    let schema_alignment = ruliad_answer_schema_alignment_summary(&items, &completions);
    let completion_degeneracy =
        ruliad_completion_degeneracy_summary(&generated_token_rows, close_token_id);
    let examples = ruliad_probe_examples(
        &items,
        &completions,
        training.events.capability_probe_example_count,
    );
    emit_ruliad_correctness_metrics_with_labels(
        run_name,
        epoch,
        absolute_step,
        &report,
        bus,
        probe_name,
        metric_prefix,
        output_degeneracy,
        &examples,
        schema_alignment,
        completion_degeneracy,
    );
    Ok(report)
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct RuliadAnswerSchemaAlignmentSummary {
    key_match_rate: f64,
    mean_key_overlap: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct RuliadCompletionDegeneracySummary {
    sequence_count: usize,
    token_count: usize,
    repetition_fraction: f64,
    distinct_1_fraction: f64,
    distinct_2_fraction: f64,
    max_period_2_to_16_fraction: f64,
    max_period_2_to_64_fraction: f64,
    dominant_period_2_to_64: usize,
}

fn ruliad_answer_schema_alignment_summary(
    items: &[burn_dragon_universality::RuliadEvalItem],
    completions: &[burn_dragon_universality::RuliadCompletionRecord],
) -> RuliadAnswerSchemaAlignmentSummary {
    if items.is_empty() {
        return RuliadAnswerSchemaAlignmentSummary::default();
    }
    let completion_by_hash = completions
        .iter()
        .map(|completion| {
            (
                completion.oracle_hash.as_str(),
                completion.completion.as_str(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut exact_matches = 0usize;
    let mut overlap_ppm_sum = 0usize;
    for item in items {
        let answer = completion_by_hash
            .get(item.oracle_hash.as_str())
            .copied()
            .and_then(burn_dragon_universality::ruliad::extract_ruliad_answer);
        let alignment = burn_dragon_universality::ruliad::ruliad_answer_key_alignment(
            &item.expected_answer,
            answer.as_deref(),
        );
        exact_matches += usize::from(alignment.exact_key_match);
        overlap_ppm_sum = overlap_ppm_sum.saturating_add(alignment.overlap_ppm);
    }
    let item_count = items.len().max(1) as f64;
    RuliadAnswerSchemaAlignmentSummary {
        key_match_rate: exact_matches as f64 / item_count,
        mean_key_overlap: overlap_ppm_sum as f64 / (item_count * 1_000_000.0),
    }
}

fn ruliad_completion_degeneracy_summary(
    completions: &[Vec<i64>],
    stop_on_token: Option<i64>,
) -> Option<RuliadCompletionDegeneracySummary> {
    let trimmed_rows = completions
        .iter()
        .map(|tokens| ruliad_completion_tokens_until_stop(tokens, stop_on_token))
        .filter(|tokens| !tokens.is_empty())
        .collect::<Vec<_>>();
    let rows = trimmed_rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
    if rows.is_empty() {
        return None;
    }
    let token_count = rows.iter().map(|tokens| tokens.len()).sum::<usize>();
    Some(RuliadCompletionDegeneracySummary {
        sequence_count: rows.len(),
        token_count,
        repetition_fraction: repeated_token_fraction(&rows),
        distinct_1_fraction: row_weighted_distinct_n_fraction(&rows, 1),
        distinct_2_fraction: row_weighted_distinct_n_fraction(&rows, 2),
        max_period_2_to_16_fraction: row_weighted_max_period_fraction(&rows, 2..=16).1,
        max_period_2_to_64_fraction: row_weighted_max_period_fraction(&rows, 2..=64).1,
        dominant_period_2_to_64: row_weighted_max_period_fraction(&rows, 2..=64).0,
    })
}

fn ruliad_completion_tokens_until_stop(tokens: &[i64], stop_on_token: Option<i64>) -> Vec<i64> {
    let Some(stop) = stop_on_token else {
        return tokens.to_vec();
    };
    match tokens.iter().position(|token| *token == stop) {
        Some(index) => tokens[..=index].to_vec(),
        None => tokens.to_vec(),
    }
}

fn repeated_token_fraction(rows: &[&[i64]]) -> f64 {
    let mut repeats = 0usize;
    let mut comparisons = 0usize;
    for tokens in rows {
        for pair in tokens.windows(2) {
            comparisons = comparisons.saturating_add(1);
            repeats += usize::from(pair[0] == pair[1]);
        }
    }
    ratio_usize(repeats, comparisons)
}

fn row_weighted_distinct_n_fraction(rows: &[&[i64]], n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let mut distinct_sum = 0usize;
    let mut window_sum = 0usize;
    for tokens in rows.iter().copied().filter(|tokens| tokens.len() >= n) {
        let window_count = tokens.len() + 1 - n;
        window_sum = window_sum.saturating_add(window_count);
        distinct_sum = distinct_sum.saturating_add(
            tokens
                .windows(n)
                .map(|window| window.to_vec())
                .collect::<HashSet<_>>()
                .len(),
        );
    }
    ratio_usize(distinct_sum, window_sum)
}

fn row_weighted_max_period_fraction(
    rows: &[&[i64]],
    periods: impl IntoIterator<Item = usize>,
) -> (usize, f64) {
    periods
        .into_iter()
        .map(|period| (period, row_weighted_period_fraction(rows, period)))
        .max_by(|(_, left), (_, right)| {
            left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or((0, 0.0))
}

fn row_weighted_period_fraction(rows: &[&[i64]], period: usize) -> f64 {
    if period == 0 {
        return 0.0;
    }
    let mut matches = 0usize;
    let mut comparisons = 0usize;
    for tokens in rows
        .iter()
        .copied()
        .filter(|tokens| tokens.len() >= period.saturating_mul(2))
    {
        comparisons = comparisons.saturating_add(tokens.len() - period);
        matches = matches.saturating_add(
            (period..tokens.len())
                .filter(|idx| tokens[*idx] == tokens[*idx - period])
                .count(),
        );
    }
    ratio_usize(matches, comparisons)
}

fn ruliad_probe_examples(
    items: &[burn_dragon_universality::RuliadEvalItem],
    completions: &[burn_dragon_universality::RuliadCompletionRecord],
    limit: usize,
) -> Vec<CapabilityProbeExample> {
    if limit == 0 {
        return Vec::new();
    }
    let completion_by_hash = completions
        .iter()
        .map(|completion| {
            (
                completion.oracle_hash.as_str(),
                completion.completion.as_str(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut examples = Vec::with_capacity(limit.min(items.len()));
    for item in items {
        let completion = completion_by_hash.get(item.oracle_hash.as_str()).copied();
        let score =
            burn_dragon_universality::ruliad::score_ruliad_item_completion(item, completion);
        if score.verifier_match() {
            continue;
        }
        let extracted = completion.map(burn_dragon_universality::ruliad::extract_ruliad_completion);
        let actual = extracted
            .as_ref()
            .and_then(|completion| completion.answer.clone());
        examples.push(CapabilityProbeExample {
            label: format!("{}:{}", item.family, item.task_kind),
            prompt: compact_probe_example_text(&item.prompt, 512),
            expected: compact_probe_example_text(&item.expected_answer, 256),
            actual: actual.map(|answer| compact_probe_example_text(&answer, 256)),
            completion: compact_probe_example_text(completion.unwrap_or_default(), 512),
            status: format!("{:?}", score.status),
            reason: if completion.is_none() {
                "missing_completion".to_string()
            } else if extracted
                .as_ref()
                .is_none_or(|completion| completion.answer.is_none())
            {
                "malformed_completion".to_string()
            } else {
                "answer_mismatch".to_string()
            },
            generated_tokens: extracted
                .as_ref()
                .map(|completion| completion.generated_token_count)
                .unwrap_or_default(),
        });
        if examples.len() >= limit {
            break;
        }
    }
    examples
}

fn compact_probe_example_text(text: &str, max_chars: usize) -> String {
    let mut compact = text.replace('\r', "\\r").replace('\n', "\\n");
    let char_count = compact.chars().count();
    if char_count <= max_chars {
        return compact;
    }
    let keep = max_chars.saturating_sub(3);
    compact = compact.chars().take(keep).collect();
    compact.push_str("...");
    compact
}

fn emit_ruliad_correctness_metrics(
    run_name: &str,
    epoch: usize,
    absolute_step: usize,
    report: &burn_dragon_universality::RuliadEvalReport,
    bus: &TrainingEventBus,
) {
    emit_ruliad_correctness_metrics_with_labels(
        run_name,
        epoch,
        absolute_step,
        report,
        bus,
        "ruliad_correctness",
        None,
        None,
        &[],
        RuliadAnswerSchemaAlignmentSummary::default(),
        None,
    );
}

fn emit_ruliad_correctness_metrics_with_labels(
    run_name: &str,
    epoch: usize,
    absolute_step: usize,
    report: &burn_dragon_universality::RuliadEvalReport,
    bus: &TrainingEventBus,
    probe_name: &str,
    metric_prefix: Option<&str>,
    output_degeneracy: Option<&crate::train::steps::OutputDegeneracyStats>,
    examples: &[CapabilityProbeExample],
    schema_alignment: RuliadAnswerSchemaAlignmentSummary,
    completion_degeneracy: Option<RuliadCompletionDegeneracySummary>,
) {
    let item_count = report.item_count.max(1) as f64;
    let competence = ruliad_competence_key(report).unwrap_or_default();
    let metrics = [
        ("Ruliad Eval Items", report.item_count as f64),
        ("Ruliad Eval Scored Items", report.scored_count as f64),
        ("Ruliad Competence Score", ruliad_competence_score(report)),
        (
            "Ruliad Competence Verifier PPM",
            competence.verifier_ppm as f64,
        ),
        (
            "Ruliad Competence Semantic PPM",
            competence.semantic_ppm as f64,
        ),
        (
            "Ruliad Competence Partial PPM",
            competence.partial_ppm as f64,
        ),
        (
            "Ruliad Competence Certificate PPM",
            competence.certificate_ppm as f64,
        ),
        (
            "Ruliad Competence Completion Health PPM",
            competence.completion_health_ppm as f64,
        ),
        ("Ruliad Exact Accuracy", f64::from(report.exact_accuracy)),
        (
            "Ruliad Semantic Accuracy",
            f64::from(report.semantic_accuracy),
        ),
        (
            "Ruliad Verifier Accuracy",
            f64::from(report.verifier_accuracy),
        ),
        (
            "Ruliad Partial Credit Rate",
            f64::from(report.partial_credit_rate),
        ),
        (
            "Ruliad Schema Valid Wrong Rate",
            report.schema_valid_wrong_count as f64 / item_count,
        ),
        (
            "Ruliad Malformed Completion Rate",
            report.malformed_completion_count as f64 / item_count,
        ),
        (
            "Ruliad Missing Completion Rate",
            report.missing_completion_count as f64 / item_count,
        ),
        (
            "Ruliad Mean Partial Progress",
            f64::from(report.mean_partial_progress),
        ),
        (
            "Ruliad Answer Field Accuracy",
            f64::from(report.answer_field_accuracy),
        ),
        (
            "Ruliad Answer Termination Rate",
            f64::from(report.answer_termination_rate),
        ),
        (
            "Ruliad Certificate Prefix Coverage",
            f64::from(report.mean_certificate_prefix_coverage),
        ),
        (
            "Ruliad Mean Completion Tokens",
            f64::from(report.mean_completion_tokens),
        ),
        (
            "Ruliad Answer Key Match Rate",
            schema_alignment.key_match_rate,
        ),
        (
            "Ruliad Answer Key Overlap",
            schema_alignment.mean_key_overlap,
        ),
    ];
    for (name, value) in metrics {
        let metric_name = metric_prefix
            .map(|prefix| format!("{prefix} {name}"))
            .unwrap_or_else(|| name.to_string());
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: run_name.to_string(),
            split: TrainingMetricSplit::Valid,
            epoch,
            step_in_epoch: 0,
            absolute_step,
            name: metric_name,
            value,
            running_value: value,
        });
    }
    if let Some(degeneracy) = completion_degeneracy {
        for (name, value) in [
            (
                "Ruliad Completion Repetition Fraction",
                degeneracy.repetition_fraction,
            ),
            (
                "Ruliad Completion Distinct-1 Fraction",
                degeneracy.distinct_1_fraction,
            ),
            (
                "Ruliad Completion Distinct-2 Fraction",
                degeneracy.distinct_2_fraction,
            ),
            (
                "Ruliad Completion Max Period 2..16 Fraction",
                degeneracy.max_period_2_to_16_fraction,
            ),
            (
                "Ruliad Completion Max Period 2..64 Fraction",
                degeneracy.max_period_2_to_64_fraction,
            ),
            (
                "Ruliad Completion Dominant Period 2..64",
                degeneracy.dominant_period_2_to_64 as f64,
            ),
        ] {
            let metric_name = metric_prefix
                .map(|prefix| format!("{prefix} {name}"))
                .unwrap_or_else(|| name.to_string());
            let _ = bus.send_metric_sample(TrainingMetricSample {
                run_id: run_name.to_string(),
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
    let _ = bus.send_capability_probe_sample(ruliad_capability_probe_sample(
        run_name,
        epoch,
        absolute_step,
        report,
        competence,
        probe_name,
        output_degeneracy,
        examples,
        completion_degeneracy,
    ));
}

fn emit_ruliad_capability_gate_metrics(
    run_name: &str,
    epoch: usize,
    absolute_step: usize,
    report: &burn_dragon_universality::RuliadEvalReport,
    output_degeneracy: Option<&crate::train::steps::OutputDegeneracyStats>,
    gates: &burn_dragon_train::TrainingGatesConfig,
    bus: &TrainingEventBus,
) -> RuliadCapabilityGateStatus {
    let status = ruliad_capability_gate_status(report, output_degeneracy, gates);
    let (_, _, _, completion_health_rate) = ruliad_capability_rates(report);
    for (name, value) in [
        (
            "Ruliad Capability Gate Passed",
            if status.passed { 1.0 } else { 0.0 },
        ),
        (
            "Ruliad Capability Gate Failure Count",
            status.reasons.len() as f64,
        ),
        (
            "Ruliad Capability Completion Health Rate",
            completion_health_rate,
        ),
    ] {
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: run_name.to_string(),
            split: TrainingMetricSplit::Valid,
            epoch,
            step_in_epoch: 0,
            absolute_step,
            name: name.to_string(),
            value,
            running_value: value,
        });
    }
    if gates.enabled && !status.passed {
        let _ = bus.send_gate_event(TrainingGateEvent {
            run_id: run_name.to_string(),
            gate: "ruliad_capability_gate_failed".to_string(),
            action: TrainingGateAction::Alert,
            severity: TrainingGateSeverity::Warning,
            epoch: Some(epoch),
            absolute_step: Some(absolute_step),
            message: format!(
                "ruliad capability gate failed: {}",
                status.reasons.join(", ")
            ),
        });
    }
    status
}

fn ruliad_capability_probe_sample(
    run_name: &str,
    epoch: usize,
    absolute_step: usize,
    report: &burn_dragon_universality::RuliadEvalReport,
    competence: RuliadCompetenceKey,
    probe_name: &str,
    output_degeneracy: Option<&crate::train::steps::OutputDegeneracyStats>,
    examples: &[CapabilityProbeExample],
    completion_degeneracy: Option<RuliadCompletionDegeneracySummary>,
) -> CapabilityProbeSample {
    let item_count = report.item_count.max(1) as f64;
    let mut group_buckets = Vec::new();
    extend_ruliad_capability_groups(&mut group_buckets, "difficulty", &report.difficulty_scores);
    extend_ruliad_capability_groups(&mut group_buckets, "family", &report.family_scores);
    extend_ruliad_capability_groups(&mut group_buckets, "task", &report.task_scores);
    extend_ruliad_capability_groups(&mut group_buckets, "domain", &report.math_domain_scores);
    extend_ruliad_capability_groups(&mut group_buckets, "mode", &report.reasoning_mode_scores);

    CapabilityProbeSample {
        run_id: run_name.to_string(),
        split: TrainingMetricSplit::Valid,
        epoch,
        absolute_step,
        probe_name: probe_name.to_string(),
        item_count: report.item_count,
        scored_count: report.scored_count,
        exact_rate: f64::from(report.exact_accuracy),
        semantic_rate: f64::from(report.semantic_accuracy),
        verifier_rate: f64::from(report.verifier_accuracy),
        partial_credit_rate: f64::from(report.partial_credit_rate),
        schema_valid_wrong_rate: report.schema_valid_wrong_count as f64 / item_count,
        malformed_rate: report.malformed_completion_count as f64 / item_count,
        missing_rate: report.missing_completion_count as f64 / item_count,
        certificate_rate: f64::from(competence.certificate_ppm) / 1_000_000.0,
        completion_health_rate: f64::from(competence.completion_health_ppm) / 1_000_000.0,
        mean_partial_progress: f64::from(report.mean_partial_progress),
        answer_field_accuracy: f64::from(report.answer_field_accuracy),
        answer_termination_rate: f64::from(report.answer_termination_rate),
        mean_completion_tokens: f64::from(report.mean_completion_tokens),
        achieved_difficulty_level: ruliad_achieved_verifier_difficulty(report),
        output_entropy_bits: output_degeneracy.map(|stats| stats.entropy_bits),
        output_distinct_2_fraction: output_degeneracy.map(|stats| stats.distinct_2_fraction),
        completion_repetition_fraction: completion_degeneracy
            .map(|stats| stats.repetition_fraction),
        completion_distinct_1_fraction: completion_degeneracy
            .map(|stats| stats.distinct_1_fraction),
        completion_distinct_2_fraction: completion_degeneracy
            .map(|stats| stats.distinct_2_fraction),
        completion_max_period_2_to_16_fraction: completion_degeneracy
            .map(|stats| stats.max_period_2_to_16_fraction),
        completion_max_period_2_to_64_fraction: completion_degeneracy
            .map(|stats| stats.max_period_2_to_64_fraction),
        completion_dominant_period_2_to_64: completion_degeneracy
            .map(|stats| stats.dominant_period_2_to_64),
        group_buckets,
        examples: examples.to_vec(),
    }
}

fn extend_ruliad_capability_groups(
    output: &mut Vec<CapabilityProbeGroupMetric>,
    prefix: &str,
    groups: &[burn_dragon_universality::RuliadEvalGroupScore],
) {
    output.extend(groups.iter().map(|group| CapabilityProbeGroupMetric {
        label: format!("{prefix}:{}", group.label),
        item_count: group.count,
        exact_rate: f64::from(group.exact_accuracy),
        semantic_rate: f64::from(group.semantic_accuracy),
        verifier_rate: f64::from(group.verifier_accuracy),
        partial_credit_rate: f64::from(group.partial_credit_rate),
        schema_valid_wrong_rate: ratio_usize(group.schema_valid_wrong_count, group.count),
        malformed_rate: ratio_usize(group.malformed_completion_count, group.count),
        missing_rate: ratio_usize(group.missing_completion_count, group.count),
        mean_partial_progress: f64::from(group.mean_partial_progress),
        answer_field_accuracy: f64::from(group.answer_field_accuracy),
        answer_termination_rate: f64::from(group.answer_termination_rate),
    }));
}

fn ratio_usize(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn ruliad_achieved_verifier_difficulty(
    report: &burn_dragon_universality::RuliadEvalReport,
) -> Option<usize> {
    report
        .difficulty_scores
        .iter()
        .filter(|group| group.verifier_accuracy > 0.0)
        .filter_map(|group| group.label.strip_prefix('d')?.parse::<usize>().ok())
        .max()
}

fn run_source_weighted_validation_forward_only<B>(
    env: &ForwardEggrollTrainEnvironment<'_, B>,
    valid_model: &LanguageTrainModel<B>,
    epoch: usize,
    steps_per_epoch: usize,
    bus: &TrainingEventBus,
) -> Result<Option<f64>>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
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
        let Some(batch) = dataset.sample_source_weighted_validation_batch::<B>(
            epoch,
            absolute_step,
            env.training.batch_size,
            env.summary_event_token_ids.as_deref(),
            env.device,
        ) else {
            break;
        };
        let output = ValidStep::step(valid_model, batch);
        let loss_value: LossValue<B> = output.adapt();
        let loss = mean_scalar_from_loss(loss_value.value());
        count += 1;
        total += loss;
        let _ = bus.send_metric_sample(TrainingMetricSample {
            run_id: env.run_name.to_string(),
            split: TrainingMetricSplit::Valid,
            epoch,
            step_in_epoch: count,
            absolute_step,
            name: "Source Weighted Loss".to_string(),
            value: loss,
            running_value: total / count as f64,
        });
    }

    Ok((count > 0).then_some(total / count as f64))
}

fn dataset_eos_id(dataset: Option<&Arc<Dataset>>) -> Option<i64> {
    dataset
        .and_then(|dataset| dataset.tokenizer().eos_id())
        .map(i64::from)
}

fn source_selection_telemetry_due<B>(env: &TrainEnvironment<'_, B>, absolute_step: usize) -> bool
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

fn source_selection_telemetry_due_for(
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

fn train_loss_metric_name(training: &TrainingHyperparameters) -> &'static str {
    if training.tbptt_persist_across_steps {
        METRIC_STREAM_WARM_LOSS
    } else {
        METRIC_LOSS
    }
}

fn emit_source_selection_telemetry<B>(
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

fn emit_source_selection_telemetry_sample(
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
    let snapshot = recorded_snapshot.or_else(|| dataset.source_selection_snapshot());
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

fn emit_source_selection_capability_feedback_sample(
    run_name: &str,
    source_selection_dataset: Option<&Arc<Dataset>>,
    absolute_step: usize,
    report: &burn_dragon_universality::RuliadEvalReport,
    bus: &TrainingEventBus,
) {
    let Some(dataset) = source_selection_dataset else {
        return;
    };
    let Some(snapshot) = dataset.record_ruliad_capability_feedback(report) else {
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

fn emit_output_degeneracy<B>(
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

fn emit_output_degeneracy_sample(
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

fn emit_output_degeneracy_sample_with_prefix(
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
            run_id: run_name.to_string(),
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
            run_id: run_name.to_string(),
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

fn decode_degeneracy_preview(
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

fn decode_degeneracy_tokens(dataset: Option<&Arc<Dataset>>, tokens: &[i64]) -> String {
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

fn emit_continual_backprop_telemetry<B>(
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
        && absolute_step % env.training.events.continual_backprop_every_steps.max(1) != 0
    {
        return;
    }
    *last_emitted_optimizer_step = telemetry.optimizer_step;
    let _ = bus.send_continual_backprop_sample(ContinualBackpropSample {
        run_id: env.run_name.to_string(),
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

fn emit_predictive_coding_telemetry<B>(
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
    let _ = bus.send_predictive_coding_sample(PredictiveCodingSample {
        run_id: env.run_name.to_string(),
        epoch: Some(epoch),
        absolute_step,
        optimizer_step,
        chunks_seen: snapshot.chunks_seen,
        chunks_corrected: snapshot.chunks_corrected,
        inference_steps: snapshot.inference_steps,
        skipped_empty_state: snapshot.skipped_empty_state,
        energy_before: snapshot.energy_before_mean(),
        energy_after: snapshot.energy_after_mean(),
        energy_delta,
        grad_norm_mean,
        grad_norm_max,
        delta_rms_mean,
        elapsed_ms: snapshot.elapsed_ms(),
    });
    for (name, value) in [
        ("Predictive Coding Energy Delta", energy_delta),
        ("Predictive Coding Grad Norm Mean", grad_norm_mean),
        ("Predictive Coding Grad Norm Max", grad_norm_max),
        ("Predictive Coding Delta RMS", delta_rms_mean),
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
            run_id: env.run_name.to_string(),
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

fn emit_latent_reasoning_telemetry<B>(
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
            run_id: env.run_name.to_string(),
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DynamicsControlOutcome {
    Continue,
    Stop,
}

fn apply_pending_dynamics_control<B>(
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
    if event.run_id != env.run_name {
        return DynamicsControlOutcome::Continue;
    }
    apply_dynamics_control_event(env, &event, active, optimizer, model)
}

fn apply_dynamics_control_event<B>(
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

fn handle_post_validation_dynamics_control<B, S>(
    env: &TrainEnvironment<'_, B>,
    slot: &DragonDynamicsControlSlot,
    active: &mut ActiveDynamicsControl,
    optimizer: &mut crate::train::continual_backprop::LanguageOptimizer<B>,
    scheduler: &mut S,
    model: &mut LanguageTrainModel<B>,
    current_model_config: &mut DragonConfig,
    epoch: usize,
) -> Result<DynamicsControlOutcome>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    S: LrScheduler + Clone + 'static,
{
    let Some(event) = slot.take() else {
        return Ok(DynamicsControlOutcome::Continue);
    };
    if event.run_id != env.run_name {
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

fn apply_continual_learning_stability_policy<B>(
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
    let valid_loss = validation.loss;
    let policy = &env.training.dynamics;
    let mut recovery_requested = false;
    let ruliad_correctness_improved = validation_ruliad_correctness_improved(&validation, state);
    let improved = state.best_valid_loss.is_none_or(|best| {
        valid_loss < best * (1.0 - env.training.gates.plateau_min_relative_improvement)
    });
    if improved {
        state.best_valid_loss = Some(valid_loss);
        state.consecutive_validation_regressions = 0;
        state.consecutive_ruliad_correctness_regressions = 0;
        emit_dynamics_control(
            env,
            bus,
            policy,
            DynamicsMode::Stable,
            epoch,
            absolute_step,
            None,
            "validation improved; returning stability controls to baseline".to_string(),
        );
    } else if let Some(best) = state.best_valid_loss {
        if valid_loss > best * (1.0 + env.training.gates.validation_regression_max_relative) {
            if ruliad_correctness_improved {
                state.consecutive_validation_regressions = 0;
                emit_policy_gate_with_action(
                    env,
                    bus,
                    "continual_learning_validation_regression_suppressed_by_ruliad_progress",
                    TrainingGateAction::Alert,
                    TrainingGateSeverity::Info,
                    epoch,
                    absolute_step,
                    format!(
                        "teacher-forced validation worsened but ruliad correctness improved: best loss {:.6}, current {:.6}; suppressing rollback",
                        best, valid_loss
                    ),
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
                bus,
                "continual_learning_validation_regression",
                TrainingGateAction::Alert,
                TrainingGateSeverity::Warning,
                epoch,
                absolute_step,
                message.clone(),
            );
            emit_dynamics_control(
                env,
                bus,
                policy,
                mode,
                epoch,
                absolute_step,
                rollback_epoch,
                message,
            );
            recovery_requested = true;
        }
    }

    if let Some(report) = validation.ruliad_eval_report.as_ref() {
        update_capability_run_control_state(
            env,
            report,
            validation.output_degeneracy.as_ref(),
            epoch,
            absolute_step,
            state,
            bus,
            &mut recovery_requested,
        );
    }

    if let Some(report) = validation.ruliad_eval_report.as_ref()
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
        if verifier_improved {
            state.best_ruliad_verifier_accuracy = Some(verifier_accuracy);
        } else if state.best_ruliad_verifier_accuracy.is_none() {
            state.best_ruliad_verifier_accuracy = Some(verifier_accuracy);
        }
        if partial_improved {
            state.best_ruliad_partial_progress = Some(partial_progress);
        } else if state.best_ruliad_partial_progress.is_none() {
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
        if verifier_regressed || partial_regressed {
            state.consecutive_ruliad_correctness_regressions = state
                .consecutive_ruliad_correctness_regressions
                .saturating_add(1);
        } else if verifier_improved || partial_improved {
            state.consecutive_ruliad_correctness_regressions = 0;
        }
        if state.consecutive_ruliad_correctness_regressions >= 1 && !recovery_requested {
            let rollback_epoch = state.best_checkpoint_epoch;
            let mode = if rollback_epoch.is_some() {
                DynamicsMode::RollbackRecovery
            } else {
                DynamicsMode::ValidationRecovery
            };
            let message = format!(
                "ruliad correctness regression detected: verifier {:.3}->{:.3}, partial_progress {:.3}->{:.3}; requesting {:?}{}",
                verifier_best,
                verifier_accuracy,
                partial_best,
                partial_progress,
                mode,
                rollback_epoch
                    .map(|epoch| format!(" to checkpoint epoch {epoch}"))
                    .unwrap_or_default()
            );
            emit_policy_gate_with_action(
                env,
                bus,
                "continual_learning_ruliad_correctness_regression",
                TrainingGateAction::Alert,
                TrainingGateSeverity::Warning,
                epoch,
                absolute_step,
                message.clone(),
            );
            emit_dynamics_control(
                env,
                bus,
                policy,
                mode,
                epoch,
                absolute_step,
                rollback_epoch,
                message,
            );
            recovery_requested = true;
        }
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
        let rollback_epoch = state.best_checkpoint_epoch;
        let mode = if hard_collapse && rollback_epoch.is_some() {
            DynamicsMode::RollbackRecovery
        } else if hard_collapse {
            DynamicsMode::HardRecovery
        } else {
            DynamicsMode::PlasticityRecovery
        };
        emit_policy_gate_with_action(
            env,
            bus,
            "continual_learning_output_degeneracy",
            TrainingGateAction::Alert,
            TrainingGateSeverity::Warning,
            epoch,
            absolute_step,
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
        );
        if hard_collapse || !recovery_requested {
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
                bus,
                policy,
                mode,
                epoch,
                absolute_step,
                (mode == DynamicsMode::RollbackRecovery)
                    .then_some(rollback_epoch)
                    .flatten(),
                message,
            );
        }
    }
}

fn emit_dynamics_control<B>(
    env: &TrainEnvironment<'_, B>,
    bus: &TrainingEventBus,
    policy: &burn_dragon_train::train::events::DynamicsEquilibriumPolicy,
    mode: DynamicsMode,
    epoch: usize,
    absolute_step: usize,
    rollback_to_epoch: Option<usize>,
    reason: String,
) where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
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
        run_id: env.run_name.to_string(),
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

fn output_degeneracy_tripped(
    gates: &burn_dragon_train::TrainingGatesConfig,
    degeneracy: &crate::train::steps::OutputDegeneracyStats,
) -> bool {
    output_diversity_degeneracy(gates, degeneracy)
}

fn output_distribution_collapse(
    gates: &burn_dragon_train::TrainingGatesConfig,
    degeneracy: &crate::train::steps::OutputDegeneracyStats,
) -> bool {
    degeneracy.entropy_bits < gates.degeneracy_entropy_min_bits
        || degeneracy.mean_max_probability > gates.degeneracy_max_probability_max
}

fn output_diversity_degeneracy(
    gates: &burn_dragon_train::TrainingGatesConfig,
    degeneracy: &crate::train::steps::OutputDegeneracyStats,
) -> bool {
    degeneracy.argmax_unique_fraction < gates.degeneracy_argmax_unique_min_fraction
        || degeneracy.distinct_2_fraction < gates.degeneracy_distinct_2_min_fraction
        || degeneracy.eos_fraction > gates.degeneracy_eos_max_fraction
        || degeneracy.repetition_fraction > gates.degeneracy_repetition_max_fraction
        || periodic_structure_degeneracy(gates, degeneracy)
}

fn output_degeneracy_is_confident(
    _gates: &burn_dragon_train::TrainingGatesConfig,
    degeneracy: &crate::train::steps::OutputDegeneracyStats,
) -> bool {
    degeneracy.mean_max_probability >= 0.25
}

fn uncertain_argmax_loop(
    gates: &burn_dragon_train::TrainingGatesConfig,
    degeneracy: &crate::train::steps::OutputDegeneracyStats,
) -> bool {
    output_diversity_degeneracy(gates, degeneracy)
        && !output_distribution_collapse(gates, degeneracy)
        && !output_degeneracy_is_confident(gates, degeneracy)
}

fn hard_argmax_loop_collapse(
    gates: &burn_dragon_train::TrainingGatesConfig,
    degeneracy: &crate::train::steps::OutputDegeneracyStats,
) -> bool {
    degeneracy.repetition_fraction > gates.degeneracy_repetition_max_fraction
        || degeneracy.eos_fraction > gates.degeneracy_eos_max_fraction
        || (periodic_structure_high(gates, degeneracy)
            && output_distribution_collapse(gates, degeneracy)
            && (low_diversity_collapse(gates, degeneracy) || short_period_argmax_loop(degeneracy)))
}

fn low_diversity_collapse(
    gates: &burn_dragon_train::TrainingGatesConfig,
    degeneracy: &crate::train::steps::OutputDegeneracyStats,
) -> bool {
    degeneracy.argmax_unique_fraction < gates.degeneracy_argmax_unique_min_fraction
        || degeneracy.distinct_2_fraction < gates.degeneracy_distinct_2_min_fraction
}

fn periodic_structure_high(
    gates: &burn_dragon_train::TrainingGatesConfig,
    degeneracy: &crate::train::steps::OutputDegeneracyStats,
) -> bool {
    degeneracy.period_2_fraction > gates.degeneracy_period_2_max_fraction
        || degeneracy.period_3_fraction > gates.degeneracy_period_3_max_fraction
        || degeneracy.max_period_2_to_16_fraction > gates.degeneracy_period_2_to_16_max_fraction
        || degeneracy.max_period_2_to_64_fraction > gates.degeneracy_period_2_to_64_max_fraction
}

fn periodic_structure_degeneracy(
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

fn short_period_argmax_loop(degeneracy: &crate::train::steps::OutputDegeneracyStats) -> bool {
    (2..=8).contains(&degeneracy.dominant_period_2_to_64)
}

fn hard_output_collapse<B>(
    env: &TrainEnvironment<'_, B>,
    degeneracy: &crate::train::steps::OutputDegeneracyStats,
) -> bool
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    hard_output_collapse_for_gates(&env.training.gates, degeneracy)
}

fn hard_output_collapse_for_gates(
    gates: &burn_dragon_train::TrainingGatesConfig,
    degeneracy: &crate::train::steps::OutputDegeneracyStats,
) -> bool {
    (output_distribution_collapse(gates, degeneracy)
        && (low_diversity_collapse(gates, degeneracy)
            || hard_argmax_loop_collapse(gates, degeneracy)))
        || (output_degeneracy_is_confident(gates, degeneracy)
            && hard_argmax_loop_collapse(gates, degeneracy))
}

fn emit_policy_gate<B>(
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
        bus,
        gate,
        TrainingGateAction::Alert,
        TrainingGateSeverity::Warning,
        epoch,
        absolute_step,
        message,
    );
}

fn emit_policy_gate_with_action<B>(
    env: &TrainEnvironment<'_, B>,
    bus: &TrainingEventBus,
    gate: &str,
    action: TrainingGateAction,
    severity: TrainingGateSeverity,
    epoch: usize,
    absolute_step: usize,
    message: String,
) where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let _ = bus.send_gate_event(TrainingGateEvent {
        run_id: env.run_name.to_string(),
        gate: gate.to_string(),
        action,
        severity,
        epoch: Some(epoch),
        absolute_step: Some(absolute_step),
        message,
    });
}

fn mean_scalar_from_loss<B: BackendTrait>(tensor: Tensor<B, 1>) -> f64 {
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

fn historical_best_validation(run_dir: &Path, max_epoch: usize) -> HistoricalBestValidation {
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

fn save_dragon_model_checkpoint<B>(
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

fn save_dragon_training_state_checkpoint<B, S>(
    run_dir: &Path,
    epoch: usize,
    model: &LanguageTrainModel<B>,
    model_config: &DragonConfig,
    optimizer: &crate::train::continual_backprop::LanguageOptimizer<B>,
    scheduler: &S,
    dynamics_control: &ActiveDynamicsControl,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    S: LrScheduler + Clone + 'static,
{
    save_dragon_model_checkpoint(run_dir, epoch, &model.model)?;
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

fn source_selection_state_checkpoint_path(run_dir: &Path, epoch: usize) -> PathBuf {
    run_dir
        .join("checkpoint")
        .join(format!("source-selection-state-{epoch}.json"))
}

fn save_source_selection_state_checkpoint(
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

fn checkpoint_artifact_epoch(name: &str) -> Option<usize> {
    for (prefix, suffix) in [
        ("model-", ".bin"),
        ("optimizer-", ".bin"),
        ("scheduler-", ".bin"),
        ("dynamics-", ".json"),
        ("model-config-", ".json"),
        ("source-selection-state-", ".json"),
    ] {
        if let Some(epoch) = name
            .strip_prefix(prefix)
            .and_then(|value| value.strip_suffix(suffix))
            .and_then(|value| value.parse::<usize>().ok())
        {
            return Some(epoch);
        }
    }
    None
}

fn prune_dragon_model_checkpoints(
    run_dir: &Path,
    current_epoch: usize,
    best_epoch: Option<usize>,
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
    if let Some(best_epoch) = best_epoch {
        keep_epochs.insert(best_epoch);
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

fn load_dragon_model_checkpoint<B>(
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

fn load_dragon_training_state_checkpoint<B, S>(
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

fn apply_dynamic_neuron_scale<B>(
    env: &TrainEnvironment<'_, B>,
    model: &mut LanguageTrainModel<B>,
    optimizer: &mut crate::train::continual_backprop::LanguageOptimizer<B>,
    current_model_config: &mut DragonConfig,
    scale_generation: &mut usize,
    request: ModelScaleRequest,
    epoch: usize,
    absolute_step: usize,
    bus: &TrainingEventBus,
    batch_size: usize,
    gradient_accumulation_steps: usize,
) -> Result<Option<(usize, usize)>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let current_latent_total = model.model.latent_total_capacity();
    let skip = |reason: String, bus: &TrainingEventBus| {
        let _ = bus.send_model_scale_skipped(ModelScaleSkipped {
            run_id: env.run_name.to_string(),
            epoch: Some(epoch),
            absolute_step: Some(absolute_step),
            from_capacity_units: current_latent_total,
            requested_capacity_units: Some(request.to_capacity_units),
            reason,
        });
    };

    if request.run_id != env.run_name {
        skip(
            format!(
                "scale request run_id {} does not match active run {}",
                request.run_id, env.run_name
            ),
            bus,
        );
        return Ok(None);
    }
    if request.from_capacity_units != current_latent_total {
        skip(
            format!(
                "scale request source capacity {} does not match active latent_total {}",
                request.from_capacity_units, current_latent_total
            ),
            bus,
        );
        return Ok(None);
    }
    if request.to_capacity_units <= current_latent_total {
        skip(
            format!(
                "scale request target {} must exceed current latent_total {}",
                request.to_capacity_units, current_latent_total
            ),
            bus,
        );
        return Ok(None);
    }
    if request.to_capacity_units > env.training.neuron_scaling.max_latent_total {
        skip(
            format!(
                "scale request target {} exceeds configured max_latent_total {}",
                request.to_capacity_units, env.training.neuron_scaling.max_latent_total
            ),
            bus,
        );
        return Ok(None);
    }
    if request.to_capacity_units % current_model_config.n_embd != 0 {
        skip(
            format!(
                "scale request target {} is not divisible by n_embd {}",
                request.to_capacity_units, current_model_config.n_embd
            ),
            bus,
        );
        return Ok(None);
    }
    if request.to_capacity_units % current_model_config.n_head != 0 {
        skip(
            format!(
                "scale request target {} is not divisible by n_head {}",
                request.to_capacity_units, current_model_config.n_head
            ),
            bus,
        );
        return Ok(None);
    }
    if request.to_capacity_units % env.parallel_config.tensor.size != 0 {
        skip(
            format!(
                "scale request target {} is not divisible by tensor parallel size {}",
                request.to_capacity_units, env.parallel_config.tensor.size
            ),
            bus,
        );
        return Ok(None);
    }

    let mut target_config = current_model_config.clone();
    target_config.mlp_internal_dim_multiplier = request.to_capacity_units / target_config.n_embd;
    let (widened, report) = model
        .model
        .widen_latent_total(target_config.clone(), env.device)
        .map_err(|message| anyhow!("failed to widen Dragon latent_total in-process: {message}"))?;
    model.model = widened;
    *model = model
        .clone()
        .with_continual_backprop(&env.training.continual_backprop)
        .with_gradient_scale_schedule(
            env.training,
            env.epochs
                .saturating_mul(env.train_loader.num_items().max(1)),
        )
        .with_neuron_scale_stabilization(
            report.old_latent_total,
            report.new_latent_total,
            &env.training.neuron_scaling.stabilization,
        );
    optimizer.prepare_after_neuron_scale(model);
    optimizer.refresh_continual_backprop_fresh_model(DragonModel::<B>::new(
        target_config.clone(),
        env.device,
    ));
    *current_model_config = target_config;
    *scale_generation = scale_generation.saturating_add(1);

    let _ = bus.send_model_scale_applied(ModelScaleApplied {
        run_id: env.run_name.to_string(),
        epoch: Some(epoch),
        absolute_step: Some(absolute_step),
        from_capacity_units: report.old_latent_total,
        to_capacity_units: report.new_latent_total,
        scale_generation: *scale_generation,
        batch_size: Some(batch_size),
        gradient_accumulation_steps: Some(gradient_accumulation_steps),
        message: format!(
            "applied Dragon neuron scaling {} -> {} in-process; optimizer_state_policy=drop_widened_param_moments; stabilization_freeze_base_steps={} stabilization_unfreeze_ramp_steps={}",
            report.old_latent_total,
            report.new_latent_total,
            env.training.neuron_scaling.stabilization.freeze_base_steps,
            env.training.neuron_scaling.stabilization.unfreeze_ramp_steps,
        ),
    });
    info!(
        "applied in-process Dragon neuron scaling {} -> {} at epoch={} absolute_step={}",
        report.old_latent_total, report.new_latent_total, epoch, absolute_step
    );
    Ok(Some((report.old_latent_total, report.new_latent_total)))
}

#[cfg(feature = "ddp")]
struct CollectiveSessionGuard<B: BackendTrait> {
    peer_id: PeerId,
    _marker: PhantomData<B>,
}

#[cfg(feature = "ddp")]
impl<B: BackendTrait> CollectiveSessionGuard<B> {
    fn register(
        peer_id: PeerId,
        device: B::Device,
        config: burn_collective::CollectiveConfig,
    ) -> Result<Self> {
        info!("registering DDP collective session for peer_id={peer_id}");
        register::<B>(peer_id, device, config)
            .map_err(|err| anyhow!("failed to register DDP collective session: {err:?}"))?;
        info!("registered DDP collective session for peer_id={peer_id}");
        Ok(Self {
            peer_id,
            _marker: PhantomData,
        })
    }
}

#[cfg(feature = "ddp")]
impl<B: BackendTrait> Drop for CollectiveSessionGuard<B> {
    fn drop(&mut self) {
        let _ = finish_collective::<B>(self.peer_id);
    }
}

#[cfg(feature = "ddp")]
fn shard_bounds(
    total_items: usize,
    shard_index: usize,
    shard_count: usize,
) -> Result<(usize, usize)> {
    if shard_count == 0 {
        return Err(anyhow!("cannot shard a dataloader across zero ranks"));
    }
    if shard_index >= shard_count {
        return Err(anyhow!(
            "rank-local dataloader shard {shard_index} is out of range for shard_count={shard_count}"
        ));
    }
    if total_items < shard_count {
        return Err(anyhow!(
            "rank-local dataloader sharding requires at least one step per rank (steps={total_items}, world_size={shard_count})"
        ));
    }

    let base = total_items / shard_count;
    let remainder = total_items % shard_count;
    let start = shard_index * base + shard_index.min(remainder);
    let width = base + usize::from(shard_index < remainder);
    Ok((start, start + width))
}

#[cfg(feature = "ddp")]
fn shard_dataloader<B, I>(
    loader: Arc<dyn DataLoader<B, I>>,
    shard_index: usize,
    shard_count: usize,
    label: &str,
) -> Result<Arc<dyn DataLoader<B, I>>>
where
    B: BackendTrait + 'static,
    I: 'static,
{
    if shard_count <= 1 {
        return Ok(loader);
    }

    let total_items = loader.num_items();
    let (start, end) = shard_bounds(total_items, shard_index, shard_count)
        .with_context(|| format!("failed to shard {label} dataloader"))?;
    Ok(loader.slice(start, end))
}

#[cfg(feature = "ddp")]
fn mean_scalar_from_tensor<B: BackendTrait>(tensor: Tensor<B, 1>) -> f64 {
    tensor
        .mean()
        .into_data()
        .iter::<f64>()
        .next()
        .unwrap_or(0.0)
}

#[cfg(feature = "ddp")]
fn reduce_mean_scalar<B: BackendTrait>(peer_id: PeerId, tensor: Tensor<B, 1>) -> Result<f64> {
    let reduced = all_reduce::<B>(
        peer_id,
        tensor.into_primitive().tensor(),
        ReduceOperation::Mean,
    )
    .map_err(|err| anyhow!("failed to all-reduce scalar metric: {err:?}"))?;
    Ok(mean_scalar_from_tensor(Tensor::<B, 1>::from_primitive(
        TensorPrimitive::Float(reduced),
    )))
}

#[cfg(feature = "ddp")]
fn process_group_peer_id(runtime: &ParallelRuntime) -> PeerId {
    runtime.global_rank.into()
}

#[cfg(feature = "ddp")]
fn process_group_data_shard(
    runtime: &ParallelRuntime,
    config: &ParallelConfig,
) -> Result<(
    usize,
    usize,
    Option<PipelineRankAssignment>,
    Option<PipelineParallelLayout>,
)> {
    let layout = resolve_pipeline_parallel_layout(runtime, config)?;
    if let Some(layout) = layout {
        let assignment = layout.assignment(runtime.global_rank).clone();
        return Ok((
            assignment.data_parallel_rank,
            layout.data_parallel_size,
            Some(assignment),
            Some(layout),
        ));
    }

    Ok((runtime.global_rank, runtime.world_size, None, None))
}

#[cfg(feature = "ddp")]
fn all_reduce_gradients_in_module_order<B, M>(
    module: &M,
    grads: &mut GradientsParams,
    peer_id: PeerId,
    op: ReduceOperation,
) -> Result<()>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    struct GradientAllReduceVisitor<'a, B: AutodiffBackend> {
        grads: &'a mut GradientsParams,
        peer_id: PeerId,
        op: ReduceOperation,
        trace_grads: bool,
        index: usize,
        error: Option<anyhow::Error>,
        _marker: PhantomData<B>,
    }

    impl<B: AutodiffBackend> burn::module::ModuleVisitor<B> for GradientAllReduceVisitor<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            if self.error.is_some() {
                return;
            }

            self.index += 1;
            let grad_index = self.index;

            let grad = match self.grads.remove::<B::InnerBackend, D>(param.id) {
                Some(grad) => grad,
                None => {
                    if self.trace_grads && grad_index <= 12 {
                        info!(
                            "process-group DDP peer_id={} gradient[{grad_index}] missing, zero-filling shape={:?}",
                            self.peer_id,
                            param.val().shape().dims::<D>()
                        );
                    }
                    param.val().inner().zeros_like()
                }
            };

            if self.trace_grads && grad_index <= 12 {
                info!(
                    "process-group DDP peer_id={} gradient[{grad_index}] entering all-reduce shape={:?}",
                    self.peer_id,
                    grad.shape().dims::<D>()
                );
            }

            match all_reduce::<B::InnerBackend>(
                self.peer_id,
                grad.into_primitive().tensor(),
                self.op,
            ) {
                Ok(reduced) => {
                    if self.trace_grads && grad_index <= 12 {
                        info!(
                            "process-group DDP peer_id={} gradient[{grad_index}] completed all-reduce",
                            self.peer_id
                        );
                    }
                    self.grads.register(
                        param.id,
                        Tensor::<B::InnerBackend, D>::from_primitive(TensorPrimitive::Float(
                            reduced,
                        )),
                    )
                }
                Err(err) => {
                    self.error = Some(anyhow!(
                        "failed to all-reduce process-group DDP gradients: {err:?}"
                    ));
                }
            }
        }
    }

    let trace_grads = true;
    let mut visitor = GradientAllReduceVisitor::<B> {
        grads,
        peer_id,
        op,
        trace_grads,
        index: 0,
        error: None,
        _marker: PhantomData,
    };
    module.visit(&mut visitor);

    if let Some(err) = visitor.error {
        return Err(err);
    }

    Ok(())
}

#[cfg(feature = "ddp")]
fn scale_gradients_in_module_order<B, M>(module: &M, grads: &mut GradientsParams, scale: f32)
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    if (scale - 1.0).abs() <= f32::EPSILON {
        return;
    }

    struct GradientScaleVisitor<'a, B: AutodiffBackend> {
        grads: &'a mut GradientsParams,
        scale: f32,
        _marker: PhantomData<B>,
    }

    impl<B: AutodiffBackend> burn::module::ModuleVisitor<B> for GradientScaleVisitor<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            if let Some(grad) = self.grads.remove::<B::InnerBackend, D>(param.id) {
                self.grads.register(param.id, grad.mul_scalar(self.scale));
            }
        }
    }

    let mut visitor = GradientScaleVisitor::<B> {
        grads,
        scale,
        _marker: PhantomData,
    };
    module.visit(&mut visitor);
}

#[cfg(feature = "ddp")]
fn reduce_sum_scalar<B: BackendTrait>(peer_id: PeerId, tensor: Tensor<B, 1>) -> Result<f64> {
    let reduced = all_reduce::<B>(
        peer_id,
        tensor.into_primitive().tensor(),
        ReduceOperation::Sum,
    )
    .map_err(|err| anyhow!("failed to all-reduce scalar sum: {err:?}"))?;
    Ok(mean_scalar_from_tensor(Tensor::<B, 1>::from_primitive(
        TensorPrimitive::Float(reduced),
    )))
}

#[cfg(feature = "ddp")]
fn scalar_tensor<B: BackendTrait>(device: &B::Device, value: f32) -> Tensor<B, 1> {
    Tensor::<B, 1>::from_floats([value], device)
}

#[cfg(feature = "ddp")]
fn scalar_flag<B: BackendTrait>(device: &B::Device, enabled: bool) -> Tensor<B, 1> {
    scalar_tensor::<B>(device, if enabled { 1.0 } else { 0.0 })
}

#[cfg(feature = "ddp")]
fn broadcast_float_tensor_rooted<B: BackendTrait, const D: usize>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    tensor: Option<Tensor<B, D>>,
) -> Result<Tensor<B, D>> {
    let root_tensor = if global_rank == root_rank {
        Some(
            tensor
                .ok_or_else(|| anyhow!("collective root rank {root_rank} is missing a tensor"))?
                .into_primitive()
                .tensor(),
        )
    } else {
        None
    };
    let broadcasted = broadcast::<B>(peer_id, root_tensor).map_err(|err| {
        anyhow!("failed to broadcast rooted tensor from rank {root_rank}: {err:?}")
    })?;
    Ok(Tensor::<B, D>::from_primitive(TensorPrimitive::Float(
        broadcasted,
    )))
}

#[cfg(feature = "ddp")]
fn broadcast_usize_rooted<B: BackendTrait>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    device: &B::Device,
    value: Option<usize>,
) -> Result<usize> {
    let tensor = broadcast_float_tensor_rooted::<B, 1>(
        peer_id,
        global_rank,
        root_rank,
        value.map(|value| scalar_tensor::<B>(device, value as f32)),
    )?;
    Ok(mean_scalar_from_tensor(tensor).round().max(0.0) as usize)
}

#[cfg(feature = "ddp")]
fn broadcast_bool_rooted<B: BackendTrait>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    device: &B::Device,
    value: Option<bool>,
) -> Result<bool> {
    let tensor = broadcast_float_tensor_rooted::<B, 1>(
        peer_id,
        global_rank,
        root_rank,
        value.map(|value| scalar_flag::<B>(device, value)),
    )?;
    Ok(mean_scalar_from_tensor(tensor) >= 0.5)
}

#[cfg(feature = "ddp")]
fn broadcast_int_tensor_rooted<B: AutodiffBackend, const D: usize>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    tensor: Option<Tensor<B, D, Int>>,
) -> Result<Tensor<B, D, Int>> {
    let broadcasted = broadcast_float_tensor_rooted::<B::InnerBackend, D>(
        peer_id,
        global_rank,
        root_rank,
        tensor.map(|tensor| tensor.float().inner()),
    )?;
    Ok(Tensor::<B, D>::from_inner(broadcasted).int())
}

#[cfg(feature = "ddp")]
fn broadcast_optional_int_tensor_rooted<B: AutodiffBackend, const D: usize>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    device: &B::Device,
    tensor: Option<Tensor<B, D, Int>>,
) -> Result<Option<Tensor<B, D, Int>>> {
    let has_tensor = broadcast_bool_rooted::<B::InnerBackend>(
        peer_id,
        global_rank,
        root_rank,
        device,
        Some(tensor.is_some()),
    )?;
    if !has_tensor {
        return Ok(None);
    }
    broadcast_int_tensor_rooted(peer_id, global_rank, root_rank, tensor).map(Some)
}

#[cfg(feature = "ddp")]
fn broadcast_sequence_batch_rooted<B: AutodiffBackend>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    device: &B::Device,
    batch: Option<SequenceBatch<B>>,
) -> Result<SequenceBatch<B>> {
    let inputs = broadcast_int_tensor_rooted(
        peer_id,
        global_rank,
        root_rank,
        batch.as_ref().map(|batch| batch.inputs.clone()),
    )?;
    let targets = broadcast_int_tensor_rooted(
        peer_id,
        global_rank,
        root_rank,
        batch.as_ref().map(|batch| batch.targets.clone()),
    )?;
    let summary_event_mask = broadcast_optional_int_tensor_rooted(
        peer_id,
        global_rank,
        root_rank,
        device,
        batch
            .as_ref()
            .and_then(|batch| batch.summary_event_mask.clone()),
    )?;
    let loss_mask = broadcast_optional_int_tensor_rooted(
        peer_id,
        global_rank,
        root_rank,
        device,
        batch.as_ref().and_then(|batch| batch.loss_mask.clone()),
    )?;
    let reset_stream_state = broadcast_bool_rooted::<B::InnerBackend>(
        peer_id,
        global_rank,
        root_rank,
        device,
        Some(batch.as_ref().is_some_and(|batch| batch.reset_stream_state)),
    )?;

    Ok(SequenceBatch {
        inputs,
        targets,
        loss_mask,
        summary_event_mask,
        ruliad_policy_batch: None,
        reset_stream_state,
    })
}

#[cfg(feature = "ddp")]
fn detach_pipeline_state_to_inner<B: AutodiffBackend>(
    state: &LanguagePipelineState<B>,
) -> LanguagePipelineState<B::InnerBackend> {
    LanguagePipelineState::from_parts(
        state.current().clone().detach().inner(),
        state
            .residual_history()
            .iter()
            .cloned()
            .map(|tensor| tensor.detach().inner())
            .collect(),
    )
}

#[cfg(feature = "ddp")]
fn attach_pipeline_state_require_grad<B: AutodiffBackend>(
    state: LanguagePipelineState<B::InnerBackend>,
) -> LanguagePipelineState<B> {
    let (current, residual_history) = state.into_parts();
    LanguagePipelineState::from_parts(
        Tensor::<B, 4>::from_inner(current).require_grad(),
        residual_history
            .into_iter()
            .map(|tensor| Tensor::<B, 4>::from_inner(tensor).require_grad())
            .collect(),
    )
}

#[cfg(feature = "ddp")]
fn broadcast_pipeline_state_rooted<B: AutodiffBackend>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    device: &B::Device,
    state: Option<&LanguagePipelineState<B>>,
) -> Result<LanguagePipelineState<B::InnerBackend>> {
    let history_len = broadcast_usize_rooted::<B::InnerBackend>(
        peer_id,
        global_rank,
        root_rank,
        device,
        state.map(|state| state.residual_history().len()),
    )?;
    let current = broadcast_float_tensor_rooted::<B::InnerBackend, 4>(
        peer_id,
        global_rank,
        root_rank,
        state.map(|state| state.current().clone().detach().inner()),
    )?;
    let residual_history = (0..history_len)
        .map(|index| {
            broadcast_float_tensor_rooted::<B::InnerBackend, 4>(
                peer_id,
                global_rank,
                root_rank,
                state.map(|state| state.residual_history()[index].clone().detach().inner()),
            )
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(LanguagePipelineState::from_parts(current, residual_history))
}

#[cfg(feature = "ddp")]
fn broadcast_pipeline_state_inner_rooted<B: BackendTrait>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    device: &B::Device,
    state: Option<&LanguagePipelineState<B>>,
) -> Result<LanguagePipelineState<B>> {
    let history_len = broadcast_usize_rooted::<B>(
        peer_id,
        global_rank,
        root_rank,
        device,
        state.map(|state| state.residual_history().len()),
    )?;
    let current = broadcast_float_tensor_rooted::<B, 4>(
        peer_id,
        global_rank,
        root_rank,
        state.map(|state| state.current().clone()),
    )?;
    let residual_history = (0..history_len)
        .map(|index| {
            broadcast_float_tensor_rooted::<B, 4>(
                peer_id,
                global_rank,
                root_rank,
                state.map(|state| state.residual_history()[index].clone()),
            )
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(LanguagePipelineState::from_parts(current, residual_history))
}

#[cfg(feature = "ddp")]
fn pipeline_surrogate_loss<B: AutodiffBackend>(
    output_state: &LanguagePipelineState<B>,
    grad_state: LanguagePipelineState<B::InnerBackend>,
) -> Tensor<B, 1> {
    let (grad_current, grad_history) = grad_state.into_parts();
    assert_eq!(
        output_state.residual_history().len(),
        grad_history.len(),
        "pipeline residual history length mismatch"
    );

    let mut surrogate = output_state
        .current()
        .clone()
        .mul(Tensor::<B, 4>::from_inner(grad_current))
        .sum();
    for (residual, grad) in output_state
        .residual_history()
        .iter()
        .zip(grad_history.into_iter())
    {
        surrogate = surrogate + residual.clone().mul(Tensor::<B, 4>::from_inner(grad)).sum();
    }
    surrogate
}

#[cfg(feature = "ddp")]
fn pipeline_input_grad_state<B: AutodiffBackend>(
    input_state: &LanguagePipelineState<B>,
    grads: &mut B::Gradients,
) -> LanguagePipelineState<B::InnerBackend> {
    LanguagePipelineState::from_parts(
        input_state
            .current()
            .grad_remove(grads)
            .unwrap_or_else(|| input_state.current().clone().inner().zeros_like()),
        input_state
            .residual_history()
            .iter()
            .map(|tensor| {
                tensor
                    .grad_remove(grads)
                    .unwrap_or_else(|| tensor.clone().inner().zeros_like())
            })
            .collect(),
    )
}

#[cfg(feature = "ddp")]
fn slice_batch_int<B: BackendTrait>(
    tensor: Tensor<B, 2, Int>,
    range: std::ops::Range<usize>,
) -> Tensor<B, 2, Int> {
    let [_batch, block_size] = tensor.shape().dims();
    tensor.slice([range.start..range.end, 0..block_size])
}

#[cfg(feature = "ddp")]
fn pipeline_replica_root_rank(layout: &PipelineParallelLayout, data_parallel_rank: usize) -> usize {
    data_parallel_rank * layout.stage_count
}

#[cfg(feature = "ddp")]
fn global_rank_for_virtual_stage(
    plan: &PipelinePlan,
    layout: &PipelineParallelLayout,
    data_parallel_rank: usize,
    virtual_stage_id: usize,
) -> usize {
    let physical_stage_id = plan.assignment(virtual_stage_id).physical_stage_id;
    data_parallel_rank * layout.stage_count + physical_stage_id
}

#[cfg(feature = "ddp")]
struct DistributedPipelineForwardCache<B: AutodiffBackend> {
    input_state: Option<LanguagePipelineState<B>>,
    output_state: Option<LanguagePipelineState<B>>,
    loss: Option<Tensor<B, 1>>,
}

#[cfg(feature = "ddp")]
fn save_process_group_checkpoint<B, O, S>(
    run_dir: &Path,
    epoch: usize,
    learner: &burn_train::Learner<
        burn_train::LearningComponentsMarker<B, S, LanguageTrainModel<B>, O>,
    >,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    O: Optimizer<LanguageTrainModel<B>, B> + 'static,
    S: LrScheduler + 'static,
{
    let checkpoint_dir = run_dir.join("checkpoint");
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    FileCheckpointer::new(recorder, &checkpoint_dir, "model")
        .save(epoch, learner.model().model.into_record())
        .with_context(|| {
            format!(
                "failed to save process-group model checkpoint {epoch} in {}",
                checkpoint_dir.display()
            )
        })?;
    Ok(())
}

#[cfg(feature = "ddp")]
fn load_process_group_checkpoint<B, O, S>(
    run_dir: &Path,
    epoch: usize,
    device: &B::Device,
    mut learner: burn_train::Learner<
        burn_train::LearningComponentsMarker<B, S, LanguageTrainModel<B>, O>,
    >,
) -> Result<burn_train::Learner<burn_train::LearningComponentsMarker<B, S, LanguageTrainModel<B>, O>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    O: Optimizer<LanguageTrainModel<B>, B> + 'static,
    S: LrScheduler + 'static,
{
    let checkpoint_dir = run_dir.join("checkpoint");
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let model_record = FileCheckpointer::new(recorder.clone(), &checkpoint_dir, "model")
        .restore(epoch, device)
        .with_context(|| {
            format!(
                "failed to restore process-group model checkpoint {epoch} from {}",
                checkpoint_dir.display()
            )
        })?;
    learner.load_model(model_record);

    let optim_path = checkpoint_dir.join(format!("optim-{epoch}.bin"));
    if optim_path.is_file() {
        let optim_record = FileCheckpointer::new(recorder.clone(), &checkpoint_dir, "optim")
            .restore(epoch, device)
            .with_context(|| {
                format!(
                    "failed to restore process-group optimizer checkpoint {epoch} from {}",
                    checkpoint_dir.display()
                )
            })?;
        learner.load_optim(optim_record);
    }

    let scheduler_path = checkpoint_dir.join(format!("scheduler-{epoch}.bin"));
    if scheduler_path.is_file() {
        let scheduler_record = FileCheckpointer::new(recorder, &checkpoint_dir, "scheduler")
            .restore(epoch, device)
            .with_context(|| {
                format!(
                    "failed to restore process-group scheduler checkpoint {epoch} from {}",
                    checkpoint_dir.display()
                )
            })?;
        learner.load_scheduler(scheduler_record);
    }

    Ok(learner)
}

#[cfg(feature = "ddp")]
fn run_process_group_validation<B, O, S>(
    env: &TrainEnvironment<'_, B>,
    learner: &burn_train::Learner<
        burn_train::LearningComponentsMarker<B, S, LanguageTrainModel<B>, O>,
    >,
) -> Option<f64>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    O: Optimizer<LanguageTrainModel<B>, B> + 'static,
    S: LrScheduler + 'static,
{
    if !env.parallel_runtime.is_primary() {
        return None;
    }

    let model = learner.model().valid();
    let mut iterator = env.valid_loader.iter();
    let mut total = 0.0;
    let mut count = 0usize;

    while let Some(item) = iterator.next() {
        let output = model.step(item);
        let loss_value: LossValue<ValidBackend<B>> = output.adapt();
        total += mean_scalar_from_tensor(loss_value.value());
        count += 1;
    }

    (count > 0).then_some(total / count as f64)
}

#[cfg(feature = "ddp")]
struct DistributedPipelineTrainStepResult {
    grads: GradientsParams,
    mean_train_loss: Option<f64>,
}

#[cfg(feature = "ddp")]
fn distributed_pipeline_train_step<B>(
    peer_id: PeerId,
    model: &LanguageTrainModel<B>,
    batch: SequenceBatch<B>,
    layout: &PipelineParallelLayout,
    assignment: &PipelineRankAssignment,
    device: &B::Device,
    reduce_metric_loss: bool,
) -> Result<DistributedPipelineTrainStepResult>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let plan = model
        .pipeline_plan
        .as_ref()
        .ok_or_else(|| anyhow!("distributed pipeline step requires a pipeline plan"))?;
    let [batch_size, _block_size] = batch.inputs.shape().dims();
    let ranges = split_microbatch_ranges(batch_size, plan.microbatches)?;
    let chunk_inputs = ranges
        .iter()
        .cloned()
        .map(|range| slice_batch_int(batch.inputs.clone(), range))
        .collect::<Vec<_>>();
    let chunk_targets = ranges
        .iter()
        .cloned()
        .map(|range| slice_batch_int(batch.targets.clone(), range))
        .collect::<Vec<_>>();
    let chunk_masks = ranges
        .iter()
        .cloned()
        .map(|range| {
            batch
                .summary_event_mask
                .clone()
                .map(|mask| slice_batch_int(mask, range))
        })
        .collect::<Vec<_>>();
    let mut chunk_states = (0..plan.microbatches)
        .map(|_| model.model.init_state())
        .collect::<Vec<ModelState<B>>>();
    let mut forward_cache = HashMap::<(usize, usize), DistributedPipelineForwardCache<B>>::new();
    let mut incoming_forward =
        HashMap::<(usize, usize), LanguagePipelineState<B::InnerBackend>>::new();
    let mut incoming_backward =
        HashMap::<(usize, usize), LanguagePipelineState<B::InnerBackend>>::new();
    let mut local_accumulator = GradientsAccumulator::new();
    let mut local_loss: Option<Tensor<B::InnerBackend, 1>> = None;
    let last_virtual_stage_id = plan.total_virtual_stages.saturating_sub(1);

    for event in &plan.events {
        let microbatch_id = event.microbatch_id;
        let local_forward_output = if event.kind
            == burn_dragon_train::train::pipeline::PipelineEventKind::Forward
            && event.physical_stage_id == assignment.pipeline_stage_id
        {
            let input_state = if event.virtual_stage_id == 0 {
                model
                    .model
                    .begin_language_pipeline(chunk_inputs[microbatch_id].clone())
            } else {
                let input_state = incoming_forward
                    .remove(&(event.virtual_stage_id, microbatch_id))
                    .ok_or_else(|| {
                        anyhow!(
                            "missing forward pipeline state for virtual_stage={} microbatch={microbatch_id}",
                            event.virtual_stage_id
                        )
                    })?;
                attach_pipeline_state_require_grad::<B>(input_state)
            };
            let cached_input = (event.virtual_stage_id > 0).then_some(input_state.clone());
            let output_state = model.model.forward_language_pipeline_stage_with_state(
                input_state,
                &mut chunk_states[microbatch_id],
                plan.assignment(event.virtual_stage_id).layer_range.clone(),
                chunk_masks[microbatch_id].clone(),
            );

            if event.virtual_stage_id == last_virtual_stage_id {
                let hidden = model.model.finish_language_pipeline_hidden_with_state(
                    output_state,
                    &mut chunk_states[microbatch_id],
                );
                let weight = ranges[microbatch_id].len() as f32 / batch_size as f32;
                let loss = model
                    .model
                    .language_loss_from_hidden(hidden, chunk_targets[microbatch_id].clone())
                    .mul_scalar(weight);
                if reduce_metric_loss {
                    local_loss = Some(match local_loss {
                        Some(accumulated) => accumulated + loss.clone().detach().inner(),
                        None => loss.clone().detach().inner(),
                    });
                }
                forward_cache.insert(
                    (event.virtual_stage_id, microbatch_id),
                    DistributedPipelineForwardCache {
                        input_state: cached_input,
                        output_state: None,
                        loss: Some(loss),
                    },
                );
                None
            } else {
                forward_cache.insert(
                    (event.virtual_stage_id, microbatch_id),
                    DistributedPipelineForwardCache {
                        input_state: cached_input,
                        output_state: Some(output_state.clone()),
                        loss: None,
                    },
                );
                Some(output_state)
            }
        } else {
            None
        };

        if event.kind == burn_dragon_train::train::pipeline::PipelineEventKind::Forward
            && event.virtual_stage_id < last_virtual_stage_id
        {
            for replica_id in 0..layout.data_parallel_size {
                let sender_rank =
                    global_rank_for_virtual_stage(plan, layout, replica_id, event.virtual_stage_id);
                let receiver_rank = global_rank_for_virtual_stage(
                    plan,
                    layout,
                    replica_id,
                    event.virtual_stage_id + 1,
                );

                if sender_rank == receiver_rank {
                    if assignment.data_parallel_rank == replica_id
                        && assignment.global_rank == receiver_rank
                    {
                        let forwarded = detach_pipeline_state_to_inner(
                            local_forward_output.as_ref().ok_or_else(|| {
                                anyhow!(
                                    "missing local forward state for virtual_stage={} microbatch={microbatch_id}",
                                    event.virtual_stage_id
                                )
                            })?,
                        );
                        incoming_forward
                            .insert((event.virtual_stage_id + 1, microbatch_id), forwarded);
                    }
                    continue;
                }

                let broadcasted = broadcast_pipeline_state_rooted(
                    peer_id,
                    assignment.global_rank,
                    sender_rank,
                    device,
                    (assignment.data_parallel_rank == replica_id
                        && assignment.global_rank == sender_rank)
                        .then_some(local_forward_output.as_ref())
                        .flatten(),
                )?;
                if assignment.data_parallel_rank == replica_id
                    && assignment.global_rank == receiver_rank
                {
                    incoming_forward
                        .insert((event.virtual_stage_id + 1, microbatch_id), broadcasted);
                }
            }
        }

        let local_backward_grad = if event.kind
            == burn_dragon_train::train::pipeline::PipelineEventKind::Backward
            && event.physical_stage_id == assignment.pipeline_stage_id
        {
            let cached = forward_cache
                .remove(&(event.virtual_stage_id, microbatch_id))
                .ok_or_else(|| {
                    anyhow!(
                        "missing backward cache for virtual_stage={} microbatch={microbatch_id}",
                        event.virtual_stage_id
                    )
                })?;

            let mut grads = if event.virtual_stage_id == last_virtual_stage_id {
                cached
                    .loss
                    .ok_or_else(|| {
                        anyhow!(
                            "missing terminal loss for virtual_stage={} microbatch={microbatch_id}",
                            event.virtual_stage_id
                        )
                    })?
                    .backward()
            } else {
                let output_state = cached.output_state.as_ref().ok_or_else(|| {
                    anyhow!(
                        "missing stage output for virtual_stage={} microbatch={microbatch_id}",
                        event.virtual_stage_id
                    )
                })?;
                let grad_state = incoming_backward
                        .remove(&(event.virtual_stage_id, microbatch_id))
                        .ok_or_else(|| {
                            anyhow!(
                                "missing backward pipeline gradient for virtual_stage={} microbatch={microbatch_id}",
                                event.virtual_stage_id
                            )
                        })?;
                pipeline_surrogate_loss(output_state, grad_state).backward()
            };

            let input_grad = cached
                .input_state
                .as_ref()
                .map(|input_state| pipeline_input_grad_state(input_state, &mut grads));
            local_accumulator.accumulate(model, GradientsParams::from_grads(grads, model));
            input_grad
        } else {
            None
        };

        if event.kind == burn_dragon_train::train::pipeline::PipelineEventKind::Backward
            && event.virtual_stage_id > 0
        {
            for replica_id in 0..layout.data_parallel_size {
                let sender_rank =
                    global_rank_for_virtual_stage(plan, layout, replica_id, event.virtual_stage_id);
                let receiver_rank = global_rank_for_virtual_stage(
                    plan,
                    layout,
                    replica_id,
                    event.virtual_stage_id - 1,
                );

                if sender_rank == receiver_rank {
                    if assignment.data_parallel_rank == replica_id
                        && assignment.global_rank == receiver_rank
                    {
                        let grad_state = local_backward_grad.clone().ok_or_else(|| {
                            anyhow!(
                                "missing local backward gradient for virtual_stage={} microbatch={microbatch_id}",
                                event.virtual_stage_id
                            )
                        })?;
                        incoming_backward
                            .insert((event.virtual_stage_id - 1, microbatch_id), grad_state);
                    }
                    continue;
                }

                let broadcasted = broadcast_pipeline_state_inner_rooted::<B::InnerBackend>(
                    peer_id,
                    assignment.global_rank,
                    sender_rank,
                    device,
                    (assignment.data_parallel_rank == replica_id
                        && assignment.global_rank == sender_rank)
                        .then_some(local_backward_grad.as_ref())
                        .flatten(),
                )?;
                if assignment.data_parallel_rank == replica_id
                    && assignment.global_rank == receiver_rank
                {
                    incoming_backward
                        .insert((event.virtual_stage_id - 1, microbatch_id), broadcasted);
                }
            }
        }
    }

    let mean_train_loss = if reduce_metric_loss {
        let reduced_loss = reduce_sum_scalar::<B::InnerBackend>(
            peer_id,
            if assignment.is_last_stage() {
                local_loss.unwrap_or_else(|| Tensor::<B::InnerBackend, 1>::zeros([1], device))
            } else {
                Tensor::<B::InnerBackend, 1>::zeros([1], device)
            },
        )?;
        Some(reduced_loss / layout.data_parallel_size as f64)
    } else {
        None
    };

    Ok(DistributedPipelineTrainStepResult {
        grads: local_accumulator.grads(),
        mean_train_loss,
    })
}

#[cfg(feature = "ddp")]
fn train_with_collective_pipeline_scheduler<B, O, S>(
    env: &TrainEnvironment<'_, B>,
    mut learner: burn_train::Learner<
        burn_train::LearningComponentsMarker<B, S, LanguageTrainModel<B>, O>,
    >,
    local_train_loader: Arc<dyn DataLoader<B, SequenceBatch<B>>>,
    peer_id: PeerId,
    layout: PipelineParallelLayout,
    assignment: PipelineRankAssignment,
) -> Result<DragonModel<ValidBackend<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    O: Optimizer<LanguageTrainModel<B>, B> + 'static,
    S: LrScheduler + 'static,
{
    let global_train_steps = env.train_loader.num_items();
    if global_train_steps % layout.data_parallel_size != 0 {
        return Err(anyhow!(
            "parallel.pipeline.enabled process-group execution requires env.train_loader.num_items() divisible by parallel.data.size so every replica executes the same number of collectives (got {} steps across {} replicas)",
            global_train_steps,
            layout.data_parallel_size
        ));
    }

    let local_train_steps = local_train_loader.num_items();
    let expected_local_train_steps = global_train_steps / layout.data_parallel_size;
    if local_train_steps != expected_local_train_steps {
        return Err(anyhow!(
            "parallel.pipeline.enabled process-group execution expected {} local steps for dp_rank={} but resolved {}",
            expected_local_train_steps,
            assignment.data_parallel_rank,
            local_train_steps
        ));
    }
    let metric_every = env.training.log_frequency.max(1);
    let grad_accumulation = env.training.gradient_accumulation_steps.max(1);
    let logical_replica_count = layout.data_parallel_size;
    let start_epoch = env
        .resume_checkpoint_epoch
        .map(|epoch| epoch + 1)
        .unwrap_or(1);

    for epoch in start_epoch..=env.epochs {
        info!(
            "Executing process-group pipeline epoch {} on global_rank={} stage={} dp_rank={}",
            epoch,
            assignment.global_rank,
            assignment.pipeline_stage_id,
            assignment.data_parallel_rank
        );

        let mut iterator = local_train_loader.iter();
        let mut iteration = 0usize;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0usize;

        while iteration < local_train_steps {
            let mut batch = None;
            for replica_id in 0..layout.data_parallel_size {
                let batch_root_rank = pipeline_replica_root_rank(&layout, replica_id);
                let replica_root_batch = if assignment.data_parallel_rank == replica_id
                    && assignment.global_rank == batch_root_rank
                {
                    iterator.next()
                } else {
                    None
                };
                let replica_batch = broadcast_sequence_batch_rooted(
                    peer_id,
                    assignment.global_rank,
                    batch_root_rank,
                    env.device,
                    replica_root_batch,
                )?;
                if assignment.data_parallel_rank == replica_id {
                    batch = Some(replica_batch);
                }
            }
            let batch = batch.ok_or_else(|| {
                anyhow!(
                    "missing local replica batch for dp_rank={} at iteration={iteration}",
                    assignment.data_parallel_rank
                )
            })?;

            iteration += 1;
            for _ in 0..logical_replica_count {
                learner.lr_step();
            }

            let absolute_step = epoch
                .saturating_sub(1)
                .saturating_mul(local_train_steps)
                .saturating_add(iteration.saturating_sub(1));
            let source_selection_due = source_selection_telemetry_due(env, absolute_step);
            let log_train_metrics = iteration % metric_every == 0 || iteration == local_train_steps;
            let step = distributed_pipeline_train_step(
                peer_id,
                &learner.model(),
                batch,
                &layout,
                &assignment,
                env.device,
                source_selection_due || log_train_metrics,
            )?;
            if source_selection_due
                && let (Some(dataset), Some(mean_train_loss)) = (
                    env.source_selection_dataset
                        .as_ref()
                        .filter(|dataset| dataset.uses_live_source_selection()),
                    step.mean_train_loss,
                )
            {
                let _ = dataset.record_source_selection_loss(absolute_step, mean_train_loss as f32);
            }

            accumulator.accumulate(&learner.model(), step.grads);
            accumulation_current += 1;

            if grad_accumulation <= accumulation_current {
                let mut grads = accumulator.grads();
                all_reduce_gradients_in_module_order(
                    &learner.model(),
                    &mut grads,
                    peer_id,
                    ReduceOperation::Sum,
                )?;
                scale_gradients_in_module_order::<B, _>(
                    &learner.model(),
                    &mut grads,
                    1.0 / layout.data_parallel_size as f32,
                );
                learner.optimizer_step(grads);
                accumulation_current = 0;
            }

            if env.parallel_runtime.is_primary()
                && log_train_metrics
                && let Some(mean_train_loss) = step.mean_train_loss
            {
                let progress = iterator.progress();
                let global_iteration = epoch
                    .saturating_sub(1)
                    .saturating_mul(logical_replica_count.saturating_mul(local_train_steps))
                    .saturating_add(iteration.saturating_mul(logical_replica_count));
                info!(
                    "train epoch={} local_step={}/{} global_iteration={} loss={:.4} lr={:.6} global_progress={}/{}",
                    epoch,
                    progress.items_processed,
                    progress.items_total,
                    global_iteration,
                    mean_train_loss,
                    learner.lr_current(),
                    epoch,
                    env.epochs
                );
            }
        }

        if env.parallel_runtime.is_primary() {
            if let Some(valid_loss) = run_process_group_validation::<B, O, S>(env, &learner) {
                info!("valid epoch={} loss={valid_loss:.4}", epoch);
            }
            save_process_group_checkpoint::<B, O, S>(env.run_dir, epoch, &learner)?;
        }
    }

    Ok(learner.model().valid().model)
}

#[cfg(feature = "ddp")]
fn train_with_collective_scheduler<B, O, S>(
    env: &TrainEnvironment<'_, B>,
    model: LanguageTrainModel<B>,
    optimizer: O,
    scheduler: S,
    collective: burn_collective::CollectiveConfig,
    peer_id: PeerId,
) -> Result<DragonModel<ValidBackend<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    O: Optimizer<LanguageTrainModel<B>, B> + 'static,
    S: LrScheduler + 'static,
{
    let _session = CollectiveSessionGuard::<B::InnerBackend>::register(
        peer_id,
        env.device.clone(),
        collective,
    )?;

    let (data_shard_index, data_shard_count, pipeline_assignment, pipeline_layout) =
        process_group_data_shard(env.parallel_runtime, env.parallel_config)?;

    let local_train_loader = shard_dataloader(
        Arc::clone(&env.train_loader),
        data_shard_index,
        data_shard_count,
        "train",
    )?;

    let metric_every = env.training.log_frequency.max(1);
    let grad_accumulation = env.training.gradient_accumulation_steps.max(1);
    let local_train_steps = local_train_loader.num_items();
    let mut learner = burn_train::Learner::new(model, optimizer, scheduler);
    if let Some(checkpoint) = env.resume_checkpoint_epoch {
        learner =
            load_process_group_checkpoint::<B, O, S>(env.run_dir, checkpoint, env.device, learner)?;
    }
    let start_epoch = env
        .resume_checkpoint_epoch
        .map(|epoch| epoch + 1)
        .unwrap_or(1);

    info!(
        "training strategy: mode={:?} replicas={} local_rank={} global_rank={} local_train_steps={} start_epoch={}",
        env.parallel_runtime.mode,
        env.parallel_runtime.world_size,
        env.parallel_runtime.local_rank,
        env.parallel_runtime.global_rank,
        local_train_steps,
        start_epoch
    );
    if let (Some(layout), Some(assignment)) = (&pipeline_layout, &pipeline_assignment) {
        info!(
            "process-group pipeline topology: {} rank={} stage={} dp_rank={} predecessor={:?} successor={:?} pipeline_group={:?} dp_group={:?}",
            layout.summary(),
            assignment.global_rank,
            assignment.pipeline_stage_id,
            assignment.data_parallel_rank,
            assignment.predecessor_global_rank,
            assignment.successor_global_rank,
            assignment.pipeline_group_ranks,
            assignment.data_parallel_group_ranks,
        );
    }

    if let (Some(layout), Some(assignment)) = (pipeline_layout.clone(), pipeline_assignment.clone())
    {
        return train_with_collective_pipeline_scheduler(
            env,
            learner,
            local_train_loader,
            peer_id,
            layout,
            assignment,
        );
    }

    for epoch in start_epoch..=env.epochs {
        info!(
            "Executing process-group DDP epoch {} on global_rank={}",
            epoch, env.parallel_runtime.global_rank
        );

        let mut iterator = local_train_loader.iter();
        let mut iteration = 0usize;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0usize;
        let logical_replica_count = env.parallel_runtime.world_size;
        while let Some(item) = iterator.next() {
            iteration += 1;
            for _ in 0..logical_replica_count {
                learner.lr_step();
            }

            let item = learner.train_step(item);
            let absolute_step = epoch
                .saturating_sub(1)
                .saturating_mul(local_train_steps)
                .saturating_add(iteration.saturating_sub(1));
            let source_selection_due = source_selection_telemetry_due(env, absolute_step);
            let log_train_metrics = iteration % metric_every == 0 || iteration == local_train_steps;
            let mean_train_loss = if source_selection_due || log_train_metrics {
                let train_output = item.item.sync();
                let loss_value: LossValue<ValidBackend<B>> = train_output.adapt();
                Some(reduce_mean_scalar::<ValidBackend<B>>(
                    peer_id,
                    loss_value.value(),
                )?)
            } else {
                None
            };
            if source_selection_due
                && let (Some(dataset), Some(mean_train_loss)) = (
                    env.source_selection_dataset
                        .as_ref()
                        .filter(|dataset| dataset.uses_live_source_selection()),
                    mean_train_loss,
                )
            {
                let _ = dataset.record_source_selection_loss(absolute_step, mean_train_loss as f32);
            }

            accumulator.accumulate(&learner.model(), item.grads);
            accumulation_current += 1;

            if grad_accumulation <= accumulation_current {
                info!(
                    "process-group DDP rank={} iteration={} entering gradient all-reduce",
                    env.parallel_runtime.global_rank, iteration
                );
                let mut grads = accumulator.grads();
                // Fresh multi-process launches instantiate random ParamIds per rank, so
                // cross-rank gradient sync must follow deterministic module traversal order.
                all_reduce_gradients_in_module_order(
                    &learner.model(),
                    &mut grads,
                    peer_id,
                    ReduceOperation::Mean,
                )?;
                info!(
                    "process-group DDP rank={} iteration={} completed gradient all-reduce",
                    env.parallel_runtime.global_rank, iteration
                );
                learner.optimizer_step(grads);
                accumulation_current = 0;
            }

            if env.parallel_runtime.is_primary()
                && log_train_metrics
                && let Some(mean_train_loss) = mean_train_loss
            {
                let progress = iterator.progress();
                let global_iteration = epoch
                    .saturating_sub(1)
                    .saturating_mul(logical_replica_count.saturating_mul(local_train_steps))
                    .saturating_add(iteration.saturating_mul(logical_replica_count));
                info!(
                    "train epoch={} local_step={}/{} global_iteration={} loss={:.4} lr={:.6} global_progress={}/{}",
                    epoch,
                    progress.items_processed,
                    progress.items_total,
                    global_iteration,
                    mean_train_loss,
                    learner.lr_current(),
                    epoch,
                    env.epochs
                );
            }
        }

        if env.parallel_runtime.is_primary() {
            if let Some(valid_loss) = run_process_group_validation::<B, O, S>(env, &learner) {
                info!("valid epoch={} loss={valid_loss:.4}", epoch);
            }
            save_process_group_checkpoint::<B, O, S>(env.run_dir, epoch, &learner)?;
        }
    }

    Ok(learner.model().valid().model)
}

#[cfg(feature = "ddp")]
fn train_with_process_group_scheduler<B, O, S>(
    env: &TrainEnvironment<'_, B>,
    model: LanguageTrainModel<B>,
    optimizer: O,
    scheduler: S,
) -> Result<DragonModel<ValidBackend<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    O: Optimizer<LanguageTrainModel<B>, B> + 'static,
    S: LrScheduler + 'static,
{
    let collective = resolve_collective_config(env.parallel_runtime, env.parallel_config)?;
    train_with_collective_scheduler::<B, O, S>(
        env,
        model,
        optimizer,
        scheduler,
        collective,
        process_group_peer_id(env.parallel_runtime),
    )
}

pub fn resolve_lr_scheduler(
    optimizer_cfg: &OptimizerConfig,
    total_steps: usize,
    override_num_iters: Option<usize>,
    model_config: &DragonConfig,
) -> Result<ResolvedLrScheduler> {
    burn_dragon_train::train::pipeline::resolve_lr_scheduler(
        optimizer_cfg,
        total_steps,
        override_num_iters,
        model_config.n_embd,
    )
}

pub fn resolve_train_schedule(
    training: &TrainingHyperparameters,
    steps_per_epoch: usize,
) -> Result<TrainSchedule> {
    burn_dragon_train::train::pipeline::resolve_train_schedule(
        training.epochs,
        training.max_iters,
        steps_per_epoch,
        "training",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::data::dataloader::{DataLoaderIterator, Progress};
    #[cfg(feature = "ddp")]
    use burn::module::list_param_ids;
    use burn::tensor::TensorData;
    use burn_autodiff::Autodiff;
    #[cfg(feature = "ddp")]
    use burn_collective::reset_collective;
    use burn_ndarray::NdArray;
    use burn_train::checkpoint::CheckpointingAction;
    #[cfg(feature = "ddp")]
    use std::sync::{Mutex, OnceLock};
    #[cfg(feature = "ddp")]
    use tempfile::tempdir;

    type TestBackend = Autodiff<NdArray<f32>>;
    type TestValidBackend = ValidBackend<TestBackend>;
    type TestForwardBackend = NdArray<f32>;

    fn degeneracy_stats(
        entropy_bits: f64,
        mean_max_probability: f64,
        distinct_2_fraction: f64,
        repetition_fraction: f64,
    ) -> crate::train::steps::OutputDegeneracyStats {
        crate::train::steps::OutputDegeneracyStats {
            token_count: 128,
            entropy_bits,
            mean_max_probability,
            argmax_unique_fraction: 0.02,
            eos_fraction: 0.0,
            repetition_fraction,
            distinct_1_fraction: 0.02,
            distinct_2_fraction,
            period_2_fraction: 0.0,
            period_3_fraction: 0.0,
            max_period_2_to_16_fraction: 0.0,
            max_period_2_to_64_fraction: 0.0,
            dominant_period_2_to_64: 0,
            prompt_max_period_2_to_64_fraction: 0.0,
            prompt_dominant_period_2_to_64: 0,
            prompt_tokens: Vec::new(),
            generated_tokens: Vec::new(),
        }
    }

    fn ruliad_degeneracy_gates() -> burn_dragon_train::TrainingGatesConfig {
        burn_dragon_train::TrainingGatesConfig {
            degeneracy_entropy_min_bits: 2.0,
            degeneracy_max_probability_max: 0.90,
            degeneracy_argmax_unique_min_fraction: 0.08,
            degeneracy_distinct_2_min_fraction: 0.20,
            degeneracy_repetition_max_fraction: 0.75,
            degeneracy_period_2_max_fraction: 0.90,
            degeneracy_period_3_max_fraction: 0.90,
            degeneracy_period_2_to_16_max_fraction: 0.90,
            degeneracy_period_2_to_64_max_fraction: 0.90,
            degeneracy_patience: 1,
            ..burn_dragon_train::TrainingGatesConfig::default()
        }
    }

    fn ruliad_eval_report(
        verifier_accuracy: f32,
        semantic_accuracy: f32,
        mean_partial_progress: f32,
        certificate_prefix_coverage: f32,
    ) -> burn_dragon_universality::RuliadEvalReport {
        let item_count = 100usize;
        burn_dragon_universality::RuliadEvalReport {
            version: burn_dragon_universality::ruliad::RULIAD_EVAL_REPORT_VERSION,
            reasoning_score_version:
                burn_dragon_universality::ruliad::RULIAD_REASONING_SCORE_VERSION,
            dataset_name: "test".to_string(),
            item_count,
            scored_count: item_count,
            exact_match_count: 0,
            semantic_match_count: (semantic_accuracy.clamp(0.0, 1.0) * item_count as f32).round()
                as usize,
            verifier_match_count: (verifier_accuracy.clamp(0.0, 1.0) * item_count as f32).round()
                as usize,
            partial_credit_count: (mean_partial_progress.clamp(0.0, 1.0) * item_count as f32)
                .round() as usize,
            schema_valid_wrong_count: 0,
            malformed_completion_count: 0,
            missing_completion_count: 0,
            unexpected_completion_count: 0,
            exact_accuracy: 0.0,
            semantic_accuracy,
            verifier_accuracy,
            partial_credit_rate: mean_partial_progress,
            mean_partial_progress,
            answer_field_correct_count: (mean_partial_progress.clamp(0.0, 1.0) * item_count as f32)
                .round() as usize,
            answer_field_expected_count: item_count,
            answer_field_accuracy: mean_partial_progress,
            answer_terminated_count: item_count,
            answer_termination_rate: 1.0,
            mean_certificate_prefix_coverage: certificate_prefix_coverage,
            mean_completion_tokens: 12.0,
            canary_count: 0,
            canary_semantic_match_count: 0,
            family_scores: Vec::new(),
            task_scores: Vec::new(),
            difficulty_scores: Vec::new(),
            math_domain_scores: Vec::new(),
            reasoning_mode_scores: Vec::new(),
            failures: Vec::new(),
        }
    }

    #[test]
    fn checkpoint_promotion_rejects_loss_only_ruliad_progress_when_free_run_is_flat() {
        let gates = ruliad_degeneracy_gates();
        let best_competence = ruliad_competence_key(&ruliad_eval_report(0.0, 0.0, 0.0, 0.0));
        let validation = DynamicValidationReport {
            loss: 0.01,
            source_weighted_loss: None,
            stream_warm_loss: None,
            output_degeneracy: None,
            ruliad_eval_report: Some(ruliad_eval_report(0.0, 0.0, 0.0, 0.0)),
        };

        assert!(!should_promote_checkpoint(
            &validation,
            Some(1.0),
            best_competence,
            &gates
        ));
    }

    #[test]
    fn checkpoint_promotion_prefers_free_run_ruliad_competence_over_teacher_forced_loss() {
        let gates = ruliad_degeneracy_gates();
        let best_competence = ruliad_competence_key(&ruliad_eval_report(0.0, 0.0, 0.0, 0.0));
        let validation = DynamicValidationReport {
            loss: 1.5,
            source_weighted_loss: None,
            stream_warm_loss: None,
            output_degeneracy: None,
            ruliad_eval_report: Some(ruliad_eval_report(0.01, 0.01, 0.10, 0.10)),
        };

        assert!(should_promote_checkpoint(
            &validation,
            Some(1.0),
            best_competence,
            &gates
        ));
    }

    #[test]
    fn checkpoint_promotion_rejects_loss_only_when_capability_gate_fails() {
        let mut gates = ruliad_degeneracy_gates();
        gates.capability_schema_wrong_max_rate = 0.25;
        let mut report = ruliad_eval_report(0.25, 0.25, 0.25, 0.25);
        report.schema_valid_wrong_count = 40;
        let validation = DynamicValidationReport {
            loss: 0.01,
            source_weighted_loss: None,
            stream_warm_loss: None,
            output_degeneracy: None,
            ruliad_eval_report: Some(report),
        };

        assert!(!should_promote_checkpoint(&validation, None, None, &gates));
    }

    #[test]
    fn ruliad_correctness_progress_suppresses_loss_only_regression() {
        let state = ContinualLearningStabilityState {
            best_ruliad_verifier_accuracy: Some(0.1875),
            best_ruliad_partial_progress: Some(0.2917),
            ..Default::default()
        };
        let validation = DynamicValidationReport {
            loss: 0.454,
            ruliad_eval_report: Some(ruliad_eval_report(0.21875, 0.21875, 0.2917, 0.0)),
            ..Default::default()
        };

        assert!(validation_ruliad_correctness_improved(&validation, &state));
    }

    #[test]
    fn flat_ruliad_correctness_does_not_suppress_loss_regression() {
        let state = ContinualLearningStabilityState {
            best_ruliad_verifier_accuracy: Some(0.1875),
            best_ruliad_partial_progress: Some(0.2917),
            ..Default::default()
        };
        let validation = DynamicValidationReport {
            loss: 0.454,
            ruliad_eval_report: Some(ruliad_eval_report(0.1875, 0.1875, 0.2917, 0.0)),
            ..Default::default()
        };

        assert!(!validation_ruliad_correctness_improved(&validation, &state));
    }

    #[test]
    fn ruliad_correctness_regression_threshold_ignores_one_item_probe_noise() {
        assert!(!ruliad_metric_materially_regressed(
            0.21875, 0.1875, 32, 0.125
        ));
    }

    #[test]
    fn ruliad_correctness_regression_threshold_flags_material_probe_drop() {
        assert!(ruliad_metric_materially_regressed(
            0.21875, 0.1875, 128, 0.125
        ));
    }

    #[test]
    fn ruliad_capability_gate_status_flags_malformed_missing_and_output_collapse() {
        let mut gates = ruliad_degeneracy_gates();
        gates.capability_malformed_max_rate = 0.02;
        gates.capability_missing_max_rate = 0.02;
        gates.capability_completion_health_min_rate = 0.80;
        gates.capability_output_entropy_min_bits = 1.25;
        gates.capability_distinct_2_min_fraction = 0.30;
        let mut report = ruliad_eval_report(0.25, 0.25, 0.25, 0.25);
        report.malformed_completion_count = 5;
        report.missing_completion_count = 3;
        let stats = degeneracy_stats(0.5, 0.9, 0.1, 0.0);

        let status = ruliad_capability_gate_status(&report, Some(&stats), &gates);

        assert!(!status.passed);
        assert!(
            status
                .reasons
                .iter()
                .any(|reason| reason.starts_with("malformed_rate="))
        );
        assert!(
            status
                .reasons
                .iter()
                .any(|reason| reason.starts_with("missing_rate="))
        );
        assert!(
            status
                .reasons
                .iter()
                .any(|reason| reason.starts_with("output_entropy_bits="))
        );
        assert!(
            status
                .reasons
                .iter()
                .any(|reason| reason.starts_with("output_distinct2="))
        );
    }

    #[test]
    fn ruliad_capability_gate_status_respects_disabled_gates() {
        let mut gates = ruliad_degeneracy_gates();
        gates.enabled = false;
        gates.capability_schema_wrong_max_rate = 0.0;
        let mut report = ruliad_eval_report(0.0, 0.0, 0.0, 0.0);
        report.schema_valid_wrong_count = report.item_count;

        let status = ruliad_capability_gate_status(&report, None, &gates);

        assert!(status.passed);
        assert!(status.reasons.is_empty());
    }

    #[test]
    fn capability_run_control_warns_during_grace_then_recovers_after_first_pass_regression() {
        let dir = tempfile::tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let parallel_config = burn_dragon_train::ParallelConfig::default();
        let parallel_runtime =
            resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");
        let device = burn::tensor::Device::<TestBackend>::default();
        let valid_device = burn::tensor::Device::<TestValidBackend>::default();
        let mut training = tiny_training_hparams();
        training.events.flush_every_steps = 1;
        training.gates = burn_dragon_train::TrainingGatesConfig {
            capability_grace_epochs: 3,
            capability_regression_patience_epochs: 2,
            capability_required_after_first_pass: true,
            capability_schema_wrong_max_rate: 0.25,
            ..ruliad_degeneracy_gates()
        };
        let model_config = tiny_model_config();
        let devices = vec![device.clone()];
        let env = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name: "capability-run-control-smoke",
            backend_name: "cpu",
            training: &training,
            resume_checkpoint_epoch: None,
            model_config: &model_config,
            device: &device,
            devices: &devices,
            train_dataset: None,
            valid_dataset: None,
            train_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<TestBackend>(
                &device,
                &[0, 1, 2, 3, 4, 5, 6, 7],
                &[1, 2, 3, 4, 5, 6, 7, 0],
                [2, 4],
            )])),
            valid_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<
                TestValidBackend,
            >(
                &valid_device,
                &[0, 0, 1, 1, 2, 2, 3, 3],
                &[0, 1, 1, 2, 2, 3, 3, 0],
                [2, 4],
            )])),
            source_selection_dataset: None,
            summary_event_token_ids: None,
            neuron_scaling_slot: None,
            epochs: 1,
            total_steps: 1,
            valid_steps: 1,
        };
        let handles = crate::train::events::build_training_event_handles(
            env.run_name,
            &run_dir,
            1,
            &training,
            None,
            None,
            None,
        )
        .expect("event handles");
        let bus = handles.metric_logger.bus();
        let mut state = ContinualLearningStabilityState::default();
        let mut bad_report = ruliad_eval_report(0.25, 0.25, 0.25, 0.25);
        bad_report.schema_valid_wrong_count = 80;
        let good_report = ruliad_eval_report(0.25, 0.25, 0.25, 0.25);

        apply_continual_learning_stability_policy(
            &env,
            DynamicValidationReport {
                loss: 1.0,
                ruliad_eval_report: Some(bad_report.clone()),
                ..Default::default()
            },
            1,
            0,
            &mut state,
            &bus,
        );
        apply_continual_learning_stability_policy(
            &env,
            DynamicValidationReport {
                loss: 0.9,
                ruliad_eval_report: Some(good_report),
                ..Default::default()
            },
            4,
            1,
            &mut state,
            &bus,
        );
        apply_continual_learning_stability_policy(
            &env,
            DynamicValidationReport {
                loss: 0.9,
                ruliad_eval_report: Some(bad_report.clone()),
                ..Default::default()
            },
            5,
            2,
            &mut state,
            &bus,
        );
        apply_continual_learning_stability_policy(
            &env,
            DynamicValidationReport {
                loss: 0.9,
                ruliad_eval_report: Some(bad_report),
                ..Default::default()
            },
            6,
            3,
            &mut state,
            &bus,
        );
        let _ = bus.flush();
        drop(handles);

        let events = read_training_events(&run_dir);
        assert!(events.iter().any(|event| {
            event.get("type").and_then(|value| value.as_str()) == Some("gate")
                && event.get("gate").and_then(|value| value.as_str())
                    == Some("continual_learning_capability_gate_grace")
        }));
        assert!(events.iter().any(|event| {
            event.get("type").and_then(|value| value.as_str()) == Some("dynamics_control")
                && event.get("mode").and_then(|value| value.as_str()) == Some("validation_recovery")
        }));
        assert_eq!(state.first_capability_pass_epoch, Some(4));
        assert_eq!(state.consecutive_capability_gate_failures, 2);
    }

    #[test]
    fn ruliad_correctness_regression_rolls_back_to_promoted_checkpoint() {
        let dir = tempfile::tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let parallel_config = burn_dragon_train::ParallelConfig::default();
        let parallel_runtime =
            resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");
        let device = burn::tensor::Device::<TestBackend>::default();
        let valid_device = burn::tensor::Device::<TestValidBackend>::default();
        let mut training = tiny_training_hparams();
        training.events.flush_every_steps = 1;
        training.gates = ruliad_degeneracy_gates();
        let model_config = tiny_model_config();
        let devices = vec![device.clone()];
        let env = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name: "ruliad-regression-rollback-target-smoke",
            backend_name: "cpu",
            training: &training,
            resume_checkpoint_epoch: None,
            model_config: &model_config,
            device: &device,
            devices: &devices,
            train_dataset: None,
            valid_dataset: None,
            train_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<TestBackend>(
                &device,
                &[0, 1, 2, 3, 4, 5, 6, 7],
                &[1, 2, 3, 4, 5, 6, 7, 0],
                [2, 4],
            )])),
            valid_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<
                TestValidBackend,
            >(
                &valid_device,
                &[0, 0, 1, 1, 2, 2, 3, 3],
                &[0, 1, 1, 2, 2, 3, 3, 0],
                [2, 4],
            )])),
            source_selection_dataset: None,
            summary_event_token_ids: None,
            neuron_scaling_slot: None,
            epochs: 1,
            total_steps: 1,
            valid_steps: 1,
        };
        let handles = crate::train::events::build_training_event_handles(
            env.run_name,
            &run_dir,
            1,
            &training,
            None,
            None,
            None,
        )
        .expect("event handles");
        let bus = handles.metric_logger.bus();
        let mut report = ruliad_eval_report(0.1328125, 0.1328125, 0.21875, 0.0);
        report.item_count = 128;
        report.scored_count = 128;
        let mut state = ContinualLearningStabilityState {
            best_valid_loss: Some(0.397696),
            best_checkpoint_epoch: Some(4),
            best_ruliad_verifier_accuracy: Some(0.203125),
            best_ruliad_partial_progress: Some(0.3125),
            ..Default::default()
        };
        apply_continual_learning_stability_policy(
            &env,
            DynamicValidationReport {
                loss: 0.357596,
                source_weighted_loss: None,
                stream_warm_loss: None,
                output_degeneracy: None,
                ruliad_eval_report: Some(report),
            },
            5,
            2559,
            &mut state,
            &bus,
        );
        let _ = bus.flush();
        drop(handles);

        let control = read_training_events(&run_dir)
            .into_iter()
            .rev()
            .find(|event| {
                event.get("type").and_then(|value| value.as_str()) == Some("dynamics_control")
            })
            .expect("dynamics control event");
        assert_eq!(
            control.get("mode").and_then(|value| value.as_str()),
            Some("rollback_recovery")
        );
        assert_eq!(
            control
                .get("rollback_to_epoch")
                .and_then(|value| value.as_u64()),
            Some(4)
        );
    }

    #[test]
    fn output_degeneracy_policy_warns_on_low_confidence_argmax_loop() {
        let gates = ruliad_degeneracy_gates();
        let stats = degeneracy_stats(11.0, 0.03, 0.03, 0.94);

        assert!(uncertain_argmax_loop(&gates, &stats));
        assert!(output_degeneracy_tripped(&gates, &stats));
        assert!(!hard_output_collapse_for_gates(&gates, &stats));
    }

    #[test]
    fn quiet_progress_renderer_is_default_for_training_efficiency() {
        assert!(quiet_progress_renderer_enabled_for("quiet"));
        assert!(quiet_progress_renderer_enabled_for("off"));
        assert!(quiet_progress_renderer_enabled_for(""));
        assert!(!quiet_progress_renderer_enabled_for("progress"));
        assert!(!quiet_progress_renderer_enabled_for("default"));
    }

    #[test]
    fn output_degeneracy_policy_keeps_low_diversity_nonperiodic_output_soft() {
        let gates = ruliad_degeneracy_gates();
        let stats = degeneracy_stats(3.5, 0.52, 0.05, 0.04);

        assert!(!uncertain_argmax_loop(&gates, &stats));
        assert!(output_degeneracy_tripped(&gates, &stats));
        assert!(!hard_output_collapse_for_gates(&gates, &stats));
    }

    #[test]
    fn output_degeneracy_policy_accepts_structured_low_entropy_output() {
        let gates = ruliad_degeneracy_gates();
        let mut stats = degeneracy_stats(1.35, 0.78, 0.38, 0.01);
        stats.argmax_unique_fraction = 0.13;
        stats.distinct_1_fraction = 0.13;
        stats.period_2_fraction = 0.02;
        stats.period_3_fraction = 0.24;
        stats.max_period_2_to_16_fraction = 0.36;
        stats.max_period_2_to_64_fraction = 0.36;
        stats.dominant_period_2_to_64 = 6;

        assert!(!output_degeneracy_tripped(&gates, &stats));
        assert!(!hard_output_collapse_for_gates(&gates, &stats));
    }

    #[test]
    fn output_degeneracy_policy_ignores_periodic_but_diverse_structure() {
        let mut gates = ruliad_degeneracy_gates();
        gates.degeneracy_period_2_to_16_max_fraction = 0.40;
        gates.degeneracy_period_2_to_64_max_fraction = 0.40;
        let mut stats = degeneracy_stats(3.99, 0.07, 0.60, 0.20);
        stats.argmax_unique_fraction = 0.50;
        stats.distinct_1_fraction = 0.50;
        stats.period_2_fraction = 0.05;
        stats.period_3_fraction = 0.03;
        stats.max_period_2_to_16_fraction = 0.46;
        stats.max_period_2_to_64_fraction = 0.46;
        stats.dominant_period_2_to_64 = 11;

        assert!(!uncertain_argmax_loop(&gates, &stats));
        assert!(!output_degeneracy_tripped(&gates, &stats));
        assert!(!hard_output_collapse_for_gates(&gates, &stats));
    }

    #[test]
    fn output_degeneracy_policy_flags_short_period_argmax_loop() {
        let mut gates = ruliad_degeneracy_gates();
        gates.degeneracy_period_2_to_16_max_fraction = 0.50;
        gates.degeneracy_period_2_to_64_max_fraction = 0.50;
        let mut stats = degeneracy_stats(3.20, 0.61, 0.55, 0.08);
        stats.argmax_unique_fraction = 0.45;
        stats.distinct_1_fraction = 0.45;
        stats.period_2_fraction = 0.04;
        stats.period_3_fraction = 0.05;
        stats.max_period_2_to_16_fraction = 0.58;
        stats.max_period_2_to_64_fraction = 0.58;
        stats.dominant_period_2_to_64 = 4;

        assert!(!uncertain_argmax_loop(&gates, &stats));
        assert!(output_degeneracy_tripped(&gates, &stats));
        assert!(!hard_output_collapse_for_gates(&gates, &stats));
    }

    #[test]
    fn output_degeneracy_policy_keeps_low_alphabet_periodic_structure_soft() {
        let mut gates = ruliad_degeneracy_gates();
        gates.degeneracy_entropy_min_bits = 1.35;
        gates.degeneracy_max_probability_max = 0.82;
        gates.degeneracy_argmax_unique_min_fraction = 0.20;
        gates.degeneracy_distinct_2_min_fraction = 0.35;
        gates.degeneracy_repetition_max_fraction = 0.45;
        gates.degeneracy_period_2_max_fraction = 0.35;
        gates.degeneracy_period_3_max_fraction = 0.40;
        gates.degeneracy_period_2_to_16_max_fraction = 0.50;
        gates.degeneracy_period_2_to_64_max_fraction = 0.50;
        let mut stats = degeneracy_stats(2.293, 0.617, 0.319, 0.010);
        stats.argmax_unique_fraction = 0.172;
        stats.distinct_1_fraction = 0.172;
        stats.period_2_fraction = 0.0;
        stats.period_3_fraction = 0.005;
        stats.max_period_2_to_16_fraction = 0.573;
        stats.max_period_2_to_64_fraction = 0.573;
        stats.dominant_period_2_to_64 = 14;

        assert!(!uncertain_argmax_loop(&gates, &stats));
        assert!(output_degeneracy_tripped(&gates, &stats));
        assert!(!hard_output_collapse_for_gates(&gates, &stats));
    }

    #[test]
    fn continual_learning_hard_output_degeneracy_requests_recovery_without_direct_stop() {
        let dir = tempfile::tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let parallel_config = burn_dragon_train::ParallelConfig::default();
        let parallel_runtime =
            resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");
        let device = burn::tensor::Device::<TestBackend>::default();
        let valid_device = burn::tensor::Device::<TestValidBackend>::default();
        let mut training = tiny_training_hparams();
        training.events.flush_every_steps = 1;
        training.gates = ruliad_degeneracy_gates();
        let model_config = tiny_model_config();
        let devices = vec![device.clone()];
        let env = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name: "hard-output-degeneracy-alert-smoke",
            backend_name: "cpu",
            training: &training,
            resume_checkpoint_epoch: None,
            model_config: &model_config,
            device: &device,
            devices: &devices,
            train_dataset: None,
            valid_dataset: None,
            train_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<TestBackend>(
                &device,
                &[0, 1, 2, 3, 4, 5, 6, 7],
                &[1, 2, 3, 4, 5, 6, 7, 0],
                [2, 4],
            )])),
            valid_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<
                TestValidBackend,
            >(
                &valid_device,
                &[0, 0, 1, 1, 2, 2, 3, 3],
                &[0, 1, 1, 2, 2, 3, 3, 0],
                [2, 4],
            )])),
            source_selection_dataset: None,
            summary_event_token_ids: None,
            neuron_scaling_slot: None,
            epochs: 1,
            total_steps: 1,
            valid_steps: 1,
        };
        let handles = crate::train::events::build_training_event_handles(
            env.run_name,
            &run_dir,
            1,
            &training,
            None,
            None,
            None,
        )
        .expect("event handles");
        let bus = handles.metric_logger.bus();
        let mut state = ContinualLearningStabilityState::default();
        apply_continual_learning_stability_policy(
            &env,
            DynamicValidationReport {
                loss: 1.0,
                source_weighted_loss: None,
                stream_warm_loss: None,
                output_degeneracy: Some(degeneracy_stats(3.5, 0.52, 0.05, 0.04)),
                ruliad_eval_report: None,
            },
            1,
            0,
            &mut state,
            &bus,
        );
        let _ = bus.flush();
        drop(handles);

        let gate = read_training_events(&run_dir)
            .into_iter()
            .find(|event| {
                event.get("type").and_then(|value| value.as_str()) == Some("gate")
                    && event.get("gate").and_then(|value| value.as_str())
                        == Some("continual_learning_output_degeneracy")
            })
            .expect("output degeneracy gate event");

        assert_eq!(
            gate.get("action").and_then(|value| value.as_str()),
            Some("alert")
        );
        assert_eq!(
            gate.get("severity").and_then(|value| value.as_str()),
            Some("warning")
        );
        let control = read_training_events(&run_dir)
            .into_iter()
            .find(|event| {
                event.get("type").and_then(|value| value.as_str()) == Some("dynamics_control")
                    && event.get("mode").and_then(|value| value.as_str())
                        == Some("plasticity_recovery")
            })
            .expect("output degeneracy should request plasticity recovery");
        assert_eq!(
            control
                .get("stop_if_repeated")
                .and_then(|value| value.as_bool()),
            Some(false)
        );
    }

    #[test]
    fn disabled_dynamics_policy_emits_gate_without_recovery_control() {
        let dir = tempfile::tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let parallel_config = burn_dragon_train::ParallelConfig::default();
        let parallel_runtime =
            resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");
        let device = burn::tensor::Device::<TestBackend>::default();
        let valid_device = burn::tensor::Device::<TestValidBackend>::default();
        let mut training = tiny_training_hparams();
        training.events.flush_every_steps = 1;
        training.gates = ruliad_degeneracy_gates();
        training.dynamics.enabled = false;
        let model_config = tiny_model_config();
        let devices = vec![device.clone()];
        let env = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name: "disabled-dynamics-no-recovery-control",
            backend_name: "cpu",
            training: &training,
            resume_checkpoint_epoch: None,
            model_config: &model_config,
            device: &device,
            devices: &devices,
            train_dataset: None,
            valid_dataset: None,
            train_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<TestBackend>(
                &device,
                &[0, 1, 2, 3, 4, 5, 6, 7],
                &[1, 2, 3, 4, 5, 6, 7, 0],
                [2, 4],
            )])),
            valid_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<
                TestValidBackend,
            >(
                &valid_device,
                &[0, 0, 1, 1, 2, 2, 3, 3],
                &[0, 1, 1, 2, 2, 3, 3, 0],
                [2, 4],
            )])),
            source_selection_dataset: None,
            summary_event_token_ids: None,
            neuron_scaling_slot: None,
            epochs: 1,
            total_steps: 1,
            valid_steps: 1,
        };
        let handles = crate::train::events::build_training_event_handles(
            env.run_name,
            &run_dir,
            1,
            &training,
            None,
            None,
            None,
        )
        .expect("event handles");
        let bus = handles.metric_logger.bus();
        let mut state = ContinualLearningStabilityState::default();
        apply_continual_learning_stability_policy(
            &env,
            DynamicValidationReport {
                loss: 1.0,
                source_weighted_loss: None,
                stream_warm_loss: None,
                output_degeneracy: Some(degeneracy_stats(3.5, 0.52, 0.05, 0.04)),
                ruliad_eval_report: None,
            },
            1,
            0,
            &mut state,
            &bus,
        );
        let _ = bus.flush();
        drop(handles);

        let events = read_training_events(&run_dir);
        assert!(
            events.iter().any(|event| {
                event.get("type").and_then(|value| value.as_str()) == Some("gate")
                    && event.get("gate").and_then(|value| value.as_str())
                        == Some("continual_learning_output_degeneracy")
            }),
            "degeneracy gate should still be visible when dynamics controls are disabled"
        );
        assert!(
            events.iter().all(|event| {
                event.get("type").and_then(|value| value.as_str()) != Some("dynamics_control")
            }),
            "disabled dynamics must not emit recovery controls"
        );
    }

    #[test]
    fn ruliad_correctness_metrics_emit_verifier_rates() {
        let dir = tempfile::tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let mut training = tiny_training_hparams();
        training.events.flush_every_steps = 1;
        let handles = crate::train::events::build_training_event_handles(
            "ruliad-correctness-metric-smoke",
            &run_dir,
            1,
            &training,
            None,
            None,
            None,
        )
        .expect("event handles");
        let bus = handles.metric_logger.bus();
        let report = burn_dragon_universality::RuliadEvalReport {
            version: burn_dragon_universality::ruliad::RULIAD_EVAL_REPORT_VERSION,
            reasoning_score_version:
                burn_dragon_universality::ruliad::RULIAD_REASONING_SCORE_VERSION,
            dataset_name: "test".to_string(),
            item_count: 4,
            scored_count: 4,
            exact_match_count: 1,
            semantic_match_count: 2,
            verifier_match_count: 2,
            partial_credit_count: 3,
            schema_valid_wrong_count: 1,
            malformed_completion_count: 1,
            missing_completion_count: 0,
            unexpected_completion_count: 0,
            exact_accuracy: 0.25,
            semantic_accuracy: 0.5,
            verifier_accuracy: 0.5,
            partial_credit_rate: 0.75,
            mean_partial_progress: 0.625,
            answer_field_correct_count: 5,
            answer_field_expected_count: 8,
            answer_field_accuracy: 0.625,
            answer_terminated_count: 3,
            answer_termination_rate: 0.75,
            mean_certificate_prefix_coverage: 0.5,
            mean_completion_tokens: 12.0,
            canary_count: 0,
            canary_semantic_match_count: 0,
            family_scores: Vec::new(),
            task_scores: Vec::new(),
            difficulty_scores: vec![burn_dragon_universality::RuliadEvalGroupScore {
                label: "d7".to_string(),
                count: 4,
                exact_match_count: 1,
                semantic_match_count: 2,
                verifier_match_count: 2,
                partial_credit_count: 3,
                schema_valid_wrong_count: 1,
                malformed_completion_count: 1,
                missing_completion_count: 0,
                exact_accuracy: 0.25,
                semantic_accuracy: 0.5,
                verifier_accuracy: 0.5,
                partial_credit_rate: 0.75,
                mean_partial_progress: 0.625,
                answer_field_correct_count: 5,
                answer_field_expected_count: 8,
                answer_field_accuracy: 0.625,
                answer_terminated_count: 3,
                answer_termination_rate: 0.75,
            }],
            math_domain_scores: Vec::new(),
            reasoning_mode_scores: Vec::new(),
            failures: Vec::new(),
        };
        emit_ruliad_correctness_metrics("ruliad-correctness-metric-smoke", 3, 17, &report, &bus);
        let _ = bus.flush();
        drop(handles);

        let events = read_training_events(&run_dir);
        let metric_value = |name: &str| {
            events
                .iter()
                .find(|event| {
                    event.get("type").and_then(|value| value.as_str()) == Some("metric")
                        && event.get("split").and_then(|value| value.as_str()) == Some("valid")
                        && event.get("name").and_then(|value| value.as_str()) == Some(name)
                })
                .and_then(|event| event.get("value"))
                .and_then(|value| value.as_f64())
                .unwrap_or_else(|| panic!("missing metric {name}"))
        };
        assert_eq!(metric_value("Ruliad Eval Items"), 4.0);
        assert_eq!(metric_value("Ruliad Verifier Accuracy"), 0.5);
        assert_eq!(metric_value("Ruliad Competence Verifier PPM"), 500_000.0);
        assert_eq!(
            metric_value("Ruliad Competence Completion Health PPM"),
            500_000.0
        );
        assert_eq!(metric_value("Ruliad Answer Field Accuracy"), 0.625);
        assert_eq!(metric_value("Ruliad Answer Termination Rate"), 0.75);
        assert_eq!(metric_value("Ruliad Malformed Completion Rate"), 0.25);
        let capability = events
            .iter()
            .find(|event| {
                event.get("type").and_then(|value| value.as_str()) == Some("capability_probe")
            })
            .expect("capability probe event");
        assert_eq!(
            capability
                .get("probe_name")
                .and_then(|value| value.as_str()),
            Some("ruliad_correctness")
        );
        assert_eq!(
            capability
                .get("achieved_difficulty_level")
                .and_then(|value| value.as_u64()),
            Some(7)
        );
        assert_eq!(
            capability
                .get("verifier_rate")
                .and_then(|value| value.as_f64()),
            Some(0.5)
        );
        assert_eq!(
            capability
                .get("answer_field_accuracy")
                .and_then(|value| value.as_f64()),
            Some(0.625)
        );
        assert_eq!(
            capability
                .get("answer_termination_rate")
                .and_then(|value| value.as_f64()),
            Some(0.75)
        );
        let capability_jsonl =
            std::fs::read_to_string(run_dir.join("events/capability_probe.jsonl"))
                .expect("capability probe jsonl");
        assert!(capability_jsonl.contains("\"probe_name\":\"ruliad_correctness\""));
    }

    #[test]
    fn ruliad_probe_examples_capture_mismatched_completion() {
        let item = burn_dragon_universality::RuliadEvalItem {
            oracle_hash: "hash-a".to_string(),
            sample_index: 7,
            split: burn_dragon_universality::SampleSplit::Validation,
            family: "proof_tree".to_string(),
            task_kind: "prove_theorem".to_string(),
            math_domains: vec!["category_theory".to_string()],
            reasoning_modes: vec!["equational_reasoning".to_string()],
            prompt: "[R2 hash-a v1 P/thm/proof]\nA:ok,l,r\n!:".to_string(),
            expected_answer: "ok=1;l=2;r=2".to_string(),
            difficulty_level: Some(3),
            spec: None,
        };
        let completion = burn_dragon_universality::RuliadCompletionRecord {
            oracle_hash: "hash-a".to_string(),
            completion: "!:ok=0;l=2;r=9\n[/R2]\n".to_string(),
        };

        let examples = ruliad_probe_examples(&[item], &[completion], 4);

        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].label, "proof_tree:prove_theorem");
        assert_eq!(examples[0].expected, "ok=1;l=2;r=2");
        assert_eq!(examples[0].actual.as_deref(), Some("ok=0;l=2;r=9"));
        assert_eq!(examples[0].status, "Partial");
        assert_eq!(examples[0].reason, "answer_mismatch");
        assert!(examples[0].prompt.contains("\\nA:ok,l,r\\n!:"));
        assert_eq!(examples[0].generated_tokens, 2);
    }

    #[test]
    fn ruliad_completion_degeneracy_summary_tracks_periodic_answers() {
        let summary =
            ruliad_completion_degeneracy_summary(&[vec![1, 2, 1, 2, 1, 2], vec![3, 4, 5, 6]], None)
                .expect("summary");

        assert_eq!(summary.sequence_count, 2);
        assert_eq!(summary.token_count, 10);
        assert!(summary.distinct_2_fraction < 1.0);
        assert_eq!(summary.dominant_period_2_to_64, 2);
        assert!(
            summary.max_period_2_to_64_fraction > 0.5,
            "{}",
            summary.max_period_2_to_64_fraction
        );
    }

    #[test]
    fn ruliad_completion_degeneracy_summary_trims_after_close_token() {
        let summary = ruliad_completion_degeneracy_summary(
            &[vec![10, 11, 99, 7, 7, 7, 7, 7], vec![12, 13, 99, 8, 8, 8]],
            Some(99),
        )
        .expect("summary");

        assert_eq!(summary.sequence_count, 2);
        assert_eq!(summary.token_count, 6);
        assert!(summary.repetition_fraction < 0.1, "{summary:?}");
        assert!(summary.max_period_2_to_64_fraction < 0.1, "{summary:?}");
    }

    #[test]
    fn ruliad_capability_gate_metrics_emit_failure_state() {
        let dir = tempfile::tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let mut training = tiny_training_hparams();
        training.events.flush_every_steps = 1;
        training.gates.capability_schema_wrong_max_rate = 0.25;
        training.gates.capability_malformed_max_rate = 0.02;
        training.gates.capability_completion_health_min_rate = 0.80;
        training.gates.capability_output_entropy_min_bits = 1.25;
        training.gates.capability_distinct_2_min_fraction = 0.30;
        let handles = crate::train::events::build_training_event_handles(
            "ruliad-capability-gate-metric-smoke",
            &run_dir,
            1,
            &training,
            None,
            None,
            None,
        )
        .expect("event handles");
        let bus = handles.metric_logger.bus();
        let mut report = ruliad_eval_report(0.25, 0.25, 0.25, 0.25);
        report.schema_valid_wrong_count = 40;
        report.malformed_completion_count = 5;
        let stats = degeneracy_stats(0.5, 0.9, 0.1, 0.0);
        emit_ruliad_capability_gate_metrics(
            "ruliad-capability-gate-metric-smoke",
            4,
            19,
            &report,
            Some(&stats),
            &training.gates,
            &bus,
        );
        let _ = bus.flush();
        drop(handles);

        let events = read_training_events(&run_dir);
        let metric_value = |name: &str| {
            events
                .iter()
                .find(|event| {
                    event.get("type").and_then(|value| value.as_str()) == Some("metric")
                        && event.get("split").and_then(|value| value.as_str()) == Some("valid")
                        && event.get("name").and_then(|value| value.as_str()) == Some(name)
                })
                .and_then(|event| event.get("value"))
                .and_then(|value| value.as_f64())
                .unwrap_or_else(|| panic!("missing metric {name}"))
        };
        assert_eq!(metric_value("Ruliad Capability Gate Passed"), 0.0);
        assert!(metric_value("Ruliad Capability Gate Failure Count") >= 4.0);
        assert!(events.iter().any(|event| {
            event.get("type").and_then(|value| value.as_str()) == Some("gate")
                && event.get("gate").and_then(|value| value.as_str())
                    == Some("ruliad_capability_gate_failed")
        }));
    }

    #[test]
    fn latent_eval_step_sweep_sorts_and_deduplicates_steps() {
        let mut training = tiny_training_hparams();
        training.latent_reasoning.eval_step_sweep = vec![8, 1, 4, 1, 2];

        assert_eq!(latent_eval_step_sweep(&training), vec![1, 2, 4, 8]);
    }

    #[test]
    fn ruliad_correctness_eval_step_metrics_use_distinct_probe_name() {
        let dir = tempfile::tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let mut training = tiny_training_hparams();
        training.events.flush_every_steps = 1;
        let handles = crate::train::events::build_training_event_handles(
            "ruliad-eval-step-metric-smoke",
            &run_dir,
            1,
            &training,
            None,
            None,
            None,
        )
        .expect("event handles");
        let bus = handles.metric_logger.bus();
        let report = burn_dragon_universality::RuliadEvalReport {
            version: burn_dragon_universality::ruliad::RULIAD_EVAL_REPORT_VERSION,
            reasoning_score_version:
                burn_dragon_universality::ruliad::RULIAD_REASONING_SCORE_VERSION,
            dataset_name: "test".to_string(),
            item_count: 2,
            scored_count: 2,
            exact_match_count: 0,
            semantic_match_count: 1,
            verifier_match_count: 1,
            partial_credit_count: 1,
            schema_valid_wrong_count: 0,
            malformed_completion_count: 0,
            missing_completion_count: 0,
            unexpected_completion_count: 0,
            exact_accuracy: 0.0,
            semantic_accuracy: 0.5,
            verifier_accuracy: 0.5,
            partial_credit_rate: 0.5,
            mean_partial_progress: 0.5,
            answer_field_correct_count: 1,
            answer_field_expected_count: 2,
            answer_field_accuracy: 0.5,
            answer_terminated_count: 2,
            answer_termination_rate: 1.0,
            mean_certificate_prefix_coverage: 0.5,
            mean_completion_tokens: 8.0,
            canary_count: 0,
            canary_semantic_match_count: 0,
            family_scores: Vec::new(),
            task_scores: Vec::new(),
            difficulty_scores: Vec::new(),
            math_domain_scores: Vec::new(),
            reasoning_mode_scores: Vec::new(),
            failures: Vec::new(),
        };
        emit_ruliad_correctness_metrics_with_labels(
            "ruliad-eval-step-metric-smoke",
            2,
            32,
            &report,
            &bus,
            "ruliad_correctness_eval_steps_8",
            Some("Ruliad Eval Steps 8"),
            None,
            &[],
            RuliadAnswerSchemaAlignmentSummary::default(),
            None,
        );
        let _ = bus.flush();
        drop(handles);

        let events = read_training_events(&run_dir);
        assert!(events.iter().any(|event| {
            event.get("type").and_then(|value| value.as_str()) == Some("metric")
                && event.get("name").and_then(|value| value.as_str())
                    == Some("Ruliad Eval Steps 8 Ruliad Verifier Accuracy")
        }));
        assert!(events.iter().any(|event| {
            event.get("type").and_then(|value| value.as_str()) == Some("capability_probe")
                && event.get("probe_name").and_then(|value| value.as_str())
                    == Some("ruliad_correctness_eval_steps_8")
        }));
    }

    #[test]
    fn file_metric_best_strategy_tracks_best_value() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut strategy = FileMetricBestCheckpointingStrategy::new(
            dir.path(),
            &LossMetric::<TestValidBackend>::new(),
            burn_train::metric::store::Direction::Lowest,
            burn_train::metric::store::Split::Valid,
        );

        let previous_best = strategy.update_best_candidate(1, 3.5);

        assert_eq!(previous_best, None);
        assert_eq!(strategy.best_epoch, Some(1));
        assert_eq!(strategy.best_value, Some(3.5));
    }

    #[test]
    fn file_metric_best_strategy_replaces_only_on_improvement() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut strategy = FileMetricBestCheckpointingStrategy::new(
            dir.path(),
            &LossMetric::<TestValidBackend>::new(),
            burn_train::metric::store::Direction::Lowest,
            burn_train::metric::store::Split::Valid,
        );
        strategy.best_epoch = Some(2);
        strategy.best_value = Some(3.2);

        let worse_previous_best = strategy.update_best_candidate(3, 3.3);
        assert_eq!(worse_previous_best, None);
        assert_eq!(strategy.best_epoch, Some(2));
        assert_eq!(strategy.best_value, Some(3.2));

        let better_previous_best = strategy.update_best_candidate(4, 3.1);
        assert_eq!(better_previous_best, Some(2));
        assert_eq!(strategy.best_epoch, Some(4));
        assert_eq!(strategy.best_value, Some(3.1));
    }

    fn write_metric_log(run_dir: &Path, split: &str, epoch: usize, values: &[f64]) {
        let epoch_dir = run_dir.join(split).join(format!("epoch-{epoch}"));
        fs::create_dir_all(&epoch_dir).expect("create epoch dir");
        let path = epoch_dir.join("Loss.log");
        let content = values
            .iter()
            .map(|value| format!("{value},1"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(path, content).expect("write metric log");
    }

    fn apply_checkpoint_actions(run_dir: &Path, epoch: usize, actions: &[CheckpointingAction]) {
        let checkpoint_dir = run_dir.join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).expect("create checkpoint dir");
        for action in actions {
            match action {
                CheckpointingAction::Save => {
                    for prefix in ["model", "optim", "scheduler"] {
                        fs::write(
                            checkpoint_dir.join(format!("{prefix}-{epoch}.bin")),
                            format!("{prefix}-{epoch}"),
                        )
                        .expect("write checkpoint file");
                    }
                }
                CheckpointingAction::Delete(epoch) => {
                    for prefix in ["model", "optim", "scheduler"] {
                        let path = checkpoint_dir.join(format!("{prefix}-{epoch}.bin"));
                        if path.exists() {
                            fs::remove_file(path).expect("remove checkpoint file");
                        }
                    }
                }
            }
        }
    }

    fn retained_model_epochs(run_dir: &Path) -> Vec<usize> {
        let checkpoint_dir = run_dir.join("checkpoint");
        let mut epochs = fs::read_dir(&checkpoint_dir)
            .expect("read checkpoint dir")
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                let name = path.file_name()?.to_str()?;
                let epoch = name
                    .strip_prefix("model-")?
                    .strip_suffix(".bin")?
                    .parse::<usize>()
                    .ok()?;
                Some(epoch)
            })
            .collect::<Vec<_>>();
        epochs.sort_unstable();
        epochs
    }

    fn write_dynamic_checkpoint_bundle(checkpoint_dir: &Path, epoch: usize) {
        for prefix in ["model", "optimizer", "scheduler"] {
            fs::write(
                checkpoint_dir.join(format!("{prefix}-{epoch}.bin")),
                format!("{prefix}-{epoch}"),
            )
            .expect("write dynamic checkpoint record");
        }
        for prefix in ["dynamics", "model-config"] {
            fs::write(
                checkpoint_dir.join(format!("{prefix}-{epoch}.json")),
                format!(r#"{{"epoch":{epoch}}}"#),
            )
            .expect("write dynamic checkpoint json");
        }
        fs::write(
            checkpoint_dir.join(format!("source-selection-state-{epoch}.json")),
            format!(r#"{{"epoch":{epoch}}}"#),
        )
        .expect("write source-selection state checkpoint");
    }

    fn append_validation_event(run_dir: &Path, epoch: usize, loss: f64) {
        let events_dir = run_dir.join("events");
        fs::create_dir_all(&events_dir).expect("create events dir");
        let path = events_dir.join("training_events.jsonl");
        let mut content = fs::read_to_string(&path).unwrap_or_default();
        content.push_str(&format!(
            r#"{{"type":"validation_finished","run_id":"test","epoch":{epoch},"absolute_step":{epoch},"loss":{loss}}}"#
        ));
        content.push('\n');
        fs::write(path, content).expect("append validation event");
    }

    #[test]
    fn historical_best_validation_recovers_loss_and_available_checkpoint_epoch() {
        let dir = tempfile::tempdir().expect("tempdir");
        let checkpoint_dir = dir.path().join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).expect("checkpoint dir");
        for epoch in [9, 10] {
            fs::write(
                checkpoint_dir.join(format!("model-{epoch}.bin")),
                format!("model-{epoch}"),
            )
            .expect("write checkpoint");
        }
        append_validation_event(dir.path(), 8, 0.789);
        append_validation_event(dir.path(), 9, 0.797);
        append_validation_event(dir.path(), 10, 0.821);
        append_validation_event(dir.path(), 11, 0.700);

        let historical = historical_best_validation(dir.path(), 10);

        assert_eq!(
            historical,
            HistoricalBestValidation {
                best_loss: Some(0.789),
                best_checkpoint_epoch: Some(9),
            }
        );
    }

    #[test]
    fn historical_best_validation_keeps_true_best_checkpoint_when_present() {
        let dir = tempfile::tempdir().expect("tempdir");
        let checkpoint_dir = dir.path().join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).expect("checkpoint dir");
        for epoch in [8, 9, 10] {
            fs::write(
                checkpoint_dir.join(format!("model-{epoch}.bin")),
                format!("model-{epoch}"),
            )
            .expect("write checkpoint");
        }
        append_validation_event(dir.path(), 8, 0.789);
        append_validation_event(dir.path(), 9, 0.797);
        append_validation_event(dir.path(), 10, 0.821);

        let historical = historical_best_validation(dir.path(), 10);

        assert_eq!(
            historical,
            HistoricalBestValidation {
                best_loss: Some(0.789),
                best_checkpoint_epoch: Some(8),
            }
        );
    }

    #[test]
    fn file_metric_best_strategy_preserves_old_best_outside_keep_last_window() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut strategy = FileMetricBestCheckpointingStrategy::new(
            dir.path(),
            &LossMetric::<TestValidBackend>::new(),
            burn_train::metric::store::Direction::Lowest,
            burn_train::metric::store::Split::Valid,
        );

        let means = [
            2.0, 1.9, 1.8, 1.7, 1.6, 1.55, 1.53, 1.52, 1.515, 1.51, 1.509, 1.508, 1.507, 1.506,
            1.505, 1.504, 1.503, 1.502, 1.497, 1.501, 1.510, 1.512, 1.511, 1.499, 1.513, 1.514,
            1.502, 1.520, 1.506, 1.530,
        ];

        for (index, mean) in means.iter().enumerate() {
            let epoch = index + 1;
            write_metric_log(dir.path(), "valid", epoch, &[*mean]);
            let actions = strategy.actions_for_epoch(epoch);
            apply_checkpoint_actions(dir.path(), epoch, &actions);
        }

        assert_eq!(strategy.best_epoch, Some(19));
        assert_eq!(retained_model_epochs(dir.path()), vec![19, 29, 30]);
    }

    #[test]
    fn dynamic_scheduler_checkpoint_pruning_keeps_recent_and_best() {
        let dir = tempfile::tempdir().expect("tempdir");
        let checkpoint_dir = dir.path().join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).expect("checkpoint dir");
        for epoch in 1..=5 {
            write_dynamic_checkpoint_bundle(&checkpoint_dir, epoch);
        }

        prune_dragon_model_checkpoints(dir.path(), 5, Some(2)).expect("prune checkpoints");

        assert_eq!(retained_model_epochs(dir.path()), vec![2, 4, 5]);
        for kept_epoch in [2, 4, 5] {
            for file in [
                format!("model-{kept_epoch}.bin"),
                format!("optimizer-{kept_epoch}.bin"),
                format!("scheduler-{kept_epoch}.bin"),
                format!("dynamics-{kept_epoch}.json"),
                format!("model-config-{kept_epoch}.json"),
                format!("source-selection-state-{kept_epoch}.json"),
            ] {
                assert!(
                    checkpoint_dir.join(file).is_file(),
                    "expected checkpoint bundle artifact for kept epoch {kept_epoch}"
                );
            }
        }
        for pruned_epoch in [1, 3] {
            for file in [
                format!("model-{pruned_epoch}.bin"),
                format!("optimizer-{pruned_epoch}.bin"),
                format!("scheduler-{pruned_epoch}.bin"),
                format!("dynamics-{pruned_epoch}.json"),
                format!("model-config-{pruned_epoch}.json"),
                format!("source-selection-state-{pruned_epoch}.json"),
            ] {
                assert!(
                    !checkpoint_dir.join(file).exists(),
                    "expected checkpoint bundle artifact to be pruned for epoch {pruned_epoch}"
                );
            }
        }
    }

    #[test]
    fn file_metric_best_strategy_deletes_old_best_after_replacement() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut strategy = FileMetricBestCheckpointingStrategy::new(
            dir.path(),
            &LossMetric::<TestValidBackend>::new(),
            burn_train::metric::store::Direction::Lowest,
            burn_train::metric::store::Split::Valid,
        );

        for (epoch, mean) in [(1, 3.0), (2, 2.0), (3, 2.5), (4, 1.5)] {
            write_metric_log(dir.path(), "valid", epoch, &[mean]);
            let actions = strategy.actions_for_epoch(epoch);
            apply_checkpoint_actions(dir.path(), epoch, &actions);
        }

        assert_eq!(strategy.best_epoch, Some(4));
        assert_eq!(retained_model_epochs(dir.path()), vec![3, 4]);
    }

    #[test]
    fn file_metric_best_strategy_rehydrates_history_when_resuming() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut strategy = FileMetricBestCheckpointingStrategy::new(
            dir.path(),
            &LossMetric::<TestValidBackend>::new(),
            burn_train::metric::store::Direction::Lowest,
            burn_train::metric::store::Split::Valid,
        );

        for (epoch, mean) in [(1, 3.0), (2, 1.5), (3, 2.0), (4, 2.1), (5, 2.2), (6, 2.3)] {
            write_metric_log(dir.path(), "valid", epoch, &[mean]);
        }
        for epoch in [2, 5, 6] {
            apply_checkpoint_actions(dir.path(), epoch, &[CheckpointingAction::Save]);
        }

        write_metric_log(dir.path(), "valid", 7, &[2.4]);
        let actions = strategy.actions_for_epoch(7);
        apply_checkpoint_actions(dir.path(), 7, &actions);

        assert_eq!(strategy.best_epoch, Some(2));
        assert_eq!(retained_model_epochs(dir.path()), vec![2, 6, 7]);
    }

    #[test]
    fn file_metric_best_strategy_recomputes_history_when_new_best_log_arrives_late() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut strategy = FileMetricBestCheckpointingStrategy::new(
            dir.path(),
            &LossMetric::<TestValidBackend>::new(),
            burn_train::metric::store::Direction::Lowest,
            burn_train::metric::store::Split::Valid,
        );

        for epoch in 1..=23 {
            let mean = if epoch == 23 {
                1.50
            } else {
                2.0 + epoch as f64 * 0.01
            };
            write_metric_log(dir.path(), "valid", epoch, &[mean]);
            let actions = strategy.actions_for_epoch(epoch);
            apply_checkpoint_actions(dir.path(), epoch, &actions);
        }

        for epoch in 24..=28 {
            write_metric_log(dir.path(), "valid", epoch, &[1.60 + epoch as f64 * 0.001]);
            let actions = strategy.actions_for_epoch(epoch);
            apply_checkpoint_actions(dir.path(), epoch, &actions);
        }

        let actions = strategy.actions_for_epoch(29);
        apply_checkpoint_actions(dir.path(), 29, &actions);
        write_metric_log(dir.path(), "valid", 29, &[1.48]);

        write_metric_log(dir.path(), "valid", 30, &[1.49]);
        let actions = strategy.actions_for_epoch(30);
        apply_checkpoint_actions(dir.path(), 30, &actions);

        assert_eq!(strategy.best_epoch, Some(29));
        assert_eq!(retained_model_epochs(dir.path()), vec![29, 30]);
    }

    #[derive(Clone)]
    struct StaticSequenceLoader<B: BackendTrait> {
        items: Vec<SequenceBatch<B>>,
    }

    impl<B: BackendTrait> StaticSequenceLoader<B> {
        fn new(items: Vec<SequenceBatch<B>>) -> Self {
            Self { items }
        }
    }

    struct StaticSequenceIterator<B: BackendTrait> {
        items: Vec<SequenceBatch<B>>,
        index: usize,
    }

    impl<B: BackendTrait> Iterator for StaticSequenceIterator<B> {
        type Item = SequenceBatch<B>;

        fn next(&mut self) -> Option<Self::Item> {
            let item = self.items.get(self.index).cloned();
            if item.is_some() {
                self.index += 1;
            }
            item
        }
    }

    impl<B: BackendTrait> DataLoaderIterator<SequenceBatch<B>> for StaticSequenceIterator<B> {
        fn progress(&self) -> Progress {
            Progress::new(self.index, self.items.len())
        }
    }

    impl<B> DataLoader<B, SequenceBatch<B>> for StaticSequenceLoader<B>
    where
        B: BackendTrait + 'static,
    {
        fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<SequenceBatch<B>> + 'a> {
            Box::new(StaticSequenceIterator {
                items: self.items.clone(),
                index: 0,
            })
        }

        fn num_items(&self) -> usize {
            self.items.len()
        }

        fn to_device(&self, _device: &B::Device) -> Arc<dyn DataLoader<B, SequenceBatch<B>>> {
            Arc::new(self.clone())
        }

        fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, SequenceBatch<B>>> {
            let len = self.items.len();
            let start = start.min(len);
            let end = end.min(len);
            Arc::new(Self {
                items: self.items[start..end].to_vec(),
            })
        }
    }

    fn make_batch<B: BackendTrait>(
        device: &B::Device,
        inputs: &[i64],
        targets: &[i64],
        shape: [usize; 2],
    ) -> SequenceBatch<B> {
        SequenceBatch::new(
            Tensor::<B, 2, Int>::from_data(TensorData::new(inputs.to_vec(), shape), device),
            Tensor::<B, 2, Int>::from_data(TensorData::new(targets.to_vec(), shape), device),
            None,
        )
    }

    fn tensor_values<B: BackendTrait, const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
        tensor
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("tensor values")
    }

    fn max_abs_diff(lhs: Vec<f32>, rhs: Vec<f32>) -> f32 {
        assert_eq!(lhs.len(), rhs.len(), "tensor length mismatch");
        lhs.into_iter()
            .zip(rhs)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max)
    }

    fn tiny_model_config() -> DragonConfig {
        DragonConfig {
            n_layer: 1,
            n_embd: 8,
            n_head: 1,
            mlp_internal_dim_multiplier: 1,
            dropout: 0.0,
            vocab_size: 16,
            ..Default::default()
        }
    }

    fn tiny_training_hparams() -> TrainingHyperparameters {
        TrainingHyperparameters {
            block_size: 4,
            tbptt_chunk_size: None,
            tbptt_persist_across_steps: false,
            min_logical_block_size: None,
            batch_size: 2,
            seed: 1337,
            gradient_accumulation_steps: 1,
            target_effective_batch_size: None,
            epochs: Some(1),
            max_iters: 2,
            checkpoint_interval_iters: 2000,
            log_frequency: 1,
            launch_mode: burn_dragon_train::train::pipeline::TrainingLaunchMode::Fresh,
            resume_run_dir: None,
            resume_checkpoint_epoch: None,
            init_checkpoint_path: None,
            init_checkpoint_epoch: None,
            source_selection_state_path: None,
            init_transfer: Default::default(),
            continual_backprop: Default::default(),
            input_corruption: Default::default(),
            logit_entropy_floor: Default::default(),
            repeat_unlikelihood: Default::default(),
            greedy_rollout_unlikelihood: Default::default(),
            dynamics_anchor: Default::default(),
            predictive_coding: Default::default(),
            latent_reasoning: Default::default(),
            ruliad_supervision: Default::default(),
            module_lr_scales: Vec::new(),
            context_strategy: ContextStrategyConfig::Infinite,
            sequence_kernel_override: None,
            objective: Default::default(),
            gdpo: None,
            events: Default::default(),
            gates: Default::default(),
            dynamics: Default::default(),
            neuron_scaling: Default::default(),
            auto_batch_size: Default::default(),
        }
    }

    fn tiny_training_hparams_with_epochs(
        epochs: usize,
        resume_checkpoint_epoch: Option<usize>,
    ) -> TrainingHyperparameters {
        let mut training = tiny_training_hparams();
        training.epochs = Some(epochs);
        training.resume_checkpoint_epoch = resume_checkpoint_epoch;
        training
    }

    #[test]
    fn persistent_tbptt_uses_stream_loss_metric_name() {
        let mut training = tiny_training_hparams();
        training.tbptt_persist_across_steps = true;
        training.log_frequency = 7;
        training.events.source_selection_every_steps = 2;

        assert_eq!(train_loss_metric_name(&training), METRIC_STREAM_WARM_LOSS);
        assert_eq!(
            crate::train::events::train_loss_metric_frequency(&training, None),
            7
        );
        assert!(!source_selection_telemetry_due_for(&training, None, 0));
    }

    #[test]
    fn predictive_coding_state_only_control_disables_optimizer_steps() {
        let mut training = tiny_training_hparams();
        assert!(parameter_updates_enabled(&training));

        training.predictive_coding.enabled = true;
        assert!(parameter_updates_enabled(&training));

        training.predictive_coding.parameter_update =
            PredictiveCodingParameterUpdate::StateOnlyControl;
        assert!(!parameter_updates_enabled(&training));
    }

    fn objective_training_hparams(objective: TrainingObjectiveConfig) -> TrainingHyperparameters {
        let mut training = tiny_training_hparams();
        training.objective = objective;
        training
    }

    fn tiny_language_optimizer(
        training: &TrainingHyperparameters,
        model_config: &DragonConfig,
        device: &burn::tensor::Device<TestBackend>,
    ) -> crate::train::continual_backprop::LanguageOptimizer<TestBackend> {
        let optimizer_cfg = OptimizerConfig {
            name: OptimizerKind::Adamw,
            learning_rate: 1e-3,
            weight_decay: 0.0,
            weight_decay_final: None,
            lr_schedule: None,
            schedule_mode: OptimizerScheduleMode::DragonReference,
            grad_clip_norm: None,
            grad_clip_value: None,
            eggroll: burn_eggroll::EggrollConfig::default(),
            eggroll_population_execution: Default::default(),
            eggroll_auto_population: Default::default(),
            predictive_coding: Default::default(),
        };
        let fresh_model = DragonModel::<TestBackend>::new(model_config.clone(), device);
        crate::train::continual_backprop::resolve_dragon_language_optimizer::<TestBackend>(
            training,
            &optimizer_cfg,
            1,
            fresh_model,
        )
        .expect("optimizer")
    }

    #[test]
    fn eggroll_chunk_autotune_candidates_are_even_bounded_and_include_configured() {
        let optimizer_cfg = OptimizerConfig {
            name: OptimizerKind::Eggroll,
            learning_rate: 1e-2,
            weight_decay: 0.0,
            weight_decay_final: None,
            lr_schedule: None,
            schedule_mode: OptimizerScheduleMode::DragonReference,
            grad_clip_norm: None,
            grad_clip_value: None,
            eggroll: burn_eggroll::EggrollConfig {
                population: burn_eggroll::PopulationConfig {
                    population_size: 512,
                    population_chunk_size: 64,
                    rank: 1,
                    seed: 7,
                    matrix_noise: burn_eggroll::MatrixNoiseMode::default(),
                },
                ..burn_eggroll::EggrollConfig::default()
            },
            eggroll_population_execution: Default::default(),
            eggroll_auto_population: burn_dragon_train::EggrollAutoPopulationConfig {
                enabled: true,
                chunk_autotune: burn_dragon_train::EggrollChunkAutotuneConfig {
                    enabled: true,
                    candidates: vec![32, 128, 256],
                    max_probe_population_size: 128,
                },
                ..Default::default()
            },
            predictive_coding: Default::default(),
        };

        let candidates = super::resolve_eggroll_chunk_autotune_candidates(&optimizer_cfg);

        assert_eq!(candidates, vec![32, 64, 128]);
    }

    #[test]
    fn eggroll_population_execution_stacked_tensorized_matches_manual_shared_lowrank_members() {
        let device = burn::tensor::Device::<TestForwardBackend>::default();
        TestForwardBackend::seed(&device, 17);
        let model = LanguageTrainModel::new(DragonModel::<TestForwardBackend>::new(
            tiny_model_config(),
            &device,
        ));
        let batch = make_batch::<TestForwardBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 8],
            [2, 4],
        );
        let eggroll = burn_eggroll::EggrollConfig {
            sigma: 1.0e-3,
            population: burn_eggroll::PopulationConfig {
                population_size: 4,
                population_chunk_size: 4,
                rank: 2,
                seed: 99,
                matrix_noise: burn_eggroll::MatrixNoiseMode::default(),
            },
            ..burn_eggroll::EggrollConfig::default()
        };
        let optimizer_cfg = OptimizerConfig {
            name: OptimizerKind::Eggroll,
            learning_rate: 1.0e-4,
            weight_decay: 0.0,
            weight_decay_final: None,
            lr_schedule: None,
            schedule_mode: OptimizerScheduleMode::DragonReference,
            grad_clip_norm: None,
            grad_clip_value: None,
            eggroll: eggroll.clone(),
            eggroll_population_execution: burn_dragon_train::EggrollPopulationExecutionConfig {
                perturbation_scope: EggrollPerturbationScope::DragonCoreProjection,
                backend: EggrollPopulationExecutionBackend::Reference,
                population_tile_size: None,
            },
            eggroll_auto_population: Default::default(),
            predictive_coding: Default::default(),
        };
        let plan = resolve_eggroll_population_execution_plan(&optimizer_cfg, &model)
            .expect("stacked tensorized plan");
        assert_eq!(plan.executor_name(), "stacked_tensorized");

        let pair_count = 2;
        let lowrank = build_shared_lowrank_population_weights(&model, &eggroll, 3, 0, pair_count);
        let lowrank_factors =
            build_shared_lowrank_population_factors(&model, &eggroll, 3, 0, pair_count);
        let base_weights = model.model.shared_lowrank_weights();
        let stacked_logits = model
            .model
            .forward_with_shared_lowrank_population(batch.inputs.clone(), lowrank.clone());
        let factorized_logits = model.model.forward_with_shared_lowrank_population_factors(
            batch.inputs.clone(),
            lowrank_factors.clone(),
        );
        let factorized_logit_diff = max_abs_diff(
            tensor_values(stacked_logits.clone()),
            tensor_values(factorized_logits.clone()),
        );
        assert!(
            factorized_logit_diff <= 1.0e-4,
            "materialized and factorized population logits drifted by {factorized_logit_diff}"
        );
        let [base_batch, _time] = batch.inputs.shape().dims::<2>();
        let mut manual_losses = Vec::with_capacity(pair_count * 2);
        for member in 0..pair_count * 2 {
            let member_weights = SharedLowrankWeights {
                encoder: lowrank
                    .encoder
                    .clone()
                    .slice_dim(0, member..member + 1)
                    .reshape(base_weights.encoder.shape().dims::<3>()),
                encoder_v: lowrank
                    .encoder_v
                    .clone()
                    .slice_dim(0, member..member + 1)
                    .reshape(base_weights.encoder_v.shape().dims::<3>()),
                decoder: lowrank
                    .decoder
                    .clone()
                    .slice_dim(0, member..member + 1)
                    .reshape(base_weights.decoder.shape().dims::<2>()),
            };
            let manual_model = model
                .clone()
                .map_model(|dragon| dragon.with_shared_lowrank_weights(member_weights));
            let manual_logits = manual_model.model.forward(batch.inputs.clone());
            let single_lowrank = SharedLowrankPopulationWeights {
                encoder: lowrank.encoder.clone().slice_dim(0, member..member + 1),
                encoder_v: lowrank.encoder_v.clone().slice_dim(0, member..member + 1),
                decoder: lowrank.decoder.clone().slice_dim(0, member..member + 1),
            };
            let single_logits = model
                .model
                .forward_with_shared_lowrank_population(batch.inputs.clone(), single_lowrank);
            let single_logit_diff = max_abs_diff(
                tensor_values(manual_logits.clone()),
                tensor_values(single_logits),
            );
            assert!(
                single_logit_diff <= 1.0e-5,
                "manual and single tensorized member {member} logits drifted by {single_logit_diff}"
            );
            let stacked_member_logits = stacked_logits
                .clone()
                .slice_dim(0, member * base_batch..(member + 1) * base_batch);
            let stacked_logit_diff = max_abs_diff(
                tensor_values(manual_logits.clone()),
                tensor_values(stacked_member_logits),
            );
            assert!(
                stacked_logit_diff <= 1.0e-5,
                "manual and stacked tensorized member {member} logits drifted by {stacked_logit_diff}"
            );
            let factorized_member_logits = factorized_logits
                .clone()
                .slice_dim(0, member * base_batch..(member + 1) * base_batch);
            let factorized_member_diff = max_abs_diff(
                tensor_values(manual_logits),
                tensor_values(factorized_member_logits),
            );
            assert!(
                factorized_member_diff <= 1.0e-4,
                "manual and factorized member {member} logits drifted by {factorized_member_diff}"
            );
            manual_losses.push(eggroll_batch_loss_tensor(&manual_model, batch.clone()));
        }
        let stacked_tensorized = evaluate_eggroll_population_chunk(
            &plan,
            &model,
            batch.clone(),
            &eggroll,
            3,
            0,
            pair_count,
        )
        .expect("stacked tensorized losses");
        let factorized_optimizer_cfg = OptimizerConfig {
            eggroll_population_execution: burn_dragon_train::EggrollPopulationExecutionConfig {
                backend: EggrollPopulationExecutionBackend::Factorized,
                perturbation_scope: EggrollPerturbationScope::DragonCoreProjection,
                population_tile_size: None,
            },
            ..optimizer_cfg
        };
        let factorized_plan =
            resolve_eggroll_population_execution_plan(&factorized_optimizer_cfg, &model)
                .expect("factorized tensorized plan");
        assert_eq!(factorized_plan.executor_name(), "factorized_tensorized");
        let factorized_tensorized = evaluate_eggroll_population_chunk(
            &factorized_plan,
            &model,
            batch,
            &eggroll,
            3,
            0,
            pair_count,
        )
        .expect("factorized tensorized losses");
        let manual = scalar_values_from_loss_tensors(manual_losses);

        assert_eq!(stacked_tensorized.len(), manual.len());
        assert_eq!(factorized_tensorized.len(), manual.len());
        assert!(stacked_tensorized.iter().all(|loss| loss.is_finite()));
        assert!(factorized_tensorized.iter().all(|loss| loss.is_finite()));
        for ((expected, actual), factorized) in manual
            .iter()
            .zip(stacked_tensorized.iter())
            .zip(factorized_tensorized.iter())
        {
            assert!(
                (expected - actual).abs() <= 1.0e-4,
                "manual={expected} stacked={actual}"
            );
            assert!(
                (expected - factorized).abs() <= 1.0e-4,
                "manual={expected} factorized={factorized}"
            );
        }
    }

    #[test]
    fn eggroll_training_dynamics_are_bounded_against_adamw() {
        let dir = tempfile::tempdir().expect("tempdir");
        let report = crate::train::optimizer_dynamics::run_optimizer_dynamics_suite(
            &crate::train::optimizer_dynamics::OptimizerDynamicsConfig::default(),
            &[17, 29, 53],
            dir.path(),
        )
        .expect("optimizer dynamics suite");

        eprintln!("optimizer dynamics suite: {report:#?}");
        for pair in &report.pairs {
            assert!(pair.adamw.initial_train_loss.is_finite());
            assert!(pair.adamw.final_train_loss.is_finite());
            assert!(pair.adamw.initial_loss.is_finite());
            assert!(pair.adamw.final_loss.is_finite());
            assert!(pair.eggroll.initial_train_loss.is_finite());
            assert!(pair.eggroll.final_train_loss.is_finite());
            assert!(pair.eggroll.initial_loss.is_finite());
            assert!(pair.eggroll.final_loss.is_finite());
            assert!(
                pair.adamw.loss_delta() > 0.0,
                "AdamW should learn the deterministic comparison task: {pair:?}"
            );
            assert!(
                pair.eggroll.train_loss_delta() > 0.0,
                "EGGROLL should reduce train loss in the deterministic comparison task: {pair:?}"
            );
            assert!(
                pair.eggroll.final_loss <= pair.eggroll.initial_loss + 0.05,
                "EGGROLL should not severely regress the deterministic comparison task: {pair:?}"
            );
            assert!(
                pair.eggroll.evaluations_per_second() >= pair.adamw.evaluations_per_second() * 0.02,
                "EGGROLL eval throughput is pathologically low: {pair:?}"
            );
        }
        let min_mean_eggroll_loss_delta = 0.01;
        assert!(
            report.mean_eggroll_loss_delta() > min_mean_eggroll_loss_delta,
            "tensorized EGGROLL should learn a positive average signal: {report:#?}"
        );
        assert!(
            report.mean_eggroll_train_loss_delta() > 0.03,
            "tensorized EGGROLL should reduce train loss by a measurable average signal: {report:#?}"
        );
        let min_adamw_fraction = 0.005;
        assert!(
            report.mean_eggroll_loss_delta() >= report.mean_adamw_loss_delta() * min_adamw_fraction,
            "tensorized EGGROLL should retain a positive fraction of AdamW quality on the deterministic comparison task: {report:#?}"
        );
    }

    #[test]
    fn manual_adamw_loop_matches_learner_dynamics() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = crate::train::optimizer_dynamics::OptimizerDynamicsConfig {
            epochs: 4,
            max_iters: 16,
            log_frequency: 4,
            seed: 29,
            ..crate::train::optimizer_dynamics::OptimizerDynamicsConfig::default()
        };
        let learner = crate::train::optimizer_dynamics::run_optimizer_dynamics(
            crate::train::optimizer_dynamics::OptimizerDynamicsKind::AdamW,
            &config,
            &dir.path().join("learner"),
        )
        .expect("learner adamw dynamics");
        let manual = crate::train::optimizer_dynamics::run_manual_adamw_optimizer_dynamics(&config)
            .expect("manual adamw dynamics");

        eprintln!("adamw learner={learner:#?} manual={manual:#?}");
        assert!(
            manual.loss_delta() > 0.0,
            "manual AdamW loop should learn the deterministic comparison task: {manual:?}"
        );
        assert!(
            manual.loss_delta() >= learner.loss_delta() * 0.60,
            "manual AdamW loop should be in the same quality regime as burn_train::Learner: learner={learner:?} manual={manual:?}"
        );
    }

    #[test]
    fn eggroll_update_preserves_train_step_gradients() {
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 29);
        let model_config = tiny_model_config();
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            model_config.clone(),
            &device,
        ));
        let batch = make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 8],
            [2, 4],
        );
        let eggroll = burn_eggroll::EggrollConfig {
            sigma: 0.0025,
            update: burn_eggroll::EggrollUpdateConfig {
                learning_rate: 1.0e-8,
                ..burn_eggroll::EggrollUpdateConfig::default()
            },
            ..burn_eggroll::EggrollConfig::default()
        };
        let mut eggroll_state =
            burn_dragon_eggroll::EggrollModuleOptimizerState::<TestBackend>::new();
        let (updated, _metrics) = burn_dragon_eggroll::apply_antithetic_update(
            model,
            &eggroll,
            0,
            &[burn_dragon_eggroll::AntitheticFitness {
                pair_index: 0,
                plus: 0.0,
                minus: 1.0,
            }],
            &mut eggroll_state,
        )
        .expect("eggroll update");
        let item = burn_train::TrainStep::step(&updated, batch);
        let raw_gradient_count = item.grads.len();
        let mut accumulator = GradientsAccumulator::new();
        accumulator.accumulate(&updated, item.grads);
        let grads = accumulator.grads();
        let accumulated_gradient_count = grads.len();

        eprintln!(
            "eggroll-updated gradient counts raw={raw_gradient_count} accumulated={accumulated_gradient_count}"
        );
        assert!(
            raw_gradient_count > 0,
            "EGGROLL-updated model should expose train-step gradients"
        );
        assert!(
            accumulated_gradient_count > 0,
            "EGGROLL-updated model should expose accumulated gradients"
        );
    }

    #[test]
    fn eggroll_forward_only_trains_on_plain_backend() {
        let dir = tempfile::tempdir().expect("tempdir");
        let run_dir = dir.path().join("eggroll-forward-only");
        let parallel_config = burn_dragon_train::ParallelConfig::default();
        let parallel_runtime =
            resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");
        let device = burn::tensor::Device::<TestForwardBackend>::default();
        TestForwardBackend::seed(&device, 41);
        let training = tiny_training_hparams();
        let model_config = tiny_model_config();
        let optimizer_cfg = OptimizerConfig {
            name: OptimizerKind::Eggroll,
            learning_rate: 1.0e-6,
            weight_decay: 0.0,
            weight_decay_final: None,
            lr_schedule: None,
            schedule_mode: OptimizerScheduleMode::DragonReference,
            grad_clip_norm: None,
            grad_clip_value: None,
            eggroll: burn_eggroll::EggrollConfig {
                sigma: 2.5e-3,
                population: burn_eggroll::PopulationConfig {
                    population_size: 2,
                    population_chunk_size: 2,
                    rank: 1,
                    seed: 41,
                    matrix_noise: burn_eggroll::MatrixNoiseMode::default(),
                },
                update: burn_eggroll::EggrollUpdateConfig {
                    learning_rate: 1.0e-6,
                    ..burn_eggroll::EggrollUpdateConfig::default()
                },
                ..burn_eggroll::EggrollConfig::default()
            },
            eggroll_population_execution: Default::default(),
            eggroll_auto_population: Default::default(),
            predictive_coding: Default::default(),
        };
        let env = ForwardEggrollTrainEnvironment {
            parallel_runtime: &parallel_runtime,
            run_dir: &run_dir,
            run_name: "eggroll-forward-only-smoke",
            backend_name: "cpu",
            training: &training,
            resume_checkpoint_epoch: None,
            model_config: &model_config,
            device: &device,
            train_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<
                TestForwardBackend,
            >(
                &device,
                &[0, 1, 2, 3, 4, 5, 6, 7],
                &[1, 2, 3, 4, 5, 6, 7, 8],
                [2, 4],
            )])),
            valid_loader: Arc::new(StaticSequenceLoader::new(vec![make_batch::<
                TestForwardBackend,
            >(
                &device,
                &[1, 2, 3, 4, 5, 6, 7, 8],
                &[2, 3, 4, 5, 6, 7, 8, 9],
                [2, 4],
            )])),
            source_selection_dataset: None,
            summary_event_token_ids: None,
            epochs: 1,
        };
        let model = LanguageTrainModel::new(DragonModel::<TestForwardBackend>::new(
            model_config.clone(),
            &device,
        ));
        let trained = train_with_eggroll_forward_only(&env, &optimizer_cfg, model)
            .expect("forward-only EGGROLL training should not require autodiff");
        let probe = make_batch::<TestForwardBackend>(
            &device,
            &[1, 2, 3, 4, 5, 6, 7, 8],
            &[2, 3, 4, 5, 6, 7, 8, 9],
            [2, 4],
        );
        let loss =
            language_model_loss::<TestForwardBackend>(trained.forward(probe.inputs), probe.targets)
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("loss vec")[0];
        assert!(
            loss.is_finite(),
            "forward-only EGGROLL loss should be finite"
        );
        assert!(
            run_dir.join("checkpoint/model-1.bin").is_file(),
            "forward-only EGGROLL should save plain-backend checkpoints"
        );
    }

    #[test]
    fn eggroll_interval_reduces_population_evaluations() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = crate::train::optimizer_dynamics::OptimizerDynamicsConfig {
            epochs: 4,
            max_iters: 16,
            log_frequency: 4,
            seed: 29,
            eggroll_learning_rate: 1.0e-2,
            eggroll_interval_steps: 4,
            ..crate::train::optimizer_dynamics::OptimizerDynamicsConfig::default()
        };
        let report = crate::train::optimizer_dynamics::run_optimizer_dynamics(
            crate::train::optimizer_dynamics::OptimizerDynamicsKind::Eggroll,
            &config,
            dir.path(),
        )
        .expect("interval eggroll dynamics");
        let total_steps = config.epochs * 4;
        let eggroll_steps = total_steps.div_ceil(config.eggroll_interval_steps);
        let expected_forward_evaluations = eggroll_steps * config.eggroll_population_size;

        eprintln!("interval eggroll report={report:#?}");
        assert_eq!(report.forward_evaluations, expected_forward_evaluations);
        assert!(
            report.final_loss.is_finite(),
            "interval EGGROLL should produce finite validation loss: {report:?}"
        );
    }

    #[test]
    fn eggroll_baseline_is_reasonable_against_nearby_variants() {
        let dir = tempfile::tempdir().expect("tempdir");
        let base = crate::train::optimizer_dynamics::OptimizerDynamicsConfig {
            epochs: 16,
            max_iters: 64,
            log_frequency: 16,
            seed: 29,
            ..crate::train::optimizer_dynamics::OptimizerDynamicsConfig::default()
        };
        let mut center = base.clone();
        center.eggroll_fitness_normalization = burn_eggroll::FitnessNormalization::Center;
        let mut zscore = base.clone();
        zscore.eggroll_fitness_normalization = burn_eggroll::FitnessNormalization::ZScore;
        let mut adamw_update = base.clone();
        adamw_update.eggroll_update_kind = burn_eggroll::EggrollUpdateKind::Adamw;
        adamw_update.eggroll_learning_rate = 1.0e-3;
        let mut smaller_population = base.clone();
        smaller_population.eggroll_population_size = 128;
        let mut larger_population = base.clone();
        larger_population.eggroll_population_size = 512;

        let report = crate::train::optimizer_dynamics::run_eggroll_dynamics_sweep(
            &[
                crate::train::optimizer_dynamics::EggrollDynamicsCandidate::new(
                    "rank_sgd_pop256_rank4",
                    base.clone(),
                ),
                crate::train::optimizer_dynamics::EggrollDynamicsCandidate::new(
                    "center_sgd_pop256_rank4",
                    center,
                ),
                crate::train::optimizer_dynamics::EggrollDynamicsCandidate::new(
                    "zscore_sgd_pop256_rank4",
                    zscore,
                ),
                crate::train::optimizer_dynamics::EggrollDynamicsCandidate::new(
                    "rank_adamw_pop256_rank4",
                    adamw_update,
                ),
                crate::train::optimizer_dynamics::EggrollDynamicsCandidate::new(
                    "rank_sgd_pop128_rank4",
                    smaller_population,
                ),
                crate::train::optimizer_dynamics::EggrollDynamicsCandidate::new(
                    "rank_sgd_pop512_rank4",
                    larger_population,
                ),
            ],
            dir.path(),
        )
        .expect("eggroll dynamics sweep");

        eprintln!("eggroll candidate sweep: {report:#?}");
        let baseline = report
            .get("rank_sgd_pop256_rank4")
            .expect("baseline candidate");
        let best_quality = report.best_by_loss_delta().expect("quality candidate");
        let best_train = report
            .best_by_train_loss_delta()
            .expect("train-loss candidate");
        for candidate in &report.candidates {
            assert!(candidate.report.initial_train_loss.is_finite());
            assert!(candidate.report.final_train_loss.is_finite());
            assert!(candidate.report.initial_loss.is_finite());
            assert!(candidate.report.final_loss.is_finite());
        }
        assert!(
            baseline.report.final_loss <= baseline.report.initial_loss + 0.05,
            "baseline EGGROLL should not regress in the candidate sweep: {report:#?}"
        );
        assert!(
            baseline.report.train_loss_delta() > 0.0,
            "baseline EGGROLL should reduce train loss in the candidate sweep: {report:#?}"
        );
        assert!(
            best_quality.report.loss_delta() > 0.01,
            "at least one tensorized EGGROLL candidate should learn a measurable signal in the candidate sweep: {report:#?}"
        );
        assert!(
            best_train.report.train_loss_delta() > 0.02,
            "at least one tensorized EGGROLL candidate should reduce train loss by a measurable signal in the candidate sweep: {report:#?}"
        );
        assert!(
            baseline.report.loss_delta() >= best_quality.report.loss_delta() * -0.5,
            "tensorized baseline EGGROLL should not be badly dominated on quality by nearby candidates: best={best_quality:?} report={report:#?}"
        );
        assert!(
            baseline.report.evaluations_per_second()
                >= best_quality.report.evaluations_per_second() * 0.25,
            "tensorized baseline EGGROLL should remain throughput-reasonable against nearby candidates: best={best_quality:?} report={report:#?}"
        );
    }

    fn single_device_scheduler_smoke(objective: TrainingObjectiveConfig, run_name: &str) -> f32 {
        let dir = tempfile::tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let parallel_config = burn_dragon_train::ParallelConfig::default();
        let parallel_runtime =
            resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");

        let primary_device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&primary_device, 11);
        let valid_device = burn::tensor::Device::<TestValidBackend>::default();
        let train_batches = vec![
            make_batch::<TestBackend>(
                &primary_device,
                &[0, 1, 2, 3, 4, 5, 6, 7],
                &[1, 2, 3, 4, 5, 6, 7, 0],
                [2, 4],
            ),
            make_batch::<TestBackend>(
                &primary_device,
                &[7, 6, 5, 4, 3, 2, 1, 0],
                &[6, 5, 4, 3, 2, 1, 0, 7],
                [2, 4],
            ),
        ];
        let valid_batches = vec![make_batch::<TestValidBackend>(
            &valid_device,
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 1, 2, 2, 3, 3, 0],
            [2, 4],
        )];

        let training = objective_training_hparams(objective.clone());
        let model_config = tiny_model_config();
        let devices = vec![primary_device];
        let env = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name,
            backend_name: "cpu",
            training: &training,
            resume_checkpoint_epoch: None,
            model_config: &model_config,
            device: &primary_device,
            devices: &devices,
            train_dataset: None,
            valid_dataset: None,
            train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
            valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
            source_selection_dataset: None,
            summary_event_token_ids: None,
            neuron_scaling_slot: None,
            epochs: 1,
            total_steps: 2,
            valid_steps: 1,
        };
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            model_config.clone(),
            &primary_device,
        ))
        .with_training_objective(objective);
        let optimizer = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<TestBackend, LanguageTrainModel<TestBackend>>();

        let trained =
            train_with_scheduler(&env, model, optimizer, 1e-3).expect("objective scheduler train");
        assert!(run_dir.join("checkpoint").join("model-1.bin").is_file());

        let probe = make_batch::<TestValidBackend>(
            &valid_device,
            &[1, 2, 3, 4, 4, 3, 2, 1],
            &[2, 3, 4, 5, 3, 2, 1, 0],
            [2, 4],
        );
        language_model_loss::<TestValidBackend>(trained.forward(probe.inputs), probe.targets)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("loss vec")[0]
    }

    #[test]
    fn train_with_scheduler_accepts_next_token_objective_toggle() {
        let loss = single_device_scheduler_smoke(
            TrainingObjectiveConfig::NextToken,
            "single-next-token-objective-smoke",
        );
        assert!(loss.is_finite(), "next_token smoke loss must be finite");
    }

    #[test]
    fn train_with_scheduler_accepts_predictive_coding_optimizer() {
        let dir = tempfile::tempdir().expect("tempdir");
        let run_dir = dir.path().join("single-pc-optimizer-smoke");
        let parallel_config = burn_dragon_train::ParallelConfig::default();
        let parallel_runtime =
            resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");

        let primary_device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&primary_device, 19);
        let valid_device = burn::tensor::Device::<TestValidBackend>::default();
        let train_batches = vec![
            make_batch::<TestBackend>(
                &primary_device,
                &[0, 1, 2, 3, 4, 5, 6, 7],
                &[1, 2, 3, 4, 5, 6, 7, 0],
                [2, 4],
            ),
            make_batch::<TestBackend>(
                &primary_device,
                &[3, 4, 5, 6, 7, 0, 1, 2],
                &[4, 5, 6, 7, 0, 1, 2, 3],
                [2, 4],
            ),
        ];
        let valid_batches = vec![make_batch::<TestValidBackend>(
            &valid_device,
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 1, 2, 2, 3, 3, 0],
            [2, 4],
        )];

        let mut training = tiny_training_hparams();
        training.tbptt_chunk_size = Some(2);
        training.predictive_coding.enabled = true;
        training.predictive_coding.steps = 1;
        training.predictive_coding.step_size = 0.01;
        let model_config = tiny_model_config();
        let devices = vec![primary_device.clone()];
        let env = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name: "single-pc-optimizer-smoke",
            backend_name: "cpu",
            training: &training,
            resume_checkpoint_epoch: None,
            model_config: &model_config,
            device: &primary_device,
            devices: &devices,
            train_dataset: None,
            valid_dataset: None,
            train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
            valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
            source_selection_dataset: None,
            summary_event_token_ids: None,
            neuron_scaling_slot: None,
            epochs: 1,
            total_steps: 2,
            valid_steps: 1,
        };
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            model_config.clone(),
            &primary_device,
        ))
        .with_predictive_coding(training.predictive_coding.clone())
        .with_tbptt_chunk_size(training.tbptt_chunk_size);
        let optimizer_cfg = OptimizerConfig {
            name: OptimizerKind::PredictiveCoding,
            learning_rate: 1.0e-3,
            weight_decay: 0.0,
            weight_decay_final: None,
            lr_schedule: None,
            schedule_mode: OptimizerScheduleMode::DragonReference,
            grad_clip_norm: Some(1.0),
            grad_clip_value: None,
            eggroll: burn_eggroll::EggrollConfig::default(),
            eggroll_population_execution: Default::default(),
            eggroll_auto_population: Default::default(),
            predictive_coding: PredictiveCodingOptimizerConfig {
                transform: PredictiveCodingOptimizerTransform::Sgd,
                ..Default::default()
            },
        };
        let optimizer =
            resolve_optimizer::<TestBackend, LanguageTrainModel<TestBackend>>(&optimizer_cfg, 2)
                .expect("predictive coding optimizer");

        let trained =
            train_with_scheduler(&env, model, optimizer, 1e-3).expect("PC optimizer train");
        assert!(run_dir.join("checkpoint").join("model-1.bin").is_file());

        let probe = make_batch::<TestValidBackend>(
            &valid_device,
            &[1, 2, 3, 4, 4, 3, 2, 1],
            &[2, 3, 4, 5, 3, 2, 1, 0],
            [2, 4],
        );
        let loss =
            language_model_loss::<TestValidBackend>(trained.forward(probe.inputs), probe.targets)
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("loss vec")[0];
        assert!(loss.is_finite(), "PC optimizer smoke loss must be finite");
    }

    #[test]
    fn train_with_scheduler_accepts_sdft_objective_toggle() {
        let loss = single_device_scheduler_smoke(
            TrainingObjectiveConfig::Sdft(SdftObjectiveConfig {
                max_completion_tokens: 2,
                top_k: Some(1),
                generate_from_teacher: true,
                num_loss_tokens_to_skip: 1,
                ..Default::default()
            }),
            "single-sdft-objective-smoke",
        );
        assert!(loss.is_finite(), "SDFT smoke loss must be finite");
    }

    #[test]
    fn train_with_scheduler_accepts_sdpo_objective_toggle() {
        let loss = single_device_scheduler_smoke(
            TrainingObjectiveConfig::Sdpo(SdpoObjectiveConfig {
                group_size: 2,
                max_completion_tokens: 2,
                top_k: Some(1),
                ..Default::default()
            }),
            "single-sdpo-objective-smoke",
        );
        assert!(loss.is_finite(), "SDPO smoke loss must be finite");
    }

    #[test]
    fn train_with_scheduler_accepts_composite_sdft_sdpo_objective_toggle() {
        let loss = single_device_scheduler_smoke(
            TrainingObjectiveConfig::SdftSdpo(SdftSdpoObjectiveConfig {
                sdft: SdftObjectiveConfig {
                    max_completion_tokens: 2,
                    top_k: Some(1),
                    ..Default::default()
                },
                sdpo: SdpoObjectiveConfig {
                    group_size: 2,
                    max_completion_tokens: 2,
                    top_k: Some(1),
                    ..Default::default()
                },
                ..Default::default()
            }),
            "single-sdft-sdpo-objective-smoke",
        );
        assert!(
            loss.is_finite(),
            "composite SDFT/SDPO smoke loss must be finite"
        );
    }

    #[test]
    fn dynamic_neuron_scale_widens_model_in_process() {
        let dir = tempfile::tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let parallel_config = burn_dragon_train::ParallelConfig::default();
        let parallel_runtime =
            resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 7);
        let valid_device = burn::tensor::Device::<TestValidBackend>::default();
        let mut training = tiny_training_hparams();
        training.neuron_scaling.enabled = true;
        training.neuron_scaling.max_latent_total = 16;
        training.neuron_scaling.stabilization.freeze_base_steps = 1;
        training.neuron_scaling.stabilization.unfreeze_ramp_steps = 1;
        let model_config = tiny_model_config();
        let devices = vec![device.clone()];
        let train_batches = vec![make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 0],
            [2, 4],
        )];
        let valid_batches = vec![make_batch::<TestValidBackend>(
            &valid_device,
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 1, 2, 2, 3, 3, 0],
            [2, 4],
        )];
        let env = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name: "dynamic-scale-smoke",
            backend_name: "cpu",
            training: &training,
            resume_checkpoint_epoch: None,
            model_config: &model_config,
            device: &device,
            devices: &devices,
            train_dataset: None,
            valid_dataset: None,
            train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
            valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
            source_selection_dataset: None,
            summary_event_token_ids: None,
            neuron_scaling_slot: None,
            epochs: 1,
            total_steps: 1,
            valid_steps: 1,
        };
        let mut model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            model_config.clone(),
            &device,
        ))
        .with_gradient_scale_schedule(&training, 1);
        let mut optimizer = tiny_language_optimizer(&training, &model_config, &device);
        let handles = crate::train::events::build_training_event_handles(
            "dynamic-scale-smoke",
            &run_dir,
            1,
            &training,
            None,
            None,
            None,
        )
        .expect("event handles");
        let bus = handles.metric_logger.bus();
        let mut current_model_config = model_config.clone();
        let mut scale_generation = 0usize;

        let scale_result = apply_dynamic_neuron_scale(
            &env,
            &mut model,
            &mut optimizer,
            &mut current_model_config,
            &mut scale_generation,
            ModelScaleRequest {
                run_id: "dynamic-scale-smoke".to_string(),
                epoch: Some(1),
                absolute_step: Some(0),
                from_capacity_units: 8,
                to_capacity_units: 16,
                reason: "test plateau".to_string(),
            },
            1,
            0,
            &bus,
            training.batch_size,
            training.gradient_accumulation_steps,
        )
        .expect("apply scale");

        let _ = bus.flush();
        assert_eq!(scale_result, Some((8, 16)));
        assert_eq!(model.model.latent_total_capacity(), 16);
        assert_eq!(current_model_config.latent_total(), 16);
        assert_eq!(scale_generation, 1);
    }

    #[test]
    fn dynamic_neuron_scaling_scheduler_consumes_request_in_process() {
        let dir = tempfile::tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let parallel_config = burn_dragon_train::ParallelConfig::default();
        let parallel_runtime =
            resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 13);
        let valid_device = burn::tensor::Device::<TestValidBackend>::default();
        let mut training = tiny_training_hparams();
        training.neuron_scaling.enabled = true;
        training.neuron_scaling.max_latent_total = 16;
        let model_config = tiny_model_config();
        let devices = vec![device.clone()];
        let request_slot = crate::train::neuron_scaling::NeuronScaleRequestSlot::default();
        assert!(request_slot.set_if_empty(ModelScaleRequest {
            run_id: "dynamic-scale-loop-smoke".to_string(),
            epoch: Some(1),
            absolute_step: Some(0),
            from_capacity_units: 8,
            to_capacity_units: 16,
            reason: "test plateau".to_string(),
        }));
        let train_batches = vec![make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 0],
            [2, 4],
        )];
        let valid_batches = vec![make_batch::<TestValidBackend>(
            &valid_device,
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 1, 2, 2, 3, 3, 0],
            [2, 4],
        )];
        let env = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name: "dynamic-scale-loop-smoke",
            backend_name: "cpu",
            training: &training,
            resume_checkpoint_epoch: None,
            model_config: &model_config,
            device: &device,
            devices: &devices,
            train_dataset: None,
            valid_dataset: None,
            train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
            valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
            source_selection_dataset: None,
            summary_event_token_ids: None,
            neuron_scaling_slot: Some(request_slot.clone()),
            epochs: 1,
            total_steps: 1,
            valid_steps: 1,
        };
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            model_config.clone(),
            &device,
        ))
        .with_gradient_scale_schedule(&training, 1);
        let optimizer = tiny_language_optimizer(&training, &model_config, &device);

        let trained = train_with_dynamic_neuron_scaling_scheduler(&env, model, optimizer, 1e-3)
            .expect("dynamic scaling train");

        assert_eq!(trained.latent_total_capacity(), 16);
        assert!(request_slot.take().is_none());
        assert!(run_dir.join("checkpoint").join("model-1.bin").is_file());
    }

    #[test]
    fn dynamic_scheduler_throttles_train_metric_events() {
        let dir = tempfile::tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let parallel_config = burn_dragon_train::ParallelConfig::default();
        let parallel_runtime =
            resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 19);
        let valid_device = burn::tensor::Device::<TestValidBackend>::default();
        let mut training = tiny_training_hparams();
        training.log_frequency = 2;
        training.events.flush_every_steps = 1;
        training.events.degeneracy_probe_every_epochs = usize::MAX;
        let model_config = tiny_model_config();
        let devices = vec![device.clone()];
        let train_batches = vec![
            make_batch::<TestBackend>(
                &device,
                &[0, 1, 2, 3, 4, 5, 6, 7],
                &[1, 2, 3, 4, 5, 6, 7, 0],
                [2, 4],
            ),
            make_batch::<TestBackend>(
                &device,
                &[1, 2, 3, 4, 5, 6, 7, 0],
                &[2, 3, 4, 5, 6, 7, 0, 1],
                [2, 4],
            ),
            make_batch::<TestBackend>(
                &device,
                &[2, 3, 4, 5, 6, 7, 0, 1],
                &[3, 4, 5, 6, 7, 0, 1, 2],
                [2, 4],
            ),
        ];
        let valid_batches = vec![make_batch::<TestValidBackend>(
            &valid_device,
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 1, 2, 2, 3, 3, 0],
            [2, 4],
        )];
        let env = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name: "dynamic-metric-throttle-smoke",
            backend_name: "cpu",
            training: &training,
            resume_checkpoint_epoch: None,
            model_config: &model_config,
            device: &device,
            devices: &devices,
            train_dataset: None,
            valid_dataset: None,
            train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
            valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
            source_selection_dataset: None,
            summary_event_token_ids: None,
            neuron_scaling_slot: None,
            epochs: 1,
            total_steps: 3,
            valid_steps: 1,
        };
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            model_config.clone(),
            &device,
        ))
        .with_gradient_scale_schedule(&training, 3);
        let optimizer = tiny_language_optimizer(&training, &model_config, &device);

        let _trained = train_with_dynamic_neuron_scaling_scheduler(&env, model, optimizer, 1e-3)
            .expect("dynamic scheduler train");

        let events = read_training_events(&run_dir);
        let train_loss_steps = events
            .iter()
            .filter(|event| {
                event.get("type").and_then(|value| value.as_str()) == Some("metric")
                    && event.get("split").and_then(|value| value.as_str()) == Some("train")
                    && event.get("name").and_then(|value| value.as_str()) == Some("Loss")
            })
            .map(|event| {
                event
                    .get("step_in_epoch")
                    .and_then(|value| value.as_u64())
                    .expect("train loss step") as usize
            })
            .collect::<Vec<_>>();

        assert_eq!(train_loss_steps, vec![2, 3]);
    }

    #[test]
    fn dynamic_scheduler_recovery_control_scales_continual_backprop_in_training_loop() {
        let dir = tempfile::tempdir().expect("tempdir");
        let weak_events = run_recovery_cbp_scale_smoke(
            &dir.path().join("weak"),
            "dynamic-recovery-cbp-weak",
            0.5,
            1.25,
        );
        let strong_events = run_recovery_cbp_scale_smoke(
            &dir.path().join("strong"),
            "dynamic-recovery-cbp-strong",
            2.0,
            0.75,
        );

        let weak_control_scale =
            latest_dynamics_control_scale(&weak_events).expect("weak recovery control");
        let strong_control_scale =
            latest_dynamics_control_scale(&strong_events).expect("strong recovery control");
        let strong_control_max_replacements =
            latest_dynamics_control_max_replacements(&strong_events)
                .expect("strong recovery max replacements");
        let weak_effective_scale =
            latest_continual_backprop_replacement_scale(&weak_events).expect("weak CBP telemetry");
        let strong_effective_scale = latest_continual_backprop_replacement_scale(&strong_events)
            .expect("strong CBP telemetry");
        let strong_effective_max_replacements =
            latest_continual_backprop_max_replacements(&strong_events)
                .expect("strong CBP max replacements telemetry");

        assert_eq!(weak_control_scale, 0.5);
        assert_eq!(weak_effective_scale, 0.5);
        assert_eq!(strong_control_scale, 2.0);
        assert_eq!(strong_control_max_replacements, 3);
        assert_eq!(strong_effective_scale, 2.0);
        assert_eq!(strong_effective_max_replacements, 3);
        assert!(
            strong_effective_scale > weak_effective_scale,
            "strong recovery should increase realized CBP plasticity relative to weak baseline"
        );
    }

    fn run_recovery_cbp_scale_smoke(
        run_dir: &Path,
        run_name: &str,
        recovery_cbp_scale: f64,
        recovery_source_pressure: f64,
    ) -> Vec<serde_json::Value> {
        let parallel_config = burn_dragon_train::ParallelConfig::default();
        let parallel_runtime =
            resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");
        let device = burn::tensor::Device::<TestBackend>::default();
        TestBackend::seed(&device, 17);
        let valid_device = burn::tensor::Device::<TestValidBackend>::default();
        let mut training = tiny_training_hparams_with_epochs(2, None);
        training.continual_backprop.enabled = true;
        training.continual_backprop.warmup_steps = 0;
        training.continual_backprop.maturity_steps = 0;
        training.continual_backprop.cooldown_steps = 0;
        training.continual_backprop.sample_interval_steps = 1;
        training.continual_backprop.replace_interval_steps = 1;
        training.continual_backprop.replacement_rate = 0.1;
        training.continual_backprop.max_replacements_per_interval = 1;
        training.continual_backprop.lr_coupling =
            burn_dragon_train::ContinualBackpropLrCoupling::None;
        training.events.continual_backprop_every_steps = 1;
        training.events.degeneracy_probe_every_epochs = 1;
        training.events.degeneracy_probe_tokens = 8;
        training.gates.degeneracy_entropy_min_bits = 128.0;
        training.gates.degeneracy_max_probability_max = 2.0;
        training.gates.degeneracy_argmax_unique_min_fraction = 1.0;
        training.gates.degeneracy_distinct_2_min_fraction = 1.0;
        training.gates.degeneracy_repetition_max_fraction = 0.0;
        training.gates.degeneracy_period_2_max_fraction = 0.0;
        training.gates.degeneracy_period_3_max_fraction = 0.0;
        training.gates.degeneracy_period_2_to_16_max_fraction = 0.0;
        training.gates.degeneracy_period_2_to_64_max_fraction = 0.0;
        training.dynamics.soft_recovery_continual_backprop_scale = recovery_cbp_scale;
        training
            .dynamics
            .validation_recovery_continual_backprop_scale = recovery_cbp_scale;
        training.dynamics.hard_recovery_continual_backprop_scale = recovery_cbp_scale;
        training
            .dynamics
            .soft_recovery_max_replacements_per_interval = Some(3);
        training
            .dynamics
            .validation_recovery_max_replacements_per_interval = Some(3);
        training
            .dynamics
            .hard_recovery_max_replacements_per_interval = Some(3);
        training.dynamics.recovery_source_difficulty_pressure = recovery_source_pressure;
        let model_config = tiny_model_config();
        let devices = vec![device.clone()];
        let train_batches = vec![make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 0],
            [2, 4],
        )];
        let valid_batches = vec![make_batch::<TestValidBackend>(
            &valid_device,
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 1, 2, 2, 3, 3, 0],
            [2, 4],
        )];
        let env = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir,
            run_name,
            backend_name: "cpu",
            training: &training,
            resume_checkpoint_epoch: None,
            model_config: &model_config,
            device: &device,
            devices: &devices,
            train_dataset: None,
            valid_dataset: None,
            train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
            valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
            source_selection_dataset: None,
            summary_event_token_ids: None,
            neuron_scaling_slot: None,
            epochs: 2,
            total_steps: 2,
            valid_steps: 1,
        };
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            model_config.clone(),
            &device,
        ))
        .with_continual_backprop(&training.continual_backprop);
        let optimizer = tiny_language_optimizer(&training, &model_config, &device);

        let trained = train_with_dynamic_neuron_scaling_scheduler(&env, model, optimizer, 1e-3)
            .expect("dynamic recovery train");

        assert_eq!(trained.latent_total_capacity(), model_config.latent_total());
        let events = read_training_events(run_dir);
        assert!(
            events
                .iter()
                .any(|event| event.get("type").and_then(|value| value.as_str())
                    == Some("dynamics_control")
                    && event
                        .get("mode")
                        .and_then(|value| value.as_str())
                        .is_some_and(is_recovery_mode)
                    && event
                        .get("continual_backprop_scale")
                        .and_then(|value| value.as_f64())
                        .is_some_and(|scale| (scale - recovery_cbp_scale).abs() < 1.0e-9)
                    && event
                        .get("max_replacements_per_interval")
                        .and_then(|value| value.as_u64())
                        == Some(3)
                    && event
                        .get("source_difficulty_pressure")
                        .and_then(|value| value.as_f64())
                        .is_some_and(
                            |pressure| (pressure - recovery_source_pressure).abs() < 1.0e-9
                        )),
            "training events should include a plasticity recovery control: {events:#?}"
        );
        assert!(
            events
                .iter()
                .any(|event| event.get("type").and_then(|value| value.as_str())
                    == Some("continual_backprop")
                    && event
                        .get("replacement_rate_scale")
                        .and_then(|value| value.as_f64())
                        .is_some_and(|scale| (scale - recovery_cbp_scale).abs() < 1.0e-6)
                    && event
                        .get("effective_max_replacements_per_interval")
                        .and_then(|value| value.as_u64())
                        == Some(3)),
            "epoch after recovery should emit CBP telemetry using recovery scale: {events:#?}"
        );
        events
    }

    fn latest_dynamics_control_scale(events: &[serde_json::Value]) -> Option<f64> {
        events.iter().rev().find_map(|event| {
            (event.get("type").and_then(|value| value.as_str()) == Some("dynamics_control")
                && event
                    .get("mode")
                    .and_then(|value| value.as_str())
                    .is_some_and(is_recovery_mode))
            .then(|| {
                event
                    .get("continual_backprop_scale")
                    .and_then(|value| value.as_f64())
            })
            .flatten()
        })
    }

    fn latest_continual_backprop_replacement_scale(events: &[serde_json::Value]) -> Option<f64> {
        events.iter().rev().find_map(|event| {
            (event.get("type").and_then(|value| value.as_str()) == Some("continual_backprop"))
                .then(|| {
                    event
                        .get("replacement_rate_scale")
                        .and_then(|value| value.as_f64())
                })
                .flatten()
        })
    }

    fn latest_dynamics_control_max_replacements(events: &[serde_json::Value]) -> Option<usize> {
        events.iter().rev().find_map(|event| {
            (event.get("type").and_then(|value| value.as_str()) == Some("dynamics_control")
                && event
                    .get("mode")
                    .and_then(|value| value.as_str())
                    .is_some_and(is_recovery_mode))
            .then(|| {
                event
                    .get("max_replacements_per_interval")
                    .and_then(|value| value.as_u64())
                    .map(|value| value as usize)
            })
            .flatten()
        })
    }

    fn is_recovery_mode(mode: &str) -> bool {
        matches!(
            mode,
            "plasticity_recovery"
                | "validation_recovery"
                | "rollback_recovery"
                | "hard_recovery"
                | "hard_collapse"
        )
    }

    fn latest_continual_backprop_max_replacements(events: &[serde_json::Value]) -> Option<usize> {
        events.iter().rev().find_map(|event| {
            (event.get("type").and_then(|value| value.as_str()) == Some("continual_backprop"))
                .then(|| {
                    event
                        .get("effective_max_replacements_per_interval")
                        .and_then(|value| value.as_u64())
                        .map(|value| value as usize)
                })
                .flatten()
        })
    }

    fn read_training_events(run_dir: &Path) -> Vec<serde_json::Value> {
        let path = run_dir.join("events").join("training_events.jsonl");
        let content = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("training event json"))
            .collect()
    }

    #[cfg(feature = "ddp")]
    fn collective_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[cfg(feature = "ddp")]
    fn flatten_gradients_in_module_order<B, M>(module: &M, mut grads: GradientsParams) -> Vec<f32>
    where
        B: AutodiffBackend,
        M: AutodiffModule<B>,
    {
        #[derive(Default)]
        struct GradientCollector {
            values: Vec<f32>,
        }

        struct GradientCollectorVisitor<'a> {
            collector: &'a mut GradientCollector,
            grads: &'a mut GradientsParams,
        }

        impl<B: AutodiffBackend> burn::module::ModuleVisitor<B> for GradientCollectorVisitor<'_> {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                let grad = self
                    .grads
                    .remove::<B::InnerBackend, D>(param.id)
                    .unwrap_or_else(|| param.val().inner().zeros_like());
                let values = grad
                    .to_data()
                    .convert::<f32>()
                    .into_vec::<f32>()
                    .expect("gradient data");
                self.collector.values.extend(values);
            }
        }

        let mut collector = GradientCollector::default();
        let mut visitor = GradientCollectorVisitor {
            collector: &mut collector,
            grads: &mut grads,
        };
        module.visit(&mut visitor);
        collector.values
    }

    #[cfg(feature = "ddp")]
    fn mean_abs_diff(left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .sum::<f32>()
            / left.len().max(1) as f32
    }

    #[cfg(feature = "ddp")]
    fn l2_norm(values: &[f32]) -> f32 {
        values.iter().map(|value| value * value).sum::<f32>().sqrt()
    }

    #[cfg(feature = "ddp")]
    fn stage_split_surrogate_gradients(
        split_model: LanguageTrainModel<TestBackend>,
        plan: &PipelinePlan,
        batch: SequenceBatch<TestBackend>,
    ) -> Vec<f32> {
        let [batch_size, _] = batch.inputs.shape().dims();
        let ranges = split_microbatch_ranges(batch_size, plan.microbatches).expect("ranges");
        let chunk_inputs = ranges
            .iter()
            .cloned()
            .map(|range| slice_batch_int(batch.inputs.clone(), range))
            .collect::<Vec<_>>();
        let chunk_targets = ranges
            .iter()
            .cloned()
            .map(|range| slice_batch_int(batch.targets.clone(), range))
            .collect::<Vec<_>>();
        let chunk_masks = ranges
            .iter()
            .cloned()
            .map(|range| {
                batch
                    .summary_event_mask
                    .clone()
                    .map(|mask| slice_batch_int(mask, range))
            })
            .collect::<Vec<_>>();
        let mut chunk_states = (0..plan.microbatches)
            .map(|_| split_model.model.init_state())
            .collect::<Vec<_>>();
        let mut accumulator = GradientsAccumulator::new();
        let last_virtual_stage_id = plan.total_virtual_stages.saturating_sub(1);

        for microbatch_id in 0..plan.microbatches {
            let stage0_output = split_model
                .model
                .forward_language_pipeline_stage_with_state(
                    split_model
                        .model
                        .begin_language_pipeline(chunk_inputs[microbatch_id].clone()),
                    &mut chunk_states[microbatch_id],
                    plan.assignment(0).layer_range.clone(),
                    chunk_masks[microbatch_id].clone(),
                );
            let stage1_input = attach_pipeline_state_require_grad::<TestBackend>(
                detach_pipeline_state_to_inner(&stage0_output),
            );
            let stage1_input_for_grad = stage1_input.clone();
            let stage1_output = split_model
                .model
                .forward_language_pipeline_stage_with_state(
                    stage1_input,
                    &mut chunk_states[microbatch_id],
                    plan.assignment(last_virtual_stage_id).layer_range.clone(),
                    chunk_masks[microbatch_id].clone(),
                );
            let hidden = split_model
                .model
                .finish_language_pipeline_hidden_with_state(
                    stage1_output,
                    &mut chunk_states[microbatch_id],
                );
            let weight = ranges[microbatch_id].len() as f32 / batch_size as f32;
            let loss = split_model
                .model
                .language_loss_from_hidden(hidden, chunk_targets[microbatch_id].clone())
                .mul_scalar(weight);
            let mut stage1_grads = loss.backward();
            let grad_to_stage0 =
                pipeline_input_grad_state(&stage1_input_for_grad, &mut stage1_grads);
            accumulator.accumulate(
                &split_model,
                GradientsParams::from_grads(stage1_grads, &split_model),
            );

            let stage0_surrogate = pipeline_surrogate_loss(&stage0_output, grad_to_stage0);
            accumulator.accumulate(
                &split_model,
                GradientsParams::from_grads(stage0_surrogate.backward(), &split_model),
            );
        }

        flatten_gradients_in_module_order::<TestBackend, _>(&split_model, accumulator.grads())
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn train_with_scheduler_executes_local_ddp_on_ndarray() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");

        let parallel_config = burn_dragon_train::ParallelConfig {
            mode: ParallelismKind::Ddp,
            world_size: 2,
            data: burn_dragon_train::ParallelDataConfig {
                size: 2,
                ..Default::default()
            },
            ..Default::default()
        };
        let parallel_runtime =
            resolve_parallel_runtime(&parallel_config).expect("resolve local ddp runtime");

        let primary_device = burn::tensor::Device::<TestBackend>::default();
        let devices =
            resolve_training_devices::<TestBackend>(&parallel_runtime, &primary_device).unwrap();
        assert_eq!(devices.len(), 2, "expected 2 local replicas");

        let valid_device = burn::tensor::Device::<TestValidBackend>::default();
        let train_batches = vec![
            make_batch::<TestBackend>(
                &primary_device,
                &[0, 1, 2, 3, 4, 5, 6, 7],
                &[1, 2, 3, 4, 5, 6, 7, 0],
                [2, 4],
            ),
            make_batch::<TestBackend>(
                &primary_device,
                &[7, 6, 5, 4, 3, 2, 1, 0],
                &[6, 5, 4, 3, 2, 1, 0, 7],
                [2, 4],
            ),
        ];
        let valid_batches = vec![make_batch::<TestValidBackend>(
            &valid_device,
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 1, 2, 2, 3, 3, 0],
            [2, 4],
        )];

        let training = tiny_training_hparams();
        let model_config = tiny_model_config();
        let env = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name: "ddp-ndarray-smoke",
            backend_name: "cpu",
            training: &training,
            resume_checkpoint_epoch: None,
            model_config: &model_config,
            device: &primary_device,
            devices: &devices,
            train_dataset: None,
            valid_dataset: None,
            train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
            valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
            source_selection_dataset: None,
            summary_event_token_ids: None,
            neuron_scaling_slot: None,
            epochs: 1,
            total_steps: 2,
            valid_steps: 1,
        };

        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            model_config.clone(),
            &primary_device,
        ));
        let optimizer = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<TestBackend, LanguageTrainModel<TestBackend>>();

        let trained = train_with_scheduler(&env, model, optimizer, 1e-3).expect("ddp train");
        let probe = make_batch::<TestValidBackend>(
            &valid_device,
            &[1, 2, 3, 4, 4, 3, 2, 1],
            &[2, 3, 4, 5, 3, 2, 1, 0],
            [2, 4],
        );
        let loss =
            language_model_loss::<TestValidBackend>(trained.forward(probe.inputs), probe.targets)
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("loss vec")[0];

        assert!(loss.is_finite(), "ddp smoke loss must be finite");
    }

    #[test]
    fn train_with_scheduler_retains_best_valid_and_last_checkpoints() {
        let dir = tempfile::tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");

        let parallel_config = burn_dragon_train::ParallelConfig::default();
        let parallel_runtime =
            resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");

        let primary_device = burn::tensor::Device::<TestBackend>::default();
        let valid_device = burn::tensor::Device::<TestValidBackend>::default();
        let train_batches = vec![
            make_batch::<TestBackend>(
                &primary_device,
                &[0, 1, 2, 3, 4, 5, 6, 7],
                &[1, 2, 3, 4, 5, 6, 7, 0],
                [2, 4],
            ),
            make_batch::<TestBackend>(
                &primary_device,
                &[7, 6, 5, 4, 3, 2, 1, 0],
                &[6, 5, 4, 3, 2, 1, 0, 7],
                [2, 4],
            ),
        ];
        let valid_batches = vec![make_batch::<TestValidBackend>(
            &valid_device,
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 1, 2, 2, 3, 3, 0],
            [2, 4],
        )];

        let training = tiny_training_hparams_with_epochs(4, None);
        let model_config = tiny_model_config();
        let devices = vec![primary_device];
        let env = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name: "single-retention-smoke",
            backend_name: "cpu",
            training: &training,
            resume_checkpoint_epoch: None,
            model_config: &model_config,
            device: &primary_device,
            devices: &devices,
            train_dataset: None,
            valid_dataset: None,
            train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
            valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
            source_selection_dataset: None,
            summary_event_token_ids: None,
            neuron_scaling_slot: None,
            epochs: 4,
            total_steps: 8,
            valid_steps: 1,
        };
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            model_config.clone(),
            &primary_device,
        ));
        let optimizer = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<TestBackend, LanguageTrainModel<TestBackend>>();

        let _trained =
            train_with_scheduler(&env, model, optimizer, 1e-3).expect("single-device train");

        let retained = retained_model_epochs(&run_dir);
        assert!(
            retained.contains(&3),
            "third epoch should be kept as recent"
        );
        assert!(retained.contains(&4), "last epoch should be kept as recent");
        assert!(
            retained.len() <= CHECKPOINT_KEEP_LAST + 1,
            "retention should keep the recent window plus at most one older best checkpoint"
        );
        assert!(
            retained.iter().all(|epoch| (1..=4).contains(epoch)),
            "retained epochs must come from completed checkpoints"
        );
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn shard_bounds_evenly_distribute_remainder_steps() {
        assert_eq!(shard_bounds(5, 0, 2).expect("rank0"), (0, 3));
        assert_eq!(shard_bounds(5, 1, 2).expect("rank1"), (3, 5));
        assert!(shard_bounds(1, 1, 2).is_err());
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn gradient_mean_matches_combined_batch_reference_in_module_order() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let config = tiny_model_config();
        let reference = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device));
        let combined_model = reference.clone();
        let shard_a_model = reference.clone();
        let shard_b_model = reference;

        let shard_a = make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 0],
            [2, 4],
        );
        let shard_b = make_batch::<TestBackend>(
            &device,
            &[7, 6, 5, 4, 3, 2, 1, 0],
            &[6, 5, 4, 3, 2, 1, 0, 7],
            [2, 4],
        );
        let combined = make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0],
            &[1, 2, 3, 4, 5, 6, 7, 0, 6, 5, 4, 3, 2, 1, 0, 7],
            [4, 4],
        );

        let combined_grads = flatten_gradients_in_module_order::<TestBackend, _>(
            &combined_model,
            burn_train::TrainStep::step(&combined_model, combined).grads,
        );
        let shard_a_grads = flatten_gradients_in_module_order::<TestBackend, _>(
            &shard_a_model,
            burn_train::TrainStep::step(&shard_a_model, shard_a).grads,
        );
        let shard_b_grads = flatten_gradients_in_module_order::<TestBackend, _>(
            &shard_b_model,
            burn_train::TrainStep::step(&shard_b_model, shard_b).grads,
        );

        assert_eq!(combined_grads.len(), shard_a_grads.len());
        assert_eq!(combined_grads.len(), shard_b_grads.len());

        let averaged_shards = shard_a_grads
            .iter()
            .zip(shard_b_grads.iter())
            .map(|(lhs, rhs)| (lhs + rhs) * 0.5)
            .collect::<Vec<_>>();

        let mean_abs = mean_abs_diff(&combined_grads, &averaged_shards);
        let combined_norm = l2_norm(&combined_grads);
        let averaged_norm = l2_norm(&averaged_shards);

        assert!(
            mean_abs <= 1.0e-5,
            "combined-batch reference and mean rank-local gradients drifted: mean_abs_diff={mean_abs}"
        );
        assert!(
            (combined_norm - averaged_norm).abs() <= 1.0e-5,
            "gradient norms drifted: combined_norm={combined_norm} averaged_norm={averaged_norm}"
        );
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn train_with_collective_scheduler_runs_single_rank_and_writes_checkpoint() {
        let _lock = collective_test_lock().lock().expect("collective lock");
        reset_collective::<TestValidBackend>();

        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let parallel_config = burn_dragon_train::ParallelConfig {
            mode: ParallelismKind::Ddp,
            world_size: 1,
            data: burn_dragon_train::ParallelDataConfig {
                size: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let parallel_runtime = ParallelRuntime {
            mode: ParallelismKind::Ddp,
            world_size: 1,
            global_rank: 0,
            local_rank: 0,
            data_parallel_size: 1,
            local_data_parallel_size: 1,
            tensor_parallel_size: 1,
            process_group_launch: false,
        };

        let primary_device = burn::tensor::Device::<TestBackend>::default();
        let valid_device = burn::tensor::Device::<TestValidBackend>::default();
        let train_batches = vec![
            make_batch::<TestBackend>(
                &primary_device,
                &[0, 1, 2, 3, 4, 5, 6, 7],
                &[1, 2, 3, 4, 5, 6, 7, 0],
                [2, 4],
            ),
            make_batch::<TestBackend>(
                &primary_device,
                &[7, 6, 5, 4, 3, 2, 1, 0],
                &[6, 5, 4, 3, 2, 1, 0, 7],
                [2, 4],
            ),
        ];
        let valid_batches = vec![make_batch::<TestValidBackend>(
            &valid_device,
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 1, 2, 2, 3, 3, 0],
            [2, 4],
        )];

        let training = tiny_training_hparams();
        let model_config = tiny_model_config();
        let devices = vec![primary_device.clone()];
        let env = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name: "collective-single-rank",
            backend_name: "cpu",
            training: &training,
            resume_checkpoint_epoch: None,
            model_config: &model_config,
            device: &primary_device,
            devices: &devices,
            train_dataset: None,
            valid_dataset: None,
            train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
            valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
            source_selection_dataset: None,
            summary_event_token_ids: None,
            neuron_scaling_slot: None,
            epochs: 1,
            total_steps: 2,
            valid_steps: 1,
        };
        let model = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            model_config.clone(),
            &primary_device,
        ));
        let optimizer = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<TestBackend, LanguageTrainModel<TestBackend>>();
        let collective =
            resolve_collective_config(&parallel_runtime, &parallel_config).expect("collective");

        let trained =
            train_with_collective_scheduler(&env, model, optimizer, 1e-3, collective, 0.into())
                .expect("collective train");
        let probe = make_batch::<TestValidBackend>(
            &valid_device,
            &[1, 2, 3, 4, 4, 3, 2, 1],
            &[2, 3, 4, 5, 3, 2, 1, 0],
            [2, 4],
        );
        let loss =
            language_model_loss::<TestValidBackend>(trained.forward(probe.inputs), probe.targets)
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("loss vec")[0];

        assert!(loss.is_finite());
        assert!(run_dir.join("checkpoint").join("model-1.bin").is_file());

        reset_collective::<TestValidBackend>();
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn train_with_collective_scheduler_resumes_from_checkpoint_family() {
        let _lock = collective_test_lock().lock().expect("collective lock");
        reset_collective::<TestValidBackend>();

        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let parallel_config = burn_dragon_train::ParallelConfig {
            mode: ParallelismKind::Ddp,
            world_size: 1,
            data: burn_dragon_train::ParallelDataConfig {
                size: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let parallel_runtime = ParallelRuntime {
            mode: ParallelismKind::Ddp,
            world_size: 1,
            global_rank: 0,
            local_rank: 0,
            data_parallel_size: 1,
            local_data_parallel_size: 1,
            tensor_parallel_size: 1,
            process_group_launch: false,
        };

        let primary_device = burn::tensor::Device::<TestBackend>::default();
        let valid_device = burn::tensor::Device::<TestValidBackend>::default();
        let train_loader: Arc<dyn DataLoader<TestBackend, SequenceBatch<TestBackend>>> =
            Arc::new(StaticSequenceLoader::new(vec![
                make_batch::<TestBackend>(
                    &primary_device,
                    &[0, 1, 2, 3, 4, 5, 6, 7],
                    &[1, 2, 3, 4, 5, 6, 7, 0],
                    [2, 4],
                ),
                make_batch::<TestBackend>(
                    &primary_device,
                    &[7, 6, 5, 4, 3, 2, 1, 0],
                    &[6, 5, 4, 3, 2, 1, 0, 7],
                    [2, 4],
                ),
            ]));
        let valid_loader: Arc<dyn DataLoader<TestValidBackend, SequenceBatch<TestValidBackend>>> =
            Arc::new(StaticSequenceLoader::new(vec![make_batch::<
                TestValidBackend,
            >(
                &valid_device,
                &[0, 0, 1, 1, 2, 2, 3, 3],
                &[0, 1, 1, 2, 2, 3, 3, 0],
                [2, 4],
            )]));
        let devices = vec![primary_device.clone()];
        let model_config = tiny_model_config();
        let collective =
            resolve_collective_config(&parallel_runtime, &parallel_config).expect("collective");

        let training_first = tiny_training_hparams_with_epochs(1, None);
        let env_first = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name: "collective-resume",
            backend_name: "cpu",
            training: &training_first,
            resume_checkpoint_epoch: None,
            model_config: &model_config,
            device: &primary_device,
            devices: &devices,
            train_dataset: None,
            valid_dataset: None,
            train_loader: Arc::clone(&train_loader),
            valid_loader: Arc::clone(&valid_loader),
            source_selection_dataset: None,
            summary_event_token_ids: None,
            neuron_scaling_slot: None,
            epochs: 1,
            total_steps: 2,
            valid_steps: 1,
        };
        let model_first = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            model_config.clone(),
            &primary_device,
        ));
        let optimizer_first = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<TestBackend, LanguageTrainModel<TestBackend>>();
        train_with_collective_scheduler(
            &env_first,
            model_first,
            optimizer_first,
            1e-3,
            collective.clone(),
            0.into(),
        )
        .expect("first collective train");
        assert!(run_dir.join("checkpoint").join("model-1.bin").is_file());

        reset_collective::<TestValidBackend>();

        let training_resume = tiny_training_hparams_with_epochs(2, Some(1));
        let env_resume = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name: "collective-resume",
            backend_name: "cpu",
            training: &training_resume,
            resume_checkpoint_epoch: Some(1),
            model_config: &model_config,
            device: &primary_device,
            devices: &devices,
            train_dataset: None,
            valid_dataset: None,
            train_loader,
            valid_loader,
            source_selection_dataset: None,
            summary_event_token_ids: None,
            neuron_scaling_slot: None,
            epochs: 2,
            total_steps: 4,
            valid_steps: 1,
        };
        let model_resume = LanguageTrainModel::new(DragonModel::<TestBackend>::new(
            model_config.clone(),
            &primary_device,
        ));
        let optimizer_resume = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<TestBackend, LanguageTrainModel<TestBackend>>();
        let resumed = train_with_collective_scheduler(
            &env_resume,
            model_resume,
            optimizer_resume,
            1e-3,
            collective,
            0.into(),
        )
        .expect("resumed collective train");

        let probe = make_batch::<TestValidBackend>(
            &valid_device,
            &[1, 2, 3, 4, 4, 3, 2, 1],
            &[2, 3, 4, 5, 3, 2, 1, 0],
            [2, 4],
        );
        let loss =
            language_model_loss::<TestValidBackend>(resumed.forward(probe.inputs), probe.targets)
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("loss vec")[0];

        assert!(loss.is_finite());
        assert!(run_dir.join("checkpoint").join("model-2.bin").is_file());

        reset_collective::<TestValidBackend>();
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn pipeline_stage_surrogate_backward_matches_full_pipeline_gradients() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let mut config = tiny_model_config();
        config.n_layer = 2;
        let pipeline = burn_dragon_train::ParallelPipelineConfig {
            enabled: true,
            stage_count: 2,
            virtual_stages_per_rank: 1,
            schedule: burn_dragon_train::PipelineScheduleKind::Interleaved1f1b,
            microbatches: 2,
            ..Default::default()
        };
        let plan = build_pipeline_plan(config.n_layer, &pipeline).expect("plan");
        let reference_model =
            LanguageTrainModel::new(DragonModel::<TestBackend>::new(config.clone(), &device))
                .with_pipeline_plan(Some(plan.clone()));
        let split_model = reference_model.clone();

        let batch = make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 7, 6, 5, 4],
            &[1, 2, 3, 4, 6, 5, 4, 3],
            [2, 4],
        );
        let reference_grads = flatten_gradients_in_module_order::<TestBackend, _>(
            &reference_model,
            burn_train::TrainStep::step(&reference_model, batch.clone()).grads,
        );
        let split_grads = stage_split_surrogate_gradients(split_model, &plan, batch);
        let mean_abs = mean_abs_diff(&reference_grads, &split_grads);
        let reference_norm = l2_norm(&reference_grads);
        let split_norm = l2_norm(&split_grads);

        assert!(
            mean_abs <= 1.0e-5,
            "surrogate split pipeline gradients drifted from full pipeline reference: mean_abs_diff={mean_abs}"
        );
        assert!(
            (reference_norm - split_norm).abs() <= 1.0e-5,
            "split pipeline gradient norm drifted from reference: reference_norm={reference_norm} split_norm={split_norm}"
        );
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn pipeline_stage_surrogate_mean_across_replicas_matches_full_batch_gradients() {
        let device = burn::tensor::Device::<TestBackend>::default();
        let mut config = tiny_model_config();
        config.n_layer = 2;
        let pipeline = burn_dragon_train::ParallelPipelineConfig {
            enabled: true,
            stage_count: 2,
            virtual_stages_per_rank: 1,
            schedule: burn_dragon_train::PipelineScheduleKind::Interleaved1f1b,
            microbatches: 2,
            ..Default::default()
        };
        let plan = build_pipeline_plan(config.n_layer, &pipeline).expect("plan");
        let reference_model =
            LanguageTrainModel::new(DragonModel::<TestBackend>::new(config.clone(), &device))
                .with_pipeline_plan(Some(plan.clone()));

        let replica_a = make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 0],
            [2, 4],
        );
        let replica_b = make_batch::<TestBackend>(
            &device,
            &[7, 6, 5, 4, 3, 2, 1, 0],
            &[6, 5, 4, 3, 2, 1, 0, 7],
            [2, 4],
        );
        let combined = make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0],
            &[1, 2, 3, 4, 5, 6, 7, 0, 6, 5, 4, 3, 2, 1, 0, 7],
            [4, 4],
        );

        let combined_grads = flatten_gradients_in_module_order::<TestBackend, _>(
            &reference_model,
            burn_train::TrainStep::step(&reference_model, combined).grads,
        );
        let replica_a_grads =
            stage_split_surrogate_gradients(reference_model.clone(), &plan, replica_a);
        let replica_b_grads =
            stage_split_surrogate_gradients(reference_model.clone(), &plan, replica_b);
        let averaged_grads = replica_a_grads
            .iter()
            .zip(replica_b_grads.iter())
            .map(|(lhs, rhs)| (lhs + rhs) * 0.5)
            .collect::<Vec<_>>();

        let mean_abs = mean_abs_diff(&combined_grads, &averaged_grads);
        let combined_norm = l2_norm(&combined_grads);
        let averaged_norm = l2_norm(&averaged_grads);

        assert!(
            mean_abs <= 1.0e-5,
            "replica-averaged split pipeline gradients drifted from combined-batch reference: mean_abs_diff={mean_abs}"
        );
        assert!(
            (combined_norm - averaged_norm).abs() <= 1.0e-5,
            "replica-averaged split pipeline gradient norm drifted from combined-batch reference: combined_norm={combined_norm} averaged_norm={averaged_norm}"
        );
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn process_group_peer_id_uses_global_rank() {
        let runtime = ParallelRuntime {
            mode: ParallelismKind::Ddp,
            world_size: 4,
            global_rank: 3,
            local_rank: 1,
            data_parallel_size: 4,
            local_data_parallel_size: 1,
            tensor_parallel_size: 1,
            process_group_launch: true,
        };

        assert_eq!(process_group_peer_id(&runtime), 3usize.into());
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn process_group_data_shard_uses_data_parallel_rank_when_pipeline_enabled() {
        let runtime = ParallelRuntime {
            mode: ParallelismKind::Ddp,
            world_size: 4,
            global_rank: 3,
            local_rank: 1,
            data_parallel_size: 2,
            local_data_parallel_size: 1,
            tensor_parallel_size: 1,
            process_group_launch: true,
        };
        let config = burn_dragon_train::ParallelConfig {
            mode: ParallelismKind::Ddp,
            world_size: 4,
            data: burn_dragon_train::ParallelDataConfig {
                size: 2,
                ..Default::default()
            },
            pipeline: burn_dragon_train::ParallelPipelineConfig {
                enabled: true,
                stage_count: 2,
                virtual_stages_per_rank: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        let (shard_index, shard_count, assignment, layout) =
            process_group_data_shard(&runtime, &config).expect("pipeline shard");

        assert_eq!(shard_index, 1);
        assert_eq!(shard_count, 2);
        let assignment = assignment.expect("rank assignment");
        let layout = layout.expect("layout");
        assert_eq!(assignment.pipeline_stage_id, 1);
        assert_eq!(assignment.data_parallel_rank, 1);
        assert_eq!(assignment.pipeline_group_ranks, vec![2, 3]);
        assert_eq!(assignment.data_parallel_group_ranks, vec![1, 3]);
        assert_eq!(
            layout.summary(),
            "pipeline_layout=replica_major stage_count=2 virtual_stages_per_rank=1 data_parallel_size=2 world_size=4"
        );
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn fresh_models_use_random_param_ids_but_stable_module_traversal_shapes() {
        #[derive(Default)]
        struct ShapeCollector {
            shapes: Vec<Vec<usize>>,
        }

        impl<B: BackendTrait> burn::module::ModuleVisitor<B> for ShapeCollector {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                self.shapes.push(param.val().shape().dims::<D>().into());
            }
        }

        let device = burn::tensor::Device::<TestBackend>::default();
        let config = tiny_model_config();
        let model_a =
            LanguageTrainModel::new(DragonModel::<TestBackend>::new(config.clone(), &device));
        let model_b = LanguageTrainModel::new(DragonModel::<TestBackend>::new(config, &device));

        let ids_a = list_param_ids(&model_a);
        let ids_b = list_param_ids(&model_b);
        let mut shapes_a = ShapeCollector::default();
        let mut shapes_b = ShapeCollector::default();
        model_a.visit(&mut shapes_a);
        model_b.visit(&mut shapes_b);

        assert_eq!(ids_a.len(), ids_b.len());
        assert_ne!(
            ids_a, ids_b,
            "fresh models should not rely on matching ParamIds"
        );
        assert_eq!(shapes_a.shapes, shapes_b.shapes);
    }
}
