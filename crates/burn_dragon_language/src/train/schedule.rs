use crate::dataset::scheduler::TokenSequenceDataset;
use crate::train::dynamics::{ActiveDynamicsControl, DragonDynamicsControlSlot};
use crate::train::prelude::*;
use crate::train::runtime_checkpoint::{
    load_runtime_state_checkpoint, prepare_predictive_coding_checkpoint_contract,
    save_runtime_state_checkpoint, synchronize_predictive_coding_checkpoint_manifests,
};
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
use rayon::prelude::*;
#[cfg(feature = "ddp")]
use std::collections::HashMap;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::io::{BufRead, BufReader, Write};
#[cfg(feature = "ddp")]
use std::marker::PhantomData;

const CHECKPOINT_KEEP_LAST: usize = 2;
const METRIC_LOSS: &str = "Loss";
const METRIC_STREAM_WARM_LOSS: &str = "Stream Warm Loss";
const METRIC_RANDOM_COLD_LOSS: &str = "Random Cold Loss";
const METRIC_STREAM_PAIRED_WARM_LOSS: &str = "Stream Paired Warm Loss";
const METRIC_STREAM_PAIRED_COLD_LOSS: &str = "Stream Paired Cold Loss";
const METRIC_STREAM_CARRY_NLL_GAIN: &str = "Stream Carry NLL Gain";
const METRIC_STREAM_CARRY_RELATIVE_GAIN: &str = "Stream Carry Relative Gain";
const METRIC_STREAM_CARRY_PROBE_BATCHES: &str = "Stream Carry Probe Batches";
const METRIC_VALIDATION_OBJECTIVE_LOSS: &str = "Validation Objective Loss";
const METRIC_RHO_RMS: &str = "Sequence State Rho RMS";
const METRIC_RHO_SLOT_VARIANCE_RATIO: &str = "Sequence State Rho Slot Variance Ratio";
const METRIC_RHO_SLOT_REDUNDANCY: &str = "Sequence State Rho Slot Redundancy";
const METRIC_RHO_LAYERS: &str = "Sequence State Rho Layers";

fn epoch_end_absolute_step(
    epoch: usize,
    logical_steps_per_epoch: usize,
    completed_steps_in_epoch: usize,
) -> usize {
    epoch
        .saturating_sub(1)
        .saturating_mul(logical_steps_per_epoch)
        .saturating_add(completed_steps_in_epoch.saturating_sub(1))
}

fn training_interruption_reason(interrupter: &burn_train::Interrupter) -> Option<String> {
    interrupter.should_stop().then(|| {
        interrupter
            .get_message()
            .unwrap_or_else(|| "training interrupted".to_string())
    })
}

#[derive(Clone, Copy)]
struct TrainingEventContext<'a> {
    epoch: usize,
    absolute_step: usize,
    bus: &'a TrainingEventBus,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(default)]
struct ContinualLearningStabilityState {
    best_valid_loss: Option<f64>,
    best_checkpoint_epoch: Option<usize>,
    best_ruliad_competence: Option<RuliadCompetenceKey>,
    best_ruliad_recovery_competence: Option<RuliadCompetenceKey>,
    best_ruliad_policy_competence: Option<RuliadPolicyCompetenceKey>,
    best_ruliad_policy_recovery_competence: Option<RuliadPolicyCompetenceKey>,
    best_ruliad_policy_observed_competence: Option<RuliadPolicyCompetenceKey>,
    best_ruliad_policy_solve_observation: Option<BinomialObservation>,
    best_ruliad_policy_goal_observation: Option<BinomialObservation>,
    best_ruliad_policy_valid_action_observation: Option<BinomialObservation>,
    best_ruliad_checkpoint_epoch: Option<usize>,
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
    objective: crate::config::TrainingValidationObjective,
    loss: f64,
    source_weighted_loss: Option<f64>,
    stream_warm_loss: Option<f64>,
    output_degeneracy: Option<crate::train::steps::OutputDegeneracyStats>,
    ruliad_eval_report: Option<burn_dragon_universality::RuliadEvalReport>,
    ruliad_policy_rollout: Option<RuliadPolicyRolloutProbeResult>,
}

struct DynamicValidation<'a, 'env, B: AutodiffBackend> {
    env: &'a TrainEnvironment<'env, B>,
    valid_loader: &'a Arc<dyn DataLoader<ValidBackend<B>, SequenceBatch<ValidBackend<B>>>>,
    model: &'a LanguageTrainModel<B>,
    batch_size: usize,
    bus: &'a TrainingEventBus,
    context_routing: Option<&'a crate::train::PredictiveContextRoutingRuntime<B>>,
}

#[derive(Clone, Copy, Debug, Default)]
struct StreamWarmValidationReport {
    warm_loss: Option<f64>,
    paired_warm_loss: Option<f64>,
    paired_cold_loss: Option<f64>,
    carry_nll_gain: Option<f64>,
    carry_relative_gain: Option<f64>,
    paired_batches: usize,
}

impl DynamicValidationReport {
    fn primary_loss(&self) -> f64 {
        match self.objective {
            crate::config::TrainingValidationObjective::FixedHoldout => self.loss,
            crate::config::TrainingValidationObjective::SourceWeighted => {
                self.source_weighted_loss.unwrap_or(f64::NAN)
            }
            crate::config::TrainingValidationObjective::StreamWarm => {
                self.stream_warm_loss.unwrap_or(f64::NAN)
            }
        }
    }
}

fn select_validation_objective_loss(
    objective: crate::config::TrainingValidationObjective,
    fixed_holdout_loss: f64,
    source_weighted_loss: Option<f64>,
    stream_warm_loss: Option<f64>,
) -> Result<f64> {
    let loss = match objective {
        crate::config::TrainingValidationObjective::FixedHoldout => Some(fixed_holdout_loss),
        crate::config::TrainingValidationObjective::SourceWeighted => source_weighted_loss,
        crate::config::TrainingValidationObjective::StreamWarm => stream_warm_loss,
    };
    loss.filter(|loss| loss.is_finite()).ok_or_else(|| {
        anyhow::anyhow!(
            "validation objective `{}` did not produce a finite loss",
            objective.as_str()
        )
    })
}

fn emit_validation_objective_loss(
    run_name: &str,
    epoch: usize,
    absolute_step: usize,
    objective: crate::config::TrainingValidationObjective,
    loss: f64,
    bus: &TrainingEventBus,
) {
    let _ = bus.send_metric_sample(TrainingMetricSample {
        run_id: run_name.to_string().into(),
        split: TrainingMetricSplit::Valid,
        epoch,
        step_in_epoch: 0,
        absolute_step,
        name: METRIC_VALIDATION_OBJECTIVE_LOSS.to_string(),
        value: loss,
        running_value: loss,
    });
    info!(
        "valid epoch={} objective={} loss={loss:.6}",
        epoch,
        objective.as_str()
    );
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
struct RuliadCompetenceKey {
    verifier_ppm: u32,
    semantic_ppm: u32,
    partial_ppm: u32,
    certificate_ppm: u32,
    completion_health_ppm: u32,
}

/// Lexicographic deployment-quality key for verifier-constrained proof search.
///
/// Exact solve rate is primary, followed by goal completion and action validity. The remaining
/// fields break ties without collapsing semantically different failure modes into an arbitrary
/// weighted scalar.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
struct RuliadPolicyCompetenceKey {
    solve_ppm: u32,
    goal_completion_ppm: u32,
    valid_action_ppm: u32,
    expert_top1_ppm: u32,
    non_repeated_state_ppm: u32,
    non_backtrack_ppm: u32,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
struct BinomialObservation {
    successes: usize,
    trials: usize,
}

impl BinomialObservation {
    fn rate(self) -> f64 {
        ratio_usize(self.successes, self.trials)
    }

    fn prefer_stronger_evidence(self, candidate: Self, z: f64) -> Self {
        let self_lower = wilson_score_interval(self, z)
            .map(|(lower, _)| lower)
            .unwrap_or_default();
        let candidate_lower = wilson_score_interval(candidate, z)
            .map(|(lower, _)| lower)
            .unwrap_or_default();
        if candidate_lower > self_lower
            || (candidate_lower == self_lower && candidate.rate() > self.rate())
            || (candidate_lower == self_lower
                && candidate.rate() == self.rate()
                && candidate.trials > self.trials)
        {
            candidate
        } else {
            self
        }
    }
}

impl RuliadPolicyCompetenceKey {
    fn componentwise_max(self, other: Self) -> Self {
        Self {
            solve_ppm: self.solve_ppm.max(other.solve_ppm),
            goal_completion_ppm: self.goal_completion_ppm.max(other.goal_completion_ppm),
            valid_action_ppm: self.valid_action_ppm.max(other.valid_action_ppm),
            expert_top1_ppm: self.expert_top1_ppm.max(other.expert_top1_ppm),
            non_repeated_state_ppm: self
                .non_repeated_state_ppm
                .max(other.non_repeated_state_ppm),
            non_backtrack_ppm: self.non_backtrack_ppm.max(other.non_backtrack_ppm),
        }
    }
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
    let status_health = item_count.saturating_sub(unhealthy_count) as f32 / item_count as f32;
    let completion_health = status_health
        * report.mean_completion_quality.clamp(0.0, 1.0)
        * report.answer_field_coverage.clamp(0.0, 1.0);
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
    // Dashboard-only bounded lexicographic score. Checkpoint promotion compares the key directly.
    const SCALE: f64 = 1_000_001.0;
    let verifier = f64::from(key.verifier_ppm) / 1_000_000.0;
    let semantic = f64::from(key.semantic_ppm) / 1_000_000.0;
    let partial = f64::from(key.partial_ppm) / 1_000_000.0;
    let certificate = f64::from(key.certificate_ppm) / 1_000_000.0;
    let completion_health = f64::from(key.completion_health_ppm) / 1_000_000.0;
    verifier
        + semantic / SCALE
        + partial / SCALE.powi(2)
        + certificate / SCALE.powi(3)
        + completion_health / SCALE.powi(4)
}

fn ruliad_policy_competence_key(
    result: &RuliadPolicyRolloutProbeResult,
) -> Option<RuliadPolicyCompetenceKey> {
    let summary = result.summary;
    let attempted_actions = summary
        .valid_actions
        .saturating_add(summary.invalid_actions);
    if summary.items == 0 || summary.scored_states == 0 || attempted_actions == 0 {
        return None;
    }
    let ppm = |numerator, denominator| ratio_to_ppm(ratio_usize(numerator, denominator) as f32);
    Some(RuliadPolicyCompetenceKey {
        solve_ppm: ppm(summary.solved, summary.items),
        goal_completion_ppm: ppm(summary.solved_goals, summary.total_goals),
        valid_action_ppm: ppm(summary.valid_actions, attempted_actions),
        expert_top1_ppm: ppm(summary.top1_expert_actions, summary.scored_states),
        non_repeated_state_ppm: 1_000_000u32
            .saturating_sub(ppm(summary.repeated_states, summary.valid_actions)),
        non_backtrack_ppm: 1_000_000u32.saturating_sub(ppm(
            summary.backtracks,
            summary.valid_actions.saturating_add(summary.backtracks),
        )),
    })
}

fn ruliad_policy_observations(
    summary: RuliadPolicyRolloutProbeSummary,
) -> (
    BinomialObservation,
    BinomialObservation,
    BinomialObservation,
) {
    (
        BinomialObservation {
            successes: summary.solved,
            trials: summary.items,
        },
        BinomialObservation {
            successes: summary.solved_goals,
            trials: summary.total_goals,
        },
        BinomialObservation {
            successes: summary.valid_actions,
            trials: summary
                .valid_actions
                .saturating_add(summary.invalid_actions),
        },
    )
}

fn wilson_score_interval(observation: BinomialObservation, z: f64) -> Option<(f64, f64)> {
    if observation.trials == 0 || !z.is_finite() || z <= 0.0 {
        return None;
    }
    let n = observation.trials as f64;
    let p = observation.successes.min(observation.trials) as f64 / n;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let radius = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;
    Some(((center - radius).max(0.0), (center + radius).min(1.0)))
}

fn binomial_observation_materially_regressed(
    best: BinomialObservation,
    current: BinomialObservation,
    minimum_best_rate: f64,
    z: f64,
) -> bool {
    if best.rate() < minimum_best_rate {
        return false;
    }
    let (Some((best_lower, _)), Some((_, current_upper))) = (
        wilson_score_interval(best, z),
        wilson_score_interval(current, z),
    ) else {
        return false;
    };
    current_upper < best_lower
}

fn competence_order<T: Copy + Ord>(current: T, best: Option<T>) -> Option<bool> {
    match best {
        None => Some(true),
        Some(best) if current > best => Some(true),
        Some(best) if current < best => Some(false),
        Some(_) => None,
    }
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
    let status_health_rate = if report.item_count == 0 {
        0.0
    } else {
        report.item_count.saturating_sub(unhealthy_count) as f64 / report.item_count as f64
    };
    let completion_health_rate = status_health_rate
        * f64::from(report.mean_completion_quality.clamp(0.0, 1.0))
        * f64::from(report.answer_field_coverage.clamp(0.0, 1.0));
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
    if report.verifier_accuracy <= f32::EPSILON {
        reasons.push("verifier_rate=0.000<=0".to_string());
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
    if report.scored_count > 1
        && report.expected_answer_distinct_fraction
            >= gates.capability_answer_distinct_min_fraction as f32
        && report.actual_answer_distinct_fraction
            < gates.capability_answer_distinct_min_fraction as f32
    {
        reasons.push(format!(
            "answer_distinct={:.3}<{} with expected_distinct={:.3}",
            report.actual_answer_distinct_fraction,
            gates.capability_answer_distinct_min_fraction,
            report.expected_answer_distinct_fraction
        ));
    }
    if report.field_value_distinct_ratio < gates.capability_field_value_distinct_ratio_min as f32 {
        reasons.push(format!(
            "field_value_distinct_ratio={:.3}<{}",
            report.field_value_distinct_ratio, gates.capability_field_value_distinct_ratio_min
        ));
    }
    if report.actual_field_value_dominant_fraction
        > gates.capability_field_value_dominance_max as f32
    {
        reasons.push(format!(
            "field_value_dominance={:.3}>{}",
            report.actual_field_value_dominant_fraction, gates.capability_field_value_dominance_max
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

fn ruliad_completion_quality_collapse(
    report: &burn_dragon_universality::RuliadEvalReport,
    gates: &burn_dragon_train::TrainingGatesConfig,
) -> bool {
    if report.item_count == 0 || report.scored_count == 0 {
        return true;
    }
    let (_, _, _, completion_health_rate) = ruliad_capability_rates(report);
    completion_health_rate < gates.capability_completion_health_min_rate
        || (report.scored_count > 1
            && report.expected_answer_distinct_fraction
                >= gates.capability_answer_distinct_min_fraction as f32
            && report.actual_answer_distinct_fraction
                < gates.capability_answer_distinct_min_fraction as f32)
        || report.field_value_distinct_ratio
            < gates.capability_field_value_distinct_ratio_min as f32
        || report.actual_field_value_dominant_fraction
            > gates.capability_field_value_dominance_max as f32
}

fn validation_capability_gate_status(
    validation: &DynamicValidationReport,
    training: &crate::config::TrainingHyperparameters,
) -> RuliadCapabilityGateStatus {
    ruliad_deployment_capability_gate_status(
        validation.ruliad_eval_report.as_ref(),
        validation.ruliad_policy_rollout.as_ref(),
        validation.output_degeneracy.as_ref(),
        training,
    )
}

fn ruliad_deployment_capability_gate_status(
    free_run: Option<&burn_dragon_universality::RuliadEvalReport>,
    closed_loop_policy: Option<&RuliadPolicyRolloutProbeResult>,
    output_degeneracy: Option<&crate::train::steps::OutputDegeneracyStats>,
    training: &crate::config::TrainingHyperparameters,
) -> RuliadCapabilityGateStatus {
    let contract = training.ruliad_policy_probe.checkpoint_capability_contract;
    let mut reasons = Vec::new();
    if contract.requires_free_run() {
        match free_run {
            Some(report) => reasons.extend(
                ruliad_capability_gate_status(report, output_degeneracy, &training.gates)
                    .reasons
                    .into_iter()
                    .map(|reason| format!("free_run:{reason}")),
            ),
            None => reasons.push("free_run:missing_probe".to_string()),
        }
    }
    if contract.requires_closed_loop_policy() {
        match closed_loop_policy {
            Some(result) => reasons.extend(
                ruliad_policy_promotion_gate_status(
                    result.summary,
                    training.ruliad_policy_probe.promotion_gate,
                )
                .reasons
                .into_iter()
                .map(|reason| format!("closed_loop_policy:{reason}")),
            ),
            None => reasons.push("closed_loop_policy:missing_probe".to_string()),
        }
    }
    RuliadCapabilityGateStatus {
        passed: reasons.is_empty(),
        reasons,
    }
}

fn emit_ruliad_deployment_capability_gate_metrics(
    run_name: &str,
    epoch: usize,
    absolute_step: usize,
    status: &RuliadCapabilityGateStatus,
    bus: &TrainingEventBus,
) {
    for (name, value) in [
        (
            "Ruliad Deployment Capability Gate Passed",
            if status.passed { 1.0 } else { 0.0 },
        ),
        (
            "Ruliad Deployment Capability Gate Failure Count",
            status.reasons.len() as f64,
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
}

fn validation_capability_quality_collapse(
    validation: &DynamicValidationReport,
    training: &crate::config::TrainingHyperparameters,
) -> bool {
    let contract = training.ruliad_policy_probe.checkpoint_capability_contract;
    let free_run_collapsed = contract.requires_free_run()
        && validation
            .ruliad_eval_report
            .as_ref()
            .is_some_and(|report| ruliad_completion_quality_collapse(report, &training.gates));
    let policy_collapsed = contract.requires_closed_loop_policy()
        && validation
            .ruliad_policy_rollout
            .as_ref()
            .is_some_and(|result| {
                let summary = result.summary;
                let attempted_actions = summary
                    .valid_actions
                    .saturating_add(summary.invalid_actions);
                summary.items == 0
                    || summary.scored_states == 0
                    || attempted_actions == 0
                    || summary.valid_actions == 0
            });
    free_run_collapsed || policy_collapsed
}

fn validation_ruliad_capability_improved(
    validation: &DynamicValidationReport,
    state: &ContinualLearningStabilityState,
    contract: crate::config::RuliadCheckpointCapabilityContract,
) -> bool {
    let free_improved = validation
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
        });
    let policy_improved = validation
        .ruliad_policy_rollout
        .as_ref()
        .and_then(ruliad_policy_competence_key)
        .is_some_and(|current| {
            state
                .best_ruliad_policy_observed_competence
                .is_some_and(|best| current > best)
        });
    match contract {
        crate::config::RuliadCheckpointCapabilityContract::FreeRunText => free_improved,
        crate::config::RuliadCheckpointCapabilityContract::ClosedLoopPolicy => policy_improved,
        crate::config::RuliadCheckpointCapabilityContract::Joint => {
            let free_non_regressing = validation
                .ruliad_eval_report
                .as_ref()
                .and_then(ruliad_competence_key)
                .is_some_and(|current| {
                    state
                        .best_ruliad_competence
                        .is_none_or(|best| current >= best)
                });
            let policy_non_regressing = validation
                .ruliad_policy_rollout
                .as_ref()
                .and_then(ruliad_policy_competence_key)
                .is_some_and(|current| {
                    state
                        .best_ruliad_policy_observed_competence
                        .is_none_or(|best| current >= best)
                });
            free_non_regressing && policy_non_regressing && (free_improved || policy_improved)
        }
    }
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
    validation: &DynamicValidationReport,
    state: &mut ContinualLearningStabilityState,
    recovery_requested: &mut bool,
    event: TrainingEventContext<'_>,
) where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let epoch = event.epoch;
    let gates = &env.training.gates;
    if !gates.enabled {
        return;
    }
    let status = validation_capability_gate_status(validation, env.training);
    if status.passed {
        if state.first_capability_pass_epoch.is_none() {
            state.first_capability_pass_epoch = Some(epoch);
            emit_policy_gate_with_action(
                env,
                "continual_learning_capability_gate_first_pass",
                TrainingGateAction::Alert,
                TrainingGateSeverity::Info,
                "ruliad capability gate passed for the first time; capability-gated auxiliaries and regression tracking may open".to_string(),
                event,
            );
        }
        state.last_capability_pass_epoch = Some(epoch);
        state.consecutive_capability_gate_failures = 0;
        return;
    }

    let reasons = status.reasons.join(", ");
    let quality_collapse = validation_capability_quality_collapse(validation, env.training);
    let before_grace =
        state.first_capability_pass_epoch.is_none() && epoch <= gates.capability_grace_epochs;
    if before_grace {
        if quality_collapse && !*recovery_requested {
            let message = format!(
                "ruliad completion quality/diversity collapsed during capability grace window; requesting SourceCapabilityRecovery while keeping continual backprop active: {reasons}"
            );
            emit_policy_gate_with_action(
                env,
                "continual_learning_capability_quality_recovery",
                TrainingGateAction::Alert,
                TrainingGateSeverity::Warning,
                message.clone(),
                event,
            );
            emit_dynamics_control(
                env,
                &env.training.dynamics,
                DynamicsMode::SourceCapabilityRecovery,
                None,
                message,
                event,
            );
            *recovery_requested = true;
        }
        emit_policy_gate_with_action(
            env,
            "continual_learning_capability_gate_grace",
            TrainingGateAction::Alert,
            TrainingGateSeverity::Info,
            format!(
                "ruliad capability gate has not passed during grace window epoch {epoch}/{}: {reasons}",
                gates.capability_grace_epochs
            ),
            event,
        );
        return;
    }

    state.consecutive_capability_gate_failures =
        state.consecutive_capability_gate_failures.saturating_add(1);
    let failures = state.consecutive_capability_gate_failures;
    if !quality_collapse {
        emit_policy_gate_with_action(
            env,
            "continual_learning_checkpoint_promotion_ineligible",
            TrainingGateAction::Alert,
            TrainingGateSeverity::Info,
            format!(
                "checkpoint is ineligible for deployment promotion ({failures} consecutive validation epochs), but no hard capability collapse was observed: {reasons}"
            ),
            event,
        );
        return;
    }

    let require_failure_recovery =
        !gates.capability_required_after_first_pass || state.first_capability_pass_epoch.is_some();
    if !require_failure_recovery {
        emit_policy_gate_with_action(
            env,
            "continual_learning_capability_gate_not_ready",
            TrainingGateAction::Alert,
            TrainingGateSeverity::Warning,
            format!("ruliad capability gate still has not passed: {reasons}"),
            event,
        );
        return;
    }

    emit_policy_gate_with_action(
        env,
        "continual_learning_capability_quality_regression",
        TrainingGateAction::Alert,
        TrainingGateSeverity::Warning,
        format!(
            "ruliad capability quality collapsed after capability was required ({failures}/{}): {reasons}",
            gates.capability_regression_patience_epochs
        ),
        event,
    );
    if failures < gates.capability_regression_patience_epochs || *recovery_requested {
        return;
    }

    let rollback_epoch = capability_rollback_checkpoint_epoch(state);
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
        &env.training.dynamics,
        mode,
        rollback_epoch,
        message,
        event,
    );
    *recovery_requested = true;
}

fn capability_rollback_checkpoint_epoch(state: &ContinualLearningStabilityState) -> Option<usize> {
    state
        .best_ruliad_checkpoint_epoch
        .or(state.best_checkpoint_epoch)
}

fn update_ruliad_recovery_competence(
    validation: &DynamicValidationReport,
    contract: crate::config::RuliadCheckpointCapabilityContract,
    policy_gate: crate::config::RuliadPolicyPromotionGateConfig,
    gates: &burn_dragon_train::TrainingGatesConfig,
    best_free_run: &mut Option<RuliadCompetenceKey>,
    best_policy: &mut Option<RuliadPolicyCompetenceKey>,
) -> bool {
    let free_candidate = || {
        let report = validation.ruliad_eval_report.as_ref()?;
        let competence = ruliad_competence_key(report)?;
        if (!competence.has_free_run_correctness()
            && competence.partial_ppm == 0
            && competence.certificate_ppm == 0)
            || !ruliad_capability_gate_status(report, validation.output_degeneracy.as_ref(), gates)
                .passed
            || validation
                .output_degeneracy
                .as_ref()
                .is_some_and(|stats| hard_output_collapse_for_gates(gates, stats))
        {
            return None;
        }
        Some(competence)
    };
    let policy_candidate = || {
        let rollout = validation.ruliad_policy_rollout.as_ref()?;
        let competence = ruliad_policy_competence_key(rollout)?;
        ruliad_policy_promotion_gate_status(rollout.summary, policy_gate)
            .passed
            .then_some(competence)
    };

    match contract {
        crate::config::RuliadCheckpointCapabilityContract::FreeRunText => {
            let Some(competence) = free_candidate() else {
                return false;
            };
            if best_free_run.is_some_and(|best| competence <= best) {
                return false;
            }
            *best_free_run = Some(competence);
        }
        crate::config::RuliadCheckpointCapabilityContract::ClosedLoopPolicy => {
            let Some(competence) = policy_candidate() else {
                return false;
            };
            if best_policy.is_some_and(|best| competence <= best) {
                return false;
            }
            *best_policy = Some(competence);
        }
        crate::config::RuliadCheckpointCapabilityContract::Joint => {
            let (Some(free), Some(policy)) = (free_candidate(), policy_candidate()) else {
                return false;
            };
            let free_order = competence_order(free, *best_free_run);
            let policy_order = competence_order(policy, *best_policy);
            if matches!(free_order, Some(false))
                || matches!(policy_order, Some(false))
                || (free_order.is_none() && policy_order.is_none())
            {
                return false;
            }
            *best_free_run = Some(free);
            *best_policy = Some(policy);
        }
    }
    true
}

fn update_ruliad_recovery_checkpoint<B>(
    env: &TrainEnvironment<'_, B>,
    validation: &DynamicValidationReport,
    epoch: usize,
    state: &mut ContinualLearningStabilityState,
) -> bool
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    if !update_ruliad_recovery_competence(
        validation,
        env.training
            .ruliad_policy_probe
            .checkpoint_capability_contract,
        env.training.ruliad_policy_probe.promotion_gate,
        &env.training.gates,
        &mut state.best_ruliad_recovery_competence,
        &mut state.best_ruliad_policy_recovery_competence,
    ) {
        return false;
    }
    state.best_ruliad_checkpoint_epoch = Some(epoch);
    true
}

fn should_promote_checkpoint(
    validation: &DynamicValidationReport,
    best_loss: Option<f64>,
    best_free_run_competence: Option<RuliadCompetenceKey>,
    best_policy_competence: Option<RuliadPolicyCompetenceKey>,
    training: &crate::config::TrainingHyperparameters,
) -> bool {
    let loss_improved = best_loss.is_none_or(|best| validation.primary_loss() < best);
    let contract = training.ruliad_policy_probe.checkpoint_capability_contract;

    let free_run = || {
        let report = validation.ruliad_eval_report.as_ref()?;
        let competence = ruliad_competence_key(report)?;
        if !ruliad_capability_gate_status(
            report,
            validation.output_degeneracy.as_ref(),
            &training.gates,
        )
        .passed
        {
            return None;
        }
        Some((competence, best_free_run_competence))
    };
    let policy = || {
        let rollout = validation.ruliad_policy_rollout.as_ref()?;
        let competence = ruliad_policy_competence_key(rollout)?;
        if !ruliad_policy_promotion_gate_status(
            rollout.summary,
            training.ruliad_policy_probe.promotion_gate,
        )
        .passed
        {
            return None;
        }
        Some((competence, best_policy_competence))
    };
    match contract {
        crate::config::RuliadCheckpointCapabilityContract::FreeRunText => {
            let Some((current, best)) = free_run() else {
                // Preserve historical behavior when a non-Ruliad run has no capability report.
                return validation.ruliad_eval_report.is_none() && loss_improved;
            };
            competence_order(current, best).unwrap_or_else(|| {
                current.has_free_run_correctness()
                    && validation
                        .output_degeneracy
                        .as_ref()
                        .is_none_or(|stats| !output_degeneracy_tripped(&training.gates, stats))
                    && loss_improved
            })
        }
        crate::config::RuliadCheckpointCapabilityContract::ClosedLoopPolicy => {
            let Some((current, best)) = policy() else {
                return false;
            };
            competence_order(current, best).unwrap_or(loss_improved)
        }
        crate::config::RuliadCheckpointCapabilityContract::Joint => {
            let (Some((current_free, best_free)), Some((current_policy, best_policy))) =
                (free_run(), policy())
            else {
                return false;
            };
            let free_order = competence_order(current_free, best_free);
            let policy_order = competence_order(current_policy, best_policy);
            if matches!(free_order, Some(false)) || matches!(policy_order, Some(false)) {
                return false;
            }
            matches!(free_order, Some(true))
                || matches!(policy_order, Some(true))
                || (current_free.has_free_run_correctness() && loss_improved)
        }
    }
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
    step_in_epoch: usize,
    value: f64,
    running_value: f64,
    event: TrainingEventContext<'_>,
) {
    let TrainingEventContext {
        epoch,
        absolute_step,
        bus,
    } = event;
    let Some(name) = teacher_forced_validation_metric_name(source_selection_dataset) else {
        return;
    };
    let _ = bus.send_metric_sample(TrainingMetricSample {
        run_id: run_name.to_string().into(),
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

fn prepare_local_predictive_coding_contract<B>(
    env: &TrainEnvironment<'_, B>,
    model: &LanguageTrainModel<B>,
) -> Result<Option<burn_pc::PcCheckpointManifest>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let manifest = model.predictive_coding_checkpoint_manifest();
    let require_exact = matches!(
        env.training.launch_mode,
        burn_dragon_train::train::pipeline::TrainingLaunchMode::ResumeExactRun
    );
    prepare_predictive_coding_checkpoint_contract(
        env.run_dir,
        env.resume_checkpoint_epoch,
        manifest.as_ref(),
        require_exact,
    )?;
    Ok(manifest)
}

mod distributed;
mod dynamic;
mod dynamics;
mod eggroll;
mod latent_validation;
mod ruliad_evaluation;
mod ruliad_rollout;
mod ruliad_validation;
mod telemetry;

use distributed::*;
use dynamic::*;
use dynamics::*;
use latent_validation::*;
use ruliad_evaluation::*;
use ruliad_rollout::*;
use ruliad_validation::*;
use telemetry::*;

#[cfg(test)]
use eggroll::*;

pub use distributed::{resolve_lr_scheduler, resolve_train_schedule};
pub(crate) use dynamic::train_with_dynamic_neuron_scaling_scheduler;
pub(crate) use eggroll::{
    autotune_eggroll_population_chunk_size, train_with_eggroll_forward_only, train_with_scheduler,
};
pub use ruliad_evaluation::{RuliadModelEvaluation, evaluate_ruliad_model_free_run};

#[cfg(test)]
mod tests;
