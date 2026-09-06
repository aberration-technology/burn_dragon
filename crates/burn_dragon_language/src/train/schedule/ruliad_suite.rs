//! Checkpoint-isolated Ruliad capability evaluation.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use super::*;

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct RuliadEvaluationSuiteOptions {
    pub panel_seed: u64,
    pub free_run_items: usize,
    pub policy_items: usize,
    pub difficulty_levels: usize,
    pub training_batch_size: usize,
    pub include_closed_loop_rollout: bool,
    pub epoch: usize,
    pub absolute_step: usize,
    pub dataset_name: String,
}

impl Default for RuliadEvaluationSuiteOptions {
    fn default() -> Self {
        Self {
            panel_seed: 0,
            free_run_items: 32,
            policy_items: 32,
            difficulty_levels: 4,
            training_batch_size: 32,
            include_closed_loop_rollout: true,
            epoch: 0,
            absolute_step: 0,
            dataset_name: "ruliad_checkpoint_evaluation".to_string(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, PartialEq)]
pub struct RuliadConstrainedPolicyEvaluation {
    pub items: usize,
    pub equivalent_top1_rate: f64,
    pub preferred_top1_rate: f64,
    pub equivalent_nll: f64,
    pub valid_invalid_margin: f64,
    pub canonical_equivalent_top1_rate: f64,
    pub canonical_preferred_top1_rate: f64,
    pub worst_presentation_equivalent_top1_rate: f64,
    pub orbit_js_divergence: f64,
    pub orbit_top1_consensus_fraction: f64,
    pub context_swap_top1_change_rate: f64,
    pub context_swap_equivalent_probability_drop: f64,
    pub counterfactual_target_top1_change_rate: f64,
    pub counterfactual_target_equivalent_probability_gain: f64,
    pub elapsed_ms: f64,
}

#[derive(Clone, Copy, Debug, Default, Serialize, PartialEq)]
pub struct RuliadPolicyRolloutEvaluation {
    pub items: usize,
    pub solve_rate: f64,
    pub goal_completion_rate: f64,
    pub valid_action_rate: f64,
    pub invalid_action_rate: f64,
    pub repeated_state_rate: f64,
    pub backtrack_rate: f64,
    pub mean_backtracks: f64,
    pub top1_expert_rate: f64,
    pub frontier_exhaustion_rate: f64,
    pub mean_steps: f64,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct RuliadEvaluationSuiteReport {
    pub version: u32,
    /// Full floating-point tensor identity, checked again after all probes.
    pub model_tensor_fingerprint_sha256: String,
    pub panel_fingerprint_sha256: String,
    /// Free generation from the canonical transfer prompt, which is not copied
    /// verbatim from the training serialization.
    pub free_run: RuliadModelEvaluation,
    /// Free generation from the exact training-document answer slot. Comparing
    /// this with `free_run` separates decoder learning from schema transfer.
    pub training_serialization_free_run: RuliadModelEvaluation,
    /// Free generation from the same verifier-derived prompt contract used by
    /// the constrained proof policy. This isolates action rendering from
    /// document-prompt transfer.
    pub policy_context_free_run: RuliadModelEvaluation,
    /// Verifier-enumerated action selection followed by deterministic rendering.
    /// This is a typed deployment path, not unconstrained autoregressive generation.
    pub structured_policy_decode: RuliadStructuredPolicyEvaluation,
    pub policy_controls: RuliadPolicyControlEvaluation,
    pub constrained_policy: RuliadConstrainedPolicyEvaluation,
    pub constrained_policy_by_difficulty: BTreeMap<usize, RuliadConstrainedPolicyEvaluation>,
    pub constrained_policy_by_source: BTreeMap<String, RuliadConstrainedPolicyEvaluation>,
    pub closed_loop_rollout: Option<RuliadPolicyRolloutEvaluation>,
    pub rollout_by_difficulty: BTreeMap<usize, RuliadPolicyRolloutEvaluation>,
    pub rollout_by_source: BTreeMap<String, RuliadPolicyRolloutEvaluation>,
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn constrained_evaluation(
    summary: RuliadCorrectnessConstrainedPolicySummary,
) -> RuliadConstrainedPolicyEvaluation {
    RuliadConstrainedPolicyEvaluation {
        items: summary.items,
        equivalent_top1_rate: ratio(summary.equivalent_top1, summary.items),
        preferred_top1_rate: ratio(summary.preferred_top1, summary.items),
        equivalent_nll: summary.equivalent_nll_sum / summary.items.max(1) as f64,
        valid_invalid_margin: summary.valid_invalid_margin_sum
            / summary.valid_invalid_margin_items.max(1) as f64,
        canonical_equivalent_top1_rate: ratio(
            summary.canonical_equivalent_top1,
            summary.canonical_items,
        ),
        canonical_preferred_top1_rate: ratio(
            summary.canonical_preferred_top1,
            summary.canonical_items,
        ),
        worst_presentation_equivalent_top1_rate: ratio(
            summary.worst_presentation_equivalent_top1,
            summary.worst_presentation_items,
        ),
        orbit_js_divergence: summary.orbit_js_divergence_sum / summary.items.max(1) as f64,
        orbit_top1_consensus_fraction: summary.orbit_top1_consensus_fraction_sum
            / summary.items.max(1) as f64,
        context_swap_top1_change_rate: ratio(
            summary.context_swap_top1_changes,
            summary.context_swap_items,
        ),
        context_swap_equivalent_probability_drop: summary
            .context_swap_equivalent_probability_drop_sum
            / summary.context_swap_items.max(1) as f64,
        counterfactual_target_top1_change_rate: ratio(
            summary.counterfactual_target_top1_changes,
            summary.counterfactual_target_items,
        ),
        counterfactual_target_equivalent_probability_gain: summary
            .counterfactual_target_equivalent_probability_gain_sum
            / summary.counterfactual_target_items.max(1) as f64,
        elapsed_ms: summary.elapsed_ms,
    }
}

fn rollout_evaluation(summary: RuliadPolicyRolloutProbeSummary) -> RuliadPolicyRolloutEvaluation {
    let attempted_actions = summary
        .valid_actions
        .saturating_add(summary.invalid_actions);
    RuliadPolicyRolloutEvaluation {
        items: summary.items,
        solve_rate: ratio(summary.solved, summary.items),
        goal_completion_rate: ratio(summary.solved_goals, summary.total_goals),
        valid_action_rate: ratio(summary.valid_actions, attempted_actions),
        invalid_action_rate: ratio(summary.invalid_actions, attempted_actions),
        repeated_state_rate: ratio(summary.repeated_states, summary.valid_actions),
        backtrack_rate: ratio(
            summary.backtracks,
            summary.valid_actions.saturating_add(summary.backtracks),
        ),
        mean_backtracks: ratio(summary.backtracks, summary.items),
        top1_expert_rate: ratio(summary.top1_expert_actions, summary.scored_states),
        frontier_exhaustion_rate: ratio(summary.frontier_exhaustions, summary.items),
        mean_steps: ratio(summary.steps, summary.items),
    }
}

fn panel_fingerprint(
    base_items: &[crate::dataset::RuliadValidationProbeItem],
    training_serialization_items: &[crate::dataset::RuliadValidationProbeItem],
    policy_items: &[crate::dataset::RuliadValidationProbeItem],
    policy_context_items: &[crate::dataset::RuliadValidationProbeItem],
) -> Result<String> {
    let bytes = serde_json::to_vec(&(
        base_items,
        training_serialization_items,
        policy_items,
        policy_context_items,
    ))
    .context("serialize checkpoint-evaluation Ruliad panel")?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

#[allow(clippy::too_many_arguments)]
fn evaluate_free_run_panel<B>(
    dataset: &Dataset,
    model: &LanguageTrainModel<B>,
    training: &TrainingHyperparameters,
    options: &RuliadEvaluationSuiteOptions,
    items: &[crate::dataset::RuliadValidationProbeItem],
    dataset_name: &str,
    device: &B::Device,
) -> Result<RuliadModelEvaluation>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    let evaluation = evaluate_ruliad_correctness_validation_for_items_core(
        None,
        None,
        dataset,
        model,
        options.epoch,
        options.absolute_step,
        device,
        training,
        items,
        options.training_batch_size.max(1),
        dataset_name,
        None,
        None,
        None,
        None,
        RuliadProbeDecodeMode::FreeRun,
        None,
    )?;
    Ok(RuliadModelEvaluation {
        report: evaluation.report,
        item_count: evaluation.item_count,
        teacher_forced: evaluation.teacher_forced,
        mean_generated_model_tokens: evaluation.mean_generated_model_tokens,
        elapsed_ms: evaluation.elapsed_ms,
        generation_mean_batch_rows: evaluation.generation_stats.mean_batch_rows,
        generation_maximum_batch_rows: evaluation.generation_stats.maximum_batch_rows,
        generation_maximum_in_flight_rows: evaluation.generation_stats.maximum_in_flight_rows,
        generation_batched_row_fraction: ratio(
            evaluation.generation_stats.batched_rows,
            evaluation.item_count,
        ),
    })
}

pub fn evaluate_ruliad_model_suite<B>(
    dataset: &Dataset,
    model: &LanguageTrainModel<B>,
    training: &TrainingHyperparameters,
    options: &RuliadEvaluationSuiteOptions,
    device: &B::Device,
) -> Result<RuliadEvaluationSuiteReport>
where
    B: BackendTrait + Clone + 'static,
    B::Device: Clone,
{
    let started = burn_dragon_time::Instant::now();
    let stage = |name: &str| {
        eprintln!(
            "ruliad checkpoint evaluation stage={name} elapsed_ms={}",
            started.elapsed().as_millis()
        );
    };
    if options.free_run_items == 0 {
        return Err(anyhow!("Ruliad evaluation requires free_run_items > 0"));
    }
    if options.policy_items == 0 {
        return Err(anyhow!("Ruliad evaluation requires policy_items > 0"));
    }
    if options.difficulty_levels == 0 {
        return Err(anyhow!("Ruliad evaluation requires difficulty_levels > 0"));
    }
    stage("panel_and_identity");
    let model_tensor_fingerprint_sha256 =
        crate::train::model_identity::model_tensor_fingerprint::<B, _>(&model.model)?;

    let base_items = dataset.sample_ruliad_validation_probe_items_stratified_fixed(
        options.panel_seed,
        options.free_run_items,
        options.difficulty_levels,
        crate::dataset::RuliadValidationPromptMode::CanonicalTransfer,
    );
    let training_serialization_items = dataset
        .sample_ruliad_validation_probe_items_stratified_fixed(
            options.panel_seed,
            options.free_run_items,
            options.difficulty_levels,
            crate::dataset::RuliadValidationPromptMode::TrainingSerialization,
        );
    let policy_items = dataset.sample_ruliad_task_probe_items_fixed(
        options.panel_seed,
        options.policy_items,
        burn_dragon_universality::RuliadTaskKind::SelectProofAction.label(),
        options.difficulty_levels,
    );
    if base_items.len() != options.free_run_items {
        return Err(anyhow!(
            "Ruliad evaluation materialized {} free-run items, expected {}",
            base_items.len(),
            options.free_run_items
        ));
    }
    if training_serialization_items.len() != options.free_run_items {
        return Err(anyhow!(
            "Ruliad evaluation materialized {} training-serialization items, expected {}",
            training_serialization_items.len(),
            options.free_run_items
        ));
    }
    if policy_items.len() != options.policy_items {
        return Err(anyhow!(
            "Ruliad evaluation materialized {} policy items, expected {}",
            policy_items.len(),
            options.policy_items
        ));
    }
    for (canonical, matched) in base_items.iter().zip(&training_serialization_items) {
        if canonical.item.oracle_hash != matched.item.oracle_hash
            || canonical.item.expected_answer != matched.item.expected_answer
        {
            return Err(anyhow!(
                "canonical and training-serialization panels are not item-aligned"
            ));
        }
    }
    let policy_context_items =
        ruliad_policy_context_probe_items(dataset, &policy_items, &training.ruliad_policy_probe)?;
    if policy_context_items.len() != options.policy_items {
        return Err(anyhow!(
            "Ruliad evaluation materialized {} policy-context items, expected {}",
            policy_context_items.len(),
            options.policy_items
        ));
    }
    let panel_fingerprint_sha256 = panel_fingerprint(
        &base_items,
        &training_serialization_items,
        &policy_items,
        &policy_context_items,
    )?;

    stage("canonical_free_run");
    let free_run = evaluate_free_run_panel(
        dataset,
        model,
        training,
        options,
        &base_items,
        &options.dataset_name,
        device,
    )?;
    stage("training_serialization_free_run");
    let training_serialization_free_run = evaluate_free_run_panel(
        dataset,
        model,
        training,
        options,
        &training_serialization_items,
        &format!("{}_training_serialization", options.dataset_name),
        device,
    )?;
    stage("policy_context_free_run");
    let policy_context_free_run = evaluate_free_run_panel(
        dataset,
        model,
        training,
        options,
        &policy_context_items,
        &format!("{}_policy_context", options.dataset_name),
        device,
    )?;

    let mut evaluation_training = training.clone();
    evaluation_training.ruliad_policy_probe.enabled = true;
    evaluation_training.ruliad_policy_probe.items = options.policy_items;
    evaluation_training
        .ruliad_policy_probe
        .stratified_difficulty_levels = options.difficulty_levels;
    let event_dir = tempfile::tempdir().context("create Ruliad evaluation event directory")?;
    let mut handles = crate::train::events::build_training_event_handles(
        "ruliad-checkpoint-evaluation",
        event_dir.path(),
        1,
        &evaluation_training,
        None,
        None,
        None,
    )?;
    let bus = handles.metric_logger.bus();
    stage("constrained_policy");
    let constrained_policy = run_ruliad_correctness_constrained_policy_probe(
        "ruliad-checkpoint-evaluation",
        dataset,
        model,
        options.epoch,
        options.absolute_step,
        device,
        &evaluation_training,
        &policy_items,
        &bus,
        RuliadPolicyControlMode::Checkpoint,
    )?;
    let rollout = options
        .include_closed_loop_rollout
        .then(|| {
            stage("closed_loop_rollout");
            run_ruliad_policy_rollout_probe(
                "ruliad-checkpoint-evaluation",
                dataset,
                model,
                options.epoch,
                options.absolute_step,
                device,
                &evaluation_training,
                &policy_items,
                &bus,
            )
        })
        .transpose()?;
    handles.metric_logger.finish();

    let constrained_policy_by_difficulty = constrained_policy
        .difficulty_summaries
        .iter()
        .map(|(difficulty, summary)| (*difficulty, constrained_evaluation(*summary)))
        .collect();
    let constrained_policy_by_source = constrained_policy
        .source_summaries
        .iter()
        .map(|(source, summary)| (source.clone(), constrained_evaluation(*summary)))
        .collect();
    let structured_policy_decode = constrained_policy
        .structured_decode
        .clone()
        .ok_or_else(|| anyhow!("Ruliad structured-policy decode was not evaluated"))?;

    let (closed_loop_rollout, rollout_by_difficulty, rollout_by_source) = match rollout {
        Some(result) => (
            Some(rollout_evaluation(result.summary)),
            result
                .difficulty_summaries
                .into_iter()
                .map(|(difficulty, summary)| (difficulty, rollout_evaluation(summary)))
                .collect(),
            result
                .source_summaries
                .into_iter()
                .map(|(source, summary)| (source, rollout_evaluation(summary)))
                .collect(),
        ),
        None => (None, BTreeMap::new(), BTreeMap::new()),
    };

    stage("verify_final_identity");
    anyhow::ensure!(
        model_tensor_fingerprint_sha256
            == crate::train::model_identity::model_tensor_fingerprint::<B, _>(&model.model)?,
        "checkpoint evaluation mutated model parameters"
    );
    Ok(RuliadEvaluationSuiteReport {
        version: 8,
        model_tensor_fingerprint_sha256,
        panel_fingerprint_sha256,
        free_run,
        training_serialization_free_run,
        policy_context_free_run,
        structured_policy_decode,
        policy_controls: constrained_policy
            .controls
            .ok_or_else(|| anyhow!("Ruliad checkpoint policy controls were not evaluated"))?,
        constrained_policy: constrained_evaluation(constrained_policy.summary),
        constrained_policy_by_difficulty,
        constrained_policy_by_source,
        closed_loop_rollout,
        rollout_by_difficulty,
        rollout_by_source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rollout_metrics_use_the_same_denominators_as_training_telemetry() {
        let report = rollout_evaluation(RuliadPolicyRolloutProbeSummary {
            items: 4,
            solved: 2,
            steps: 12,
            valid_actions: 6,
            invalid_actions: 2,
            repeated_states: 3,
            backtracks: 2,
            scored_states: 8,
            top1_expert_actions: 6,
            frontier_exhaustions: 1,
            solved_goals: 3,
            total_goals: 4,
            ..Default::default()
        });

        assert_eq!(report.solve_rate, 0.5);
        assert_eq!(report.goal_completion_rate, 0.75);
        assert_eq!(report.valid_action_rate, 0.75);
        assert_eq!(report.repeated_state_rate, 0.5);
        assert_eq!(report.backtrack_rate, 0.25);
        assert_eq!(report.top1_expert_rate, 0.75);
        assert_eq!(report.mean_steps, 3.0);
    }

    #[test]
    fn zero_denominators_produce_finite_checkpoint_metrics() {
        let rollout = rollout_evaluation(RuliadPolicyRolloutProbeSummary::default());
        let constrained =
            constrained_evaluation(RuliadCorrectnessConstrainedPolicySummary::default());

        assert!(rollout.solve_rate.is_finite());
        assert!(rollout.valid_action_rate.is_finite());
        assert!(constrained.equivalent_top1_rate.is_finite());
        assert!(constrained.valid_invalid_margin.is_finite());
    }
}
