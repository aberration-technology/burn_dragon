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
    pub panel_fingerprint_sha256: String,
    pub free_run: RuliadModelEvaluation,
    pub constrained_policy: RuliadConstrainedPolicyEvaluation,
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
    policy_items: &[crate::dataset::RuliadValidationProbeItem],
) -> Result<String> {
    let bytes = serde_json::to_vec(&(base_items, policy_items))
        .context("serialize checkpoint-evaluation Ruliad panel")?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
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
    if options.free_run_items == 0 {
        return Err(anyhow!("Ruliad evaluation requires free_run_items > 0"));
    }
    if options.policy_items == 0 {
        return Err(anyhow!("Ruliad evaluation requires policy_items > 0"));
    }
    if options.difficulty_levels == 0 {
        return Err(anyhow!("Ruliad evaluation requires difficulty_levels > 0"));
    }

    let base_items = dataset.sample_ruliad_validation_probe_items_stratified_fixed(
        options.panel_seed,
        options.free_run_items,
        options.difficulty_levels,
        crate::dataset::RuliadValidationPromptMode::CanonicalTransfer,
    );
    let policy_items = dataset.sample_ruliad_validation_probe_items_stratified(
        0,
        0,
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
    if policy_items.len() != options.policy_items {
        return Err(anyhow!(
            "Ruliad evaluation materialized {} policy items, expected {}",
            policy_items.len(),
            options.policy_items
        ));
    }
    let panel_fingerprint_sha256 = panel_fingerprint(&base_items, &policy_items)?;

    let free_run = evaluate_ruliad_correctness_validation_for_items_core(
        None,
        None,
        dataset,
        model,
        options.epoch,
        options.absolute_step,
        device,
        training,
        &base_items,
        options.training_batch_size.max(1),
        &options.dataset_name,
        None,
        None,
        None,
        None,
        RuliadProbeDecodeMode::FreeRun,
        None,
    )?;
    let free_run = RuliadModelEvaluation {
        report: free_run.report,
        item_count: free_run.item_count,
        elapsed_ms: free_run.elapsed_ms,
        generation_mean_batch_rows: free_run.generation_stats.mean_batch_rows,
        generation_maximum_batch_rows: free_run.generation_stats.maximum_batch_rows,
        generation_maximum_in_flight_rows: free_run.generation_stats.maximum_in_flight_rows,
        generation_batched_row_fraction: ratio(
            free_run.generation_stats.batched_rows,
            free_run.item_count,
        ),
    };

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
    )?;
    let rollout = options
        .include_closed_loop_rollout
        .then(|| {
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

    Ok(RuliadEvaluationSuiteReport {
        version: 1,
        panel_fingerprint_sha256,
        free_run,
        constrained_policy: constrained_evaluation(constrained_policy),
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
