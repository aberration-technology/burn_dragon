use std::sync::{Arc, Mutex};

use crate::config::{LocalPredictiveCodingConfig, LocalPredictiveCodingSolver};

#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize)]
pub struct LocalPredictiveCodingStepReport {
    pub solver: LocalPredictiveCodingSolver,
    pub inference_steps: usize,
    pub dual_steps: usize,
    pub factors: usize,
    pub local_vjp_calls: usize,
    /// VJPs induced by a later chunk's terminal-rho adjoint.
    pub temporal_state_vjp_calls: usize,
    pub fused_temporal_vjp_calls: usize,
    pub global_backward_calls: usize,
    pub gradient_tensors: usize,
    /// Logical direct-feedback forward-factor updates. These may be executed
    /// by one batched kernel even when every depth factor contributes.
    pub direct_forward_updates: usize,
    /// Logical Kolen-Pollack feedback matrices updated by the step.
    pub feedback_parameter_updates: usize,
    /// Exact local adjoint teacher waves used to calibrate feedback.
    pub adjoint_teacher_updates: usize,
    /// Adjoint updates computed without an exact reverse-depth teacher. This
    /// includes local feedback rules and analytic first-order residual probes.
    pub adjoint_local_updates: usize,
    pub parameter_updates: usize,
    pub energy_before: Option<f64>,
    pub energy_after: Option<f64>,
    pub grad_norm_mean: Option<f64>,
    pub grad_norm_max: Option<f64>,
    pub delta_rms_mean: Option<f64>,
    /// Fraction of primal activity rows clipped on the final inference step.
    pub clip_fraction_mean: Option<f64>,
    pub constraint_rms: Option<f64>,
    pub dual_rms: Option<f64>,
    pub composite_signal_rms: Option<f64>,
    pub elapsed_ns: u128,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LocalPredictiveCodingProfileSnapshot {
    pub steps: u64,
    pub inference_steps: u64,
    pub dual_steps: u64,
    pub factors: u64,
    pub local_vjp_calls: u64,
    pub temporal_state_vjp_calls: u64,
    pub fused_temporal_vjp_calls: u64,
    pub global_backward_calls: u64,
    pub gradient_tensors: u64,
    pub direct_forward_updates: u64,
    pub feedback_parameter_updates: u64,
    pub adjoint_teacher_updates: u64,
    pub adjoint_local_updates: u64,
    pub adjoint_calibration_samples: u64,
    /// Logical local derivative/update intents. These are not necessarily
    /// outer optimizer applications when gradients are accumulated over
    /// recurrent chunks.
    pub parameter_updates: u64,
    /// Actual calls to the optimizer update transform.
    pub optimizer_updates: u64,
    pub structured_terminal_steps: u64,
    pub structured_terminal_skipped_steps: u64,
    pub structured_terminal_groups: u64,
    pub structured_terminal_rows: u64,
    pub elapsed_ns: u128,
    pub last_energy_before: Option<f64>,
    pub last_energy_after: Option<f64>,
    pub last_grad_norm_mean: Option<f64>,
    pub last_grad_norm_max: Option<f64>,
    pub last_delta_rms_mean: Option<f64>,
    pub last_clip_fraction_mean: Option<f64>,
    pub last_constraint_rms: Option<f64>,
    pub last_dual_rms: Option<f64>,
    pub last_composite_signal_rms: Option<f64>,
    pub last_adjoint_calibration_loss: Option<f64>,
    pub last_adjoint_cosine_alignment: Option<f64>,
    pub last_adjoint_prediction_teacher_norm_ratio: Option<f64>,
    pub last_adjoint_update_rms: Option<f64>,
}

pub(crate) fn validate_step_execution_contract(
    config: &LocalPredictiveCodingConfig,
    report: &LocalPredictiveCodingStepReport,
) {
    burn_pc::PcExecutionCounters {
        local_parameter_vjp_calls: report.local_vjp_calls as u64,
        global_parameter_backward_calls: report.global_backward_calls as u64,
        direct_forward_updates: report.direct_forward_updates as u64,
        feedback_parameter_updates: report.feedback_parameter_updates as u64,
        parameter_updates: report.parameter_updates as u64,
        ..burn_pc::PcExecutionCounters::default()
    }
    .validate_against(&config.execution_contract())
    .expect("local predictive-coding step violated its configured execution contract");
}

/// Run-scoped local-PC telemetry shared by the train model and its ECS run
/// entity. Multiple pipelines in one process never contend on one global slot.
#[derive(Debug, Clone, Default)]
pub struct LocalPredictiveCodingProfile {
    inner: Arc<Mutex<LocalPredictiveCodingProfileSnapshot>>,
}

impl LocalPredictiveCodingProfile {
    pub fn reset(&self) {
        if let Ok(mut profile) = self.inner.lock() {
            *profile = LocalPredictiveCodingProfileSnapshot::default();
        }
    }

    pub fn snapshot(&self) -> LocalPredictiveCodingProfileSnapshot {
        self.inner
            .lock()
            .map(|profile| *profile)
            .unwrap_or_default()
    }

    pub fn take(&self) -> LocalPredictiveCodingProfileSnapshot {
        self.inner
            .lock()
            .map(|mut profile| std::mem::take(&mut *profile))
            .unwrap_or_default()
    }

    pub(crate) fn record(&self, report: LocalPredictiveCodingStepReport) {
        if let Ok(mut profile) = self.inner.lock() {
            profile.steps = profile.steps.saturating_add(1);
            profile.inference_steps = profile
                .inference_steps
                .saturating_add(report.inference_steps as u64);
            profile.dual_steps = profile.dual_steps.saturating_add(report.dual_steps as u64);
            profile.factors = profile.factors.saturating_add(report.factors as u64);
            profile.local_vjp_calls = profile
                .local_vjp_calls
                .saturating_add(report.local_vjp_calls as u64);
            profile.temporal_state_vjp_calls = profile
                .temporal_state_vjp_calls
                .saturating_add(report.temporal_state_vjp_calls as u64);
            profile.fused_temporal_vjp_calls = profile
                .fused_temporal_vjp_calls
                .saturating_add(report.fused_temporal_vjp_calls as u64);
            profile.global_backward_calls = profile
                .global_backward_calls
                .saturating_add(report.global_backward_calls as u64);
            profile.gradient_tensors = profile
                .gradient_tensors
                .saturating_add(report.gradient_tensors as u64);
            profile.direct_forward_updates = profile
                .direct_forward_updates
                .saturating_add(report.direct_forward_updates as u64);
            profile.feedback_parameter_updates = profile
                .feedback_parameter_updates
                .saturating_add(report.feedback_parameter_updates as u64);
            profile.adjoint_teacher_updates = profile
                .adjoint_teacher_updates
                .saturating_add(report.adjoint_teacher_updates as u64);
            profile.adjoint_local_updates = profile
                .adjoint_local_updates
                .saturating_add(report.adjoint_local_updates as u64);
            profile.parameter_updates = profile
                .parameter_updates
                .saturating_add(report.parameter_updates as u64);
            profile.elapsed_ns = profile.elapsed_ns.saturating_add(report.elapsed_ns);
            profile.last_energy_before = report.energy_before;
            profile.last_energy_after = report.energy_after;
            profile.last_grad_norm_mean = report.grad_norm_mean;
            profile.last_grad_norm_max = report.grad_norm_max;
            profile.last_delta_rms_mean = report.delta_rms_mean;
            profile.last_clip_fraction_mean = report.clip_fraction_mean;
            profile.last_constraint_rms = report.constraint_rms;
            profile.last_dual_rms = report.dual_rms;
            profile.last_composite_signal_rms = report.composite_signal_rms;
        }
    }

    pub(crate) fn record_optimizer_updates(&self, count: usize) {
        if let Ok(mut profile) = self.inner.lock() {
            profile.optimizer_updates = profile.optimizer_updates.saturating_add(count as u64);
        }
    }

    pub(crate) fn record_structured_terminal(&self, groups: usize, rows: usize) {
        if let Ok(mut profile) = self.inner.lock() {
            profile.structured_terminal_steps = profile.structured_terminal_steps.saturating_add(1);
            profile.structured_terminal_groups = profile
                .structured_terminal_groups
                .saturating_add(groups as u64);
            profile.structured_terminal_rows =
                profile.structured_terminal_rows.saturating_add(rows as u64);
        }
    }

    pub(crate) fn record_adjoint_calibration(
        &self,
        loss: f64,
        cosine_alignment: f64,
        prediction_teacher_norm_ratio: f64,
        update_rms: f64,
    ) {
        if let Ok(mut profile) = self.inner.lock() {
            profile.adjoint_calibration_samples =
                profile.adjoint_calibration_samples.saturating_add(1);
            profile.last_adjoint_calibration_loss = Some(loss);
            profile.last_adjoint_cosine_alignment = Some(cosine_alignment);
            profile.last_adjoint_prediction_teacher_norm_ratio =
                Some(prediction_teacher_norm_ratio);
            profile.last_adjoint_update_rms = Some(update_rms);
        }
    }

    pub(crate) fn record_global_structured_terminal(
        &self,
        groups: usize,
        rows: usize,
        elapsed_ns: u128,
    ) {
        if let Ok(mut profile) = self.inner.lock() {
            profile.steps = profile.steps.saturating_add(1);
            profile.factors = profile.factors.saturating_add(1);
            profile.global_backward_calls = profile.global_backward_calls.saturating_add(1);
            profile.parameter_updates = profile.parameter_updates.saturating_add(1);
            profile.structured_terminal_steps = profile.structured_terminal_steps.saturating_add(1);
            profile.structured_terminal_groups = profile
                .structured_terminal_groups
                .saturating_add(groups as u64);
            profile.structured_terminal_rows =
                profile.structured_terminal_rows.saturating_add(rows as u64);
            profile.elapsed_ns = profile.elapsed_ns.saturating_add(elapsed_ns);
        }
    }

    pub(crate) fn record_structured_terminal_skip(&self) {
        if let Ok(mut profile) = self.inner.lock() {
            profile.structured_terminal_skipped_steps =
                profile.structured_terminal_skipped_steps.saturating_add(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjoint_calibration_diagnostics_are_run_scoped_and_take_resets_them() {
        let first = LocalPredictiveCodingProfile::default();
        let second = LocalPredictiveCodingProfile::default();
        first.record_adjoint_calibration(0.25, 0.75, 1.25, 0.01);
        first.record_adjoint_calibration(0.125, 0.875, 1.125, 0.005);

        let snapshot = first.take();
        assert_eq!(snapshot.adjoint_calibration_samples, 2);
        assert_eq!(snapshot.last_adjoint_calibration_loss, Some(0.125));
        assert_eq!(snapshot.last_adjoint_cosine_alignment, Some(0.875));
        assert_eq!(
            snapshot.last_adjoint_prediction_teacher_norm_ratio,
            Some(1.125)
        );
        assert_eq!(snapshot.last_adjoint_update_rms, Some(0.005));
        assert_eq!(
            first.snapshot(),
            LocalPredictiveCodingProfileSnapshot::default()
        );
        assert_eq!(
            second.snapshot(),
            LocalPredictiveCodingProfileSnapshot::default()
        );
    }

    #[test]
    fn derivative_intents_are_distinct_from_optimizer_applications() {
        let profile = LocalPredictiveCodingProfile::default();
        profile.record_global_structured_terminal(2, 8, 10);
        profile.record_global_structured_terminal(2, 8, 10);
        profile.record_optimizer_updates(1);

        let snapshot = profile.snapshot();
        assert_eq!(snapshot.parameter_updates, 2);
        assert_eq!(snapshot.optimizer_updates, 1);
    }
}
