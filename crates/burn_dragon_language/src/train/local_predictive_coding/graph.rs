use crate::config::{
    LocalPredictiveCodingAdjointConditioning, LocalPredictiveCodingConfig,
    PredictiveCodingFactorReduction,
};

/// Materialized, backend-independent factor graph for one Dragon block.
/// Activity nodes are inferred per batch; the token target remains clamped.
pub fn dragon_predictive_coding_graph(layers: usize) -> burn_pc::PcGraphSpec {
    assert!(
        layers > 0,
        "Dragon predictive coding requires at least one layer"
    );
    let target_id = burn_pc::PcNodeId((layers + 1).try_into().expect("layer count fits u32"));
    let mut nodes = (0..=layers)
        .map(|layer| burn_pc::PcNodeSpec {
            id: burn_pc::PcNodeId(layer.try_into().expect("layer count fits u32")),
            name: format!("activity_{layer}"),
            clamped: layer == 0,
        })
        .collect::<Vec<_>>();
    nodes.push(burn_pc::PcNodeSpec {
        id: target_id,
        name: "token_target".to_string(),
        clamped: true,
    });
    let mut factors = (0..layers)
        .map(|layer| burn_pc::PcFactorSpec {
            id: burn_pc::PcFactorId(layer.try_into().expect("layer count fits u32")),
            name: format!("dragon_layer_{layer}"),
            parents: vec![burn_pc::PcNodeId(
                layer.try_into().expect("layer count fits u32"),
            )],
            target: burn_pc::PcNodeId((layer + 1).try_into().expect("layer count fits u32")),
        })
        .collect::<Vec<_>>();
    factors.push(burn_pc::PcFactorSpec {
        id: burn_pc::PcFactorId(layers.try_into().expect("layer count fits u32")),
        name: "next_token".to_string(),
        parents: vec![burn_pc::PcNodeId(
            layers.try_into().expect("layer count fits u32"),
        )],
        target: target_id,
    });
    burn_pc::PcGraphSpec::new(nodes, factors)
}

/// Stable identity persisted with exact-resume checkpoints and exchanged by
/// distributed orchestration before a peer can participate in optimization.
pub fn dragon_predictive_coding_checkpoint_manifest(
    layers: usize,
    config: &LocalPredictiveCodingConfig,
) -> anyhow::Result<burn_pc::PcCheckpointManifest> {
    let graph_digest = dragon_predictive_coding_graph(layers).stable_fingerprint()?;
    let learning_schedule = match config.learning_schedule {
        burn_pc::PcLearningSchedule::Equilibrium => "equilibrium",
        burn_pc::PcLearningSchedule::Incremental => "incremental",
    };
    let parameterization = match config.parameterization {
        burn_pc::PcParameterizationKind::Standard => "standard",
        burn_pc::PcParameterizationKind::MuPc => "mu_pc",
    };
    let shared_reuse_reduction = match config.shared_reuse_reduction {
        burn_pc::PcSharedReuseReduction::Sum => "sum",
        burn_pc::PcSharedReuseReduction::Mean => "mean",
        burn_pc::PcSharedReuseReduction::RootMeanSquare => "root_mean_square",
    };
    let factor_reduction = match config.factor_reduction {
        PredictiveCodingFactorReduction::Sum => "sum",
        PredictiveCodingFactorReduction::Mean => "mean",
    };
    let gradient_norm_scope = match config.inference.gradient_norm_scope {
        burn_pc::PcGradientNormScope::Global => "global",
        burn_pc::PcGradientNormScope::PerSample => "per_sample",
        burn_pc::PcGradientNormScope::PerRow => "per_row",
    };
    let max_grad_norm = config.inference.max_grad_norm.map_or_else(
        || "none".to_string(),
        |value| format!("{:08x}", value.to_bits()),
    );
    let alm_gradient_norm_scope = match config.augmented_lagrangian.gradient_norm_scope {
        burn_pc::PcGradientNormScope::Global => "global",
        burn_pc::PcGradientNormScope::PerSample => "per_sample",
        burn_pc::PcGradientNormScope::PerRow => "per_row",
    };
    let alm_max_grad_norm = config
        .augmented_lagrangian
        .max_primal_grad_norm
        .map_or_else(
            || "none".to_string(),
            |value| format!("{:08x}", value.to_bits()),
        );
    let consensus_max_norm = config.tied_consensus.max_update_norm.map_or_else(
        || "none".to_string(),
        |value| format!("{:08x}", value.to_bits()),
    );
    let adjoint_max_update = config
        .amortized_adjoint
        .calibration
        .max_update_norm
        .map_or_else(
            || "none".to_string(),
            |value| format!("{:08x}", value.to_bits()),
        );
    let feedback_initialization = match config.direct_feedback.initialization {
        burn_pc::PcFeedbackInitialization::Gaussian => "gaussian",
        burn_pc::PcFeedbackInitialization::Identity => "identity",
    };
    let adjoint_predictor = match config.amortized_adjoint.predictor {
        burn_pc::PcAdjointPredictorKind::DirectLinear => "direct_linear",
        burn_pc::PcAdjointPredictorKind::ResidualConditioned => "residual_conditioned",
    };
    let adjoint_conditioning = match config.adjoint_conditioning {
        LocalPredictiveCodingAdjointConditioning::LocalResidual => "local_residual",
        LocalPredictiveCodingAdjointConditioning::TerminalDisplacement => "terminal_displacement",
    };
    let terminal_criterion = match config.terminal_criterion {
        crate::config::LocalPredictiveCodingTerminalCriterion::NextToken => "next_token",
        crate::config::LocalPredictiveCodingTerminalCriterion::RuliadVerifierSet => {
            "ruliad_verifier_set"
        }
        crate::config::LocalPredictiveCodingTerminalCriterion::RuliadVerifierSetJoint => {
            "ruliad_verifier_set_joint"
        }
    };
    let temporal_credit_mode = match config.temporal_credit.mode {
        burn_pc::PcTemporalCreditMode::Detached => "detached",
        burn_pc::PcTemporalCreditMode::ExactWindow => "exact_window",
    };
    let program_digest = format!(
        "dragon-pc-program-v11;solver={};next_token_solver={};terminal={terminal_criterion};schedule={learning_schedule};parameterization={parameterization};shared_reuse={shared_reuse_reduction};factor_reduction={factor_reduction};temporal_credit={temporal_credit_mode};temporal_window_chunks={};inference_steps={};step_size={:08x};latent_decay={:08x};max_grad_norm={max_grad_norm};gradient_norm_scope={gradient_norm_scope};eps={:08x};alm_steps={};alm_primal_step={:08x};alm_dual_step={:08x};alm_penalty={:08x};alm_max_grad_norm={alm_max_grad_norm};alm_gradient_norm_scope={alm_gradient_norm_scope};alm_eps={:08x};prediction_precision={:08x};incremental_parameter_step_scale={:016x};dkp_preliminary_step={:08x};dkp_feedback_step={:08x};dkp_forward_decay={:08x};dkp_feedback_decay={:08x};dkp_signal_scale={:08x};dkp_feedback_initialization={feedback_initialization};adjoint_enabled={};adjoint_warmup={};adjoint_every={};adjoint_predictor={adjoint_predictor};adjoint_conditioning={adjoint_conditioning};adjoint_conditioning_clip={:08x};adjoint_lr={:08x};adjoint_decay={:08x};adjoint_max_update={adjoint_max_update};adjoint_eps={:08x};consensus_damping={:08x};consensus_min_curvature={:08x};consensus_max_norm={consensus_max_norm};consensus_eps={:08x}",
        config.solver.as_str(),
        config.next_token_solver().as_str(),
        config.temporal_credit.window_chunks,
        config.inference.steps,
        config.inference.step_size.to_bits(),
        config.inference.latent_decay.to_bits(),
        config.inference.eps.to_bits(),
        config.augmented_lagrangian.steps,
        config.augmented_lagrangian.primal_step_size.to_bits(),
        config.augmented_lagrangian.dual_step_size.to_bits(),
        config.augmented_lagrangian.penalty.to_bits(),
        config.augmented_lagrangian.eps.to_bits(),
        config.prediction_precision.to_bits(),
        config.incremental_parameter_step_scale.to_bits(),
        config.direct_feedback.preliminary_step_size.to_bits(),
        config.direct_feedback.feedback_step_size.to_bits(),
        config.direct_feedback.forward_weight_decay.to_bits(),
        config.direct_feedback.feedback_weight_decay.to_bits(),
        config.direct_feedback.signal_scale.to_bits(),
        config.amortized_adjoint.enabled,
        config.amortized_adjoint.teacher_warmup_updates,
        config.amortized_adjoint.teacher_every_updates,
        config.amortized_adjoint.conditioning_clip.to_bits(),
        config.amortized_adjoint.calibration.learning_rate.to_bits(),
        config.amortized_adjoint.calibration.weight_decay.to_bits(),
        config.amortized_adjoint.calibration.eps.to_bits(),
        config.tied_consensus.damping.to_bits(),
        config.tied_consensus.min_curvature.to_bits(),
        config.tied_consensus.eps.to_bits(),
    );
    Ok(burn_pc::PcCheckpointManifest {
        schema_version: burn_pc::PcCheckpointManifest::CURRENT_SCHEMA_VERSION,
        graph_digest,
        program_digest,
        algorithm: "dragon_local_predictive_coding_v1".to_string(),
        learning_schedule: config.learning_schedule,
        execution_contract: config.execution_contract(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{LocalPredictiveCodingObjectiveRoutingConfig, LocalPredictiveCodingSolver};

    #[test]
    fn objective_solver_route_is_bound_to_checkpoint_program_identity() {
        let base = LocalPredictiveCodingConfig {
            solver: LocalPredictiveCodingSolver::ErrorEquilibrium,
            ..LocalPredictiveCodingConfig::default()
        };
        let routed = LocalPredictiveCodingConfig {
            objective_routing: LocalPredictiveCodingObjectiveRoutingConfig {
                next_token_solver: Some(LocalPredictiveCodingSolver::FixedPrediction),
            },
            ..base.clone()
        };

        let base = dragon_predictive_coding_checkpoint_manifest(4, &base)
            .expect("base checkpoint manifest");
        let routed = dragon_predictive_coding_checkpoint_manifest(4, &routed)
            .expect("routed checkpoint manifest");
        assert_ne!(base.program_digest, routed.program_digest);
        assert!(
            routed
                .program_digest
                .contains("next_token_solver=fixed_prediction")
        );
    }
}
