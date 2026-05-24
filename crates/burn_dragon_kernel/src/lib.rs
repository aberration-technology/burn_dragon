#![recursion_limit = "256"]

//! Fused Dragon execution kernels and compiled-plan helpers.
//!
//! Preferred library-facing surface:
//! - [`api::recurrent`]
//! - [`api::spatial`]
//! - [`api::graph`]
//! - [`api::expert`] for lower-level kernel-plan access

mod dense_causal_attention;
mod fusion_compat;
mod profiling;
mod recurrent;
mod relu_lowrank;
mod sequence;

pub mod api {
    //! Curated public surface for the fused execution layer.
    //!
    //! This mirrors the active kernel families instead of exposing the entire file/module layout.

    pub use crate::kernels::{attention, projection, recurrent};

    pub mod expert {
        //! Lower-level fused-kernel surface for advanced callers.

        pub use crate::kernels;
    }
}

/// Namespaced fused-kernel families exposed by this crate.
pub mod kernels {
    /// Sequence-kernel family namespace used by the language line.
    pub mod sequence {
        pub use crate::sequence::{gdn2, linear, mamba3};
    }

    /// Dense causal attention kernels used by the focused linear-attention path.
    pub mod attention {
        pub use crate::dense_causal_attention::{
            CompiledDenseCausalAttentionPlan, supports_dense_causal_attention_backend,
            try_fused_dense_causal_attention_wgpu, try_fused_dense_causal_attention_wgpu_with_plan,
        };
    }

    /// Core recurrent attention kernels.
    pub mod recurrent {
        pub use crate::recurrent::{
            CompiledRecurrentAttentionPlan, RecurrentAttentionOutput, RecurrentProfileSnapshot,
            recurrent_profile_reset, recurrent_profile_snapshot,
            supports_backend as supports_recurrent_backend, try_fused_recurrent_attention_wgpu,
            try_fused_recurrent_attention_wgpu_with_plan,
        };
    }

    /// Fused low-rank projection kernels used in recurrent x/y projection paths.
    pub mod projection {
        pub use crate::relu_lowrank::{
            LowrankForwardRouteProfileSnapshot, LowrankGradInputExecutor,
            LowrankProjectionProfileSnapshot, relu_lowrank_forward_profile_reset,
            relu_lowrank_forward_profile_snapshot, relu_lowrank_forward_route_profile_reset,
            relu_lowrank_forward_route_profile_snapshot, relu_lowrank_grad_input_profile_reset,
            relu_lowrank_grad_input_profile_snapshot, relu_lowrank_grad_weight_profile_reset,
            relu_lowrank_grad_weight_profile_snapshot, supports_relu_lowrank_projection_backend,
            try_fused_relu_lowrank_projection_wgpu,
            try_fused_relu_lowrank_projection_wgpu_with_executor,
        };
    }
}
