//! Gated DeltaNet 2 kernel feature gates.
//!
//! Gated DeltaNet 2 execution surfaces.
//!
//! The chunk-WY executor uses a Burn custom-autodiff wrapper over explicit CubeCL kernels on CUDA.
//! Its default CUDA backward builds the per-chunk WY basis, solves the strict triangular system,
//! and applies the gate-aware VJP in one fused kernel.

mod custom_backward;
mod runtime;

use burn::tensor::backend::Backend as BackendTrait;

pub use custom_backward::{GatedDeltaNet2CustomBackwardOutput, try_gdn2_chunk_wy_custom_backward};

fn env_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            matches!(
                value.as_str(),
                "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
            )
        })
        .unwrap_or(false)
}

pub fn use_gdn2_wgpu_recurrent_forward_experimental() -> bool {
    env_enabled("BURN_DRAGON_GDN2_WGPU_RECURRENT_FORWARD")
}

pub fn use_gdn2_wgpu_chunk_wy_experimental() -> bool {
    env_enabled("BURN_DRAGON_GDN2_WGPU_CHUNK_WY")
}

pub fn use_gdn2_wgpu_custom_backward_experimental() -> bool {
    env_enabled("BURN_DRAGON_GDN2_WGPU_CUSTOM_BACKWARD")
}

pub fn gdn2_profile_enabled() -> bool {
    env_enabled("BURN_DRAGON_GDN2_PROFILE")
}

pub fn supports_gdn2_chunk_wy_backend<B: BackendTrait>() -> bool {
    let _ = core::marker::PhantomData::<B>;
    false
}
