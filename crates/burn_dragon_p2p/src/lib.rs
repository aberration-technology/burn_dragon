#![forbid(unsafe_code)]

pub mod auth;
#[cfg(any(all(feature = "wasm-peer", target_arch = "wasm32"), test))]
pub(crate) mod browser_data;
#[cfg(any(all(feature = "wasm-peer", target_arch = "wasm32"), test))]
#[cfg(any(
    target_arch = "wasm32",
    all(test, feature = "native", feature = "wasm-peer")
))]
pub(crate) mod browser_record;
pub mod build_info;
pub mod capability;
#[cfg(feature = "native")]
pub mod capability_reprobe;
pub mod capability_state;
pub mod config;
#[cfg(feature = "native")]
pub mod deployment;
#[cfg(feature = "native")]
pub mod experiments;
pub mod logging;
#[cfg(feature = "native")]
pub mod manifests;
pub mod profile;
#[cfg(any(feature = "wasm-peer", feature = "native"))]
pub(crate) mod seeded_fitness;
pub(crate) mod stream_batch;

pub mod admin;
#[cfg(feature = "native")]
pub mod native;
#[cfg(feature = "native")]
pub mod native_runtime;
pub mod p2p_adapter;

#[cfg(all(feature = "wasm-ui", feature = "wasm-peer", target_arch = "wasm32"))]
pub mod wasm;
