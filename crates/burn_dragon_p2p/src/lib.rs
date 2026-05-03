#![forbid(unsafe_code)]

pub mod auth;
#[cfg(any(feature = "wasm-peer", test))]
pub(crate) mod browser_data;
pub mod build_info;
pub mod capability;
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

pub mod admin;
#[cfg(feature = "native")]
pub mod native;
#[cfg(feature = "native")]
pub mod native_runtime;
pub mod p2p_adapter;

#[cfg(all(feature = "wasm-ui", feature = "wasm-peer", target_arch = "wasm32"))]
pub mod wasm;
