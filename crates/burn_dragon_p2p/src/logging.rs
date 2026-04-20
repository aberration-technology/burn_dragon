#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]

use std::sync::Once;

static LOGGING_INIT: Once = Once::new();

#[cfg(target_arch = "wasm32")]
pub fn init_browser_logging() {
    LOGGING_INIT.call_once(|| {
        let _ = console_log::init_with_level(log::Level::Info);
    });
}

#[cfg(not(target_arch = "wasm32"))]
pub fn init_native_logging() {
    LOGGING_INIT.call_once(|| {
        let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .format_timestamp_secs()
            .format_target(false)
            .try_init();
    });
}
