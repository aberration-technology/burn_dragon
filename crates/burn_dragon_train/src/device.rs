#[cfg(feature = "train")]
use std::sync::{Mutex, OnceLock};

#[cfg(feature = "train")]
pub(crate) fn device_allocation_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}
