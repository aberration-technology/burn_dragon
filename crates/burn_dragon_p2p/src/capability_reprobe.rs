use std::fs;
use std::time::Duration;

use crate::capability::DragonTrainingFootprint;
use crate::config::DragonNativeCapabilityReprobePolicy;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeMemorySnapshot {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

pub fn read_native_memory_snapshot() -> Option<NativeMemorySnapshot> {
    let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
    parse_linux_meminfo(&meminfo)
}

fn parse_linux_meminfo(meminfo: &str) -> Option<NativeMemorySnapshot> {
    let mut total_kib = None;
    let mut available_kib = None;
    for line in meminfo.lines() {
        let mut fields = line.split_whitespace();
        match fields.next()? {
            "MemTotal:" => total_kib = fields.next().and_then(|value| value.parse::<u64>().ok()),
            "MemAvailable:" => {
                available_kib = fields.next().and_then(|value| value.parse::<u64>().ok())
            }
            _ => {}
        }
    }
    Some(NativeMemorySnapshot {
        total_bytes: total_kib?.saturating_mul(1024),
        available_bytes: available_kib?.saturating_mul(1024),
    })
}

pub fn native_reprobe_required_available_bytes(
    policy: &DragonNativeCapabilityReprobePolicy,
    footprint: &DragonTrainingFootprint,
) -> u64 {
    footprint
        .estimated_training_bytes
        .saturating_mul(u64::from(policy.memory_headroom_percent))
        .div_ceil(100)
        .max(policy.min_available_memory_bytes)
}

pub fn evaluate_native_reprobe(
    policy: &DragonNativeCapabilityReprobePolicy,
    footprint: &DragonTrainingFootprint,
    trainer_budget_bytes: Option<u64>,
    memory: Option<NativeMemorySnapshot>,
) -> Result<(), String> {
    policy.validate().map_err(|error| error.to_string())?;
    if let Some(budget) = trainer_budget_bytes
        && footprint.estimated_training_bytes > budget
    {
        return Err(format!(
            "estimated training footprint {} bytes still exceeds trainer budget {} bytes",
            footprint.estimated_training_bytes, budget
        ));
    }
    let memory = memory.ok_or_else(|| {
        "host memory availability is unavailable; trainer recovery remains read-only".to_owned()
    })?;
    let required = native_reprobe_required_available_bytes(policy, footprint);
    if memory.available_bytes < required {
        return Err(format!(
            "available host memory {} bytes is below safe recovery requirement {} bytes",
            memory.available_bytes, required
        ));
    }
    Ok(())
}

pub fn native_reprobe_backoff(
    policy: &DragonNativeCapabilityReprobePolicy,
    failure_count: u32,
) -> Duration {
    let exponent = failure_count.saturating_sub(1).min(20);
    let multiplier = 1_u64.checked_shl(exponent).unwrap_or(u64::MAX);
    Duration::from_secs(
        policy
            .interval_secs
            .saturating_mul(multiplier)
            .min(policy.max_interval_secs),
    )
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NativeReprobeTracker {
    consecutive_successes: u32,
}

impl NativeReprobeTracker {
    pub fn observe(
        &mut self,
        policy: &DragonNativeCapabilityReprobePolicy,
        probe: Result<(), String>,
    ) -> Result<bool, String> {
        match probe {
            Ok(()) => {
                self.consecutive_successes = self.consecutive_successes.saturating_add(1);
                Ok(self.consecutive_successes >= policy.required_successes)
            }
            Err(error) => {
                self.consecutive_successes = 0;
                Err(error)
            }
        }
    }

    pub fn reset(&mut self) {
        self.consecutive_successes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn footprint(training_bytes: u64) -> DragonTrainingFootprint {
        DragonTrainingFootprint {
            estimated_parameter_bytes: training_bytes / 4,
            estimated_optimizer_state_bytes: training_bytes / 2,
            estimated_activation_bytes: training_bytes / 4,
            estimated_training_bytes: training_bytes,
            estimated_checkpoint_bytes: training_bytes / 4,
            estimated_shard_bytes: 1,
            estimated_tokens_per_second: 1.0,
        }
    }

    #[test]
    fn linux_memory_parser_uses_mem_available() {
        let snapshot = parse_linux_meminfo(
            "MemTotal:       100000 kB\nMemFree:         1000 kB\nMemAvailable:   60000 kB\n",
        )
        .expect("memory snapshot");
        assert_eq!(snapshot.total_bytes, 100_000 * 1024);
        assert_eq!(snapshot.available_bytes, 60_000 * 1024);
    }

    #[test]
    fn reprobe_requires_budget_and_available_memory_headroom() {
        let policy = DragonNativeCapabilityReprobePolicy {
            min_available_memory_bytes: 1,
            memory_headroom_percent: 125,
            ..DragonNativeCapabilityReprobePolicy::default()
        };
        let footprint = footprint(1_000);
        assert!(
            evaluate_native_reprobe(
                &policy,
                &footprint,
                Some(999),
                Some(NativeMemorySnapshot {
                    total_bytes: 10_000,
                    available_bytes: 10_000,
                }),
            )
            .is_err()
        );
        assert!(
            evaluate_native_reprobe(
                &policy,
                &footprint,
                Some(2_000),
                Some(NativeMemorySnapshot {
                    total_bytes: 10_000,
                    available_bytes: 1_249,
                }),
            )
            .is_err()
        );
        assert!(
            evaluate_native_reprobe(
                &policy,
                &footprint,
                Some(2_000),
                Some(NativeMemorySnapshot {
                    total_bytes: 10_000,
                    available_bytes: 1_250,
                }),
            )
            .is_ok()
        );
    }

    #[test]
    fn reprobe_tracker_requires_a_success_streak_and_resets_on_failure() {
        let policy = DragonNativeCapabilityReprobePolicy {
            required_successes: 2,
            ..DragonNativeCapabilityReprobePolicy::default()
        };
        let mut tracker = NativeReprobeTracker::default();
        assert!(!tracker.observe(&policy, Ok(())).expect("first probe"));
        assert!(tracker.observe(&policy, Err("pressure".into())).is_err());
        assert!(!tracker.observe(&policy, Ok(())).expect("restarted probe"));
        assert!(tracker.observe(&policy, Ok(())).expect("second probe"));
    }

    #[test]
    fn reprobe_backoff_is_bounded() {
        let policy = DragonNativeCapabilityReprobePolicy {
            interval_secs: 10,
            max_interval_secs: 40,
            ..DragonNativeCapabilityReprobePolicy::default()
        };
        assert_eq!(native_reprobe_backoff(&policy, 1), Duration::from_secs(10));
        assert_eq!(native_reprobe_backoff(&policy, 2), Duration::from_secs(20));
        assert_eq!(native_reprobe_backoff(&policy, 20), Duration::from_secs(40));
    }
}
