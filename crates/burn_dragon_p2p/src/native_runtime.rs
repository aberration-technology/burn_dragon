use std::collections::BTreeSet;
use std::ops::{Deref, DerefMut};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::Result;
use burn::tensor::backend::AutodiffBackend;
use burn_ecs::prelude::App;
use burn_ecs::{
    PipelineComputeClass, PipelineParticipation, TrainingAppExt, TrainingEventBusConfig,
    TrainingPlugins, TrainingRunConfig, TrainingRunOptions, TrainingRuntimeThread,
};
use burn_p2p::{
    ControlHandle, NodeTelemetrySnapshot, PeerRole, PeerRoleSet, RunningNode, RuntimeStatus,
    SelectedWorkloadProject, TelemetryHandle,
    ecs::{
        P2pCapabilityAssessment, P2pTrainingEcsObserver, P2pTrainingEventBus,
        P2pTrainingEventBusStats, P2pTrainingIngressPlugin,
    },
};

use crate::capability_reprobe::{
    NativeReprobeTracker, evaluate_native_reprobe, native_reprobe_backoff,
    read_native_memory_snapshot,
};
use crate::capability_state::is_probable_trainer_fit_failure;
use crate::config::DragonNativeTarget;
use crate::experiments::common::{DragonProjectFamily, PreparedNativePeer};

const MONITOR_POLL_INTERVAL: Duration = Duration::from_millis(500);
const DROP_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

pub struct ManagedRunningNativePeer<B>
where
    B: AutodiffBackend + Clone + 'static,
{
    prepared: Option<PreparedNativePeer<B>>,
    running: Option<RunningNode<SelectedWorkloadProject<DragonProjectFamily<B>>>>,
    p2p_event_thread: Option<TrainingRuntimeThread>,
    p2p_event_bus: P2pTrainingEventBus,
    stop_flag: Arc<AtomicBool>,
    monitor_thread: Option<JoinHandle<()>>,
}

impl<B> ManagedRunningNativePeer<B>
where
    B: AutodiffBackend + Clone + 'static,
{
    fn stop_and_join(&mut self, timeout: Option<Duration>) -> Result<()> {
        self.stop_flag.store(true, Ordering::SeqCst);
        if let Some(running) = self.running.take() {
            let _ = running.shutdown();
            match timeout {
                Some(timeout) => {
                    let _ = running.await_termination_timeout(timeout)?;
                }
                None => {
                    let _ = running.await_termination()?;
                }
            }
        }
        if let Some(handle) = self.monitor_thread.take() {
            let _ = handle.join();
        }
        if let Some(thread) = self.p2p_event_thread.take() {
            thread.shutdown()?;
        }
        Ok(())
    }

    pub fn prepared(&self) -> &PreparedNativePeer<B> {
        self.prepared
            .as_ref()
            .expect("managed native peer should retain prepared peer")
    }

    pub fn telemetry(&self) -> TelemetryHandle {
        self.running
            .as_ref()
            .expect("managed native peer should retain running node")
            .telemetry()
    }

    pub fn control_handle(&self) -> ControlHandle {
        self.running
            .as_ref()
            .expect("managed native peer should retain running node")
            .control_handle()
    }

    pub fn snapshot(&self) -> NodeTelemetrySnapshot {
        self.telemetry().snapshot()
    }

    /// Returns point-in-time pressure and delivery counters for the run's ECS ingress.
    pub fn p2p_event_bus_stats(&self) -> P2pTrainingEventBusStats {
        self.p2p_event_bus.stats()
    }

    pub fn shutdown(&self) -> Result<()> {
        self.stop_flag.store(true, Ordering::SeqCst);
        self.running
            .as_ref()
            .expect("managed native peer should retain running node")
            .shutdown()
    }

    pub fn await_termination(mut self) -> Result<PreparedNativePeer<B>> {
        self.stop_and_join(None)?;
        Ok(self
            .prepared
            .take()
            .expect("managed native peer should retain prepared peer"))
    }

    pub fn await_termination_timeout(mut self, timeout: Duration) -> Result<PreparedNativePeer<B>> {
        self.stop_and_join(Some(timeout))?;
        Ok(self
            .prepared
            .take()
            .expect("managed native peer should retain prepared peer"))
    }
}

impl<B> Drop for ManagedRunningNativePeer<B>
where
    B: AutodiffBackend + Clone + 'static,
{
    fn drop(&mut self) {
        let _ = self.stop_and_join(Some(DROP_SHUTDOWN_TIMEOUT));
    }
}

impl<B> Deref for ManagedRunningNativePeer<B>
where
    B: AutodiffBackend + Clone + 'static,
{
    type Target = RunningNode<SelectedWorkloadProject<DragonProjectFamily<B>>>;

    fn deref(&self) -> &Self::Target {
        self.running
            .as_ref()
            .expect("managed native peer should retain running node")
    }
}

impl<B> DerefMut for ManagedRunningNativePeer<B>
where
    B: AutodiffBackend + Clone + 'static,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.running
            .as_mut()
            .expect("managed native peer should retain running node")
    }
}

pub fn spawn_prepared_native_peer<B>(
    prepared: PreparedNativePeer<B>,
) -> Result<ManagedRunningNativePeer<B>>
where
    B: AutodiffBackend + Clone + 'static,
{
    let p2p_run_id = format!(
        "p2p-{}-{}",
        prepared
            .manifests
            .experiment_directory
            .first()
            .map(|entry| entry.experiment_id.as_str())
            .unwrap_or("dragon"),
        prepared
            .manifests
            .experiment_directory
            .first()
            .map(|entry| entry.current_revision_id.as_str())
            .unwrap_or("revision"),
    );
    let event_bus_config = TrainingEventBusConfig::default();
    let (p2p_plugin, p2p_event_bus) =
        P2pTrainingIngressPlugin::channel(event_bus_config.queue_capacity);
    let p2p_run = TrainingRunConfig::new(
        p2p_run_id.clone(),
        p2p_run_id.clone(),
        prepared.storage_root.join("ecs/p2p"),
        1,
    );
    let p2p_run_options = TrainingRunOptions {
        sinks: prepared.config.training.events.clone(),
        gates: prepared.config.training.gates.clone(),
        ..TrainingRunOptions::default()
    };
    let p2p_event_thread = TrainingRuntimeThread::spawn(
        move || {
            let mut app = App::new();
            app.add_plugins(TrainingPlugins).add_plugins(p2p_plugin);
            app.try_add_training_run_with(p2p_run, p2p_run_options)?;
            Ok(app)
        },
        event_bus_config,
    )?;
    let running = prepared
        .builder
        .clone()
        .with_training_window_observer(P2pTrainingEcsObserver::new(
            p2p_run_id.clone(),
            p2p_event_bus.clone(),
        ))
        .spawn()?;
    let capability = native_p2p_capability_assessment(&prepared, &p2p_run_id);
    p2p_event_bus.send_capability(capability)?;
    let stop_flag = Arc::new(AtomicBool::new(false));
    let trainer_requested = matches!(
        prepared.target_decision.requested_target,
        DragonNativeTarget::Auto | DragonNativeTarget::Trainer
    );
    let monitor_thread = if trainer_requested {
        let prepared_for_monitor = prepared.clone();
        let event_bus = p2p_event_bus.clone();
        let run_id = p2p_run_id.clone();
        let stop_flag_for_thread = Arc::clone(&stop_flag);
        let telemetry = running.telemetry();
        let control = running.control_handle();
        Some(
            thread::Builder::new()
                .name("dragon-native-capability-monitor".into())
                .spawn(move || {
                    run_native_capability_monitor(
                        prepared_for_monitor,
                        event_bus,
                        run_id,
                        stop_flag_for_thread,
                        telemetry,
                        control,
                    )
                })?,
        )
    } else {
        None
    };

    Ok(ManagedRunningNativePeer {
        prepared: Some(prepared),
        running: Some(running),
        p2p_event_thread: Some(p2p_event_thread),
        p2p_event_bus,
        stop_flag,
        monitor_thread,
    })
}

fn run_native_capability_monitor<B>(
    prepared: PreparedNativePeer<B>,
    event_bus: burn_p2p::ecs::P2pTrainingEventBus,
    run_id: String,
    stop_flag: Arc<AtomicBool>,
    telemetry: TelemetryHandle,
    control: ControlHandle,
) where
    B: AutodiffBackend + Clone + 'static,
{
    let policy = prepared.capability_reprobe_policy.clone();
    let trainer_roles = native_trainer_roles(&prepared.backend_label);
    let initial_roles = telemetry.snapshot().configured_roles;
    let mut trainer_active = contains_trainer_role(&initial_roles);
    let mut failure_count = prepared
        .runtime_downgrade_failure_count()
        .unwrap_or_default();
    let mut probe_failures = 0_u32;
    let mut tracker = NativeReprobeTracker::default();
    let mut last_handled_fit_error = None::<String>;
    let mut last_probe_reason = None::<String>;
    let mut next_probe_at = Instant::now()
        + if trainer_active {
            Duration::ZERO
        } else {
            Duration::from_secs(policy.cooldown_secs)
        };

    while !stop_flag.load(Ordering::SeqCst) {
        let snapshot = telemetry.snapshot();
        if snapshot.status == RuntimeStatus::Failed {
            break;
        }

        if trainer_active {
            if let Some(error) = snapshot.last_error.as_deref()
                && is_probable_trainer_fit_failure(error)
                && last_handled_fit_error.as_deref() != Some(error)
                && control
                    .update_roles(native_read_only_roles(), Duration::from_secs(2))
                    .is_ok()
            {
                let _ = event_bus.send_capability(read_only_capability_assessment(
                    &run_id,
                    format!("runtime trainer fit failure: {error}"),
                ));
                let _ =
                    prepared.persist_runtime_training_failure_with_source(error, "runtime-monitor");
                failure_count = prepared
                    .runtime_downgrade_failure_count()
                    .unwrap_or_else(|_| failure_count.saturating_add(1));
                last_handled_fit_error = Some(error.to_owned());
                trainer_active = false;
                tracker.reset();
                probe_failures = 0;
                next_probe_at = Instant::now()
                    + Duration::from_secs(policy.cooldown_secs)
                        .max(native_reprobe_backoff(&policy, failure_count.max(1)));
            }
        } else if policy.enabled && Instant::now() >= next_probe_at {
            let probe = evaluate_native_reprobe(
                &policy,
                &prepared.footprint,
                prepared.target_decision.trainer_memory_budget_bytes,
                read_native_memory_snapshot(),
            );
            match tracker.observe(&policy, probe) {
                Ok(true) => {
                    let acknowledgement = snapshot
                        .last_error
                        .as_deref()
                        .filter(|error| is_probable_trainer_fit_failure(error))
                        .map(|error| {
                            control.acknowledge_runtime_error(error, Duration::from_secs(2))
                        })
                        .transpose();
                    let restored = acknowledgement
                        .and_then(|_| {
                            control.update_roles(trainer_roles.clone(), Duration::from_secs(2))
                        })
                        .and_then(|_| prepared.clear_runtime_downgrade());
                    match restored {
                        Ok(()) => {
                            let _ = event_bus.send_capability(trainer_capability_assessment(
                                &run_id,
                                &prepared.backend_label,
                                "native capability recovery probe accepted".into(),
                            ));
                            trainer_active = true;
                            failure_count = 0;
                            probe_failures = 0;
                            tracker.reset();
                            last_handled_fit_error = None;
                            last_probe_reason = None;
                        }
                        Err(error) => {
                            probe_failures = probe_failures.saturating_add(1);
                            let reason =
                                format!("native capability recovery transition failed: {error}");
                            if last_probe_reason.as_deref() != Some(reason.as_str()) {
                                let _ = event_bus.send_capability(read_only_capability_assessment(
                                    &run_id,
                                    reason.clone(),
                                ));
                                last_probe_reason = Some(reason);
                            }
                            next_probe_at = Instant::now()
                                + native_reprobe_backoff(&policy, probe_failures.max(1));
                        }
                    }
                }
                Ok(false) => {
                    probe_failures = 0;
                    next_probe_at = Instant::now() + Duration::from_secs(policy.interval_secs);
                }
                Err(reason) => {
                    probe_failures = probe_failures.saturating_add(1);
                    if last_probe_reason.as_deref() != Some(reason.as_str()) {
                        let _ = event_bus.send_capability(read_only_capability_assessment(
                            &run_id,
                            format!("native capability recovery deferred: {reason}"),
                        ));
                        last_probe_reason = Some(reason);
                    }
                    next_probe_at =
                        Instant::now() + native_reprobe_backoff(&policy, probe_failures.max(1));
                }
            }
        }

        thread::sleep(MONITOR_POLL_INTERVAL);
    }
}

fn native_read_only_roles() -> PeerRoleSet {
    PeerRoleSet::new([PeerRole::Viewer])
}

fn native_trainer_roles(backend_label: &str) -> PeerRoleSet {
    if backend_label.eq_ignore_ascii_case("cpu") || backend_label.eq_ignore_ascii_case("ndarray") {
        PeerRoleSet::new([PeerRole::TrainerCpu])
    } else {
        PeerRoleSet::new([PeerRole::TrainerGpu])
    }
}

fn contains_trainer_role(roles: &PeerRoleSet) -> bool {
    roles.contains(&PeerRole::TrainerCpu) || roles.contains(&PeerRole::TrainerGpu)
}

fn read_only_capability_assessment(run_id: &str, reason: String) -> P2pCapabilityAssessment {
    P2pCapabilityAssessment {
        run_id: run_id.into(),
        participation: PipelineParticipation::Observer,
        compute: PipelineComputeClass::None,
        supported_participation: BTreeSet::from([
            PipelineParticipation::Observer,
            PipelineParticipation::Validator,
        ]),
        reason,
    }
}

fn trainer_capability_assessment(
    run_id: &str,
    backend_label: &str,
    reason: String,
) -> P2pCapabilityAssessment {
    P2pCapabilityAssessment {
        run_id: run_id.into(),
        participation: PipelineParticipation::Trainer,
        compute: if backend_label.eq_ignore_ascii_case("cpu")
            || backend_label.eq_ignore_ascii_case("ndarray")
        {
            PipelineComputeClass::Cpu
        } else {
            PipelineComputeClass::Accelerator
        },
        supported_participation: BTreeSet::from([
            PipelineParticipation::Observer,
            PipelineParticipation::Validator,
            PipelineParticipation::Trainer,
        ]),
        reason,
    }
}

fn native_p2p_capability_assessment<B>(
    prepared: &PreparedNativePeer<B>,
    run_id: &str,
) -> P2pCapabilityAssessment
where
    B: AutodiffBackend + Clone + 'static,
{
    let participation = match (
        prepared.target_decision.effective_target,
        prepared.target_decision.can_train,
    ) {
        (DragonNativeTarget::Auto | DragonNativeTarget::Trainer, true) => {
            PipelineParticipation::Trainer
        }
        (DragonNativeTarget::Auto | DragonNativeTarget::Trainer, false) => {
            PipelineParticipation::Observer
        }
        (DragonNativeTarget::Validator, _) => PipelineParticipation::Validator,
        (DragonNativeTarget::Reducer, _) => PipelineParticipation::Aggregator,
    };
    let compute = if participation == PipelineParticipation::Observer {
        PipelineComputeClass::None
    } else if prepared.backend_label.eq_ignore_ascii_case("cpu") {
        PipelineComputeClass::Cpu
    } else {
        PipelineComputeClass::Accelerator
    };
    let mut supported_participation = BTreeSet::from([PipelineParticipation::Observer]);
    supported_participation.insert(participation);
    if participation.can_train() {
        supported_participation.insert(PipelineParticipation::Validator);
    }
    P2pCapabilityAssessment {
        run_id: run_id.into(),
        participation,
        compute,
        supported_participation,
        reason: prepared
            .target_decision
            .downgrade_reason
            .clone()
            .unwrap_or_else(|| "native capability assessment accepted".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_runtime_downgrade_uses_read_only_network_roles() {
        let roles = native_read_only_roles();
        assert!(roles.contains(&PeerRole::Viewer));
        assert!(!roles.contains(&PeerRole::TrainerGpu));
        assert!(!roles.contains(&PeerRole::TrainerCpu));
    }
}
