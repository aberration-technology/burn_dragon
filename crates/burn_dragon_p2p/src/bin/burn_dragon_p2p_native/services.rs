//! Head-mirror and validator service loops plus publication helpers.

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn run_prepared_head_mirror<B>(
    prepared: PreparedNativePeer<B>,
    config: &DragonNativePeerConfig,
    auth_bundle: Option<&DragonNativeAuthBundle>,
    backend: BackendArg,
    status_interval_secs: u64,
    head_sync_interval_secs: u64,
    initialize_head_on_start: bool,
    restore_head_on_start: bool,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
{
    let experiment_entry = prepared
        .manifests
        .experiment_directory
        .first()
        .cloned()
        .ok_or_else(|| anyhow!("prepared head mirror is missing an experiment"))?;
    eprintln!(
        "starting burn_dragon head mirror: experiment={} backend={} target={:?} can_train={} edge={} seeds={} storage_root={}",
        prepared.experiment_kind.workload_slug(),
        backend.as_label(),
        prepared.target_decision.effective_target,
        prepared.target_decision.can_train,
        config.effective_edge_base_url().unwrap_or("<none>"),
        config.effective_seed_node_urls().len(),
        config.storage_root.display(),
    );
    if let Some(reason) = prepared.target_decision.downgrade_reason.as_deref() {
        eprintln!("capability decision: {reason}");
    }
    if !prepared.target_decision.can_train {
        eprintln!(
            "head mirror continuing with estimated training footprint above the configured budget; target={:?}",
            prepared.target_decision.effective_target,
        );
    }

    let mut running = spawn_prepared_native_peer(prepared)?;
    wait_for_runtime_ready(&running, RUNTIME_READY_TIMEOUT)?;
    let shutdown_requested = Arc::new(AtomicBool::new(false));
    let shutdown_requested_for_handler = Arc::clone(&shutdown_requested);
    let control = running.control_handle();
    ctrlc::set_handler(move || {
        if !shutdown_requested_for_handler.swap(true, Ordering::SeqCst) {
            let _ = control.shutdown();
        }
    })
    .context("failed to install ctrl-c handler")?;

    let experiment = running.mainnet().experiment(
        experiment_entry.study_id.clone(),
        experiment_entry.experiment_id.clone(),
        experiment_entry.current_revision_id.clone(),
    );
    let edge_registration = auth_bundle
        .and_then(|auth| {
            auth.session_id.as_ref().and_then(|session_id| {
                let edge_base_url = auth
                    .edge_base_url
                    .clone()
                    .or_else(|| config.effective_edge_base_url().map(ToOwned::to_owned));
                edge_base_url.map(|edge_base_url| (edge_base_url, session_id.clone()))
            })
        })
        .map(|(edge_base_url, session_id)| {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .context("failed to build async runtime for head mirror edge registration")?;
            Ok::<_, anyhow::Error>((runtime, edge_base_url, session_id))
        })
        .transpose()?;
    let status_interval = Duration::from_secs(status_interval_secs.max(1));
    let head_sync_interval = Duration::from_secs(head_sync_interval_secs.max(1));
    let mut last_status = Instant::now()
        .checked_sub(status_interval)
        .unwrap_or_else(Instant::now);
    let mut last_head_sync = Instant::now()
        .checked_sub(head_sync_interval)
        .unwrap_or_else(Instant::now);
    let mut served_head_id = None;
    let mut edge_registered_head_id = None;

    loop {
        if last_head_sync.elapsed() >= head_sync_interval {
            let head = sync_or_initialize_latest_head_provider(
                &mut running,
                &experiment,
                initialize_head_on_start,
                restore_head_on_start,
                &mut served_head_id,
                HeadProviderSyncMode::LatestPromoted,
                "head-mirror",
            )?;
            let snapshot = running.snapshot();
            let visible_promoted = latest_visible_promoted_head_announcement(
                &snapshot.control_plane,
                &experiment,
                head.as_ref(),
            );
            if let (Some(announcement), Some((registration_runtime, edge_base_url, session_id))) =
                (visible_promoted.as_ref(), edge_registration.as_ref())
            {
                if edge_registered_head_id.as_ref() != Some(&announcement.head.head_id) {
                    match register_live_head_with_edge_options(
                        registration_runtime,
                        edge_base_url,
                        session_id,
                        Some(&experiment_entry),
                        announcement,
                    ) {
                        Ok(()) => {
                            eprintln!(
                                "head-mirror-edge-visible-head-registered head_id={} provider={}",
                                announcement.head.head_id.as_str(),
                                announcement
                                    .provider_peer_id
                                    .as_ref()
                                    .map(|peer_id| peer_id.as_str())
                                    .unwrap_or("-"),
                            );
                            edge_registered_head_id = Some(announcement.head.head_id.clone());
                        }
                        Err(error) => {
                            eprintln!(
                                "head-mirror-edge-visible-head-registration-failed head_id={} provider={} error={error}",
                                announcement.head.head_id.as_str(),
                                announcement
                                    .provider_peer_id
                                    .as_ref()
                                    .map(|peer_id| peer_id.as_str())
                                    .unwrap_or("-"),
                            );
                            if let (Some(head), Some(local_peer_id)) =
                                (head.as_ref(), snapshot.local_peer_id.clone())
                                && should_register_edge_local_fallback(
                                    &announcement.head,
                                    head,
                                    edge_registered_head_id.as_ref(),
                                )
                            {
                                let local_announcement = edge_local_head_announcement(
                                    head,
                                    &experiment,
                                    local_peer_id.clone(),
                                )?;
                                match register_live_head_with_edge_options(
                                    registration_runtime,
                                    edge_base_url,
                                    session_id,
                                    Some(&experiment_entry),
                                    &local_announcement,
                                ) {
                                    Ok(()) => {
                                        eprintln!(
                                            "head-mirror-edge-local-fallback-registered head_id={} provider={} superseded_head={}",
                                            local_announcement.head.head_id.as_str(),
                                            local_peer_id.as_str(),
                                            announcement.head.head_id.as_str(),
                                        );
                                        edge_registered_head_id =
                                            Some(local_announcement.head.head_id.clone());
                                    }
                                    Err(fallback_error) => {
                                        eprintln!(
                                            "head-mirror-edge-local-fallback-registration-failed head_id={} provider={} superseded_head={} error={fallback_error}",
                                            local_announcement.head.head_id.as_str(),
                                            local_peer_id.as_str(),
                                            announcement.head.head_id.as_str(),
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            } else if let (Some(head), Some((registration_runtime, edge_base_url, session_id))) =
                (head.as_ref(), edge_registration.as_ref())
            {
                let snapshot = running.snapshot();
                if let Some(local_peer_id) = snapshot.local_peer_id
                    && edge_registered_head_id.as_ref() != Some(&head.head_id)
                {
                    let announcement =
                        edge_local_head_announcement(head, &experiment, local_peer_id.clone())?;
                    if let Err(error) = register_live_head_with_edge_options(
                        registration_runtime,
                        edge_base_url,
                        session_id,
                        Some(&experiment_entry),
                        &announcement,
                    ) {
                        eprintln!(
                            "head-mirror-edge-local-registration-failed head_id={} provider={} error={error}",
                            head.head_id.as_str(),
                            local_peer_id.as_str(),
                        );
                    } else {
                        eprintln!(
                            "head-mirror-edge-local-registered head_id={} provider={}",
                            head.head_id.as_str(),
                            local_peer_id.as_str(),
                        );
                        edge_registered_head_id = Some(head.head_id.clone());
                    }
                }
            }
            last_head_sync = Instant::now();
        }

        let snapshot = running.snapshot();
        if status_interval_secs > 0 && last_status.elapsed() >= status_interval {
            eprintln!(
                "head-mirror-status status={:?} node_state={:?} connected_peers={} served_head={} edge_registered_head={} last_error={}",
                snapshot.status,
                snapshot.node_state,
                snapshot.connected_peers,
                served_head_id
                    .as_ref()
                    .map(|head_id| head_id.as_str())
                    .unwrap_or("-"),
                edge_registered_head_id
                    .as_ref()
                    .map(|head_id| head_id.as_str())
                    .unwrap_or("-"),
                operator_visible_last_error(snapshot.last_error.as_deref())
                    .as_deref()
                    .unwrap_or("-"),
            );
            last_status = Instant::now();
        }

        match snapshot.status {
            RuntimeStatus::Failed => {
                let reason = snapshot
                    .last_error
                    .unwrap_or_else(|| "peer runtime failed".into());
                let _ = running.shutdown();
                let _ = running.await_termination_timeout(SHUTDOWN_TIMEOUT);
                bail!("head mirror failed: {reason}");
            }
            RuntimeStatus::Stopped => {
                let _prepared = running.await_termination_timeout(SHUTDOWN_TIMEOUT)?;
                eprintln!("head mirror stopped cleanly");
                return Ok(());
            }
            _ => {}
        }

        thread::sleep(STATUS_POLL_INTERVAL);
    }
}

pub(super) fn edge_local_head_announcement(
    head: &HeadDescriptor,
    experiment: &ExperimentHandle,
    local_peer_id: PeerId,
) -> Result<HeadAnnouncement> {
    Ok(HeadAnnouncement {
        overlay: experiment.overlay_set()?.heads,
        provider_peer_id: Some(local_peer_id),
        head: head.clone(),
        announced_at: chrono::Utc::now(),
    })
}

pub(super) fn should_register_edge_local_fallback(
    failed_visible_head: &HeadDescriptor,
    local_head: &HeadDescriptor,
    edge_registered_head_id: Option<&HeadId>,
) -> bool {
    failed_visible_head.head_id != local_head.head_id
        && edge_registered_head_id != Some(&local_head.head_id)
}

pub(super) fn latest_visible_promoted_head_announcement(
    snapshot: &ControlPlaneSnapshot,
    experiment: &ExperimentHandle,
    baseline: Option<&HeadDescriptor>,
) -> Option<HeadAnnouncement> {
    snapshot
        .head_announcements
        .iter()
        .filter(|announcement| announcement.provider_peer_id.is_some())
        .filter(|announcement| head_matches_experiment(&announcement.head, experiment))
        .filter(|announcement| {
            baseline.is_none_or(|baseline| head_is_newer_than(&announcement.head, baseline))
        })
        .max_by(|left, right| {
            left.head
                .global_step
                .cmp(&right.head.global_step)
                .then(left.head.created_at.cmp(&right.head.created_at))
                .then(left.announced_at.cmp(&right.announced_at))
        })
        .cloned()
}

pub(super) fn head_matches_experiment(
    head: &HeadDescriptor,
    experiment: &ExperimentHandle,
) -> bool {
    head.study_id == experiment.study_id
        && head.experiment_id == experiment.experiment_id
        && head.revision_id == experiment.revision_id
}

pub(super) fn head_is_newer_than(candidate: &HeadDescriptor, baseline: &HeadDescriptor) -> bool {
    candidate.global_step > baseline.global_step
        || (candidate.global_step == baseline.global_step
            && candidate.created_at > baseline.created_at
            && candidate.head_id != baseline.head_id)
}

pub(super) fn run_prepared_validator_daemon<B>(
    prepared: PreparedNativePeer<B>,
    config: &DragonNativePeerConfig,
    backend: BackendArg,
    status_interval_secs: u64,
    validation_interval_millis: u64,
    initialize_head_on_start: bool,
    restore_head_on_start: bool,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
{
    let experiment_entry = prepared
        .manifests
        .experiment_directory
        .first()
        .cloned()
        .ok_or_else(|| anyhow!("prepared validator manifest bundle is missing an experiment"))?;
    let diffusion_promotion = directory_entry_promotes_with_diffusion(&experiment_entry);
    eprintln!(
        "starting burn_dragon validator daemon: experiment={} backend={} target={:?} can_train={} promotion={} edge={} seeds={} storage_root={}",
        prepared.experiment_kind.workload_slug(),
        backend.as_label(),
        prepared.target_decision.effective_target,
        prepared.target_decision.can_train,
        if diffusion_promotion {
            "diffusion-steady-state"
        } else {
            "validator-quorum"
        },
        config.effective_edge_base_url().unwrap_or("<none>"),
        config.effective_seed_node_urls().len(),
        config.storage_root.display(),
    );
    if let Some(reason) = prepared.target_decision.downgrade_reason.as_deref() {
        eprintln!("capability decision: {reason}");
    }
    if prepared.target_decision.effective_target
        != burn_dragon_p2p::config::DragonNativeTarget::Validator
    {
        bail!(
            "validator daemon requires effective validator target; resolved {:?}",
            prepared.target_decision.effective_target
        );
    }

    let mut running = spawn_prepared_native_peer(prepared)?;
    wait_for_runtime_ready(&running, RUNTIME_READY_TIMEOUT)?;
    let ready_snapshot = running.snapshot();
    eprintln!(
        "validator-runtime-ready local_peer_id={} connected_peers={}",
        ready_snapshot
            .local_peer_id
            .as_ref()
            .map(|peer_id| peer_id.as_str())
            .unwrap_or("-"),
        ready_snapshot.connected_peers,
    );
    let shutdown_requested = Arc::new(AtomicBool::new(false));
    let shutdown_requested_for_handler = Arc::clone(&shutdown_requested);
    let control = running.control_handle();
    ctrlc::set_handler(move || {
        if !shutdown_requested_for_handler.swap(true, Ordering::SeqCst) {
            let _ = control.shutdown();
        }
    })
    .context("failed to install ctrl-c handler")?;

    let experiment = running.mainnet().experiment(
        experiment_entry.study_id,
        experiment_entry.experiment_id,
        experiment_entry.current_revision_id,
    );
    let mut served_head_id = None;
    let mut served_head = None;
    let mut evaluated_head_id = None;

    let status_interval = Duration::from_secs(status_interval_secs.max(1));
    let validation_interval = Duration::from_millis(validation_interval_millis.max(25));
    let head_sync_interval = Duration::from_secs(status_interval_secs.clamp(1, 5));
    let mut last_status = Instant::now()
        .checked_sub(status_interval)
        .unwrap_or_else(Instant::now);
    let mut last_validation = Instant::now()
        .checked_sub(validation_interval)
        .unwrap_or_else(Instant::now);
    let mut last_head_sync = Instant::now()
        .checked_sub(head_sync_interval)
        .unwrap_or_else(Instant::now);
    let mut head_sync_attempts = 0_u64;

    loop {
        if last_head_sync.elapsed() >= head_sync_interval {
            head_sync_attempts = head_sync_attempts.saturating_add(1);
            match sync_or_initialize_latest_head_provider(
                &mut running,
                &experiment,
                initialize_head_on_start,
                restore_head_on_start,
                &mut served_head_id,
                if diffusion_promotion {
                    HeadProviderSyncMode::LatestPromoted
                } else {
                    HeadProviderSyncMode::DirectoryCurrent
                },
                "validator",
            ) {
                Ok(Some(head)) => served_head = Some(head),
                Ok(None) => {
                    if head_sync_attempts == 1 || head_sync_attempts.is_multiple_of(12) {
                        let snapshot = running.snapshot();
                        eprintln!(
                            "validator-head-sync-waiting attempts={} connected_peers={} local_heads={} node_state={:?} last_error={}",
                            head_sync_attempts,
                            snapshot.connected_peers,
                            snapshot.control_plane.head_announcements.len(),
                            snapshot.node_state,
                            operator_visible_last_error(snapshot.last_error.as_deref())
                                .as_deref()
                                .unwrap_or("-"),
                        );
                    }
                }
                Err(error) => {
                    eprintln!("validator-head-sync-error: {error}");
                }
            }
            last_head_sync = Instant::now();
        }

        let snapshot = running.snapshot();
        if status_interval_secs > 0 && last_status.elapsed() >= status_interval {
            eprintln!(
                "validator-status status={:?} node_state={:?} connected_peers={} served_head={} evaluated_head={} last_error={}",
                snapshot.status,
                snapshot.node_state,
                snapshot.connected_peers,
                served_head_id
                    .as_ref()
                    .map(|head_id| head_id.as_str())
                    .unwrap_or("-"),
                evaluated_head_id
                    .as_ref()
                    .map(|head_id: &burn_p2p::HeadId| head_id.as_str())
                    .unwrap_or("-"),
                operator_visible_last_error(snapshot.last_error.as_deref())
                    .as_deref()
                    .unwrap_or("-"),
            );
            last_status = Instant::now();
        }

        match snapshot.status {
            RuntimeStatus::Failed => {
                let reason = snapshot
                    .last_error
                    .unwrap_or_else(|| "validator runtime failed".into());
                let _ = running.shutdown();
                let _ = running.await_termination_timeout(SHUTDOWN_TIMEOUT);
                bail!("validator runtime failed: {reason}");
            }
            RuntimeStatus::Stopped => {
                let _prepared = running.await_termination_timeout(SHUTDOWN_TIMEOUT)?;
                eprintln!("validator stopped cleanly");
                return Ok(());
            }
            _ => {}
        }

        if last_validation.elapsed() >= validation_interval {
            if served_head.is_none() {
                last_validation = Instant::now();
                thread::sleep(STATUS_POLL_INTERVAL);
                continue;
            }

            if diffusion_promotion
                && let Some(head) = served_head.as_ref()
                && evaluated_head_id.as_ref() != Some(&head.head_id)
            {
                let started = Instant::now();
                match running.evaluate_and_record_materialized_head(
                    &experiment,
                    head,
                    burn_p2p::EvalSplit::Validation,
                ) {
                    Ok(evaluation) => {
                        eprintln!(
                            "validator-head-evaluated head_id={} revision={} global_step={} samples={} metrics={} elapsed_ms={}",
                            head.head_id.as_str(),
                            head.revision_id.as_str(),
                            head.global_step,
                            evaluation.report.sample_count,
                            evaluation.report.metric_values.len(),
                            started.elapsed().as_millis(),
                        );
                        evaluated_head_id = Some(head.head_id.clone());
                    }
                    Err(error) => {
                        eprintln!(
                            "validator-head-evaluation-error head_id={} revision={} error={error}",
                            head.head_id.as_str(),
                            head.revision_id.as_str(),
                        );
                    }
                }
            }

            if diffusion_promotion {
                if let Err(error) = running.advance_diffusion_steady_state(&experiment, None, None)
                {
                    eprintln!("validator-diffusion-pass-error: {error}");
                }
            } else {
                match running.validate_candidates_once(&experiment) {
                    Ok(Some(outcome)) => {
                        eprintln!(
                            "validator-promoted merged_head_id={} global_step={}",
                            outcome.merged_head.head_id.as_str(),
                            outcome.merged_head.global_step,
                        );
                    }
                    Ok(None) => {}
                    Err(error) => {
                        eprintln!("validator-validation-pass-error: {error}");
                    }
                }
                let validation_snapshot = running.snapshot();
                if let Some(local_peer_id) = validation_snapshot.local_peer_id.as_ref()
                    && let Some(binding) = validation_snapshot
                        .control_plane
                        .reduction_certificate_announcements
                        .iter()
                        .filter(|announcement| {
                            announcement.certificate.promoter_peer_id == *local_peer_id
                        })
                        .max_by_key(|announcement| announcement.certificate.issued_at)
                        .and_then(|announcement| announcement.certificate.evaluation.as_ref())
                    && evaluated_head_id.as_ref() != Some(&binding.head_id)
                {
                    eprintln!(
                        "validator-head-attested head_id={} artifact={} eval_protocol={} eval_report={}",
                        binding.head_id.as_str(),
                        binding.artifact_id.as_str(),
                        binding.eval_protocol_id.as_str(),
                        binding.eval_report_id.as_str(),
                    );
                    evaluated_head_id = Some(binding.head_id.clone());
                }
            }
            last_validation = Instant::now();
        }

        thread::sleep(STATUS_POLL_INTERVAL);
    }
}

pub(super) fn wait_for_runtime_ready<B>(
    running: &ManagedRunningNativePeer<B>,
    timeout: Duration,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
{
    let deadline = Instant::now() + timeout;
    loop {
        let snapshot = running.snapshot();
        if snapshot.local_peer_id.is_some() && !snapshot.listen_addresses.is_empty() {
            return Ok(());
        }
        if snapshot.status == RuntimeStatus::Failed {
            bail!(
                "peer runtime failed before becoming ready: {}",
                snapshot.last_error.as_deref().unwrap_or("unknown error"),
            );
        }
        if snapshot.status == RuntimeStatus::Stopped {
            bail!("peer runtime stopped before becoming ready");
        }
        if Instant::now() >= deadline {
            bail!("peer runtime did not become ready within {:?}", timeout);
        }
        thread::sleep(STATUS_POLL_INTERVAL);
    }
}

pub(super) fn ensure_p2p_publication_connectivity<B>(
    running: &ManagedRunningNativePeer<B>,
    config: &DragonNativePeerConfig,
    context: &str,
    timeout: Duration,
) -> Result<usize>
where
    B: AutodiffBackend + Clone + 'static,
{
    let bootstrap_peers = config.effective_bootstrap_peers()?;
    if bootstrap_peers.is_empty() {
        let connected_peers = running.snapshot().connected_peers;
        eprintln!(
            "train-window-once progress: p2p connectivity check skipped context={context:?} reason=no-bootstrap-peers connected_peers={connected_peers}"
        );
        return Ok(connected_peers);
    }

    let control = running.control_handle();
    let deadline = Instant::now() + timeout;
    let mut last_dial = Instant::now()
        .checked_sub(TRAIN_WINDOW_P2P_REDIAL_INTERVAL)
        .unwrap_or_else(Instant::now);
    let mut last_dial_errors = Vec::new();

    loop {
        let snapshot = running.snapshot();
        if snapshot.connected_peers > 0 {
            eprintln!(
                "train-window-once progress: p2p connectivity ready context={context:?} connected_peers={} seeds={}",
                snapshot.connected_peers,
                bootstrap_peers.len(),
            );
            return Ok(snapshot.connected_peers);
        }
        match snapshot.status {
            RuntimeStatus::Failed => {
                bail!(
                    "train-window-once runtime failed while waiting for p2p connectivity {context:?}: {}",
                    snapshot.last_error.as_deref().unwrap_or("unknown error"),
                );
            }
            RuntimeStatus::Stopped => {
                bail!(
                    "train-window-once runtime stopped while waiting for p2p connectivity {context:?}"
                );
            }
            _ => {}
        }
        if Instant::now() >= deadline {
            let last_error = operator_visible_last_error(snapshot.last_error.as_deref())
                .unwrap_or_else(|| "-".into());
            let seed_preview = bootstrap_peers
                .iter()
                .take(4)
                .map(|address| address.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            let dial_errors = if last_dial_errors.is_empty() {
                "-".to_owned()
            } else {
                last_dial_errors.join("; ")
            };
            bail!(
                "train-window-once p2p connectivity unavailable {context:?} after {}s; connected_peers=0 seeds={} seed_preview=[{}] last_error={} dial_errors={}",
                timeout.as_secs(),
                bootstrap_peers.len(),
                seed_preview,
                last_error,
                dial_errors,
            );
        }

        if last_dial.elapsed() >= TRAIN_WINDOW_P2P_REDIAL_INTERVAL {
            last_dial_errors.clear();
            for address in &bootstrap_peers {
                if let Err(error) = control.dial_address(address.clone()) {
                    last_dial_errors.push(format!("{}: {error}", address.as_str()));
                }
            }
            let last_error = operator_visible_last_error(snapshot.last_error.as_deref())
                .unwrap_or_else(|| "-".into());
            eprintln!(
                "train-window-once progress: waiting for p2p connectivity context={context:?} connected_peers=0 seeds={} dial_errors={} last_error={}",
                bootstrap_peers.len(),
                last_dial_errors.len(),
                last_error,
            );
            last_dial = Instant::now();
        }

        thread::sleep(STATUS_POLL_INTERVAL);
    }
}

pub(super) fn publish_train_window_head<B>(
    running: &ManagedRunningNativePeer<B>,
    experiment: &ExperimentHandle,
    local_peer_id: &PeerId,
    head: &HeadDescriptor,
    context: &str,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
{
    running
        .control_handle()
        .publish_head(HeadAnnouncement {
            overlay: experiment.overlay_set()?.heads,
            provider_peer_id: Some(local_peer_id.clone()),
            head: head.clone(),
            announced_at: chrono::Utc::now(),
        })
        .with_context(|| {
            format!(
                "failed to announce train-window-once head {} {context}",
                head.head_id.as_str()
            )
        })?;
    eprintln!(
        "train-window-once progress: announced published head context={context:?} head={} step={}",
        head.head_id.as_str(),
        head.global_step,
    );
    Ok(())
}
