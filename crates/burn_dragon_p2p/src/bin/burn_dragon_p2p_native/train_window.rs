//! One-shot train windows, head synchronization, and edge publication.

use super::*;

pub(super) fn run_prepared_train_window_once<B>(
    prepared: PreparedNativePeer<B>,
    config: &DragonNativePeerConfig,
    auth_bundle: Option<&DragonNativeAuthBundle>,
    backend: BackendArg,
    options: TrainWindowOnceRunOptions<'_>,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
{
    let experiment_entry = prepared
        .manifests
        .experiment_directory
        .first()
        .cloned()
        .ok_or_else(|| anyhow!("prepared native peer is missing an experiment"))?;
    eprintln!(
        "starting burn_dragon train-window-once: experiment={} backend={} target={:?} can_train={} edge={} seeds={} storage_root={}",
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
    if !prepared.target_decision.can_train
        || !matches!(
            prepared.target_decision.effective_target,
            DragonNativeTarget::Auto | DragonNativeTarget::Trainer
        )
    {
        bail!(
            "train-window-once requires a trainer-capable target; resolved {:?}",
            prepared.target_decision.effective_target
        );
    }

    let started = Instant::now();
    eprintln!("train-window-once progress: spawning native peer runtime");
    let mut running = spawn_prepared_native_peer(prepared)?;
    let edge_registration = if options.mirror_live_head_to_edge {
        train_window_edge_registration(config, auth_bundle)?
    } else {
        None
    };
    let report_result = (|| -> Result<TrainWindowOnceReport> {
        eprintln!("train-window-once progress: waiting for runtime readiness");
        wait_for_runtime_ready(&running, RUNTIME_READY_TIMEOUT)?;
        let local_peer_id = running
            .snapshot()
            .local_peer_id
            .ok_or_else(|| anyhow!("peer runtime did not report a local peer id"))?;
        eprintln!(
            "train-window-once progress: runtime ready peer={} elapsed_ms={}",
            local_peer_id,
            started.elapsed().as_millis()
        );
        ensure_p2p_publication_connectivity(
            &running,
            config,
            "before canonical head sync",
            TRAIN_WINDOW_P2P_CONNECTIVITY_TIMEOUT,
        )?;
        let experiment = running.mainnet().experiment(
            experiment_entry.study_id.clone(),
            experiment_entry.experiment_id.clone(),
            experiment_entry.current_revision_id.clone(),
        );
        let mut served_head_id = None;
        eprintln!("train-window-once progress: resolving active canonical head");
        let base_head = wait_for_head_provider(
            &mut running,
            &experiment,
            options.initialize_head_on_start,
            options.restore_head_on_start,
            &mut served_head_id,
            "trainer",
            Duration::from_secs(options.head_sync_timeout_secs.max(1)),
        )?;
        eprintln!(
            "train-window-once progress: active head ready head={} step={} served_head={:?} elapsed_ms={}",
            base_head.head_id,
            base_head.global_step,
            served_head_id,
            started.elapsed().as_millis()
        );
        eprintln!("train-window-once progress: preparing pinned trainer state");
        eprintln!(
            "train-window-once progress: trainer ready; running one training window elapsed_ms={}",
            started.elapsed().as_millis()
        );
        let outcome = running.train_window_once_with_pinned_head(&experiment, Some(&base_head))?;
        let train_loss = outcome
            .report
            .stats
            .get("train_loss")
            .or_else(|| outcome.report.stats.get("loss"));
        eprintln!(
            "train-window-once progress: window published head={} step={} artifact={} train_loss={:?} data_fetch_ms={} publish_ms={} elapsed_ms={}",
            outcome.head.head_id,
            outcome.head.global_step,
            outcome.artifact.artifact_id,
            train_loss,
            outcome.timing.data_fetch_time_ms,
            outcome.timing.publish_latency_ms,
            started.elapsed().as_millis()
        );
        ensure_p2p_publication_connectivity(
            &running,
            config,
            "after local training before diffusion publication",
            TRAIN_WINDOW_P2P_CONNECTIVITY_TIMEOUT,
        )?;
        publish_train_window_head(
            &running,
            &experiment,
            &local_peer_id,
            &outcome.head,
            "after local training",
        )?;
        let mut diffusion_settlement = None;
        if options.settle_diffusion || options.serve_after_publish_secs > 0 {
            if directory_entry_promotes_with_diffusion(&experiment_entry) {
                let passes_requested = if options.settle_diffusion {
                    options.diffusion_settle_passes.max(1)
                } else {
                    0
                };
                let mut passes_completed = 0_u32;
                if options.settle_diffusion {
                    for pass in 1..=passes_requested {
                        eprintln!(
                            "train-window-once progress: diffusion settle pass={} starting elapsed_ms={}",
                            pass,
                            started.elapsed().as_millis(),
                        );
                        ensure_p2p_publication_connectivity(
                            &running,
                            config,
                            "before diffusion settle pass",
                            TRAIN_WINDOW_P2P_CONNECTIVITY_TIMEOUT,
                        )?;
                        publish_train_window_head(
                            &running,
                            &experiment,
                            &local_peer_id,
                            &outcome.head,
                            "before diffusion settle pass",
                        )?;
                        running.advance_diffusion_steady_state(
                            &experiment,
                            Some(outcome.lease.window_id),
                            Some(&base_head.head_id),
                        )?;
                        passes_completed = pass;
                        let snapshot = running.snapshot();
                        eprintln!(
                            "train-window-once progress: diffusion settle pass={} connected_peers={} merge_windows={} updates={} attestations={} certificates={} merges={} elapsed_ms={}",
                            pass,
                            snapshot.connected_peers,
                            snapshot.control_plane.merge_window_announcements.len(),
                            snapshot.control_plane.update_announcements.len(),
                            snapshot
                                .control_plane
                                .trainer_promotion_attestation_announcements
                                .len(),
                            snapshot
                                .control_plane
                                .diffusion_promotion_certificate_announcements
                                .len(),
                            snapshot.control_plane.merge_announcements.len(),
                            started.elapsed().as_millis(),
                        );
                        thread::sleep(Duration::from_millis(250));
                    }
                }
                if options.serve_after_publish_secs > 0 {
                    let serve_for = Duration::from_secs(options.serve_after_publish_secs);
                    let serve_deadline = Instant::now() + serve_for;
                    let status_interval = Duration::from_secs(5);
                    let mut last_status = Instant::now()
                        .checked_sub(status_interval)
                        .unwrap_or_else(Instant::now);
                    let mut last_head_announcement = last_status;
                    eprintln!(
                        "train-window-once progress: serving published artifact for {}s elapsed_ms={}",
                        options.serve_after_publish_secs,
                        started.elapsed().as_millis()
                    );
                    ensure_p2p_publication_connectivity(
                        &running,
                        config,
                        "before serving published artifact",
                        TRAIN_WINDOW_P2P_CONNECTIVITY_TIMEOUT,
                    )?;
                    while Instant::now() < serve_deadline {
                        let mut connected_peers = running.snapshot().connected_peers;
                        if connected_peers == 0 {
                            connected_peers = ensure_p2p_publication_connectivity(
                                &running,
                                config,
                                "while serving published artifact",
                                TRAIN_WINDOW_P2P_CONNECTIVITY_TIMEOUT,
                            )?;
                        }
                        if last_head_announcement.elapsed() >= status_interval {
                            publish_train_window_head(
                                &running,
                                &experiment,
                                &local_peer_id,
                                &outcome.head,
                                "while serving published artifact",
                            )?;
                            last_head_announcement = Instant::now();
                        }
                        let snapshot = running.snapshot();
                        if last_status.elapsed() >= status_interval {
                            eprintln!(
                                "train-window-once progress: serving status connected_peers={} merge_windows={} updates={} attestations={} certificates={} merges={} last_error={} elapsed_ms={}",
                                connected_peers,
                                snapshot.control_plane.merge_window_announcements.len(),
                                snapshot.control_plane.update_announcements.len(),
                                snapshot
                                    .control_plane
                                    .trainer_promotion_attestation_announcements
                                    .len(),
                                snapshot
                                    .control_plane
                                    .diffusion_promotion_certificate_announcements
                                    .len(),
                                snapshot.control_plane.merge_announcements.len(),
                                operator_visible_last_error(snapshot.last_error.as_deref())
                                    .as_deref()
                                    .unwrap_or("-"),
                                started.elapsed().as_millis(),
                            );
                            last_status = Instant::now();
                        }
                        match snapshot.status {
                            RuntimeStatus::Failed => {
                                let reason = snapshot
                                    .last_error
                                    .unwrap_or_else(|| "peer runtime failed".into());
                                bail!("train-window-once runtime failed while serving: {reason}");
                            }
                            RuntimeStatus::Stopped => {
                                bail!("train-window-once runtime stopped while serving");
                            }
                            _ => {}
                        }
                        thread::sleep(STATUS_POLL_INTERVAL);
                    }
                }
                let snapshot = running.snapshot();
                diffusion_settlement = Some(diffusion_settlement_report(
                    &snapshot.control_plane,
                    true,
                    passes_requested,
                    passes_completed,
                    options.serve_after_publish_secs,
                ));
            } else {
                eprintln!(
                    "train-window-once progress: diffusion settlement requested but experiment promotion mode is not diffusion-steady-state"
                );
                let snapshot = running.snapshot();
                diffusion_settlement = Some(diffusion_settlement_report(
                    &snapshot.control_plane,
                    false,
                    0,
                    0,
                    options.serve_after_publish_secs,
                ));
            }
        }
        let mirrored_edge_head = if let Some((registration_runtime, edge_base_url, session_id)) =
            edge_registration.as_ref()
        {
            let announcement = HeadAnnouncement {
                overlay: experiment.overlay_set()?.heads,
                provider_peer_id: Some(local_peer_id.clone()),
                head: outcome.head.clone(),
                announced_at: chrono::Utc::now(),
            };
            eprintln!(
                "train-window-once progress: mirroring settled and served artifact to edge head={} artifact={} elapsed_ms={}",
                announcement.head.head_id.as_str(),
                announcement.head.artifact_id.as_str(),
                started.elapsed().as_millis(),
            );
            Some(
                mirror_head_artifact_with_edge(
                    registration_runtime,
                    edge_base_url,
                    session_id,
                    &announcement,
                )
                .with_context(|| {
                    format!(
                        "failed to mirror settled published head {} artifact {} to edge",
                        announcement.head.head_id.as_str(),
                        announcement.head.artifact_id.as_str()
                    )
                })?,
            )
        } else {
            None
        };
        if let (Some((registration_runtime, edge_base_url, session_id)), Some(edge_announcement)) =
            (edge_registration.as_ref(), mirrored_edge_head)
        {
            register_edge_head_and_directory(
                registration_runtime,
                edge_base_url,
                session_id,
                Some(&experiment_entry),
                edge_announcement,
                Some(&local_peer_id),
            )
            .with_context(|| {
                format!(
                    "failed to register mirrored head {} on edge after diffusion settlement",
                    outcome.head.head_id.as_str(),
                )
            })?;
        }
        Ok(TrainWindowOnceReport {
            experiment_kind: running.prepared().experiment_kind,
            backend: backend.as_label().into(),
            edge_base_url: config.effective_edge_base_url().map(ToOwned::to_owned),
            seed_node_count: config.effective_seed_node_urls().len(),
            effective_target: format!("{:?}", running.prepared().target_decision.effective_target),
            can_train: running.prepared().target_decision.can_train,
            downgrade_reason: running.prepared().target_decision.downgrade_reason.clone(),
            local_peer_id: local_peer_id.as_str().to_owned(),
            base_head_id: base_head.head_id.as_str().to_owned(),
            base_global_step: base_head.global_step,
            published_head_id: outcome.head.head_id.as_str().to_owned(),
            published_global_step: outcome.head.global_step,
            artifact_id: outcome.artifact.artifact_id.as_str().to_owned(),
            contribution_receipt_id: outcome.contribution.receipt_id.as_str().to_owned(),
            lease_window_id: outcome.lease.window_id.0.to_string(),
            lease_microshard_count: outcome.lease.microshards.len(),
            timing: TrainWindowOnceTimingReport {
                data_fetch_time_ms: outcome.timing.data_fetch_time_ms,
                publish_latency_ms: outcome.timing.publish_latency_ms,
            },
            diffusion_settlement,
            metrics: outcome.report.stats,
        })
    })();

    let shutdown_result = running.shutdown();
    let termination_result = running.await_termination_timeout(SHUTDOWN_TIMEOUT);

    if let Err(error) = shutdown_result {
        eprintln!("train-window-once shutdown error: {error}");
    }
    if let Err(error) = termination_result {
        match &report_result {
            Ok(_) => return Err(error),
            Err(_) => eprintln!("train-window-once termination error: {error}"),
        }
    }

    let report = report_result?;
    if options.require_head_advanced && report.published_global_step <= report.base_global_step {
        bail!(
            "train-window-once did not advance the experiment head: base step {} published step {}",
            report.base_global_step,
            report.published_global_step
        );
    }
    write_output(options.output, options.output_format, &report)
}

pub(super) fn train_window_edge_registration(
    config: &DragonNativePeerConfig,
    auth_bundle: Option<&DragonNativeAuthBundle>,
) -> Result<Option<(tokio::runtime::Runtime, String, String)>> {
    let Some((edge_base_url, session_id)) = auth_bundle.and_then(|auth| {
        auth.session_id.as_ref().and_then(|session_id| {
            let edge_base_url = auth
                .edge_base_url
                .clone()
                .or_else(|| config.effective_edge_base_url().map(ToOwned::to_owned));
            edge_base_url.map(|edge_base_url| (edge_base_url, session_id.clone()))
        })
    }) else {
        return Ok(None);
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to build async runtime for train-window edge registration")?;
    Ok(Some((runtime, edge_base_url, session_id)))
}

pub(super) fn wait_for_head_provider<B>(
    running: &mut ManagedRunningNativePeer<B>,
    experiment: &burn_p2p::ExperimentHandle,
    initialize_head_on_start: bool,
    restore_head_on_start: bool,
    served_head_id: &mut Option<burn_p2p::HeadId>,
    log_prefix: &str,
    timeout: Duration,
) -> Result<burn_p2p::HeadDescriptor>
where
    B: AutodiffBackend + Clone + 'static,
{
    let deadline = Instant::now() + timeout;
    let started = Instant::now();
    let mut attempts = 0_u64;
    let mut last_error = None;
    loop {
        attempts += 1;
        match sync_or_initialize_latest_head_provider(
            running,
            experiment,
            initialize_head_on_start,
            restore_head_on_start,
            served_head_id,
            HeadProviderSyncMode::DirectoryCurrent,
            log_prefix,
        ) {
            Ok(Some(head)) => return Ok(head),
            Ok(None) => {}
            Err(error) => last_error = Some(error.to_string()),
        }

        if Instant::now() >= deadline {
            let detail = last_error
                .map(|error| format!("; last error: {error}"))
                .unwrap_or_default();
            bail!(
                "no experiment head became available within {:?}; rerun with --initialize-head-on-start true or seed a head first{}",
                timeout,
                detail
            );
        }

        if attempts == 1 || attempts.is_multiple_of(10) {
            let snapshot = running.snapshot();
            let last_snapshot_heads = snapshot
                .last_snapshot
                .as_ref()
                .map(|snapshot| snapshot.head_announcements.len())
                .unwrap_or(0);
            eprintln!(
                "{log_prefix}-head-waiting elapsed_ms={} attempts={} connected_peers={} local_heads={} last_snapshot_peer={} last_snapshot_heads={} node_state={:?} last_error={}",
                started.elapsed().as_millis(),
                attempts,
                snapshot.connected_peers,
                snapshot.control_plane.head_announcements.len(),
                snapshot
                    .last_snapshot_peer_id
                    .as_ref()
                    .map(|peer_id| peer_id.as_str())
                    .unwrap_or("-"),
                last_snapshot_heads,
                snapshot.node_state,
                operator_visible_last_error(snapshot.last_error.as_deref())
                    .as_deref()
                    .unwrap_or("-"),
            );
        }
        thread::sleep(STATUS_POLL_INTERVAL);
    }
}

#[derive(Clone, Copy)]
pub(super) enum HeadProviderSyncMode {
    DirectoryCurrent,
    LatestPromoted,
}

pub(super) fn sync_or_initialize_latest_head_provider<B>(
    running: &mut ManagedRunningNativePeer<B>,
    experiment: &burn_p2p::ExperimentHandle,
    initialize_head_on_start: bool,
    restore_head_on_start: bool,
    served_head_id: &mut Option<burn_p2p::HeadId>,
    sync_mode: HeadProviderSyncMode,
    log_prefix: &str,
) -> Result<Option<burn_p2p::HeadDescriptor>>
where
    B: AutodiffBackend + Clone + 'static,
{
    let restored = if restore_head_on_start {
        eprintln!("{log_prefix}-head-restore-start");
        match running.restore_experiment_head(experiment) {
            Ok(head) => {
                if let Some(head) = head.as_ref() {
                    eprintln!(
                        "{log_prefix}-head-restored id={} global_step={}",
                        head.head_id.as_str(),
                        head.global_step,
                    );
                }
                head
            }
            Err(error) if initialize_head_on_start => {
                eprintln!(
                    "{log_prefix}-head-restore-failed error={error}; continuing with sync/initialize"
                );
                None
            }
            Err(error) => return Err(error),
        }
    } else {
        None
    };

    let synced_result = match sync_mode {
        HeadProviderSyncMode::DirectoryCurrent => running.sync_experiment_head(experiment),
        HeadProviderSyncMode::LatestPromoted => {
            running.sync_latest_promoted_experiment_head(experiment)
        }
    };
    let synced = match synced_result {
        Ok(Some(head)) => {
            eprintln!(
                "{log_prefix}-head-synced id={} global_step={}",
                head.head_id.as_str(),
                head.global_step,
            );
            Some(head)
        }
        Ok(None) => None,
        Err(error) if restored.is_some() => {
            eprintln!(
                "{log_prefix}-head-sync-failed error={error}; keeping restored head candidate"
            );
            None
        }
        Err(error) if initialize_head_on_start => {
            eprintln!(
                "{log_prefix}-head-sync-failed error={error}; falling back to local genesis initialization if no restored head is available"
            );
            None
        }
        Err(error) => return Err(error),
    };

    let (head, source) = match select_latest_head_candidate(restored, synced) {
        Some(candidate) => candidate,
        None if initialize_head_on_start => {
            eprintln!("{log_prefix}-initializing local genesis head");
            let head = running.initialize_local_head(experiment)?;
            eprintln!(
                "{log_prefix}-initialized genesis head id={} global_step={}",
                head.head_id.as_str(),
                head.global_step,
            );
            (head, "initialized")
        }
        None => return Ok(None),
    };

    if source == "restored" && !running.adopt_known_head_if_present(experiment, &head)? {
        bail!(
            "{log_prefix}-head-restored id={} artifact={} could not be re-adopted",
            head.head_id.as_str(),
            head.artifact_id.as_str()
        );
    }
    eprintln!(
        "{log_prefix}-head-selected source={} id={} global_step={}",
        source,
        head.head_id.as_str(),
        head.global_step,
    );
    serve_head_provider(running, experiment, head, served_head_id, log_prefix).map(Some)
}

pub(super) fn select_latest_head_candidate(
    restored: Option<burn_p2p::HeadDescriptor>,
    synced: Option<burn_p2p::HeadDescriptor>,
) -> Option<(burn_p2p::HeadDescriptor, &'static str)> {
    match (restored, synced) {
        (Some(restored), Some(synced)) if restored.global_step > synced.global_step => {
            Some((restored, "restored"))
        }
        (Some(_), Some(synced)) => Some((synced, "synced")),
        (Some(restored), None) => Some((restored, "restored")),
        (None, Some(synced)) => Some((synced, "synced")),
        (None, None) => None,
    }
}

pub(super) fn serve_head_provider<B>(
    running: &mut ManagedRunningNativePeer<B>,
    experiment: &burn_p2p::ExperimentHandle,
    head: burn_p2p::HeadDescriptor,
    served_head_id: &mut Option<burn_p2p::HeadId>,
    log_prefix: &str,
) -> Result<burn_p2p::HeadDescriptor>
where
    B: AutodiffBackend + Clone + 'static,
{
    // Re-announce the locally materialized head on every sync pass so late
    // browser peers can always discover at least one live provider.
    running.publish_head_provider(experiment, &head)?;

    if served_head_id.as_ref() != Some(&head.head_id) {
        eprintln!(
            "{log_prefix}-serving head id={} global_step={}",
            head.head_id.as_str(),
            head.global_step,
        );
        *served_head_id = Some(head.head_id.clone());
    }

    Ok(head)
}

pub(super) fn directory_entry_promotes_with_diffusion(entry: &ExperimentDirectoryEntry) -> bool {
    entry.merge_topology_policy().is_some_and(|policy| {
        matches!(
            policy.promotion_policy.mode,
            HeadPromotionMode::DiffusionSteadyState
        )
    })
}

pub(super) fn diffusion_settlement_report(
    snapshot: &ControlPlaneSnapshot,
    enabled: bool,
    passes_requested: u32,
    passes_completed: u32,
    served_after_publish_secs: u64,
) -> DiffusionSettlementReport {
    DiffusionSettlementReport {
        enabled,
        passes_requested,
        passes_completed,
        served_after_publish_secs,
        merge_windows: snapshot.merge_window_announcements.len(),
        updates: snapshot.update_announcements.len(),
        attestations: snapshot.trainer_promotion_attestation_announcements.len(),
        certificates: snapshot.diffusion_promotion_certificate_announcements.len(),
        merges: snapshot.merge_announcements.len(),
    }
}

pub(super) fn register_live_head_with_edge_options(
    runtime: &tokio::runtime::Runtime,
    edge_base_url: &str,
    session_id: &str,
    directory_template: Option<&ExperimentDirectoryEntry>,
    announcement: &HeadAnnouncement,
) -> Result<()> {
    let source_provider_peer_id = announcement
        .provider_peer_id
        .as_ref()
        .ok_or_else(|| anyhow!("live head registration requires a provider peer id"))?
        .clone();
    let edge_announcement =
        mirror_head_artifact_with_edge(runtime, edge_base_url, session_id, announcement)?;
    register_edge_head_and_directory(
        runtime,
        edge_base_url,
        session_id,
        directory_template,
        edge_announcement,
        Some(&source_provider_peer_id),
    )
}

pub(super) fn mirror_head_artifact_with_edge(
    runtime: &tokio::runtime::Runtime,
    edge_base_url: &str,
    session_id: &str,
    announcement: &HeadAnnouncement,
) -> Result<HeadAnnouncement> {
    let provider_peer_id = announcement
        .provider_peer_id
        .as_ref()
        .ok_or_else(|| anyhow!("artifact mirror requires a provider peer id"))?;
    let mirror = runtime
        .block_on(mirror_peer_artifact(
            edge_base_url,
            session_id,
            burn_p2p_publish::PeerArtifactMirrorRequest {
                artifact_id: announcement.head.artifact_id.clone(),
                provider_peer_ids: vec![provider_peer_id.clone()],
                timeout_ms: Some(EDGE_HEAD_ARTIFACT_MIRROR_TIMEOUT_MILLIS),
            },
        ))
        .with_context(|| {
            format!(
                "failed to mirror head artifact {} from provider {} before live head registration",
                announcement.head.artifact_id.as_str(),
                provider_peer_id.as_str()
            )
        })?;
    let mirrored_provider_peer_id = mirror.mirrored_provider_peer_id.clone().ok_or_else(|| {
        anyhow!(
            "edge mirror response for artifact {} did not include a mirrored provider peer id",
            announcement.head.artifact_id.as_str()
        )
    })?;
    eprintln!(
        "head-mirror-edge-artifact-mirrored artifact_id={} source_provider={} edge_provider={} bytes={} chunks={}",
        mirror.artifact_id.as_str(),
        mirror.mirrored_from.as_str(),
        mirrored_provider_peer_id.as_str(),
        mirror.bytes_len,
        mirror.chunk_count,
    );

    Ok(mirrored_edge_head_announcement(
        announcement,
        mirrored_provider_peer_id,
    ))
}

pub(super) fn register_edge_head_and_directory(
    runtime: &tokio::runtime::Runtime,
    edge_base_url: &str,
    session_id: &str,
    directory_template: Option<&ExperimentDirectoryEntry>,
    edge_announcement: HeadAnnouncement,
    source_provider_peer_id: Option<&PeerId>,
) -> Result<()> {
    let _ = runtime.block_on(register_live_head(
        edge_base_url,
        session_id,
        edge_announcement.clone(),
    ))?;
    eprintln!(
        "head-mirror-edge-head-registered head_id={} provider={} source_provider={}",
        edge_announcement.head.head_id.as_str(),
        edge_announcement
            .provider_peer_id
            .as_ref()
            .map(|peer_id| peer_id.as_str())
            .unwrap_or("-"),
        source_provider_peer_id
            .map(|peer_id| peer_id.as_str())
            .unwrap_or("-"),
    );
    if let Some(directory_template) = directory_template {
        let mut directory_entries =
            runtime.block_on(fetch_signed_directory_entries(edge_base_url, session_id))?;
        if upsert_directory_entry_current_head(
            &mut directory_entries,
            directory_template,
            edge_announcement.head.head_id.clone(),
        ) {
            let _ = runtime.block_on(rollout_directory_entries(
                edge_base_url,
                session_id,
                directory_entries,
            ))?;
            eprintln!(
                "head-mirror-edge-directory-updated head_id={}",
                edge_announcement.head.head_id.as_str(),
            );
        }
    }
    Ok(())
}

pub(super) fn mirrored_edge_head_announcement(
    announcement: &HeadAnnouncement,
    mirrored_provider_peer_id: PeerId,
) -> HeadAnnouncement {
    let mut edge_announcement = announcement.clone();
    edge_announcement.provider_peer_id = Some(mirrored_provider_peer_id);
    edge_announcement
}
