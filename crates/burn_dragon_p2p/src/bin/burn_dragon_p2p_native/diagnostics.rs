//! Swarm probes, deployment diagnostics, and administrative reports.

use super::*;

pub(super) fn probe_swarm(args: ProbeSwarmArgs) -> Result<()> {
    let timeout = Duration::from_secs(args.timeout_secs);
    let started = Instant::now();
    let network_id = NetworkId::new(args.network_id.clone());
    let protocols = ProtocolSet::for_network(&network_id)
        .with_context(|| format!("failed to build protocol set for {}", args.network_id))?;
    let transport_policy =
        RuntimeTransportPolicy::native_for_roles(&PeerRoleSet::default_trainer());
    let mut shell = NativeControlPlaneShell::new(protocols.control, transport_policy)
        .context("failed to initialize native control-plane swarm")?;
    let local_peer_id = shell.local_peer_id().to_string();
    let address = SwarmAddress::new(args.address.clone())
        .with_context(|| format!("invalid swarm address {}", args.address))?;
    if let Some(listen_address) = probe_swarm_listen_address_for_target(address.as_str()) {
        shell
            .listen_on(SwarmAddress::new(listen_address)?)
            .with_context(|| {
                format!(
                    "failed to open required local listener before probing {}",
                    args.address
                )
            })?;
    }
    shell
        .dial(address)
        .with_context(|| format!("failed to enqueue swarm dial to {}", args.address))?;

    let deadline = Instant::now() + timeout;
    let mut connected_peer_id = None;
    let mut events = Vec::new();
    let mut last_error = None;

    while connected_peer_id.is_none() && events.len() < args.max_events {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        let wait_for = deadline.duration_since(now).min(Duration::from_millis(500));
        let Some(event) = shell.wait_event(wait_for) else {
            continue;
        };
        match &event {
            LiveControlPlaneEvent::ConnectionEstablished { peer_id } => {
                connected_peer_id = Some(peer_id.clone());
            }
            LiveControlPlaneEvent::OutgoingConnectionError { message, .. }
            | LiveControlPlaneEvent::IncomingConnectionError { message }
            | LiveControlPlaneEvent::InboundFailure { message, .. }
            | LiveControlPlaneEvent::ResponseSendFailure { message, .. }
            | LiveControlPlaneEvent::RequestFailure { message, .. } => {
                last_error = Some(message.clone());
            }
            _ => {}
        }
        events.push(event);
    }

    let connected = connected_peer_id.is_some();
    let (snapshot, snapshot_error) = if args.fetch_snapshot {
        match connected_peer_id.as_deref() {
            Some(peer_id) => match shell.fetch_snapshot(
                peer_id,
                Duration::from_secs(args.snapshot_timeout_secs.max(1)),
            ) {
                Ok(snapshot) => (Some(probe_swarm_snapshot_summary(&snapshot)), None),
                Err(error) => (None, Some(error.to_string())),
            },
            None => (None, Some("not connected".into())),
        }
    } else {
        (None, None)
    };
    let elapsed_millis = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
    let report = ProbeSwarmReport {
        network_id: args.network_id,
        address: args.address,
        local_peer_id,
        connected,
        connected_peer_id,
        elapsed_millis,
        events,
        snapshot,
        snapshot_error,
        last_error,
    };
    write_output(None, args.output_format, &report)?;
    if !connected {
        bail!(
            "swarm probe did not establish a connection within {:?}",
            timeout
        );
    }
    Ok(())
}

pub(super) fn probe_swarm_listen_address_for_target(address: &str) -> Option<&'static str> {
    if address.split('/').any(|segment| segment == "webrtc-direct") {
        Some("/ip4/0.0.0.0/udp/0/webrtc-direct")
    } else {
        None
    }
}

pub(super) fn probe_swarm_snapshot_summary(
    snapshot: &ControlPlaneSnapshot,
) -> ProbeSwarmSnapshotSummary {
    let heads = snapshot
        .head_announcements
        .iter()
        .map(|announcement| ProbeSwarmHeadSummary {
            provider_peer_id: announcement
                .provider_peer_id
                .as_ref()
                .map(|peer_id| peer_id.as_str().to_owned()),
            study_id: announcement.head.study_id.as_str().to_owned(),
            experiment_id: announcement.head.experiment_id.as_str().to_owned(),
            revision_id: announcement.head.revision_id.as_str().to_owned(),
            head_id: announcement.head.head_id.as_str().to_owned(),
            parent_head_id: announcement
                .head
                .parent_head_id
                .as_ref()
                .map(|head_id| head_id.as_str().to_owned()),
            artifact_id: announcement.head.artifact_id.as_str().to_owned(),
            global_step: announcement.head.global_step,
        })
        .collect();
    let directory_entries = snapshot
        .directory_announcements
        .iter()
        .flat_map(|announcement| announcement.entries.iter())
        .map(|entry| ProbeSwarmDirectoryEntrySummary {
            study_id: entry.study_id.as_str().to_owned(),
            experiment_id: entry.experiment_id.as_str().to_owned(),
            revision_id: entry.current_revision_id.as_str().to_owned(),
            current_head_id: entry
                .current_head_id
                .as_ref()
                .map(|head_id| head_id.as_str().to_owned()),
        })
        .collect();
    ProbeSwarmSnapshotSummary {
        head_announcements: snapshot.head_announcements.len(),
        directory_announcements: snapshot.directory_announcements.len(),
        peer_directory_announcements: snapshot.peer_directory_announcements.len(),
        merge_announcements: snapshot.merge_announcements.len(),
        merge_window_announcements: snapshot.merge_window_announcements.len(),
        update_announcements: snapshot.update_announcements.len(),
        aggregate_proposal_announcements: snapshot.aggregate_proposal_announcements.len(),
        reduction_certificate_announcements: snapshot.reduction_certificate_announcements.len(),
        validation_quorum_announcements: snapshot.validation_quorum_announcements.len(),
        trainer_promotion_attestation_announcements: snapshot
            .trainer_promotion_attestation_announcements
            .len(),
        diffusion_promotion_certificate_announcements: snapshot
            .diffusion_promotion_certificate_announcements
            .len(),
        heads,
        directory_entries,
    }
}

pub(super) fn build_profile(args: BuildProfileArgs) -> Result<()> {
    let config = load_training_config(&args.training_config_paths)?;
    let profile = build_profile_from_local_config(
        &config,
        args.experiment_kind.into_config(),
        args.revision_id.as_deref(),
        args.browser_climbmix_manifest_url.as_deref(),
    )?;
    write_output(args.output.as_deref(), args.output_format, &profile)
}

pub(super) fn resolve_config(args: ResolveConfigArgs) -> Result<()> {
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        args.edge_url,
        args.seed_node_urls,
        Some(args.capability_policy),
    )?;
    write_output(None, args.output_format, &config)
}

pub(super) fn assess_capability(args: AssessCapabilityArgs) -> Result<()> {
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        None,
        Vec::new(),
        Some(args.capability_policy),
    )?;
    let report = CapabilityAssessmentReport {
        config_path: args.config.clone(),
        experiment_kind: args.experiment_kind.into_config(),
        backend: args.backend.as_label().into(),
        assessment: assess_native_peer(
            &config,
            args.experiment_kind.into_config(),
            args.backend.as_label(),
        )?,
    };
    write_output(None, args.output_format, &report)
}

pub(super) fn deployment_diagnostics(args: DeploymentDiagnosticsArgs) -> Result<()> {
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        args.edge_url,
        args.seed_node_urls,
        None,
    )?;
    let report = collect_deployment_diagnostics(
        &config,
        args.experiment_kind.into_config(),
        args.backend.as_label(),
        DeploymentDiagnosticsOptions {
            check_metrics_catchup: args.check_metrics_catchup,
            check_auth_authorize: args.check_auth_authorize,
            check_artifact_head_view: args.check_artifact_head_view,
            require_head_published: args.require_head_published,
            require_head_advanced: args.require_head_advanced,
            require_directory_entry_published: args.require_directory_entry_published,
            require_revision_contract: args.require_revision_contract,
            require_metrics_catchup: args.require_metrics_catchup,
            require_auth_authorize: args.require_auth_authorize,
            require_artifact_head_view: args.require_artifact_head_view,
        },
    );
    write_output(args.output.as_deref(), args.output_format, &report)?;
    if args.assert_ready {
        assert_deployment_ready(&report)?;
    }
    Ok(())
}

pub(super) fn doctor(args: DoctorArgs) -> Result<()> {
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        args.edge_url,
        args.seed_node_urls,
        Some(args.capability_policy),
    )?;
    fs::create_dir_all(&config.storage_root).with_context(|| {
        format!(
            "failed to create native storage root {}",
            config.storage_root.display()
        )
    })?;
    let experiment_kind = args.experiment_kind.into_config();
    let backend = args.backend.as_label().to_owned();
    let capability = assess_native_peer(&config, experiment_kind, &backend)?;
    let mut checks = Vec::new();
    checks.push(DoctorCheck {
        name: "storage_root".into(),
        ok: true,
        message: config.storage_root.display().to_string(),
    });
    checks.push(DoctorCheck {
        name: "capability".into(),
        ok: capability.target_decision.can_train,
        message: capability
            .target_decision
            .downgrade_reason
            .clone()
            .unwrap_or_else(|| "native trainer capability accepted".into()),
    });
    let edge_base_url = config.effective_edge_base_url().map(ToOwned::to_owned);
    let mut edge_snapshot = None;
    if let Some(edge_url) = edge_base_url.as_deref() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("failed to build async runtime for doctor edge snapshot")?;
        match runtime.block_on(fetch_edge_snapshot(edge_url)) {
            Ok(snapshot) => {
                checks.push(DoctorCheck {
                    name: "edge_snapshot".into(),
                    ok: true,
                    message: format!(
                        "{} entries from {}",
                        snapshot.directory.entries.len(),
                        edge_url
                    ),
                });
                edge_snapshot = Some(DoctorEdgeSnapshotReport {
                    network_id: snapshot.network_id.as_str().to_owned(),
                    protocol_major: snapshot.protocol_major,
                    minimum_client_version: snapshot.minimum_client_version.to_string(),
                    auth_enabled: snapshot.auth_enabled,
                    directory_entries: snapshot.directory.entries.len(),
                    browser_mode: format!("{:?}", snapshot.browser_mode),
                });
            }
            Err(error) => checks.push(DoctorCheck {
                name: "edge_snapshot".into(),
                ok: false,
                message: error.to_string(),
            }),
        }
    } else {
        checks.push(DoctorCheck {
            name: "edge_snapshot".into(),
            ok: false,
            message: "no edge_base_url configured".into(),
        });
    }
    checks.push(DoctorCheck {
        name: "seed_nodes".into(),
        ok: !config.effective_seed_node_urls().is_empty(),
        message: format!(
            "{} configured seed(s)",
            config.effective_seed_node_urls().len()
        ),
    });
    let ready = checks.iter().all(|check| check.ok);
    let report = DoctorReport {
        config_path: args.config,
        experiment_kind,
        backend,
        storage_root: config.storage_root.clone(),
        edge_base_url,
        seed_node_count: config.effective_seed_node_urls().len(),
        install_features: args.backend.default_enabled_features_label().into(),
        capability,
        edge_snapshot,
        checks,
        ready,
    };
    write_output(args.output.as_deref(), args.output_format, &report)?;
    if args.assert_ready && !ready {
        bail!("native peer doctor checks did not pass");
    }
    Ok(())
}

pub(super) fn admin_export_directory(args: AdminExportDirectoryArgs) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to build async runtime for directory export")?;
    let entries = runtime.block_on(fetch_directory_entries(&args.edge_url))?;
    let report = entries
        .into_iter()
        .map(|entry| AdminDirectoryEntryReport {
            dragon_profile: DragonExperimentProfile::from_entry_metadata(&entry)
                .ok()
                .flatten(),
            entry,
        })
        .collect::<Vec<_>>();
    write_output(None, args.output_format, &report)
}

pub(super) fn admin_rollout_profile(args: AdminRolloutProfileArgs) -> Result<()> {
    let requested_edge_url = args.edge_url.clone();
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        requested_edge_url.clone(),
        Vec::new(),
        None,
    )?;
    let auth_bundle = resolve_or_login_native_auth_bundle(
        &config,
        args.experiment_kind.into_config(),
        args.backend,
        NativeAuthResolutionOptions {
            auth_bundle_path: Some(args.auth_bundle.as_path()),
            auth_bundle_format: args.auth_bundle_format,
            principal_hint: None,
            session_ttl_secs: DEFAULT_SESSION_TTL_SECS,
            callback_timeout_secs: DEFAULT_AUTH_CALLBACK_TIMEOUT_SECS,
        },
    )?;
    let edge_base_url = requested_edge_url
        .or_else(|| auth_bundle.edge_base_url.clone())
        .or_else(|| config.effective_edge_base_url().map(ToOwned::to_owned))
        .ok_or_else(|| anyhow!("no edge base URL configured for admin rollout"))?;
    let session_id = auth_bundle
        .session_id
        .clone()
        .ok_or_else(|| anyhow!("auth bundle is missing a session_id for admin rollout"))?;

    let local_config = config.clone().with_network_overrides(None, None);
    let manifests = prepared_manifests(
        &local_config,
        args.experiment_kind.into_config(),
        args.backend,
    )?;
    let mut replacement = manifests
        .experiment_directory
        .first()
        .cloned()
        .ok_or_else(|| anyhow!("manifest bundle is missing an experiment directory entry"))?;

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to build async runtime for admin rollout")?;
    let mut directory_entries =
        runtime.block_on(fetch_signed_directory_entries(&edge_base_url, &session_id))?;
    let revision_contract_changed = directory_entries
        .iter()
        .find(|entry| {
            entry.study_id == replacement.study_id
                && entry.experiment_id == replacement.experiment_id
                && entry.current_revision_id == replacement.current_revision_id
        })
        .is_some_and(|existing| !directory_revision_contract_matches(existing, &replacement));
    let preserved_current_head_id =
        if args.reset_current_head_to_visible_root || revision_contract_changed {
            None
        } else {
            preserve_directory_entry_current_head(&directory_entries, &mut replacement)
        };
    let mut recovered_current_head_id = None;
    let mut reset_current_head_id = None;
    if !revision_contract_changed
        && (args.reset_current_head_to_visible_root
            || (replacement.current_head_id.is_none()
                && args.recover_current_head_from_visible_root))
    {
        let snapshot = runtime.block_on(fetch_edge_snapshot(&edge_base_url))?;
        let recovered =
            recover_directory_current_head_from_visible_roots(&replacement, &snapshot.heads);
        if args.reset_current_head_to_visible_root && recovered.is_none() {
            bail!(
                "cannot reset current head for experiment={} revision={} because no visible root head was available",
                replacement.experiment_id.as_str(),
                replacement.current_revision_id.as_str()
            );
        }
        if let Some(head_id) = recovered.as_ref() {
            replacement.current_head_id = Some(head_id.clone());
        }
        if args.reset_current_head_to_visible_root {
            reset_current_head_id = recovered;
        } else {
            recovered_current_head_id = recovered;
        }
    }
    if revision_contract_changed {
        replacement.current_head_id = None;
    }
    upsert_directory_entry(&mut directory_entries, replacement.clone());
    let result = runtime.block_on(rollout_directory_entries(
        &edge_base_url,
        &session_id,
        directory_entries.clone(),
    ))?;

    write_output(
        None,
        args.output_format,
        &AdminRolloutReport {
            edge_base_url,
            experiment_id: replacement.experiment_id.as_str().to_owned(),
            revision_id: replacement.current_revision_id.as_str().to_owned(),
            current_head_id: replacement
                .current_head_id
                .as_ref()
                .map(|head_id| head_id.as_str().to_owned()),
            preserved_current_head_id: preserved_current_head_id
                .as_ref()
                .map(|head_id| head_id.as_str().to_owned()),
            recovered_current_head_id: recovered_current_head_id
                .as_ref()
                .map(|head_id| head_id.as_str().to_owned()),
            reset_current_head_id: reset_current_head_id
                .as_ref()
                .map(|head_id| head_id.as_str().to_owned()),
            revision_contract_changed,
            directory_entries: directory_entries.len(),
            result,
        },
    )
}
