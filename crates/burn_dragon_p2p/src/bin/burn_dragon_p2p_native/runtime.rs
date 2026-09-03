//! Native peer preparation, capability resolution, and runtime startup.

use super::*;

pub(super) fn train_window_once(args: TrainWindowOnceArgs) -> Result<()> {
    let mut config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        args.edge_url,
        args.seed_node_urls,
        Some(args.capability_policy),
    )?;
    args.training_overrides.apply_to(&mut config);
    ensure_training_backend_runtime_accessible(args.backend)?;
    let auth_bundle = resolve_or_login_native_auth_bundle(
        &config,
        args.experiment_kind.into_config(),
        args.backend,
        NativeAuthResolutionOptions {
            auth_bundle_path: args.auth_bundle.as_deref(),
            auth_bundle_format: args.auth_bundle_format,
            principal_hint: None,
            session_ttl_secs: DEFAULT_SESSION_TTL_SECS,
            callback_timeout_secs: DEFAULT_AUTH_CALLBACK_TIMEOUT_SECS,
        },
    )?;
    let run_options = TrainWindowOnceRunOptions {
        initialize_head_on_start: args.initialize_head_on_start,
        restore_head_on_start: args.restore_head_on_start,
        output: args.output.as_deref(),
        output_format: args.output_format,
        require_head_advanced: args.require_head_advanced,
        head_sync_timeout_secs: args.head_sync_timeout_secs,
        settle_diffusion: args.settle_diffusion,
        diffusion_settle_passes: args.diffusion_settle_passes,
        serve_after_publish_secs: args.serve_after_publish_secs,
        mirror_live_head_to_edge: args.mirror_live_head_to_edge,
    };

    with_prepared_native_peer!(
        args.experiment_kind.into_config(),
        args.backend,
        &config,
        Some(&auth_bundle),
        |prepared| {
            run_prepared_train_window_once(
                prepared,
                &config,
                Some(&auth_bundle),
                args.backend,
                run_options,
            )
        }
    )
}

pub(super) fn native_target_artifact_id(backend: BackendArg) -> &'static str {
    match backend {
        BackendArg::Cpu => "native-cpu",
        BackendArg::Wgpu => "native-wgpu",
        BackendArg::Cuda => "native-cuda",
        BackendArg::Rocm => "native-rocm",
    }
}

pub(super) fn resolve_native_target_artifact_hash(
    snapshot: &burn_p2p::BrowserEdgeSnapshot,
    override_hash: Option<String>,
) -> Result<ContentId> {
    if let Some(target_artifact_hash) = override_hash.as_deref().map(str::trim)
        && !target_artifact_hash.is_empty()
    {
        return Ok(ContentId::new(target_artifact_hash));
    }

    let mut allowed = snapshot
        .allowed_target_artifact_hashes
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    if allowed.is_empty()
        && let Some(trust_bundle) = snapshot.trust_bundle.as_ref()
    {
        allowed.extend(trust_bundle.allowed_target_artifact_hashes.iter().cloned());
    }
    if allowed.is_empty() {
        bail!(
            "edge snapshot is missing allowed target artifact hashes; pass --target-artifact-hash explicitly"
        )
    }
    if allowed.len() == 1 {
        return Ok(allowed.remove(0));
    }

    let nativeish = allowed
        .into_iter()
        .filter(|hash| {
            let label = hash.as_str().to_ascii_lowercase();
            label.contains("native") || !label.contains("browser")
        })
        .collect::<Vec<_>>();
    if nativeish.len() == 1 {
        return Ok(nativeish.into_iter().next().expect("nativeish hash exists"));
    }

    bail!(
        "edge snapshot advertises multiple target artifact hashes; pass --target-artifact-hash explicitly"
    )
}

pub(super) fn native_release_manifest_for_snapshot(
    config: &DragonNativePeerConfig,
    snapshot: &burn_p2p::BrowserEdgeSnapshot,
    backend: BackendArg,
    target_artifact_hash: Option<String>,
) -> Result<ClientReleaseManifest> {
    let trust_bundle = snapshot
        .trust_bundle
        .as_ref()
        .ok_or_else(|| anyhow!("edge snapshot is missing a trust bundle"))?;
    let release_train_hash = snapshot
        .required_release_train_hash
        .clone()
        .unwrap_or_else(|| trust_bundle.required_release_train_hash.clone());
    let release_manifest = ClientReleaseManifest {
        project_family_id: trust_bundle.project_family_id.clone(),
        release_train_hash,
        target_artifact_id: native_target_artifact_id(backend).into(),
        target_artifact_hash: resolve_native_target_artifact_hash(snapshot, target_artifact_hash)?,
        target_platform: ClientPlatform::Native,
        app_semver: config.app_semver.clone(),
        git_commit: config
            .git_commit
            .clone()
            .or_else(build_info::embedded_git_commit_owned)
            .unwrap_or_else(|| "unknown".into()),
        cargo_lock_hash: ContentId::new("dragon-native-auth-lock"),
        burn_version_string: "0.21.0".into(),
        enabled_features_hash: ContentId::new(
            config
                .enabled_features_label
                .clone()
                .unwrap_or_else(|| backend.default_enabled_features_label().into()),
        ),
        protocol_major: snapshot.protocol_major,
        supported_workloads: Vec::new(),
        built_at: chrono::Utc::now(),
    };
    release_manifest
        .validate_for_edge_snapshot(snapshot)
        .map_err(|error| {
            anyhow!("native release manifest is incompatible with edge snapshot: {error}")
        })?;
    Ok(release_manifest)
}

pub(super) fn run_peer(args: RunPeerArgs) -> Result<()> {
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        args.edge_url,
        args.seed_node_urls,
        Some(args.capability_policy),
    )?;
    ensure_training_backend_runtime_accessible(args.backend)?;
    let auth_bundle = Some(resolve_or_login_native_auth_bundle(
        &config,
        args.experiment_kind.into_config(),
        args.backend,
        NativeAuthResolutionOptions {
            auth_bundle_path: args.auth_bundle.as_deref(),
            auth_bundle_format: args.auth_bundle_format,
            principal_hint: None,
            session_ttl_secs: DEFAULT_SESSION_TTL_SECS,
            callback_timeout_secs: DEFAULT_AUTH_CALLBACK_TIMEOUT_SECS,
        },
    )?);

    with_prepared_native_peer!(
        args.experiment_kind.into_config(),
        args.backend,
        &config,
        auth_bundle.as_ref(),
        |prepared| run_prepared_peer(
            prepared,
            &config,
            NativePeerServiceOptions {
                backend: args.backend,
                status_interval_secs: args.status_interval_secs,
                initialize_head_on_start: args.initialize_head_on_start,
                restore_head_on_start: args.restore_head_on_start,
                head_sync_interval_secs: args.head_sync_interval_secs,
                trainer_daemon: None,
            },
        )
    )
}

pub(super) fn run_trainer_daemon(args: RunTrainerDaemonArgs) -> Result<()> {
    let mut config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        args.edge_url,
        args.seed_node_urls,
        Some(args.capability_policy),
    )?;
    args.training_overrides.apply_to(&mut config);
    config.target = Some(DragonNativeTarget::Trainer);
    ensure_training_backend_runtime_accessible(args.backend)?;
    let auth_bundle = Some(resolve_or_login_native_auth_bundle(
        &config,
        args.experiment_kind.into_config(),
        args.backend,
        NativeAuthResolutionOptions {
            auth_bundle_path: args.auth_bundle.as_deref(),
            auth_bundle_format: args.auth_bundle_format,
            principal_hint: None,
            session_ttl_secs: DEFAULT_SESSION_TTL_SECS,
            callback_timeout_secs: DEFAULT_AUTH_CALLBACK_TIMEOUT_SECS,
        },
    )?);
    let policy = TrainerDaemonPolicy {
        minimum_step_interval: Duration::from_secs(args.minimum_step_interval_secs),
        failure_backoff: Duration::from_secs(args.failure_backoff_secs.max(1)),
        max_consecutive_failures: args.max_consecutive_failures.max(1),
        max_protocol_steps: (args.max_protocol_steps > 0).then_some(args.max_protocol_steps),
    };

    with_prepared_native_peer!(
        args.experiment_kind.into_config(),
        args.backend,
        &config,
        auth_bundle.as_ref(),
        |prepared| run_prepared_peer(
            prepared,
            &config,
            NativePeerServiceOptions {
                backend: args.backend,
                status_interval_secs: args.status_interval_secs,
                initialize_head_on_start: args.initialize_head_on_start,
                restore_head_on_start: args.restore_head_on_start,
                head_sync_interval_secs: args.head_sync_interval_secs,
                trainer_daemon: Some(policy),
            },
        )
    )
}

pub(super) fn run_head_mirror(args: RunHeadMirrorArgs) -> Result<()> {
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        args.edge_url,
        args.seed_node_urls,
        Some(args.capability_policy),
    )?;
    ensure_training_backend_runtime_accessible(args.backend)?;
    let auth_bundle = Some(resolve_or_login_native_auth_bundle(
        &config,
        args.experiment_kind.into_config(),
        args.backend,
        NativeAuthResolutionOptions {
            auth_bundle_path: args.auth_bundle.as_deref(),
            auth_bundle_format: args.auth_bundle_format,
            principal_hint: None,
            session_ttl_secs: DEFAULT_SESSION_TTL_SECS,
            callback_timeout_secs: DEFAULT_AUTH_CALLBACK_TIMEOUT_SECS,
        },
    )?);

    with_prepared_native_peer!(
        args.experiment_kind.into_config(),
        args.backend,
        &config,
        auth_bundle.as_ref(),
        |prepared| run_prepared_head_mirror(
            prepared,
            &config,
            auth_bundle.as_ref(),
            args.backend,
            args.status_interval_secs,
            args.head_sync_interval_secs,
            args.initialize_head_on_start,
            args.restore_head_on_start,
        )
    )
}

pub(super) fn run_validator_daemon(args: RunValidatorDaemonArgs) -> Result<()> {
    ensure_validator_read_only(args.initialize_head_on_start)?;
    let mut config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        args.edge_url,
        args.seed_node_urls,
        Some(args.capability_policy),
    )?;
    args.training_overrides.apply_to(&mut config);
    config.target = Some(DragonNativeTarget::Validator);
    ensure_training_backend_runtime_accessible(args.backend)?;
    let auth_bundle = Some(resolve_or_login_native_auth_bundle(
        &config,
        args.experiment_kind.into_config(),
        args.backend,
        NativeAuthResolutionOptions {
            auth_bundle_path: args.auth_bundle.as_deref(),
            auth_bundle_format: args.auth_bundle_format,
            principal_hint: None,
            session_ttl_secs: DEFAULT_SESSION_TTL_SECS,
            callback_timeout_secs: DEFAULT_AUTH_CALLBACK_TIMEOUT_SECS,
        },
    )?);

    with_prepared_native_peer!(
        args.experiment_kind.into_config(),
        args.backend,
        &config,
        auth_bundle.as_ref(),
        |prepared| run_prepared_validator_daemon(
            prepared,
            &config,
            args.backend,
            args.status_interval_secs,
            args.validation_interval_millis,
            false,
            args.restore_head_on_start,
        )
    )
}

pub(super) fn ensure_validator_read_only(initialize_head_on_start: bool) -> Result<()> {
    if initialize_head_on_start {
        bail!(
            "validator daemons are read-only and cannot initialize model heads; start a trainer or head mirror to bootstrap the revision"
        );
    }
    Ok(())
}

pub(super) fn mark_runtime_failure(args: MarkRuntimeFailureArgs) -> Result<()> {
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        None,
        Vec::new(),
        Some(args.capability_policy),
    )?;
    let assessment = assess_native_peer(
        &config,
        args.experiment_kind.into_config(),
        args.backend.as_label(),
    )?;
    let record = persist_native_downgrade(
        NativeDowngradeScope {
            storage_root: &config.storage_root,
            experiment_kind: args.experiment_kind.into_config(),
            backend_label: args.backend.as_label(),
            model_config: &assessment.model_config,
            batch_size: assessment.batch_size,
            block_size: assessment.block_size,
        },
        NativeDowngradeObservation {
            footprint: &assessment.footprint,
            trainer_budget_bytes: assessment.target_decision.trainer_memory_budget_bytes,
            downgrade_to: "trainer",
            reason: &args.reason,
            source: &args.source,
        },
    )?;
    write_output(None, OutputFormat::Json, &record)
}

pub(super) fn clear_downgrade(args: ClearDowngradeArgs) -> Result<()> {
    let config = resolved_config(
        args.config.as_deref(),
        args.config_format,
        None,
        Vec::new(),
        None,
    )?;
    let assessment = assess_native_peer(
        &config,
        args.experiment_kind.into_config(),
        args.backend.as_label(),
    )?;
    clear_native_downgrade(NativeDowngradeScope {
        storage_root: &config.storage_root,
        experiment_kind: args.experiment_kind.into_config(),
        backend_label: args.backend.as_label(),
        model_config: &assessment.model_config,
        batch_size: assessment.batch_size,
        block_size: assessment.block_size,
    })?;
    Ok(())
}

pub(super) fn resolved_config(
    path: Option<&Path>,
    format: ConfigFormat,
    edge_url: Option<String>,
    seed_node_urls: Vec<String>,
    capability_policy: Option<CapabilityPolicyArgs>,
) -> Result<DragonNativePeerConfig> {
    let mut config = if let Some(path) = path {
        load_native_config(path, format)?
    } else {
        default_mainnet_native_config()
    };
    config = config.with_network_overrides(
        edge_url,
        (!seed_node_urls.is_empty()).then_some(seed_node_urls),
    );
    if let Some(capability_policy) = capability_policy {
        config.capability_policy = capability_policy.apply_to(config.capability_policy.clone());
    }
    let _ = config.effective_bootstrap_peers()?;
    Ok(config)
}

pub(super) fn default_mainnet_storage_root() -> PathBuf {
    if let Some(root) = env::var_os(NATIVE_STORAGE_ROOT_ENV) {
        return PathBuf::from(root);
    }
    if let Some(root) = env::var_os("XDG_DATA_HOME") {
        return PathBuf::from(root)
            .join("burn_dragon_p2p")
            .join("mainnet-native");
    }
    if let Some(home) = env::var_os("HOME") {
        return PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("burn_dragon_p2p")
            .join("mainnet-native");
    }
    PathBuf::from("runs/p2p/mainnet-native")
}

pub(super) fn default_mainnet_native_config() -> DragonNativePeerConfig {
    DragonNativePeerConfig {
        training_overrides: Default::default(),
        training_config_paths: Vec::new(),
        storage_root: default_mainnet_storage_root(),
        network: DragonPeerNetworkConfig::default()
            .with_edge_base_url(Some(DEFAULT_MAINNET_EDGE_BASE_URL.to_owned()))
            .with_seed_node_urls(Some(
                DEFAULT_MAINNET_SEED_NODE_URLS
                    .iter()
                    .map(|seed| (*seed).to_owned())
                    .collect(),
            )),
        target: Some(DragonNativeTarget::Trainer),
        identity: burn_p2p::IdentityConfig::Persistent,
        bootstrap_peers: Vec::new(),
        manifest: DragonManifestSeed {
            project_family_id: DEFAULT_MAINNET_PROJECT_FAMILY_ID.into(),
            network_id: DEFAULT_MAINNET_NETWORK_ID.into(),
            study_id: DEFAULT_MAINNET_STUDY_ID.into(),
            experiment_id: DEFAULT_MAINNET_EXPERIMENT_ID.into(),
            revision_id: DEFAULT_MAINNET_REVISION_ID.into(),
            display_name: "burn_dragon mainnet NCA pre-pre-training".into(),
            description: "burn_dragon mainnet native trainer".into(),
            ..DragonManifestSeed::default()
        },
        app_semver: semver::Version::parse(env!("CARGO_PKG_VERSION"))
            .expect("valid burn_dragon version"),
        git_commit: build_info::embedded_git_commit_owned(),
        enabled_features_label: None,
        auth: None,
        capability_policy: DragonCapabilityPolicy::default(),
        shard_export: None,
        existing_shard_dataset: None,
    }
}

pub(super) fn prepared_manifests(
    config: &DragonNativePeerConfig,
    experiment_kind: DragonExperimentKind,
    backend: BackendArg,
) -> Result<DragonManifestBundle> {
    let placeholder_auth = DragonNativeAuthBundle {
        auth_config: AuthConfig::new(),
        trust_bundle_endpoint: "https://edge.invalid/trust-bundle".into(),
        edge_base_url: None,
        session_id: None,
        principal_id: None,
        enrollment: None,
        session: None,
        certificate_not_after: None,
    };
    with_prepared_native_peer!(
        experiment_kind,
        backend,
        config,
        Some(&placeholder_auth),
        |prepared| Ok(prepared.manifests)
    )
}

pub(super) fn requested_scopes_for_config(
    config: &DragonNativePeerConfig,
) -> BTreeSet<ExperimentScope> {
    let experiment_id = ExperimentId::new(config.manifest.experiment_id.clone());
    match config.target_or_default() {
        DragonNativeTarget::Validator => managed_validator_scopes(&experiment_id),
        DragonNativeTarget::Auto | DragonNativeTarget::Trainer | DragonNativeTarget::Reducer => {
            standard_experiment_scopes(&experiment_id)
        }
    }
}

pub(super) fn standard_experiment_scopes(
    experiment_id: &ExperimentId,
) -> BTreeSet<ExperimentScope> {
    BTreeSet::from([
        ExperimentScope::Connect,
        ExperimentScope::Discover,
        ExperimentScope::Train {
            experiment_id: experiment_id.clone(),
        },
        ExperimentScope::Archive {
            experiment_id: experiment_id.clone(),
        },
    ])
}

pub(super) fn contains_native_trainer_role(roles: &PeerRoleSet) -> bool {
    roles.contains(&PeerRole::TrainerCpu) || roles.contains(&PeerRole::TrainerGpu)
}

pub(super) fn trainer_daemon_step_eligible(
    roles: &PeerRoleSet,
    connected_peers: usize,
    canonical_head_ready: bool,
    awaiting_canonical_promotion: bool,
) -> bool {
    connected_peers > 0
        && canonical_head_ready
        && !awaiting_canonical_promotion
        && contains_native_trainer_role(roles)
}

pub(super) fn managed_trainer_scopes(experiment_id: &ExperimentId) -> BTreeSet<ExperimentScope> {
    standard_experiment_scopes(experiment_id)
}

pub(super) fn managed_validator_scopes(experiment_id: &ExperimentId) -> BTreeSet<ExperimentScope> {
    BTreeSet::from([
        ExperimentScope::Connect,
        ExperimentScope::Discover,
        ExperimentScope::Validate {
            experiment_id: experiment_id.clone(),
        },
        ExperimentScope::Archive {
            experiment_id: experiment_id.clone(),
        },
    ])
}

pub(super) fn ensure_training_backend_runtime_accessible(backend: BackendArg) -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        match backend {
            BackendArg::Cuda => {
                if !Path::new("/dev/nvidiactl").exists() || !Path::new("/dev/nvidia0").exists() {
                    bail!(
                        "cuda backend requested but NVIDIA device nodes are not visible; expected /dev/nvidiactl and /dev/nvidia0. Run on a CUDA host/container with NVIDIA driver devices mounted, or use `login --backend cuda` separately to refresh auth without starting training."
                    );
                }
            }
            BackendArg::Rocm => {
                if !Path::new("/dev/kfd").exists() || !Path::new("/dev/dri").exists() {
                    bail!(
                        "rocm backend requested but ROCm device nodes are not visible; expected /dev/kfd and /dev/dri. Run on a ROCm host/container with AMD GPU devices mounted, or use `login --backend rocm` separately to refresh auth without starting training."
                    );
                }
            }
            BackendArg::Cpu | BackendArg::Wgpu => {}
        }
    }
    Ok(())
}

pub(super) fn run_prepared_peer<B>(
    prepared: PreparedNativePeer<B>,
    config: &DragonNativePeerConfig,
    options: NativePeerServiceOptions,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
{
    let NativePeerServiceOptions {
        backend,
        status_interval_secs,
        initialize_head_on_start,
        restore_head_on_start,
        head_sync_interval_secs,
        trainer_daemon,
    } = options;
    let process_kind = if trainer_daemon.is_some() {
        "trainer daemon"
    } else {
        "native peer"
    };
    eprintln!(
        "starting burn_dragon {process_kind}: experiment={} backend={} target={:?} can_train={} edge={} seeds={} storage_root={}",
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

    let experiment_entry = prepared
        .manifests
        .experiment_directory
        .first()
        .cloned()
        .ok_or_else(|| anyhow!("prepared native peer is missing an experiment"))?;
    let mut running = spawn_prepared_native_peer(prepared)?;
    let mut served_head_id = None;
    let requires_head = initialize_head_on_start
        || restore_head_on_start
        || head_sync_interval_secs > 0
        || trainer_daemon.is_some();
    if requires_head {
        wait_for_runtime_ready(&running, RUNTIME_READY_TIMEOUT)?;
        let experiment = running.mainnet().experiment(
            experiment_entry.study_id.clone(),
            experiment_entry.experiment_id.clone(),
            experiment_entry.current_revision_id.clone(),
        );
        let _ = sync_or_initialize_latest_head_provider(
            &mut running,
            &experiment,
            initialize_head_on_start,
            restore_head_on_start,
            &mut served_head_id,
            HeadProviderSyncMode::DirectoryCurrent,
            "peer",
        )?;
    }
    let shutdown_requested = Arc::new(AtomicBool::new(false));
    let shutdown_requested_for_handler = Arc::clone(&shutdown_requested);
    let control = running.control_handle();
    ctrlc::set_handler(move || {
        if !shutdown_requested_for_handler.swap(true, Ordering::SeqCst) {
            let _ = control.shutdown();
        }
    })
    .context("failed to install ctrl-c handler")?;

    let status_interval = Duration::from_secs(status_interval_secs.max(1));
    let mut last_status = Instant::now()
        .checked_sub(status_interval)
        .unwrap_or_else(Instant::now);
    let experiment = if requires_head {
        Some(running.mainnet().experiment(
            experiment_entry.study_id.clone(),
            experiment_entry.experiment_id.clone(),
            experiment_entry.current_revision_id.clone(),
        ))
    } else {
        None
    };
    let head_sync_interval = Duration::from_secs(head_sync_interval_secs.max(1));
    let mut last_head_sync = Instant::now();
    let mut next_training_at = Instant::now();
    let mut consecutive_training_failures = 0_u32;
    let mut completed_protocol_steps = 0_u64;
    let mut pending_artifact_base_head = None;

    loop {
        let periodic_head_sync = (head_sync_interval_secs > 0 || trainer_daemon.is_some())
            && last_head_sync.elapsed() >= head_sync_interval;
        if (periodic_head_sync || pending_artifact_base_head.is_some())
            && let Some(experiment) = experiment.as_ref()
        {
            let latest = sync_or_initialize_latest_head_provider(
                &mut running,
                experiment,
                false,
                false,
                &mut served_head_id,
                HeadProviderSyncMode::LatestPromoted,
                "peer",
            )?;
            if let (Some(base_head_id), Some(latest)) =
                (pending_artifact_base_head.as_ref(), latest.as_ref())
                && latest.head_id != *base_head_id
            {
                eprintln!(
                    "trainer-daemon canonical-advance base={} canonical={} step={}",
                    base_head_id, latest.head_id, latest.global_step
                );
                pending_artifact_base_head = None;
                next_training_at = Instant::now();
            }
            last_head_sync = Instant::now();
        }

        if trainer_daemon
            .and_then(|policy| policy.max_protocol_steps)
            .is_some_and(|limit| completed_protocol_steps >= limit)
            && pending_artifact_base_head.is_none()
        {
            eprintln!(
                "trainer-daemon protocol-step-limit-reached completed_steps={} canonical_head={}",
                completed_protocol_steps,
                served_head_id
                    .as_ref()
                    .map(|head_id| head_id.as_str())
                    .unwrap_or("-"),
            );
            running.shutdown()?;
            let _prepared = running.await_termination_timeout(SHUTDOWN_TIMEOUT)?;
            eprintln!("trainer daemon stopped cleanly");
            return Ok(());
        }

        let snapshot = running.snapshot();
        if let (Some(policy), Some(experiment)) = (trainer_daemon, experiment.as_ref())
            && Instant::now() >= next_training_at
            && trainer_daemon_step_eligible(
                &snapshot.configured_roles,
                snapshot.connected_peers,
                served_head_id.is_some(),
                pending_artifact_base_head.is_some(),
            )
        {
            eprintln!(
                "trainer-daemon step-start connected_peers={} roles={:?}",
                snapshot.connected_peers, snapshot.configured_roles
            );
            match running.train_protocol_once(experiment) {
                Ok(TrainingProtocolStepOutcome::ArtifactWindow(outcome)) => {
                    consecutive_training_failures = 0;
                    let base_head_id = outcome.head.parent_head_id.clone().ok_or_else(|| {
                        anyhow!(
                            "published artifact-window head {} is missing its base parent",
                            outcome.head.head_id
                        )
                    })?;
                    eprintln!(
                        "trainer-daemon artifact-window-complete window={} head={} artifact={} train_steps={:?} data_fetch_ms={} publish_ms={}",
                        outcome.lease.window_id.0,
                        outcome.head.head_id,
                        outcome.artifact.artifact_id,
                        outcome
                            .report
                            .stats
                            .get("train_steps")
                            .or_else(|| outcome.report.stats.get("batch_count")),
                        outcome.timing.data_fetch_time_ms,
                        outcome.timing.publish_latency_ms,
                    );
                    if directory_entry_promotes_with_diffusion(&experiment_entry)
                        && let Err(error) = running.advance_diffusion_steady_state(
                            experiment,
                            Some(outcome.lease.window_id),
                            Some(&base_head_id),
                        )
                    {
                        eprintln!(
                            "trainer-daemon diffusion-settlement-deferred window={} error={error}",
                            outcome.lease.window_id.0
                        );
                    }
                    pending_artifact_base_head = Some(base_head_id);
                    next_training_at = Instant::now() + policy.minimum_step_interval;
                    completed_protocol_steps = completed_protocol_steps.saturating_add(1);
                }
                Ok(TrainingProtocolStepOutcome::DiLoCoRound(outcome)) => {
                    consecutive_training_failures = 0;
                    eprintln!(
                        "trainer-daemon diloco-round-complete round={} next_round={} group={} reducer={} participants={} inner_steps={} gradient_bytes={} checkpoint={}",
                        outcome.completed_round.round_id,
                        outcome.next_round_cursor.round_id,
                        outcome.group_id,
                        outcome.reducer_peer_id,
                        outcome.participant_peer_ids.len(),
                        outcome.local_inner_report.steps_completed,
                        outcome.local_gradient_manifest.total_encoded_bytes,
                        outcome
                            .published_checkpoint
                            .as_ref()
                            .map(|head| head.head_id.as_str())
                            .unwrap_or("-"),
                    );
                    next_training_at = Instant::now() + policy.minimum_step_interval;
                    completed_protocol_steps = completed_protocol_steps.saturating_add(1);
                }
                Err(error) => {
                    consecutive_training_failures = consecutive_training_failures.saturating_add(1);
                    eprintln!(
                        "trainer-daemon step-failed consecutive_failures={} max_failures={} retry_in_secs={} error={error}",
                        consecutive_training_failures,
                        policy.max_consecutive_failures,
                        policy.failure_backoff.as_secs(),
                    );
                    if consecutive_training_failures >= policy.max_consecutive_failures {
                        bail!(
                            "trainer daemon exceeded {} consecutive protocol-step failures: {error}",
                            policy.max_consecutive_failures
                        );
                    }
                    next_training_at = Instant::now() + policy.failure_backoff;
                }
            }
        }

        let snapshot = running.snapshot();
        if status_interval_secs > 0 && last_status.elapsed() >= status_interval {
            let ingress = running.p2p_event_bus_stats();
            eprintln!(
                "peer-status process_kind={} status={:?} node_state={:?} roles={:?} connected_peers={} completed_protocol_steps={} training_failures={} canonical_head_ready={} served_head={} waiting_for_canonical={} ecs_ingress_depth={} ecs_ingress_capacity={} ecs_ingress_high_watermark={} ecs_ingress_full={} ecs_ingress_disconnected={} last_error={}",
                process_kind,
                snapshot.status,
                snapshot.node_state,
                snapshot.configured_roles,
                snapshot.connected_peers,
                completed_protocol_steps,
                consecutive_training_failures,
                served_head_id.is_some(),
                served_head_id
                    .as_ref()
                    .map(|head_id| head_id.as_str())
                    .unwrap_or("-"),
                served_head_id.is_none() || pending_artifact_base_head.is_some(),
                ingress.queue_depth,
                ingress.queue_capacity,
                ingress.queue_high_watermark,
                ingress.sends_full,
                ingress.send_disconnects,
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
                bail!("peer runtime failed: {reason}");
            }
            RuntimeStatus::Stopped => {
                let _prepared = running.await_termination_timeout(SHUTDOWN_TIMEOUT)?;
                eprintln!("peer stopped cleanly");
                return Ok(());
            }
            _ => {}
        }

        thread::sleep(STATUS_POLL_INTERVAL);
    }
}
