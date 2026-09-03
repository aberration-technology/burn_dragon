use super::*;
use std::collections::BTreeMap;
use std::sync::Mutex;

use burn_p2p::{
    AuthProvider, DatasetViewId, EdgeEnrollmentConfig, ExperimentId, ExperimentOptInPolicy,
    ExperimentResourceRequirements, ExperimentVisibility, NodeCertificate, NodeCertificateClaims,
    OverlayTopic, PeerId, PeerRole, PrincipalClaims, PrincipalSession, ProjectFamilyId, RevisionId,
    RevocationEpoch, StudyId, WorkloadId,
};
use burn_p2p_core::{SignatureAlgorithm, SignatureMetadata};
use chrono::Utc;
use semver::Version;
use tempfile::tempdir;

fn test_enrollment(requested_scopes: BTreeSet<ExperimentScope>) -> EdgeEnrollmentConfig {
    EdgeEnrollmentConfig {
        network_id: NetworkId::new("dragon-native-auth-testnet"),
        project_family_id: ProjectFamilyId::new("burn-dragon-language"),
        protocol_major: 0,
        app_semver: semver::Version::parse(env!("CARGO_PKG_VERSION"))
            .expect("valid burn_dragon version"),
        release_train_hash: ContentId::new("dragon-native-auth-release"),
        target_artifact_id: "native-cpu".into(),
        target_artifact_hash: ContentId::new("burn-dragon-native"),
        login_path: "/login/github".into(),
        device_path: None,
        callback_path: "/callback/github".into(),
        trusted_callback_header: None,
        trusted_callback_token: None,
        enroll_path: "/enroll".into(),
        trust_bundle_path: "/trust".into(),
        requested_scopes,
        session_ttl_secs: 1800,
    }
}

fn test_session(enrollment: &EdgeEnrollmentConfig) -> PrincipalSession {
    let now = Utc::now();
    PrincipalSession {
        session_id: ContentId::new("dragon-native-auth-session"),
        network_id: enrollment.network_id.clone(),
        claims: PrincipalClaims {
            principal_id: PrincipalId::new("github-native-cli"),
            provider: AuthProvider::GitHub,
            display_name: "native cli".into(),
            org_memberships: BTreeSet::new(),
            group_memberships: BTreeSet::new(),
            granted_roles: PeerRoleSet::new([PeerRole::TrainerCpu, PeerRole::Archive]),
            granted_scopes: enrollment.requested_scopes.clone(),
            custom_claims: BTreeMap::new(),
            issued_at: now,
            expires_at: now + chrono::Duration::minutes(30),
        },
        issued_at: now,
        expires_at: now + chrono::Duration::minutes(30),
    }
}

fn test_certificate(
    enrollment: &EdgeEnrollmentConfig,
    session: &PrincipalSession,
    identity: &burn_p2p::EdgePeerIdentity,
) -> NodeCertificate {
    let now = Utc::now();
    NodeCertificate::new(
        Version::new(0, 1, 0),
        NodeCertificateClaims {
            network_id: enrollment.network_id.clone(),
            project_family_id: enrollment.project_family_id.clone(),
            release_train_hash: enrollment.release_train_hash.clone(),
            target_artifact_hash: enrollment.target_artifact_hash.clone(),
            peer_id: identity.peer_id.clone(),
            peer_public_key_hex: identity.peer_public_key_hex.clone(),
            principal_id: session.claims.principal_id.clone(),
            provider: session.claims.provider.clone(),
            granted_roles: session.claims.granted_roles.clone(),
            experiment_scopes: enrollment.requested_scopes.clone(),
            client_policy_hash: identity.client_policy_hash.clone(),
            auth_policy_snapshot: None,
            not_before: now,
            not_after: now + chrono::Duration::minutes(30),
            serial: identity.serial,
            revocation_epoch: RevocationEpoch(0),
        },
        SignatureMetadata {
            signer: PeerId::new("dragon-native-auth-issuer"),
            key_id: "dragon-native-auth-key".into(),
            algorithm: SignatureAlgorithm::Ed25519,
            signed_at: now,
            signature_hex: "00".into(),
        },
    )
    .expect("test certificate")
}

fn post_form(callback_url: &str, fields: &[(&str, String)]) -> Result<String> {
    let url = Url::parse(callback_url)?;
    let host = url
        .host_str()
        .ok_or_else(|| anyhow!("callback url missing host"))?;
    let port = url
        .port()
        .ok_or_else(|| anyhow!("callback url missing port"))?;
    let mut body = url::form_urlencoded::Serializer::new(String::new());
    for (key, value) in fields {
        body.append_pair(key, value);
    }
    let body = body.finish();
    let target = match url.query() {
        Some(query) => format!("{}?{query}", url.path()),
        None => url.path().to_owned(),
    };
    let mut stream = TcpStream::connect((host, port))?;
    write!(
        stream,
        "POST {target} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/x-www-form-urlencoded\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )?;
    stream.flush()?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    Ok(response)
}

fn test_experiment_entry() -> ExperimentDirectoryEntry {
    ExperimentDirectoryEntry {
        network_id: NetworkId::new("burn-dragon-mainnet"),
        study_id: StudyId::new("burn-dragon-mainnet"),
        experiment_id: ExperimentId::new("nca-prepretraining"),
        workload_id: WorkloadId::new("dragon-nca-prepretraining-cpu"),
        display_name: "NCA".into(),
        model_schema_hash: ContentId::new("schema"),
        dataset_view_id: DatasetViewId::new("dataset"),
        resource_requirements: ExperimentResourceRequirements {
            minimum_roles: BTreeSet::new(),
            minimum_device_memory_bytes: None,
            minimum_system_memory_bytes: Some(1),
            estimated_download_bytes: 1,
            estimated_window_seconds: 30,
        },
        visibility: ExperimentVisibility::Public,
        opt_in_policy: ExperimentOptInPolicy::Open,
        current_revision_id: RevisionId::new("nca-r1"),
        current_head_id: None,
        allowed_roles: PeerRoleSet::new([PeerRole::TrainerCpu]),
        allowed_scopes: BTreeSet::from([ExperimentScope::Connect]),
        training_protocol: Default::default(),
        metadata: BTreeMap::new(),
    }
}

#[test]
fn directory_entry_promotes_with_diffusion_reads_merge_topology_metadata() {
    let mut entry = test_experiment_entry();
    assert!(!directory_entry_promotes_with_diffusion(&entry));

    let policy = burn_p2p::MergeTopologyPolicy {
        promotion_policy: burn_p2p::HeadPromotionPolicy {
            mode: HeadPromotionMode::DiffusionSteadyState,
            diffusion: Some(burn_p2p::DiffusionSteadyStatePolicy::default()),
            ..burn_p2p::HeadPromotionPolicy::default()
        },
        ..burn_p2p::MergeTopologyPolicy::default()
    };
    entry.metadata.insert(
        "burn_p2p.revision.merge_topology.policy_json".into(),
        serde_json::to_string(&policy).expect("serialize merge topology policy"),
    );

    assert!(directory_entry_promotes_with_diffusion(&entry));
}

fn test_head_announcement(provider_peer_id: Option<PeerId>) -> HeadAnnouncement {
    HeadAnnouncement {
        overlay: OverlayTopic::control(NetworkId::new("burn-dragon-mainnet")),
        provider_peer_id,
        head: burn_p2p::HeadDescriptor {
            head_id: burn_p2p::HeadId::new("head-1"),
            study_id: StudyId::new("burn-dragon-mainnet"),
            experiment_id: ExperimentId::new("nca-prepretraining"),
            revision_id: RevisionId::new("nca-r1"),
            artifact_id: burn_p2p::ArtifactId::new("artifact-1"),
            parent_head_id: None,
            global_step: 0,
            created_at: Utc::now(),
            metrics: BTreeMap::new(),
        },
        announced_at: Utc::now(),
    }
}

fn test_head_descriptor(head_id: &str, global_step: u64) -> burn_p2p::HeadDescriptor {
    burn_p2p::HeadDescriptor {
        head_id: burn_p2p::HeadId::new(head_id),
        study_id: StudyId::new("burn-dragon-mainnet"),
        experiment_id: ExperimentId::new("nca-prepretraining"),
        revision_id: RevisionId::new("nca-r1"),
        artifact_id: burn_p2p::ArtifactId::new(format!("artifact-{head_id}")),
        parent_head_id: None,
        global_step,
        created_at: Utc::now(),
        metrics: BTreeMap::new(),
    }
}

fn test_experiment_handle() -> ExperimentHandle {
    ExperimentHandle {
        network_id: NetworkId::new("burn-dragon-mainnet"),
        study_id: StudyId::new("burn-dragon-mainnet"),
        experiment_id: ExperimentId::new("nca-prepretraining"),
        revision_id: RevisionId::new("nca-r1"),
    }
}

fn test_head_announcement_for(head: burn_p2p::HeadDescriptor, provider: &str) -> HeadAnnouncement {
    HeadAnnouncement {
        overlay: OverlayTopic::control(NetworkId::new("burn-dragon-mainnet")),
        provider_peer_id: (!provider.is_empty()).then(|| PeerId::new(provider)),
        head,
        announced_at: Utc::now(),
    }
}

#[test]
fn latest_head_candidate_keeps_restored_head_when_network_is_stale() {
    let restored = test_head_descriptor("head-window-2", 2);
    let synced = test_head_descriptor("head-genesis", 0);

    let (selected, source) =
        select_latest_head_candidate(Some(restored), Some(synced)).expect("selected head");

    assert_eq!(source, "restored");
    assert_eq!(selected.head_id.as_str(), "head-window-2");
    assert_eq!(selected.global_step, 2);
}

#[test]
fn latest_head_candidate_prefers_synced_head_when_it_is_current() {
    let restored = test_head_descriptor("head-window-1", 1);
    let synced = test_head_descriptor("head-window-2", 2);

    let (selected, source) =
        select_latest_head_candidate(Some(restored), Some(synced)).expect("selected head");

    assert_eq!(source, "synced");
    assert_eq!(selected.head_id.as_str(), "head-window-2");
    assert_eq!(selected.global_step, 2);
}

#[test]
fn visible_promoted_head_candidate_prefers_provider_backed_newer_head() {
    let experiment = test_experiment_handle();
    let mut served = test_head_descriptor("head-window-2", 2);
    let mut stale = test_head_descriptor("head-window-1", 1);
    let mut promoted = test_head_descriptor("head-window-3", 3);
    let mut providerless = test_head_descriptor("head-window-4", 4);
    let base_time = Utc::now();
    served.created_at = base_time;
    stale.created_at = base_time - chrono::Duration::seconds(1);
    promoted.created_at = base_time + chrono::Duration::seconds(1);
    providerless.created_at = base_time + chrono::Duration::seconds(2);

    let snapshot = ControlPlaneSnapshot {
        head_announcements: vec![
            test_head_announcement_for(providerless, ""),
            test_head_announcement_for(stale, "provider-stale"),
            test_head_announcement_for(promoted, "provider-promoted"),
        ],
        ..ControlPlaneSnapshot::default()
    };

    let selected = latest_visible_promoted_head_announcement(&snapshot, &experiment, Some(&served))
        .expect("newer provider-backed head");

    assert_eq!(selected.head.head_id.as_str(), "head-window-3");
    assert_eq!(
        selected.provider_peer_id.as_ref().map(|peer| peer.as_str()),
        Some("provider-promoted"),
    );
}

#[test]
fn edge_local_head_announcement_uses_local_provider() {
    let experiment = test_experiment_handle();
    let head = test_head_descriptor("head-window-2", 2);
    let local_peer_id = PeerId::new("local-head-mirror");

    let announcement = edge_local_head_announcement(&head, &experiment, local_peer_id.clone())
        .expect("local head announcement");

    assert_eq!(announcement.head, head);
    assert_eq!(announcement.provider_peer_id, Some(local_peer_id));
    assert_eq!(
        announcement.overlay,
        experiment
            .overlay_set()
            .expect("test experiment handle has an overlay")
            .heads
    );
}

#[test]
fn edge_local_fallback_is_selected_after_unreachable_newer_head() {
    let visible = test_head_descriptor("head-window-4", 4);
    let local = test_head_descriptor("head-window-3", 3);

    assert!(should_register_edge_local_fallback(&visible, &local, None));
    assert!(!should_register_edge_local_fallback(
        &visible,
        &local,
        Some(&local.head_id),
    ));
    assert!(!should_register_edge_local_fallback(
        &visible, &visible, None,
    ));
}

fn spawn_single_response_server(
    status: &'static str,
    body: &'static str,
) -> (String, Arc<Mutex<Vec<String>>>, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
    let address = listener.local_addr().expect("test server address");
    let requests = Arc::new(Mutex::new(Vec::new()));
    let requests_for_thread = Arc::clone(&requests);
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept request");
        let mut buffer = [0; 4096];
        let read = stream.read(&mut buffer).expect("read request");
        let request = String::from_utf8_lossy(&buffer[..read]);
        let path = request
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or("<missing>")
            .to_owned();
        requests_for_thread
            .lock()
            .expect("requests lock")
            .push(path);
        write!(
                stream,
                "HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .expect("write response");
        stream.flush().expect("flush response");
    });
    (format!("http://{address}"), requests, handle)
}

#[test]
fn default_mainnet_native_config_targets_public_nca_profile() {
    let config = default_mainnet_native_config();
    let expected_seeds = DEFAULT_MAINNET_SEED_NODE_URLS
        .iter()
        .map(|seed| (*seed).to_owned())
        .collect::<Vec<_>>();

    assert!(config.training_config_paths.is_empty());
    assert_eq!(config.target, Some(DragonNativeTarget::Trainer));
    assert_eq!(
        config.effective_edge_base_url(),
        Some(DEFAULT_MAINNET_EDGE_BASE_URL)
    );
    assert_eq!(config.effective_seed_node_urls(), expected_seeds);
    assert_eq!(
        config.manifest.project_family_id,
        DEFAULT_MAINNET_PROJECT_FAMILY_ID
    );
    assert_eq!(config.manifest.network_id, DEFAULT_MAINNET_NETWORK_ID);
    assert_eq!(config.manifest.study_id, DEFAULT_MAINNET_STUDY_ID);
    assert_eq!(config.manifest.experiment_id, DEFAULT_MAINNET_EXPERIMENT_ID);
    assert_eq!(config.manifest.revision_id, DEFAULT_MAINNET_REVISION_ID);
}

#[test]
fn trainer_daemon_step_gate_requires_network_canonical_head_and_trainer_role() {
    let trainer = PeerRoleSet::new([PeerRole::TrainerGpu]);
    let observer = PeerRoleSet::new([PeerRole::Viewer]);

    assert!(trainer_daemon_step_eligible(&trainer, 1, true, false));
    assert!(!trainer_daemon_step_eligible(&trainer, 0, true, false));
    assert!(!trainer_daemon_step_eligible(&trainer, 1, false, false));
    assert!(!trainer_daemon_step_eligible(&trainer, 1, true, true));
    assert!(!trainer_daemon_step_eligible(&observer, 1, true, false));
}

#[test]
fn native_join_commands_default_to_mainnet_wgpu_and_head_sync() {
    let run_peer = Cli::try_parse_from(["burn_dragon_p2p_native", "run-peer"])
        .expect("parse run-peer defaults");
    let CommandKind::RunPeer(run_peer) = run_peer.command else {
        panic!("expected run-peer command");
    };
    assert!(run_peer.config.is_none());
    assert_eq!(run_peer.experiment_kind, ExperimentKindArg::Nca);
    assert_eq!(run_peer.backend, BackendArg::Wgpu);
    assert!(run_peer.restore_head_on_start);
    assert_eq!(
        run_peer.head_sync_interval_secs,
        DEFAULT_HEAD_SYNC_INTERVAL_SECS
    );

    let trainer_daemon = Cli::try_parse_from([
        "burn_dragon_p2p_native",
        "run-trainer-daemon",
        "--max-protocol-steps",
        "2",
        "--training-batch-size",
        "3",
    ])
    .expect("parse trainer daemon defaults");
    let CommandKind::RunTrainerDaemon(trainer_daemon) = trainer_daemon.command else {
        panic!("expected run-trainer-daemon command");
    };
    assert!(trainer_daemon.config.is_none());
    assert_eq!(trainer_daemon.experiment_kind, ExperimentKindArg::Nca);
    assert_eq!(trainer_daemon.backend, BackendArg::Wgpu);
    assert!(!trainer_daemon.initialize_head_on_start);
    assert!(trainer_daemon.restore_head_on_start);
    assert_eq!(
        trainer_daemon.head_sync_interval_secs,
        DEFAULT_HEAD_SYNC_INTERVAL_SECS
    );
    assert_eq!(trainer_daemon.max_protocol_steps, 2);
    assert_eq!(trainer_daemon.training_overrides.batch_size, Some(3));

    let validator = Cli::try_parse_from([
        "burn_dragon_p2p_native",
        "run-validator-daemon",
        "--experiment-kind",
        "nca",
    ])
    .expect("parse validator daemon defaults");
    let CommandKind::RunValidatorDaemon(validator) = validator.command else {
        panic!("expected run-validator-daemon command");
    };
    assert!(!validator.initialize_head_on_start);
    assert!(validator.restore_head_on_start);
    assert!(ensure_validator_read_only(false).is_ok());
    assert!(
        ensure_validator_read_only(true)
            .expect_err("validator initialization must fail closed")
            .to_string()
            .contains("read-only")
    );

    let doctor =
        Cli::try_parse_from(["burn_dragon_p2p_native", "doctor"]).expect("parse doctor defaults");
    let CommandKind::Doctor(doctor) = doctor.command else {
        panic!("expected doctor command");
    };
    assert!(doctor.config.is_none());
    assert_eq!(doctor.experiment_kind, ExperimentKindArg::Nca);
    assert_eq!(doctor.backend, BackendArg::Wgpu);

    let train_once = Cli::try_parse_from([
        "burn_dragon_p2p_native",
        "train-window-once",
        "--backend",
        "webgpu",
    ])
    .expect("parse train-window-once defaults");
    let CommandKind::TrainWindowOnce(train_once) = train_once.command else {
        panic!("expected train-window-once command");
    };
    assert!(train_once.config.is_none());
    assert_eq!(train_once.experiment_kind, ExperimentKindArg::Nca);
    assert_eq!(train_once.backend, BackendArg::Wgpu);
    assert!(train_once.initialize_head_on_start);
    assert!(train_once.restore_head_on_start);
    assert!(!train_once.settle_diffusion);
    assert_eq!(train_once.diffusion_settle_passes, 3);
    assert_eq!(train_once.serve_after_publish_secs, 0);
    assert!(!train_once.mirror_live_head_to_edge);
    assert_eq!(train_once.training_overrides.batch_size, None);
    assert_eq!(train_once.training_overrides.max_iters, None);
    assert_eq!(train_once.training_overrides.max_eval_batches, None);

    let no_restore = Cli::try_parse_from([
        "burn_dragon_p2p_native",
        "train-window-once",
        "--initialize-head-on-start",
        "false",
        "--restore-head-on-start",
        "false",
        "--training-batch-size",
        "1",
        "--training-max-iters",
        "4",
        "--evaluation-max-batches",
        "1",
        "--settle-diffusion",
        "--diffusion-settle-passes",
        "7",
        "--serve-after-publish-secs",
        "30",
        "--mirror-live-head-to-edge",
    ])
    .expect("parse explicit head flags");
    let CommandKind::TrainWindowOnce(no_restore) = no_restore.command else {
        panic!("expected train-window-once command");
    };
    assert!(!no_restore.initialize_head_on_start);
    assert!(!no_restore.restore_head_on_start);
    assert!(no_restore.settle_diffusion);
    assert_eq!(no_restore.diffusion_settle_passes, 7);
    assert_eq!(no_restore.serve_after_publish_secs, 30);
    assert!(no_restore.mirror_live_head_to_edge);
    assert_eq!(no_restore.training_overrides.batch_size, Some(1));
    assert_eq!(no_restore.training_overrides.max_iters, Some(4));
    assert_eq!(no_restore.training_overrides.max_eval_batches, Some(1));

    let admin_rollout = Cli::try_parse_from([
        "burn_dragon_p2p_native",
        "admin-rollout-profile",
        "--experiment-kind",
        "nca",
        "--backend",
        "cpu",
        "--auth-bundle",
        "/tmp/auth.json",
        "--reset-current-head-to-visible-root",
    ])
    .expect("parse admin rollout repair flags");
    let CommandKind::AdminRolloutProfile(admin_rollout) = admin_rollout.command else {
        panic!("expected admin-rollout-profile command");
    };
    assert!(admin_rollout.config.is_none());
    assert!(admin_rollout.reset_current_head_to_visible_root);

    let provision = Cli::try_parse_from([
        "burn_dragon_p2p_native",
        "admin-provision-revision-contract",
        "--experiment-kind",
        "nca",
        "--backend",
        "cpu",
        "--auth-bundle",
        "/tmp/auth.json",
        "--authority-key",
        "/tmp/authority.key",
        "--contract-out",
        "/tmp/nca-r1.revision-contract.json",
    ])
    .expect("parse revision-contract provisioning command");
    let CommandKind::AdminProvisionRevisionContract(provision) = provision.command else {
        panic!("expected admin-provision-revision-contract command");
    };
    assert_eq!(provision.wait_timeout_secs, 600);
    assert_eq!(provision.poll_interval_secs, 5);
}

#[test]
fn validator_config_requests_validate_scopes() {
    let mut config = default_mainnet_native_config();
    config.target = Some(DragonNativeTarget::Validator);
    let scopes = requested_scopes_for_config(&config);
    let experiment_id = ExperimentId::new(DEFAULT_MAINNET_EXPERIMENT_ID);
    assert!(scopes.contains(&ExperimentScope::Connect));
    assert!(scopes.contains(&ExperimentScope::Discover));
    assert!(scopes.contains(&ExperimentScope::Validate {
        experiment_id: experiment_id.clone()
    }));
    assert!(scopes.contains(&ExperimentScope::Archive { experiment_id }));
    assert!(!scopes.iter().any(|scope| {
        matches!(
            scope,
            ExperimentScope::Train {
                experiment_id
            } if experiment_id.as_str() == DEFAULT_MAINNET_EXPERIMENT_ID
        )
    }));
}

#[test]
fn head_mirror_registration_requires_artifact_mirror_before_live_head() {
    let (edge_base_url, requests, server) =
        spawn_single_response_server("502 Bad Gateway", "mirror unavailable");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let announcement = test_head_announcement(Some(PeerId::new(
        "12D3KooWCPbD9DgsaDHtPC6cC6DsvLNL64rtfo8UsQCVMBuazuuP",
    )));

    let error = register_live_head_with_edge_options(
        &runtime,
        &edge_base_url,
        "session-1",
        Some(&test_experiment_entry()),
        &announcement,
    )
    .expect_err("mirror failure should block live head registration");

    assert!(
        error
            .to_string()
            .contains("failed to mirror head artifact artifact-1"),
        "{error:#}"
    );
    server.join().expect("server thread");
    assert_eq!(
        *requests.lock().expect("requests lock"),
        vec!["/admin/artifacts/mirror-peer".to_owned()]
    );
}

#[test]
fn head_mirror_registration_uses_edge_provider_after_mirror() {
    let source_provider = PeerId::new("12D3KooWCPbD9DgsaDHtPC6cC6DsvLNL64rtfo8UsQCVMBuazuuP");
    let edge_provider = PeerId::new("12D3KooWJLKDYyWyB26bcJwV3u2ASqXvewHdKWRLkTe8xH7gb63");
    let announcement = test_head_announcement(Some(source_provider));
    let edge_announcement = mirrored_edge_head_announcement(&announcement, edge_provider.clone());

    assert_eq!(edge_announcement.provider_peer_id, Some(edge_provider));
    assert_eq!(edge_announcement.head, announcement.head);
    assert_eq!(edge_announcement.overlay, announcement.overlay);
}

#[test]
fn artifact_mirror_can_complete_before_canonical_head_registration() {
    let source_provider = PeerId::new("12D3KooWCPbD9DgsaDHtPC6cC6DsvLNL64rtfo8UsQCVMBuazuuP");
    let edge_provider = PeerId::new("12D3KooWJLKDYyWyB26bcJwV3u2ASqXvewHdKWRLkTe8xH7gb63");
    let (edge_base_url, requests, server) = spawn_single_response_server(
        "200 OK",
        r#"{"artifact_id":"artifact-1","mirrored_from":"12D3KooWCPbD9DgsaDHtPC6cC6DsvLNL64rtfo8UsQCVMBuazuuP","mirrored_provider_peer_id":"12D3KooWJLKDYyWyB26bcJwV3u2ASqXvewHdKWRLkTe8xH7gb63","bytes_len":1024,"chunk_count":2}"#,
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let announcement = test_head_announcement(Some(source_provider));

    let mirrored =
        mirror_head_artifact_with_edge(&runtime, &edge_base_url, "session-1", &announcement)
            .expect("mirror artifact");

    server.join().expect("server thread");
    assert_eq!(mirrored.provider_peer_id, Some(edge_provider));
    assert_eq!(mirrored.head, announcement.head);
    assert_eq!(
        *requests.lock().expect("requests lock"),
        vec!["/admin/artifacts/mirror-peer".to_owned()]
    );
}

#[test]
fn native_backend_labels_match_install_features() {
    assert_eq!(BackendArg::Cpu.default_enabled_features_label(), "native");
    assert_eq!(
        BackendArg::Wgpu.default_enabled_features_label(),
        "native,wgpu"
    );
    assert_eq!(
        BackendArg::Cuda.default_enabled_features_label(),
        "native,cuda"
    );
    assert_eq!(
        BackendArg::Rocm.default_enabled_features_label(),
        "native,rocm"
    );
    assert_eq!(native_target_artifact_id(BackendArg::Rocm), "native-rocm");
}

#[test]
fn native_cli_browser_auth_url_targets_pages_callback() {
    let storage = tempdir().expect("storage");
    let (_, identity) = edge_peer_identity_for_storage(storage.path(), None).expect("identity");
    let bootstrap = NativeCliBridgeBootstrap {
        edge_base_url: "https://edge.dragon.example".into(),
        site_base_url: "https://dragon.example".into(),
        target_artifact_id: "native-cpu".into(),
        app_semver: "0.21.0".into(),
        git_commit: "test".into(),
        enabled_features_label: "native".into(),
        requested_scopes: BTreeSet::from([ExperimentScope::Connect]),
        session_ttl_secs: 1800,
        principal_hint: Some("alice".into()),
        identity,
    };

    let url = native_cli_browser_auth_url(&bootstrap, "http://127.0.0.1:43123/callback", "nonce-1")
        .expect("bridge url");
    let parsed = Url::parse(&url).expect("parse bridge url");
    assert_eq!(parsed.scheme(), "https");
    assert_eq!(parsed.host_str(), Some("dragon.example"));
    assert_eq!(parsed.path(), "/callback/github");
    let query = parsed.query_pairs().collect::<BTreeMap<_, _>>();
    assert_eq!(
        query.get("native_cli").map(|value| value.as_ref()),
        Some("1")
    );
    assert!(query.contains_key("native_auth_bootstrap"));
    assert!(!query.contains_key("native_authorize"));
}

#[test]
fn browser_site_base_url_override_avoids_edge_hostname_guessing() {
    assert_eq!(
        resolve_browser_site_base_url(
            "https://edge-staging.dragon.example",
            Some("https://staging.dragon.example/"),
        )
        .expect("browser site base url"),
        "https://staging.dragon.example"
    );
    assert_eq!(
        resolve_browser_site_base_url("https://edge.dragon.example", None)
            .expect("inferred browser site base url"),
        "https://dragon.example"
    );
}

#[test]
fn probe_swarm_opens_listener_for_webrtc_direct_targets() {
    assert_eq!(
        probe_swarm_listen_address_for_target(
            "/dns4/edge.dragon.example/udp/443/webrtc-direct/certhash/uEiabc"
        ),
        Some("/ip4/0.0.0.0/udp/0/webrtc-direct")
    );
    assert_eq!(
        probe_swarm_listen_address_for_target("/dns4/edge.dragon.example/tcp/4001"),
        None
    );
}

#[test]
fn native_browser_auth_listener_accepts_bridge_auth_result_and_updates_cache() {
    let storage = tempdir().expect("storage");
    let (_, identity) = edge_peer_identity_for_storage(storage.path(), None).expect("identity");
    let requested_scopes = BTreeSet::from([
        ExperimentScope::Connect,
        ExperimentScope::Train {
            experiment_id: ExperimentId::new("nca-prepretraining"),
        },
    ]);
    let enrollment = test_enrollment(requested_scopes);
    let session = test_session(&enrollment);
    let certificate = test_certificate(&enrollment, &session, &identity);
    let auth_result = NativeCliBridgeAuthResult {
        edge_base_url: "https://edge.dragon.example".into(),
        enrollment,
        session,
        certificate,
    };
    let listener = start_native_browser_auth_listener().expect("listener");
    let callback_url = listener.callback_url.clone();
    let nonce = listener.nonce.clone();
    let response = post_form(
        &callback_url,
        &[
            ("native_nonce", nonce),
            (
                "auth_result_json",
                serde_json::to_string(&auth_result).expect("auth result json"),
            ),
        ],
    )
    .expect("post callback form");
    assert!(response.starts_with("HTTP/1.1 200 OK"));

    let callback = listener
        .wait(Duration::from_secs(2))
        .expect("auth callback");
    let NativeBrowserAuthCallback::AuthResult(result) = callback else {
        panic!("expected bridge auth result");
    };
    assert_eq!(result.session.session_id, auth_result.session.session_id);

    let authenticated =
        finalize_native_auth_session_from_bridge_result(storage.path(), &result, None)
            .expect("finalize native auth");
    assert!(native_auth_bundle_is_fresh(&authenticated.auth));
    assert!(authenticated.auth.auth_config.local_peer_auth.is_some());
    let cached = load_cached_native_auth_bundle(storage.path())
        .expect("load cached auth")
        .expect("cached auth");
    assert_eq!(cached.session_id, authenticated.auth.session_id);
}
