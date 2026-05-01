// Browser UI and connect-state tests live here to keep wasm/mod.rs focused on runtime code.
use super::{
    BROWSER_APP_CONNECTING_REFRESH_INTERVAL_MILLIS, BROWSER_APP_DEGRADED_REFRESH_INTERVAL_MILLIS,
    BROWSER_APP_REFRESH_INTERVAL_MILLIS, DRAGON_UI_EVENT_LIMIT, DragonBrowserTransportOverride,
    DragonHeroTone, DragonPeerUiContext, DragonReadinessStepId, DragonStepStatus,
    DragonTrainingActionContext, DragonUiEventCandidate, DragonUiEventKind,
    browser_app_refresh_interval_millis, browser_session_is_authenticated,
    browser_view_machine_state_json, connect_config, dragon_browser_training_action_ready,
    dragon_global_training_detail, dragon_global_training_summary, dragon_live_notice,
    dragon_local_training_detail, dragon_local_training_summary, dragon_network_detail,
    dragon_peer_ui_state, dragon_push_ui_event, dragon_runtime_mode_detail,
    dragon_runtime_mode_summary, dragon_session_metric_view, dragon_slice_progress_summary,
    dragon_training_action_state, dragon_transport_summary, dragon_window_progress_detail,
    dragon_window_summary, filter_seed_urls_for_transport,
    filter_signed_seed_advertisement_for_transport, normalized_browser_callback_url,
    retained_refresh_transport_warning,
};
use crate::config::{DragonBrowserAppConfig, DragonPeerNetworkConfig};
use burn_p2p::{
    AuthProvider, BrowserMode, ContentId, ExperimentScope, NetworkId, PeerId, PeerRoleSet,
    PrincipalClaims, PrincipalId, PrincipalSession, ProfileMode, SocialMode,
};
use burn_p2p_app::AdminSessionSummaryView;
use burn_p2p_browser::{
    BrowserAppClientView, BrowserAppNetworkView, BrowserAppPerformanceView, BrowserAppSurface,
    BrowserAppTrainingView, BrowserAppValidationView, BrowserAppViewerView, BrowserSessionState,
    BrowserTransportKind,
};
use burn_p2p_core::{
    BrowserDirectorySnapshot, BrowserEdgeMode, BrowserEdgePaths, BrowserEdgeSnapshot,
    BrowserLeaderboardSnapshot, BrowserSeedAdvertisement, BrowserSeedRecord,
    BrowserSeedTransportKind, BrowserSeedTransportPolicy, BrowserSwarmStatus,
    BrowserTransportFamily, BrowserTransportSurface, SchemaEnvelope, SignatureAlgorithm,
    SignatureMetadata, SignedPayload,
};
use chrono::Utc;
use semver::Version;
use std::collections::{BTreeMap, BTreeSet};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

fn sample_browser_app_config() -> DragonBrowserAppConfig {
    DragonBrowserAppConfig {
        network: DragonPeerNetworkConfig::default()
            .with_edge_base_url(Some("https://edge.example".into()))
            .with_seed_node_urls(Some(vec![
                "/dns4/bootstrap.example/udp/4001/webrtc-direct/certhash/uEiAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA".into(),
                "/dns4/bootstrap.example/tcp/443/wss".into(),
            ])),
        selected_experiment_id: Some("nca-prepretraining".into()),
        selected_revision_id: Some("rev-browser".into()),
        requested_scopes: BTreeSet::from([
            ExperimentScope::Connect,
            ExperimentScope::Discover,
        ]),
        require_edge_auth: true,
        #[cfg(feature = "wasm-peer")]
        training: None,
    }
}

fn sample_edge_snapshot() -> BrowserEdgeSnapshot {
    BrowserEdgeSnapshot {
        network_id: NetworkId::new("burn-dragon-mainnet"),
        protocol_major: 0,
        minimum_client_version: semver::Version::new(0, 0, 0),
        edge_mode: BrowserEdgeMode::Peer,
        browser_mode: BrowserMode::Trainer,
        social_mode: SocialMode::Public,
        profile_mode: ProfileMode::Public,
        transports: BrowserTransportSurface {
            webrtc_direct: true,
            webtransport_gateway: true,
            wss_fallback: true,
        },
        paths: BrowserEdgePaths::default(),
        auth_enabled: true,
        login_providers: Vec::new(),
        required_release_train_hash: Some(ContentId::new("train-browser")),
        allowed_target_artifact_hashes: BTreeSet::from([ContentId::new("artifact-browser")]),
        directory: BrowserDirectorySnapshot {
            network_id: NetworkId::new("burn-dragon-mainnet"),
            generated_at: Utc::now(),
            entries: Vec::new(),
        },
        heads: Vec::new(),
        leaderboard: BrowserLeaderboardSnapshot {
            network_id: NetworkId::new("burn-dragon-mainnet"),
            score_version: "leaderboard_score_v1".into(),
            entries: Vec::new(),
            captured_at: Utc::now(),
        },
        trust_bundle: None,
        captured_at: Utc::now(),
    }
}

fn sample_signed_seed_advertisement() -> SignedPayload<SchemaEnvelope<BrowserSeedAdvertisement>> {
    SignedPayload::new(
        SchemaEnvelope::new(
            "burn_p2p.browser_seed_advertisement",
            Version::new(0, 1, 0),
            BrowserSeedAdvertisement {
                schema_version: u32::from(burn_p2p_core::SCHEMA_VERSION),
                network_id: NetworkId::new("burn-dragon-mainnet"),
                issued_at: Utc::now(),
                expires_at: Utc::now() + chrono::Duration::minutes(10),
                transport_policy: BrowserSeedTransportPolicy {
                    preferred: vec![
                        BrowserSeedTransportKind::WebRtcDirect,
                        BrowserSeedTransportKind::WssFallback,
                    ],
                    allow_fallback_wss: true,
                },
                seeds: vec![BrowserSeedRecord {
                    peer_id: Some(PeerId::new("seed-browser")),
                    multiaddrs: vec![
                        "/dns4/bootstrap.example/udp/4001/webrtc-direct/certhash/uEiAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA".into(),
                        "/dns4/bootstrap.example/tcp/443/wss".into(),
                    ],
                }],
            },
        ),
        SignatureMetadata {
            signer: PeerId::new("bootstrap"),
            key_id: "browser-seeds".into(),
            algorithm: SignatureAlgorithm::Ed25519,
            signed_at: Utc::now(),
            signature_hex: "deadbeef".into(),
        },
    )
    .expect("signed browser seed advertisement")
}

fn sample_browser_view() -> BrowserAppClientView {
    BrowserAppClientView {
        network_id: "burn-dragon-mainnet".into(),
        default_surface: BrowserAppSurface::Train,
        runtime_label: "ready".into(),
        runtime_detail: "browser runtime ready".into(),
        capability_summary: "full".into(),
        session_label: "authenticated".into(),
        selected_experiment: None,
        viewer: BrowserAppViewerView {
            visible_experiments: 0,
            visible_heads: 0,
            leaderboard_entries: 0,
            signed_directory_ready: false,
            signed_leaderboard_ready: false,
            experiments_preview: Vec::new(),
            leaderboard_preview: Vec::new(),
        },
        validation: BrowserAppValidationView {
            validate_available: false,
            can_validate: false,
            current_head_id: None,
            metrics_sync_at: None,
            pending_receipts: 0,
            validation_status: None,
            checked_chunks: None,
            emitted_receipt_id: None,
            evaluation_summary: None,
            metric_preview: Vec::new(),
        },
        training: BrowserAppTrainingView {
            train_available: true,
            can_train: true,
            active_assignment: None,
            active_training_lease: None,
            slice_status: "pending".into(),
            latest_head_id: None,
            active_head_artifact_ready: false,
            active_head_artifact_source: None,
            active_head_artifact_error: None,
            cached_chunk_artifacts: 0,
            cached_microshards: 0,
            pending_receipts: 0,
            max_window_secs: None,
            last_window_secs: None,
            optimizer_steps: None,
            accepted_samples: None,
            slice_target_samples: None,
            slice_remaining_samples: None,
            last_loss: None,
            publish_latency_ms: None,
            throughput_summary: None,
            last_artifact_id: None,
            last_receipt_id: None,
        },
        network: BrowserAppNetworkView {
            edge_base_url: "https://edge.example".into(),
            transport: BrowserTransportKind::WebRtcDirect.label().into(),
            node_state: "IdleReady".into(),
            direct_peers: 0,
            observed_peers: 0,
            estimated_network_size: 0,
            accepted_receipts: 0,
            certified_merges: 0,
            in_flight_transfers: 0,
            network_note: "test".into(),
            swarm_status: BrowserSwarmStatus::default(),
            metrics_live_ready: false,
            last_directory_sync_at: None,
            last_error: None,
            performance: None,
            diffusion: None,
        },
    }
}

fn ready_training_action_context(view: &BrowserAppClientView) -> DragonTrainingActionContext<'_> {
    DragonTrainingActionContext {
        view: Some(view),
        browser_can_attempt_dynamic_training: true,
        edge_configured: true,
        direct_transport_ready: true,
        requires_active_head_artifact: true,
        local_training_pending: false,
        local_training_failure: None,
        downgrade_reason: None,
    }
}

fn assign_training_work(view: &mut BrowserAppClientView) {
    view.training.active_assignment = Some("assignment-1".into());
}

fn set_current_head(view: &mut BrowserAppClientView) {
    view.training.latest_head_id = Some("head-1".into());
}

fn cache_current_head_from_p2p(view: &mut BrowserAppClientView) {
    set_current_head(view);
    view.training.active_head_artifact_ready = true;
    view.training.active_head_artifact_source = Some("p2p".into());
}

fn make_training_ready(view: &mut BrowserAppClientView) {
    assign_training_work(view);
    cache_current_head_from_p2p(view);
}

#[wasm_bindgen_test]
fn callback_url_normalizes_to_site_root() {
    assert_eq!(
        normalized_browser_callback_url("/callback/github", "?code=abc&state=xyz", ""),
        "/"
    );
    assert_eq!(
        normalized_browser_callback_url(
            "/repo/callback/github",
            "?code=abc&edge=https%3A%2F%2Fedge.example",
            "#frag",
        ),
        "/repo/?edge=https%3A%2F%2Fedge.example#frag"
    );
}

#[wasm_bindgen_test]
fn browser_transport_filter_keeps_only_selected_seed_family() {
    let seeds = vec![
        "/dns4/bootstrap.example/udp/4001/webrtc-direct/certhash/uEiAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA".to_owned(),
        "/dns4/bootstrap.example/tcp/443/wss".to_owned(),
    ];
    assert_eq!(
        filter_seed_urls_for_transport(seeds.clone(), DragonBrowserTransportOverride::Wss),
        vec!["/dns4/bootstrap.example/tcp/443/wss".to_owned()]
    );

    let mut advertisement = sample_signed_seed_advertisement();
    filter_signed_seed_advertisement_for_transport(
        &mut advertisement,
        DragonBrowserTransportOverride::Wss,
    );
    let payload = advertisement.payload.payload;
    assert_eq!(
        payload.transport_policy.preferred,
        vec![BrowserSeedTransportKind::WssFallback]
    );
    assert!(payload.transport_policy.allow_fallback_wss);
    assert_eq!(
        payload.seeds[0].multiaddrs,
        vec!["/dns4/bootstrap.example/tcp/443/wss".to_owned()]
    );

    let mut direct_advertisement = sample_signed_seed_advertisement();
    filter_signed_seed_advertisement_for_transport(
        &mut direct_advertisement,
        DragonBrowserTransportOverride::WebRtcDirect,
    );
    let direct_payload = direct_advertisement.payload.payload;
    assert_eq!(
        direct_payload.transport_policy.preferred,
        vec![BrowserSeedTransportKind::WebRtcDirect]
    );
    assert!(!direct_payload.transport_policy.allow_fallback_wss);
    assert_eq!(
        direct_payload.seeds[0].multiaddrs,
        vec!["/dns4/bootstrap.example/udp/4001/webrtc-direct/certhash/uEiAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA".to_owned()]
    );
}

#[wasm_bindgen_test]
fn browser_session_authentication_requires_session_claims() {
    assert!(!browser_session_is_authenticated(
        &BrowserSessionState::default()
    ));

    let now = Utc::now();
    let session = BrowserSessionState {
        session: Some(PrincipalSession {
            session_id: ContentId::new("session-browser-test"),
            network_id: NetworkId::new("burn-dragon-mainnet"),
            claims: PrincipalClaims {
                principal_id: PrincipalId::new("principal-browser-test"),
                provider: AuthProvider::Static {
                    authority: "test".into(),
                },
                display_name: "Browser Test".into(),
                org_memberships: BTreeSet::new(),
                group_memberships: BTreeSet::new(),
                granted_roles: PeerRoleSet::default(),
                granted_scopes: BTreeSet::new(),
                custom_claims: BTreeMap::new(),
                issued_at: now,
                expires_at: now,
            },
            issued_at: now,
            expires_at: now,
        }),
        ..BrowserSessionState::default()
    };
    assert!(browser_session_is_authenticated(&session));
}

#[wasm_bindgen_test]
fn dragon_live_notice_keeps_plain_connecting_state_quiet() {
    let mut view = sample_browser_view();
    view.network.swarm_status.desired_transport = Some(BrowserTransportFamily::WebRtcDirect);

    assert!(dragon_live_notice(Some(&view), false, true).is_none());
}

#[wasm_bindgen_test]
fn dragon_live_notice_reports_checkpoint_and_slice_wait_states() {
    let mut view = sample_browser_view();
    assign_training_work(&mut view);

    let checkpoint_notice =
        dragon_live_notice(Some(&view), false, true).expect("checkpoint notice");
    assert_eq!(checkpoint_notice.label, "syncing");
    assert_eq!(checkpoint_notice.detail, "syncing checkpoint");

    cache_current_head_from_p2p(&mut view);
    assert!(dragon_live_notice(Some(&view), false, true).is_none());
}

#[wasm_bindgen_test]
fn dragon_live_notice_prefers_local_training_state() {
    let notice = dragon_live_notice(Some(&sample_browser_view()), true, true).expect("training");
    assert_eq!(notice.label, "training");
    assert_eq!(notice.detail, "running a local training window in this tab");
    assert_eq!(notice.tone, "accent");
}

#[wasm_bindgen_test]
fn dragon_browser_training_action_ready_requires_settled_trainer_state() {
    let mut view = sample_browser_view();
    assert!(!dragon_browser_training_action_ready(
        Some(&view),
        true,
        true
    ));

    assign_training_work(&mut view);
    assert!(!dragon_browser_training_action_ready(
        Some(&view),
        true,
        true
    ));

    set_current_head(&mut view);
    assert!(!dragon_browser_training_action_ready(
        Some(&view),
        true,
        true
    ));

    cache_current_head_from_p2p(&mut view);
    assert!(dragon_browser_training_action_ready(
        Some(&view),
        true,
        true
    ));
    assert!(dragon_live_notice(Some(&view), false, true).is_none());
}

#[wasm_bindgen_test]
fn dragon_browser_canary_profile_skips_checkpoint_artifact_gate() {
    let mut view = sample_browser_view();
    view.runtime_label = "train".into();
    view.runtime_detail = "slice loads when training starts".into();
    view.network.direct_peers = 1;
    view.network.swarm_status.connected_transport = Some(BrowserTransportFamily::WebRtcDirect);
    view.training.can_train = true;
    assign_training_work(&mut view);
    set_current_head(&mut view);
    view.training.active_head_artifact_ready = false;
    view.training.active_head_artifact_error = Some("p2p artifact handoff timed out".into());

    assert!(!dragon_browser_training_action_ready(
        Some(&view),
        true,
        true
    ));
    assert!(dragon_browser_training_action_ready(
        Some(&view),
        true,
        false
    ));
    assert!(dragon_live_notice(Some(&view), false, false).is_none());

    let action = dragon_training_action_state(DragonTrainingActionContext {
        requires_active_head_artifact: false,
        ..ready_training_action_context(&view)
    })
    .expect("canary training action");
    assert!(action.enabled);
    assert_eq!(action.label, "run browser training");
    assert!(action.detail.contains("checkpoint load is skipped"));

    let context = DragonPeerUiContext {
        view: Some(&view),
        status_message: "",
        has_session: true,
        auth_bootstrap_pending: false,
        needs_sign_in: false,
        ready_to_connect: false,
        edge_configured: true,
        browser_can_attempt_dynamic_training: true,
        direct_transport_ready: true,
        requires_active_head_artifact: false,
        local_training_pending: false,
        local_training_failure: None,
        downgrade_reason: None,
        training_action_state: Some(&action),
        session_metric: None,
    };
    let ui = dragon_peer_ui_state(&context);
    assert_eq!(ui.hero.label, "ready to train");
    assert!(ui.hero.detail.contains("checkpoint load skipped"));
    let checkpoint = ui
        .readiness
        .iter()
        .find(|step| step.id == DragonReadinessStepId::Checkpoint)
        .expect("checkpoint step");
    assert_eq!(checkpoint.status, DragonStepStatus::Done);
    assert_eq!(
        checkpoint.detail,
        "checkpoint load skipped by training profile"
    );
}

#[wasm_bindgen_test]
fn dragon_browser_training_action_ready_blocks_joining_or_missing_transport() {
    let mut view = sample_browser_view();
    make_training_ready(&mut view);
    view.runtime_label = "joining train".into();
    view.runtime_detail = "syncing checkpoint before training".into();

    assert!(!dragon_browser_training_action_ready(
        Some(&view),
        true,
        true
    ));

    view.runtime_label = "train".into();
    view.runtime_detail = "slice loads when training starts".into();
    assert!(!dragon_browser_training_action_ready(
        Some(&view),
        false,
        true
    ));
    assert!(dragon_browser_training_action_ready(
        Some(&view),
        true,
        true
    ));
}

#[wasm_bindgen_test]
fn dragon_transport_and_progress_summaries_reflect_truthful_runtime_state() {
    let mut view = sample_browser_view();
    view.network.direct_peers = 2;
    make_training_ready(&mut view);
    view.training.cached_microshards = 2;
    view.training.last_window_secs = Some(9);
    view.training.max_window_secs = Some(30);
    view.training.accepted_samples = Some(96);
    view.training.slice_target_samples = Some(128);
    view.training.slice_remaining_samples = Some(32);

    assert_eq!(
        dragon_transport_summary(Some(&view)),
        "webrtc-direct · 2 peers"
    );
    assert_eq!(dragon_window_summary(Some(&view), false), "9s of 30s");
    assert_eq!(
        dragon_slice_progress_summary(Some(&view)),
        "96/128 · 32 left"
    );
}

#[wasm_bindgen_test]
fn dragon_transport_and_progress_summaries_degrade_cleanly_while_waiting() {
    let mut view = sample_browser_view();
    view.network.swarm_status.desired_transport = Some(BrowserTransportFamily::WebRtcDirect);

    assert_eq!(
        dragon_transport_summary(Some(&view)),
        "webrtc-direct pending"
    );
    assert_eq!(dragon_network_detail(Some(&view)), "connecting");
    assert_eq!(
        dragon_slice_progress_summary(Some(&view)),
        "waiting for work"
    );

    assign_training_work(&mut view);
    assert_eq!(
        dragon_slice_progress_summary(Some(&view)),
        "syncing checkpoint"
    );

    cache_current_head_from_p2p(&mut view);
    view.training.max_window_secs = Some(30);
    assert_eq!(dragon_slice_progress_summary(Some(&view)), "loads on run");
    assert_eq!(dragon_window_summary(Some(&view), false), "30s max");
    assert_eq!(
        dragon_window_progress_detail(Some(&view), "30s max"),
        "window cap 30s"
    );
}

#[wasm_bindgen_test]
fn dragon_network_detail_surfaces_direct_transport_error() {
    let mut view = sample_browser_view();
    view.network.swarm_status.desired_transport = Some(BrowserTransportFamily::WebRtcDirect);
    view.network.swarm_status.last_error = Some("browser direct swarm could not dial any supported seed candidate: /ip4/3.149.166.58/udp/443/webrtc-direct/certhash/uEiCZZAGOMSXZiiWY2Mi8hejsmqxCPWT3Qs3uZ9EO5uxbrA: Failed to negotiate transport protocol(s)".into());

    assert_eq!(dragon_network_detail(Some(&view)), "connection issue");
    assert_eq!(
        dragon_transport_summary(Some(&view)),
        "webrtc-direct failed"
    );
    let notice = dragon_live_notice(Some(&view), false, true).expect("direct wait notice");
    assert_eq!(notice.label, "peer connection");
    assert_eq!(
        notice.detail,
        "direct peer connection failed. full dial error is logged in the browser console."
    );
}

#[wasm_bindgen_test]
fn retained_refresh_transport_warning_ignores_stale_errors_after_direct_connect() {
    let mut view = sample_browser_view();
    view.network.swarm_status.desired_transport = Some(BrowserTransportFamily::WebRtcDirect);
    view.network.swarm_status.last_error = Some("direct dial timeout".into());
    assert_eq!(
        retained_refresh_transport_warning(&view),
        Some("direct dial timeout")
    );

    view.network.swarm_status.connected_transport = Some(BrowserTransportFamily::WebRtcDirect);
    assert!(retained_refresh_transport_warning(&view).is_none());

    view.network.swarm_status.connected_transport = None;
    view.network.direct_peers = 1;
    assert!(retained_refresh_transport_warning(&view).is_none());
}

#[wasm_bindgen_test]
fn browser_machine_state_suppresses_stale_errors_after_direct_connect() {
    let mut view = sample_browser_view();
    view.network.swarm_status.desired_transport = Some(BrowserTransportFamily::WebRtcDirect);
    view.network.swarm_status.last_error = Some("direct dial timeout".into());

    let pending_state: serde_json::Value =
        serde_json::from_str(&browser_view_machine_state_json(&view)).expect("machine state");
    assert_eq!(pending_state["last_error"], "direct dial timeout");

    view.network.direct_peers = 1;
    let connected_state: serde_json::Value =
        serde_json::from_str(&browser_view_machine_state_json(&view)).expect("machine state");
    assert!(connected_state["last_error"].is_null());
}

#[wasm_bindgen_test]
fn browser_refresh_interval_slows_when_direct_transport_is_stuck() {
    let mut view = sample_browser_view();
    view.network.swarm_status.desired_transport = Some(BrowserTransportFamily::WebRtcDirect);
    assert_eq!(
        browser_app_refresh_interval_millis(Some(&view)),
        BROWSER_APP_CONNECTING_REFRESH_INTERVAL_MILLIS
    );

    view.network.swarm_status.last_error = Some("direct dial timeout".into());
    assert_eq!(
        browser_app_refresh_interval_millis(Some(&view)),
        BROWSER_APP_DEGRADED_REFRESH_INTERVAL_MILLIS
    );

    view.network.swarm_status.desired_transport = None;
    assert_eq!(
        browser_app_refresh_interval_millis(Some(&view)),
        BROWSER_APP_REFRESH_INTERVAL_MILLIS
    );
}

#[wasm_bindgen_test]
fn dragon_connected_summary_prefers_high_signal_metrics() {
    let mut view = sample_browser_view();
    view.training.last_window_secs = Some(9);
    view.training.max_window_secs = Some(30);
    view.training.accepted_samples = Some(96);
    view.training.slice_target_samples = Some(128);
    view.training.slice_remaining_samples = Some(32);
    view.training.throughput_summary = Some("8.0 sample/s".into());
    view.training.last_loss = Some("0.421".into());
    view.network.direct_peers = 3;
    view.network.estimated_network_size = 9;
    view.network.performance = Some(BrowserAppPerformanceView {
        scope_summary: "visible peers".into(),
        captured_at: "2026-04-18T00:00:00Z".into(),
        training_throughput: "128.0 sample/s".into(),
        validation_throughput: "16.0 sample/s".into(),
        wait_time: "2s".into(),
        idle_time: "1s".into(),
    });
    let training_action_state = dragon_training_action_state(ready_training_action_context(&view));

    assert_eq!(
        dragon_local_training_summary(Some(&view), false),
        "8.0 sample/s"
    );
    assert_eq!(
        dragon_local_training_detail(Some(&view), training_action_state.as_ref()),
        "loss 0.421 · last window 9s of 30s"
    );
    assert_eq!(
        dragon_global_training_summary(Some(&view)),
        "128.0 sample/s"
    );
    assert_eq!(
        dragon_global_training_detail(Some(&view)),
        "validation 16.0 sample/s"
    );
    assert_eq!(
        dragon_window_progress_detail(Some(&view), "9s of 30s"),
        "32 left · eta 4s"
    );
    assert_eq!(
        dragon_network_detail(Some(&view)),
        "3 direct · ~9 recently seen"
    );
}

#[wasm_bindgen_test]
fn dragon_training_action_state_explains_observe_and_ready_modes() {
    let mut view = sample_browser_view();
    view.runtime_label = "observe".into();
    view.runtime_detail = "watching heads and standings".into();
    view.network.direct_peers = 2;
    view.training.can_train = false;

    let observe = dragon_training_action_state(ready_training_action_context(&view))
        .expect("observe training state");
    assert!(!observe.enabled);
    assert_eq!(observe.label, "observe mode");
    assert!(observe.detail.contains("watching the network"));

    view.training.can_train = true;
    make_training_ready(&mut view);

    let ready = dragon_training_action_state(ready_training_action_context(&view))
        .expect("ready training state");
    assert!(ready.enabled);
    assert_eq!(ready.label, "run browser training");
    assert!(ready.detail.contains("downloads when the run starts"));
}

#[wasm_bindgen_test]
fn dragon_peer_ui_state_promotes_ready_training_path() {
    let mut view = sample_browser_view();
    view.network.direct_peers = 1;
    view.network.swarm_status.connected_transport = Some(BrowserTransportFamily::WebRtcDirect);
    make_training_ready(&mut view);
    view.training.cached_microshards = 4;
    let action = dragon_training_action_state(ready_training_action_context(&view));

    let context = DragonPeerUiContext {
        view: Some(&view),
        status_message: "",
        has_session: true,
        auth_bootstrap_pending: false,
        needs_sign_in: false,
        ready_to_connect: false,
        edge_configured: true,
        browser_can_attempt_dynamic_training: true,
        direct_transport_ready: true,
        requires_active_head_artifact: true,
        local_training_pending: false,
        local_training_failure: None,
        downgrade_reason: None,
        training_action_state: action.as_ref(),
        session_metric: None,
    };
    let ui = dragon_peer_ui_state(&context);

    assert_eq!(ui.hero.label, "ready to train");
    assert_eq!(ui.hero.tone, DragonHeroTone::Ready);
    assert!(ui.hero.detail.contains("checkpoint synced"));
    assert_eq!(
        ui.readiness
            .iter()
            .find(|step| step.id == DragonReadinessStepId::TrainingReady)
            .expect("train step")
            .status,
        DragonStepStatus::Done
    );
    assert!(ui.metrics.iter().any(|metric| metric.title == "network"));
}

#[wasm_bindgen_test]
fn dragon_peer_ui_state_explains_direct_transport_failure() {
    let mut view = sample_browser_view();
    view.network.swarm_status.desired_transport = Some(BrowserTransportFamily::WebRtcDirect);
    view.network.swarm_status.last_error = Some("direct dial timeout".into());
    let action = dragon_training_action_state(DragonTrainingActionContext {
        direct_transport_ready: false,
        ..ready_training_action_context(&view)
    });

    let context = DragonPeerUiContext {
        view: Some(&view),
        status_message: "",
        has_session: true,
        auth_bootstrap_pending: false,
        needs_sign_in: false,
        ready_to_connect: false,
        edge_configured: true,
        browser_can_attempt_dynamic_training: true,
        direct_transport_ready: false,
        requires_active_head_artifact: true,
        local_training_pending: false,
        local_training_failure: None,
        downgrade_reason: None,
        training_action_state: action.as_ref(),
        session_metric: None,
    };
    let ui = dragon_peer_ui_state(&context);

    assert_eq!(ui.hero.label, "direct peer connection failed");
    assert_eq!(ui.hero.tone, DragonHeroTone::Blocked);
    assert_eq!(ui.event_candidate.kind, super::DragonUiEventKind::Error);
    assert_eq!(
        ui.readiness
            .iter()
            .find(|step| step.id == DragonReadinessStepId::DirectPeer)
            .expect("peer step")
            .status,
        DragonStepStatus::Blocked
    );
}

#[wasm_bindgen_test]
fn dragon_peer_ui_state_explains_browser_capability_block() {
    let mut view = sample_browser_view();
    view.runtime_label = "observe".into();
    view.runtime_detail = "watching heads and standings".into();
    view.training.can_train = false;
    let downgrade_reason = "webgpu unavailable; downgrading browser peer to verifier/observer";
    let action = dragon_training_action_state(DragonTrainingActionContext {
        downgrade_reason: Some(downgrade_reason),
        ..ready_training_action_context(&view)
    });

    let context = DragonPeerUiContext {
        view: Some(&view),
        status_message: "",
        has_session: true,
        auth_bootstrap_pending: false,
        needs_sign_in: false,
        ready_to_connect: false,
        edge_configured: true,
        browser_can_attempt_dynamic_training: true,
        direct_transport_ready: true,
        requires_active_head_artifact: true,
        local_training_pending: false,
        local_training_failure: None,
        downgrade_reason: Some(downgrade_reason),
        training_action_state: action.as_ref(),
        session_metric: None,
    };
    let ui = dragon_peer_ui_state(&context);

    assert_eq!(ui.hero.label, "observe mode");
    assert_eq!(ui.hero.tone, DragonHeroTone::Blocked);
    assert!(ui.hero.detail.contains("webgpu unavailable"));
    assert_eq!(
        ui.readiness
            .iter()
            .find(|step| step.id == DragonReadinessStepId::BrowserCapabilities)
            .expect("browser step")
            .detail,
        "webgpu unavailable"
    );
    assert_eq!(
        ui.readiness
            .iter()
            .find(|step| step.id == DragonReadinessStepId::TrainingReady)
            .expect("train step")
            .detail,
        "webgpu unavailable"
    );
    assert_eq!(ui.event_candidate.kind, DragonUiEventKind::Error);
    assert_eq!(ui.event_candidate.label, "webgpu unavailable");
    assert_eq!(ui.event_candidate.detail.as_deref(), Some(downgrade_reason));

    let mut blocked_view = view.clone();
    blocked_view.runtime_label = "blocked".into();
    blocked_view.runtime_detail = "training blocked".into();
    let blocked_action = dragon_training_action_state(DragonTrainingActionContext {
        downgrade_reason: Some(downgrade_reason),
        ..ready_training_action_context(&blocked_view)
    });
    let blocked_context = DragonPeerUiContext {
        view: Some(&blocked_view),
        status_message: "",
        has_session: true,
        auth_bootstrap_pending: false,
        needs_sign_in: false,
        ready_to_connect: false,
        edge_configured: true,
        browser_can_attempt_dynamic_training: true,
        direct_transport_ready: true,
        requires_active_head_artifact: true,
        local_training_pending: false,
        local_training_failure: None,
        downgrade_reason: Some(downgrade_reason),
        training_action_state: blocked_action.as_ref(),
        session_metric: None,
    };
    let blocked_ui = dragon_peer_ui_state(&blocked_context);

    assert_eq!(blocked_ui.hero.label, "blocked");
    assert_eq!(blocked_ui.hero.detail, downgrade_reason);
    assert_eq!(
        dragon_runtime_mode_detail(
            Some(&blocked_view),
            true,
            blocked_action.as_ref(),
            false,
            Some(downgrade_reason),
        ),
        downgrade_reason
    );
}

#[wasm_bindgen_test]
fn dragon_session_metric_uses_operator_copy_for_github_admin_identity() {
    let session = AdminSessionSummaryView {
        session_label: "admin session ready".into(),
        principal_label: Some("github-admin-mosure".into()),
        provider_label: Some("GitHub".into()),
        session_id: Some("session-browser".into()),
        rollout_enabled: true,
    };
    let metric = dragon_session_metric_view(&session, Some("github-a…mosure"), true)
        .expect("session metric");

    assert_eq!(metric.value, "admin ready");
    assert_eq!(metric.detail, "github · mosure");
    assert!(!metric.value.contains('…'));
    assert!(!metric.detail.contains('…'));
}

#[wasm_bindgen_test]
fn dragon_ui_events_collapse_repeated_display_messages() {
    let waiting = DragonUiEventCandidate {
        key: "peer:waiting:1".into(),
        kind: DragonUiEventKind::Peer,
        label: "waiting for peers".into(),
        detail: None,
    };
    let waiting_with_detail = DragonUiEventCandidate {
        key: "peer:waiting:2".into(),
        kind: DragonUiEventKind::Peer,
        label: "waiting for peers".into(),
        detail: Some("training unlocks after a direct peer connects".into()),
    };

    let events = dragon_push_ui_event(Vec::new(), &waiting, 1_000.0);
    let events = dragon_push_ui_event(events, &waiting, 2_000.0);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].at_ms, 1_000.0);

    let events = dragon_push_ui_event(events, &waiting_with_detail, 3_000.0);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].at_ms, 3_000.0);
    assert_eq!(events[0].detail, waiting_with_detail.detail);

    let mut events = events;
    for index in 0..(DRAGON_UI_EVENT_LIMIT + 3) {
        let candidate = DragonUiEventCandidate {
            key: format!("training:{index}"),
            kind: DragonUiEventKind::Training,
            label: format!("training event {index}"),
            detail: None,
        };
        events = dragon_push_ui_event(events, &candidate, 4_000.0 + index as f64);
    }
    assert_eq!(events.len(), DRAGON_UI_EVENT_LIMIT);
}

#[wasm_bindgen_test]
fn dragon_joining_state_uses_runtime_detail_without_redundant_action_status() {
    let mut view = sample_browser_view();
    view.runtime_label = "joining train".into();
    view.runtime_detail = "connecting peer transport".into();
    view.training.can_train = true;

    let action = dragon_training_action_state(DragonTrainingActionContext {
        direct_transport_ready: false,
        ..ready_training_action_context(&view)
    });
    assert!(action.is_none());
    assert_eq!(
        dragon_local_training_detail(Some(&view), action.as_ref()),
        "connecting peer transport"
    );
    assert_eq!(
        dragon_runtime_mode_detail(Some(&view), false, action.as_ref(), false, None),
        "connecting peer transport"
    );
}

#[wasm_bindgen_test]
fn dragon_training_action_state_prioritizes_persisted_downgrade_reason() {
    let mut view = sample_browser_view();
    view.runtime_label = "observe".into();
    view.runtime_detail = "watching heads and standings".into();
    view.training.can_train = false;

    let blocked = dragon_training_action_state(DragonTrainingActionContext {
        direct_transport_ready: false,
        downgrade_reason: Some("persisted trainer failure for this workload fingerprint"),
        ..ready_training_action_context(&view)
    })
    .expect("downgraded training state");
    assert!(!blocked.enabled);
    assert_eq!(blocked.label, "trainer downgraded");
    assert_eq!(
        blocked.detail,
        "persisted trainer failure for this workload fingerprint"
    );
}

#[wasm_bindgen_test]
fn dragon_runtime_mode_summary_exposes_friendly_connected_states() {
    let mut view = sample_browser_view();
    view.runtime_label = "observe".into();
    view.runtime_detail = "watching heads and standings".into();
    view.network.direct_peers = 2;
    view.training.can_train = false;
    let observe_training_action =
        dragon_training_action_state(ready_training_action_context(&view));

    assert_eq!(
        dragon_runtime_mode_summary(
            Some(&view),
            true,
            observe_training_action.as_ref(),
            false,
            false
        ),
        "watching"
    );
    assert_eq!(
        dragon_runtime_mode_detail(
            Some(&view),
            true,
            observe_training_action.as_ref(),
            false,
            None
        ),
        "watching network state. training turns on when trainer work is available."
    );

    view.training.can_train = true;
    make_training_ready(&mut view);
    let ready_training_action = dragon_training_action_state(ready_training_action_context(&view));

    assert_eq!(
        dragon_runtime_mode_summary(
            Some(&view),
            true,
            ready_training_action.as_ref(),
            false,
            false
        ),
        "ready to train"
    );
}

#[wasm_bindgen_test]
fn dragon_connect_config_reuses_embedded_bootstrap_when_network_matches() {
    let config = sample_browser_app_config();
    let snapshot = sample_edge_snapshot();
    let signed_seed_advertisement = sample_signed_seed_advertisement();

    let connect = connect_config(
        &config,
        &config,
        Some(&snapshot),
        Some(&signed_seed_advertisement),
    )
    .expect("connect config");

    assert_eq!(
        connect.seed_node_urls,
        vec!["/dns4/bootstrap.example/udp/4001/webrtc-direct/certhash/uEiAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA".to_owned()]
    );
    assert_eq!(connect.bootstrap_snapshot, Some(snapshot));
    let mut expected_signed_seed_advertisement = signed_seed_advertisement;
    filter_signed_seed_advertisement_for_transport(
        &mut expected_signed_seed_advertisement,
        DragonBrowserTransportOverride::WebRtcDirect,
    );
    assert_eq!(
        connect.bootstrap_signed_seed_advertisement,
        Some(expected_signed_seed_advertisement)
    );
}

#[wasm_bindgen_test]
fn dragon_connect_config_discards_embedded_bootstrap_when_overrides_diverge() {
    let config = sample_browser_app_config();
    let override_config = config.clone().with_network_overrides(
        Some("https://override-edge.example".into()),
        Some(vec!["/dns4/override.example/tcp/443/wss".into()]),
    );
    let snapshot = sample_edge_snapshot();
    let signed_seed_advertisement = sample_signed_seed_advertisement();

    let connect = connect_config(
        &config,
        &override_config,
        Some(&snapshot),
        Some(&signed_seed_advertisement),
    )
    .expect("connect config");

    assert_eq!(
        connect.seed_node_urls,
        vec!["/dns4/override.example/tcp/443/wss".to_owned()]
    );
    assert_eq!(connect.bootstrap_snapshot, None);
    assert_eq!(connect.bootstrap_signed_seed_advertisement, None);
}
