use burn_p2p_app::AdminSessionSummaryView;
use burn_p2p_browser::{BrowserAppClientView, browser_transport_kind};

use super::{
    BROWSER_APP_CONNECTING_REFRESH_INTERVAL_MILLIS, BROWSER_APP_DEGRADED_REFRESH_INTERVAL_MILLIS,
    BROWSER_APP_REFRESH_INTERVAL_MILLIS, DRAGON_UI_EVENT_LIMIT, active_direct_transport_error,
};

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct DragonLiveNotice {
    pub(super) label: &'static str,
    pub(super) detail: String,
    pub(super) tone: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct DragonTrainingActionState {
    pub(super) label: &'static str,
    pub(super) detail: String,
    pub(super) enabled: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DragonHeroTone {
    Ready,
    Working,
    Waiting,
    Blocked,
}

impl DragonHeroTone {
    pub(super) fn class(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Working => "working",
            Self::Waiting => "waiting",
            Self::Blocked => "blocked",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DragonReadinessStepId {
    Edge,
    BrowserCapabilities,
    Transport,
    DirectPeer,
    Assignment,
    Checkpoint,
    TrainingReady,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DragonStepStatus {
    Done,
    Active,
    Waiting,
    Blocked,
}

impl DragonStepStatus {
    pub(super) fn class(self) -> &'static str {
        match self {
            Self::Done => "done",
            Self::Active => "active",
            Self::Waiting => "waiting",
            Self::Blocked => "blocked",
        }
    }

    pub(super) fn marker(self) -> &'static str {
        match self {
            Self::Done => "✓",
            Self::Active => "…",
            Self::Waiting => "—",
            Self::Blocked => "!",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct DragonReadinessStepView {
    pub(super) id: DragonReadinessStepId,
    pub(super) label: &'static str,
    pub(super) status: DragonStepStatus,
    pub(super) detail: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct DragonHeroView {
    pub(super) label: String,
    pub(super) detail: String,
    pub(super) tone: DragonHeroTone,
    pub(super) animate: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct DragonMetricCardView {
    pub(super) title: &'static str,
    pub(super) value: String,
    pub(super) detail: String,
    pub(super) tone: DragonHeroTone,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct DragonSessionMetricView {
    pub(super) value: String,
    pub(super) detail: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DragonUiEventKind {
    Edge,
    Capability,
    Transport,
    Peer,
    Assignment,
    Checkpoint,
    Training,
    Error,
}

impl DragonUiEventKind {
    pub(super) fn class(self) -> &'static str {
        match self {
            Self::Edge => "edge",
            Self::Capability => "capability",
            Self::Transport => "transport",
            Self::Peer => "peer",
            Self::Assignment => "assignment",
            Self::Checkpoint => "checkpoint",
            Self::Training => "training",
            Self::Error => "error",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct DragonUiEvent {
    pub(super) at_ms: f64,
    pub(super) kind: DragonUiEventKind,
    pub(super) label: String,
    pub(super) detail: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct DragonUiEventCandidate {
    pub(super) key: String,
    pub(super) kind: DragonUiEventKind,
    pub(super) label: String,
    pub(super) detail: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct DragonPeerUiState {
    pub(super) hero: DragonHeroView,
    pub(super) readiness: Vec<DragonReadinessStepView>,
    pub(super) metrics: Vec<DragonMetricCardView>,
    pub(super) event_candidate: DragonUiEventCandidate,
}

pub(super) struct DragonPeerUiContext<'a> {
    pub(super) view: Option<&'a BrowserAppClientView>,
    pub(super) status_message: &'a str,
    pub(super) has_session: bool,
    pub(super) auth_bootstrap_pending: bool,
    pub(super) needs_sign_in: bool,
    pub(super) ready_to_connect: bool,
    pub(super) edge_configured: bool,
    pub(super) browser_can_attempt_dynamic_training: bool,
    pub(super) direct_transport_ready: bool,
    pub(super) requires_active_head_artifact: bool,
    pub(super) local_training_pending: bool,
    pub(super) local_training_failure: Option<&'a str>,
    pub(super) downgrade_reason: Option<&'a str>,
    pub(super) training_action_state: Option<&'a DragonTrainingActionState>,
    pub(super) session_metric: Option<DragonSessionMetricView>,
}

pub(super) fn browser_app_refresh_interval_millis(view: Option<&BrowserAppClientView>) -> u32 {
    let Some(view) = view else {
        return BROWSER_APP_REFRESH_INTERVAL_MILLIS;
    };
    let direct_connect_pending = view.network.swarm_status.connected_transport.is_none()
        && view.network.direct_peers == 0
        && view.network.swarm_status.desired_transport.is_some();
    if direct_connect_pending {
        return if active_direct_transport_error(view).is_some() {
            BROWSER_APP_DEGRADED_REFRESH_INTERVAL_MILLIS
        } else {
            BROWSER_APP_CONNECTING_REFRESH_INTERVAL_MILLIS
        };
    }
    if view.runtime_label.starts_with("joining ") || view.runtime_label.starts_with("catchup ") {
        return BROWSER_APP_CONNECTING_REFRESH_INTERVAL_MILLIS;
    }
    BROWSER_APP_REFRESH_INTERVAL_MILLIS
}

#[cfg(test)]
pub(super) fn dragon_live_notice(
    view: Option<&BrowserAppClientView>,
    local_training_pending: bool,
    requires_active_head_artifact: bool,
) -> Option<DragonLiveNotice> {
    if local_training_pending {
        return Some(DragonLiveNotice {
            label: "training",
            detail: "running a local training window in this tab".into(),
            tone: "accent",
        });
    }

    let view = view?;
    if view.runtime_label == "blocked" {
        return Some(DragonLiveNotice {
            label: "blocked",
            detail: view.runtime_detail.clone(),
            tone: "neutral",
        });
    }
    if view.training.can_train
        && view.network.swarm_status.connected_transport.is_none()
        && view.network.direct_peers == 0
        && view.network.swarm_status.desired_transport.is_some()
    {
        if active_direct_transport_error(view).is_some() {
            return Some(DragonLiveNotice {
                label: "peer connection",
                detail: "direct peer connection failed. full dial error is logged in the browser console.".into(),
                tone: "neutral",
            });
        }
        return None;
    }

    let active_head_artifact_ready =
        !requires_active_head_artifact || view.training.active_head_artifact_ready;
    match (
        view.training.can_train,
        view.training.active_assignment.as_ref(),
        view.training.latest_head_id.as_ref(),
        active_head_artifact_ready,
    ) {
        (true, None, _, _) => Some(DragonLiveNotice {
            label: "waiting",
            detail: "waiting for work".into(),
            tone: "neutral",
        }),
        (true, Some(_), None, _) => Some(DragonLiveNotice {
            label: "syncing",
            detail: if requires_active_head_artifact {
                "syncing checkpoint".into()
            } else {
                "waiting for current head".into()
            },
            tone: "neutral",
        }),
        (true, Some(_), Some(_), false) => Some(DragonLiveNotice {
            label: "syncing",
            detail: view
                .training
                .active_head_artifact_error
                .clone()
                .unwrap_or_else(|| "syncing checkpoint".into()),
            tone: "neutral",
        }),
        _ => None,
    }
}

#[derive(Clone, Copy)]
pub(super) struct DragonTrainingActionContext<'a> {
    pub(super) view: Option<&'a BrowserAppClientView>,
    pub(super) browser_can_attempt_dynamic_training: bool,
    pub(super) edge_configured: bool,
    pub(super) direct_transport_ready: bool,
    pub(super) requires_active_head_artifact: bool,
    pub(super) local_training_pending: bool,
    pub(super) local_training_failure: Option<&'a str>,
    pub(super) downgrade_reason: Option<&'a str>,
}

pub(super) fn dragon_training_action_state(
    context: DragonTrainingActionContext<'_>,
) -> Option<DragonTrainingActionState> {
    let DragonTrainingActionContext {
        view,
        browser_can_attempt_dynamic_training,
        edge_configured,
        direct_transport_ready,
        requires_active_head_artifact,
        local_training_pending,
        local_training_failure,
        downgrade_reason,
    } = context;

    if !browser_can_attempt_dynamic_training {
        return None;
    }
    if local_training_pending {
        return Some(DragonTrainingActionState {
            label: "stop training",
            detail: "training continues window by window until stopped".into(),
            enabled: true,
        });
    }
    if let Some(error) = local_training_failure {
        return Some(DragonTrainingActionState {
            label: "retry browser training",
            detail: error.to_owned(),
            enabled: true,
        });
    }

    let view = view?;
    if !edge_configured {
        return Some(DragonTrainingActionState {
            label: "training unavailable",
            detail: "edge configuration is missing".into(),
            enabled: false,
        });
    }
    if view.runtime_label.starts_with("joining ") || view.runtime_label.starts_with("catchup ") {
        return None;
    }
    if view.runtime_label == "blocked" {
        return Some(DragonTrainingActionState {
            label: "training blocked",
            detail: downgrade_reason
                .map(str::to_owned)
                .unwrap_or_else(|| view.runtime_detail.clone()),
            enabled: false,
        });
    }
    if !view.training.can_train {
        return Some(DragonTrainingActionState {
            label: if downgrade_reason.is_some() {
                "trainer downgraded"
            } else {
                "observe mode"
            },
            detail: downgrade_reason
                .map(str::to_owned)
                .unwrap_or_else(|| {
                    "this tab is connected and watching the network. training turns on after trainer work is available.".into()
                }),
            enabled: false,
        });
    }
    if !direct_transport_ready {
        return Some(DragonTrainingActionState {
            label: "waiting for peers",
            detail: "training unlocks after a direct webrtc peer connects".into(),
            enabled: false,
        });
    }
    if view.training.active_assignment.is_none() {
        return Some(DragonTrainingActionState {
            label: "waiting for work",
            detail: "connected to the swarm. training turns on when this peer receives a training assignment.".into(),
            enabled: false,
        });
    }
    if view.training.latest_head_id.is_none() {
        return Some(DragonTrainingActionState {
            label: "syncing checkpoint",
            detail: "loading the current head before browser training can start".into(),
            enabled: false,
        });
    }
    if requires_active_head_artifact && !view.training.active_head_artifact_ready {
        return Some(DragonTrainingActionState {
            label: "syncing checkpoint",
            detail: view
                .training
                .active_head_artifact_error
                .clone()
                .unwrap_or_else(|| {
                    "fetching the active head over p2p before browser training can start".into()
                }),
            enabled: false,
        });
    }

    if dragon_browser_training_action_ready(
        Some(view),
        direct_transport_ready,
        requires_active_head_artifact,
    ) {
        Some(DragonTrainingActionState {
            label: "run browser training",
            detail: if !requires_active_head_artifact {
                "manual in this tab. checkpoint load is skipped by the active training profile."
                    .into()
            } else if view.training.active_head_artifact_source.as_deref() == Some("edge-fallback")
            {
                "manual in this tab. checkpoint is cached via edge fallback; p2p artifact serving is degraded."
                    .into()
            } else if view.training.cached_microshards > 0 {
                "manual in this tab. the assigned slice is already cached for this browser peer."
                    .into()
            } else {
                "manual in this tab. the assigned slice downloads when the run starts.".into()
            },
            enabled: true,
        })
    } else {
        Some(DragonTrainingActionState {
            label: "training unavailable",
            detail: "browser training is not ready yet".into(),
            enabled: false,
        })
    }
}

fn dragon_short_block_reason(reason: &str) -> String {
    let reason = reason.trim();
    if reason.is_empty() {
        return "training blocked".into();
    }
    let short = reason
        .split([';', '.'])
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(reason);
    short.to_owned()
}

fn dragon_browser_capability_block_summary(
    downgrade_reason: Option<&str>,
    browser_can_attempt_dynamic_training: bool,
) -> String {
    if let Some(reason) = downgrade_reason {
        return dragon_short_block_reason(reason);
    }
    if !browser_can_attempt_dynamic_training {
        return "webgpu or worker unavailable".into();
    }
    "browser capability blocked".into()
}

fn dragon_browser_capability_block_detail(
    downgrade_reason: Option<&str>,
    browser_can_attempt_dynamic_training: bool,
) -> String {
    if let Some(reason) = downgrade_reason {
        return reason.to_owned();
    }
    if !browser_can_attempt_dynamic_training {
        return "browser training needs WebGPU available to the page and worker. this tab can still watch the network.".into();
    }
    "browser capability policy blocked local training. open advanced diagnostics for capability details.".into()
}

fn dragon_training_block_detail(
    view: Option<&BrowserAppClientView>,
    downgrade_reason: Option<&str>,
    browser_can_attempt_dynamic_training: bool,
) -> String {
    if downgrade_reason.is_some() || !browser_can_attempt_dynamic_training {
        return dragon_browser_capability_block_summary(
            downgrade_reason,
            browser_can_attempt_dynamic_training,
        );
    }
    view.and_then(|view| {
        let detail = view.runtime_detail.trim();
        (!detail.is_empty()).then(|| detail.to_owned())
    })
    .unwrap_or_else(|| "training blocked".into())
}

pub(super) fn dragon_peer_ui_state(context: &DragonPeerUiContext<'_>) -> DragonPeerUiState {
    let hero = dragon_hero_view(context);
    let readiness = dragon_readiness_steps(context);
    let metrics = dragon_metric_cards(context, hero.tone);
    let event_candidate = dragon_ui_event_candidate(&hero, &readiness);
    DragonPeerUiState {
        hero,
        readiness,
        metrics,
        event_candidate,
    }
}

fn dragon_hero_view(context: &DragonPeerUiContext<'_>) -> DragonHeroView {
    let view = context.view;
    let status_message = context.status_message;
    let auth_bootstrap_pending = context.auth_bootstrap_pending;
    let needs_sign_in = context.needs_sign_in;
    let ready_to_connect = context.ready_to_connect;
    let edge_configured = context.edge_configured;
    let browser_can_attempt_dynamic_training = context.browser_can_attempt_dynamic_training;
    let direct_transport_ready = context.direct_transport_ready;
    let requires_active_head_artifact = context.requires_active_head_artifact;
    let local_training_pending = context.local_training_pending;
    let local_training_failure = context.local_training_failure;
    let downgrade_reason = context.downgrade_reason;
    let training_action_state = context.training_action_state;

    if local_training_pending {
        return DragonHeroView {
            label: "training…".into(),
            detail: "running a local training window in this tab".into(),
            tone: DragonHeroTone::Working,
            animate: true,
        };
    }
    if let Some(error) = local_training_failure {
        return DragonHeroView {
            label: "training failed".into(),
            detail: error.to_owned(),
            tone: DragonHeroTone::Blocked,
            animate: false,
        };
    }
    if auth_bootstrap_pending {
        return DragonHeroView {
            label: "connecting".into(),
            detail: "checking session and edge state".into(),
            tone: DragonHeroTone::Working,
            animate: true,
        };
    }
    if !edge_configured {
        return DragonHeroView {
            label: "blocked".into(),
            detail: "edge config is missing. set an edge url before this browser peer can connect."
                .into(),
            tone: DragonHeroTone::Blocked,
            animate: false,
        };
    }
    if needs_sign_in {
        return DragonHeroView {
            label: "sign in required".into(),
            detail: "github session is needed before this browser peer can join the network".into(),
            tone: DragonHeroTone::Waiting,
            animate: false,
        };
    }
    if ready_to_connect {
        return DragonHeroView {
            label: "ready to connect".into(),
            detail: "edge config loaded. connect this browser to the peer network.".into(),
            tone: DragonHeroTone::Waiting,
            animate: false,
        };
    }
    if let Some(view) = view {
        if view.runtime_label == "blocked" {
            return DragonHeroView {
                label: "blocked".into(),
                detail: downgrade_reason
                    .map(str::to_owned)
                    .unwrap_or_else(|| view.runtime_detail.clone()),
                tone: DragonHeroTone::Blocked,
                animate: false,
            };
        }
        if active_direct_transport_error(view).is_some() {
            return DragonHeroView {
                label: "direct peer connection failed".into(),
                detail: "browser training needs a direct WebRTC peer before it can start. retry connect, switch transport, or open diagnostics.".into(),
                tone: DragonHeroTone::Blocked,
                animate: false,
            };
        }
        if let Some(action) = training_action_state.filter(|state| state.enabled) {
            let checkpoint_detail = if requires_active_head_artifact {
                "checkpoint synced"
            } else {
                "checkpoint load skipped"
            };
            let detail = if view.training.cached_microshards > 0 {
                format!("direct peer connected · {checkpoint_detail} · assigned slice cached")
            } else {
                format!("direct peer connected · {checkpoint_detail} · assigned slice ready")
            };
            return DragonHeroView {
                label: action
                    .label
                    .strip_prefix("run browser training")
                    .map(|_| "ready to train")
                    .unwrap_or(action.label)
                    .into(),
                detail,
                tone: DragonHeroTone::Ready,
                animate: false,
            };
        }
        if view.runtime_label.starts_with("joining ") {
            return DragonHeroView {
                label: "connecting".into(),
                detail: dragon_runtime_mode_detail(
                    Some(view),
                    direct_transport_ready,
                    training_action_state,
                    false,
                    downgrade_reason,
                ),
                tone: DragonHeroTone::Working,
                animate: true,
            };
        }
        if view.runtime_label.starts_with("catchup ") {
            return DragonHeroView {
                label: "syncing checkpoint".into(),
                detail: dragon_runtime_mode_detail(
                    Some(view),
                    direct_transport_ready,
                    training_action_state,
                    false,
                    downgrade_reason,
                ),
                tone: DragonHeroTone::Working,
                animate: true,
            };
        }
        if !browser_can_attempt_dynamic_training {
            return DragonHeroView {
                label: "observe mode".into(),
                detail: dragon_browser_capability_block_detail(None, false),
                tone: DragonHeroTone::Blocked,
                animate: false,
            };
        }
        if let Some(reason) = downgrade_reason {
            return DragonHeroView {
                label: "observe mode".into(),
                detail: reason.to_owned(),
                tone: DragonHeroTone::Blocked,
                animate: false,
            };
        }
        if let Some(action) = training_action_state {
            return DragonHeroView {
                label: action.label.into(),
                detail: action.detail.clone(),
                tone: if action.label == "waiting for peers" {
                    DragonHeroTone::Waiting
                } else {
                    DragonHeroTone::Working
                },
                animate: matches!(
                    action.label,
                    "waiting for peers" | "waiting for work" | "syncing checkpoint"
                ),
            };
        }
        return DragonHeroView {
            label: dragon_runtime_mode_summary(
                Some(view),
                direct_transport_ready,
                training_action_state,
                false,
                false,
            ),
            detail: dragon_runtime_mode_detail(
                Some(view),
                direct_transport_ready,
                training_action_state,
                false,
                downgrade_reason,
            ),
            tone: DragonHeroTone::Waiting,
            animate: false,
        };
    }
    let fallback_detail = if status_message.trim().is_empty() {
        "browser runtime not connected".into()
    } else {
        status_message.to_owned()
    };
    DragonHeroView {
        label: "waiting".into(),
        detail: fallback_detail,
        tone: DragonHeroTone::Waiting,
        animate: false,
    }
}

fn dragon_readiness_steps(context: &DragonPeerUiContext<'_>) -> Vec<DragonReadinessStepView> {
    let view = context.view;
    let auth_bootstrap_pending = context.auth_bootstrap_pending;
    let edge_configured = context.edge_configured;
    let browser_can_attempt_dynamic_training = context.browser_can_attempt_dynamic_training;
    let direct_transport_ready = context.direct_transport_ready;
    let requires_active_head_artifact = context.requires_active_head_artifact;
    let local_training_pending = context.local_training_pending;
    let local_training_failure = context.local_training_failure;
    let downgrade_reason = context.downgrade_reason;
    let training_action_state = context.training_action_state;

    let direct_error = view.and_then(active_direct_transport_error);
    let capability_block_summary = dragon_browser_capability_block_summary(
        downgrade_reason,
        browser_can_attempt_dynamic_training,
    );
    let training_block_detail =
        dragon_training_block_detail(view, downgrade_reason, browser_can_attempt_dynamic_training);
    let edge_status = if !edge_configured {
        DragonStepStatus::Blocked
    } else if auth_bootstrap_pending {
        DragonStepStatus::Active
    } else {
        DragonStepStatus::Done
    };
    let browser_status = if browser_can_attempt_dynamic_training && downgrade_reason.is_none() {
        DragonStepStatus::Done
    } else if auth_bootstrap_pending {
        DragonStepStatus::Active
    } else {
        DragonStepStatus::Blocked
    };
    let transport_status = if direct_error.is_some() {
        DragonStepStatus::Blocked
    } else if view.is_some_and(|view| view.network.swarm_status.connected_transport.is_some()) {
        DragonStepStatus::Done
    } else if view.is_some_and(|view| view.network.swarm_status.desired_transport.is_some()) {
        DragonStepStatus::Active
    } else {
        DragonStepStatus::Waiting
    };
    let peer_status = if direct_error.is_some() {
        DragonStepStatus::Blocked
    } else if direct_transport_ready {
        DragonStepStatus::Done
    } else if view.is_some() {
        DragonStepStatus::Active
    } else {
        DragonStepStatus::Waiting
    };
    let assignment_status = match view {
        Some(view) if view.training.active_assignment.is_some() => DragonStepStatus::Done,
        Some(view) if view.training.can_train && direct_transport_ready => DragonStepStatus::Active,
        Some(view) if !view.training.can_train => DragonStepStatus::Waiting,
        Some(_) => DragonStepStatus::Waiting,
        None => DragonStepStatus::Waiting,
    };
    let checkpoint_status = if requires_active_head_artifact {
        match view {
            Some(view) if view.training.active_head_artifact_ready => DragonStepStatus::Done,
            Some(view) if view.training.latest_head_id.is_some() => DragonStepStatus::Active,
            Some(view) if view.training.active_assignment.is_some() => DragonStepStatus::Active,
            Some(_) => DragonStepStatus::Waiting,
            None => DragonStepStatus::Waiting,
        }
    } else {
        match view {
            Some(view) if view.training.latest_head_id.is_some() => DragonStepStatus::Done,
            Some(view) if view.training.active_assignment.is_some() => DragonStepStatus::Active,
            Some(_) => DragonStepStatus::Waiting,
            None => DragonStepStatus::Waiting,
        }
    };
    let training_status = if local_training_pending {
        DragonStepStatus::Active
    } else if local_training_failure.is_some() {
        DragonStepStatus::Blocked
    } else if training_action_state.is_some_and(|state| state.enabled) {
        DragonStepStatus::Done
    } else if view.is_some_and(|view| view.runtime_label == "blocked")
        || downgrade_reason.is_some()
        || !browser_can_attempt_dynamic_training
    {
        DragonStepStatus::Blocked
    } else {
        DragonStepStatus::Waiting
    };

    vec![
        DragonReadinessStepView {
            id: DragonReadinessStepId::Edge,
            label: "edge",
            status: edge_status,
            detail: match edge_status {
                DragonStepStatus::Done => "edge reachable".into(),
                DragonStepStatus::Active => "checking edge".into(),
                DragonStepStatus::Blocked => "edge config missing".into(),
                DragonStepStatus::Waiting => "waiting for edge".into(),
            },
        },
        DragonReadinessStepView {
            id: DragonReadinessStepId::BrowserCapabilities,
            label: "browser",
            status: browser_status,
            detail: match browser_status {
                DragonStepStatus::Done => "WebGPU available".into(),
                DragonStepStatus::Active => "probing browser".into(),
                DragonStepStatus::Blocked => capability_block_summary.clone(),
                DragonStepStatus::Waiting => "waiting for browser".into(),
            },
        },
        DragonReadinessStepView {
            id: DragonReadinessStepId::Transport,
            label: "transport",
            status: transport_status,
            detail: match transport_status {
                DragonStepStatus::Done => "transport selected".into(),
                DragonStepStatus::Active => "dialing peer".into(),
                DragonStepStatus::Blocked => "direct transport failed".into(),
                DragonStepStatus::Waiting => "transport pending".into(),
            },
        },
        DragonReadinessStepView {
            id: DragonReadinessStepId::DirectPeer,
            label: "peer",
            status: peer_status,
            detail: match peer_status {
                DragonStepStatus::Done => "direct peer connected".into(),
                DragonStepStatus::Active => "waiting for peers".into(),
                DragonStepStatus::Blocked => "peer connection failed".into(),
                DragonStepStatus::Waiting => "waiting for peers".into(),
            },
        },
        DragonReadinessStepView {
            id: DragonReadinessStepId::Assignment,
            label: "assignment",
            status: assignment_status,
            detail: match assignment_status {
                DragonStepStatus::Done => "work assigned".into(),
                DragonStepStatus::Active => "waiting for work".into(),
                DragonStepStatus::Blocked => "no assignment".into(),
                DragonStepStatus::Waiting => "waiting for work".into(),
            },
        },
        DragonReadinessStepView {
            id: DragonReadinessStepId::Checkpoint,
            label: "checkpoint",
            status: checkpoint_status,
            detail: match checkpoint_status {
                DragonStepStatus::Done => {
                    if !requires_active_head_artifact {
                        "checkpoint load skipped by training profile".into()
                    } else {
                        match view
                            .and_then(|view| view.training.active_head_artifact_source.as_deref())
                        {
                            Some("p2p") => "checkpoint synced over p2p".into(),
                            Some("edge-fallback") => "checkpoint synced via edge fallback".into(),
                            _ => "checkpoint synced".into(),
                        }
                    }
                }
                DragonStepStatus::Active => {
                    if !requires_active_head_artifact {
                        "waiting for current head".into()
                    } else {
                        view.and_then(|view| view.training.active_head_artifact_error.clone())
                            .unwrap_or_else(|| "syncing checkpoint over p2p".into())
                    }
                }
                DragonStepStatus::Blocked => "checkpoint unavailable".into(),
                DragonStepStatus::Waiting => "checkpoint pending".into(),
            },
        },
        DragonReadinessStepView {
            id: DragonReadinessStepId::TrainingReady,
            label: "train",
            status: training_status,
            detail: match training_status {
                DragonStepStatus::Done => "ready".into(),
                DragonStepStatus::Active => "training".into(),
                DragonStepStatus::Blocked => local_training_failure
                    .map(str::to_owned)
                    .unwrap_or(training_block_detail),
                DragonStepStatus::Waiting => "not ready yet".into(),
            },
        },
    ]
}

fn dragon_metric_cards(
    context: &DragonPeerUiContext<'_>,
    hero_tone: DragonHeroTone,
) -> Vec<DragonMetricCardView> {
    let view = context.view;
    let has_session = context.has_session;
    let direct_transport_ready = context.direct_transport_ready;
    let local_training_pending = context.local_training_pending;
    let downgrade_reason = context.downgrade_reason;
    let training_action_state = context.training_action_state;

    let runtime_summary = dragon_runtime_mode_summary(
        view,
        direct_transport_ready,
        training_action_state,
        false,
        local_training_pending,
    );
    let runtime_detail = dragon_runtime_mode_detail(
        view,
        direct_transport_ready,
        training_action_state,
        local_training_pending,
        downgrade_reason,
    );
    let window_summary = dragon_window_summary(view, local_training_pending);
    let mut metrics = vec![
        DragonMetricCardView {
            title: "network",
            value: dragon_transport_summary(view),
            detail: dragon_network_detail(view),
            tone: if direct_transport_ready {
                DragonHeroTone::Ready
            } else {
                hero_tone
            },
        },
        DragonMetricCardView {
            title: "mode",
            value: runtime_summary,
            detail: runtime_detail,
            tone: hero_tone,
        },
        DragonMetricCardView {
            title: "local",
            value: dragon_local_training_summary(view, local_training_pending),
            detail: dragon_local_training_detail(view, training_action_state),
            tone: if local_training_pending {
                DragonHeroTone::Working
            } else {
                DragonHeroTone::Waiting
            },
        },
        DragonMetricCardView {
            title: "swarm",
            value: dragon_global_training_summary(view),
            detail: dragon_global_training_detail(view),
            tone: DragonHeroTone::Waiting,
        },
        DragonMetricCardView {
            title: "slice",
            value: dragon_slice_progress_summary(view),
            detail: dragon_window_progress_detail(view, &window_summary),
            tone: if training_action_state.is_some_and(|state| state.enabled) {
                DragonHeroTone::Ready
            } else {
                DragonHeroTone::Waiting
            },
        },
    ];
    if has_session || view.is_some_and(|view| !view.session_label.trim().is_empty()) {
        let session_metric = context
            .session_metric
            .clone()
            .or_else(|| dragon_fallback_session_metric(view, has_session));
        metrics.push(DragonMetricCardView {
            title: "session",
            value: session_metric
                .as_ref()
                .map(|metric| metric.value.clone())
                .unwrap_or_else(|| "signed in".into()),
            detail: session_metric
                .map(|metric| metric.detail)
                .unwrap_or_else(|| "session active".into()),
            tone: DragonHeroTone::Ready,
        });
    }
    metrics
}

pub(super) fn dragon_session_metric_view(
    session: &AdminSessionSummaryView,
    fallback_label: Option<&str>,
    has_session: bool,
) -> Option<DragonSessionMetricView> {
    let fallback_label = fallback_label
        .map(str::trim)
        .filter(|label| !label.is_empty() && !matches!(*label, "guest" | "local node"));
    if !has_session && fallback_label.is_none() {
        return None;
    }

    let value = if session.rollout_enabled {
        "admin ready".into()
    } else if has_session {
        "signed in".into()
    } else {
        fallback_label
            .map(dragon_session_value_from_label)
            .unwrap_or_else(|| "signed in".into())
    };
    let detail = dragon_session_metric_detail(session, fallback_label);
    Some(DragonSessionMetricView { value, detail })
}

fn dragon_fallback_session_metric(
    view: Option<&BrowserAppClientView>,
    has_session: bool,
) -> Option<DragonSessionMetricView> {
    let label = view
        .map(|view| view.session_label.trim())
        .filter(|label| !label.is_empty() && !matches!(*label, "guest" | "local node"));
    if !has_session && label.is_none() {
        return None;
    }
    Some(DragonSessionMetricView {
        value: if has_session {
            "signed in".into()
        } else {
            label
                .map(dragon_session_value_from_label)
                .unwrap_or_else(|| "signed in".into())
        },
        detail: "session active".into(),
    })
}

fn dragon_session_value_from_label(label: &str) -> String {
    let label = label.trim();
    if label.is_empty() || label.contains('…') || label.len() > 16 {
        return "signed in".into();
    }
    label.replace(['_', '-'], " ")
}

fn dragon_session_metric_detail(
    session: &AdminSessionSummaryView,
    fallback_label: Option<&str>,
) -> String {
    let provider = session
        .provider_label
        .as_deref()
        .map(dragon_metric_provider_label);
    let principal = session
        .principal_label
        .as_deref()
        .and_then(|principal| dragon_metric_principal_label(principal, provider.as_deref()))
        .or_else(|| fallback_label.and_then(|label| dragon_metric_principal_label(label, None)));

    match (provider, principal) {
        (Some(provider), Some(principal)) => format!("{provider} · {principal}"),
        (Some(provider), None) => format!("{provider} session active"),
        (None, Some(principal)) => format!("operator · {principal}"),
        (None, None) => "session active".into(),
    }
}

fn dragon_metric_provider_label(provider: &str) -> String {
    let provider = provider.trim();
    if provider.eq_ignore_ascii_case("github") {
        "github".into()
    } else if provider.len() <= 14 {
        provider.to_ascii_lowercase()
    } else {
        "session".into()
    }
}

fn dragon_metric_principal_label(principal: &str, provider: Option<&str>) -> Option<String> {
    let mut label = principal.trim();
    if label.is_empty() || matches!(label, "authenticated" | "signed in" | "guest") {
        return None;
    }
    if provider == Some("github") {
        label = label
            .strip_prefix("github-admin-")
            .or_else(|| label.strip_prefix("github-"))
            .unwrap_or(label);
    }
    if label.contains('…') || label.len() > 18 {
        return None;
    }
    Some(label.replace(['_', '-'], " "))
}

fn dragon_ui_event_candidate(
    hero: &DragonHeroView,
    readiness: &[DragonReadinessStepView],
) -> DragonUiEventCandidate {
    if let Some(step) = readiness
        .iter()
        .find(|step| step.status == DragonStepStatus::Blocked)
    {
        return DragonUiEventCandidate {
            key: format!("{:?}:blocked:{}", step.id, step.detail),
            kind: DragonUiEventKind::Error,
            label: step.detail.clone(),
            detail: dragon_blocked_event_detail(step, hero),
        };
    }
    if let Some(step) = readiness
        .iter()
        .find(|step| step.status == DragonStepStatus::Active)
    {
        return DragonUiEventCandidate {
            key: format!("{:?}:active:{}", step.id, step.detail),
            kind: dragon_readiness_event_kind(step.id),
            label: step.detail.clone(),
            detail: None,
        };
    }
    DragonUiEventCandidate {
        key: format!("hero:{}:{}", hero.tone.class(), hero.label),
        kind: match hero.tone {
            DragonHeroTone::Ready => DragonUiEventKind::Training,
            DragonHeroTone::Working => DragonUiEventKind::Transport,
            DragonHeroTone::Waiting => DragonUiEventKind::Peer,
            DragonHeroTone::Blocked => DragonUiEventKind::Error,
        },
        label: hero.label.clone(),
        detail: dragon_ui_event_detail(&hero.label, &hero.detail),
    }
}

fn dragon_blocked_event_detail(
    step: &DragonReadinessStepView,
    hero: &DragonHeroView,
) -> Option<String> {
    if hero.tone == DragonHeroTone::Blocked {
        let detail = dragon_ui_event_detail(&step.detail, &hero.detail);
        if detail.is_some() {
            return detail;
        }
    }
    match step.id {
        DragonReadinessStepId::Edge => Some("set an edge url before this browser peer can connect".into()),
        DragonReadinessStepId::BrowserCapabilities => Some("browser training needs WebGPU and worker support. this tab can still watch the network.".into()),
        DragonReadinessStepId::Transport | DragonReadinessStepId::DirectPeer => {
            Some("training needs a direct WebRTC peer before it can start".into())
        }
        DragonReadinessStepId::Assignment => Some("training starts after this peer receives work".into()),
        DragonReadinessStepId::Checkpoint => Some("training starts after the current checkpoint is available".into()),
        DragonReadinessStepId::TrainingReady => Some("resolve the blocked readiness step before training can run".into()),
    }
}

fn dragon_ui_event_detail(label: &str, detail: &str) -> Option<String> {
    let detail = detail.trim();
    if detail.is_empty() || detail.eq_ignore_ascii_case(label.trim()) {
        None
    } else {
        Some(detail.to_owned())
    }
}

fn dragon_ui_event_matches_candidate(
    event: &DragonUiEvent,
    candidate: &DragonUiEventCandidate,
) -> bool {
    event.kind == candidate.kind
        && event.label == candidate.label
        && event.detail == candidate.detail
}

fn dragon_ui_event_replaces_candidate(
    event: &DragonUiEvent,
    candidate: &DragonUiEventCandidate,
) -> bool {
    event.kind == candidate.kind && event.label == candidate.label
}

pub(super) fn dragon_push_ui_event(
    mut events: Vec<DragonUiEvent>,
    candidate: &DragonUiEventCandidate,
    at_ms: f64,
) -> Vec<DragonUiEvent> {
    if events
        .first()
        .is_some_and(|event| dragon_ui_event_matches_candidate(event, candidate))
    {
        return events;
    }
    events.retain(|event| !dragon_ui_event_replaces_candidate(event, candidate));
    events.insert(
        0,
        DragonUiEvent {
            at_ms,
            kind: candidate.kind,
            label: candidate.label.clone(),
            detail: candidate.detail.clone(),
        },
    );
    events.truncate(DRAGON_UI_EVENT_LIMIT);
    events
}

fn dragon_readiness_event_kind(id: DragonReadinessStepId) -> DragonUiEventKind {
    match id {
        DragonReadinessStepId::Edge => DragonUiEventKind::Edge,
        DragonReadinessStepId::BrowserCapabilities => DragonUiEventKind::Capability,
        DragonReadinessStepId::Transport => DragonUiEventKind::Transport,
        DragonReadinessStepId::DirectPeer => DragonUiEventKind::Peer,
        DragonReadinessStepId::Assignment => DragonUiEventKind::Assignment,
        DragonReadinessStepId::Checkpoint => DragonUiEventKind::Checkpoint,
        DragonReadinessStepId::TrainingReady => DragonUiEventKind::Training,
    }
}

pub(super) fn dragon_ui_now_ms() -> f64 {
    #[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
    {
        js_sys::Date::now()
    }
    #[cfg(not(all(feature = "wasm-ui", target_arch = "wasm32")))]
    {
        0.0
    }
}

pub(super) fn dragon_browser_training_action_ready(
    view: Option<&BrowserAppClientView>,
    direct_transport_ready: bool,
    requires_active_head_artifact: bool,
) -> bool {
    let Some(view) = view else {
        return false;
    };
    if !direct_transport_ready || !view.training.can_train {
        return false;
    }
    if view.runtime_label.starts_with("joining ") || view.runtime_label == "blocked" {
        return false;
    }
    view.training.active_assignment.is_some()
        && view.training.latest_head_id.is_some()
        && (!requires_active_head_artifact || view.training.active_head_artifact_ready)
}

pub(super) fn dragon_window_summary(
    view: Option<&BrowserAppClientView>,
    local_training_pending: bool,
) -> String {
    if local_training_pending {
        return "running".into();
    }
    let Some(view) = view else {
        return "pending".into();
    };
    match (
        view.training.last_window_secs,
        view.training.max_window_secs,
    ) {
        (Some(last), Some(max)) => format!("{last}s of {max}s"),
        (Some(last), None) => format!("{last}s last"),
        (None, Some(max)) => format!("{max}s max"),
        (None, None) => "waiting".into(),
    }
}

fn parse_leading_rate_per_second(summary: &str) -> Option<f64> {
    summary.split_whitespace().next()?.parse::<f64>().ok()
}

fn format_compact_duration(seconds: u64) -> String {
    match seconds {
        0 => "<1s".into(),
        1..=59 => format!("{seconds}s"),
        60..=3599 => {
            let minutes = seconds / 60;
            let remainder = seconds % 60;
            if remainder == 0 {
                format!("{minutes}m")
            } else {
                format!("{minutes}m {remainder}s")
            }
        }
        _ => {
            let hours = seconds / 3600;
            let remainder = seconds % 3600;
            let minutes = remainder / 60;
            if minutes == 0 {
                format!("{hours}h")
            } else {
                format!("{hours}h {minutes}m")
            }
        }
    }
}

fn dragon_window_eta_summary(view: Option<&BrowserAppClientView>) -> Option<String> {
    let view = view?;
    let remaining = view.training.slice_remaining_samples?;
    let throughput = view.training.throughput_summary.as_deref()?;
    let rate = parse_leading_rate_per_second(throughput)?;
    if !rate.is_finite() || rate <= 0.0 {
        return None;
    }
    let eta_seconds = ((remaining as f64) / rate).ceil() as u64;
    Some(format_compact_duration(eta_seconds))
}

pub(super) fn dragon_slice_progress_summary(view: Option<&BrowserAppClientView>) -> String {
    let Some(view) = view else {
        return "waiting".into();
    };
    if view.training.active_assignment.is_none() {
        return "waiting for work".into();
    }
    if view.training.latest_head_id.is_none() {
        return "syncing checkpoint".into();
    }
    if view.training.cached_microshards == 0 {
        return "loads on run".into();
    }
    match (
        view.training.accepted_samples,
        view.training.slice_target_samples,
        view.training.slice_remaining_samples,
    ) {
        (Some(done), Some(target), Some(remaining)) => {
            format!("{done}/{target} · {remaining} left")
        }
        (Some(done), Some(target), None) => format!("{done}/{target}"),
        _ => {
            if let Some(max_window_secs) = view.training.max_window_secs {
                format!("{max_window_secs}s max")
            } else {
                view.training.slice_status.clone()
            }
        }
    }
}

pub(super) fn dragon_transport_summary(view: Option<&BrowserAppClientView>) -> String {
    let Some(view) = view else {
        return "offline".into();
    };
    let transport = dragon_transport_target_label(view);
    if transport.is_empty() {
        return "offline".into();
    }
    if view.network.direct_peers > 0 {
        let peer_label = if view.network.direct_peers == 1 {
            "peer"
        } else {
            "peers"
        };
        return format!("{transport} · {} {peer_label}", view.network.direct_peers);
    }
    if view.network.swarm_status.connected_transport.is_none()
        && view.network.swarm_status.desired_transport.is_some()
    {
        if active_direct_transport_error(view).is_some() {
            return format!("{transport} failed");
        }
        return format!("{transport} pending");
    }
    transport.to_owned()
}

fn dragon_transport_target_label(view: &BrowserAppClientView) -> String {
    if let Some(connected) = view.network.swarm_status.connected_transport.as_ref() {
        return browser_transport_kind(connected).label().into();
    }
    if let Some(desired) = view.network.swarm_status.desired_transport.as_ref() {
        return browser_transport_kind(desired).label().into();
    }
    let fallback = view.network.transport.trim();
    if fallback.is_empty() {
        "offline".into()
    } else {
        fallback.to_owned()
    }
}

pub(super) fn dragon_network_detail(view: Option<&BrowserAppClientView>) -> String {
    let Some(view) = view else {
        return "edge snapshot only".into();
    };
    if view.network.direct_peers > 0 {
        if view.network.estimated_network_size > view.network.direct_peers {
            return format!(
                "{} direct · ~{} recently seen",
                view.network.direct_peers, view.network.estimated_network_size
            );
        }
        let peer_label = if view.network.direct_peers == 1 {
            "direct peer"
        } else {
            "direct peers"
        };
        return format!("{} {peer_label}", view.network.direct_peers);
    }
    if view.network.swarm_status.connected_transport.is_none()
        && view.network.swarm_status.desired_transport.is_some()
    {
        if active_direct_transport_error(view).is_some() {
            return "connection issue".into();
        }
        return "connecting".into();
    }
    if view.network.estimated_network_size > 0 {
        return format!(
            "~{} recently seen from the current network view",
            view.network.estimated_network_size
        );
    }
    "edge snapshot only".into()
}

pub(super) fn dragon_window_progress_detail(
    view: Option<&BrowserAppClientView>,
    window_summary: &str,
) -> String {
    let Some(view) = view else {
        return window_summary.into();
    };
    match (
        view.training.slice_remaining_samples,
        view.training.slice_target_samples,
    ) {
        (Some(remaining), Some(_target)) => {
            if let Some(eta) = dragon_window_eta_summary(Some(view)) {
                format!("{remaining} left · eta {eta}")
            } else {
                format!("{remaining} left · {window_summary}")
            }
        }
        (Some(remaining), None) => {
            if let Some(eta) = dragon_window_eta_summary(Some(view)) {
                format!("{remaining} left · eta {eta}")
            } else {
                format!("{remaining} left")
            }
        }
        _ => {
            if let Some(max_window_secs) = view.training.max_window_secs {
                format!("window cap {max_window_secs}s")
            } else {
                window_summary.into()
            }
        }
    }
}

pub(super) fn dragon_local_training_summary(
    view: Option<&BrowserAppClientView>,
    local_training_pending: bool,
) -> String {
    if local_training_pending {
        return "training…".into();
    }
    let Some(view) = view else {
        return "waiting".into();
    };
    if let Some(summary) = view.training.throughput_summary.clone() {
        return summary;
    }
    if view.training.last_window_secs.is_some() {
        return dragon_window_summary(Some(view), false);
    }
    if view.training.can_train {
        "idle".into()
    } else {
        "watching".into()
    }
}

pub(super) fn dragon_local_training_detail(
    view: Option<&BrowserAppClientView>,
    training_action_state: Option<&DragonTrainingActionState>,
) -> String {
    let Some(view) = view else {
        return "no active browser runtime".into();
    };
    if let Some(loss) = view.training.last_loss.as_ref() {
        if view.training.last_window_secs.is_some() || view.training.max_window_secs.is_some() {
            return format!(
                "loss {loss} · last window {}",
                dragon_window_summary(Some(view), false)
            );
        }
        return format!("loss {loss}");
    }
    if view.runtime_label.starts_with("joining ") || view.runtime_label.starts_with("catchup ") {
        let detail = view.runtime_detail.trim();
        if !detail.is_empty() {
            return detail.to_owned();
        }
    }
    training_action_state
        .map(|state| state.detail.clone())
        .unwrap_or_else(|| "waiting for the browser runtime".into())
}

pub(super) fn dragon_global_training_summary(view: Option<&BrowserAppClientView>) -> String {
    view.and_then(|view| {
        view.network
            .performance
            .as_ref()
            .map(|performance| performance.training_throughput.clone())
    })
    .unwrap_or_else(|| "waiting".into())
}

pub(super) fn dragon_global_training_detail(view: Option<&BrowserAppClientView>) -> String {
    let Some(performance) = view.and_then(|view| view.network.performance.as_ref()) else {
        return "network throughput has not been observed yet".into();
    };
    format!("validation {}", performance.validation_throughput)
}

pub(super) fn dragon_runtime_mode_summary(
    view: Option<&BrowserAppClientView>,
    direct_transport_ready: bool,
    training_action_state: Option<&DragonTrainingActionState>,
    auth_bootstrap_pending_active: bool,
    local_training_pending: bool,
) -> String {
    let Some(view) = view else {
        return if auth_bootstrap_pending_active {
            "bootstrapping".into()
        } else {
            "idle".into()
        };
    };
    if local_training_pending {
        return "training now".into();
    }
    if view.runtime_label.starts_with("joining ") || view.runtime_label.starts_with("catchup ") {
        return "syncing".into();
    }
    if view.runtime_label == "blocked" {
        return "blocked".into();
    }
    if training_action_state.is_some_and(|state| state.enabled) {
        return "ready to train".into();
    }
    match view.runtime_label.as_str() {
        "observe" => {
            if direct_transport_ready {
                "watching".into()
            } else {
                "connecting".into()
            }
        }
        "validate" => "validating".into(),
        "portal" => "portal only".into(),
        "train" => "training path".into(),
        _ => view.runtime_label.clone(),
    }
}

pub(super) fn dragon_runtime_mode_detail(
    view: Option<&BrowserAppClientView>,
    direct_transport_ready: bool,
    training_action_state: Option<&DragonTrainingActionState>,
    local_training_pending: bool,
    downgrade_reason: Option<&str>,
) -> String {
    let Some(view) = view else {
        return "browser runtime not connected".into();
    };
    if local_training_pending {
        return "running a local browser training window in this tab".into();
    }
    if view.runtime_label == "observe" {
        if let Some(reason) = downgrade_reason {
            return reason.to_owned();
        }
        return if direct_transport_ready {
            "watching network state. training turns on when trainer work is available.".into()
        } else {
            "waiting for a direct peer connection".into()
        };
    }
    if training_action_state.is_some_and(|state| state.enabled) {
        return "direct peers connected and checkpoint synced. run training when ready.".into();
    }
    if view.runtime_label == "blocked" {
        return downgrade_reason
            .map(str::to_owned)
            .unwrap_or_else(|| view.runtime_detail.clone());
    }
    if matches!(view.runtime_label.as_str(), "train" | "validate") {
        return view.runtime_detail.clone();
    }
    if view.runtime_label.starts_with("joining ") {
        let detail = view.runtime_detail.trim();
        return if detail.is_empty() {
            "connecting to the peer network".into()
        } else {
            detail.to_owned()
        };
    }
    if view.runtime_label.starts_with("catchup ") {
        let detail = view.runtime_detail.trim();
        return if detail.is_empty() {
            "syncing the current checkpoint".into()
        } else {
            detail.to_owned()
        };
    }
    training_action_state
        .map(|state| state.detail.clone())
        .unwrap_or_else(|| view.runtime_detail.clone())
}
