#[cfg(feature = "native")]
use anyhow::{Context, Result, anyhow, bail};
#[cfg(feature = "native")]
use chrono::{DateTime, Utc};
#[cfg(feature = "native")]
use reqwest::Client;
#[cfg(feature = "native")]
use serde::Serialize;
#[cfg(feature = "native")]
use url::Url;

#[cfg(feature = "native")]
use crate::auth::fetch_edge_snapshot;
#[cfg(feature = "native")]
use crate::capability::DragonNativeCapabilityAssessment;
#[cfg(feature = "native")]
use crate::config::{DragonExperimentKind, DragonManifestSeed, DragonNativePeerConfig};
#[cfg(feature = "native")]
use crate::native::assess_native_peer;
#[cfg(feature = "native")]
use crate::profile::{
    DragonResolvedProfileSource, find_matching_entry, resolve_native_training_profile,
};

#[cfg(feature = "native")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeploymentDiagnosticsOptions {
    pub check_metrics_catchup: bool,
    pub check_auth_authorize: bool,
    pub check_artifact_head_view: bool,
    pub require_head_published: bool,
    pub require_directory_entry_published: bool,
    pub require_metrics_catchup: bool,
    pub require_auth_authorize: bool,
    pub require_artifact_head_view: bool,
}

#[cfg(feature = "native")]
#[derive(Debug, Serialize)]
pub struct DeploymentDiagnosticsReport {
    pub captured_at: DateTime<Utc>,
    pub config: DeploymentConfigSummary,
    pub capability: DeploymentCheck<DragonNativeCapabilityAssessment>,
    pub edge_snapshot: DeploymentCheck<DeploymentEdgeSnapshotSummary>,
    pub profile_resolution: DeploymentCheck<DeploymentProfileResolutionSummary>,
    pub metrics_catchup: Option<DeploymentCheck<DeploymentHttpProbe>>,
    pub auth_authorize: Option<DeploymentCheck<DeploymentAuthorizeProbe>>,
    pub artifact_head_view: Option<DeploymentCheck<DeploymentArtifactHeadProbe>>,
    pub readiness: DeploymentReadinessReport,
}

#[cfg(feature = "native")]
#[derive(Debug, Serialize)]
pub struct DeploymentConfigSummary {
    pub experiment_kind: DragonExperimentKind,
    pub backend: String,
    pub edge_base_url: Option<String>,
    pub seed_node_urls: Vec<String>,
    pub storage_root: std::path::PathBuf,
    pub has_local_training_config: bool,
    pub training_config_paths: Vec<std::path::PathBuf>,
    pub manifest: DragonManifestSeed,
}

#[cfg(feature = "native")]
#[derive(Debug, Serialize)]
pub struct DeploymentEdgeSnapshotSummary {
    pub network_id: String,
    pub auth_enabled: bool,
    pub login_providers: Vec<DeploymentLoginProviderSummary>,
    pub directory_entries: usize,
    pub matching_directory_entry_present: bool,
    pub matching_head_present: bool,
    pub matching_head_id: Option<String>,
    pub captured_at: DateTime<Utc>,
}

#[cfg(feature = "native")]
#[derive(Debug, Serialize)]
pub struct DeploymentLoginProviderSummary {
    pub label: String,
    pub login_path: String,
    pub callback_path: Option<String>,
    pub device_path: Option<String>,
}

#[cfg(feature = "native")]
#[derive(Debug, Serialize)]
pub struct DeploymentProfileResolutionSummary {
    pub source: DragonResolvedProfileSource,
    pub manifest_seed: DragonManifestSeed,
    pub has_directory_entry: bool,
    pub block_size: usize,
    pub batch_size: usize,
}

#[cfg(feature = "native")]
#[derive(Debug, Serialize)]
pub struct DeploymentHttpProbe {
    pub url: String,
    pub status_code: u16,
    pub body_preview: Option<String>,
}

#[cfg(feature = "native")]
#[derive(Debug, Serialize)]
pub struct DeploymentAuthorizeProbe {
    pub provider_label: String,
    pub login_path: String,
    pub redirect_uri: Option<String>,
    pub missing_query_params: Vec<String>,
}

#[cfg(feature = "native")]
#[derive(Debug, Serialize)]
pub struct DeploymentArtifactHeadProbe {
    pub url: String,
    pub status_code: u16,
    pub head_id: String,
}

#[cfg(feature = "native")]
#[derive(Debug, Serialize)]
pub struct DeploymentReadinessReport {
    pub ready: bool,
    pub blocking_issues: Vec<String>,
    pub observed_warnings: Vec<String>,
}

#[cfg(feature = "native")]
#[derive(Debug, Serialize)]
pub struct DeploymentCheck<T> {
    pub ok: bool,
    pub value: Option<T>,
    pub error: Option<String>,
}

#[cfg(feature = "native")]
pub fn collect_deployment_diagnostics(
    config: &DragonNativePeerConfig,
    experiment_kind: DragonExperimentKind,
    backend: &str,
    options: DeploymentDiagnosticsOptions,
) -> DeploymentDiagnosticsReport {
    let config_summary = DeploymentConfigSummary {
        experiment_kind,
        backend: backend.into(),
        edge_base_url: config.effective_edge_base_url().map(ToOwned::to_owned),
        seed_node_urls: config.effective_seed_node_urls(),
        storage_root: config.storage_root.clone(),
        has_local_training_config: !config.training_config_paths.is_empty(),
        training_config_paths: config.training_config_paths.clone(),
        manifest: config.manifest.clone(),
    };

    let capability = match assess_native_peer(config, experiment_kind, backend) {
        Ok(value) => DeploymentCheck::ok(value),
        Err(error) => DeploymentCheck::err(error),
    };

    let edge_snapshot = match fetch_deployment_edge_snapshot(config, experiment_kind) {
        Ok(value) => DeploymentCheck::ok(value),
        Err(error) => DeploymentCheck::err(error),
    };

    let profile_resolution = match resolve_native_training_profile(config, experiment_kind, true) {
        Ok(resolved) => DeploymentCheck::ok(DeploymentProfileResolutionSummary {
            source: resolved.source,
            manifest_seed: resolved.manifest_seed,
            has_directory_entry: resolved.directory_entry.is_some(),
            block_size: resolved.config.training.block_size,
            batch_size: resolved.config.training.batch_size,
        }),
        Err(error) => DeploymentCheck::err(error),
    };

    let metrics_catchup = if options.check_metrics_catchup {
        Some(match probe_metrics_catchup(config, experiment_kind) {
            Ok(value) => DeploymentCheck::ok(value),
            Err(error) => DeploymentCheck::err(error),
        })
    } else {
        None
    };

    let auth_authorize = if options.check_auth_authorize {
        Some(match probe_auth_authorize(config) {
            Ok(value) => DeploymentCheck::ok(value),
            Err(error) => DeploymentCheck::err(error),
        })
    } else {
        None
    };

    let artifact_head_view = if options.check_artifact_head_view {
        Some(
            match probe_artifact_head_view(config, edge_snapshot.value.as_ref()) {
                Ok(value) => DeploymentCheck::ok(value),
                Err(error) => DeploymentCheck::err(error),
            },
        )
    } else {
        None
    };

    let readiness = evaluate_deployment_readiness(
        &capability,
        &edge_snapshot,
        &profile_resolution,
        metrics_catchup.as_ref(),
        auth_authorize.as_ref(),
        artifact_head_view.as_ref(),
        options.require_head_published,
        options.require_directory_entry_published,
        options.require_metrics_catchup,
        options.require_auth_authorize,
        options.require_artifact_head_view,
    );

    DeploymentDiagnosticsReport {
        captured_at: Utc::now(),
        config: config_summary,
        capability,
        edge_snapshot,
        profile_resolution,
        metrics_catchup,
        auth_authorize,
        artifact_head_view,
        readiness,
    }
}

#[cfg(feature = "native")]
pub fn assert_deployment_ready(report: &DeploymentDiagnosticsReport) -> Result<()> {
    if report.readiness.ready {
        return Ok(());
    }
    bail!(
        "deployment readiness failed: {}",
        report.readiness.blocking_issues.join(", ")
    )
}

#[cfg(feature = "native")]
fn fetch_deployment_edge_snapshot(
    config: &DragonNativePeerConfig,
    experiment_kind: DragonExperimentKind,
) -> Result<DeploymentEdgeSnapshotSummary> {
    let edge_base_url = config
        .effective_edge_base_url()
        .ok_or_else(|| anyhow!("no edge base URL configured"))?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to build async runtime for deployment diagnostics")?;
    let snapshot = runtime.block_on(fetch_edge_snapshot(edge_base_url))?;
    let matching_directory_entry = find_matching_entry(
        &snapshot.directory.entries,
        Some(&config.manifest.experiment_id),
        Some(&config.manifest.revision_id),
        Some(experiment_kind),
    )?;
    let matching_head = snapshot
        .heads
        .iter()
        .find(|head| head.experiment_id.as_str() == config.manifest.experiment_id);
    Ok(DeploymentEdgeSnapshotSummary {
        network_id: snapshot.network_id.as_str().to_owned(),
        auth_enabled: snapshot.auth_enabled,
        login_providers: snapshot
            .login_providers
            .into_iter()
            .map(|provider| DeploymentLoginProviderSummary {
                label: provider.label,
                login_path: provider.login_path,
                callback_path: provider.callback_path,
                device_path: provider.device_path,
            })
            .collect(),
        directory_entries: snapshot.directory.entries.len(),
        matching_directory_entry_present: matching_directory_entry.is_some(),
        matching_head_present: matching_head.is_some(),
        matching_head_id: matching_head.map(|head| head.head_id.as_str().to_owned()),
        captured_at: snapshot.captured_at,
    })
}

#[cfg(feature = "native")]
fn probe_metrics_catchup(
    config: &DragonNativePeerConfig,
    experiment_kind: DragonExperimentKind,
) -> Result<DeploymentHttpProbe> {
    let edge_base_url = config
        .effective_edge_base_url()
        .ok_or_else(|| anyhow!("no edge base URL configured"))?;
    let url = format!(
        "{}/metrics/catchup/{}",
        edge_base_url.trim_end_matches('/'),
        experiment_kind.workload_slug()
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to build async runtime for metrics probe")?;
    runtime.block_on(async {
        let response = Client::new().get(&url).send().await?;
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            bail!("{} {}", status.as_u16(), trim_preview(&body));
        }
        Ok(DeploymentHttpProbe {
            url,
            status_code: status.as_u16(),
            body_preview: (!body.trim().is_empty()).then(|| trim_preview(&body)),
        })
    })
}

#[cfg(feature = "native")]
fn probe_auth_authorize(config: &DragonNativePeerConfig) -> Result<DeploymentAuthorizeProbe> {
    let edge_base_url = config
        .effective_edge_base_url()
        .ok_or_else(|| anyhow!("no edge base URL configured"))?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to build async runtime for auth authorize probe")?;
    runtime.block_on(async {
        let snapshot = fetch_edge_snapshot(edge_base_url).await?;
        let provider = snapshot
            .login_providers
            .into_iter()
            .find(|provider| !provider.login_path.trim().is_empty())
            .ok_or_else(|| anyhow!("edge snapshot has no login providers"))?;
        let login_url = format!(
            "{}{}",
            edge_base_url.trim_end_matches('/'),
            provider.login_path
        );
        let payload = serde_json::json!({
            "network_id": snapshot.network_id.as_str(),
            "principal_hint": null,
            "requested_scopes": ["Connect"],
        });
        let response = Client::new().post(&login_url).json(&payload).send().await?;
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            bail!("{} {}", status.as_u16(), trim_preview(&body));
        }
        let payload: serde_json::Value =
            serde_json::from_str(&body).context("failed to decode auth login response")?;
        let authorize_url = payload
            .get("authorize_url")
            .and_then(|value| value.as_str())
            .ok_or_else(|| anyhow!("login response is missing authorize_url"))?;
        let parsed = Url::parse(authorize_url)
            .with_context(|| format!("invalid authorize_url: {authorize_url}"))?;
        let query = parsed.query_pairs().into_owned().collect::<Vec<_>>();
        let missing_query_params = ["client_id", "redirect_uri", "state"]
            .into_iter()
            .filter(|key| {
                !query
                    .iter()
                    .any(|(candidate, value)| candidate == key && !value.is_empty())
            })
            .map(str::to_owned)
            .collect::<Vec<_>>();
        Ok(DeploymentAuthorizeProbe {
            provider_label: provider.label,
            login_path: provider.login_path,
            redirect_uri: query
                .iter()
                .find(|(key, _)| key == "redirect_uri")
                .map(|(_, value)| value.clone()),
            missing_query_params,
        })
    })
}

#[cfg(feature = "native")]
fn probe_artifact_head_view(
    config: &DragonNativePeerConfig,
    edge_snapshot: Option<&DeploymentEdgeSnapshotSummary>,
) -> Result<DeploymentArtifactHeadProbe> {
    let edge_base_url = config
        .effective_edge_base_url()
        .ok_or_else(|| anyhow!("no edge base URL configured"))?;
    let head_id = edge_snapshot
        .and_then(|snapshot| snapshot.matching_head_id.clone())
        .ok_or_else(|| anyhow!("edge snapshot does not expose a matching head id"))?;
    let url = format!(
        "{}/artifacts/heads/{}",
        edge_base_url.trim_end_matches('/'),
        head_id
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to build async runtime for head artifact probe")?;
    runtime.block_on(async {
        let response = Client::new().get(&url).send().await?;
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            bail!("{} {}", status.as_u16(), trim_preview(&body));
        }
        serde_json::from_str::<serde_json::Value>(&body)
            .context("failed to decode artifact head response")?;
        Ok(DeploymentArtifactHeadProbe {
            url,
            status_code: status.as_u16(),
            head_id,
        })
    })
}

pub fn evaluate_deployment_readiness(
    capability: &DeploymentCheck<DragonNativeCapabilityAssessment>,
    edge_snapshot: &DeploymentCheck<DeploymentEdgeSnapshotSummary>,
    profile_resolution: &DeploymentCheck<DeploymentProfileResolutionSummary>,
    metrics_catchup: Option<&DeploymentCheck<DeploymentHttpProbe>>,
    auth_authorize: Option<&DeploymentCheck<DeploymentAuthorizeProbe>>,
    artifact_head_view: Option<&DeploymentCheck<DeploymentArtifactHeadProbe>>,
    require_head_published: bool,
    require_directory_entry_published: bool,
    require_metrics_catchup: bool,
    require_auth_authorize: bool,
    require_artifact_head_view: bool,
) -> DeploymentReadinessReport {
    let mut blocking_issues = Vec::new();
    let mut observed_warnings = Vec::new();

    if !capability.ok {
        blocking_issues.push("native_capability_assessment_failed".into());
    } else if !capability
        .value
        .as_ref()
        .is_some_and(|assessment| assessment.target_decision.can_train)
    {
        blocking_issues.push("native_capability_cannot_train".into());
    }

    if !edge_snapshot.ok {
        blocking_issues.push("edge_snapshot_unavailable".into());
    } else if let Some(snapshot) = edge_snapshot.value.as_ref() {
        if !snapshot.matching_directory_entry_present {
            if require_directory_entry_published {
                blocking_issues.push("matching_directory_entry_missing".into());
            } else {
                observed_warnings.push("matching_directory_entry_missing".into());
            }
        }
        if !snapshot.matching_head_present {
            if require_head_published {
                blocking_issues.push("matching_experiment_head_missing".into());
            } else {
                observed_warnings.push("matching_experiment_head_missing".into());
            }
        }
    }

    if !profile_resolution.ok {
        blocking_issues.push("training_profile_resolution_failed".into());
    }

    if let Some(metrics_catchup) = metrics_catchup {
        if !metrics_catchup.ok && require_metrics_catchup {
            blocking_issues.push("metrics_catchup_probe_failed".into());
        } else if !metrics_catchup.ok {
            observed_warnings.push("metrics_catchup_probe_failed".into());
        }
    }

    if let Some(auth_authorize) = auth_authorize {
        if !auth_authorize.ok {
            if require_auth_authorize {
                blocking_issues.push("auth_authorize_probe_failed".into());
            } else {
                observed_warnings.push("auth_authorize_probe_failed".into());
            }
        } else if auth_authorize
            .value
            .as_ref()
            .is_some_and(|probe| !probe.missing_query_params.is_empty())
        {
            if require_auth_authorize {
                blocking_issues.push("auth_authorize_query_incomplete".into());
            } else {
                observed_warnings.push("auth_authorize_query_incomplete".into());
            }
        }
    }

    if let Some(artifact_head_view) = artifact_head_view {
        if !artifact_head_view.ok {
            if require_artifact_head_view {
                blocking_issues.push("artifact_head_view_probe_failed".into());
            } else {
                observed_warnings.push("artifact_head_view_probe_failed".into());
            }
        }
    }

    DeploymentReadinessReport {
        ready: blocking_issues.is_empty(),
        blocking_issues,
        observed_warnings,
    }
}

#[cfg(feature = "native")]
fn trim_preview(body: &str) -> String {
    const LIMIT: usize = 240;
    let trimmed = body.trim();
    if trimmed.len() <= LIMIT {
        trimmed.to_owned()
    } else {
        format!("{}...", &trimmed[..LIMIT])
    }
}

#[cfg(feature = "native")]
impl<T> DeploymentCheck<T> {
    pub fn ok(value: T) -> Self {
        Self {
            ok: true,
            value: Some(value),
            error: None,
        }
    }

    pub fn err(error: anyhow::Error) -> Self {
        Self {
            ok: false,
            value: None,
            error: Some(format!("{error:#}")),
        }
    }
}

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;
    use crate::capability::{DragonNativeTargetDecision, DragonTrainingFootprint};
    use crate::config::DragonNativeTarget;
    use burn_dragon_language::DragonConfig;

    fn capability_check(can_train: bool) -> DeploymentCheck<DragonNativeCapabilityAssessment> {
        DeploymentCheck::ok(DragonNativeCapabilityAssessment {
            experiment_kind: DragonExperimentKind::NcaPrepretraining,
            backend_label: "cpu".into(),
            model_config: DragonConfig::default(),
            batch_size: 4,
            block_size: 128,
            footprint: DragonTrainingFootprint {
                estimated_parameter_bytes: 1,
                estimated_optimizer_state_bytes: 1,
                estimated_activation_bytes: 1,
                estimated_training_bytes: 1,
                estimated_checkpoint_bytes: 1,
                estimated_shard_bytes: 1,
                estimated_tokens_per_second: 1.0,
            },
            target_decision: DragonNativeTargetDecision {
                requested_target: DragonNativeTarget::Trainer,
                effective_target: DragonNativeTarget::Trainer,
                can_train,
                trainer_memory_budget_bytes: Some(1),
                downgrade_reason: None,
            },
        })
    }

    fn edge_check(head_present: bool) -> DeploymentCheck<DeploymentEdgeSnapshotSummary> {
        DeploymentCheck::ok(DeploymentEdgeSnapshotSummary {
            network_id: "burn-dragon-mainnet".into(),
            auth_enabled: true,
            login_providers: vec![DeploymentLoginProviderSummary {
                label: "github".into(),
                login_path: "/login/github".into(),
                callback_path: Some("/callback/github".into()),
                device_path: None,
            }],
            directory_entries: 1,
            matching_directory_entry_present: true,
            matching_head_present: head_present,
            matching_head_id: head_present.then(|| "head-1".into()),
            captured_at: Utc::now(),
        })
    }

    fn profile_check() -> DeploymentCheck<DeploymentProfileResolutionSummary> {
        DeploymentCheck::ok(DeploymentProfileResolutionSummary {
            source: DragonResolvedProfileSource::BuiltinFallback,
            manifest_seed: DragonManifestSeed::default(),
            has_directory_entry: false,
            block_size: 128,
            batch_size: 4,
        })
    }

    #[test]
    fn deployment_readiness_requires_head_when_requested() {
        let readiness = evaluate_deployment_readiness(
            &capability_check(true),
            &edge_check(false),
            &profile_check(),
            None,
            None,
            None,
            true,
            false,
            false,
            false,
            false,
        );

        assert!(!readiness.ready);
        assert!(
            readiness
                .blocking_issues
                .contains(&"matching_experiment_head_missing".to_owned())
        );
    }

    #[test]
    fn deployment_readiness_treats_missing_head_as_warning_when_not_required() {
        let readiness = evaluate_deployment_readiness(
            &capability_check(true),
            &edge_check(false),
            &profile_check(),
            None,
            None,
            None,
            false,
            false,
            false,
            false,
            false,
        );

        assert!(readiness.ready);
        assert!(
            readiness
                .observed_warnings
                .contains(&"matching_experiment_head_missing".to_owned())
        );
    }

    #[test]
    fn deployment_readiness_blocks_on_incomplete_auth_query_when_required() {
        let readiness = evaluate_deployment_readiness(
            &capability_check(true),
            &edge_check(true),
            &profile_check(),
            None,
            Some(&DeploymentCheck::ok(DeploymentAuthorizeProbe {
                provider_label: "github".into(),
                login_path: "/login/github".into(),
                redirect_uri: None,
                missing_query_params: vec!["redirect_uri".into()],
            })),
            None,
            false,
            false,
            false,
            true,
            false,
        );

        assert!(!readiness.ready);
        assert!(
            readiness
                .blocking_issues
                .contains(&"auth_authorize_query_incomplete".to_owned())
        );
    }

    #[test]
    fn deployment_readiness_requires_directory_entry_when_requested() {
        let mut edge = edge_check(true);
        if let Some(snapshot) = edge.value.as_mut() {
            snapshot.matching_directory_entry_present = false;
            snapshot.directory_entries = 0;
        }
        let readiness = evaluate_deployment_readiness(
            &capability_check(true),
            &edge,
            &profile_check(),
            None,
            None,
            None,
            true,
            true,
            false,
            false,
            false,
        );

        assert!(!readiness.ready);
        assert!(
            readiness
                .blocking_issues
                .contains(&"matching_directory_entry_missing".to_owned())
        );
    }

    #[test]
    fn deployment_readiness_requires_artifact_head_view_when_requested() {
        let readiness = evaluate_deployment_readiness(
            &capability_check(true),
            &edge_check(true),
            &profile_check(),
            None,
            None,
            Some(&DeploymentCheck::err(anyhow!("502 bad gateway"))),
            true,
            true,
            false,
            false,
            true,
        );

        assert!(!readiness.ready);
        assert!(
            readiness
                .blocking_issues
                .contains(&"artifact_head_view_probe_failed".to_owned())
        );
    }
}
