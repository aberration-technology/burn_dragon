use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, ensure};
use burn_p2p::{
    BrowserEdgeSnapshot, ClientPlatform, ClientReleaseManifest, ContentId, ExperimentId,
    ExperimentScope, ProjectFamilyId,
};
use burn_p2p_core::{BrowserSeedAdvertisement, SchemaEnvelope, SignedPayload};
use clap::{ArgAction, Args};
use serde::Serialize;
use wasm_bindgen_cli_support::Bindgen;

const BROWSER_BIN: &str = "burn_dragon_p2p_browser";
const DEFAULT_OUT_DIR: &str = "target/xtask/browser-site";
const EDGE_FETCH_MAX_ATTEMPTS: usize = 5;
const BROWSER_APP_LOADER: &str = r#"import init from "./burn_dragon_p2p_browser.js";

await init({ module_or_path: new URL("./burn_dragon_p2p_browser_bg.wasm", import.meta.url) });
"#;
const INDEX_HTML_TEMPLATE: &str = r##"<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="color-scheme" content="dark" />
    <meta name="theme-color" content="#000000" />
    <title>burn_dragon</title>
    <link rel="stylesheet" href="__ASSET_PREFIX__/browser-app.css" />
  </head>
  <body>
    <div id="main"></div>
    <script type="module" src="__ASSET_PREFIX__/browser-app-loader.js"></script>
  </body>
</html>
"##;
const EXTRA_STYLESHEET: &str = r#"
:root {
  --bg: #000000;
  --bg-elevated: #050505;
  --panel: rgba(8, 8, 8, 0.9);
  --panel-strong: rgba(12, 12, 12, 0.96);
  --panel-soft: rgba(10, 10, 10, 0.78);
  --ink: #ffffff;
  --muted: rgba(255, 255, 255, 0.72);
  --line: rgba(255, 255, 255, 0.12);
  --line-strong: rgba(216, 124, 124, 0.26);
  --accent: #d87c7c;
  --accent-strong: #c96b6b;
  --accent-soft: rgba(216, 124, 124, 0.14);
  --accent-cool: #d87c7c;
  --shadow: 0 28px 80px rgba(0, 0, 0, 0.42);
  color-scheme: dark;
  font-family: "Iosevka Etoile", "IBM Plex Sans", system-ui, sans-serif;
}

html {
  background: var(--bg);
}

body {
  background:
    radial-gradient(circle at top center, rgba(216, 124, 124, 0.12), transparent 28%),
    linear-gradient(180deg, #000000, #030303 35%, #000000 100%);
}

.dragon-browser-shell {
  gap: 22px;
}

.dragon-browser-shell .browser-hero {
  gap: 18px;
  padding: 26px 26px 22px;
}

.hero {
  background:
    radial-gradient(circle at top left, rgba(216, 124, 124, 0.16), transparent 26%),
    radial-gradient(circle at 88% 0%, rgba(216, 124, 124, 0.08), transparent 24%),
    linear-gradient(135deg, rgba(10, 10, 10, 0.98), rgba(6, 6, 6, 0.98));
  border-color: var(--line-strong);
}

.status-pill-accent,
.status-chip.accent,
.activity-notice-accent {
  border-color: rgba(216, 124, 124, 0.32);
  background: rgba(216, 124, 124, 0.1);
  color: var(--accent);
}

.surface-tab:hover {
  border-color: rgba(216, 124, 124, 0.3);
  background: rgba(216, 124, 124, 0.08);
}

.surface-tab.is-active {
  border-color: rgba(216, 124, 124, 0.52);
  background:
    linear-gradient(180deg, rgba(216, 124, 124, 0.14), rgba(201, 107, 107, 0.08)),
    rgba(255, 255, 255, 0.02);
}

.leaderboard-row.is-local,
.experiment-card-button.is-selected,
.directory-row.is-selected {
  border-color: rgba(216, 124, 124, 0.36);
}

.directory-row:hover {
  border-color: rgba(216, 124, 124, 0.26);
}

.experiment-card-button.is-selected {
  background: rgba(216, 124, 124, 0.08);
}

.directory-row.is-selected {
  background:
    linear-gradient(180deg, rgba(216, 124, 124, 0.06), rgba(255, 255, 255, 0.02)),
    rgba(255, 255, 255, 0.03);
}

.browser-spotlight {
  border-color: rgba(216, 124, 124, 0.22);
  background:
    radial-gradient(circle at top right, rgba(216, 124, 124, 0.08), transparent 24%),
    linear-gradient(145deg, rgba(10, 10, 10, 0.98), rgba(6, 6, 6, 0.98));
}

.action-button,
.pill {
  border-color: rgba(216, 124, 124, 0.3);
  background: rgba(216, 124, 124, 0.1);
}

.action-button-primary {
  border-color: rgba(216, 124, 124, 0.36);
  background: linear-gradient(180deg, rgba(216, 124, 124, 0.18), rgba(201, 107, 107, 0.12));
}

.dragon-site-footer {
  margin-top: auto;
  display: flex;
  justify-content: center;
  padding: 4px 0 2px;
}

.dragon-site-footer-links {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  gap: 1rem;
  font-family: ui-monospace, monospace;
}

.dragon-site-footer a {
  color: var(--accent);
  text-decoration: none;
  transition: color 0.2s ease;
}

.dragon-site-footer a:hover {
  color: var(--ink);
  text-decoration: underline;
}

.dragon-browser-shell .browser-hero-copy {
  max-width: 46rem;
}

.dragon-eyebrow-row {
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
  line-height: 1;
}

.dragon-eyebrow-row .eyebrow {
  display: inline-flex;
  align-items: center;
  margin-bottom: 0;
  line-height: 1;
}

.dragon-eyebrow-rattle {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 1.2ch;
  color: var(--accent);
  font-family: "Iosevka Etoile", "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.92em;
  line-height: 1;
  transform: translateY(-0.08em);
}

.dragon-browser-shell .browser-hero-bar {
  justify-content: flex-start;
  padding-top: 0;
  border-top: 0;
}

.dragon-browser-shell .browser-quick-card strong {
  line-height: 1.35;
}

.dragon-hero-actions {
  align-self: stretch;
}

.dragon-connection-editor {
  display: grid;
  gap: 12px;
  width: auto;
  max-width: 100%;
}

.dragon-advanced-settings {
  padding-top: 4px;
}

.dragon-metric-band {
  margin-top: 8px;
}

.dragon-landing-grid {
  display: grid;
  gap: 14px;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
}

.dragon-landing-card {
  display: grid;
  gap: 10px;
  padding: 18px;
  border-radius: 18px;
  border: 1px solid var(--line);
  background:
    radial-gradient(circle at top right, rgba(216, 124, 124, 0.08), transparent 30%),
    rgba(255, 255, 255, 0.025);
}

.dragon-panel-stack {
  display: grid;
  gap: 16px;
}

.dragon-live-shell-wrap {
  width: min(100%, 820px);
  margin: 0 auto;
}

.dragon-live-shell {
  gap: 18px;
}

.dragon-live-summary {
  gap: 18px;
}

.dragon-live-status-row {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  justify-content: flex-start;
}

.dragon-live-status-pill {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 2rem;
  padding: 0 0.9rem;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.03);
  color: var(--muted);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: lowercase;
}

.dragon-live-status-pill-syncing,
.dragon-live-status-pill-connecting,
.dragon-live-status-pill-training {
  border-color: rgba(216, 124, 124, 0.26);
  background: rgba(216, 124, 124, 0.08);
  color: var(--accent);
}

.dragon-live-status-pill-ready {
  border-color: rgba(255, 255, 255, 0.14);
  background: rgba(255, 255, 255, 0.04);
  color: var(--ink);
}

.dragon-live-stats {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
}

.dragon-live-stats .stat-tile {
  min-width: 0;
  min-height: 112px;
  justify-content: space-between;
  padding: 16px 18px;
  border-radius: 18px;
  border: 1px solid var(--line);
  background:
    radial-gradient(circle at top right, rgba(216, 124, 124, 0.06), transparent 32%),
    rgba(255, 255, 255, 0.025);
}

.dragon-live-stats .stat-tile span {
  color: var(--muted);
  letter-spacing: 0.08em;
  text-transform: lowercase;
}

.dragon-live-stats .stat-tile strong {
  font-size: 1.04rem;
  line-height: 1.22;
  overflow-wrap: anywhere;
  word-break: break-word;
}

.dragon-live-stats .stat-tile .stat-detail {
  overflow-wrap: anywhere;
  word-break: break-word;
}

.dragon-live-actions {
  display: grid;
  gap: 8px;
  padding-top: 2px;
  justify-items: start;
}

.dragon-live-actions .action-button[disabled] {
  opacity: 0.58;
  cursor: not-allowed;
}

.dragon-live-action-note {
  margin: 0;
  max-width: 34rem;
  color: var(--muted);
  line-height: 1.45;
}

.dragon-live-keyvalues {
  margin-top: 4px;
}

.dragon-live-keyvalues .keyvalue-row {
  align-items: flex-start;
}

.dragon-live-keyvalues .keyvalue-row strong {
  max-width: 64%;
  overflow-wrap: anywhere;
  word-break: break-word;
}

.dragon-live-shell .section-detail {
  max-width: 34rem;
}

.dragon-public-copy {
  max-width: 48rem;
}

.dragon-operator-summary {
  margin-top: 4px;
}

.dragon-admin-actions {
  margin-top: 4px;
}

.dragon-editor-grid {
  display: grid;
  gap: 16px;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
}

.dragon-editor-grid-wide {
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
}

.dragon-admin-surface,
.dragon-admin-gate {
  margin-top: 2px;
}

.dragon-editor-field {
  display: grid;
  gap: 8px;
}

.dragon-text-input,
.dragon-textarea {
  width: 100%;
  box-sizing: border-box;
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 12px 14px;
  font: inherit;
  color: var(--ink);
  background: rgba(255, 255, 255, 0.03);
}

.dragon-text-input::placeholder,
.dragon-textarea::placeholder {
  color: rgba(221, 213, 199, 0.55);
}

.dragon-textarea {
  min-height: 16rem;
  resize: vertical;
  font-family: "Iosevka Etoile", "IBM Plex Mono", monospace;
  line-height: 1.5;
}

.burn-dragon-p2p-bootstrap-loading,
.burn-dragon-p2p-bootstrap-error {
  max-width: 1180px;
  margin: 0 auto;
  padding: 28px clamp(18px, 4vw, 36px) 40px;
}

.burn-dragon-p2p-bootstrap-error pre {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  margin: 0;
  padding: 16px;
  border-radius: 16px;
  border: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.03);
}

@media (max-width: 960px) {
  .dragon-editor-grid-wide {
    grid-template-columns: 1fr;
  }
}
"#;

#[derive(Clone, Debug, Serialize)]
struct DragonPeerNetworkConfigDocument {
    #[serde(skip_serializing_if = "Option::is_none")]
    edge_base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed_node_urls: Option<Vec<String>>,
}

#[derive(Clone, Debug, Serialize)]
struct DragonBrowserAppConfigDocument {
    network: DragonPeerNetworkConfigDocument,
    #[serde(skip_serializing_if = "Option::is_none")]
    selected_experiment_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    selected_revision_id: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    requested_scopes: BTreeSet<ExperimentScope>,
    require_edge_auth: bool,
}

#[derive(Clone, Debug, Serialize)]
struct DragonBrowserSiteBootstrapDocument {
    config: DragonBrowserAppConfigDocument,
    #[serde(skip_serializing_if = "Option::is_none")]
    release_manifest: Option<ClientReleaseManifest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    edge_snapshot: Option<BrowserEdgeSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    signed_seed_advertisement: Option<SignedPayload<SchemaEnvelope<BrowserSeedAdvertisement>>>,
}

#[derive(Clone, Debug, Args)]
pub struct BuildBrowserSiteArgs {
    #[arg(long, default_value = DEFAULT_OUT_DIR)]
    pub out_dir: PathBuf,
    #[arg(long)]
    pub edge_url: Option<String>,
    #[arg(long = "seed-node-url", value_delimiter = ',', action = ArgAction::Append)]
    pub seed_node_urls: Vec<String>,
    #[arg(long)]
    pub selected_experiment_id: Option<String>,
    #[arg(long)]
    pub selected_revision_id: Option<String>,
    #[arg(
        long = "require-edge-auth",
        alias = "require-github-auth",
        default_value_t = true,
        action = ArgAction::Set
    )]
    pub require_edge_auth: bool,
}

impl Default for BuildBrowserSiteArgs {
    fn default() -> Self {
        Self {
            out_dir: PathBuf::from(DEFAULT_OUT_DIR),
            edge_url: None,
            seed_node_urls: Vec::new(),
            selected_experiment_id: None,
            selected_revision_id: None,
            require_edge_auth: true,
        }
    }
}

pub fn build_browser_site_default() -> Result<()> {
    build_browser_site(&BuildBrowserSiteArgs::default())
}

pub fn build_browser_site(args: &BuildBrowserSiteArgs) -> Result<()> {
    let out_dir = workspace_root().join(&args.out_dir);
    if out_dir.exists() {
        fs::remove_dir_all(&out_dir)
            .with_context(|| format!("failed to clear {}", out_dir.display()))?;
    }
    fs::create_dir_all(&out_dir)
        .with_context(|| format!("failed to create {}", out_dir.display()))?;

    build_browser_wasm_bundle(&out_dir)?;
    write_site_shell(&out_dir, args)?;
    Ok(())
}

fn build_browser_wasm_bundle(out_dir: &Path) -> Result<()> {
    let workspace_root = workspace_root();
    run_step(
        Command::new(cargo_bin())
            .current_dir(&workspace_root)
            .arg("build")
            .arg("--manifest-path")
            .arg(workspace_root.join("Cargo.toml"))
            .arg("-p")
            .arg("burn_dragon_p2p")
            .arg("--target")
            .arg("wasm32-unknown-unknown")
            .arg("--profile")
            .arg("wasm-release")
            .arg("--no-default-features")
            .arg("--features")
            .arg("wasm-ui,wasm-peer,wgpu")
            .arg("--bin")
            .arg(BROWSER_BIN),
        "cargo build burn_dragon_p2p browser wasm bundle",
    )?;

    let wasm_input = workspace_root
        .join("target/wasm32-unknown-unknown/wasm-release")
        .join(format!("{BROWSER_BIN}.wasm"));
    ensure!(
        wasm_input.exists(),
        "missing wasm output at {}",
        wasm_input.display()
    );

    let mut bindgen = Bindgen::new();
    bindgen.input_path(&wasm_input).out_name(BROWSER_BIN);
    bindgen
        .web(true)
        .context("configure wasm-bindgen web target")?;
    bindgen
        .generate(out_dir)
        .context("generate wasm-bindgen browser bundle")?;
    Ok(())
}

fn write_site_shell(out_dir: &Path, args: &BuildBrowserSiteArgs) -> Result<()> {
    write_html_page(out_dir, "index.html", ".")?;
    write_html_page(out_dir, "404.html", ".")?;
    write_html_page(out_dir, "callback/github/index.html", "../..")?;
    write_html_page(out_dir, "callback/oidc/index.html", "../..")?;
    write_html_page(out_dir, "callback/oauth/index.html", "../..")?;
    fs::write(out_dir.join(".nojekyll"), "")
        .with_context(|| format!("failed to write {}/.nojekyll", out_dir.display()))?;
    fs::write(out_dir.join("browser-app-loader.js"), BROWSER_APP_LOADER)
        .with_context(|| format!("failed to write loader in {}", out_dir.display()))?;
    fs::write(
        out_dir.join("browser-app.css"),
        format!(
            "{}\n{}",
            burn_p2p_app::browser_app_stylesheet(),
            EXTRA_STYLESHEET
        ),
    )
    .with_context(|| format!("failed to write stylesheet in {}", out_dir.display()))?;
    fs::write(
        out_dir.join("browser-app-config.json"),
        serde_json::to_vec_pretty(&browser_site_bootstrap_json(args)?)?,
    )
    .with_context(|| format!("failed to write browser config in {}", out_dir.display()))?;
    Ok(())
}

fn write_html_page(out_dir: &Path, relative_path: &str, asset_prefix: &str) -> Result<()> {
    let page_path = out_dir.join(relative_path);
    if let Some(parent) = page_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let html = INDEX_HTML_TEMPLATE.replace("__ASSET_PREFIX__", asset_prefix);
    fs::write(&page_path, html)
        .with_context(|| format!("failed to write {}", page_path.display()))?;
    Ok(())
}

fn browser_site_bootstrap_json(
    args: &BuildBrowserSiteArgs,
) -> Result<DragonBrowserSiteBootstrapDocument> {
    let edge_url = args
        .edge_url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let seed_node_urls: Vec<String> = args
        .seed_node_urls
        .iter()
        .map(String::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect();

    let selected_experiment_id = args
        .selected_experiment_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let selected_revision_id = args
        .selected_revision_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let edge_snapshot = edge_url
        .as_deref()
        .map(fetch_browser_edge_snapshot)
        .transpose()?;
    let signed_seed_advertisement =
        if let (Some(edge_url), Some(snapshot)) = (edge_url.as_deref(), edge_snapshot.as_ref()) {
            fetch_signed_seed_advertisement(edge_url, snapshot)?
        } else {
            None
        };

    Ok(DragonBrowserSiteBootstrapDocument {
        config: DragonBrowserAppConfigDocument {
            network: DragonPeerNetworkConfigDocument {
                edge_base_url: edge_url,
                seed_node_urls: (!seed_node_urls.is_empty()).then_some(seed_node_urls.clone()),
            },
            selected_experiment_id: selected_experiment_id.clone(),
            selected_revision_id,
            requested_scopes: browser_site_requested_scopes(selected_experiment_id.as_deref()),
            require_edge_auth: args.require_edge_auth,
        },
        release_manifest: edge_snapshot
            .as_ref()
            .map(browser_release_manifest_from_snapshot),
        edge_snapshot,
        signed_seed_advertisement,
    })
}

fn browser_site_requested_scopes(
    selected_experiment_id: Option<&str>,
) -> BTreeSet<ExperimentScope> {
    let mut requested_scopes =
        BTreeSet::from([ExperimentScope::Connect, ExperimentScope::Discover]);
    if let Some(experiment_id) = selected_experiment_id {
        let experiment_id = ExperimentId::new(experiment_id);
        requested_scopes.insert(ExperimentScope::Train {
            experiment_id: experiment_id.clone(),
        });
        requested_scopes.insert(ExperimentScope::Archive { experiment_id });
    }
    requested_scopes
}

fn fetch_browser_edge_snapshot(edge_url: &str) -> Result<BrowserEdgeSnapshot> {
    let edge_url = edge_url.trim_end_matches('/');
    let snapshot_url = format!("{edge_url}/portal/snapshot");
    edge_get_with_retry(&snapshot_url, "browser edge snapshot")?
        .error_for_status()
        .context("browser edge snapshot returned a non-success status")?
        .json::<BrowserEdgeSnapshot>()
        .context("decode browser edge snapshot JSON")
}

fn fetch_signed_seed_advertisement(
    edge_url: &str,
    snapshot: &BrowserEdgeSnapshot,
) -> Result<Option<SignedPayload<SchemaEnvelope<BrowserSeedAdvertisement>>>> {
    let edge_url = edge_url.trim_end_matches('/');
    let seed_path = snapshot.paths.browser_seed_advertisement_path.trim();
    let seed_url = if seed_path.starts_with('/') {
        format!("{edge_url}{seed_path}")
    } else {
        format!("{edge_url}/{seed_path}")
    };
    let response = edge_get_with_retry(&seed_url, "signed browser seed advertisement")?;
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    response
        .error_for_status()
        .context("browser seed advertisement returned a non-success status")?
        .json::<SignedPayload<SchemaEnvelope<BrowserSeedAdvertisement>>>()
        .map(Some)
        .context("decode signed browser seed advertisement JSON")
}

fn edge_get_with_retry(url: &str, resource_name: &str) -> Result<reqwest::blocking::Response> {
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(20))
        .build()
        .context("build browser site HTTP client")?;
    let mut last_error = None;

    for attempt in 1..=EDGE_FETCH_MAX_ATTEMPTS {
        match client.get(url).send() {
            Ok(response) if should_retry_edge_status(response.status()) => {
                last_error = Some(anyhow!(
                    "{resource_name} returned transient status {}",
                    response.status()
                ));
            }
            Ok(response) => return Ok(response),
            Err(error) => {
                last_error = Some(anyhow!(
                    "fetch {resource_name} from {url} (attempt {attempt}): {error}"
                ));
            }
        }

        if attempt < EDGE_FETCH_MAX_ATTEMPTS {
            thread::sleep(Duration::from_secs(attempt as u64));
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow!("fetch {resource_name} from {url} failed")))
}

fn should_retry_edge_status(status: reqwest::StatusCode) -> bool {
    status.is_server_error()
        || status == reqwest::StatusCode::REQUEST_TIMEOUT
        || status == reqwest::StatusCode::TOO_MANY_REQUESTS
}

fn current_app_semver() -> semver::Version {
    semver::Version::parse(env!("CARGO_PKG_VERSION")).expect("valid burn_dragon version")
}

fn browser_release_manifest_from_snapshot(snapshot: &BrowserEdgeSnapshot) -> ClientReleaseManifest {
    let target_artifact_hash = snapshot
        .allowed_target_artifact_hashes
        .iter()
        .next()
        .cloned()
        .or_else(|| {
            snapshot
                .trust_bundle
                .as_ref()
                .and_then(|bundle| bundle.allowed_target_artifact_hashes.iter().next().cloned())
        })
        .unwrap_or_else(|| ContentId::new("dragon-browser-client-artifact"));
    let release_train_hash = snapshot
        .required_release_train_hash
        .clone()
        .or_else(|| {
            snapshot
                .trust_bundle
                .as_ref()
                .map(|bundle| bundle.required_release_train_hash.clone())
        })
        .unwrap_or_else(|| ContentId::new("dragon-browser-client-train"));
    let project_family_id = snapshot
        .trust_bundle
        .as_ref()
        .map(|bundle| bundle.project_family_id.clone())
        .unwrap_or_else(|| ProjectFamilyId::new("burn-dragon-language"));

    ClientReleaseManifest {
        project_family_id,
        release_train_hash,
        target_artifact_id: "browser-wasm".into(),
        target_artifact_hash,
        target_platform: ClientPlatform::Browser,
        app_semver: current_app_semver(),
        git_commit: "browser-site".into(),
        cargo_lock_hash: ContentId::new("dragon-browser-site-lock"),
        burn_version_string: "0.21.0-pre.3".into(),
        enabled_features_hash: ContentId::new("dragon-browser-site-features"),
        protocol_major: 0,
        supported_workloads: Vec::new(),
        built_at: snapshot.captured_at,
    }
}

fn cargo_bin() -> String {
    std::env::var("CARGO").unwrap_or_else(|_| "cargo".into())
}

fn run_step(command: &mut Command, label: &str) -> Result<()> {
    let output = command.output().with_context(|| format!("run {label}"))?;
    ensure!(
        output.status.success(),
        "{label} failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    Ok(())
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask should live under workspace root")
        .to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::{INDEX_HTML_TEMPLATE, should_retry_edge_status, write_html_page};
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn browser_shell_template_includes_main_mount_node() {
        assert!(INDEX_HTML_TEMPLATE.contains("id=\"main\""));
    }

    #[test]
    fn generated_html_page_includes_main_mount_node() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("unix time")
            .as_nanos();
        let temp: PathBuf = std::env::temp_dir().join(format!("burn-dragon-browser-site-{unique}"));
        std::fs::create_dir_all(&temp).expect("create temp dir");
        write_html_page(&temp, "index.html", ".").expect("write html page");
        let html = std::fs::read_to_string(temp.join("index.html")).expect("read html");
        assert!(html.contains("<div id=\"main\"></div>"));
        std::fs::remove_dir_all(&temp).expect("remove temp dir");
    }

    #[test]
    fn transient_edge_statuses_retry() {
        assert!(should_retry_edge_status(reqwest::StatusCode::BAD_GATEWAY));
        assert!(should_retry_edge_status(
            reqwest::StatusCode::SERVICE_UNAVAILABLE
        ));
        assert!(should_retry_edge_status(
            reqwest::StatusCode::GATEWAY_TIMEOUT
        ));
        assert!(should_retry_edge_status(
            reqwest::StatusCode::REQUEST_TIMEOUT
        ));
        assert!(should_retry_edge_status(
            reqwest::StatusCode::TOO_MANY_REQUESTS
        ));
        assert!(!should_retry_edge_status(reqwest::StatusCode::NOT_FOUND));
        assert!(!should_retry_edge_status(reqwest::StatusCode::UNAUTHORIZED));
    }
}
