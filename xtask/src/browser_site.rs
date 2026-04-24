use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, ensure};
use burn_dragon_p2p::config::DragonBrowserTrainingConfig;
use burn_dragon_p2p::profile::browser_training_config_from_directory_entries;
use burn_p2p::{
    BrowserEdgeSnapshot, ClientPlatform, ClientReleaseManifest, ContentId, ExperimentId,
    ExperimentScope, ProjectFamilyId,
};
use burn_p2p_core::{
    BrowserSeedAdvertisement, BrowserSeedTransportKind, SchemaEnvelope, SignedPayload,
};
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
    <link rel="icon" href="__ASSET_PREFIX__/favicon.svg" type="image/svg+xml" />
    <link rel="stylesheet" href="__ASSET_PREFIX__/browser-app.css" />
  </head>
  <body>
    <div id="main"></div>
    <script type="module" src="__ASSET_PREFIX__/browser-app-loader.js"></script>
  </body>
</html>
"##;
const FAVICON_SVG: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="12" fill="#000000"/>
  <path fill="#d87c7c" d="M15 46V18h17c10 0 17 5 17 14s-7 14-17 14H15Zm10-8h7c5 0 8-2 8-6s-3-6-8-6h-7v12Z"/>
</svg>
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
  background: #000000;
}

.dragon-browser-shell {
  gap: 22px;
}

.dragon-browser-shell .browser-hero {
  gap: 18px;
  padding: 26px 26px 22px;
}

.hero {
  background: linear-gradient(180deg, rgba(10, 10, 10, 0.98), rgba(6, 6, 6, 0.98));
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
  background: linear-gradient(180deg, rgba(10, 10, 10, 0.98), rgba(6, 6, 6, 0.98));
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
  align-items: center;
  position: relative;
  padding: 6px 0 2px;
}

.dragon-site-footer-build {
  position: absolute;
  left: 0;
  bottom: 2px;
  color: rgba(255, 255, 255, 0.14);
  font-family: ui-monospace, monospace;
  font-size: 0.58rem;
  letter-spacing: 0.08em;
  user-select: text;
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
  background: rgba(255, 255, 255, 0.025);
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
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.dragon-live-stats .stat-tile .stat-detail {
  margin-bottom: 0;
  color: var(--muted);
  line-height: 1.35;
  overflow-wrap: anywhere;
  word-break: break-word;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.dragon-live-actions {
  display: grid;
  gap: 10px;
  padding-top: 2px;
  justify-items: start;
}

.dragon-live-action-status {
  display: grid;
  gap: 4px;
  max-width: 34rem;
  padding: 0;
  color: var(--muted);
}

.dragon-live-action-status span {
  color: var(--ink);
  font-family: ui-monospace, monospace;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: lowercase;
}

.dragon-live-action-status p {
  margin: 0;
  line-height: 1.45;
}

.dragon-live-reset-button {
  opacity: 0.74;
}

.dragon-live-action-note {
  margin: 0;
  max-width: 34rem;
  color: var(--muted);
  line-height: 1.45;
}

.dragon-debug-transport-error {
  max-width: 100%;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.025);
  color: var(--muted);
}

.dragon-debug-transport-error summary {
  cursor: pointer;
  padding: 10px 12px;
  font-family: ui-monospace, monospace;
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: lowercase;
}

.dragon-debug-transport-error pre {
  max-height: 12rem;
  margin: 0;
  padding: 0 12px 12px;
  overflow: auto;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  color: rgba(255, 255, 255, 0.68);
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

main.dragon-browser-shell {
  width: min(100%, 1040px);
  max-width: 1040px;
  min-width: 0;
  overflow-x: hidden;
}

.dragon-browser-shell h1,
.dragon-browser-shell h2,
.dragon-browser-shell h3,
.dragon-browser-shell strong {
  letter-spacing: 0;
}

.dragon-peer-hero {
  gap: 16px;
  padding: 22px;
  border-radius: 10px;
  background: rgba(6, 6, 6, 0.98);
  overflow: hidden;
  max-width: 100%;
}

.dragon-peer-hero-ready {
  border-color: rgba(111, 211, 164, 0.34);
  box-shadow: 0 0 0 1px rgba(111, 211, 164, 0.04), 0 18px 48px rgba(0, 0, 0, 0.28);
}

.dragon-peer-hero-working {
  border-color: rgba(216, 124, 124, 0.34);
}

.dragon-peer-hero-blocked {
  border-color: rgba(255, 127, 127, 0.28);
}

.dragon-peer-hero-main {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 16px;
  align-items: start;
}

.dragon-peer-mark {
  width: 42px;
  height: 42px;
  display: grid;
  place-items: center;
  border: 1px solid var(--line);
  border-radius: 8px;
  color: var(--accent);
  background: rgba(255, 255, 255, 0.025);
  font-family: ui-monospace, monospace;
  font-size: 0.62rem;
  letter-spacing: 0.06em;
  text-transform: lowercase;
}

.dragon-peer-hero-copy {
  display: grid;
  gap: 10px;
  min-width: 0;
}

.dragon-peer-tone {
  display: inline-flex;
  align-items: center;
  min-height: 22px;
  padding: 3px 8px;
  border-radius: 999px;
  border: 1px solid var(--line);
  color: var(--muted);
  font-family: ui-monospace, monospace;
  font-size: 0.66rem;
  text-transform: lowercase;
}

.dragon-peer-tone-ready {
  border-color: rgba(111, 211, 164, 0.28);
  color: var(--success);
}

.dragon-peer-tone-working {
  border-color: rgba(216, 124, 124, 0.3);
  color: var(--accent);
}

.dragon-peer-tone-blocked {
  border-color: rgba(255, 127, 127, 0.28);
  color: var(--danger);
}

.dragon-peer-hero-title {
  max-width: none;
  font-size: clamp(2rem, 5vw, 3rem);
  line-height: 1.04;
  overflow-wrap: anywhere;
}

.dragon-peer-hero-detail {
  max-width: 58ch;
  color: var(--muted);
  overflow-wrap: anywhere;
}

.dragon-primary-action-bar {
  display: grid;
  gap: 12px;
  padding: 14px 16px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.018);
  min-width: 0;
  max-width: 100%;
}

.dragon-readiness-shell {
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.018);
  min-width: 0;
  max-width: 100%;
}

.dragon-readiness {
  display: grid;
  gap: 8px;
  grid-template-columns: repeat(7, minmax(0, 1fr));
  list-style: none;
  margin: 0;
  padding: 0;
}

.dragon-step {
  min-width: 0;
  display: grid;
  gap: 5px;
  padding: 10px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.018);
}

.dragon-step-marker {
  width: 22px;
  height: 22px;
  display: inline-grid;
  place-items: center;
  border-radius: 999px;
  border: 1px solid var(--line);
  color: var(--muted);
  font-family: ui-monospace, monospace;
  font-size: 0.72rem;
}

.dragon-step-label {
  color: var(--ink);
  font-size: 0.78rem;
  font-weight: 700;
  text-transform: lowercase;
}

.dragon-step-detail {
  color: var(--muted);
  font-size: 0.76rem;
  line-height: 1.35;
  overflow-wrap: anywhere;
}

.dragon-step-done .dragon-step-marker {
  border-color: rgba(111, 211, 164, 0.32);
  color: var(--success);
}

.dragon-step-active {
  border-color: rgba(216, 124, 124, 0.3);
}

.dragon-step-active .dragon-step-marker {
  border-color: rgba(216, 124, 124, 0.36);
  color: var(--accent);
}

.dragon-step-blocked {
  border-color: rgba(255, 127, 127, 0.28);
}

.dragon-step-blocked .dragon-step-marker {
  border-color: rgba(255, 127, 127, 0.32);
  color: var(--danger);
}

.dragon-metrics-grid {
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(auto-fit, minmax(156px, 1fr));
  align-items: stretch;
}

.dragon-card {
  min-width: 0;
  max-width: 100%;
  height: 100%;
  min-height: 124px;
  display: grid;
  grid-template-rows: 0.9rem minmax(2.35rem, auto) minmax(2.35rem, auto);
  align-content: start;
  gap: 9px;
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.022);
}

.dragon-metric-ready {
  border-top-color: rgba(111, 211, 164, 0.42);
}

.dragon-metric-working {
  border-top-color: rgba(216, 124, 124, 0.42);
}

.dragon-metric-blocked {
  border-top-color: rgba(255, 127, 127, 0.36);
}

.dragon-card-title {
  color: var(--muted);
  font-size: 0.72rem;
  line-height: 0.9rem;
  letter-spacing: 0;
  text-transform: lowercase;
  white-space: nowrap;
}

.dragon-card-value {
  align-self: start;
  color: var(--ink);
  font-size: 1.05rem;
  font-weight: 700;
  line-height: 1.22;
  overflow-wrap: anywhere;
  font-variant-numeric: tabular-nums;
}

.dragon-card-detail {
  align-self: start;
  color: var(--muted);
  font-size: 0.82rem;
  line-height: 1.4;
  overflow-wrap: anywhere;
}

.dragon-activity-panel {
  border-radius: 8px;
  min-width: 0;
  max-width: 100%;
}

.dragon-activity-feed {
  display: grid;
  gap: 8px;
  list-style: none;
  margin: 0;
  padding: 0;
}

.dragon-activity-event {
  display: grid;
  grid-template-columns: 72px minmax(0, 0.8fr) minmax(0, 1.4fr);
  gap: 10px;
  align-items: baseline;
  padding: 9px 0;
  border-top: 1px solid rgba(255, 255, 255, 0.055);
}

.dragon-activity-event:first-child {
  border-top: 0;
}

.dragon-activity-event time {
  color: rgba(255, 255, 255, 0.46);
  font-family: ui-monospace, monospace;
  font-size: 0.72rem;
  font-variant-numeric: tabular-nums;
}

.dragon-activity-label {
  color: var(--ink);
  font-size: 0.86rem;
  overflow-wrap: anywhere;
}

.dragon-activity-detail {
  color: var(--muted);
  font-size: 0.82rem;
  line-height: 1.35;
  overflow-wrap: anywhere;
}

.dragon-activity-event-error .dragon-activity-label {
  color: var(--danger);
}

.dragon-diagnostics-drawer {
  border-radius: 8px;
}

.dragon-diagnostics-summary {
  display: flex;
  gap: 12px;
  align-items: baseline;
  justify-content: space-between;
  cursor: pointer;
  color: var(--ink);
}

.dragon-diagnostics-summary span {
  font-weight: 700;
}

.dragon-diagnostics-summary small {
  color: var(--muted);
  font-size: 0.76rem;
}

.dragon-diagnostics-grid {
  display: grid;
  gap: 16px;
  margin-top: 18px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.dragon-diagnostics-section {
  min-width: 0;
  display: grid;
  gap: 12px;
}

.dragon-machine-state {
  max-height: 18rem;
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
  .dragon-readiness {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .dragon-diagnostics-grid {
    grid-template-columns: 1fr;
  }

  .dragon-editor-grid-wide {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 600px) {
  main.dragon-browser-shell {
    width: 100%;
    max-width: 100%;
    padding-left: 16px;
    padding-right: 16px;
  }

  .dragon-peer-hero {
    padding: 18px;
  }

  .dragon-peer-hero-main {
    grid-template-columns: 1fr;
  }

  .dragon-peer-mark {
    width: 36px;
    height: 36px;
  }

  .dragon-peer-hero-title {
    font-size: 2rem;
  }

  .dragon-peer-hero-detail {
    max-width: 100%;
  }

  .dragon-readiness {
    grid-template-columns: 1fr;
  }

  .dragon-activity-event {
    grid-template-columns: 1fr;
    gap: 4px;
  }

  .dragon-diagnostics-summary {
    display: grid;
    gap: 4px;
    justify-content: start;
  }
}

@media (prefers-reduced-motion: reduce) {
  .dragon-eyebrow-rattle {
    display: none;
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
    #[serde(skip_serializing_if = "Option::is_none")]
    training: Option<DragonBrowserTrainingConfig>,
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
    fs::write(out_dir.join("favicon.svg"), FAVICON_SVG)
        .with_context(|| format!("failed to write favicon.svg in {}", out_dir.display()))?;
    fs::write(out_dir.join("favicon.ico"), FAVICON_SVG)
        .with_context(|| format!("failed to write favicon.ico in {}", out_dir.display()))?;
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
    let seed_node_urls = prefer_validated_browser_seed_urls(
        args.seed_node_urls
            .iter()
            .map(String::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .collect(),
    );

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
    let edge_snapshot =
        fetch_optional_browser_edge_snapshot(edge_url.as_deref(), seed_node_urls.as_slice())?;
    let mut signed_seed_advertisement =
        if let (Some(edge_url), Some(snapshot)) = (edge_url.as_deref(), edge_snapshot.as_ref()) {
            fetch_signed_seed_advertisement(edge_url, snapshot)?
        } else {
            None
        };
    if let Some(advertisement) = signed_seed_advertisement.as_mut() {
        prefer_validated_browser_seed_advertisement(advertisement);
    }
    let training = resolve_browser_training_config(
        edge_snapshot.as_ref(),
        selected_experiment_id.as_deref(),
        selected_revision_id.as_deref(),
    )?;

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
            training,
        },
        release_manifest: edge_snapshot
            .as_ref()
            .map(browser_release_manifest_from_snapshot),
        edge_snapshot,
        signed_seed_advertisement,
    })
}

fn resolve_browser_training_config(
    edge_snapshot: Option<&BrowserEdgeSnapshot>,
    selected_experiment_id: Option<&str>,
    selected_revision_id: Option<&str>,
) -> Result<Option<DragonBrowserTrainingConfig>> {
    let Some(selected_experiment_id) = selected_experiment_id else {
        return Ok(None);
    };

    let Some(snapshot) = edge_snapshot else {
        return Ok(None);
    };

    let training = browser_training_config_from_directory_entries(
        &snapshot.directory.entries,
        Some(selected_experiment_id),
        selected_revision_id,
    )?;
    training.ok_or_else(|| {
        anyhow!(
            "selected experiment `{selected_experiment_id}` does not publish a browser training profile"
        )
    })
    .map(Some)
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

fn fetch_optional_browser_edge_snapshot(
    edge_url: Option<&str>,
    seed_node_urls: &[String],
) -> Result<Option<BrowserEdgeSnapshot>> {
    let Some(edge_url) = edge_url else {
        return Ok(None);
    };

    match fetch_browser_edge_snapshot(edge_url) {
        Ok(snapshot) => Ok(Some(snapshot)),
        Err(error) if !seed_node_urls.is_empty() => {
            eprintln!("warning: failed to embed browser edge snapshot from {edge_url}: {error:#}");
            eprintln!(
                "warning: continuing without embedded snapshot because explicit browser seed URLs were provided; the browser app will fetch live edge state at runtime"
            );
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

fn is_webrtc_direct_browser_seed(value: &str) -> bool {
    let segments = value
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>();
    segments.contains(&"webrtc-direct")
        && segments
            .first()
            .is_some_and(|segment| matches!(*segment, "ip4" | "ip6"))
        && segments.contains(&"certhash")
}

fn prefer_validated_browser_seed_urls(seed_urls: Vec<String>) -> Vec<String> {
    if seed_urls
        .iter()
        .any(|value| is_webrtc_direct_browser_seed(value))
    {
        return seed_urls
            .into_iter()
            .filter(|value| is_webrtc_direct_browser_seed(value))
            .collect();
    }
    seed_urls
}

fn prefer_validated_browser_seed_advertisement(
    advertisement: &mut SignedPayload<SchemaEnvelope<BrowserSeedAdvertisement>>,
) {
    let payload = &mut advertisement.payload.payload;
    let has_webrtc_direct = payload
        .seeds
        .iter()
        .flat_map(|record| record.multiaddrs.iter())
        .any(|value| is_webrtc_direct_browser_seed(value));
    if !has_webrtc_direct {
        return;
    }

    for seed in &mut payload.seeds {
        seed.multiaddrs
            .retain(|value| is_webrtc_direct_browser_seed(value));
    }
    payload.seeds.retain(|seed| !seed.multiaddrs.is_empty());
    payload.transport_policy.preferred = vec![BrowserSeedTransportKind::WebRtcDirect];
    payload.transport_policy.allow_fallback_wss = false;
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

fn current_git_commit() -> String {
    let workspace_root = workspace_root();
    let output = Command::new("git")
        .current_dir(&workspace_root)
        .args(["rev-parse", "--short=12", "HEAD"])
        .output();
    match output {
        Ok(output) if output.status.success() => {
            let commit = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if commit.is_empty() {
                "unknown".into()
            } else {
                commit
            }
        }
        _ => "unknown".into(),
    }
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
        git_commit: current_git_commit(),
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
    println!("running {label}");
    let status = command.status().with_context(|| format!("run {label}"))?;
    ensure!(status.success(), "{label} failed with status {status}");
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
    use super::{
        INDEX_HTML_TEMPLATE, fetch_optional_browser_edge_snapshot,
        prefer_validated_browser_seed_urls, resolve_browser_training_config,
        should_retry_edge_status, write_html_page, write_site_shell,
    };
    use crate::browser_site::BuildBrowserSiteArgs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn browser_shell_template_includes_main_mount_node() {
        assert!(INDEX_HTML_TEMPLATE.contains("id=\"main\""));
        assert!(INDEX_HTML_TEMPLATE.contains("favicon.svg"));
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
        assert!(html.contains("favicon.svg"));
        std::fs::remove_dir_all(&temp).expect("remove temp dir");
    }

    #[test]
    fn generated_site_shell_writes_favicon_fallbacks() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("unix time")
            .as_nanos();
        let temp: PathBuf = std::env::temp_dir().join(format!("burn-dragon-browser-site-{unique}"));
        std::fs::create_dir_all(&temp).expect("create temp dir");
        write_site_shell(&temp, &BuildBrowserSiteArgs::default()).expect("write site shell");
        assert!(temp.join("favicon.svg").is_file());
        assert!(temp.join("favicon.ico").is_file());
        std::fs::remove_dir_all(&temp).expect("remove temp dir");
    }

    #[test]
    fn validated_browser_seed_preference_strips_unvalidated_wss_fallback() {
        assert_eq!(
            prefer_validated_browser_seed_urls(vec![
                "/dns4/edge.dragon.aberration.technology/tcp/443/wss".to_owned(),
                "/ip4/1.2.3.4/udp/443/webrtc-direct/certhash/uEiAbc".to_owned(),
            ]),
            vec!["/ip4/1.2.3.4/udp/443/webrtc-direct/certhash/uEiAbc".to_owned()]
        );
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

    #[test]
    fn selected_experiment_without_snapshot_uses_runtime_training_resolution() {
        let training =
            resolve_browser_training_config(None, Some("nca-prepretraining"), Some("nca-r1"))
                .expect("resolve training config");
        assert!(training.is_none());
    }

    #[test]
    fn explicit_seed_urls_allow_runtime_snapshot_fallback() {
        let snapshot = fetch_optional_browser_edge_snapshot(
            Some("https://127.0.0.1:9"),
            &["/ip4/127.0.0.1/udp/443/webrtc-direct/certhash/uEiAbc".to_owned()],
        )
        .expect("runtime fallback should not fail site generation");
        assert!(snapshot.is_none());
    }
}
