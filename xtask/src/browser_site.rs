use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, ensure};
use burn_p2p::{ExperimentId, RevisionId};
use burn_p2p_browser::BrowserSiteBootstrapConfig;
use clap::{ArgAction, Args};
use wasm_bindgen_cli_support::Bindgen;

const BROWSER_BIN: &str = "burn_dragon_p2p_browser";
const DEFAULT_OUT_DIR: &str = "target/xtask/browser-site";
const BROWSER_APP_LOADER: &str = r#"import init from "./burn_dragon_p2p_browser.js";

await init({ module_or_path: new URL("./burn_dragon_p2p_browser_bg.wasm", import.meta.url) });
"#;
const INDEX_HTML: &str = r#"<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>burn_dragon p2p</title>
    <link rel="stylesheet" href="./browser-app.css" />
  </head>
  <body>
    <script type="module" src="./browser-app-loader.js"></script>
  </body>
</html>
"#;
const EXTRA_STYLESHEET: &str = r#"
:root {
  color-scheme: light;
  font-family: "Iosevka Etoile", "IBM Plex Sans", system-ui, sans-serif;
  background: #eef2f8;
  color: #121826;
}

body {
  margin: 0;
  min-height: 100vh;
  background:
    radial-gradient(circle at top, rgba(78, 111, 255, 0.12), transparent 40%),
    linear-gradient(180deg, #f8fbff 0%, #eef2f8 100%);
}

.burn-dragon-p2p-app {
  max-width: 1120px;
  margin: 0 auto;
  padding: 2rem 1.25rem 3rem;
  display: grid;
  gap: 1rem;
}

.burn-dragon-p2p-app > div,
.burn-dragon-p2p-app > section {
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(18, 24, 38, 0.08);
  border-radius: 16px;
  padding: 1rem 1.1rem;
  box-shadow: 0 20px 60px rgba(15, 23, 42, 0.08);
}

.burn-dragon-p2p-app input,
.burn-dragon-p2p-app textarea {
  width: 100%;
  box-sizing: border-box;
  margin-top: 0.4rem;
  padding: 0.75rem 0.9rem;
  border-radius: 12px;
  border: 1px solid rgba(18, 24, 38, 0.14);
  background: rgba(255, 255, 255, 0.96);
}

.burn-dragon-p2p-app textarea {
  min-height: 12rem;
  resize: vertical;
  font-family: "Iosevka Etoile", "IBM Plex Mono", monospace;
}

.burn-dragon-p2p-app button {
  margin-right: 0.75rem;
  margin-bottom: 0.5rem;
  padding: 0.7rem 1rem;
  border: none;
  border-radius: 999px;
  background: #1f4fff;
  color: white;
  font-weight: 600;
  cursor: pointer;
}

.burn-dragon-p2p-bootstrap-loading,
.burn-dragon-p2p-bootstrap-error {
  max-width: 720px;
  margin: 10vh auto;
  padding: 2rem;
}

.burn-dragon-p2p-bootstrap-error pre {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
"#;

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
        default_value_t = true
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
    fs::write(out_dir.join("index.html"), INDEX_HTML)
        .with_context(|| format!("failed to write {}/index.html", out_dir.display()))?;
    fs::write(out_dir.join("404.html"), INDEX_HTML)
        .with_context(|| format!("failed to write {}/404.html", out_dir.display()))?;
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
        serde_json::to_vec_pretty(&browser_site_bootstrap_json(args))?,
    )
    .with_context(|| format!("failed to write browser config in {}", out_dir.display()))?;
    Ok(())
}

fn browser_site_bootstrap_json(args: &BuildBrowserSiteArgs) -> BrowserSiteBootstrapConfig {
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

    let config = BrowserSiteBootstrapConfig::new(edge_url)
        .with_seed_node_urls(seed_node_urls)
        .with_edge_auth_requirement(args.require_edge_auth);
    match args
        .selected_experiment_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some(experiment_id) => config.with_selection(
            ExperimentId::new(experiment_id),
            args.selected_revision_id
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(RevisionId::new),
        ),
        None => config,
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
