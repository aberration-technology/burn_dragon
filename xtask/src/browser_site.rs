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
const INDEX_HTML_TEMPLATE: &str = r#"<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>burn_dragon p2p</title>
    <link rel="stylesheet" href="__ASSET_PREFIX__/browser-app.css" />
  </head>
  <body>
    <div id="main"></div>
    <script type="module" src="__ASSET_PREFIX__/browser-app-loader.js"></script>
  </body>
</html>
"#;
const EXTRA_STYLESHEET: &str = r#"
:root {
  font-family: "Iosevka Etoile", "IBM Plex Sans", system-ui, sans-serif;
}

.dragon-browser-shell {
  gap: 22px;
}

.dragon-hero-actions {
  align-self: stretch;
}

.dragon-connection-editor {
  display: grid;
  gap: 12px;
  width: min(100%, 900px);
}

.dragon-metric-band {
  margin-top: 8px;
}

.dragon-panel-stack {
  display: grid;
  gap: 16px;
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
        serde_json::to_vec_pretty(&browser_site_bootstrap_json(args))?,
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

#[cfg(test)]
mod tests {
    use super::{INDEX_HTML_TEMPLATE, write_html_page};
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
}
