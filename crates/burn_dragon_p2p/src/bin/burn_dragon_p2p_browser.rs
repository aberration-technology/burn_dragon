#![forbid(unsafe_code)]

#[cfg(target_arch = "wasm32")]
use burn_dragon_p2p::config::{DragonBrowserAppConfig, DragonBrowserSiteBootstrap};
#[cfg(target_arch = "wasm32")]
use burn_dragon_p2p::wasm::DragonBrowserApp;
#[cfg(target_arch = "wasm32")]
use burn_p2p_browser::BrowserSiteBootstrapConfig;
#[cfg(target_arch = "wasm32")]
use dioxus::prelude::*;

#[cfg(target_arch = "wasm32")]
const DEFAULT_BOOTSTRAP_PATH: &str = "browser-app-config.json";

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(untagged)]
enum BrowserBootstrapDocument {
    Dragon(Box<DragonBrowserSiteBootstrap>),
    Site(BrowserSiteBootstrapConfig),
}

#[cfg(target_arch = "wasm32")]
impl BrowserBootstrapDocument {
    fn into_dragon(self) -> DragonBrowserSiteBootstrap {
        match self {
            Self::Dragon(bootstrap) => *bootstrap,
            Self::Site(config) => DragonBrowserSiteBootstrap {
                config: DragonBrowserAppConfig::from_site_config(config),
                release_manifest: None,
            },
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn default_bootstrap() -> DragonBrowserSiteBootstrap {
    DragonBrowserSiteBootstrap {
        config: DragonBrowserAppConfig::from_site_config(BrowserSiteBootstrapConfig::new(None)),
        release_manifest: None,
    }
}

#[cfg(target_arch = "wasm32")]
fn bootstrap_path_from_window_query() -> String {
    let Some(window) = web_sys::window() else {
        return DEFAULT_BOOTSTRAP_PATH.into();
    };
    let search = window.location().search().unwrap_or_default();
    let query = search.strip_prefix('?').unwrap_or(&search);
    url::form_urlencoded::parse(query.as_bytes())
        .find_map(|(key, value)| (key == "config").then(|| value.into_owned()))
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_BOOTSTRAP_PATH.into())
}

#[cfg(target_arch = "wasm32")]
async fn load_browser_site_bootstrap() -> Result<DragonBrowserSiteBootstrap, String> {
    let path = bootstrap_path_from_window_query();
    match gloo_net::http::Request::get(&path).send().await {
        Ok(response) if response.status() == 404 && path == DEFAULT_BOOTSTRAP_PATH => {
            Ok(default_bootstrap())
        }
        Ok(response) if response.ok() => {
            let body = response
                .text()
                .await
                .map_err(|error| format!("failed to read {path}: {error}"))?;
            serde_json::from_str::<BrowserBootstrapDocument>(&body)
                .map(BrowserBootstrapDocument::into_dragon)
                .map_err(|error| format!("failed to decode {path}: {error}"))
        }
        Ok(response) => Err(format!("failed to load {path}: http {}", response.status())),
        Err(_error) if path == DEFAULT_BOOTSTRAP_PATH => Ok(default_bootstrap()),
        Err(error) => Err(format!("failed to fetch {path}: {error}")),
    }
}

#[cfg(target_arch = "wasm32")]
#[component]
fn App() -> Element {
    let bootstrap = use_resource(|| async move { load_browser_site_bootstrap().await });

    match &*bootstrap.read_unchecked() {
        Some(Ok(bootstrap)) => {
            let bootstrap = bootstrap.clone();
            rsx! {
                DragonBrowserApp {
                    config: bootstrap.config,
                    release_manifest: bootstrap.release_manifest,
                }
            }
        }
        Some(Err(error)) => rsx! {
            main {
                class: "burn-dragon-p2p-bootstrap-error",
                h1 { "burn_dragon p2p" }
                p { "Failed to load browser app bootstrap." }
                pre { "{error}" }
            }
        },
        None => rsx! {
            main {
                class: "burn-dragon-p2p-bootstrap-loading",
                h1 { "burn_dragon p2p" }
                p { "Loading browser peer shell..." }
            }
        },
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {
    dioxus::launch(App);
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!("burn_dragon_p2p_browser must be built for wasm32-unknown-unknown");
    std::process::exit(1);
}
