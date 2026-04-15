#![forbid(unsafe_code)]

#[cfg(target_arch = "wasm32")]
use burn_dragon_p2p::config::{DragonBrowserAppConfig, DragonBrowserSiteBootstrap};
#[cfg(target_arch = "wasm32")]
use burn_dragon_p2p::wasm::DragonBrowserApp;
#[cfg(target_arch = "wasm32")]
use burn_p2p_browser::BrowserSiteBootstrapConfig;
#[cfg(target_arch = "wasm32")]
use dioxus::prelude::*;

const DEFAULT_BOOTSTRAP_PATH: &str = "browser-app-config.json";

fn callback_site_root_prefix(pathname: &str) -> Option<String> {
    let (prefix, _) = pathname.split_once("/callback/")?;
    let prefix = prefix.trim_end_matches('/');
    Some(if prefix.is_empty() {
        "/".to_owned()
    } else {
        format!("{prefix}/")
    })
}

fn default_bootstrap_path_for_pathname(pathname: &str) -> String {
    callback_site_root_prefix(pathname)
        .map(|prefix| format!("{prefix}{DEFAULT_BOOTSTRAP_PATH}"))
        .unwrap_or_else(|| DEFAULT_BOOTSTRAP_PATH.into())
}

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
    let location = window.location();
    let search = location.search().unwrap_or_default();
    let query = search.strip_prefix('?').unwrap_or(&search);
    if let Some(path) = url::form_urlencoded::parse(query.as_bytes())
        .find_map(|(key, value)| (key == "config").then(|| value.into_owned()))
        .filter(|value| !value.trim().is_empty())
    {
        return path;
    }
    let pathname = location.pathname().unwrap_or_default();
    default_bootstrap_path_for_pathname(&pathname)
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
            main { class: "browser-app-shell burn-dragon-p2p-bootstrap-error",
                section { class: "panel hero browser-hero",
                    div { class: "browser-hero-grid",
                        div { class: "browser-hero-copy",
                            div { class: "eyebrow", "burn_dragon" }
                            h1 { class: "app-title", "bootstrap load failed" }
                            p { class: "app-subtitle", "The browser app shell loaded, but its bootstrap config could not be fetched or decoded." }
                        }
                    }
                    pre { "{error}" }
                }
            }
        },
        None => rsx! {
            main { class: "browser-app-shell burn-dragon-p2p-bootstrap-loading",
                section { class: "panel hero browser-hero",
                    div { class: "browser-hero-grid",
                        div { class: "browser-hero-copy",
                            div { class: "eyebrow", "burn_dragon" }
                            h1 { class: "app-title", "loading browser shell" }
                            p { class: "app-subtitle", "Resolving the edge, auth, and experiment bootstrap config." }
                        }
                    }
                }
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

#[cfg(test)]
mod tests {
    use super::{callback_site_root_prefix, default_bootstrap_path_for_pathname};

    #[test]
    fn callback_paths_resolve_site_root_prefix() {
        assert_eq!(
            callback_site_root_prefix("/callback/github").as_deref(),
            Some("/")
        );
        assert_eq!(
            callback_site_root_prefix("/repo/callback/github").as_deref(),
            Some("/repo/")
        );
    }

    #[test]
    fn callback_paths_use_root_bootstrap_config() {
        assert_eq!(
            default_bootstrap_path_for_pathname("/callback/github"),
            "/browser-app-config.json"
        );
        assert_eq!(
            default_bootstrap_path_for_pathname("/repo/callback/github"),
            "/repo/browser-app-config.json"
        );
        assert_eq!(
            default_bootstrap_path_for_pathname("/repo/"),
            "browser-app-config.json"
        );
    }
}
