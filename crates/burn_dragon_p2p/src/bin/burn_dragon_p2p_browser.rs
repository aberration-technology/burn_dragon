#![forbid(unsafe_code)]

#[cfg(target_arch = "wasm32")]
use burn_dragon_p2p::auth::{native_cli_bridge_mode_active, resume_or_complete_native_cli_bridge};
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
                edge_snapshot: None,
                signed_seed_advertisement: None,
            },
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn default_bootstrap() -> DragonBrowserSiteBootstrap {
    DragonBrowserSiteBootstrap {
        config: DragonBrowserAppConfig::from_site_config(BrowserSiteBootstrapConfig::new(None)),
        release_manifest: None,
        edge_snapshot: None,
        signed_seed_advertisement: None,
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
fn NativeCliBridgeApp() -> Element {
    let status = use_signal(|| "preparing native CLI login.".to_owned());
    let error = use_signal(|| None::<String>);
    let started = use_signal(|| false);

    {
        let mut status = status;
        let mut error = error;
        let mut started = started;
        use_effect(move || {
            if *started.read() {
                return;
            }
            started.set(true);
            spawn(async move {
                match resume_or_complete_native_cli_bridge().await {
                    Ok(true) => {
                        status.set("continuing GitHub sign-in for the native CLI.".into());
                    }
                    Ok(false) => {
                        error.set(Some(
                            "native CLI auth bridge was not armed for this page load.".into(),
                        ));
                    }
                    Err(bridge_error) => {
                        error.set(Some(bridge_error.to_string()));
                    }
                }
            });
        });
    }

    rsx! {
        main { class: "browser-app-shell burn-dragon-p2p-native-cli-bridge",
            section { class: "panel hero browser-hero",
                div { class: "browser-hero-grid",
                    div { class: "browser-hero-copy",
                        div { class: "eyebrow", "burn_dragon" }
                        h1 { class: "app-title", "cli login" }
                        p { class: "app-subtitle", "relaying GitHub auth back to the native CLI." }
                    }
                }
                if let Some(error) = &*error.read() {
                    pre { "{error}" }
                } else {
                    p { class: "app-subtitle", "{status}" }
                    p { class: "app-subtitle", "this page does not start the browser peer." }
                }
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[component]
fn BrowserBootstrapApp() -> Element {
    let bootstrap = use_resource(|| async move { load_browser_site_bootstrap().await });

    match &*bootstrap.read_unchecked() {
        Some(Ok(bootstrap)) => {
            let bootstrap = bootstrap.clone();
            rsx! {
                DragonBrowserApp {
                    config: bootstrap.config,
                    release_manifest: bootstrap.release_manifest,
                    edge_snapshot: bootstrap.edge_snapshot,
                    signed_seed_advertisement: bootstrap.signed_seed_advertisement,
                }
            }
        }
        Some(Err(error)) => rsx! {
            main { class: "browser-app-shell burn-dragon-p2p-bootstrap-error",
                section { class: "panel hero browser-hero",
                    div { class: "browser-hero-grid",
                        div { class: "browser-hero-copy",
                            div { class: "eyebrow", "burn_dragon" }
                            h1 { class: "app-title", "app load failed" }
                            p { class: "app-subtitle", "could not load the site config." }
                        }
                    }
                    pre { "{error}" }
                }
            }
        },
        None => rsx! {},
    }
}

#[cfg(target_arch = "wasm32")]
#[component]
fn App() -> Element {
    if native_cli_bridge_mode_active() {
        rsx! { NativeCliBridgeApp {} }
    } else {
        rsx! { BrowserBootstrapApp {} }
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {
    console_error_panic_hook::set_once();
    burn_dragon_p2p::logging::init_browser_logging();
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
