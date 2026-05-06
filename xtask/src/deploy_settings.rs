use std::collections::BTreeSet;
use std::net::IpAddr;
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, bail, ensure, Context, Result};
use burn_p2p::{BrowserEdgeSnapshot, ExperimentId, ExperimentScope};
use burn_p2p_core::{BrowserSeedAdvertisement, SchemaEnvelope, SignedPayload};
use clap::Args;
use serde::Serialize;

const EDGE_FETCH_MAX_ATTEMPTS: usize = 3;
const DEFAULT_BROWSER_APP_BASE_URL: &str = "https://dragon.aberration.technology";
const DEFAULT_EDGE_BASE_URL: &str = "https://edge.dragon.aberration.technology";
const DEFAULT_EXPERIMENT_ID: &str = "nca-prepretraining";
const DEFAULT_REVISION_ID: &str = "nca-r1";

#[derive(Debug, Clone, Args)]
pub struct ResolvePagesDeploySettingsArgs {
    #[arg(long)]
    pub environment: String,
    #[arg(long, default_value = "")]
    pub edge_base_url_input: String,
    #[arg(long, default_value = "")]
    pub edge_base_url_from_env: String,
    #[arg(long, default_value = "")]
    pub seed_node_urls_input: String,
    #[arg(long, default_value = "")]
    pub seed_node_urls_from_env: String,
    #[arg(long, default_value = "")]
    pub selected_experiment_id_input: String,
    #[arg(long, default_value = "")]
    pub selected_experiment_id_from_env: String,
    #[arg(long, default_value = "")]
    pub selected_revision_id_input: String,
    #[arg(long, default_value = "")]
    pub selected_revision_id_from_env: String,
    #[arg(long, default_value = "")]
    pub canary_principal_id_input: String,
    #[arg(long, default_value = "")]
    pub canary_principal_id_from_env: String,
    #[arg(long, default_value = "")]
    pub browser_app_base_url_from_env: String,
    #[arg(long, default_value = "")]
    pub edge_domain_name_from_env: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PagesDeploySettings {
    pub edge_base_url: String,
    pub browser_app_base_url: String,
    pub selected_experiment_id: String,
    pub selected_revision_id: String,
    pub canary_principal_id: String,
    pub seed_node_urls: Vec<String>,
    pub site_host: String,
    pub requested_scopes: BTreeSet<ExperimentScope>,
}

pub fn resolve_pages_deploy_settings(args: &ResolvePagesDeploySettingsArgs) -> Result<()> {
    let settings = resolve_pages_deploy_settings_inner(args)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&settings).context("serialize pages deploy settings")?
    );
    Ok(())
}

pub fn resolve_pages_deploy_settings_inner(
    args: &ResolvePagesDeploySettingsArgs,
) -> Result<PagesDeploySettings> {
    let browser_app_base_url = first_nonempty([
        &args.browser_app_base_url_from_env,
        DEFAULT_BROWSER_APP_BASE_URL,
    ])
    .to_owned();

    let edge_base_url = first_nonempty([
        &args.edge_base_url_input,
        &args.edge_base_url_from_env,
        &edge_base_url_from_domain(&args.edge_domain_name_from_env),
        DEFAULT_EDGE_BASE_URL,
    ])
    .to_owned();

    let selected_experiment_id = first_nonempty([
        &args.selected_experiment_id_input,
        &args.selected_experiment_id_from_env,
        DEFAULT_EXPERIMENT_ID,
    ])
    .to_owned();
    let selected_revision_id = first_nonempty([
        &args.selected_revision_id_input,
        &args.selected_revision_id_from_env,
        DEFAULT_REVISION_ID,
    ])
    .to_owned();
    let default_canary_principal_id = default_canary_principal_id(&args.environment);
    let canary_principal_id = first_nonempty([
        &args.canary_principal_id_input,
        &args.canary_principal_id_from_env,
        default_canary_principal_id.as_str(),
    ])
    .to_owned();

    let requested_scopes = browser_requested_scopes(&selected_experiment_id);
    let site_host = reqwest::Url::parse(&browser_app_base_url)
        .ok()
        .and_then(|value| value.host_str().map(ToOwned::to_owned))
        .unwrap_or_default();

    let seed_node_urls = resolve_seed_node_urls(
        &edge_base_url,
        &args.seed_node_urls_input,
        &args.seed_node_urls_from_env,
    )?;

    ensure!(
        !seed_node_urls.is_empty(),
        "browser pages deploy requires at least one browser-capable seed multiaddr"
    );

    Ok(PagesDeploySettings {
        edge_base_url,
        browser_app_base_url,
        selected_experiment_id,
        selected_revision_id,
        canary_principal_id,
        seed_node_urls,
        site_host,
        requested_scopes,
    })
}

fn resolve_seed_node_urls(
    edge_base_url: &str,
    seed_node_urls_input: &str,
    seed_node_urls_from_env: &str,
) -> Result<Vec<String>> {
    let mut seed_urls = dedupe_csv_seed_urls(seed_node_urls_input);
    if seed_urls.is_empty() {
        seed_urls = fetch_browser_seed_urls(edge_base_url)?;
    }
    if seed_urls.is_empty() {
        seed_urls = dedupe_csv_seed_urls(seed_node_urls_from_env);
    }
    seed_urls = prefer_validated_browser_seed_urls(seed_urls);
    seed_urls = canonicalize_browser_seed_urls(edge_base_url, seed_urls);

    let browser_capable_seed_urls = seed_urls
        .into_iter()
        .filter(|value| is_dialable_browser_seed(value))
        .collect::<Vec<_>>();

    ensure!(
        !browser_capable_seed_urls.is_empty(),
        "browser pages deploy requires at least one browser-capable seed multiaddr (expected webrtc-direct, webtransport, or wss/ws)"
    );
    Ok(browser_capable_seed_urls)
}

fn fetch_browser_seed_urls(edge_base_url: &str) -> Result<Vec<String>> {
    let edge_base_url = edge_base_url.trim_end_matches('/');
    let host = reqwest::Url::parse(edge_base_url)
        .ok()
        .and_then(|value| value.host_str().map(ToOwned::to_owned))
        .unwrap_or_else(|| "edge.dragon.aberration.technology".to_owned());
    let mut seed_urls = Vec::new();
    let mut signed_fetch_error = None;

    for attempt in 1..=EDGE_FETCH_MAX_ATTEMPTS {
        match fetch_signed_seed_advertisement(edge_base_url) {
            Ok(Some(advertisement)) => {
                seed_urls = advertisement_seed_urls(&advertisement);
                signed_fetch_error = None;
                break;
            }
            Ok(None) => {
                signed_fetch_error = Some(anyhow!(
                    "signed browser seed advertisement payload was missing"
                ));
                break;
            }
            Err(error) => {
                signed_fetch_error = Some(anyhow!("{error:#}"));
                if attempt < EDGE_FETCH_MAX_ATTEMPTS {
                    thread::sleep(Duration::from_secs(2));
                }
            }
        }
    }

    let mut snapshot_transports = None;
    if seed_urls.is_empty() || !seed_urls.iter().any(|value| is_direct_browser_seed(value)) {
        snapshot_transports = Some(fetch_browser_edge_snapshot(edge_base_url)?);
    }

    let direct_transport_expected = snapshot_transports
        .as_ref()
        .map(snapshot_advertises_direct_transports)
        .unwrap_or(false);
    let direct_seed_available = seed_urls.iter().any(|value| is_direct_browser_seed(value));

    if direct_transport_expected && !direct_seed_available {
        let detail = signed_fetch_error
            .map(|error| format!("signed_seed_fetch_error={error:#}"))
            .unwrap_or_else(|| "signed_browser_seeds_missing_direct_transport".to_owned());
        bail!(
            "browser pages deploy refusing to publish degraded WSS-only config while direct browser transports are advertised; {detail}"
        );
    }

    if seed_urls.is_empty()
        && snapshot_transports
            .as_ref()
            .is_some_and(|snapshot| snapshot.transports.wss_fallback)
    {
        seed_urls.push(format!("/dns4/{host}/tcp/443/wss"));
    }

    Ok(dedupe_seed_urls(seed_urls))
}

fn fetch_browser_edge_snapshot(edge_base_url: &str) -> Result<BrowserEdgeSnapshot> {
    let snapshot_url = format!("{}/portal/snapshot", edge_base_url.trim_end_matches('/'));
    edge_get_with_retry(&snapshot_url, "browser edge snapshot")?
        .error_for_status()
        .context("browser edge snapshot returned a non-success status")?
        .json::<BrowserEdgeSnapshot>()
        .context("decode browser edge snapshot JSON")
}

fn fetch_signed_seed_advertisement(
    edge_base_url: &str,
) -> Result<Option<SignedPayload<SchemaEnvelope<BrowserSeedAdvertisement>>>> {
    let url = format!(
        "{}/browser/seeds/signed",
        edge_base_url.trim_end_matches('/')
    );
    let response = edge_get_with_retry(&url, "signed browser seed advertisement")?;
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
        .context("build deploy settings HTTP client")?;
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
            thread::sleep(Duration::from_secs(2));
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow!("fetch {resource_name} from {url} failed")))
}

fn should_retry_edge_status(status: reqwest::StatusCode) -> bool {
    status.is_server_error()
        || status == reqwest::StatusCode::REQUEST_TIMEOUT
        || status == reqwest::StatusCode::TOO_MANY_REQUESTS
}

fn browser_requested_scopes(selected_experiment_id: &str) -> BTreeSet<ExperimentScope> {
    let experiment_id = ExperimentId::new(selected_experiment_id);
    BTreeSet::from([
        ExperimentScope::Connect,
        ExperimentScope::Discover,
        ExperimentScope::Train {
            experiment_id: experiment_id.clone(),
        },
        ExperimentScope::Archive { experiment_id },
    ])
}

fn advertisement_seed_urls(
    advertisement: &SignedPayload<SchemaEnvelope<BrowserSeedAdvertisement>>,
) -> Vec<String> {
    advertisement
        .payload
        .payload
        .seeds
        .iter()
        .flat_map(|record| record.multiaddrs.iter())
        .map(String::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn dedupe_csv_seed_urls(value: &str) -> Vec<String> {
    dedupe_seed_urls(
        value
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .collect(),
    )
}

pub(crate) fn dedupe_seed_urls(seed_urls: Vec<String>) -> Vec<String> {
    let mut seen = BTreeSet::new();
    seed_urls
        .into_iter()
        .filter(|value| seen.insert(value.clone()))
        .collect()
}

pub(crate) fn canonicalize_browser_seed_urls(
    edge_base_url: &str,
    seed_urls: Vec<String>,
) -> Vec<String> {
    let Some(edge_host) = browser_seed_dns_host(edge_base_url) else {
        return dedupe_seed_urls(seed_urls);
    };
    canonicalize_browser_seed_urls_for_host(&edge_host, seed_urls)
}

pub(crate) fn canonicalize_browser_seed_urls_for_host(
    edge_host: &str,
    seed_urls: Vec<String>,
) -> Vec<String> {
    let seed_urls = dedupe_seed_urls(seed_urls);
    let canonical_seeds = seed_urls
        .iter()
        .filter(|value| canonicalize_browser_seed_url(edge_host, (*value).clone()) == **value)
        .cloned()
        .collect::<BTreeSet<_>>();
    dedupe_seed_urls(
        seed_urls
            .into_iter()
            .map(|value| {
                let rewritten = canonicalize_browser_seed_url(edge_host, value.clone());
                if rewritten != value && canonical_seeds.contains(&rewritten) {
                    value
                } else {
                    rewritten
                }
            })
            .collect(),
    )
}

pub(crate) fn browser_seed_dns_host(edge_base_url: &str) -> Option<String> {
    reqwest::Url::parse(edge_base_url)
        .ok()
        .and_then(|value| value.host_str().map(ToOwned::to_owned))
        .filter(|host| host.parse::<IpAddr>().is_err())
}

fn canonicalize_browser_seed_url(edge_host: &str, seed_url: String) -> String {
    let segments = multiaddr_segments(&seed_url);
    if segments.len() < 3
        || !matches!(segments.first().copied(), Some("ip4" | "ip6"))
        || !is_dialable_browser_seed(&seed_url)
    {
        return seed_url;
    }
    format!("/dns4/{edge_host}/{}", segments[2..].join("/"))
}

pub(crate) fn is_dialable_browser_seed(value: &str) -> bool {
    let segments = multiaddr_segments(value);
    is_direct_browser_seed(value)
        || segments
            .iter()
            .any(|segment| matches!(*segment, "wss" | "ws"))
}

pub(crate) fn is_webrtc_direct_browser_seed(value: &str) -> bool {
    let segments = multiaddr_segments(value);
    segments.contains(&"webrtc-direct")
        && segments
            .first()
            .is_some_and(|segment| matches!(*segment, "ip4" | "ip6" | "dns4" | "dns6"))
        && segments.contains(&"certhash")
}

pub(crate) fn is_direct_browser_seed(value: &str) -> bool {
    let segments = multiaddr_segments(value);
    if is_webrtc_direct_browser_seed(value) {
        return true;
    }
    if segments.contains(&"webtransport") {
        return segments.contains(&"quic-v1") && segments.contains(&"certhash");
    }
    false
}

pub(crate) fn prefer_validated_browser_seed_urls(seed_urls: Vec<String>) -> Vec<String> {
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

fn snapshot_advertises_direct_transports(snapshot: &BrowserEdgeSnapshot) -> bool {
    snapshot.transports.webrtc_direct || snapshot.transports.webtransport_gateway
}

fn multiaddr_segments(value: &str) -> Vec<&str> {
    value
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect()
}

fn edge_base_url_from_domain(edge_domain_name: &str) -> String {
    normalized_value(edge_domain_name)
        .map(|domain| format!("https://{domain}"))
        .unwrap_or_default()
}

fn default_canary_principal_id(environment: &str) -> String {
    if environment.trim() == "production" {
        "browser-canary-mainnet-nca".to_owned()
    } else {
        format!("browser-canary-{}-nca", environment.trim())
    }
}

fn first_nonempty<'a>(values: impl IntoIterator<Item = &'a str>) -> &'a str {
    values
        .into_iter()
        .find(|value| !value.trim().is_empty())
        .unwrap_or("")
}

fn normalized_value(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

#[cfg(test)]
mod tests {
    use super::{
        browser_requested_scopes, canonicalize_browser_seed_urls, dedupe_csv_seed_urls,
        default_canary_principal_id, edge_base_url_from_domain, first_nonempty,
        is_dialable_browser_seed, is_direct_browser_seed, is_webrtc_direct_browser_seed,
        prefer_validated_browser_seed_urls, resolve_pages_deploy_settings_inner,
        ResolvePagesDeploySettingsArgs,
    };
    use burn_p2p::ExperimentScope;

    #[test]
    fn canary_principal_defaults_follow_environment() {
        assert_eq!(
            default_canary_principal_id("production"),
            "browser-canary-mainnet-nca"
        );
        assert_eq!(
            default_canary_principal_id("staging"),
            "browser-canary-staging-nca"
        );
    }

    #[test]
    fn edge_domain_fallback_builds_https_url() {
        assert_eq!(
            edge_base_url_from_domain("edge.dragon.aberration.technology"),
            "https://edge.dragon.aberration.technology"
        );
        assert!(edge_base_url_from_domain("").is_empty());
    }

    #[test]
    fn browser_seed_filter_accepts_expected_transports() {
        assert!(is_dialable_browser_seed(
            "/ip4/1.2.3.4/udp/443/webrtc-direct/certhash/uEiAbc"
        ));
        assert!(is_direct_browser_seed(
            "/ip4/1.2.3.4/udp/443/webrtc-direct/certhash/uEiAbc"
        ));
        assert!(is_dialable_browser_seed(
            "/dns4/edge.dragon.aberration.technology/tcp/443/wss"
        ));
        assert!(!is_direct_browser_seed(
            "/dns4/edge.dragon.aberration.technology/tcp/443/wss"
        ));
        assert!(is_webrtc_direct_browser_seed(
            "/ip4/1.2.3.4/udp/443/webrtc-direct/certhash/uEiAbc"
        ));
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
    fn dns_webrtc_direct_seed_is_treated_as_direct() {
        let seed = "/dns4/edge.dragon.aberration.technology/udp/443/webrtc-direct/certhash/uEiAbc";
        assert!(is_dialable_browser_seed(seed));
        assert!(is_direct_browser_seed(seed));
        assert!(is_webrtc_direct_browser_seed(seed));
    }

    #[test]
    fn canonicalize_browser_seed_urls_rewrites_ip_based_browser_seeds_to_edge_dns_host() {
        assert_eq!(
            canonicalize_browser_seed_urls(
                "https://edge.dragon.aberration.technology",
                vec![
                    "/ip4/3.149.166.58/udp/443/webrtc-direct/certhash/uEiAbc".to_owned(),
                    "/ip4/3.149.166.58/tcp/443/wss".to_owned(),
                ],
            ),
            vec![
                "/dns4/edge.dragon.aberration.technology/udp/443/webrtc-direct/certhash/uEiAbc"
                    .to_owned(),
                "/dns4/edge.dragon.aberration.technology/tcp/443/wss".to_owned(),
            ]
        );
        assert_eq!(
            canonicalize_browser_seed_urls(
                "https://edge.dragon.aberration.technology",
                vec![
                    "/dns4/edge.dragon.aberration.technology/udp/443/webrtc-direct/certhash/uEiAbc"
                        .to_owned(),
                    "/ip4/3.149.166.58/udp/443/webrtc-direct/certhash/uEiAbc".to_owned(),
                ],
            ),
            vec![
                "/dns4/edge.dragon.aberration.technology/udp/443/webrtc-direct/certhash/uEiAbc"
                    .to_owned(),
                "/ip4/3.149.166.58/udp/443/webrtc-direct/certhash/uEiAbc".to_owned(),
            ]
        );
    }

    #[test]
    fn dedupe_csv_seed_urls_preserves_order() {
        assert_eq!(
            dedupe_csv_seed_urls(" a ,b,a ,, b "),
            vec!["a".to_owned(), "b".to_owned()]
        );
    }

    #[test]
    fn first_nonempty_skips_blank_values() {
        assert_eq!(first_nonempty(["", "  ", "x", "y"]), "x");
    }

    #[test]
    fn resolved_settings_use_explicit_values_before_defaults() {
        let args = ResolvePagesDeploySettingsArgs {
            environment: "staging".to_owned(),
            edge_base_url_input: "https://custom-edge.example".to_owned(),
            edge_base_url_from_env: "https://ignored-edge.example".to_owned(),
            seed_node_urls_input: "/dns4/example.com/tcp/443/wss".to_owned(),
            seed_node_urls_from_env: String::new(),
            selected_experiment_id_input: "climbmix-pretraining".to_owned(),
            selected_experiment_id_from_env: "ignored".to_owned(),
            selected_revision_id_input: "climbmix-r1".to_owned(),
            selected_revision_id_from_env: "ignored".to_owned(),
            canary_principal_id_input: "browser-canary-custom".to_owned(),
            canary_principal_id_from_env: "ignored".to_owned(),
            browser_app_base_url_from_env: "https://dragon.example".to_owned(),
            edge_domain_name_from_env: "ignored.example".to_owned(),
        };

        let settings = resolve_pages_deploy_settings_inner(&args).expect("resolve settings");
        assert_eq!(settings.edge_base_url, "https://custom-edge.example");
        assert_eq!(settings.browser_app_base_url, "https://dragon.example");
        assert_eq!(settings.selected_experiment_id, "climbmix-pretraining");
        assert_eq!(settings.selected_revision_id, "climbmix-r1");
        assert_eq!(settings.canary_principal_id, "browser-canary-custom");
        assert_eq!(
            settings.seed_node_urls,
            vec!["/dns4/example.com/tcp/443/wss".to_owned()]
        );
        assert_eq!(settings.site_host, "dragon.example");
        assert!(settings
            .requested_scopes
            .contains(&ExperimentScope::Connect));
        assert_eq!(
            settings.requested_scopes,
            browser_requested_scopes("climbmix-pretraining")
        );
    }
}
