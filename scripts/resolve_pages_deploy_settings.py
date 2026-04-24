#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ipaddress
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from typing import Any


EDGE_FETCH_MAX_ATTEMPTS = 20
EDGE_FETCH_RETRY_DELAY_SECONDS = 3
DEFAULT_BROWSER_APP_BASE_URL = "https://dragon.aberration.technology"
DEFAULT_EDGE_BASE_URL = "https://edge.dragon.aberration.technology"
DEFAULT_EXPERIMENT_ID = "nca-prepretraining"
DEFAULT_REVISION_ID = "nca-r1"
WEBRTC_DIRECT_BROWSER_HOST_PROTOCOLS = {"ip4", "ip6", "dns", "dns4", "dns6", "dnsaddr"}


def first_nonempty(*values: str) -> str:
    for value in values:
        if value.strip():
            return value.strip()
    return ""


def default_canary_principal_id(environment: str) -> str:
    environment = environment.strip()
    if environment == "production":
        return "browser-canary-mainnet-nca"
    return f"browser-canary-{environment}-nca"


def edge_base_url_from_domain(edge_domain_name: str) -> str:
    edge_domain_name = edge_domain_name.strip()
    if not edge_domain_name:
        return ""
    return f"https://{edge_domain_name}"


def dedupe(values: list[str]) -> list[str]:
    return list(OrderedDict.fromkeys(values))


def dedupe_csv_seed_urls(value: str) -> list[str]:
    return dedupe([part.strip() for part in value.split(",") if part.strip()])


def multiaddr_segments(value: str) -> list[str]:
    return [segment for segment in value.split("/") if segment]


def is_webrtc_direct_browser_seed(value: str) -> bool:
    segments = multiaddr_segments(value)
    return (
        "webrtc-direct" in segments
        and bool(segments)
        and segments[0] in WEBRTC_DIRECT_BROWSER_HOST_PROTOCOLS
        and "certhash" in segments
    )


def is_direct_browser_seed(value: str) -> bool:
    segments = multiaddr_segments(value)
    if is_webrtc_direct_browser_seed(value):
        return True
    if "webtransport" in segments:
        return "quic-v1" in segments and "certhash" in segments
    return False


def is_dialable_browser_seed(value: str) -> bool:
    segments = multiaddr_segments(value)
    return is_direct_browser_seed(value) or any(segment in {"wss", "ws"} for segment in segments)


def prefer_validated_browser_seed_urls(seed_urls: list[str]) -> list[str]:
    if any(is_webrtc_direct_browser_seed(value) for value in seed_urls):
        return [value for value in seed_urls if is_webrtc_direct_browser_seed(value)]
    return seed_urls


def browser_seed_dns_host(edge_base_url: str) -> str:
    host = urllib.parse.urlparse(edge_base_url).hostname or ""
    if not host:
        return ""
    try:
        ipaddress.ip_address(host)
        return ""
    except ValueError:
        return host


def canonicalize_browser_seed_url(edge_host: str, seed_url: str) -> str:
    segments = multiaddr_segments(seed_url)
    if (
        len(segments) < 3
        or segments[0] not in {"ip4", "ip6"}
        or not is_dialable_browser_seed(seed_url)
    ):
        return seed_url
    return f"/dns4/{edge_host}/{'/'.join(segments[2:])}"


def canonicalize_browser_seed_urls(edge_base_url: str, seed_urls: list[str]) -> list[str]:
    edge_host = browser_seed_dns_host(edge_base_url)
    if not edge_host:
        return dedupe(seed_urls)
    return dedupe([canonicalize_browser_seed_url(edge_host, value) for value in seed_urls])


def fetch_json(url: str, resource_name: str) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, EDGE_FETCH_MAX_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(url, headers={"accept": "application/json"})
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise
            last_error = error
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error

        if attempt < EDGE_FETCH_MAX_ATTEMPTS:
            time.sleep(EDGE_FETCH_RETRY_DELAY_SECONDS)

    raise RuntimeError(f"fetch {resource_name} from {url} failed: {last_error}")


def advertisement_seed_urls(advertisement: Any) -> list[str]:
    seed_records = (
        advertisement.get("payload", {})
        .get("payload", {})
        .get("seeds", [])
    )
    values: list[str] = []
    for record in seed_records:
        values.extend(str(value).strip() for value in record.get("multiaddrs", []))
    return dedupe([value for value in values if value])


def fetch_signed_seed_advertisement(edge_base_url: str) -> Any | None:
    url = f"{edge_base_url.rstrip('/')}/browser/seeds/signed"
    try:
        return fetch_json(url, "signed browser seed advertisement")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def fetch_browser_edge_snapshot(edge_base_url: str) -> Any:
    url = f"{edge_base_url.rstrip('/')}/portal/snapshot"
    return fetch_json(url, "browser edge snapshot")


def snapshot_advertises_direct_transports(snapshot: Any) -> bool:
    transports = snapshot.get("transports", {})
    return bool(transports.get("webrtc_direct") or transports.get("webtransport_gateway"))


def resolve_seed_node_urls(
    edge_base_url: str,
    seed_node_urls_input: str,
    seed_node_urls_from_env: str,
) -> list[str]:
    seed_urls = dedupe_csv_seed_urls(seed_node_urls_input)
    if not seed_urls:
        seed_urls = dedupe_csv_seed_urls(seed_node_urls_from_env)

    signed_fetch_error: Exception | None = None
    snapshot: Any | None = None

    if not seed_urls:
        try:
            advertisement = fetch_signed_seed_advertisement(edge_base_url)
            if advertisement is not None:
                seed_urls = advertisement_seed_urls(advertisement)
        except Exception as error:  # noqa: BLE001 - deployment diagnostics should retain detail.
            signed_fetch_error = error

    seed_urls = canonicalize_browser_seed_urls(
        edge_base_url,
        prefer_validated_browser_seed_urls(seed_urls),
    )

    if not seed_urls or not any(is_direct_browser_seed(value) for value in seed_urls):
        snapshot = fetch_browser_edge_snapshot(edge_base_url)

    if (
        snapshot is not None
        and snapshot_advertises_direct_transports(snapshot)
        and not any(is_direct_browser_seed(value) for value in seed_urls)
    ):
        detail = (
            f"signed_seed_fetch_error={signed_fetch_error}"
            if signed_fetch_error is not None
            else "signed_browser_seeds_missing_direct_transport"
        )
        raise RuntimeError(
            "browser pages deploy refusing to publish degraded WSS-only config while "
            f"direct browser transports are advertised; {detail}"
        )

    if not seed_urls and snapshot is not None and snapshot.get("transports", {}).get("wss_fallback"):
        host = urllib.parse.urlparse(edge_base_url).hostname or "edge.dragon.aberration.technology"
        seed_urls.append(f"/dns4/{host}/tcp/443/wss")

    browser_capable_seed_urls = [value for value in seed_urls if is_dialable_browser_seed(value)]
    if not browser_capable_seed_urls:
        raise RuntimeError(
            "browser pages deploy requires at least one browser-capable seed multiaddr "
            "(expected webrtc-direct, webtransport, or wss/ws)"
        )
    return dedupe(browser_capable_seed_urls)


def requested_scopes(selected_experiment_id: str) -> list[Any]:
    return [
        "Connect",
        "Discover",
        {"Train": {"experiment_id": selected_experiment_id}},
        {"Archive": {"experiment_id": selected_experiment_id}},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", required=True)
    parser.add_argument("--edge-base-url-input", default="")
    parser.add_argument("--edge-base-url-from-env", default="")
    parser.add_argument("--seed-node-urls-input", default="")
    parser.add_argument("--seed-node-urls-from-env", default="")
    parser.add_argument("--selected-experiment-id-input", default="")
    parser.add_argument("--selected-experiment-id-from-env", default="")
    parser.add_argument("--selected-revision-id-input", default="")
    parser.add_argument("--selected-revision-id-from-env", default="")
    parser.add_argument("--canary-principal-id-input", default="")
    parser.add_argument("--canary-principal-id-from-env", default="")
    parser.add_argument("--browser-app-base-url-from-env", default="")
    parser.add_argument("--edge-domain-name-from-env", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    browser_app_base_url = first_nonempty(
        args.browser_app_base_url_from_env,
        DEFAULT_BROWSER_APP_BASE_URL,
    )
    edge_base_url = first_nonempty(
        args.edge_base_url_input,
        args.edge_base_url_from_env,
        edge_base_url_from_domain(args.edge_domain_name_from_env),
        DEFAULT_EDGE_BASE_URL,
    )
    selected_experiment_id = first_nonempty(
        args.selected_experiment_id_input,
        args.selected_experiment_id_from_env,
        DEFAULT_EXPERIMENT_ID,
    )
    selected_revision_id = first_nonempty(
        args.selected_revision_id_input,
        args.selected_revision_id_from_env,
        DEFAULT_REVISION_ID,
    )
    canary_principal_id = first_nonempty(
        args.canary_principal_id_input,
        args.canary_principal_id_from_env,
        default_canary_principal_id(args.environment),
    )
    site_host = urllib.parse.urlparse(browser_app_base_url).hostname or ""
    seed_node_urls = resolve_seed_node_urls(
        edge_base_url,
        args.seed_node_urls_input,
        args.seed_node_urls_from_env,
    )
    print(
        json.dumps(
            {
                "edge_base_url": edge_base_url,
                "browser_app_base_url": browser_app_base_url,
                "selected_experiment_id": selected_experiment_id,
                "selected_revision_id": selected_revision_id,
                "canary_principal_id": canary_principal_id,
                "seed_node_urls": seed_node_urls,
                "site_host": site_host,
                "requested_scopes": requested_scopes(selected_experiment_id),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - keep deployment failures compact.
        print(f"resolve pages deploy settings failed: {error}", file=sys.stderr)
        raise
