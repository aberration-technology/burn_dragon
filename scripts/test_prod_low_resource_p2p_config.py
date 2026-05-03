#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import re


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TF_ROOT = REPO_ROOT / "crates" / "burn_dragon_p2p" / "deploy" / "terraform" / "aws"


def terraform_default(text: str, variable_name: str) -> str:
    pattern = re.compile(
        rf'variable "{re.escape(variable_name)}" \{{.*?default\s+=\s+([^\n]+)',
        re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        raise AssertionError(f"missing Terraform default for {variable_name}")
    raw = match.group(1).strip()
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    return raw


def require_contains(text: str, snippet: str, label: str) -> None:
    if snippet not in text:
        raise AssertionError(f"missing {label}: {snippet}")


def require_default(text: str, variable_name: str, expected: str) -> None:
    actual = terraform_default(text, variable_name)
    if actual != expected:
        raise AssertionError(
            f"Terraform default {variable_name} = {actual!r}, expected {expected!r}"
        )


def main() -> None:
    main_tf = (TF_ROOT / "main.tf").read_text()
    variables_tf = (TF_ROOT / "variables.tf").read_text()
    user_data = (TF_ROOT / "templates" / "user-data.sh.tftpl").read_text()
    service_unit = (
        TF_ROOT / "templates" / "burn-p2p-bootstrap.service.tftpl"
    ).read_text()
    secret_sync = (
        TF_ROOT / "templates" / "bootstrap-secret-sync.sh.tftpl"
    ).read_text()

    # Keep the production deployment on the fixed-cost, browser-first profile.
    for variable_name, expected in [
        ("bootstrap_install_source", "crate"),
        ("instance_type", "t3a.small"),
        ("root_volume_size_gib", "32"),
        ("use_retained_bootstrap_data_volume", "false"),
        ("enable_data_volume_snapshots", "false"),
        ("enable_managed_control_plane_redis", "false"),
        ("managed_trainer_desired_capacity", "0"),
        ("managed_trainer_backend", "cpu"),
        ("managed_validator_enabled", "false"),
    ]:
        require_default(variables_tf, variable_name, expected)

    for snippet, label in [
        ("p2p_webrtc_port              = 443", "browser WebRTC-direct UDP/443"),
        (
            'bootstrap_state_storage_mode        = local.use_retained_bootstrap_data_volume ? "retained-ebs-volume" : "root-volume"',
            "root-volume state default",
        ),
        (
            'control_plane_state_backend         = local.managed_control_plane_redis_enabled ? "redis" : "local-file"',
            "local-file control-plane state default",
        ),
        (
            "managed_control_plane_redis_enabled = var.enable_managed_control_plane_redis",
            "managed Redis opt-in toggle",
        ),
        (
            "managed_trainer_enabled                    = var.managed_trainer_desired_capacity > 0",
            "managed trainer opt-in toggle",
        ),
        (
            "bootstrap_head_mirror_seed_node_urls         = local.bootstrap_peer_internal_multiaddrs",
            "head mirror internal bootstrap seeding",
        ),
        (
            "head_artifact_mirror_source_roots = [\n        local.bootstrap_head_mirror_storage_root,\n      ]",
            "edge bootstrap local head mirror artifact source",
        ),
        ('preset = "BootstrapOnly"', "bootstrap-only burn_p2p preset"),
        (
            "bootstrap_addresses = local.bootstrap_peer_internal_multiaddrs",
            "bootstrap self-seed address",
        ),
        ('"/ip4/0.0.0.0/tcp/${var.p2p_port}"', "bootstrap TCP listener"),
        (
            '"/ip4/0.0.0.0/udp/${var.p2p_port}/quic-v1"',
            "bootstrap QUIC listener",
        ),
        (
            '"/ip4/0.0.0.0/udp/${local.p2p_webrtc_port}/webrtc-direct"',
            "bootstrap WebRTC-direct listener",
        ),
        ("authority = null", "bootstrap authority disabled"),
        ("allow_dev_admin_token = false", "dev admin token disabled"),
        ("browser_edge_enabled = true", "browser edge service enabled"),
        ('browser_mode         = "Trainer"', "browser trainer mode"),
        ('social_mode          = "Public"', "public social surface"),
        ('profile_mode         = "Public"', "public profile surface"),
        (
            "operator_state_backend = local.managed_control_plane_redis_enabled ? {",
            "operator state Redis opt-in",
        ),
        (
            "session_state_backend = local.managed_control_plane_redis_enabled ? {",
            "auth session state Redis opt-in",
        ),
        ('kind                    = "S3Compatible"', "S3 artifact target"),
        ('publication_mode        = "Hybrid"', "hybrid artifact publication"),
        ('access_mode             = "Authenticated"', "authenticated artifacts"),
        ("supports_signed_urls    = true", "signed artifact URLs"),
        ("multipart_threshold_bytes = 16777216", "S3 multipart threshold"),
        ("signed_url_ttl_secs       = 900", "short signed URL TTL"),
        ('identity = "Persistent"', "persistent bootstrap peer identity"),
        (
            "bootstrap_peers = local.bootstrap_peer_internal_multiaddrs",
            "internal peer bootstrap list",
        ),
        (
            '"/dns4/${var.edge_domain_name}/tcp/${var.p2p_port}"',
            "public DNS TCP address",
        ),
        (
            '"/dns4/${var.edge_domain_name}/udp/${var.p2p_port}/quic-v1"',
            "public DNS QUIC address",
        ),
        (
            '"/ip4/PUBLIC_IP/udp/${local.p2p_webrtc_port}/webrtc-direct"',
            "runtime-rewritten public IPv4 WebRTC-direct address",
        ),
        ("persist_provider_tokens     = false", "provider token persistence disabled"),
        ("session_ttl_seconds = 86400", "one-day auth session TTL"),
        ("minimum_revocation_epoch = 1", "revocation epoch enabled"),
        ('strategy             = "KRegularGossip"', "diffusion gossip merge topology"),
        ("reducer_replication  = 0", "no always-on reducer replicas"),
        ("target_leaf_cohort   = 3", "small diffusion leaf cohort"),
        ('mode                  = "DiffusionSteadyState"', "diffusion promotion policy"),
        ("validator_quorum      = 1", "single-validator low-resource quorum"),
        ("promote_serve_head    = true", "serve-head promotion"),
        (
            "artifact_sync_timeout_secs   = 120",
            "large-checkpoint diffusion artifact sync timeout",
        ),
        ("allow_solo_promotion         = true", "solo promotion fallback"),
    ]:
        require_contains(main_tf, snippet, label)

    require_contains(
        user_data,
        "--features admin-http,metrics,metrics-indexer,artifact-publish,artifact-download,artifact-fs,artifact-s3,browser-edge,browser-join,${bootstrap_auth_feature},rbac,social",
        "prod bootstrap burn_p2p feature set",
    )
    require_contains(
        user_data,
        'address.replace("PUBLIC_IP", sys.argv[1])',
        "runtime public IPv4 WebRTC-direct rewrite",
    )
    require_contains(
        secret_sync,
        'AWS_CLI_CONNECT_TIMEOUT="$${AWS_CLI_CONNECT_TIMEOUT:-5}"',
        "bounded bootstrap secret sync connect timeout",
    )
    require_contains(
        secret_sync,
        'auth_client_id="$${BURN_P2P_AUTH_CLIENT_ID:-}"',
        "cached bootstrap auth client id reuse",
    )
    require_contains(
        secret_sync,
        'if [ -s "$AUTHORITY_KEY_PATH" ]; then',
        "cached bootstrap authority key reuse",
    )
    require_contains(service_unit, "LimitNOFILE=${limit_nofile}", "bootstrap fd limit")
    require_contains(service_unit, "TimeoutStartSec=90", "bounded bootstrap startup")
    require_contains(main_tf, "limit_nofile       = 262144", "bootstrap fd limit value")

    print("prod-low-resource-p2p-config-ok")


if __name__ == "__main__":
    main()
