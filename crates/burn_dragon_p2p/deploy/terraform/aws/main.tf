data "aws_caller_identity" "current" {}

data "aws_partition" "current" {}

data "aws_route53_zone" "selected" {
  name         = endswith(var.route53_zone_name, ".") ? var.route53_zone_name : "${var.route53_zone_name}."
  private_zone = false
}

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

data "aws_kms_alias" "ssm" {
  name = "alias/aws/ssm"
}

data "aws_cloudfront_cache_policy" "caching_optimized" {
  name = "Managed-CachingOptimized"
}

locals {
  tags = merge(
    var.tags,
    {
      Application        = "burn-dragon-p2p"
      Environment        = var.environment_name
      ManagedBy          = "terraform"
      Stack              = var.stack_name
      TerraformWorkspace = terraform.workspace
    },
  )

  auth_connector_kind = lower(trimspace(var.auth_connector_kind))
  auth_oauth_enabled  = contains(["github", "oidc", "oauth"], local.auth_connector_kind)
  auth_redirect_path = local.auth_connector_kind == "github" ? "/callback/github" : (
    local.auth_connector_kind == "oidc" ? "/callback/oidc" : (
      local.auth_connector_kind == "oauth" ? "/callback/oauth" : null
    )
  )
  auth_endpoint_overrides = {
    authorize_base_url = trimspace(var.auth_authorize_base_url) != "" ? trimspace(var.auth_authorize_base_url) : null
    exchange_url       = trimspace(var.auth_exchange_url) != "" ? trimspace(var.auth_exchange_url) : null
    token_url          = trimspace(var.auth_token_url) != "" ? trimspace(var.auth_token_url) : null
    api_base_url       = trimspace(var.auth_api_base_url) != "" ? trimspace(var.auth_api_base_url) : null
    userinfo_url       = trimspace(var.auth_userinfo_url) != "" ? trimspace(var.auth_userinfo_url) : null
    refresh_url        = trimspace(var.auth_refresh_url) != "" ? trimspace(var.auth_refresh_url) : null
    revoke_url         = trimspace(var.auth_revoke_url) != "" ? trimspace(var.auth_revoke_url) : null
    jwks_url           = trimspace(var.auth_jwks_url) != "" ? trimspace(var.auth_jwks_url) : null
  }
  auth_endpoint_overrides_nonnull = {
    for key, value in local.auth_endpoint_overrides : key => value
    if value != null
  }
  github_auth_endpoint_defaults = {
    token_url = "https://github.com/login/oauth/access_token"
  }
  auth_principals          = try(jsondecode(var.auth_principals_json), [])
  bootstrap_install_source = lower(trimspace(var.bootstrap_install_source))
  secret_parameter_names = {
    auth_client_id                    = "${var.secret_parameter_prefix}/auth_client_id"
    auth_client_secret                = "${var.secret_parameter_prefix}/auth_client_secret"
    authority_key                     = "${var.secret_parameter_prefix}/authority_key"
    control_plane_redis_auth_token    = "${var.secret_parameter_prefix}/control_plane_redis_auth_token"
    bootstrap_head_mirror_auth_bundle = "${var.secret_parameter_prefix}/bootstrap_head_mirror_auth_bundle_json"
    trainer_auth_bundle               = "${var.secret_parameter_prefix}/trainer_auth_bundle_json"
    validator_auth_bundle             = "${var.secret_parameter_prefix}/validator_auth_bundle_json"
  }
  bootstrap_primary_private_ip = "10.42.1.10"
  bootstrap_peer_internal_multiaddrs = [
    "/ip4/${local.bootstrap_primary_private_ip}/tcp/${var.p2p_port}",
  ]
  bootstrap_data_mount_path           = "/var/lib/burn-p2p"
  bootstrap_auth_root                 = "${local.bootstrap_data_mount_path}/auth"
  bootstrap_auth_session_state_path   = "${local.bootstrap_auth_root}/session-state.cbor"
  bootstrap_peer_root                 = "${local.bootstrap_data_mount_path}/bootstrap-peer"
  bootstrap_data_snapshot_tag         = "${var.stack_name}-bootstrap-data"
  use_retained_bootstrap_data_volume  = var.use_retained_bootstrap_data_volume
  managed_control_plane_redis_enabled = var.enable_managed_control_plane_redis
  bootstrap_state_storage_mode        = local.use_retained_bootstrap_data_volume ? "retained-ebs-volume" : "root-volume"
  control_plane_state_backend         = local.managed_control_plane_redis_enabled ? "redis" : "local-file"
  cloudwatch_alarm_actions            = trimspace(var.alarm_sns_topic_arn) == "" ? [] : [trimspace(var.alarm_sns_topic_arn)]
  control_plane_dashboard_name        = "${var.stack_name}-${terraform.workspace}-control-plane"
  control_plane_dashboard_url         = "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${local.control_plane_dashboard_name}"
  managed_trainer_alarm_threshold     = max(var.managed_trainer_desired_capacity, 1)
  route53_zone_apex                   = trimsuffix(lower(trimspace(var.route53_zone_name)), ".")
  edge_domain_name_normalized         = trimsuffix(lower(trimspace(var.edge_domain_name)), ".")
  edge_base_url                       = "https://${var.edge_domain_name}"
  browser_app_base_url = trimspace(var.browser_app_base_url) != "" ? trimsuffix(
    trimspace(var.browser_app_base_url),
    "/",
  ) : null
  auth_redirect_base_url = trimspace(var.auth_redirect_base_url) != "" ? trimsuffix(
    trimspace(var.auth_redirect_base_url),
    "/",
  ) : local.browser_app_base_url
  browser_app_origin = coalesce(local.browser_app_base_url, local.edge_base_url)
  browser_app_hostname = local.browser_app_base_url == null ? null : split(
    "/",
    replace(replace(local.browser_app_base_url, "https://", ""), "http://", ""),
  )[0]
  browser_app_pages_domain_target = trimspace(var.browser_app_pages_domain_target) != "" ? trimsuffix(
    trimspace(var.browser_app_pages_domain_target),
    ".",
  ) : null
  acme_contact_email               = trimspace(var.acme_contact_email) != "" ? trimspace(var.acme_contact_email) : "admin@${trimsuffix(trimsuffix(var.route53_zone_name, "."), "/")}"
  browser_app_pages_record_enabled = local.browser_app_hostname != null && local.browser_app_pages_domain_target != null
  dataset_domain_name = trimspace(var.dataset_domain_name) != "" ? trimspace(var.dataset_domain_name) : (
    local.browser_app_hostname != null ? "datasets.${local.browser_app_hostname}" : "datasets.${var.edge_domain_name}"
  )
  dataset_domain_name_normalized = trimsuffix(lower(trimspace(local.dataset_domain_name)), ".")
  default_dataset_bucket_name = trimsuffix(
    substr(
      replace(
        replace(
          replace(
            lower("${var.stack_name}-${terraform.workspace}-${data.aws_caller_identity.current.account_id}-datasets"),
            ".",
            "-",
          ),
          "_",
          "-",
        ),
        "/",
        "-",
      ),
      0,
      63,
    ),
    "-",
  )
  dataset_bucket_name        = trimspace(var.dataset_bucket_name) != "" ? trimspace(var.dataset_bucket_name) : local.default_dataset_bucket_name
  dataset_bucket_path_prefix = trimspace(var.dataset_bucket_path_prefix) != "" ? trim(trimspace(var.dataset_bucket_path_prefix), "/") : "dragon-datasets"
  dataset_bucket_arn         = "arn:${data.aws_partition.current.partition}:s3:::${local.dataset_bucket_name}"
  dataset_bucket_object_arn  = "${local.dataset_bucket_arn}/*"
  dataset_bucket_s3_uri      = "s3://${local.dataset_bucket_name}/${local.dataset_bucket_path_prefix}"
  default_artifact_bucket_name = trimsuffix(
    substr(
      replace(
        replace(
          replace(
            lower("${var.stack_name}-${terraform.workspace}-${data.aws_caller_identity.current.account_id}-${var.aws_region}-artifacts"),
            ".",
            "-",
          ),
          "_",
          "-",
        ),
        "/",
        "-",
      ),
      0,
      63,
    ),
    "-",
  )
  artifact_bucket_name        = trimspace(var.artifact_bucket_name) != "" ? trimspace(var.artifact_bucket_name) : local.default_artifact_bucket_name
  artifact_bucket_path_prefix = trimspace(var.artifact_bucket_path_prefix) != "" ? trim(trimspace(var.artifact_bucket_path_prefix), "/") : "artifacts/${var.stack_name}/${terraform.workspace}"
  artifact_bucket_arn         = "arn:${data.aws_partition.current.partition}:s3:::${local.artifact_bucket_name}"
  artifact_bucket_object_arn  = "${local.artifact_bucket_arn}/*"
  artifact_bucket_s3_uri      = "s3://${local.artifact_bucket_name}/${local.artifact_bucket_path_prefix}"
  disaster_recovery_enabled   = trimspace(var.disaster_recovery_region) != ""
  disaster_recovery_region    = trimspace(var.disaster_recovery_region) != "" ? trimspace(var.disaster_recovery_region) : var.aws_region
  default_artifact_replica_bucket_name = trimsuffix(
    substr(
      replace(
        replace(
          replace(
            lower("${var.stack_name}-${terraform.workspace}-${data.aws_caller_identity.current.account_id}-${local.disaster_recovery_region}-artifacts-dr"),
            ".",
            "-",
          ),
          "_",
          "-",
        ),
        "/",
        "-",
      ),
      0,
      63,
    ),
    "-",
  )
  artifact_replica_bucket_name               = trimspace(var.artifact_replica_bucket_name) != "" ? trimspace(var.artifact_replica_bucket_name) : local.default_artifact_replica_bucket_name
  artifact_replica_bucket_arn                = "arn:${data.aws_partition.current.partition}:s3:::${local.artifact_replica_bucket_name}"
  artifact_replica_bucket_object_arn         = "${local.artifact_replica_bucket_arn}/*"
  artifact_replica_bucket_s3_uri             = "s3://${local.artifact_replica_bucket_name}/${local.artifact_bucket_path_prefix}"
  disaster_recovery_snapshot_copies_enabled  = local.disaster_recovery_enabled && local.use_retained_bootstrap_data_volume && var.enable_data_volume_snapshots && var.enable_disaster_recovery_snapshot_copies
  managed_trainer_enabled                    = var.managed_trainer_desired_capacity > 0
  managed_trainer_backend                    = lower(trimspace(var.managed_trainer_backend))
  managed_trainer_experiment_kind            = lower(trimspace(var.managed_trainer_experiment_kind))
  managed_trainer_min_size_effective         = var.managed_trainer_min_size > 0 ? var.managed_trainer_min_size : var.managed_trainer_desired_capacity
  managed_trainer_max_size_effective         = var.managed_trainer_max_size > 0 ? var.managed_trainer_max_size : max(var.managed_trainer_desired_capacity, 1)
  managed_trainer_features                   = local.managed_trainer_backend == "cpu" ? "native" : "native,${local.managed_trainer_backend}"
  managed_trainer_enabled_features_label     = local.managed_trainer_features
  managed_trainer_experiment_id              = local.managed_trainer_experiment_kind == "climbmix" ? "climbmix-pretraining" : "nca-prepretraining"
  managed_trainer_revision_id                = local.managed_trainer_experiment_kind == "climbmix" ? "climbmix-r1" : "nca-r1"
  managed_trainer_auth_bundle_parameter_name = trimspace(var.managed_trainer_auth_bundle_parameter_name) != "" ? trimspace(var.managed_trainer_auth_bundle_parameter_name) : local.secret_parameter_names.trainer_auth_bundle
  managed_trainer_seed_node_urls = [
    "/dns4/${var.edge_domain_name}/tcp/${var.p2p_port}",
    "/dns4/${var.edge_domain_name}/udp/${var.p2p_port}/quic-v1",
  ]
  bootstrap_head_mirror_backend                    = "cpu"
  bootstrap_head_mirror_experiment_kind            = "nca"
  bootstrap_head_mirror_experiment_id              = "nca-prepretraining"
  bootstrap_head_mirror_revision_id                = "nca-r1"
  bootstrap_head_mirror_target                     = "trainer"
  bootstrap_head_mirror_enabled_features_label     = "native"
  bootstrap_head_mirror_storage_root               = "${local.bootstrap_data_mount_path}/head-mirror"
  bootstrap_head_mirror_auth_bundle_parameter_name = trimspace(var.bootstrap_head_mirror_auth_bundle_parameter_name) != "" ? trimspace(var.bootstrap_head_mirror_auth_bundle_parameter_name) : local.secret_parameter_names.bootstrap_head_mirror_auth_bundle
  # The head mirror is colocated with the bootstrap edge on the same EC2 host.
  # Seed it against the bootstrap's in-VPC listen address instead of the public
  # edge DNS so it does not self-dial the instance's own public endpoint.
  bootstrap_head_mirror_seed_node_urls             = local.bootstrap_peer_internal_multiaddrs
  managed_validator_enabled                        = var.managed_validator_enabled
  managed_validator_experiment_kind                = lower(trimspace(var.managed_validator_experiment_kind))
  managed_validator_features                       = "native"
  managed_validator_enabled_features_label         = local.managed_validator_features
  managed_validator_experiment_id                  = local.managed_validator_experiment_kind == "climbmix" ? "climbmix-pretraining" : "nca-prepretraining"
  managed_validator_revision_id                    = local.managed_validator_experiment_kind == "climbmix" ? "climbmix-r1" : "nca-r1"
  managed_validator_auth_bundle_parameter_name     = trimspace(var.managed_validator_auth_bundle_parameter_name) != "" ? trimspace(var.managed_validator_auth_bundle_parameter_name) : local.secret_parameter_names.validator_auth_bundle
  managed_validator_seed_node_urls                 = local.managed_trainer_seed_node_urls
  auth_connector = local.auth_connector_kind == "github" ? merge(
    {
      kind          = "github"
      client_id     = "$${BURN_P2P_AUTH_CLIENT_ID}"
      client_secret = "$${BURN_P2P_AUTH_CLIENT_SECRET}"
      redirect_uri  = "$${BURN_P2P_AUTH_REDIRECT_URI}"
    },
    local.github_auth_endpoint_defaults,
    local.auth_endpoint_overrides_nonnull,
    ) : (
    local.auth_connector_kind == "oidc" ? merge(
      {
        kind          = "oidc"
        issuer        = trimspace(var.auth_oidc_issuer)
        client_id     = "$${BURN_P2P_AUTH_CLIENT_ID}"
        client_secret = "$${BURN_P2P_AUTH_CLIENT_SECRET}"
        redirect_uri  = "$${BURN_P2P_AUTH_REDIRECT_URI}"
      },
      {
        for key, value in local.auth_endpoint_overrides_nonnull : key => value
        if key != "api_base_url"
      },
      ) : (
      local.auth_connector_kind == "oauth" ? merge(
        {
          kind          = "oauth"
          provider      = trimspace(var.auth_oauth_provider)
          client_id     = "$${BURN_P2P_AUTH_CLIENT_ID}"
          client_secret = "$${BURN_P2P_AUTH_CLIENT_SECRET}"
          redirect_uri  = "$${BURN_P2P_AUTH_REDIRECT_URI}"
        },
        {
          for key, value in local.auth_endpoint_overrides_nonnull : key => value
          if key != "api_base_url"
        },
        ) : (
        local.auth_connector_kind == "external" ? {
          kind                     = "external"
          authority                = trimspace(var.auth_external_authority)
          trusted_principal_header = trimspace(var.auth_external_trusted_principal_header)
          trusted_internal_only    = var.auth_external_trusted_internal_only
          } : {
          kind = "static"
        }
      )
    )
  )

  github_required_orgs  = trimspace(var.github_required_org) == "" ? [] : [var.github_required_org]
  github_required_teams = trimspace(var.github_required_team) == "" ? [] : [var.github_required_team]
  github_admin_logins = sort(tolist(toset([
    for login in var.github_admin_logins : lower(trimspace(login))
    if trimspace(login) != ""
  ])))
  managed_climbmix_browser_dataset_base_url = format(
    "https://%s/%s/climbmix-pretraining/climbmix-r1",
    local.dataset_domain_name,
    local.dataset_bucket_path_prefix,
  )
  resolved_climbmix_browser_dataset_base_url = trimspace(var.climbmix_browser_dataset_base_url) != "" ? trimsuffix(trimspace(var.climbmix_browser_dataset_base_url), "/") : local.managed_climbmix_browser_dataset_base_url
  climbmix_browser_manifest_url = format(
    "%s/fetch-manifest.json",
    local.resolved_climbmix_browser_dataset_base_url,
  )

  dragon_experiment_scopes = [
    { "Train" = { experiment_id = "nca-prepretraining" } },
    { "Archive" = { experiment_id = "nca-prepretraining" } },
    { "Train" = { experiment_id = "climbmix-pretraining" } },
    { "Archive" = { experiment_id = "climbmix-pretraining" } },
  ]
  dragon_admin_scopes = concat(
    ["Connect", "Discover"],
    local.dragon_experiment_scopes,
    [{ "Admin" = { study_id = var.study_id } }],
  )
  nca_profile_json = trimspace(file("${path.module}/../../profiles/nca-r1.profile.json"))
  climbmix_profile = jsondecode(trimspace(file("${path.module}/../../profiles/climbmix-r1.profile.json")))
  climbmix_profile_json = jsonencode(
    local.climbmix_browser_manifest_url == null ? merge(
      local.climbmix_profile,
      {
        browser = null
      }
      ) : merge(
      local.climbmix_profile,
      {
        browser = merge(
          local.climbmix_profile.browser,
          {
            train_source = merge(
              local.climbmix_profile.browser.train_source,
              {
                manifest_url = local.climbmix_browser_manifest_url
              }
            )
          }
        )
      }
    )
  )

  nca_merge_topology_policy_json = jsonencode({
    strategy             = "KRegularGossip"
    reducer_replication  = 0
    target_leaf_cohort   = 3
    upper_fanin          = 0
    window_duration_secs = 60
    publish_jitter_ms    = 750
    staleness_windows    = 2
    promotion_policy = {
      mode                  = "DiffusionSteadyState"
      validator_quorum      = 1
      apply_single_root_ema = true
      allow_late_rollover   = true
      promote_serve_head    = true
      diffusion = {
        settlement_timeout_secs      = 45
        observation_poll_ms          = 250
        required_stable_observations = 4
        support_margin               = 1
        allow_solo_promotion         = true
      }
    }
  })
  climbmix_merge_topology_policy_json = jsonencode({
    strategy             = "KRegularGossip"
    reducer_replication  = 0
    target_leaf_cohort   = 3
    upper_fanin          = 0
    window_duration_secs = 180
    publish_jitter_ms    = 750
    staleness_windows    = 2
    promotion_policy = {
      mode                  = "DiffusionSteadyState"
      validator_quorum      = 1
      apply_single_root_ema = true
      allow_late_rollover   = true
      promote_serve_head    = true
      diffusion = {
        settlement_timeout_secs      = 45
        observation_poll_ms          = 250
        required_stable_observations = 4
        support_margin               = 1
        allow_solo_promotion         = true
      }
    }
  })

  contributor_rule = {
    principal_id   = var.github_principal_id
    display_name   = "burn_dragon mainnet contributor"
    required_orgs  = local.github_required_orgs
    required_teams = local.github_required_teams
    required_repo_access = [
      {
        repo               = var.github_required_repo
        minimum_permission = "write"
      },
    ]
    granted_roles = {
      roles = ["TrainerCpu", "TrainerGpu", "BrowserObserver", "BrowserTrainerWgpu", "Archive", "Viewer"]
    }
    granted_scopes   = concat(["Connect", "Discover"], local.dragon_experiment_scopes)
    allowed_networks = [var.network_id]
    custom_claims = {
      deployment_profile = var.environment_name
      stack              = var.stack_name
    }
  }

  admin_rules = [
    for login in local.github_admin_logins : {
      principal_id   = "github-admin-${login}"
      display_name   = "burn_dragon admin ${login}"
      provider_login = login
      required_orgs  = local.github_required_orgs
      required_teams = local.github_required_teams
      required_repo_access = [
        {
          repo               = var.github_required_repo
          minimum_permission = var.github_admin_required_repo_permission
        },
      ]
      granted_roles = {
        roles = ["TrainerCpu", "TrainerGpu", "BrowserObserver", "BrowserTrainerWgpu", "Archive", "Viewer"]
      }
      granted_scopes   = local.dragon_admin_scopes
      allowed_networks = [var.network_id]
      custom_claims = {
        deployment_profile = var.environment_name
        stack              = var.stack_name
        operator_role      = "admin"
        admin_capabilities = "all"
      }
    }
  ]
  auth_provider_policy = local.auth_connector_kind == "github" ? {
    github = {
      rules = concat([local.contributor_rule], local.admin_rules)
    }
  } : null

  experiment_directory = [
    {
      network_id        = var.network_id
      study_id          = var.study_id
      experiment_id     = "nca-prepretraining"
      workload_id       = "nca-prepretraining"
      display_name      = "NCA pre-pre-training"
      model_schema_hash = "burn-dragon-language-nca-v1"
      dataset_view_id   = "burn-dragon-universality-nca-v1"
      resource_requirements = {
        minimum_roles               = []
        minimum_device_memory_bytes = null
        minimum_system_memory_bytes = var.min_system_memory_bytes
        estimated_download_bytes    = 134217728
        estimated_window_seconds    = 60
      }
      visibility          = "OptIn"
      opt_in_policy       = "Scoped"
      current_revision_id = "nca-r1"
      current_head_id     = null
      allowed_roles = {
        roles = ["TrainerCpu", "TrainerGpu", "BrowserObserver", "BrowserTrainerWgpu", "Archive", "Viewer"]
      }
      allowed_scopes = [
        { "Train" = { experiment_id = "nca-prepretraining" } },
        { "Archive" = { experiment_id = "nca-prepretraining" } },
      ]
      metadata = {
        experiment_kind                                = "nca-prepretraining"
        stack                                          = var.stack_name
        dragon_profile_version                         = "1"
        dragon_profile_json                            = local.nca_profile_json
        "burn_p2p.revision.merge_topology.policy_json" = local.nca_merge_topology_policy_json
      }
    },
    {
      network_id        = var.network_id
      study_id          = var.study_id
      experiment_id     = "climbmix-pretraining"
      workload_id       = "climbmix-pretraining"
      display_name      = "ClimbMix pre-training"
      model_schema_hash = "burn-dragon-language-climbmix-v1"
      dataset_view_id   = "burn-dragon-climbmix-v1"
      resource_requirements = {
        minimum_roles               = []
        minimum_device_memory_bytes = null
        minimum_system_memory_bytes = var.min_system_memory_bytes
        estimated_download_bytes    = 2147483648
        estimated_window_seconds    = 180
      }
      visibility          = "OptIn"
      opt_in_policy       = "Scoped"
      current_revision_id = "climbmix-r1"
      current_head_id     = null
      allowed_roles = {
        roles = ["TrainerCpu", "TrainerGpu", "BrowserObserver", "BrowserTrainerWgpu", "Archive", "Viewer"]
      }
      allowed_scopes = [
        { "Train" = { experiment_id = "climbmix-pretraining" } },
        { "Archive" = { experiment_id = "climbmix-pretraining" } },
      ]
      metadata = {
        experiment_kind                                = "climbmix-pretraining"
        stack                                          = var.stack_name
        dragon_profile_version                         = "1"
        dragon_profile_json                            = local.climbmix_profile_json
        "burn_p2p.revision.merge_topology.policy_json" = local.climbmix_merge_topology_policy_json
      }
    },
  ]

  bootstrap_daemon_config = {
    spec = {
      preset = "BootstrapOnly"
      genesis = {
        network_id       = var.network_id
        protocol_version = var.protocol_version
        display_name     = "burn_dragon P2P ${title(var.environment_name)}"
        created_at       = "2026-01-01T00:00:00Z"
        metadata = {
          purpose = "burn-dragon-p2p"
          stack   = var.stack_name
        }
      }
      platform            = "Native"
      bootstrap_addresses = local.bootstrap_peer_internal_multiaddrs
      listen_addresses = [
        "/ip4/0.0.0.0/tcp/${var.p2p_port}",
        "/ip4/0.0.0.0/udp/${var.p2p_port}/quic-v1",
      ]
      authority = null
      archive = {
        pinned_heads                 = []
        pinned_artifacts             = []
        retain_contribution_receipts = true
      }
      admin_api = {
        supported_actions = [
          "Control",
          "ExportDiagnostics",
          "ExportDiagnosticsBundle",
          "ExportHeads",
          "ExportReceipts",
          "ExportReducerLoad",
          "ExportTrustBundle",
          "OperatorRetentionPrune",
          "RolloutAuthPolicy",
        ]
        diagnostics_enabled     = true
        receipt_exports_enabled = true
      }
    }
    http_bind_addr        = "127.0.0.1:${var.http_port}"
    allow_dev_admin_token = false
    optional_services = {
      browser_edge_enabled = true
      browser_mode         = "Trainer"
      social_mode          = "Public"
      profile_mode         = "Public"
    }
    remaining_work_units = var.remaining_work_units
    admin_signer_peer_id = "burn-dragon-bootstrap-authority"
    operator_state_backend = local.managed_control_plane_redis_enabled ? {
      kind       = "redis"
      url        = "$${BURN_P2P_SHARED_REDIS_URL}"
      key_prefix = "burn-dragon:operator-state"
    } : null
    artifact_publication = {
      targets = [
        {
          publication_target_id   = "local-default"
          label                   = "artifact-s3"
          kind                    = "S3Compatible"
          publication_mode        = "Hybrid"
          access_mode             = "Authenticated"
          allow_public_reads      = false
          supports_signed_urls    = true
          edge_proxy_required     = false
          max_artifact_size_bytes = var.local_artifact_max_size_bytes
          retention_ttl_secs      = var.local_artifact_retention_ttl_secs
          allowed_artifact_profiles = [
            "FullTrainingCheckpoint",
            "ServeCheckpoint",
            "BrowserSnapshot",
            "ManifestOnly",
          ]
          eager_alias_names         = []
          local_root                = null
          bucket                    = local.artifact_bucket_name
          endpoint                  = "https://s3.${var.aws_region}.amazonaws.com"
          region                    = var.aws_region
          access_key_id             = null
          secret_access_key         = null
          session_token             = null
          path_prefix               = local.artifact_bucket_path_prefix
          multipart_threshold_bytes = 16777216
          server_side_encryption    = var.artifact_bucket_server_side_encryption
          signed_url_ttl_secs       = 900
        },
      ]
    }
    bootstrap_peer = {
      node = {
        identity = "Persistent"
        storage = {
          root = local.bootstrap_peer_root
        }
        dataset         = null
        bootstrap_peers = local.bootstrap_peer_internal_multiaddrs
        listen_addresses = [
          "/ip4/0.0.0.0/tcp/${var.p2p_port}",
          "/ip4/0.0.0.0/udp/${var.p2p_port}/quic-v1",
        ]
      }
    }
    auth = {
      authority_name              = var.auth_authority_name
      connector                   = local.auth_connector
      authority_key_path          = "${local.bootstrap_auth_root}/bootstrap-authority.key"
      session_state_path          = local.bootstrap_auth_session_state_path
      persist_provider_tokens     = false
      issuer_key_id               = "burn-dragon-mainnet"
      project_family_id           = var.project_family_id
      required_release_train_hash = var.release_train_hash
      allowed_target_artifact_hashes = [
        var.native_target_artifact_hash,
        var.browser_target_artifact_hash,
      ]
      session_ttl_seconds = 86400
      session_state_backend = local.managed_control_plane_redis_enabled ? {
        kind       = "redis"
        url        = "$${BURN_P2P_SHARED_REDIS_URL}"
        key_prefix = "burn-dragon:auth-sessions"
      } : null
      minimum_revocation_epoch = 1
      principals               = local.auth_principals
      provider_policy          = local.auth_provider_policy
      directory_entries        = local.experiment_directory
    }
  }

  bootstrap_config_json = jsonencode(local.bootstrap_daemon_config)
  caddyfile = templatefile("${path.module}/templates/Caddyfile.tftpl", {
    acme_contact_email   = local.acme_contact_email
    edge_domain_name     = var.edge_domain_name
    http_port            = var.http_port
    browser_app_base_url = local.browser_app_base_url == null ? "" : local.browser_app_base_url
    browser_app_origin   = local.browser_app_origin
  })
  secret_sync_script = templatefile("${path.module}/templates/bootstrap-secret-sync.sh.tftpl", {
    aws_region                          = var.aws_region
    auth_client_credentials_required    = local.auth_oauth_enabled
    auth_client_id_name                 = local.secret_parameter_names.auth_client_id
    auth_client_secret_name             = local.secret_parameter_names.auth_client_secret
    auth_redirect_uri                   = local.auth_redirect_path == null ? "" : "${coalesce(local.auth_redirect_base_url, local.edge_base_url)}${local.auth_redirect_path}"
    authority_key_name                  = local.secret_parameter_names.authority_key
    authority_key_path                  = "${local.bootstrap_auth_root}/bootstrap-authority.key"
    bootstrap_node_role                 = "primary"
    control_plane_redis_enabled         = local.managed_control_plane_redis_enabled
    control_plane_redis_auth_token_name = local.secret_parameter_names.control_plane_redis_auth_token
    control_plane_redis_endpoint        = local.managed_control_plane_redis_enabled ? aws_elasticache_replication_group.control_plane[0].primary_endpoint_address : ""
    control_plane_redis_port            = local.managed_control_plane_redis_enabled ? aws_elasticache_replication_group.control_plane[0].port : 0
    edge_domain_name                    = var.edge_domain_name
  })
  bootstrap_head_mirror_config = templatefile("${path.module}/templates/bootstrap-head-mirror.toml.tftpl", {
    dragon_crate_version   = var.dragon_crate_version
    enabled_features_label = local.bootstrap_head_mirror_enabled_features_label
    storage_root           = local.bootstrap_head_mirror_storage_root
    target                 = local.bootstrap_head_mirror_target
    edge_base_url          = local.edge_base_url
    seed_node_urls         = local.bootstrap_head_mirror_seed_node_urls
    project_family_id      = var.project_family_id
    network_id             = var.network_id
    study_id               = var.study_id
    experiment_id          = local.bootstrap_head_mirror_experiment_id
    revision_id            = local.bootstrap_head_mirror_revision_id
    experiment_kind        = local.bootstrap_head_mirror_experiment_kind
  })
  bootstrap_head_mirror_fetch_auth_script = templatefile("${path.module}/templates/bootstrap-head-mirror-fetch-auth.sh.tftpl", {
    aws_region                             = var.aws_region
    head_mirror_auth_bundle_parameter_name = local.bootstrap_head_mirror_auth_bundle_parameter_name
  })
  bootstrap_head_mirror_service_unit = templatefile("${path.module}/templates/bootstrap-head-mirror.service.tftpl", {
    experiment_kind         = local.bootstrap_head_mirror_experiment_kind
    backend                 = local.bootstrap_head_mirror_backend
    config_path             = "/etc/burn_dragon_p2p/bootstrap-head-mirror.toml"
    auth_bundle_path        = "/var/lib/burn_dragon_p2p/bootstrap-head-mirror-auth-bundle.json"
    head_sync_interval_secs = 15
  })
  bootstrap_auth_feature = local.auth_connector_kind == "github" ? "auth-github" : (
    local.auth_connector_kind == "oidc" ? "auth-oidc" : (
      local.auth_connector_kind == "oauth" ? "auth-oauth" : (
        local.auth_connector_kind == "external" ? "auth-external" : "auth-static"
      )
    )
  )
}

check "github_connector_configuration" {
  assert {
    condition = local.auth_connector_kind != "github" || (
      trimspace(var.github_required_org) != "" &&
      trimspace(var.github_required_repo) != ""
    )
    error_message = "github auth deployments require github_required_org and github_required_repo."
  }
}

check "oidc_connector_configuration" {
  assert {
    condition     = local.auth_connector_kind != "oidc" || trimspace(var.auth_oidc_issuer) != ""
    error_message = "oidc auth deployments require auth_oidc_issuer."
  }
}

check "oauth_connector_configuration" {
  assert {
    condition     = local.auth_connector_kind != "oauth" || trimspace(var.auth_oauth_provider) != ""
    error_message = "oauth auth deployments require auth_oauth_provider."
  }
}

check "external_connector_configuration" {
  assert {
    condition     = local.auth_connector_kind != "external" || trimspace(var.auth_external_authority) != ""
    error_message = "external auth deployments require auth_external_authority."
  }
}

check "artifact_bucket_configuration" {
  assert {
    condition     = var.create_artifact_bucket || trimspace(var.artifact_bucket_name) != ""
    error_message = "Set artifact_bucket_name when create_artifact_bucket is false so the bootstrap host has an existing S3 bucket for direct artifact publication."
  }
}

check "artifact_replica_bucket_configuration" {
  assert {
    condition     = !local.disaster_recovery_enabled || var.create_artifact_replica_bucket || trimspace(var.artifact_replica_bucket_name) != ""
    error_message = "Set artifact_replica_bucket_name when create_artifact_replica_bucket is false so warm disaster recovery has an existing replica bucket."
  }
}

check "artifact_replica_encryption_configuration" {
  assert {
    condition     = !local.disaster_recovery_enabled || var.artifact_bucket_server_side_encryption == "AES256"
    error_message = "Warm disaster recovery currently supports AES256 artifact bucket encryption. Use the default AES256 mode when disaster_recovery_region is enabled."
  }
}

check "edge_domain_configuration" {
  assert {
    condition     = var.allow_route53_zone_apex_records || local.edge_domain_name_normalized != local.route53_zone_apex
    error_message = "edge_domain_name must not equal the Route53 hosted-zone apex unless allow_route53_zone_apex_records is set. This protects existing apex websites and CDNs from accidental replacement."
  }
}

check "dataset_domain_configuration" {
  assert {
    condition     = local.dataset_domain_name != var.edge_domain_name
    error_message = "dataset_domain_name must differ from edge_domain_name so the managed dataset CDN does not conflict with the bootstrap edge hostname."
  }

  assert {
    condition     = var.allow_route53_zone_apex_records || local.dataset_domain_name_normalized != local.route53_zone_apex
    error_message = "dataset_domain_name must not equal the Route53 hosted-zone apex unless allow_route53_zone_apex_records is set. This protects existing apex websites and CDNs from accidental replacement."
  }
}

check "browser_app_domain_configuration" {
  assert {
    condition     = local.browser_app_hostname == null || local.browser_app_hostname != local.edge_domain_name_normalized
    error_message = "browser_app_base_url must use a different hostname than edge_domain_name so the browser shell and bootstrap API edge remain split."
  }
}

check "browser_app_pages_domain_configuration" {
  assert {
    condition = !local.browser_app_pages_record_enabled || (
      local.browser_app_hostname != local.route53_zone_apex &&
      endswith(local.browser_app_hostname, ".${local.route53_zone_apex}")
    )
    error_message = "browser_app_base_url must be a non-apex hostname inside the selected Route53 zone when browser_app_pages_domain_target is set."
  }
}

check "bootstrap_install_source_configuration" {
  assert {
    condition     = local.bootstrap_install_source != "git" || trimspace(var.bootstrap_git_ref) != ""
    error_message = "Set bootstrap_git_ref when bootstrap_install_source = git."
  }
}

check "managed_trainer_scaling_configuration" {
  assert {
    condition     = !local.managed_trainer_enabled || (local.managed_trainer_min_size_effective <= var.managed_trainer_desired_capacity && local.managed_trainer_max_size_effective >= var.managed_trainer_desired_capacity)
    error_message = "Managed trainer min/max size must bracket managed_trainer_desired_capacity."
  }
}

check "managed_trainer_instance_type_configuration" {
  assert {
    condition     = !local.managed_trainer_enabled || local.managed_trainer_backend == "cpu" || startswith(lower(var.managed_trainer_instance_type), "g") || startswith(lower(var.managed_trainer_instance_type), "p")
    error_message = "Managed trainer backends wgpu and cuda require a GPU-capable instance type, for example g5.xlarge."
  }
}

resource "aws_vpc" "bootstrap" {
  cidr_block           = "10.42.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = merge(local.tags, {
    Name = "${var.stack_name}-vpc"
  })
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.bootstrap.id
  cidr_block              = "10.42.1.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = merge(local.tags, {
    Name = "${var.stack_name}-public-a"
  })
}

resource "aws_subnet" "public_secondary" {
  vpc_id                  = aws_vpc.bootstrap.id
  cidr_block              = "10.42.2.0/24"
  availability_zone       = data.aws_availability_zones.available.names[1]
  map_public_ip_on_launch = true

  tags = merge(local.tags, {
    Name = "${var.stack_name}-public-b"
  })
}

resource "aws_internet_gateway" "bootstrap" {
  vpc_id = aws_vpc.bootstrap.id

  tags = merge(local.tags, {
    Name = "${var.stack_name}-igw"
  })
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.bootstrap.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.bootstrap.id
  }

  tags = merge(local.tags, {
    Name = "${var.stack_name}-public"
  })
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "public_secondary" {
  subnet_id      = aws_subnet.public_secondary.id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "bootstrap" {
  name        = "${var.stack_name}-bootstrap"
  description = "burn_dragon_p2p bootstrap edge"
  vpc_id      = aws_vpc.bootstrap.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = var.p2p_port
    to_port     = var.p2p_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = var.p2p_port
    to_port     = var.p2p_port
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  dynamic "ingress" {
    for_each = var.ssh_cidr_blocks
    content {
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = [ingress.value]
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.tags, {
    Name = "${var.stack_name}-bootstrap"
  })
}

resource "aws_iam_role" "bootstrap" {
  name = "${var.stack_name}-bootstrap"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      },
    ]
  })

  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "ssm_managed_core" {
  role       = aws_iam_role.bootstrap.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "bootstrap_secret_access" {
  name = "${var.stack_name}-secret-access"
  role = aws_iam_role.bootstrap.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ssm:GetParameter",
          "ssm:GetParameters",
        ]
        Resource = [
          "arn:${data.aws_partition.current.partition}:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter${var.secret_parameter_prefix}/*",
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ssm:PutParameter",
        ]
        Resource = [
          "arn:${data.aws_partition.current.partition}:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter${local.secret_parameter_names.authority_key}",
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
        ]
        Resource = [data.aws_kms_alias.ssm.target_key_arn]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetBucketLocation",
          "s3:ListBucket",
          "s3:ListBucketMultipartUploads",
        ]
        Resource = [local.artifact_bucket_arn]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:AbortMultipartUpload",
          "s3:DeleteObject",
          "s3:GetObject",
          "s3:GetObjectAttributes",
          "s3:PutObject",
        ]
        Resource = [local.artifact_bucket_object_arn]
      },
    ]
  })
}

resource "aws_iam_instance_profile" "bootstrap" {
  name = "${var.stack_name}-bootstrap"
  role = aws_iam_role.bootstrap.name
}

resource "aws_security_group" "managed_trainer" {
  count       = local.managed_trainer_enabled ? 1 : 0
  name        = "${var.stack_name}-managed-trainer"
  description = "burn_dragon_p2p managed native trainer"
  vpc_id      = aws_vpc.bootstrap.id

  dynamic "ingress" {
    for_each = var.ssh_cidr_blocks
    content {
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = [ingress.value]
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.tags, {
    Name = "${var.stack_name}-managed-trainer"
  })
}

resource "aws_iam_role" "managed_trainer" {
  count = local.managed_trainer_enabled ? 1 : 0
  name  = "${var.stack_name}-managed-trainer"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      },
    ]
  })

  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "managed_trainer_ssm_managed_core" {
  count      = local.managed_trainer_enabled ? 1 : 0
  role       = aws_iam_role.managed_trainer[0].name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "managed_trainer_secret_access" {
  count = local.managed_trainer_enabled ? 1 : 0
  name  = "${var.stack_name}-managed-trainer-secret-access"
  role  = aws_iam_role.managed_trainer[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ssm:GetParameter",
          "ssm:GetParameters",
        ]
        Resource = [
          "arn:${data.aws_partition.current.partition}:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter${local.managed_trainer_auth_bundle_parameter_name}",
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
        ]
        Resource = [data.aws_kms_alias.ssm.target_key_arn]
      },
    ]
  })
}

resource "aws_iam_instance_profile" "managed_trainer" {
  count = local.managed_trainer_enabled ? 1 : 0
  name  = "${var.stack_name}-managed-trainer"
  role  = aws_iam_role.managed_trainer[0].name
}

resource "aws_launch_template" "managed_trainer" {
  count         = local.managed_trainer_enabled ? 1 : 0
  name_prefix   = "${var.stack_name}-managed-trainer-"
  image_id      = data.aws_ami.ubuntu.id
  instance_type = var.managed_trainer_instance_type

  iam_instance_profile {
    name = aws_iam_instance_profile.managed_trainer[0].name
  }

  vpc_security_group_ids = [aws_security_group.managed_trainer[0].id]

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  block_device_mappings {
    device_name = "/dev/sda1"

    ebs {
      volume_size           = var.managed_trainer_root_volume_size_gib
      volume_type           = "gp3"
      encrypted             = true
      delete_on_termination = true
    }
  }

  user_data = base64encode(templatefile("${path.module}/templates/trainer-user-data.sh.tftpl", {
    aws_region                         = var.aws_region
    dragon_crate_version               = var.managed_trainer_crate_version
    trainer_backend                    = local.managed_trainer_backend
    trainer_features                   = local.managed_trainer_features
    trainer_enabled_features_label     = local.managed_trainer_enabled_features_label
    trainer_edge_base_url              = "https://${var.edge_domain_name}"
    trainer_seed_node_urls             = local.managed_trainer_seed_node_urls
    trainer_project_family_id          = var.project_family_id
    trainer_network_id                 = var.network_id
    trainer_study_id                   = var.study_id
    trainer_experiment_id              = local.managed_trainer_experiment_id
    trainer_revision_id                = local.managed_trainer_revision_id
    trainer_experiment_kind            = local.managed_trainer_experiment_kind
    trainer_target                     = var.managed_trainer_target
    trainer_auth_bundle_parameter_name = local.managed_trainer_auth_bundle_parameter_name
  }))

  tag_specifications {
    resource_type = "instance"

    tags = merge(local.tags, {
      Name = "${var.stack_name}-managed-trainer"
      Role = "managed-trainer"
    })
  }

  update_default_version = true
}

resource "aws_autoscaling_group" "managed_trainer" {
  count                     = local.managed_trainer_enabled ? 1 : 0
  name                      = "${var.stack_name}-managed-trainer"
  min_size                  = local.managed_trainer_min_size_effective
  max_size                  = local.managed_trainer_max_size_effective
  desired_capacity          = var.managed_trainer_desired_capacity
  health_check_type         = "EC2"
  health_check_grace_period = 300
  vpc_zone_identifier       = [aws_subnet.public.id, aws_subnet.public_secondary.id]

  launch_template {
    id      = aws_launch_template.managed_trainer[0].id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "${var.stack_name}-managed-trainer"
    propagate_at_launch = true
  }

  tag {
    key                 = "Role"
    value               = "managed-trainer"
    propagate_at_launch = true
  }
}

resource "aws_security_group" "managed_validator" {
  count       = local.managed_validator_enabled ? 1 : 0
  name        = "${var.stack_name}-managed-validator"
  description = "burn_dragon_p2p managed native validator"
  vpc_id      = aws_vpc.bootstrap.id

  ingress {
    from_port   = var.p2p_port
    to_port     = var.p2p_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = var.p2p_port
    to_port     = var.p2p_port
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  dynamic "ingress" {
    for_each = var.ssh_cidr_blocks
    content {
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = [ingress.value]
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.tags, {
    Name = "${var.stack_name}-managed-validator"
  })
}

resource "aws_iam_role" "managed_validator" {
  count = local.managed_validator_enabled ? 1 : 0
  name  = "${var.stack_name}-managed-validator"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      },
    ]
  })

  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "managed_validator_ssm_managed_core" {
  count      = local.managed_validator_enabled ? 1 : 0
  role       = aws_iam_role.managed_validator[0].name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "managed_validator_secret_access" {
  count = local.managed_validator_enabled ? 1 : 0
  name  = "${var.stack_name}-managed-validator-secret-access"
  role  = aws_iam_role.managed_validator[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ssm:GetParameter",
          "ssm:GetParameters",
        ]
        Resource = [
          "arn:${data.aws_partition.current.partition}:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter${local.managed_validator_auth_bundle_parameter_name}",
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
        ]
        Resource = [data.aws_kms_alias.ssm.target_key_arn]
      },
    ]
  })
}

resource "aws_iam_instance_profile" "managed_validator" {
  count = local.managed_validator_enabled ? 1 : 0
  name  = "${var.stack_name}-managed-validator"
  role  = aws_iam_role.managed_validator[0].name
}

resource "aws_instance" "managed_validator" {
  count                  = local.managed_validator_enabled ? 1 : 0
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.managed_validator_instance_type
  subnet_id              = aws_subnet.public.id
  iam_instance_profile   = aws_iam_instance_profile.managed_validator[0].name
  vpc_security_group_ids = [aws_security_group.managed_validator[0].id]

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  root_block_device {
    volume_size           = var.managed_validator_root_volume_size_gib
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  user_data = templatefile("${path.module}/templates/validator-user-data.sh.tftpl", {
    aws_region                           = var.aws_region
    dragon_crate_version                 = var.managed_validator_crate_version
    validator_backend                    = "cpu"
    validator_features                   = local.managed_validator_features
    validator_enabled_features_label     = local.managed_validator_enabled_features_label
    validator_edge_base_url              = "https://${var.edge_domain_name}"
    validator_seed_node_urls             = local.managed_validator_seed_node_urls
    validator_project_family_id          = var.project_family_id
    validator_network_id                 = var.network_id
    validator_study_id                   = var.study_id
    validator_experiment_id              = local.managed_validator_experiment_id
    validator_revision_id                = local.managed_validator_revision_id
    validator_experiment_kind            = local.managed_validator_experiment_kind
    validator_auth_bundle_parameter_name = local.managed_validator_auth_bundle_parameter_name
    validator_validation_interval_millis = var.managed_validator_validation_interval_millis
  })

  tags = merge(local.tags, {
    Name = "${var.stack_name}-managed-validator"
    Role = "managed-validator"
  })
}

resource "aws_security_group" "control_plane_redis" {
  count = local.managed_control_plane_redis_enabled ? 1 : 0

  name        = "${var.stack_name}-control-plane-redis"
  description = "burn_dragon_p2p shared redis control plane"
  vpc_id      = aws_vpc.bootstrap.id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.bootstrap.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.tags, {
    Name = "${var.stack_name}-control-plane-redis"
  })
}

resource "aws_elasticache_subnet_group" "control_plane" {
  count = local.managed_control_plane_redis_enabled ? 1 : 0

  name       = substr(replace(lower("${var.stack_name}-${terraform.workspace}-cp"), "_", "-"), 0, 40)
  subnet_ids = [aws_subnet.public.id, aws_subnet.public_secondary.id]

  tags = local.tags
}

resource "random_password" "control_plane_redis_auth_token" {
  count = local.managed_control_plane_redis_enabled ? 1 : 0

  length  = 32
  special = false
}

resource "aws_ssm_parameter" "control_plane_redis_auth_token" {
  count = local.managed_control_plane_redis_enabled ? 1 : 0

  name      = local.secret_parameter_names.control_plane_redis_auth_token
  type      = "SecureString"
  key_id    = data.aws_kms_alias.ssm.target_key_arn
  overwrite = true
  value     = random_password.control_plane_redis_auth_token[0].result

  tags = local.tags
}

resource "aws_elasticache_replication_group" "control_plane" {
  count = local.managed_control_plane_redis_enabled ? 1 : 0

  replication_group_id       = substr(replace(lower("${var.stack_name}-${terraform.workspace}-cp"), "_", "-"), 0, 40)
  description                = "burn_dragon_p2p shared operator/session state"
  engine                     = "redis"
  engine_version             = "7.1"
  node_type                  = "cache.t4g.small"
  port                       = 6379
  parameter_group_name       = "default.redis7"
  subnet_group_name          = aws_elasticache_subnet_group.control_plane[0].name
  security_group_ids         = [aws_security_group.control_plane_redis[0].id]
  automatic_failover_enabled = false
  multi_az_enabled           = false
  num_cache_clusters         = 1
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = random_password.control_plane_redis_auth_token[0].result
  apply_immediately          = true

  tags = local.tags
}

resource "aws_s3_bucket" "artifact" {
  count = var.create_artifact_bucket ? 1 : 0

  bucket        = local.artifact_bucket_name
  force_destroy = var.artifact_bucket_force_destroy

  tags = merge(local.tags, {
    Name    = "${var.stack_name}-artifacts"
    Purpose = "burn-dragon-p2p-artifact-publication"
  })
}

resource "aws_s3_bucket_public_access_block" "artifact" {
  count = var.create_artifact_bucket ? 1 : 0

  bucket = aws_s3_bucket.artifact[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "artifact" {
  count = var.create_artifact_bucket ? 1 : 0

  bucket = aws_s3_bucket.artifact[0].id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifact" {
  count = var.create_artifact_bucket ? 1 : 0

  bucket = aws_s3_bucket.artifact[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = var.artifact_bucket_server_side_encryption
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "artifact" {
  count = var.create_artifact_bucket ? 1 : 0

  bucket = aws_s3_bucket.artifact[0].id

  rule {
    id     = "abort-incomplete-multipart-uploads"
    status = "Enabled"

    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

resource "aws_s3_bucket_policy" "artifact_deny_insecure_transport" {
  count = var.create_artifact_bucket ? 1 : 0

  bucket = aws_s3_bucket.artifact[0].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "DenyInsecureTransport"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource = [
          aws_s3_bucket.artifact[0].arn,
          "${aws_s3_bucket.artifact[0].arn}/*",
        ]
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      },
    ]
  })

  depends_on = [aws_s3_bucket_public_access_block.artifact]
}

resource "aws_s3_bucket" "dataset" {
  bucket        = local.dataset_bucket_name
  force_destroy = var.dataset_bucket_force_destroy

  tags = merge(local.tags, {
    Name    = "${var.stack_name}-datasets"
    Purpose = "burn-dragon-p2p-browser-datasets"
  })
}

resource "aws_s3_bucket_public_access_block" "dataset" {
  bucket = aws_s3_bucket.dataset.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "dataset" {
  bucket = aws_s3_bucket.dataset.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "dataset" {
  bucket = aws_s3_bucket.dataset.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = var.dataset_bucket_server_side_encryption
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "dataset" {
  bucket = aws_s3_bucket.dataset.id

  rule {
    id     = "abort-incomplete-multipart-uploads"
    status = "Enabled"

    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

resource "aws_s3_bucket_cors_configuration" "dataset" {
  bucket = aws_s3_bucket.dataset.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "HEAD"]
    allowed_origins = [local.browser_app_origin]
    expose_headers  = ["Content-Length", "Content-Type", "ETag"]
    max_age_seconds = 3600
  }
}

resource "aws_cloudfront_response_headers_policy" "dataset_cors" {
  name = "${var.stack_name}-${terraform.workspace}-dataset-cors"

  cors_config {
    access_control_allow_credentials = false
    access_control_allow_headers {
      items = ["*"]
    }
    access_control_allow_methods {
      items = ["GET", "HEAD", "OPTIONS"]
    }
    access_control_allow_origins {
      items = [local.browser_app_origin]
    }
    access_control_expose_headers {
      items = ["Content-Length", "Content-Type", "ETag"]
    }
    access_control_max_age_sec = 3600
    origin_override            = true
  }
}

resource "aws_acm_certificate" "dataset" {
  provider          = aws.us_east_1
  domain_name       = local.dataset_domain_name
  validation_method = "DNS"
  depends_on        = [aws_route53_record.dataset_caa]

  lifecycle {
    create_before_destroy = true
  }

  tags = local.tags
}

resource "aws_route53_record" "dataset_caa" {
  allow_overwrite = true
  zone_id         = data.aws_route53_zone.selected.zone_id
  name            = local.dataset_domain_name
  type            = "CAA"
  ttl             = 300
  records = [
    "0 issue \"amazon.com\"",
    "0 issue \"amazontrust.com\"",
    "0 issue \"amazonaws.com\"",
    "0 issue \"awstrust.com\"",
  ]
}

resource "aws_route53_record" "dataset_certificate_validation" {
  allow_overwrite = true
  zone_id         = data.aws_route53_zone.selected.zone_id
  name            = one(aws_acm_certificate.dataset.domain_validation_options).resource_record_name
  type            = one(aws_acm_certificate.dataset.domain_validation_options).resource_record_type
  ttl             = 60
  records         = [one(aws_acm_certificate.dataset.domain_validation_options).resource_record_value]
}

resource "aws_acm_certificate_validation" "dataset" {
  provider                = aws.us_east_1
  certificate_arn         = aws_acm_certificate.dataset.arn
  validation_record_fqdns = [aws_route53_record.dataset_certificate_validation.fqdn]
  depends_on              = [aws_route53_record.dataset_caa]
}

resource "aws_cloudfront_origin_access_control" "dataset" {
  name                              = "${var.stack_name}-${terraform.workspace}-dataset"
  description                       = "burn_dragon_p2p managed browser dataset origin"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

resource "aws_cloudfront_distribution" "dataset" {
  enabled         = true
  is_ipv6_enabled = true
  aliases         = [local.dataset_domain_name]

  origin {
    domain_name              = aws_s3_bucket.dataset.bucket_regional_domain_name
    origin_id                = "dataset-s3"
    origin_access_control_id = aws_cloudfront_origin_access_control.dataset.id

    s3_origin_config {
      origin_access_identity = ""
    }
  }

  default_cache_behavior {
    allowed_methods            = ["GET", "HEAD", "OPTIONS"]
    cached_methods             = ["GET", "HEAD", "OPTIONS"]
    cache_policy_id            = data.aws_cloudfront_cache_policy.caching_optimized.id
    compress                   = true
    response_headers_policy_id = aws_cloudfront_response_headers_policy.dataset_cors.id
    target_origin_id           = "dataset-s3"
    viewer_protocol_policy     = "redirect-to-https"
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate_validation.dataset.certificate_arn
    minimum_protocol_version = "TLSv1.2_2021"
    ssl_support_method       = "sni-only"
  }

  tags = merge(local.tags, {
    Name    = "${var.stack_name}-dataset-cdn"
    Purpose = "burn-dragon-p2p-browser-dataset-cdn"
  })
}

resource "aws_s3_bucket_policy" "dataset" {
  bucket = aws_s3_bucket.dataset.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "DenyInsecureTransport"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource = [
          aws_s3_bucket.dataset.arn,
          "${aws_s3_bucket.dataset.arn}/*",
        ]
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      },
      {
        Sid    = "AllowCloudFrontRead"
        Effect = "Allow"
        Principal = {
          Service = "cloudfront.amazonaws.com"
        }
        Action   = ["s3:GetObject"]
        Resource = ["${aws_s3_bucket.dataset.arn}/*"]
        Condition = {
          StringEquals = {
            "AWS:SourceArn" = aws_cloudfront_distribution.dataset.arn
          }
        }
      },
    ]
  })

  depends_on = [aws_s3_bucket_public_access_block.dataset]
}

resource "aws_route53_record" "dataset_distribution" {
  allow_overwrite = true
  zone_id         = data.aws_route53_zone.selected.zone_id
  name            = local.dataset_domain_name
  type            = "A"

  alias {
    name                   = aws_cloudfront_distribution.dataset.domain_name
    zone_id                = aws_cloudfront_distribution.dataset.hosted_zone_id
    evaluate_target_health = false
  }
}

resource "aws_route53_record" "dataset_distribution_ipv6" {
  allow_overwrite = true
  zone_id         = data.aws_route53_zone.selected.zone_id
  name            = local.dataset_domain_name
  type            = "AAAA"

  alias {
    name                   = aws_cloudfront_distribution.dataset.domain_name
    zone_id                = aws_cloudfront_distribution.dataset.hosted_zone_id
    evaluate_target_health = false
  }
}

resource "aws_route53_record" "browser_app_pages" {
  count           = local.browser_app_pages_record_enabled ? 1 : 0
  allow_overwrite = true
  zone_id         = data.aws_route53_zone.selected.zone_id
  name            = local.browser_app_hostname
  type            = "CNAME"
  ttl             = 300
  records         = [local.browser_app_pages_domain_target]
}

resource "aws_s3_bucket" "artifact_replica" {
  provider = aws.dr
  count    = local.disaster_recovery_enabled && var.create_artifact_replica_bucket ? 1 : 0

  bucket        = local.artifact_replica_bucket_name
  force_destroy = var.artifact_replica_bucket_force_destroy

  tags = merge(local.tags, {
    Name    = "${var.stack_name}-artifacts-dr"
    Purpose = "burn-dragon-p2p-artifact-dr-replica"
  })
}

resource "aws_s3_bucket_public_access_block" "artifact_replica" {
  provider = aws.dr
  count    = local.disaster_recovery_enabled && var.create_artifact_replica_bucket ? 1 : 0

  bucket = aws_s3_bucket.artifact_replica[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "artifact_replica" {
  provider = aws.dr
  count    = local.disaster_recovery_enabled && var.create_artifact_replica_bucket ? 1 : 0

  bucket = aws_s3_bucket.artifact_replica[0].id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifact_replica" {
  provider = aws.dr
  count    = local.disaster_recovery_enabled && var.create_artifact_replica_bucket ? 1 : 0

  bucket = aws_s3_bucket.artifact_replica[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = var.artifact_bucket_server_side_encryption
    }
  }
}

resource "aws_s3_bucket_policy" "artifact_replica_deny_insecure_transport" {
  provider = aws.dr
  count    = local.disaster_recovery_enabled && var.create_artifact_replica_bucket ? 1 : 0

  bucket = aws_s3_bucket.artifact_replica[0].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "DenyInsecureTransport"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource = [
          aws_s3_bucket.artifact_replica[0].arn,
          "${aws_s3_bucket.artifact_replica[0].arn}/*",
        ]
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      },
    ]
  })

  depends_on = [aws_s3_bucket_public_access_block.artifact_replica]
}

resource "aws_iam_role" "artifact_replication" {
  count = local.disaster_recovery_enabled ? 1 : 0

  name = "${var.stack_name}-artifact-replication"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      },
    ]
  })

  tags = local.tags
}

resource "aws_iam_role_policy" "artifact_replication" {
  count = local.disaster_recovery_enabled ? 1 : 0

  name = "${var.stack_name}-artifact-replication"
  role = aws_iam_role.artifact_replication[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetReplicationConfiguration",
          "s3:ListBucket",
        ]
        Resource = [local.artifact_bucket_arn]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObjectVersionForReplication",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectVersionTagging",
        ]
        Resource = [local.artifact_bucket_object_arn]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ReplicateObject",
          "s3:ReplicateDelete",
          "s3:ReplicateTags",
          "s3:ObjectOwnerOverrideToBucketOwner",
        ]
        Resource = [local.artifact_replica_bucket_object_arn]
      },
    ]
  })
}

resource "aws_s3_bucket_replication_configuration" "artifact" {
  count = local.disaster_recovery_enabled ? 1 : 0

  bucket = local.artifact_bucket_name
  role   = aws_iam_role.artifact_replication[0].arn

  rule {
    id       = "warm-disaster-recovery"
    priority = 1
    status   = "Enabled"

    delete_marker_replication {
      status = "Enabled"
    }

    existing_object_replication {
      status = "Enabled"
    }

    filter {
      prefix = local.artifact_bucket_path_prefix
    }

    destination {
      bucket        = local.artifact_replica_bucket_arn
      storage_class = "STANDARD"
    }
  }

  depends_on = [
    aws_s3_bucket_versioning.artifact,
    aws_s3_bucket_versioning.artifact_replica,
  ]
}

resource "aws_ebs_volume" "bootstrap_data" {
  count = local.use_retained_bootstrap_data_volume ? 1 : 0

  availability_zone = aws_subnet.public.availability_zone
  size              = var.data_volume_size_gib
  type              = var.data_volume_type
  encrypted         = true
  snapshot_id       = trimspace(var.bootstrap_primary_restore_snapshot_id) != "" ? trimspace(var.bootstrap_primary_restore_snapshot_id) : null

  tags = merge(local.tags, {
    Name           = "${var.stack_name}-bootstrap-data"
    SnapshotPolicy = local.bootstrap_data_snapshot_tag
    Persistence    = "retained-bootstrap-state"
    NodeRole       = "primary"
  })
}

resource "aws_instance" "bootstrap" {
  ami                         = data.aws_ami.ubuntu.id
  instance_type               = var.instance_type
  subnet_id                   = aws_subnet.public.id
  private_ip                  = local.bootstrap_primary_private_ip
  vpc_security_group_ids      = [aws_security_group.bootstrap.id]
  iam_instance_profile        = aws_iam_instance_profile.bootstrap.name
  associate_public_ip_address = true

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  root_block_device {
    volume_size           = var.root_volume_size_gib
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  user_data_base64 = base64gzip(templatefile("${path.module}/templates/user-data.sh.tftpl", {
    artifact_bucket_name                             = local.artifact_bucket_name
    artifact_bucket_path_prefix                      = local.artifact_bucket_path_prefix
    artifact_bucket_server_side_encryption           = var.artifact_bucket_server_side_encryption
    aws_region                                       = var.aws_region
    bootstrap_auth_feature                           = local.bootstrap_auth_feature
    bootstrap_auth_root                              = local.bootstrap_auth_root
    bootstrap_config_json                            = local.bootstrap_config_json
    bootstrap_data_device_name                       = var.data_volume_device_name
    bootstrap_data_mount_path                        = local.bootstrap_data_mount_path
    bootstrap_data_volume_id                         = local.use_retained_bootstrap_data_volume ? aws_ebs_volume.bootstrap_data[0].id : ""
    bootstrap_crate_version                          = var.bootstrap_crate_version
    bootstrap_head_mirror_auth_script                = local.bootstrap_head_mirror_fetch_auth_script
    bootstrap_head_mirror_auth_bundle_parameter_name = local.bootstrap_head_mirror_auth_bundle_parameter_name
    bootstrap_head_mirror_config                     = local.bootstrap_head_mirror_config
    bootstrap_head_mirror_service_unit               = local.bootstrap_head_mirror_service_unit
    bootstrap_git_ref                                = var.bootstrap_git_ref
    bootstrap_git_repo                               = var.bootstrap_git_repository
    bootstrap_install_source                         = local.bootstrap_install_source
    bootstrap_node_role                              = "primary"
    caddyfile                                        = local.caddyfile
    dragon_crate_version                             = var.dragon_crate_version
    dragon_git_ref                                   = var.dragon_git_ref
    dragon_git_repo                                  = var.dragon_git_repository
    authority_key_parameter_name                     = local.secret_parameter_names.authority_key
    http_port                                        = var.http_port
    secret_sync_script                               = local.secret_sync_script
    use_retained_bootstrap_data_volume               = local.use_retained_bootstrap_data_volume
  }))

  tags = merge(local.tags, {
    Name = "${var.stack_name}-bootstrap"
  })
}

resource "aws_volume_attachment" "bootstrap_data" {
  count = local.use_retained_bootstrap_data_volume ? 1 : 0

  device_name = var.data_volume_device_name
  volume_id   = aws_ebs_volume.bootstrap_data[0].id
  instance_id = aws_instance.bootstrap.id

  stop_instance_before_detaching = true
}

resource "aws_iam_role" "bootstrap_data_snapshot" {
  count = local.use_retained_bootstrap_data_volume && var.enable_data_volume_snapshots ? 1 : 0

  name = "${var.stack_name}-bootstrap-data-snapshot"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "dlm.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      },
    ]
  })

  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "bootstrap_data_snapshot" {
  count = local.use_retained_bootstrap_data_volume && var.enable_data_volume_snapshots ? 1 : 0

  role       = aws_iam_role.bootstrap_data_snapshot[0].name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AWSDataLifecycleManagerServiceRole"
}

resource "aws_dlm_lifecycle_policy" "bootstrap_data" {
  count = local.use_retained_bootstrap_data_volume && var.enable_data_volume_snapshots ? 1 : 0

  description        = "${var.stack_name} retained bootstrap data volume snapshots"
  execution_role_arn = aws_iam_role.bootstrap_data_snapshot[0].arn
  state              = "ENABLED"

  policy_details {
    resource_types = ["VOLUME"]
    target_tags = {
      SnapshotPolicy = local.bootstrap_data_snapshot_tag
    }

    schedule {
      name = "daily-bootstrap-data"

      create_rule {
        interval      = 24
        interval_unit = "HOURS"
        times         = [var.data_volume_snapshot_time_utc]
      }

      retain_rule {
        count = var.data_volume_snapshot_retention_days
      }

      copy_tags = true
      tags_to_add = merge(local.tags, {
        SnapshotSource = "${var.stack_name}-bootstrap-data"
      })

      dynamic "cross_region_copy_rule" {
        for_each = local.disaster_recovery_snapshot_copies_enabled ? [1] : []
        content {
          target    = local.disaster_recovery_region
          encrypted = true
          copy_tags = true

          retain_rule {
            interval      = var.disaster_recovery_snapshot_retention_days
            interval_unit = "DAYS"
          }
        }
      }
    }
  }

  tags = local.tags
}

resource "aws_cloudwatch_metric_alarm" "bootstrap_status_check_failed_instance" {
  count = var.enable_bootstrap_status_alarms ? 1 : 0

  alarm_name          = "${var.stack_name}-bootstrap-status-check-failed-instance"
  alarm_description   = "burn_dragon_p2p bootstrap EC2 instance status check failure"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 2
  metric_name         = "StatusCheckFailed_Instance"
  namespace           = "AWS/EC2"
  period              = 60
  statistic           = "Maximum"
  threshold           = 1
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.cloudwatch_alarm_actions
  ok_actions          = local.cloudwatch_alarm_actions

  dimensions = {
    InstanceId = aws_instance.bootstrap.id
  }

  tags = local.tags
}

resource "aws_cloudwatch_metric_alarm" "bootstrap_status_check_failed_system" {
  count = var.enable_bootstrap_status_alarms ? 1 : 0

  alarm_name          = "${var.stack_name}-bootstrap-status-check-failed-system"
  alarm_description   = "burn_dragon_p2p bootstrap EC2 system status check failure"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 2
  metric_name         = "StatusCheckFailed_System"
  namespace           = "AWS/EC2"
  period              = 60
  statistic           = "Maximum"
  threshold           = 1
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.cloudwatch_alarm_actions
  ok_actions          = local.cloudwatch_alarm_actions

  dimensions = {
    InstanceId = aws_instance.bootstrap.id
  }

  tags = local.tags
}

resource "aws_cloudwatch_metric_alarm" "control_plane_redis_engine_cpu_high" {
  count = var.enable_control_plane_operational_alarms && local.managed_control_plane_redis_enabled ? 1 : 0

  alarm_name          = "${var.stack_name}-redis-engine-cpu-high"
  alarm_description   = "burn_dragon_p2p shared Redis EngineCPUUtilization is above the configured threshold"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 5
  metric_name         = "EngineCPUUtilization"
  namespace           = "AWS/ElastiCache"
  period              = 60
  statistic           = "Average"
  threshold           = var.redis_engine_cpu_alarm_threshold_percent
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.cloudwatch_alarm_actions
  ok_actions          = local.cloudwatch_alarm_actions

  dimensions = {
    ReplicationGroupId = aws_elasticache_replication_group.control_plane[0].replication_group_id
  }

  tags = local.tags
}

resource "aws_cloudwatch_metric_alarm" "control_plane_redis_freeable_memory_low" {
  count = var.enable_control_plane_operational_alarms && local.managed_control_plane_redis_enabled ? 1 : 0

  alarm_name          = "${var.stack_name}-redis-freeable-memory-low"
  alarm_description   = "burn_dragon_p2p shared Redis freeable memory is below the configured threshold"
  comparison_operator = "LessThanOrEqualToThreshold"
  evaluation_periods  = 3
  metric_name         = "FreeableMemory"
  namespace           = "AWS/ElastiCache"
  period              = 300
  statistic           = "Average"
  threshold           = var.redis_freeable_memory_alarm_threshold_bytes
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.cloudwatch_alarm_actions
  ok_actions          = local.cloudwatch_alarm_actions

  dimensions = {
    ReplicationGroupId = aws_elasticache_replication_group.control_plane[0].replication_group_id
  }

  tags = local.tags
}

resource "aws_cloudwatch_metric_alarm" "edge_primary_health_check_unhealthy" {
  count = var.enable_control_plane_operational_alarms ? 1 : 0

  alarm_name          = "${var.stack_name}-edge-primary-unhealthy"
  alarm_description   = "burn_dragon_p2p primary edge Route53 health check is unhealthy"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 2
  metric_name         = "HealthCheckStatus"
  namespace           = "AWS/Route53"
  period              = 60
  statistic           = "Minimum"
  threshold           = 1
  treat_missing_data  = "breaching"
  alarm_actions       = local.cloudwatch_alarm_actions
  ok_actions          = local.cloudwatch_alarm_actions

  dimensions = {
    HealthCheckId = aws_route53_health_check.edge_primary.id
  }

  tags = local.tags
}

resource "aws_cloudwatch_metric_alarm" "dataset_cdn_5xx_error_rate_high" {
  count = var.enable_control_plane_operational_alarms ? 1 : 0

  alarm_name          = "${var.stack_name}-dataset-cdn-5xx-high"
  alarm_description   = "burn_dragon_p2p managed dataset CDN 5xx error rate is above the configured threshold"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 3
  metric_name         = "5xxErrorRate"
  namespace           = "AWS/CloudFront"
  period              = 300
  statistic           = "Average"
  threshold           = var.dataset_cdn_5xx_error_rate_alarm_threshold_percent
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.cloudwatch_alarm_actions
  ok_actions          = local.cloudwatch_alarm_actions

  dimensions = {
    DistributionId = aws_cloudfront_distribution.dataset.id
    Region         = "Global"
  }

  tags = local.tags
}

resource "aws_cloudwatch_metric_alarm" "managed_trainer_in_service_low" {
  count = var.enable_control_plane_operational_alarms && local.managed_trainer_enabled && var.managed_trainer_desired_capacity > 0 ? 1 : 0

  alarm_name          = "${var.stack_name}-managed-trainer-in-service-low"
  alarm_description   = "burn_dragon_p2p managed trainer in-service capacity is below the configured desired capacity"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 5
  metric_name         = "GroupInServiceInstances"
  namespace           = "AWS/AutoScaling"
  period              = 60
  statistic           = "Average"
  threshold           = local.managed_trainer_alarm_threshold
  treat_missing_data  = "breaching"
  alarm_actions       = local.cloudwatch_alarm_actions
  ok_actions          = local.cloudwatch_alarm_actions

  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.managed_trainer[0].name
  }

  tags = local.tags
}

resource "aws_cloudwatch_dashboard" "control_plane" {
  count          = var.enable_control_plane_dashboard ? 1 : 0
  dashboard_name = local.control_plane_dashboard_name
  dashboard_body = jsonencode({
    widgets = concat([
      {
        type   = "text"
        x      = 0
        y      = 0
        width  = 24
        height = 3
        properties = {
          markdown = join("\n", concat([
            "# burn_dragon control plane",
            "- edge: https://${var.edge_domain_name}",
            "- dataset cdn: https://${local.dataset_domain_name}",
            "- artifact bucket: ${local.artifact_bucket_name}",
            ], local.managed_control_plane_redis_enabled ? [
            "- control-plane state backend: redis (${aws_elasticache_replication_group.control_plane[0].primary_endpoint_address})",
            ] : [
            "- control-plane state backend: local-file on bootstrap root volume",
          ]))
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 3
        width  = 12
        height = 6
        properties = {
          title   = "Edge health checks"
          region  = var.aws_region
          period  = 60
          stat    = "Minimum"
          view    = "timeSeries"
          stacked = false
          metrics = [
            ["AWS/Route53", "HealthCheckStatus", "HealthCheckId", aws_route53_health_check.edge_primary.id, { label = "edge" }],
          ]
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 3
        width  = 12
        height = 6
        properties = {
          title   = "Bootstrap EC2 CPU"
          region  = var.aws_region
          period  = 300
          stat    = "Average"
          view    = "timeSeries"
          stacked = false
          metrics = [
            ["AWS/EC2", "CPUUtilization", "InstanceId", aws_instance.bootstrap.id, { label = "bootstrap" }],
          ]
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 15
        width  = 12
        height = 6
        properties = {
          title   = "Managed dataset CDN"
          region  = var.aws_region
          period  = 300
          stat    = "Average"
          view    = "timeSeries"
          stacked = false
          metrics = [
            ["AWS/CloudFront", "Requests", "DistributionId", aws_cloudfront_distribution.dataset.id, "Region", "Global", { label = "requests", stat = "Sum" }],
            [".", "4xxErrorRate", ".", ".", ".", ".", { label = "4xx error rate %" }],
            [".", "5xxErrorRate", ".", ".", ".", ".", { label = "5xx error rate %" }],
          ]
        }
      },
      ], local.managed_control_plane_redis_enabled ? [
      {
        type   = "metric"
        x      = 0
        y      = 9
        width  = 12
        height = 6
        properties = {
          title   = "Control plane Redis"
          region  = var.aws_region
          period  = 60
          stat    = "Average"
          view    = "timeSeries"
          stacked = false
          metrics = [
            ["AWS/ElastiCache", "EngineCPUUtilization", "ReplicationGroupId", aws_elasticache_replication_group.control_plane[0].replication_group_id, { label = "engine cpu %" }],
            [".", "CurrConnections", ".", ".", { label = "connections", yAxis = "right" }],
          ]
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 9
        width  = 12
        height = 6
        properties = {
          title   = "Control plane Redis memory"
          region  = var.aws_region
          period  = 300
          stat    = "Average"
          view    = "timeSeries"
          stacked = false
          metrics = [
            ["AWS/ElastiCache", "FreeableMemory", "ReplicationGroupId", aws_elasticache_replication_group.control_plane[0].replication_group_id, { label = "freeable memory bytes" }],
            [".", "Evictions", ".", ".", { label = "evictions", yAxis = "right" }],
          ]
        }
      },
      ] : [], local.managed_trainer_enabled ? [
      {
        type   = "metric"
        x      = 12
        y      = 15
        width  = 12
        height = 6
        properties = {
          title   = "Managed trainer pool"
          region  = var.aws_region
          period  = 60
          stat    = "Average"
          view    = "timeSeries"
          stacked = false
          metrics = [
            ["AWS/AutoScaling", "GroupDesiredCapacity", "AutoScalingGroupName", aws_autoscaling_group.managed_trainer[0].name, { label = "desired" }],
            [".", "GroupInServiceInstances", ".", ".", { label = "in service" }],
            [".", "GroupTotalInstances", ".", ".", { label = "total" }],
          ]
        }
      },
    ] : [])
  })
}

resource "aws_eip" "bootstrap" {
  domain   = "vpc"
  instance = aws_instance.bootstrap.id

  tags = merge(local.tags, {
    Name = "${var.stack_name}-bootstrap"
  })
}

resource "aws_route53_health_check" "edge_primary" {
  ip_address        = aws_eip.bootstrap.public_ip
  type              = "TCP"
  port              = 443
  request_interval  = 30
  failure_threshold = 3

  tags = merge(local.tags, {
    Name = "${var.stack_name}-edge-primary"
  })
}

resource "aws_route53_record" "edge_primary" {
  allow_overwrite = true
  zone_id         = data.aws_route53_zone.selected.zone_id
  name            = var.edge_domain_name
  type            = "A"
  ttl             = 60
  records         = [aws_eip.bootstrap.public_ip]
}
