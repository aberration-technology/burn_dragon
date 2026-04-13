variable "aws_region" {
  description = "AWS region for the burn_dragon_p2p bootstrap deployment."
  type        = string
  default     = "us-east-1"
}

variable "disaster_recovery_region" {
  description = "Optional warm-disaster-recovery AWS region. When set, Terraform enables cross-region artifact replication and cross-region copies of retained bootstrap data snapshots."
  type        = string
  default     = ""
}

variable "stack_name" {
  description = "Logical stack name used for tags and DNS outputs."
  type        = string
}

variable "environment_name" {
  description = "Human-readable environment label."
  type        = string
  default     = "production"
}

variable "route53_zone_name" {
  description = "Public Route53 hosted zone name that should contain the edge record. Defaults to the aberration.technology production zone."
  type        = string
  default     = "aberration.technology"
}

variable "edge_domain_name" {
  description = "Public domain served by the burn_p2p bootstrap browser edge. Defaults to the production root at dragon.aberration.technology."
  type        = string
  default     = "dragon.aberration.technology"
}

variable "bootstrap_install_source" {
  description = "How the bootstrap hosts install burn_p2p_bootstrap. Supported values: crate or git. Production deployments should use the published crate by default."
  type        = string
  default     = "crate"

  validation {
    condition     = contains(["crate", "git"], lower(trimspace(var.bootstrap_install_source)))
    error_message = "bootstrap_install_source must be one of: crate, git."
  }
}

variable "bootstrap_crate_version" {
  description = "Published burn_p2p_bootstrap crate version installed on the bootstrap hosts when bootstrap_install_source = crate."
  type        = string
  default     = "0.21.0-pre.12"
}

variable "bootstrap_git_repository" {
  description = "Git repository used to install burn_p2p_bootstrap when bootstrap_install_source = git."
  type        = string
  default     = "https://github.com/aberration-technology/burn_p2p.git"
}

variable "bootstrap_git_ref" {
  description = "Pinned burn_p2p git ref used to install burn_p2p_bootstrap when bootstrap_install_source = git."
  type        = string
  default     = ""
}

variable "secret_parameter_prefix" {
  description = "SSM parameter prefix used for runtime secrets read by the bootstrap host."
  type        = string
}

variable "network_id" {
  description = "burn_p2p network id exposed by the bootstrap edge."
  type        = string
  default     = "burn-dragon-mainnet"
}

variable "auth_connector_kind" {
  description = "Bootstrap auth connector kind. Supported values: github, oidc, oauth, static, external."
  type        = string
  default     = "github"

  validation {
    condition = contains(
      ["github", "oidc", "oauth", "static", "external"],
      lower(trimspace(var.auth_connector_kind))
    )
    error_message = "auth_connector_kind must be one of: github, oidc, oauth, static, external."
  }
}

variable "auth_authority_name" {
  description = "Logical authority name exposed by the bootstrap auth portal."
  type        = string
  default     = "burn-dragon-auth"
}

variable "auth_principals_json" {
  description = "Optional JSON array of bootstrap auth principals. Use this for static, external, oidc, or oauth deployments that need seeded operator/admin principals without GitHub provider-policy rules."
  type        = string
  default     = "[]"

  validation {
    condition     = can(jsondecode(var.auth_principals_json))
    error_message = "auth_principals_json must be valid JSON."
  }
}

variable "auth_authorize_base_url" {
  description = "Optional auth connector authorize endpoint override."
  type        = string
  default     = ""
}

variable "auth_exchange_url" {
  description = "Optional auth connector token exchange endpoint override."
  type        = string
  default     = ""
}

variable "auth_token_url" {
  description = "Optional auth connector token endpoint override."
  type        = string
  default     = ""
}

variable "auth_api_base_url" {
  description = "Optional auth connector API base URL override. Primarily useful for GitHub-compatible providers."
  type        = string
  default     = ""
}

variable "auth_userinfo_url" {
  description = "Optional auth connector userinfo endpoint override."
  type        = string
  default     = ""
}

variable "auth_refresh_url" {
  description = "Optional auth connector refresh endpoint override."
  type        = string
  default     = ""
}

variable "auth_revoke_url" {
  description = "Optional auth connector revoke endpoint override."
  type        = string
  default     = ""
}

variable "auth_jwks_url" {
  description = "Optional auth connector JWKS endpoint override."
  type        = string
  default     = ""
}

variable "auth_oidc_issuer" {
  description = "OIDC issuer URL when auth_connector_kind = oidc."
  type        = string
  default     = ""
}

variable "auth_oauth_provider" {
  description = "Provider label when auth_connector_kind = oauth."
  type        = string
  default     = ""
}

variable "auth_external_authority" {
  description = "Trusted external authority label when auth_connector_kind = external."
  type        = string
  default     = ""
}

variable "auth_external_trusted_principal_header" {
  description = "Trusted ingress header carrying the authenticated principal when auth_connector_kind = external."
  type        = string
  default     = "x-forwarded-user"
}

variable "auth_external_trusted_internal_only" {
  description = "Whether the external connector should trust only internal traffic for the principal header."
  type        = bool
  default     = true
}

variable "project_family_id" {
  description = "Project family id required by the GitHub auth portal."
  type        = string
  default     = "burn-dragon-language"
}

variable "study_id" {
  description = "Study id shared by the Dragon experiment directory entries."
  type        = string
  default     = "burn-dragon-mainnet"
}

variable "release_train_hash" {
  description = "Required release train hash enforced by the bootstrap auth portal."
  type        = string
  default     = "burn-dragon-mainnet-train"
}

variable "native_target_artifact_hash" {
  description = "Artifact hash label granted to native peers by the bootstrap auth portal."
  type        = string
  default     = "burn-dragon-native"
}

variable "browser_target_artifact_hash" {
  description = "Artifact hash label granted to browser peers by the bootstrap auth portal."
  type        = string
  default     = "burn-dragon-browser"
}

variable "github_required_org" {
  description = "GitHub organization required for peer admission."
  type        = string
  default     = ""
}

variable "github_required_team" {
  description = "Optional GitHub team slug (org/team) required for peer admission."
  type        = string
  default     = ""
}

variable "github_required_repo" {
  description = "Repository access rule used for peer admission."
  type        = string
  default     = "mosure/burn_dragon"
}

variable "github_admin_logins" {
  description = "Explicit GitHub logins granted session-authenticated admin rights for this deployment."
  type        = list(string)
  default     = []
}

variable "github_admin_required_repo_permission" {
  description = "Minimum GitHub repository permission required for explicitly listed admin logins."
  type        = string
  default     = "admin"
}

variable "dataset_domain_name" {
  description = "Public CloudFront hostname serving browser training datasets. Leave empty to derive datasets.<edge_domain_name>."
  type        = string
  default     = ""
}

variable "dataset_bucket_name" {
  description = "Optional desired S3 bucket name for managed browser dataset distribution. Leave empty to let Terraform derive a stable bucket name."
  type        = string
  default     = ""
}

variable "dataset_bucket_force_destroy" {
  description = "Whether Terraform may destroy the managed browser dataset S3 bucket even if it still contains published shard manifests and shard bytes."
  type        = bool
  default     = false
}

variable "dataset_bucket_path_prefix" {
  description = "Optional key prefix inside the managed browser dataset S3 bucket. Leave empty to use dragon-datasets."
  type        = string
  default     = ""
}

variable "dataset_bucket_server_side_encryption" {
  description = "Server-side encryption mode enforced for the managed browser dataset S3 bucket."
  type        = string
  default     = "AES256"
}

variable "climbmix_browser_dataset_base_url" {
  description = "Optional explicit public base URL for the browser ClimbMix shard pool. Leave empty to use the managed dataset CDN path under dataset_domain_name."
  type        = string
  default     = ""
}

variable "github_principal_id" {
  description = "Principal id assigned to admitted GitHub operators."
  type        = string
  default     = "burn-dragon-contributor"
}

variable "instance_type" {
  description = "EC2 instance type for the bootstrap edge."
  type        = string
  default     = "t3.large"
}

variable "root_volume_size_gib" {
  description = "Root EBS volume size for the bootstrap edge instance."
  type        = number
  default     = 256
}

variable "data_volume_size_gib" {
  description = "Dedicated retained EBS data volume size for bootstrap/auth/publication state."
  type        = number
  default     = 512
}

variable "data_volume_type" {
  description = "EBS volume type for the retained bootstrap data volume."
  type        = string
  default     = "gp3"
}

variable "data_volume_device_name" {
  description = "EC2 device name requested for the retained bootstrap data volume attachment."
  type        = string
  default     = "/dev/sdf"
}

variable "enable_data_volume_snapshots" {
  description = "Whether Terraform should manage automatic snapshots for the retained bootstrap data volume."
  type        = bool
  default     = true
}

variable "data_volume_snapshot_retention_days" {
  description = "How many daily retained bootstrap data volume snapshots to keep when automatic snapshots are enabled."
  type        = number
  default     = 14
}

variable "data_volume_snapshot_time_utc" {
  description = "UTC time for the daily retained bootstrap data volume snapshot, in HH:MM format."
  type        = string
  default     = "03:00"
}

variable "enable_disaster_recovery_snapshot_copies" {
  description = "Whether Terraform should copy retained bootstrap data volume snapshots into disaster_recovery_region when that region is configured."
  type        = bool
  default     = true
}

variable "disaster_recovery_snapshot_retention_days" {
  description = "How many daily copied bootstrap data snapshots to keep in disaster_recovery_region when warm-DR snapshot copies are enabled."
  type        = number
  default     = 14
}

variable "bootstrap_primary_restore_snapshot_id" {
  description = "Optional EBS snapshot id used to restore the primary bootstrap data volume. Leave empty for a normal retained-volume deployment."
  type        = string
  default     = ""
}

variable "bootstrap_secondary_restore_snapshot_id" {
  description = "Optional EBS snapshot id used to restore the secondary bootstrap data volume. Leave empty for a normal retained-volume deployment."
  type        = string
  default     = ""
}

variable "enable_bootstrap_status_alarms" {
  description = "Whether Terraform should create EC2 status-check CloudWatch alarms for the bootstrap host."
  type        = bool
  default     = true
}

variable "alarm_sns_topic_arn" {
  description = "Optional SNS topic ARN notified by bootstrap status-check CloudWatch alarms."
  type        = string
  default     = ""
}

variable "enable_control_plane_operational_alarms" {
  description = "Whether Terraform should create operational CloudWatch alarms for Redis, Route53 health checks, dataset CDN, and managed trainer capacity."
  type        = bool
  default     = true
}

variable "enable_control_plane_dashboard" {
  description = "Whether Terraform should create a shared CloudWatch dashboard for the Dragon control plane."
  type        = bool
  default     = true
}

variable "redis_engine_cpu_alarm_threshold_percent" {
  description = "Redis EngineCPUUtilization threshold that triggers the operational alarm."
  type        = number
  default     = 80
}

variable "redis_freeable_memory_alarm_threshold_bytes" {
  description = "Redis FreeableMemory threshold in bytes that triggers the low-memory operational alarm."
  type        = number
  default     = 268435456
}

variable "dataset_cdn_5xx_error_rate_alarm_threshold_percent" {
  description = "CloudFront 5xx error-rate percentage that triggers the managed dataset CDN alarm."
  type        = number
  default     = 5
}

variable "create_artifact_bucket" {
  description = "Whether Terraform should create the S3 bucket used for durable artifact publication from the bootstrap host."
  type        = bool
  default     = true
}

variable "artifact_bucket_name" {
  description = "Optional existing or desired S3 bucket name for published Dragon artifacts. Leave empty to let Terraform derive a stable bucket name."
  type        = string
  default     = ""
}

variable "artifact_bucket_force_destroy" {
  description = "Whether Terraform may destroy the artifact S3 bucket even if it still contains published checkpoints and metrics."
  type        = bool
  default     = false
}

variable "artifact_bucket_path_prefix" {
  description = "Optional key prefix inside the artifact S3 bucket. Leave empty to derive a stack/workspace-scoped prefix automatically."
  type        = string
  default     = ""
}

variable "artifact_bucket_server_side_encryption" {
  description = "Server-side encryption mode enforced for the artifact S3 bucket and direct bootstrap uploads."
  type        = string
  default     = "AES256"
}

variable "create_artifact_replica_bucket" {
  description = "Whether Terraform should create the warm-DR replica S3 bucket in disaster_recovery_region when cross-region replication is enabled."
  type        = bool
  default     = true
}

variable "artifact_replica_bucket_name" {
  description = "Optional existing or desired S3 bucket name in disaster_recovery_region for replicated Dragon artifacts. Leave empty to let Terraform derive a stable replica bucket name."
  type        = string
  default     = ""
}

variable "artifact_replica_bucket_force_destroy" {
  description = "Whether Terraform may destroy the disaster-recovery artifact replica S3 bucket even if it still contains replicated checkpoints and metrics."
  type        = bool
  default     = false
}

variable "managed_trainer_desired_capacity" {
  description = "Desired instance count for the optional managed native trainer pool. Set to 0 to disable managed trainers."
  type        = number
  default     = 0
}

variable "managed_trainer_min_size" {
  description = "Optional minimum size for the managed native trainer pool. Leave 0 to default to managed_trainer_desired_capacity."
  type        = number
  default     = 0
}

variable "managed_trainer_max_size" {
  description = "Optional maximum size for the managed native trainer pool. Leave 0 to default to managed_trainer_desired_capacity."
  type        = number
  default     = 0
}

variable "managed_trainer_instance_type" {
  description = "EC2 instance type used by the managed native trainer pool."
  type        = string
  default     = "g5.xlarge"
}

variable "managed_trainer_root_volume_size_gib" {
  description = "Root EBS volume size for managed native trainer instances."
  type        = number
  default     = 256
}

variable "managed_trainer_backend" {
  description = "Native backend used by the managed trainer pool. Supported values: cpu, wgpu, cuda."
  type        = string
  default     = "wgpu"

  validation {
    condition     = contains(["cpu", "wgpu", "cuda"], lower(trimspace(var.managed_trainer_backend)))
    error_message = "managed_trainer_backend must be one of: cpu, wgpu, cuda."
  }
}

variable "managed_trainer_experiment_kind" {
  description = "Experiment kind assigned to the managed trainer pool. Supported values: nca, climbmix."
  type        = string
  default     = "nca"

  validation {
    condition     = contains(["nca", "climbmix"], lower(trimspace(var.managed_trainer_experiment_kind)))
    error_message = "managed_trainer_experiment_kind must be one of: nca, climbmix."
  }
}

variable "managed_trainer_target" {
  description = "Native target mode used by managed trainer instances."
  type        = string
  default     = "trainer"
}

variable "managed_trainer_crate_version" {
  description = "Published burn_dragon_p2p crate version installed on managed trainer instances."
  type        = string
  default     = "0.21.0-pre.12"
}

variable "managed_trainer_auth_bundle_parameter_name" {
  description = "Optional SSM parameter name containing the JSON auth bundle used by managed trainer instances. Leave empty to derive a standard name under secret_parameter_prefix."
  type        = string
  default     = ""
}

variable "ssh_cidr_blocks" {
  description = "Optional SSH ingress CIDRs. Leave empty to rely on SSM/sessionless operation."
  type        = list(string)
  default     = []
}

variable "p2p_port" {
  description = "TCP/UDP port exposed for libp2p bootstrap traffic."
  type        = number
  default     = 4001
}

variable "http_port" {
  description = "Local burn_p2p bootstrap HTTP port behind Caddy."
  type        = number
  default     = 8787
}

variable "protocol_version" {
  description = "burn_p2p protocol version announced by the bootstrap edge."
  type        = string
  default     = "0.1.0"
}

variable "remaining_work_units" {
  description = "Bootstrap-side work unit budget surfaced to the directory."
  type        = number
  default     = 1000000
}

variable "local_artifact_retention_ttl_secs" {
  description = "Retention TTL for the local publication store."
  type        = number
  default     = 604800
}

variable "local_artifact_max_size_bytes" {
  description = "Maximum artifact size accepted by the local publication store."
  type        = number
  default     = 21474836480
}

variable "nca_min_device_memory_bytes" {
  description = "Advertised minimum trainer memory requirement for the NCA workload."
  type        = number
  default     = 2147483648
}

variable "climbmix_min_device_memory_bytes" {
  description = "Advertised minimum trainer memory requirement for the ClimbMix workload."
  type        = number
  default     = 6442450944
}

variable "min_system_memory_bytes" {
  description = "Advertised minimum system memory requirement for Dragon trainers."
  type        = number
  default     = 8589934592
}

variable "tags" {
  description = "Additional AWS tags applied to all resources."
  type        = map(string)
  default     = {}
}
