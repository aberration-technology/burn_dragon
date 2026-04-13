# burn_dragon_p2p Deploy

This folder contains the operator-facing deployment assets for the `burn_dragon_p2p` network:

- `native-peer.toml.example`: example native peer config
- `burn-dragon-p2p-native.service`: example native peer systemd unit
- `profiles/`: checked-in Dragon experiment profile sources and initial published profile payloads
- `datasets/README.md`: notes on shard-pool source material that stays out of git and can be published into the managed dataset CDN
- `terraform/aws`: the checked-in AWS bootstrap/edge deployment

The AWS Terraform root deploys a single-region bootstrap plane for the Dragon network:

- one EC2 bootstrap/edge host
- Route53 DNS for the browser edge
- no ALB or API Gateway; the browser edge is Caddy on the bootstrap host
- TLS termination via Caddy on the host
- bootstrap-local peer/runtime/auth state on the EC2 root volume by default
- bootstrap-managed direct S3 publication for checkpoint and metric artifacts using the EC2 instance role
- optional retained encrypted EBS data volume for bootstrap local peer/runtime/auth state when you want state to survive host replacement
- local file-backed auth session and operator state on the bootstrap host by default
- optional managed Redis node for auth session and operator state when you want externalized control-plane state
- bootstrap authority material synchronized through SSM and persisted for host replacement
- retained data-volume snapshots disabled by default and only relevant when retained bootstrap storage is enabled
- optional warm-disaster-recovery region with cross-region artifact replication plus retained-volume snapshot copies when retained bootstrap storage is enabled
- optional managed native trainer pool for always-on NCA or ClimbMix trainer capacity
- managed browser dataset S3 bucket plus CloudFront hostname for ClimbMix shard-pool distribution
- managed trainers are separate from the bootstrap nodes and default to disabled
- EC2, Redis, dataset CDN, Route53, and managed-trainer CloudWatch alarms, optionally wired to SNS
- shared CloudWatch dashboard for control-plane health and throughput
- configurable browser/native auth flow through `burn-p2p-bootstrap`

It does not attempt to manage every end-user native trainer. Native operators can still install and run `burn_dragon_p2p_native` locally, then point it at the deployed edge and seed URLs. The stack can also own a small managed native trainer pool for always-on capacity.

The bootstrap publishes initial Dragon experiment directory entries for:

- `nca-prepretraining`
- `climbmix-pretraining`

Those entries include Dragon profile metadata, so peers can resolve experiment and training configuration from the network instead of requiring a matching static local config.
The initial ClimbMix revision now defaults to the managed dataset CDN path under `https://datasets.dragon.aberration.technology/dragon-datasets/climbmix-pretraining/climbmix-r1`. The deploy workflow publishes `${base_url}/fetch-manifest.json` into the initial ClimbMix browser profile, and browser peers fetch only the shards they train on from that managed shard pool unless you override the base URL explicitly.

## Artifact Storage

Checkpoint artifacts, including model weights and exported metric bundles, are published directly from the bootstrap host into S3 using the EC2 instance role and the upstream `S3Compatible` publication target. When `disaster_recovery_region` is configured, Terraform also enables cross-region S3 replication into a warm-DR replica bucket.

There is no separate artifact node by default. The bootstrap/control-plane host owns artifact publication, durable artifact bytes live in S3, and bootstrap-local peer/runtime/auth/operator state lives on the root volume by default. If you opt into retained bootstrap storage, that local state moves onto a dedicated EBS volume. If you opt into managed Redis, auth session and operator state are externalized there. Cross-region retained-volume recovery is handled through copied EBS snapshots plus the restore workflow only when retained bootstrap storage is enabled.

## Managed Dataset Distribution

The stack now also owns a managed browser dataset origin for ClimbMix:

- S3 bucket for shard manifests and shard bytes
- CloudFront distribution on `dataset_domain_name`
- Route53 alias records and ACM certificate validation
- deploy-time default ClimbMix browser manifest URL derived from that managed CDN path

The intended operator entrypoint for publishing a shard pool into that managed origin is:

- `.github/workflows/publish-burn-dragon-p2p-dataset.yml`

That workflow syncs a source S3 prefix into the managed dataset bucket, invalidates the CloudFront distribution, and prints the resulting public base URL. The shard source material still stays out of git.

## One-Click GitHub Action

The intended operator entrypoint is:

- `.github/workflows/deploy-burn-dragon-p2p-aws.yml`

That workflow:

- seeds auth client credentials into AWS SSM Parameter Store when the selected auth connector needs them
- installs `burn_p2p_bootstrap` from the published crate by default, with an explicit git fallback only for testing unpublished upstream revisions
- runs `terraform fmt`, `init`, `validate`, `plan`, and `apply`
- creates or reuses the S3 bucket used for durable direct artifact publication
- optionally creates an autoscaled managed trainer pool that installs `burn_dragon_p2p_native` from crates.io and fetches its auth bundle from SSM
- auto-seeds a deploy-managed static trainer principal and mints its auth bundle after edge health when the trainer pool is enabled and no explicit bundle override secret is supplied
- configures explicit GitHub admin logins for session-authenticated admin access when the auth connector is `github`
- waits for the edge URL to answer over HTTPS
- prints the bootstrap instance details, bootstrap state-storage mode, control-plane state backend, control-plane dashboard URL, bootstrap install source/version, managed trainer pool outputs, and artifact plus dataset S3 prefixes in the workflow summary

If you trigger the workflow with a forced bootstrap replacement, Terraform replaces the bootstrap EC2 host. By default that also replaces bootstrap-local root-volume state. If you enable retained bootstrap storage, Terraform reattaches the retained data volume so local peer/runtime/auth state survives a normal rebuild. Artifact publication remains externalized in S3 either way.

The workflow still performs a Terraform plan internally before apply. That keeps the operator experience one-click without dropping the safety and auditability of a plan phase.

## Disaster Recovery Restore Workflow

The explicit restore and failover entrypoint is:

- `.github/workflows/restore-burn-dragon-p2p-aws.yml`

That workflow can:

- resolve the latest tagged retained bootstrap data-volume snapshot automatically when retained bootstrap storage is enabled
- run a `plan_only=true` disaster-recovery drill without applying
- restore the stack into a target region from explicit or auto-resolved snapshots when retained bootstrap storage is enabled
- optionally re-enable warm-DR replication on the restored stack by setting `next_disaster_recovery_region`
- reuse the normal `data_volume_size_gib` setting, but keep it greater than or equal to the source snapshot volume size when restoring from snapshots

Recommended warm-DR drill flow:

- set `terraform_workspace` to a drill-specific workspace like `dr-drill`
- set `aws_region` to the warm-DR region that receives copied snapshots
- leave `restore_from_latest_snapshots=true`
- keep `plan_only=true`

Recommended actual failover flow:

- set `terraform_workspace` to the production workspace you are failing over
- set `aws_region` to the warm-DR region
- leave `restore_from_latest_snapshots=true` unless you are pinning explicit snapshot ids
- set `plan_only=false`
- optionally set `next_disaster_recovery_region` if the restored stack should keep a warm-DR target

## GitHub Pages Browser Shell

The focused repo also ships a separate browser-shell workflow:

- `.github/workflows/deploy-burn-dragon-p2p-pages.yml`
- `.github/workflows/publish-burn-dragon-p2p-dataset.yml`

Before the workflow can publish, set the repository Pages source to `GitHub Actions` under `Settings > Pages`.

That workflow builds the standalone `burn_dragon_p2p_browser` wasm client through `xtask build-browser-site`, uploads the generated static bundle, and deploys it to GitHub Pages. The published shell is static; it still connects to the live edge URL you configure. By default, the baked browser config points at `https://dragon.aberration.technology` and derives the standard TCP and QUIC bootstrap multiaddrs from that host.

The deployed browser shell now includes an operator panel alongside the peer UI. It requests `Connect` and `Discover` by default, plus `Train` and `Validate` for the selected experiment id when one is baked into the shell. Operators can then use `Sign In (Admin)` from the browser to request an additional `ExperimentScope::Admin { study_id }` session for live directory edits. Under the default deployment, that browser login provider is GitHub.

Terraform now protects the Route53 hosted-zone apex by default. With the production defaults, the stack may manage `dragon.aberration.technology` and `datasets.dragon.aberration.technology`, but it will refuse to claim `aberration.technology` itself unless `allow_route53_zone_apex_records=true` is set explicitly in Terraform.

Optional GitHub repository variables for the Pages workflow:

- `BURN_DRAGON_P2P_PAGES_EDGE_BASE_URL`
- `BURN_DRAGON_P2P_PAGES_SEED_NODE_URLS`
- `BURN_DRAGON_P2P_PAGES_SELECTED_EXPERIMENT_ID`
- `BURN_DRAGON_P2P_PAGES_SELECTED_REVISION_ID`

None of those values are secrets. If they are omitted, the Pages workflow defaults to `https://dragon.aberration.technology`, `nca-prepretraining`, and `nca-r1`, and derives `/dns4/<edge-host>/tcp/4001` plus `/dns4/<edge-host>/udp/4001/quic-v1` automatically. Operators can still override everything with workflow inputs, `?edge=` / `?seed=` query params, or the UI at runtime.

## Required GitHub Environment Configuration

Create one or more GitHub Environments. Recommended names:

- `burn-dragon-p2p-staging`
- `burn-dragon-p2p-production`

Configure the workflow to target one of those environments. Put the following values on the selected environment.

### Required Environment Variables

- `BURN_DRAGON_P2P_AWS_ROLE_ARN`
  - AWS IAM role assumed through GitHub OIDC. This is the only required non-secret variable on the normal path.
- `BURN_DRAGON_P2P_AWS_REGION`
  - Optional AWS region for the stack. Defaults to `us-east-2`, which is the sane Midwest default.
- `BURN_DRAGON_P2P_STACK_NAME`
  - Optional Terraform stack prefix. Defaults to `burn-dragon-p2p-<environment>`.
- `BURN_DRAGON_P2P_EDGE_DOMAIN_NAME`
  - Optional public browser edge hostname override. Defaults to `dragon.aberration.technology`.
- `BURN_DRAGON_P2P_ROUTE53_ZONE_NAME`
  - Optional Route53 public zone override. Defaults to `aberration.technology`.
- `BURN_DRAGON_P2P_NETWORK_ID`
  - Optional burn_p2p network id. Defaults to `burn-dragon-mainnet`.
- `BURN_DRAGON_P2P_PROJECT_FAMILY_ID`
  - Optional burn_p2p project family id. Defaults to `burn-dragon-language`.
- `BURN_DRAGON_P2P_STUDY_ID`
  - Optional study id advertised in the experiment directory. Defaults to `burn-dragon-mainnet`.
- `BURN_DRAGON_P2P_RELEASE_TRAIN_HASH`
  - Optional release-train hash enforced by the auth portal. Defaults to `burn-dragon-mainnet-train`.

Recommended Midwest baseline:

- `BURN_DRAGON_P2P_AWS_REGION=us-east-2`
- `BURN_DRAGON_P2P_EDGE_DOMAIN_NAME=dragon.aberration.technology`
- `BURN_DRAGON_P2P_ROUTE53_ZONE_NAME=aberration.technology`
- leave `BURN_DRAGON_P2P_MANAGED_TRAINER_DESIRED_CAPACITY=0` until you explicitly want an always-on trainer
- if you do enable a trainer, start with `BURN_DRAGON_P2P_MANAGED_TRAINER_BACKEND=cpu`
- `BURN_DRAGON_P2P_BOOTSTRAP_INSTALL_SOURCE`
  - optional bootstrap installation source. Supported values: `crate` and `git`. Defaults to `crate`.
- `BURN_DRAGON_P2P_BOOTSTRAP_VERSION`
  - optional published `burn_p2p_bootstrap` crate version used when `BURN_DRAGON_P2P_BOOTSTRAP_INSTALL_SOURCE=crate`. Defaults to `0.21.0-pre.13`.
- `BURN_DRAGON_P2P_BOOTSTRAP_GIT_REF`
  - optional pinned `burn_p2p` git ref used only when `BURN_DRAGON_P2P_BOOTSTRAP_INSTALL_SOURCE=git`.

### Optional Environment Variables

- `BURN_DRAGON_P2P_AUTH_CONNECTOR_KIND`
  - auth connector kind for the bootstrap edge. Supported values:
    - `github`
    - `oidc`
    - `oauth`
    - `static`
    - `external`
  - defaults to `github`
- `BURN_DRAGON_P2P_AUTH_AUTHORITY_NAME`
  - logical authority label for the auth portal. Defaults to `burn-dragon-auth`.
- `BURN_DRAGON_P2P_AUTH_PRINCIPALS_JSON`
  - optional JSON array of seeded auth principals. This is the generic way to inject admin/operator principals for non-GitHub deployments, and it also works alongside GitHub auth if you want extra static principals.
- `BURN_DRAGON_P2P_AUTH_AUTHORIZE_BASE_URL`
- `BURN_DRAGON_P2P_AUTH_EXCHANGE_URL`
- `BURN_DRAGON_P2P_AUTH_TOKEN_URL`
- `BURN_DRAGON_P2P_AUTH_API_BASE_URL`
- `BURN_DRAGON_P2P_AUTH_USERINFO_URL`
- `BURN_DRAGON_P2P_AUTH_REFRESH_URL`
- `BURN_DRAGON_P2P_AUTH_REVOKE_URL`
- `BURN_DRAGON_P2P_AUTH_JWKS_URL`
  - optional connector endpoint overrides
- `BURN_DRAGON_P2P_AUTH_OIDC_ISSUER`
  - required when `BURN_DRAGON_P2P_AUTH_CONNECTOR_KIND=oidc`
- `BURN_DRAGON_P2P_AUTH_OAUTH_PROVIDER`
  - required when `BURN_DRAGON_P2P_AUTH_CONNECTOR_KIND=oauth`
- `BURN_DRAGON_P2P_AUTH_EXTERNAL_AUTHORITY`
  - required when `BURN_DRAGON_P2P_AUTH_CONNECTOR_KIND=external`
- `BURN_DRAGON_P2P_AUTH_EXTERNAL_TRUSTED_PRINCIPAL_HEADER`
  - trusted upstream header carrying the authenticated principal for `external` auth. Defaults to `x-forwarded-user`.
- `BURN_DRAGON_P2P_AUTH_EXTERNAL_TRUSTED_INTERNAL_ONLY`
  - whether the external connector should trust only internal ingress traffic. Defaults to `true`.
- `BURN_DRAGON_P2P_GITHUB_REQUIRED_ORG`
  - optional GitHub org required for peer admission when `auth_connector_kind=github`. Leave empty on the normal repo-gated path.
- `BURN_DRAGON_P2P_GITHUB_REQUIRED_REPO`
  - repo permission gate used for peer admission when `auth_connector_kind=github`, for example `mosure/burn_dragon`.
- `BURN_DRAGON_P2P_GITHUB_REQUIRED_TEAM`
  - optional `org/team` rule for GitHub peer admission
- `BURN_DRAGON_P2P_GITHUB_ADMIN_LOGINS`
  - optional comma-separated fallback GitHub username handles for the deploy workflow. These are plain GitHub login handles like `mosure`, not display names or email addresses. The workflow lowercases and deduplicates them. If the manual workflow input is empty, the workflow uses this value; if both are empty, it defaults to the triggering GitHub account. This setting only applies when `auth_connector_kind=github`.
- `BURN_DRAGON_P2P_GITHUB_ADMIN_REQUIRED_REPO_PERMISSION`
  - optional minimum GitHub repository permission for explicitly listed admin logins. Defaults to `admin`. This only applies when `auth_connector_kind=github`.
- `BURN_DRAGON_P2P_INSTANCE_TYPE`
  - override bootstrap host size, default `t3a.small`
- `BURN_DRAGON_P2P_ROOT_VOLUME_SIZE_GIB`
  - override encrypted EBS root size, default `32`. EC2 still requires a root EBS volume even on the cheapest path.
- `BURN_DRAGON_P2P_DATA_VOLUME_SIZE_GIB`
  - retained encrypted bootstrap/auth/publication data volume size when retained bootstrap storage is enabled, default `64`
- `BURN_DRAGON_P2P_USE_RETAINED_BOOTSTRAP_DATA_VOLUME`
  - whether Terraform should provision a separate retained bootstrap data volume. Defaults to `false`, which keeps bootstrap-local state on the root volume only.
- `BURN_DRAGON_P2P_MANAGED_TRAINER_DESIRED_CAPACITY`
  - desired instance count for the optional managed native trainer pool. Defaults to `0`, which disables the pool.
- `BURN_DRAGON_P2P_MANAGED_TRAINER_BACKEND`
  - backend used by the managed trainer pool. Supported values: `cpu`, `wgpu`, `cuda`. Defaults to `cpu` so GPU trainers stay opt-in.
- `BURN_DRAGON_P2P_MANAGED_TRAINER_EXPERIMENT_KIND`
  - experiment family assigned to the managed trainer pool. Supported values: `nca`, `climbmix`. Defaults to `nca`.
- `BURN_DRAGON_P2P_MANAGED_TRAINER_TARGET`
  - target role used by managed trainer instances. Defaults to `trainer`.
- `BURN_DRAGON_P2P_MANAGED_TRAINER_INSTANCE_TYPE`
  - EC2 instance type used by the managed trainer pool. Defaults to `m7i.large` on the CPU path.
- `BURN_DRAGON_P2P_MANAGED_TRAINER_MIN_SIZE`
  - optional autoscaling-group minimum size. Leave empty or `0` to default to the desired capacity.
- `BURN_DRAGON_P2P_MANAGED_TRAINER_MAX_SIZE`
  - optional autoscaling-group maximum size. Leave empty or `0` to default to the desired capacity.
- `BURN_DRAGON_P2P_MANAGED_TRAINER_CRATE_VERSION`
  - optional published `burn_dragon_p2p` crate version installed on managed trainer instances. Defaults to `0.21.0-pre.13`.
- `BURN_DRAGON_P2P_MANAGED_TRAINER_AUTH_BUNDLE_PARAMETER_NAME`
  - optional SSM parameter name containing the JSON auth bundle used by managed trainer instances. Leave empty to derive `/<stack>/<workspace>/bootstrap/trainer_auth_bundle_json`.
- `BURN_DRAGON_P2P_ENABLE_DATA_VOLUME_SNAPSHOTS`
  - enable or disable the Terraform-managed daily data-volume snapshot policy. Defaults to `false` and only matters when retained bootstrap storage is enabled.
- `BURN_DRAGON_P2P_DATA_VOLUME_SNAPSHOT_RETENTION_DAYS`
  - retained daily snapshot count for the bootstrap data volume. Defaults to `14`.
- `BURN_DRAGON_P2P_DISASTER_RECOVERY_REGION`
  - optional warm-disaster-recovery region, for example `us-west-2`. When set, Terraform enables cross-region artifact replication and, if retained bootstrap storage is enabled, copied retained-volume snapshots into that region.
- `BURN_DRAGON_P2P_ENABLE_DISASTER_RECOVERY_SNAPSHOT_COPIES`
  - enable or disable copied retained-volume snapshots into the warm-DR region. Defaults to `false` and only matters when retained bootstrap storage is enabled.
- `BURN_DRAGON_P2P_DISASTER_RECOVERY_SNAPSHOT_RETENTION_DAYS`
  - retained daily copied-snapshot count in the warm-DR region. Defaults to `14`.
- `BURN_DRAGON_P2P_ENABLE_BOOTSTRAP_STATUS_ALARMS`
  - enable or disable EC2 status-check CloudWatch alarms for the bootstrap host. Defaults to `true`.
- `BURN_DRAGON_P2P_ALARM_SNS_TOPIC_ARN`
  - optional SNS topic ARN used for CloudWatch operational alarms. Leave empty to create alarms without notifications.
- `BURN_DRAGON_P2P_ENABLE_MANAGED_CONTROL_PLANE_REDIS`
  - whether Terraform should provision a managed Redis node for auth session and operator state. Defaults to `false`.
- `BURN_DRAGON_P2P_ENABLE_CONTROL_PLANE_OPERATIONAL_ALARMS`
  - enable or disable control-plane alarms. With the cheap defaults this covers bootstrap EC2, dataset CDN, Route53 health-check, and managed-trainer alarms; Redis alarms appear only when managed Redis is enabled. Defaults to `true`.
- `BURN_DRAGON_P2P_ENABLE_CONTROL_PLANE_DASHBOARD`
  - enable or disable the shared CloudWatch dashboard for the Dragon control plane. Defaults to `true`.
- `BURN_DRAGON_P2P_ARTIFACT_BUCKET_NAME`
  - optional existing S3 bucket name for directly published checkpoints and metrics. Leave empty to let Terraform derive a stable unique bucket name.
- `BURN_DRAGON_P2P_ARTIFACT_BUCKET_PATH_PREFIX`
  - optional key prefix inside the artifact bucket. Defaults to `artifacts/<stack>/<workspace>`.
- `BURN_DRAGON_P2P_ARTIFACT_BUCKET_FORCE_DESTROY`
  - whether Terraform may delete a managed artifact bucket even when it still contains published data. Defaults to `false` and should stay that way for production.
- `BURN_DRAGON_P2P_ARTIFACT_BUCKET_SERVER_SIDE_ENCRYPTION`
  - server-side encryption mode for the managed artifact bucket and direct bootstrap uploads. Defaults to `AES256`. Warm-DR replication currently expects `AES256`.
- `BURN_DRAGON_P2P_CREATE_ARTIFACT_REPLICA_BUCKET`
  - whether Terraform should create the warm-DR replica artifact bucket when `BURN_DRAGON_P2P_DISASTER_RECOVERY_REGION` is set. Defaults to `true`.
- `BURN_DRAGON_P2P_ARTIFACT_REPLICA_BUCKET_NAME`
  - optional existing bucket name in the warm-DR region for replicated artifacts. Leave empty to auto-derive a stable name.
- `BURN_DRAGON_P2P_ARTIFACT_REPLICA_BUCKET_FORCE_DESTROY`
  - whether Terraform may delete a managed warm-DR replica bucket even when it still contains replicated artifacts. Defaults to `false` and should stay that way for production.
- `BURN_DRAGON_P2P_DATASET_DOMAIN_NAME`
  - optional public hostname for the managed browser dataset CDN. Defaults to `datasets.dragon.aberration.technology`.
- `BURN_DRAGON_P2P_DATASET_BUCKET_NAME`
  - optional S3 bucket name for the managed browser dataset origin. Leave empty to let Terraform derive a stable bucket name.
- `BURN_DRAGON_P2P_DATASET_BUCKET_PATH_PREFIX`
  - optional key prefix inside the managed browser dataset bucket. Defaults to `dragon-datasets`.
- `BURN_DRAGON_P2P_CLIMBMIX_BROWSER_DATASET_BASE_URL`
  - optional explicit base URL for the full browser ClimbMix shard pool. Defaults to the managed dataset CDN path under `https://datasets.dragon.aberration.technology/dragon-datasets/climbmix-pretraining/climbmix-r1`. Override it when the shard pool should live on a different CDN origin.

### Required Environment Secrets

- `BURN_DRAGON_P2P_AUTH_CLIENT_ID`
- `BURN_DRAGON_P2P_AUTH_CLIENT_SECRET`
  - generic OAuth/OIDC client credentials used when the selected auth connector needs them
- `BURN_DRAGON_P2P_GITHUB_CLIENT_ID`
- `BURN_DRAGON_P2P_GITHUB_CLIENT_SECRET`
  - legacy GitHub-specific secret names still accepted as a fallback when `auth_connector_kind=github`
- `BURN_DRAGON_P2P_TRAINER_AUTH_BUNDLE_JSON`
  - optional JSON auth bundle override written into SSM for the managed native trainer pool. Leave it unset on the normal path. When omitted and `BURN_DRAGON_P2P_MANAGED_TRAINER_DESIRED_CAPACITY > 0`, the deploy workflow seeds a managed static principal, waits for edge health, and mints the trainer auth bundle automatically.

The workflow writes auth client credentials into AWS SSM Parameter Store before `terraform apply` only when the selected auth connector needs client credentials, so they do not need to be committed into Terraform files or `.tfvars`. When the managed trainer pool is enabled, the workflow writes the trainer auth bundle into SSM after the edge is healthy: it uses the explicit override secret when provided, otherwise it auto-enrolls the managed trainer static principal and stores the generated bundle for instance boot.

There is intentionally no shared bootstrap admin token in the production flow. Admin actions are authenticated with a short-lived session id. For GitHub auth, admin capability is granted only to explicitly listed GitHub username handles that also satisfy the org/team/repo policy. For non-GitHub auth, seed explicit admin principals through `BURN_DRAGON_P2P_AUTH_PRINCIPALS_JSON`.

## Required AWS IAM Permissions

The GitHub OIDC role must be able to:

- manage the Terraform target resources in the selected AWS account
- write and overwrite SSM parameters under the chosen secret prefix
- read Route53 hosted zone metadata
- manage the bootstrap EC2 host, optional retained EBS volume, optional DLM snapshot policies, and CloudWatch alarms for the bootstrap stack
- create or update the artifact S3 bucket resources when you use the managed-bucket path, including optional warm-DR replica bucket resources

The deployed EC2 instance role is created by Terraform and needs to:

- read the SSM parameters that hold the auth client credentials when the selected auth connector uses them
- list, upload, and delete objects in the configured artifact S3 bucket so the bootstrap host can publish and prune checkpoints and metrics without static AWS keys
- when managed trainers are enabled, the managed trainer EC2 role additionally needs SSM read access for the trainer auth bundle parameter and KMS decrypt rights for the SSM key

## Managed Native Trainer Pool

The deploy can optionally provision an autoscaled native trainer pool alongside the bootstrap plane.

Current behavior:

- each managed trainer instance installs `burn_dragon_p2p_native` from crates.io
- CPU is now the default managed trainer backend, and the trainer pool still defaults to `0` instances, so no trainer and no GPU resource is created unless you opt in
- GPU trainer backends still work, but they are explicit opt-in and require a GPU-capable instance type such as `g5.xlarge`
- the instance fetches a JSON auth bundle from SSM at startup
- the instance joins the deployed edge as a native trainer for either `nca` or `climbmix`

Recommended first production setting:

- leave `BURN_DRAGON_P2P_MANAGED_TRAINER_DESIRED_CAPACITY=0` while you bring up the control plane
- if you want a low-cost always-on trainer later, start with `BURN_DRAGON_P2P_MANAGED_TRAINER_DESIRED_CAPACITY=1`
- set `BURN_DRAGON_P2P_MANAGED_TRAINER_BACKEND=cpu`
- set `BURN_DRAGON_P2P_MANAGED_TRAINER_EXPERIMENT_KIND=nca`

Operational constraint:

- the trainer auth bundle must be suitable for unattended use
- if your auth provider issues short-lived human sessions only, do not point the managed trainer pool at that bundle
- use a long-lived service principal, static principal, or equivalent operator-managed credential path instead

## Dynamic Admin Flow

The deployed network supports live experiment-directory edits without redeploying the bootstrap host.

### Security model

- admin access is session-based, not token-based
- for `github` auth:
  - the deploy workflow accepts explicit GitHub admin logins
  - each login is an exact GitHub username handle, for example `mosure`
  - matching admin sessions must also satisfy the configured org/team/repo policy
  - matching admin sessions receive:
    - `operator_role = admin`
    - `admin_capabilities = all`
    - `ExperimentScope::Admin { study_id = ... }`
- for non-GitHub auth:
  - seed admin/operator principals through `BURN_DRAGON_P2P_AUTH_PRINCIPALS_JSON`
  - include the needed roles/scopes directly in those principal records
- the bootstrap itself still exposes only a bounded admin action set:
  - control and lifecycle
  - auth-policy rollout
  - diagnostics, receipts, reducer-load, and trust-bundle exports
  - operator-retention prune

### Example seeded principals for non-GitHub auth

Use `BURN_DRAGON_P2P_AUTH_PRINCIPALS_JSON` to inject principals like:

```json
[
  {
    "principal_id": "burn-dragon-admin",
    "display_name": "burn_dragon admin",
    "org_memberships": [],
    "group_memberships": [],
    "granted_roles": { "roles": ["TrainerGpu", "Validator", "Archive"] },
    "granted_scopes": [
      "Connect",
      "Discover",
      { "Train": { "experiment_id": "nca-prepretraining" } },
      { "Validate": { "experiment_id": "nca-prepretraining" } },
      { "Archive": { "experiment_id": "nca-prepretraining" } },
      { "Train": { "experiment_id": "climbmix-pretraining" } },
      { "Validate": { "experiment_id": "climbmix-pretraining" } },
      { "Archive": { "experiment_id": "climbmix-pretraining" } },
      { "Admin": { "study_id": "burn-dragon-mainnet" } }
    ],
    "allowed_networks": ["burn-dragon-mainnet"],
    "custom_claims": {
      "operator_role": "admin",
      "admin_capabilities": "all"
    }
  }
]
```

### Initial published profiles

The initial directory entries are seeded from:

- `crates/burn_dragon_p2p/deploy/profiles/nca-r1.profile.json`
- `crates/burn_dragon_p2p/deploy/profiles/climbmix-r1.profile.json`

`BURN_DRAGON_P2P_CLIMBMIX_BROWSER_DATASET_BASE_URL` defaults to the managed dataset CDN path `https://datasets.dragon.aberration.technology/dragon-datasets/climbmix-pretraining/climbmix-r1`. Terraform publishes `${base_url}/fetch-manifest.json` into the initial ClimbMix browser profile. Browser peers still fetch only the shards they train on. With a runtime-provided training lease they use the exact assigned microshards; otherwise they use the bounded deterministic per-peer fallback advertised by the profile. The shipped Dragon browser app now reads that persisted browser training lease automatically before local training starts.

Those profile payloads are derived from the source configs in the same folder. To regenerate a profile locally:

```bash
cargo run -p burn_dragon_p2p --features native --bin burn_dragon_p2p_native -- \
  build-profile \
  --training-config crates/burn_dragon_p2p/deploy/profiles/nca-r1.training.toml \
  --experiment-kind nca \
  --output crates/burn_dragon_p2p/deploy/profiles/nca-r1.profile.json
```

For ClimbMix, pass the revision id so the browser shard-manifest URL is included in the profile:

```bash
cargo run -p burn_dragon_p2p --features native --bin burn_dragon_p2p_native -- \
  build-profile \
  --training-config crates/burn_dragon_p2p/deploy/profiles/climbmix-r1.training.toml \
  --experiment-kind climbmix \
  --revision-id climbmix-r1 \
  --output crates/burn_dragon_p2p/deploy/profiles/climbmix-r1.profile.json
```

To point the published ClimbMix browser profile at a full shard pool:

```bash
cargo run -p burn_dragon_p2p --features native --bin burn_dragon_p2p_native -- \
  build-profile \
  --training-config crates/burn_dragon_p2p/deploy/profiles/climbmix-r1.training.toml \
  --experiment-kind climbmix \
  --revision-id climbmix-r1 \
  --browser-climbmix-manifest-url https://dragon.aberration.technology/dragon-datasets/climbmix-pretraining/climbmix-r1/fetch-manifest.json \
  --output crates/burn_dragon_p2p/deploy/profiles/climbmix-r1.profile.json
```

### Roll out an updated experiment profile

The recommended operator flow is now browser-first:

1. open the deployed Pages shell or another host rendering `DragonBrowserApp`
2. click `Sign In (Admin)`
3. set the study id, for example `burn-dragon-mainnet`
4. click `Load Directory`
5. click `Load Selected Entry` for the experiment you want to change
6. paste the replacement entry JSON or update the current entry draft
7. click `Upsert Editor Entry`
8. click `Roll Out Directory`

That path uses the same session-authenticated `/admin` surface as the native operator binary. The native CLI remains available for scripted or headless rollout.

Example native fallback:

```bash
cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  begin-github-login \
  --config /path/to/native-peer.toml \
  --experiment-kind nca \
  --backend wgpu \
  --edge-url https://dragon.aberration.technology

cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  complete-github-login \
  --config /path/to/native-peer.toml \
  --pending /path/to/pending-login.json \
  --provider-code '<github-code>' \
  --auth-bundle-out /path/to/admin-auth.json

cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  admin-rollout-profile \
  --config /path/to/native-peer.toml \
  --experiment-kind nca \
  --backend wgpu \
  --auth-bundle /path/to/admin-auth.json
```

`admin-rollout-profile` uses the current local Dragon config and manifest metadata as the source of truth and pushes a replacement directory entry through `RolloutAuthPolicy`. Peers that rely on network-published Dragon profiles will pick up the new config from the directory instead of requiring a matching static local training config.

## Native Peer Join After Deploy

After the workflow finishes, use the outputs from the workflow summary:

- `edge_url`
- `seed_node_tcp_multiaddr`
- `seed_node_quic_multiaddr`

Then run the native operator binary against that network, for example:

```bash
cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  begin-github-login \
  --config /path/to/native-peer.toml \
  --experiment-kind nca \
  --backend wgpu \
  --edge-url https://dragon.aberration.technology \
  --seed-node-url /dns4/dragon.aberration.technology/tcp/4001 \
  --seed-node-url /dns4/dragon.aberration.technology/udp/4001/quic-v1
```

Native peers can leave `training_config_paths` empty and rely on the published Dragon profile metadata for experiments that have a compatible network profile.

## Browser Join After Deploy

Open the deployed `edge_url` in a browser, sign in with GitHub, and join the network from the embedded browser edge UI.

Browser peers can train directly from the published Dragon profile metadata for experiments that include a browser-capable profile.

## Terraform Root

The AWS Terraform root lives at:

- `crates/burn_dragon_p2p/deploy/terraform/aws`

Use `terraform.tfvars.example` as the starting point for any local/manual run. The GitHub Action does not require a checked-in `.tfvars` file as long as the environment variables and secrets above are configured.
