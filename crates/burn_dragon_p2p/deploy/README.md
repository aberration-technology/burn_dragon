# burn_dragon_p2p Deploy

This folder contains the operator-facing deployment assets for the `burn_dragon_p2p` network:

- `native-peer.toml.example`: example native peer config
- `burn-dragon-p2p-native.service`: example native peer systemd unit
- `profiles/`: checked-in Dragon experiment profile sources and initial published profile payloads
- `datasets/README.md`: notes on external browser shard pools that stay out of git
- `terraform/aws`: the checked-in AWS bootstrap/edge deployment

The AWS Terraform root deploys a single-region bootstrap plane for the Dragon network:

- two EC2 bootstrap/edge hosts across two AZs
- Route53 DNS for the browser edge
- TLS termination via Caddy on the host
- retained encrypted EBS data volume per bootstrap host for local peer/runtime state
- bootstrap-managed direct S3 publication for checkpoint and metric artifacts using the EC2 instance role
- shared Redis-backed auth session and operator state for multi-node control-plane continuity
- shared authority material synchronized through SSM so both bootstrap nodes serve the same auth/control plane
- daily retained data-volume snapshots with Terraform-managed DLM policy by default
- EC2 status-check CloudWatch alarms, optionally wired to SNS
- configurable browser/native auth flow through `burn-p2p-bootstrap`

It does not deploy end-user native trainer peers. Native operators still install and run `burn_dragon_p2p_native` locally, then point it at the deployed edge and seed URLs.

The bootstrap publishes initial Dragon experiment directory entries for:

- `nca-prepretraining`
- `climbmix-pretraining`

Those entries include Dragon profile metadata, so peers can resolve experiment and training configuration from the network instead of requiring a matching static local config.
The initial ClimbMix revision is intended to point at a full external browser shard pool. The
deploy workflow publishes `${base_url}/fetch-manifest.json` into the initial ClimbMix browser
profile, and browser peers fetch only the shards they train on from that external pool.

## Artifact Storage

Checkpoint artifacts, including model weights and exported metric bundles, are published directly from the bootstrap host into S3 using the EC2 instance role and the upstream `S3Compatible` publication target.

There is no separate artifact node by default. The bootstrap/control-plane hosts own artifact publication, durable artifact bytes live in S3, shared auth session plus operator state live in Redis, and each bootstrap node keeps its local peer/runtime state on its retained EBS data volume.

## One-Click GitHub Action

The intended operator entrypoint is:

- `.github/workflows/deploy-burn-dragon-p2p-aws.yml`

That workflow:

- seeds auth client credentials into AWS SSM Parameter Store when the selected auth connector needs them
- runs `terraform fmt`, `init`, `validate`, `plan`, and `apply`
- creates or reuses the S3 bucket used for durable direct artifact publication
- configures explicit GitHub admin logins for session-authenticated admin access when the auth connector is `github`
- waits for the edge URL to answer over HTTPS
- prints the primary and secondary bootstrap instance details, shared Redis endpoint, pinned bootstrap git ref, and artifact S3 prefix in the workflow summary

If you trigger the workflow with a forced bootstrap replacement, Terraform replaces the primary EC2 host. The retained primary bootstrap data volume is reattached to the replacement host, so local peer/runtime state survives a normal rebuild. Shared auth session state, operator state, and artifact publication remain externalized in Redis and S3.

The workflow still performs a Terraform plan internally before apply. That keeps the operator experience one-click without dropping the safety and auditability of a plan phase.


## GitHub Pages Browser Shell

The focused repo also ships a separate browser-shell workflow:

- `.github/workflows/deploy-burn-dragon-p2p-pages.yml`

Before the workflow can publish, set the repository Pages source to `GitHub Actions` under `Settings > Pages`.

That workflow builds the standalone `burn_dragon_p2p_browser` wasm client through `xtask build-browser-site`, uploads the generated static bundle, and deploys it to GitHub Pages. The published shell is static; it still connects to the live edge URL you configure. By default, the baked browser config points at `https://dragon.aberration.technology` and derives the standard TCP and QUIC bootstrap multiaddrs from that host.

The deployed browser shell now includes an operator panel alongside the peer UI. It requests `Connect` and `Discover` by default, plus `Train` and `Validate` for the selected experiment id when one is baked into the shell. Operators can then use `Sign In (Admin)` from the browser to request an additional `ExperimentScope::Admin { study_id }` session for live directory edits. Under the default deployment, that browser login provider is GitHub.

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
  - AWS IAM role assumed through GitHub OIDC.
- `BURN_DRAGON_P2P_AWS_REGION`
  - AWS region for the stack, for example `us-east-1`.
- `BURN_DRAGON_P2P_STACK_NAME`
  - Terraform stack prefix, for example `burn-dragon-p2p-mainnet`.
- `BURN_DRAGON_P2P_EDGE_DOMAIN_NAME`
  - Optional public browser edge hostname override. Defaults to `dragon.aberration.technology`.
- `BURN_DRAGON_P2P_ROUTE53_ZONE_NAME`
  - Optional Route53 public zone override. Defaults to `aberration.technology`.
- `BURN_DRAGON_P2P_NETWORK_ID`
  - burn_p2p network id, for example `burn-dragon-mainnet`.
- `BURN_DRAGON_P2P_PROJECT_FAMILY_ID`
  - burn_p2p project family id, usually `burn-dragon-language`.
- `BURN_DRAGON_P2P_STUDY_ID`
  - study id advertised in the experiment directory.
- `BURN_DRAGON_P2P_RELEASE_TRAIN_HASH`
  - release-train hash enforced by the auth portal.
- `BURN_DRAGON_P2P_BOOTSTRAP_GIT_REF`
  - optional pinned `burn_p2p` git ref used to install `burn_p2p_bootstrap` on the edge host. Defaults to `0c89aaf`, the first `burn_p2p` commit with ambient-IAM direct S3 publication support.

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
  - GitHub org required for peer admission when `auth_connector_kind=github`.
- `BURN_DRAGON_P2P_GITHUB_REQUIRED_REPO`
  - repo permission gate used for peer admission when `auth_connector_kind=github`, for example `mosure/burn_dragon`.
- `BURN_DRAGON_P2P_GITHUB_REQUIRED_TEAM`
  - optional `org/team` rule for GitHub peer admission
- `BURN_DRAGON_P2P_GITHUB_ADMIN_LOGINS`
  - optional comma-separated fallback GitHub username handles for the deploy workflow. These are plain GitHub login handles like `mosure`, not display names or email addresses. The workflow lowercases and deduplicates them. If the manual workflow input is empty, the workflow uses this value; if both are empty, it defaults to the triggering GitHub account. This setting only applies when `auth_connector_kind=github`.
- `BURN_DRAGON_P2P_GITHUB_ADMIN_REQUIRED_REPO_PERMISSION`
  - optional minimum GitHub repository permission for explicitly listed admin logins. Defaults to `admin`. This only applies when `auth_connector_kind=github`.
- `BURN_DRAGON_P2P_INSTANCE_TYPE`
  - override bootstrap host size, default `t3.large`
- `BURN_DRAGON_P2P_ROOT_VOLUME_SIZE_GIB`
  - override encrypted EBS root size, default `256`
- `BURN_DRAGON_P2P_DATA_VOLUME_SIZE_GIB`
  - retained encrypted bootstrap/auth/publication data volume size, default `512`
- `BURN_DRAGON_P2P_ENABLE_DATA_VOLUME_SNAPSHOTS`
  - enable or disable the Terraform-managed daily data-volume snapshot policy. Defaults to `true`.
- `BURN_DRAGON_P2P_DATA_VOLUME_SNAPSHOT_RETENTION_DAYS`
  - retained daily snapshot count for the bootstrap data volume. Defaults to `14`.
- `BURN_DRAGON_P2P_ENABLE_BOOTSTRAP_STATUS_ALARMS`
  - enable or disable EC2 status-check CloudWatch alarms for the bootstrap host. Defaults to `true`.
- `BURN_DRAGON_P2P_ALARM_SNS_TOPIC_ARN`
  - optional SNS topic ARN used for bootstrap status-check alarms. Leave empty to create alarms without notifications.
- `BURN_DRAGON_P2P_ARTIFACT_BUCKET_NAME`
  - optional existing S3 bucket name for directly published checkpoints and metrics. Leave empty to let Terraform derive a stable unique bucket name.
- `BURN_DRAGON_P2P_ARTIFACT_BUCKET_PATH_PREFIX`
  - optional key prefix inside the artifact bucket. Defaults to `artifacts/<stack>/<workspace>`.
- `BURN_DRAGON_P2P_ARTIFACT_BUCKET_FORCE_DESTROY`
  - whether Terraform may delete a managed artifact bucket even when it still contains published data. Defaults to `false` and should stay that way for production.
- `BURN_DRAGON_P2P_ARTIFACT_BUCKET_SERVER_SIDE_ENCRYPTION`
  - server-side encryption mode for the managed artifact bucket and direct bootstrap uploads. Defaults to `AES256`.
- `BURN_DRAGON_P2P_CLIMBMIX_BROWSER_DATASET_BASE_URL`
  - public base URL for the full browser ClimbMix shard pool. Defaults to `https://dragon.aberration.technology/dragon-datasets/climbmix-pretraining/climbmix-r1`. The deploy workflow publishes `${base_url}/fetch-manifest.json` into the initial ClimbMix browser profile. Override it when the shard pool lives on a different CDN origin.

### Required Environment Secrets

- `BURN_DRAGON_P2P_AUTH_CLIENT_ID`
- `BURN_DRAGON_P2P_AUTH_CLIENT_SECRET`
  - generic OAuth/OIDC client credentials used when the selected auth connector needs them
- `BURN_DRAGON_P2P_GITHUB_CLIENT_ID`
- `BURN_DRAGON_P2P_GITHUB_CLIENT_SECRET`
  - legacy GitHub-specific secret names still accepted as a fallback when `auth_connector_kind=github`

The workflow writes these secrets into AWS SSM Parameter Store before `terraform apply` only when the selected auth connector needs client credentials, so they do not need to be committed into Terraform files or `.tfvars`.

There is intentionally no shared bootstrap admin token in the production flow. Admin actions are authenticated with a short-lived session id. For GitHub auth, admin capability is granted only to explicitly listed GitHub username handles that also satisfy the org/team/repo policy. For non-GitHub auth, seed explicit admin principals through `BURN_DRAGON_P2P_AUTH_PRINCIPALS_JSON`.

## Required AWS IAM Permissions

The GitHub OIDC role must be able to:

- manage the Terraform target resources in the selected AWS account
- write and overwrite SSM parameters under the chosen secret prefix
- read Route53 hosted zone metadata
- manage the retained EBS volume, DLM snapshot policies, and CloudWatch alarms for the bootstrap stack
- create or update the artifact S3 bucket resources when you use the managed-bucket path

The deployed EC2 instance role is created by Terraform and needs to:

- read the SSM parameters that hold the auth client credentials when the selected auth connector uses them
- list, upload, and delete objects in the configured artifact S3 bucket so the bootstrap host can publish and prune checkpoints and metrics without static AWS keys

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

`BURN_DRAGON_P2P_CLIMBMIX_BROWSER_DATASET_BASE_URL` defaults to `https://dragon.aberration.technology/dragon-datasets/climbmix-pretraining/climbmix-r1`. Terraform publishes `${base_url}/fetch-manifest.json` into the initial ClimbMix browser profile. Browser peers still fetch only the shards they train on. With a runtime-provided training lease they use the exact assigned microshards; otherwise they use the bounded deterministic per-peer fallback advertised by the profile. The shipped Dragon browser app now reads that persisted browser training lease automatically before local training starts.

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
