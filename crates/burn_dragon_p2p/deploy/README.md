# burn_dragon_p2p Deploy

This folder contains the operator-facing deployment assets for the `burn_dragon_p2p` network:

- `native-peer.toml.example`: example native peer config
- `burn-dragon-p2p-native.service`: example native peer systemd unit
- `profiles/`: checked-in Dragon experiment profile sources and initial published profile payloads
- `datasets/README.md`: notes on shard-pool source material that stays out of git and can be published into the managed dataset CDN
- `terraform/aws`: the checked-in AWS bootstrap/edge deployment

The AWS Terraform root deploys a single-region bootstrap plane for the Dragon network:

- one EC2 bootstrap/edge host
- Route53 DNS for the bootstrap API/auth edge
- no ALB or API Gateway; the bootstrap API/auth edge is Caddy on the bootstrap host
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
- EC2, Redis, dataset CDN, Route53 HTTPS app health, and managed-trainer CloudWatch alarms, optionally wired to SNS
- shared CloudWatch dashboard for control-plane health and throughput
- configurable browser/native auth flow through `burn-p2p-bootstrap`

It does not attempt to manage every end-user native trainer. Native operators can still install and run `burn_dragon_p2p_native` locally, then point it at the deployed edge and seed URLs. The stack can also own a small managed native trainer pool for always-on capacity. The supported minimal production topology is now explicit: one bootstrap-only edge host plus optional trainers. The bootstrap host handles GitHub auth, seed/discovery, API/auth traffic, and S3 publication. The browser shell is intended to run from GitHub Pages, and dataset bytes are served from the managed dataset CDN. Trainer-only diffusion promotion happens across trainer peers; the standard AWS workflow no longer provisions a separate validator host.

The bootstrap publishes initial Dragon experiment directory entries for:

- `nca-prepretraining`
- `climbmix-pretraining`

Those entries include Dragon profile metadata, so peers can resolve experiment and training configuration from the network instead of requiring a matching static local config.
The initial ClimbMix revision now defaults to the managed dataset CDN path under `https://datasets.dragon.aberration.technology/dragon-datasets/climbmix-pretraining/climbmix-r1`. The deploy workflow publishes `${base_url}/fetch-manifest.json` into the initial ClimbMix browser profile, and browser peers fetch only the shards they train on from that managed shard pool unless you override the base URL explicitly.

## Ruliad Training Profiles

The ruliad profile family keeps recommended defaults and comparison baselines
separate so ablations stay reproducible:

- `profiles/ruliad-r1.jepa.training.toml`: promoted JEPA default profile.
- `profiles/ruliad-r1.training.toml`: base CE/AdamW comparison profile.
- `profiles/ruliad-r1.jepa-state.training.toml`: JEPA plus rho-state
  consistency candidate for collapse-resistance experiments.
- `profiles/ruliad-1m.jepa.training.toml` and
  `profiles/ruliad-1m-la-16k.jepa.training.toml`: larger JEPA default
  profiles.

The plain non-JEPA profiles intentionally remain available as AdamW/CE
comparison points. Treat the ruliad competence scalar as a coarse dashboard
composite; use verifier/schema/degen metrics and source-bucket telemetry when
promoting a profile.

## Minimal Topology

The cheapest supported production topology is now:

- one bootstrap-only EC2 node
- zero Redis nodes by default
- zero managed trainers by default
- zero GPU infrastructure by default

Recommended operator defaults:

- keep the managed deploy/restore default `bootstrap_install_source=git`; CI materializes the public sibling stack at `stack.lock.toml`, builds the bootstrap and head-mirror binaries from that coherent checkout, verifies their checksums, and uploads them for runtime sync
- use `bootstrap_install_source=crate` only when the selected published release intentionally matches the protocol revision in the locked stack
- leave the managed trainer pool at `0` until the control plane and browser path are stable under the intended traffic pattern
- keep restore drills on `plan_only=true` until you are intentionally executing a failover

Role split:

- bootstrap: GitHub auth, API/auth edge, seed/discovery, operator API, S3 artifact publication
- trainers: browser peers and any optional external or managed trainers

Canonical promotion in the supported deploy path uses trainer-only diffusion steady-state.

That minimal topology is a trusted-fleet deployment shape. Trainer-only
diffusion is not a sufficient authority boundary for arbitrary public peers.
Before enabling untrusted trainer membership, deploy separate validators,
require a quorum greater than one, and enable independent compact-update replay
as described in
[the production-readiness ledger](../../../docs/p2p-production-readiness.md).

For that topology, bind promotion ownership into the shared manifest seed:

```toml
[manifest.promotion]
mode = "validator_quorum"
validator_quorum = 2
```

Run trainer peers with `target = "trainer"` and at least two independent peers
with `target = "validator"`. Trainers publish candidate artifacts without
performing canonical validation. Each validator materializes the candidate,
persists its evaluation report, and publishes an attestation bound to the exact
head ID, artifact ID, evaluation protocol, and report content ID. Promotion
fails closed until the control plane can match a distinct report and reduction
certificate from every member of the required quorum. The validator daemon
does not re-evaluate the newly promoted head because that exact materialization
was already evaluated before attestation.

## Signed Revision Provisioning

Production native and browser peers must converge on the same signed revision
contract and exact full-head genesis artifact.

The managed AWS deploy and restore workflows provision the selected head-mirror
revision automatically after its canonical genesis is visible. Provisioning
runs on the bootstrap host: it loads the local full-head artifact, decodes it
through the Dragon workload adapter, computes the canonical tensor digest,
signs with the bootstrap authority key, atomically rolls the complete contract
set into the edge, and verifies that the exact bundle is present in
`/portal/snapshot`. A repeat deployment reuses an already-valid immutable
contract. The bootstrap service reloads that bundle from the retained head
mirror storage on every subsequent start. Deployment readiness fails before
Pages deployment when the selected browser revision has no valid
authority-signed contract.

The bootstrap accepts one or more JSON contract bundle files through the
comma-separated `BURN_P2P_REVISION_CONTRACT_FILES` environment variable. On
startup it:

1. loads the active authority trust bundle
2. verifies the genesis and revision-contract Ed25519 signatures
3. rejects stale content IDs, cross-revision mismatches, and conflicts
4. publishes only verified bundles in the browser-edge snapshot

Native peer config points to the same bundle with:

```toml
[manifest]
revision_contract_path = "/etc/burn_dragon_p2p/nca-r3.revision-contract.json"
require_signed_revision_contracts = true
```

Provision the referenced genesis artifact before starting peers. Verify from a
clean native storage root and a clean browser profile that both peers load the
same contract ID, genesis payload ID, tensor digest, artifact ID, and head ID.
Missing or invalid material must prevent training rather than falling back to
peer-local random initialization.

Build and verify a bundle from explicit authority material:

```bash
cargo run -p xtask -- build-revision-contract \
  --spec /secure/nca-r3.contract-spec.json \
  --artifact-store-root /var/lib/burn-p2p/artifacts \
  --authority-key /secure/authority.key \
  --output /secure/nca-r3.revision-contract.json

cargo run -p xtask -- verify-revision-contract \
  --bundle /secure/nca-r3.revision-contract.json \
  --trust-bundle /secure/trust-bundle.json \
  --artifact-store-root /var/lib/burn-p2p/artifacts
```

The build command verifies the complete artifact descriptor and every
content-addressed byte before signing. The authority supplies the canonical
tensor digest in the spec; every native and browser training peer independently
recomputes it after decoding and fails closed on mismatch.

Rotate a set of bundles into a new atomically installed directory, then roll
them out through the authenticated live admin API:

```bash
cargo run -p xtask -- rotate-revision-contracts \
  --bundle /secure/nca-r3.revision-contract.json \
  --new-authority-key /secure/next-authority.key \
  --output-dir /secure/contracts-next

cargo run -p xtask -- rollout-revision-contracts \
  --edge-url https://edge.dragon.aberration.technology \
  --bundle /secure/contracts-next/nca-r3.revision-contract.json \
  --session-id "$BURN_P2P_ADMIN_SESSION_ID" \
  --allow-signature-rotation
```

The rollout validates the complete replacement set before one atomic
control-plane update; readers never observe a partially rotated revision set.
Terraform does not manufacture authority keys. The managed deployment signs
only after canonical genesis materialization on the authority host; manual
build/rotation remains available for offline ceremonies. Secret custody,
staging rotation, and disaster-recovery rehearsal remain operator release
gates.

## Artifact Storage

Checkpoint artifacts, including model weights and exported metric bundles, are published from the bootstrap host into S3 using the EC2 instance role and the upstream `S3Compatible` publication target. The default Dragon deploy now uses a hybrid publication policy: canonical serve-checkpoint aliases are mirrored eagerly, while heavier exports remain on-demand. When `disaster_recovery_region` is configured, Terraform also enables cross-region S3 replication into a warm-DR replica bucket.

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

a successful `push` to `main` now auto-dispatches the production AWS deploy workflow from `CI`. that production deploy workflow remains the single orchestrator and still dispatches `deploy-pages.yml` only after the AWS rollout succeeds, so the browser shell stays ordered behind the live edge rollout instead of racing it. CI gates that dispatch on both the browser site artifact build and `xtask local-browser-e2e`; together they cover the local production-shaped browser/native contract without rebuilding the static site twice. Use `xtask local-prod-e2e` for the local one-shot version of the same contract when you want the site build and browser/native checks in one command.

the bootstrap runtime sync now updates the bootstrap systemd unit itself, not just `bootstrap.json` and caddy config. that keeps the live edge aligned with the repo-managed fd limit and service settings without depending on instance replacement.

That workflow:

- seeds auth client credentials into AWS SSM Parameter Store when the selected auth connector needs them
- installs `burn_p2p_bootstrap` from the published crate by default, with an explicit git fallback only for testing unpublished upstream revisions
- runs `terraform fmt`, `init`, `validate`, `plan`, and `apply`
- creates or reuses the S3 bucket used for durable direct artifact publication
- optionally creates an autoscaled managed trainer pool that installs `burn_dragon_p2p_native` from crates.io and fetches its auth bundle from SSM
- runs managed trainer instances with the protocol-aware
  `run-trainer-daemon` command; `run-peer` is reserved for non-training network
  service
- auto-seeds a deploy-managed static trainer principal and mints its auth bundle after edge health when the trainer pool is enabled and no explicit bundle override secret is supplied
- configures explicit GitHub admin logins for session-authenticated admin access when the auth connector is `github`
- waits for the edge URL to answer over HTTPS
- prints the bootstrap instance details, bootstrap state-storage mode, control-plane state backend, control-plane dashboard URL, bootstrap install source/version, managed trainer pool outputs, and artifact plus dataset S3 prefixes in the workflow summary
- derives the managed stack name as `burn-dragon-p2p-<environment>` and rejects legacy stack-name overrides or duplicate bootstrap instances for the same deployment environment

Managed production deploys install `burn_p2p_bootstrap` from the immutable public Git revision in `stack.lock.toml`. The crates.io path remains available for explicit released-stack deployments, but operators must keep its protocol revision aligned with the Dragon peer lock.

The trainer daemon follows the `TrainingProtocol` in the active signed
directory revision. The currently browser-compatible public profile uses
ArtifactWindows; the convergence-qualified native candidate uses DiLoCo.
Browsers fail closed to observer/verifier on a DiLoCo revision until browser
DiLoCo is implemented. Do not advertise one mixed native/browser trainer
cohort across those two protocol revisions.

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
- derives the managed stack name as `burn-dragon-p2p-<environment>` and rejects legacy stack-name overrides or duplicate bootstrap instances for the same deployment environment

The bootstrap inspection workflow is:

- `.github/workflows/inspect-burn-dragon-p2p-aws.yml`

It now lists every bootstrap instance tagged for the selected deployment environment and can terminate non-canonical legacy bootstrap instances when `cleanup_legacy_bootstrap_instances=true`.

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

- `.github/workflows/deploy-pages.yml`
- `.github/workflows/publish-burn-dragon-p2p-dataset.yml`

Before the workflow can publish, set the repository Pages source to `GitHub Actions` under `Settings > Pages`. Under the split-host production layout, use `dragon.aberration.technology` for the published Pages shell, `edge.dragon.aberration.technology` for the bootstrap API/auth edge, and `datasets.dragon.aberration.technology` for the managed shard CDN.

That workflow builds the standalone `burn_dragon_p2p_browser` wasm client through `xtask build-browser-site`, uploads the generated static bundle, and deploys it to GitHub Pages. The published shell is static and is intended to live at `https://dragon.aberration.technology`; it still depends on the live bootstrap API/auth edge you configure for browser control-plane bootstrap, auth, signed browser seeds, and steady-state sync. By default, the baked browser config points at `https://edge.dragon.aberration.technology`, resolves browser-capable signed seeds from that edge, and refuses to publish a degraded WSS-only shell when the edge advertises direct browser transports.

The deploy path is now split more cleanly:

- `xtask resolve-pages-deploy-settings` and `xtask resolve-pages-deploy-settings-outputs` resolve Pages defaults, signed browser seed derivation, and the "refuse degraded WSS-only publish when direct browser transports are advertised" guardrail
- `xtask build-browser-site` is the authoritative browser bundle generator
- `cargo run -p xtask -- dispatch-pages-deploy-and-wait` owns the child-workflow dispatch/watch logic used by deploy and restore
- `cargo run -p xtask -- run-live-browser-canary`, `run-pages-predeploy-canary`, and `summarize-live-browser-canary` own the shared canary execution and summary path used by Pages, deploy, restore, and the standalone live-canary workflow

`deploy-pages.yml` runs a predeploy browser training canary against the freshly built Pages artifact before upload, then runs the live browser canary after the Pages publish completes. The workflow does not succeed unless the exact static bundle about to be published can boot, connect, train through the browser path, and reach the expected browser peer contract against the configured edge.

The deployed browser shell now includes an operator panel alongside the peer UI. It requests `Connect` and `Discover` by default, plus `Train` and `Archive` for the selected experiment id when one is baked into the shell. Operators can then use `Sign In (Admin)` from the browser to request an additional `ExperimentScope::Admin { study_id }` session for live directory edits. Under the default deployment, that browser login provider is GitHub.

Terraform now protects the Route53 hosted-zone apex by default. With the production defaults, the stack may manage `edge.dragon.aberration.technology`, `dragon.aberration.technology`, and `datasets.dragon.aberration.technology`, but it will refuse to claim `aberration.technology` itself unless `allow_route53_zone_apex_records=true` is set explicitly in Terraform.

Optional GitHub repository variables for the Pages workflow:

- `BURN_DRAGON_P2P_PAGES_EDGE_BASE_URL`
- `BURN_DRAGON_P2P_PAGES_SEED_NODE_URLS`
- `BURN_DRAGON_P2P_PAGES_SELECTED_EXPERIMENT_ID`
- `BURN_DRAGON_P2P_PAGES_SELECTED_REVISION_ID`

None of those values are secrets. If they are omitted, the Pages workflow defaults to `https://edge.dragon.aberration.technology`, `nca-prepretraining`, and `nca-r3`, then derives browser-capable bootstrap material from the live edge before publishing. Operators can still override everything with workflow inputs, `?edge=` / `?seed=` query params, or the UI at runtime.

## Required GitHub Environment Configuration

Create one or more GitHub Environments. Recommended names:

- `burn-dragon-p2p-staging`
- `burn-dragon-p2p-production`

Configure the workflow to target one of those environments. Put the following values on the selected environment.

### Required Environment Variables

- `BURN_DRAGON_P2P_AWS_ROLE_ARN`
  - AWS IAM role assumed through GitHub OIDC for deploy, restore, inspect, and dataset publication workflows.
- `BURN_DRAGON_P2P_AWS_CLEANUP_ROLE_ARN`
  - separate AWS IAM role assumed through GitHub OIDC for the destructive cleanup workflow only. Keep this role distinct from `BURN_DRAGON_P2P_AWS_ROLE_ARN`.
- `BURN_DRAGON_P2P_AWS_REGION`
  - Optional AWS region for the stack. Defaults to `us-east-2`, which is the sane Midwest default.
- `BURN_DRAGON_P2P_STACK_NAME`
  - Optional Terraform stack prefix for manual or local Terraform usage. Managed GitHub deploy and restore workflows require the canonical name `burn-dragon-p2p-<environment>` and will fail if this variable is set to a different legacy alias.
- `BURN_DRAGON_P2P_EDGE_DOMAIN_NAME`
  - Optional public bootstrap API/auth hostname override. Defaults to `edge.dragon.aberration.technology`.
- `BURN_DRAGON_P2P_BROWSER_APP_BASE_URL`
  - Optional public browser-app URL. Defaults to `https://dragon.aberration.technology` and is used for the separately hosted GitHub Pages shell.
- `BURN_DRAGON_P2P_AUTH_REDIRECT_BASE_URL`
  - Optional OAuth callback base URL. Defaults to `BURN_DRAGON_P2P_BROWSER_APP_BASE_URL`, so GitHub/OIDC/OAuth flows return to the Pages host instead of the bootstrap edge.
- `BURN_DRAGON_P2P_BROWSER_APP_PAGES_DOMAIN_TARGET`
  - Optional GitHub Pages DNS target for the browser app hostname. Defaults to `<repo-owner>.github.io` in the deploy workflow and lets Terraform create the Route53 CNAME for `dragon.aberration.technology`.
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

### Production Guardrails

The production workflow path is intentionally narrower than the full Terraform surface:

- use `terraform_workspace=mainnet`
- keep the managed workflow default `bootstrap_install_source=git` so bootstrap and peer binaries are runner-built from the same locked P2P revision; the remote host does not perform a standalone Git install because `burn_p2p` intentionally uses the sibling `burn_ecs` path dependency
- keep bootstrap status alarms enabled
- keep control-plane operational alarms enabled
- keep the shared control-plane dashboard enabled
- set `BURN_DRAGON_P2P_ALARM_SNS_TOPIC_ARN` so alarms route to a real pager or operator channel
- keep the Route53 edge health check on `https://${BURN_DRAGON_P2P_EDGE_DOMAIN_NAME}/portal/snapshot`, not a raw TCP 443 probe
- keep the post-deploy Pages browser canary green before treating a browser publish as complete
- keep the fixed recurring AWS profile under `$100`
- keep `BURN_DRAGON_P2P_MANAGED_TRAINER_DESIRED_CAPACITY=0` on the normal production path; scale first through browser peers and external native peers instead of an always-on managed trainer fleet

Recommended Midwest baseline:

- `BURN_DRAGON_P2P_AWS_REGION=us-east-2`
- `BURN_DRAGON_P2P_ALARM_SNS_TOPIC_ARN=arn:aws:sns:us-east-2:<account-id>:burn-dragon-p2p-alerts`
- `BURN_DRAGON_P2P_EDGE_DOMAIN_NAME=edge.dragon.aberration.technology`
- `BURN_DRAGON_P2P_BROWSER_APP_BASE_URL=https://dragon.aberration.technology`
- `BURN_DRAGON_P2P_AUTH_REDIRECT_BASE_URL=https://dragon.aberration.technology`
- `BURN_DRAGON_P2P_BROWSER_APP_PAGES_DOMAIN_TARGET=aberration-technology.github.io`
- `BURN_DRAGON_P2P_ROUTE53_ZONE_NAME=aberration.technology`
- leave `BURN_DRAGON_P2P_MANAGED_TRAINER_DESIRED_CAPACITY=0` on the default production path to stay under `$100` fixed monthly AWS cost
- if you intentionally exceed that budget later, start with `BURN_DRAGON_P2P_MANAGED_TRAINER_BACKEND=cpu` and re-evaluate the fixed-cost estimate before apply
- `BURN_DRAGON_P2P_BOOTSTRAP_INSTALL_SOURCE`
  - optional bootstrap installation source. Supported values: `git` and `crate`. Managed workflows default to `git` and resolve the public immutable revision from `stack.lock.toml`; direct Terraform remains configurable.
- `BURN_DRAGON_P2P_BOOTSTRAP_VERSION`
  - optional published `burn_p2p_bootstrap` crate version used when `BURN_DRAGON_P2P_BOOTSTRAP_INSTALL_SOURCE=crate`. Defaults to `0.21.0`.
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
  - desired instance count for the optional managed native trainer pool. Defaults to `0`, which disables the pool. Any nonzero production pool on the default `m7i.large` path exceeds the fixed-cost guardrail and is rejected.
- `BURN_DRAGON_P2P_MANAGED_TRAINER_BACKEND`
  - backend used by the managed trainer pool. Supported values: `cpu`, `wgpu`, `cuda`, `rocm`. Defaults to `cpu` so GPU trainers stay opt-in.
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
  - optional published `burn_dragon_p2p` crate version installed on managed trainer instances. Defaults to the current repo workspace version from `Cargo.toml` when using the deployment workflows, currently `0.21.0`.
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
  - SNS topic ARN used for CloudWatch operational alarms. The production workflow guardrails require this to be non-empty so alarms route somewhere actionable.
- `BURN_DRAGON_P2P_ENABLE_MANAGED_CONTROL_PLANE_REDIS`
  - whether Terraform should provision a managed Redis node for auth session and operator state. Defaults to `false`.
- `BURN_DRAGON_P2P_ENABLE_CONTROL_PLANE_OPERATIONAL_ALARMS`
  - enable or disable control-plane alarms. With the cheap defaults this covers bootstrap EC2, dataset CDN, Route53 health-check, and managed-trainer alarms; Redis alarms appear only when managed Redis is enabled. Defaults to `true`, and the production path keeps it enabled.
- `BURN_DRAGON_P2P_ENABLE_CONTROL_PLANE_DASHBOARD`
  - enable or disable the shared CloudWatch dashboard for the Dragon control plane. Defaults to `true`, and the production path keeps it enabled.
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

## GitHub Actions IAM

Do not run the GitHub workflows with `AdministratorAccess` or a single broad shared role.

Use the split-role policy in [aws/github-actions-iam.md](./aws/github-actions-iam.md):

- `BURN_DRAGON_P2P_AWS_ROLE_ARN`
  - normal operations only: deploy, restore, inspect, dataset publication
- `BURN_DRAGON_P2P_AWS_CLEANUP_ROLE_ARN`
  - destructive cleanup only

That document includes:

- an OIDC trust policy pinned to the `burn-dragon-p2p-staging` and `burn-dragon-p2p-production` GitHub environments
- a deploy-role policy covering the actual Terraform and workflow surface
- a separate cleanup-role policy for legacy and orphan teardown
- the minimal placeholder set you need to fill in before attaching the policies

The deployed EC2 runtime roles are still created by Terraform. Those roles need:

- SSM read access for auth client credentials when the selected auth connector uses them
- S3 object access for artifact publication and pruning
- when managed trainers are enabled, SSM read access for the trainer auth bundle parameter and KMS decrypt rights for the SSM key

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
- keep the managed trainer pool disabled on the normal production path so the fixed AWS spend stays under `$100`
- add capacity first through browser peers and operator-run native peers before enabling an always-on managed trainer pool
- if you enable the managed trainer pool for the larger NCA profile, choose the backend and instance type deliberately instead of relying on the default CPU bootstrap path

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
    "granted_roles": { "roles": ["TrainerCpu", "TrainerGpu", "Archive"] },
    "granted_scopes": [
      "Connect",
      "Discover",
      { "Train": { "experiment_id": "nca-prepretraining" } },
      { "Archive": { "experiment_id": "nca-prepretraining" } },
      { "Train": { "experiment_id": "climbmix-pretraining" } },
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

- `crates/burn_dragon_p2p/deploy/profiles/nca-r3.profile.json`
- `crates/burn_dragon_p2p/deploy/profiles/climbmix-r1.profile.json`

The checked-in `ruliad-1m` profile pair is the local ruliad trainability
baseline: `profiles/ruliad-1m.training.toml` and `profiles/ruliad-1m.corpus.toml`.
It remains intentionally small, AdamW-only, live-source-selected, and
answer-completion supervised so dataset/model issues show up in an isolated CE
comparison. The default ruliad research/training path is the JEPA-only profile
family, starting with `profiles/ruliad-1m.jepa.training.toml`. The
`ruliad-r1.eggroll-smoke.toml` override is quarantined for optimizer research
and should not be treated as the default ruliad path.

`BURN_DRAGON_P2P_CLIMBMIX_BROWSER_DATASET_BASE_URL` defaults to the managed dataset CDN path `https://datasets.dragon.aberration.technology/dragon-datasets/climbmix-pretraining/climbmix-r1`. Terraform publishes `${base_url}/fetch-manifest.json` into the initial ClimbMix browser profile. Browser peers still fetch only the shards they train on. With a runtime-provided training lease they use the exact assigned microshards; otherwise they use the bounded deterministic per-peer fallback advertised by the profile. The shipped Dragon browser app now reads that persisted browser training lease automatically before local training starts.

The active `nca-r3` profile keeps the NCA corpus and `8` shared Dragon layers, `512` hidden width, `1024` total latent width, native `512` token windows, batch `6`, and `24` native training steps per window. It factorizes the NCA input embedding into cell-position, cell-state, and special-token parameters, reducing the model from about `27.3M` to `1.62M` parameters without changing the task vocabulary. Its signed browser WebGPU tier uses the mathematically equivalent dense-score short-context executor with `256` token windows, `64` token TBPTT chunks, `batch_size = 6`, `8` train batches, `1` eval batch, and a `30s` lease budget. Batch `6` is the measured practical throughput choice for the production browser tier; capability checks still downgrade peers whose reported memory cannot fit it. The smaller full head fits the bounded asynchronous browser publication path, so a browser can contribute learned weights instead of telemetry alone. `nca-r1` and `nca-r2` remain immutable historical contracts.

On an NVIDIA GB10 Chromium/Vulkan probe, the dense executor matched the recurrent executor's 64-step validation loss within `0.00003` while increasing browser training throughput from `253` to `2,111 tok/s`. In a three-sample, fresh-browser sweep of the actual eight-step production window, the complete factorized path averaged `3,330`, `4,535`, and `4,562 tok/s` at batches `4`, `6`, and `8`. Batch `6` retains 25% more batch-memory headroom than batch `8` for only a 0.6% mean-throughput difference and was faster on the cold first window. The larger arms sustained roughly `87-91%` sampled GPU SM utilization during dense compute. These are compatibility and convergence smokes, not a statistical quality claim. Browser capability checks still downgrade peers that cannot safely fit the signed workload before allocating training buffers.

Those profile payloads are derived from the source configs in the same folder. To regenerate a profile locally:

```bash
cargo run -p burn_dragon_p2p --features native --bin burn_dragon_p2p_native -- \
  build-profile \
  --training-config crates/burn_dragon_p2p/deploy/profiles/nca-r3.training.toml \
  --experiment-kind nca \
  --revision-id nca-r3 \
  --output crates/burn_dragon_p2p/deploy/profiles/nca-r3.profile.json
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
  --browser-climbmix-manifest-url https://datasets.dragon.aberration.technology/dragon-datasets/climbmix-pretraining/climbmix-r1/fetch-manifest.json \
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
  login \
  --config /path/to/native-peer.toml \
  --experiment-kind nca \
  --backend wgpu \
  --edge-url https://edge.dragon.aberration.technology \
  --auth-bundle-out /path/to/admin-auth.json

cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  admin-rollout-profile \
  --config /path/to/native-peer.toml \
  --experiment-kind nca \
  --backend wgpu \
  --auth-bundle /path/to/admin-auth.json
```

The older `begin-github-login` + `complete-github-login` path still exists for manual or headless debugging, but the primary native operator flow is now the browser-launched loopback callback path above.

`admin-rollout-profile` uses the current local Dragon config and manifest metadata as the source of truth and pushes a replacement directory entry through `RolloutAuthPolicy`. Peers that rely on network-published Dragon profiles will pick up the new config from the directory instead of requiring a matching static local training config.

## Native Peer Join After Deploy

For the public production network, the simplest native path is the published
operator binary:

```bash
cargo install --locked burn_dragon_p2p --version 0.21.0 --bin burn_dragon_p2p_native
burn_dragon_p2p_native doctor --assert-ready
burn_dragon_p2p_native login
burn_dragon_p2p_native train-window-once --require-head-advanced
burn_dragon_p2p_native run-peer
```

Keep the explicit `--version` while the production line is pre-release; without
it, Cargo can select an older stable crate instead of the current mainnet
operator.

The crate default feature set includes `native,wgpu`, so the install command
above produces the portable WebGPU backend. Use a backend-specific install only
on a host with the matching driver and toolkit libraries:

```bash
cargo install --locked burn_dragon_p2p --version 0.21.0 --bin burn_dragon_p2p_native --no-default-features --features native,cuda
cargo install --locked burn_dragon_p2p --version 0.21.0 --bin burn_dragon_p2p_native --no-default-features --features native,rocm
```

With no `--config`, the binary points at
`https://edge.dragon.aberration.technology`, uses DNS TCP/QUIC seed multiaddrs,
joins `burn-dragon-mainnet` / `nca-prepretraining` / `nca-r3`, restores the
canonical head at startup, and resyncs it every 15 seconds while `run-peer` is
alive. Set `BURN_DRAGON_P2P_NATIVE_STORAGE_ROOT` if you want the auth cache,
materialized network profile, and checkpoints somewhere other than the default
XDG data directory.

For non-default deployments, use the outputs from the workflow summary:

- `edge_url`
- `seed_node_tcp_multiaddr`
- `seed_node_quic_multiaddr`

For browser peers, do not use the deprecated Terraform `seed_node_webrtc_multiaddr` output. Dialable browser WebRTC and WebTransport addresses require runtime `certhash` material, so browser seed bootstrap should come from the signed edge advertisement (`/browser/seeds/signed`) or the generated `browser-app-config.json`, not from static Terraform outputs.

Then run the native operator binary against that network, for example:

```bash
cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  login \
  --config /path/to/native-peer.toml \
  --experiment-kind nca \
  --backend wgpu \
  --edge-url https://edge.dragon.aberration.technology \
  --seed-node-url /dns4/edge.dragon.aberration.technology/tcp/4001 \
  --seed-node-url /dns4/edge.dragon.aberration.technology/udp/4001/quic-v1
```

Native peers can leave `training_config_paths` empty and rely on the published Dragon profile metadata for experiments that have a compatible network profile.

## Browser Join After Deploy

Open the deployed `browser_app_url` in a browser, sign in with GitHub, and join the network from the published GitHub Pages shell.

Browser peers can train directly from published Dragon profile metadata for experiments that include a browser-sized training profile. Production browser profiles are expected to load the active canonical head artifact and publish canonical browser updates through the p2p artifact/update path. The browser gates keep two trust domains explicit: the bounded training lane removes `live_participant` and proves browser shell, auth, seed derivation, direct WebRTC connectivity, and local WebGPU execution without claiming canonical work; the production-profile lane preserves every authority-signed model and execution field, loads the active head over P2P, submits receipt-only contributions, and proves their durable acceptance without publishing a canary artifact as the canonical head.

Before waiting for the full CI/deploy cycle, run:

```bash
cargo run -p xtask -- local-browser-e2e
```

That local gate runs the deployment config drift checks, a focused local edge/auth/browser training receipt e2e, and a single generated-NCA Chrome/WebGPU training smoke. It is intentionally narrower than `xtask smoke` and `xtask deploy-check` so browser/bootstrap changes can fail locally before the long native + wasm CI lanes or the AWS/Pages deployment workflows.

For deployment-shaped browser peer triage, run the same Playwright canary contract
against a locally built browser artifact before publishing Pages:

```bash
export BURN_DRAGON_BROWSER_CANARY_EDGE_BASE_URL=https://edge.dragon.aberration.technology
export BURN_DRAGON_BROWSER_CANARY_PRINCIPAL_ID=browser-canary-mainnet-nca
export BURN_DRAGON_BROWSER_CANARY_CALLBACK_TOKEN=...
export BURN_DRAGON_BROWSER_CANARY_EXPERIMENT_ID=nca-prepretraining
cargo run -p xtask -- local-browser-e2e --lane canary-webrtc-direct-training --build-site
```

The command above is the detached, bounded WebGPU execution lane. Run
`--lane canary-production-profile-training` to exercise the unchanged signed
profile and require accepted, durable live receipts.

Use `--lane canary-auto-connect`, `--lane canary-webrtc-direct-connect`,
`--lane canary-webrtc-direct-checkpoint`, or `--lane canary-all` to stop at the
network/checkpoint boundary before spending time on local WebGPU training. These
lanes write replayable artifacts to `target/test-artifacts/browser-peer-e2e/`:
the report JSON, console/network logs, trace, screenshot, effective canary
config, portal snapshot, signed seed advertisement, and browser config.

If `../burn_p2p` is intentionally dirty or on a different version than Dragon's
path dependency expects, run `cargo run -p xtask -- local-browser-e2e-ci-sibling`
with the same lane flags. The command creates temporary Dragon and `burn_p2p`
worktrees, applies the current Dragon diff, checks out the CI-pinned `burn_p2p`
ref, and keeps build outputs in `target/local-browser-e2e-ci-sibling/`.

For local production-edge triage, run the canary with `BURN_DRAGON_BROWSER_CANARY_EXPECT_TRAINING=0`, `BURN_DRAGON_BROWSER_CANARY_EXPECT_CHECKPOINT_SYNC=1`, and `BURN_DRAGON_BROWSER_CANARY_TRANSPORT_MODE=webrtc-direct`. That path should fail if the browser falls back to edge artifact HTTP for the active head.

For local agentic work, run long local commands and GitHub workflow waits
through the local task broker instead of streaming logs into the agent session:

```bash
cargo run -p xtask -- agent-task run \
  --label local-browser-e2e \
  --detach \
  -- cargo run -p xtask -- local-browser-e2e

cargo run -p xtask -- agent-task status
cargo run -p xtask -- agent-task summarize TASK_ID
```

For GitHub runs, dispatch or wait through the same broker:

```bash
cargo run -p xtask -- agent-task gh-wait \
  --repo aberration-technology/burn_dragon \
  --run-id RUN_ID \
  --wait \
  --exit-status
```

The deploy dispatch commands use `xtask agent-task gh-dispatch` internally and
pass a generated `agent_task_id` into child workflows, so runs can be
rediscovered by task id instead of branch/time polling. Pull full logs only
after the broker summary identifies the failed job or step.

## Terraform Root

The AWS Terraform root lives at:

- `crates/burn_dragon_p2p/deploy/terraform/aws`

Use `terraform.tfvars.example` as the starting point for any local/manual run. The GitHub Action does not require a checked-in `.tfvars` file as long as the environment variables and secrets above are configured.
