# burn_dragon_p2p

`burn_dragon_p2p` integrates `burn_p2p` with burn_dragon language experiments.

Current supported experiment families:

- NCA pre-pre-training
- ClimbMix pre-training

The crate is intentionally split into three layers:

- [config](src/config.rs): stable experiment, auth, and backend configuration
- [native](src/native.rs): native peer preparation for CPU, WGPU, CUDA, and ROCm
- [wasm](src/wasm/mod.rs): browser auth, Dioxus UI, and WebGPU browser training

It is still a library crate first, but both operator surfaces now exist:

- browser: productized through the Dioxus component and browser runtime
- native: productized through the `burn_dragon_p2p_native` operator binary

Deployment assets live in [deploy](deploy):

- [deploy/README.md](deploy/README.md): GitHub Actions, Terraform, and required repo/environment secrets
- [deploy/profiles](deploy/profiles): initial Dragon training-profile sources and published network profile payloads
- [deploy/terraform/aws](deploy/terraform/aws): checked-in AWS bootstrap/edge Terraform root

## Target Matrix

- native CPU:
  - feature set: `native`
  - intended for validation, reducers, and low-scale local trainer smoke
- native WGPU:
  - feature set: `native,wgpu`
  - intended for native GPU trainer peers
- native CUDA:
  - feature set: `native,cuda`
  - intended for native GPU trainer peers on CUDA hosts
- native ROCm:
  - feature set: `native,rocm`
  - intended for native GPU trainer peers on ROCm hosts
- browser WebGPU:
  - feature set: `wasm-ui,wasm-peer,wgpu`
  - intended for real browser trainer and verifier participation
- browser CPU:
  - feature set: `wasm-ui,wasm-peer`
  - smoke and development only

Browser CPU is not treated as a real deployment mode. The actual browser trainer path is WebGPU.

## Features

- `native`
  - enables native learner integration and shard-backed experiment prep
- `wasm-ui`
  - enables the Dioxus browser UI and browser auth/session flows
- `wasm-peer`
  - enables browser-local Dragon training and token-source loaders
- `wgpu`
  - enables native WGPU and browser WebGPU backends
- `cuda`
  - enables native CUDA peers
- `rocm`
  - enables native ROCm peers

There is intentionally no Cargo feature called `internet-scale`. Authenticated network participation is part of the normal runtime policy of this crate. The default deployed control plane uses GitHub auth, but the peer/browser surface follows the edge's configured browser login provider.

## Auth Model

For network participation:

- native peers require an authenticated edge auth bundle
- browser peers require an authenticated browser session when `require_edge_auth` is set
- browser training submission requires WebGPU
- dynamic admin edits are authenticated with a session-backed browser or native login, not a shared bootstrap token

The relevant seams are in:

- [auth.rs](src/auth.rs)
- [native.rs](src/native.rs)
- [wasm/mod.rs](src/wasm/mod.rs)

## Automatic Trainer Downgrade

Peers do not assume they can train just because the binary was built with `wgpu`, `cuda`, or `rocm`.

Both native and browser paths now run a local preflight assessment before advertising a trainer role:

- estimate model + optimizer + activation footprint from the actual Dragon revision config
- compare that estimate against the configured trainer memory budget
- downgrade automatically when the fit looks unsafe

Current default budgets are conservative:

- native CPU: `8 GiB`
- native WGPU: `4 GiB`
- native CUDA: `6 GiB`
- native ROCm: `6 GiB`
- browser WebGPU: `2 GiB`

Fallback policy:

- native peers: `trainer -> validator`
- browser peers: `browser_trainer_wgpu -> browser_verifier`

This is still a heuristic fit model, not a portable exact VRAM probe. The important product behavior is that undersized peers should downgrade before training starts instead of crashing on first optimizer allocation.

Native and browser peers also persist downgrade state for a specific workload fingerprint:

- experiment kind
- backend
- model config
- batch size
- block size

If a trainer run fails with a probable fit error like OOM / failed allocation / device loss, the next startup comes back as validator or verifier automatically instead of retrying trainer blindly. The downgrade record stops binding automatically if the configured trainer budget increases above the recorded failed footprint, and native peers can also clear it explicitly.

The browser app now renders the local capability decision directly:

- recommended role
- estimated training footprint
- trainer memory budget
- estimated tokens/sec
- checkpoint / shard / window budgets

## Browser Data Sources

Browser-local training supports:

- inline token windows
- HTTP JSON token-window shards
- HTTP shard manifests with per-shard integrity verification
- generated NCA corpora

That covers:

- synthetic NCA pre-pre-training
- shard-backed ClimbMix pre-training

For ClimbMix, the intended browser path is the shard-manifest form. The browser fetches
`fetch-manifest.json`, selects a bounded per-peer shard subset from the full shard pool,
downloads only those shard files on demand, verifies shard byte length and content hash, and then
decodes the token-window records locally. The checked-in profile uses deterministic peer selection
with a bounded shard window instead of walking the entire manifest from the front. When the host
runtime provides an exact browser training lease, the browser uses those assigned microshards
directly instead of the deterministic fallback.

## Join Mainnet

The public mainnet defaults are built into the native operator and the Pages
browser shell. This README uses `MAINNET_EDGE_URL` only for custom deployments
or local override examples.

The deployed network can publish Dragon experiment profiles directly in the directory. When those profiles are present, peers do not need a matching static experiment config on disk.
The deployed initial ClimbMix revision should point at a full external shard pool base URL. The
AWS deploy workflow publishes `${base_url}/fetch-manifest.json` into the initial browser profile,
so browser peers still fetch only the shards they train on without relying on repo-tracked shard blobs.
When the browser runtime has already persisted an exact training lease for the current assignment,
the Dragon browser app now picks that lease up automatically before local training starts.

### Browser Peer

The browser path is the intended product surface for browser operators.

Build the standalone WebGPU browser shell:

```bash
cargo run -p xtask -- build-browser-site --edge-url "$MAINNET_EDGE_URL"
```

That writes a static site bundle to `target/xtask/browser-site/`, including:

- `index.html`
- `browser-app-loader.js`
- `burn_dragon_p2p_browser.js`
- `burn_dragon_p2p_browser_bg.wasm`
- `browser-app-config.json`

The focused repo also ships a separate Pages workflow:

- `.github/workflows/deploy-pages.yml`

Before the workflow can publish, set the repository Pages source to `GitHub Actions` under `Settings > Pages`.

The generated browser shell now includes both surfaces:

- peer surface: connect, inspect assignments, and run browser-local training
- operator surface: inspect the live experiment directory, load a specific entry into a JSON editor, and roll out a replacement directory draft with an admin-scoped session

By default the baked browser config requests `Connect` and `Discover`, plus `Train` and `Archive` for the selected experiment id when one is provided. The separate `Sign In (Admin)` action extends that request with `ExperimentScope::Admin { study_id }` for the study id entered in the operator panel. Under the default deployment, that browser login provider is GitHub.

If you embed the UI yourself instead of using the generated shell, render [DragonBrowserApp](src/wasm/mod.rs) from your Dioxus host and point it at the edge:

```rust
use burn_dragon_p2p::config::{DragonBrowserAppConfig, DragonPeerNetworkConfig};
use burn_dragon_p2p::wasm::{DragonBrowserApp, DragonBrowserAppProps};

let config = DragonBrowserAppConfig {
    network: DragonPeerNetworkConfig::default()
        .with_edge_base_url(Some(std::env::var("MAINNET_EDGE_URL").unwrap()))
        .with_seed_node_urls(None),
    selected_experiment_id: None,
    selected_revision_id: None,
    requested_scopes: Default::default(),
    require_edge_auth: true,
    training: None,
};

let props = DragonBrowserAppProps {
    config,
    release_manifest: None,
};
```

At runtime:

1. open the browser app
2. connect to `MAINNET_EDGE_URL`
3. complete the GitHub login flow
4. resolve the selected experiment from the network directory
5. join as a WebGPU trainer or verifier

The browser app also accepts network overrides from query params:

- `?edge=https://edge.example`
- `?seed=/dnsaddr/seed-1.example/tcp/4001/p2p/...`
- repeated or comma-separated `seed` values

The browser runtime still bootstraps through the edge today, then reconciles the
site config with the live signed browser seed advertisement. Browser-capable
seeds should be DNS multiaddrs with runtime `certhash` material; raw static IP
WSS fallbacks are treated as degraded when direct browser transports are
advertised. The current browser transport contract is maintained in
[`burn_p2p`'s browser transport backend doc](https://github.com/aberration-technology/burn_p2p/blob/main/docs/browser-transport-backend.md),
while the Dragon deploy defaults and Pages canary gates live in
[deploy/README.md](deploy/README.md).

If the selected directory entry includes Dragon profile metadata and explicitly
allows `BrowserTrainerWgpu`, browser training can run without a static embedded
`training` config in the host app. Production profiles that exceed the browser
WebGPU memory budget still publish browser observer/verifier connectivity, but
they omit the training payload so the UI and canary do not advertise an unsafe
browser trainer path.

### Native Peer

The native join surface is now a real operator binary:

- `burn_dragon_p2p_native resolve-config`
- `burn_dragon_p2p_native assess-capability`
- `burn_dragon_p2p_native deployment-diagnostics`
- `burn_dragon_p2p_native doctor`
- `burn_dragon_p2p_native probe-swarm`
- `burn_dragon_p2p_native build-profile`
- `burn_dragon_p2p_native admin-export-directory`
- `burn_dragon_p2p_native admin-rollout-profile`
- `burn_dragon_p2p_native login`
- `burn_dragon_p2p_native begin-github-login`
- `burn_dragon_p2p_native complete-github-login`
- `burn_dragon_p2p_native enroll-static-principal`
- `burn_dragon_p2p_native train-window-once`
- `burn_dragon_p2p_native run-peer`
- `burn_dragon_p2p_native run-head-mirror`
- `burn_dragon_p2p_native run-validator-daemon`
- `burn_dragon_p2p_native mark-runtime-failure`
- `burn_dragon_p2p_native clear-downgrade`

Install the portable native trainer. The published default feature set includes
`native,wgpu`, so this produces a WebGPU-capable binary without extra flags:

```bash
cargo install --locked burn_dragon_p2p --version 0.21.0-pre.29 --bin burn_dragon_p2p_native
```

Keep the explicit `--version` while the production line is pre-release; without
it, Cargo can select an older stable crate instead of the current mainnet
operator.

Then join the public mainnet NCA experiment:

```bash
burn_dragon_p2p_native doctor --assert-ready
burn_dragon_p2p_native login
burn_dragon_p2p_native train-window-once --require-head-advanced
burn_dragon_p2p_native run-peer
```

With no `--config`, the binary uses the public Dragon edge at
`https://edge.dragon.aberration.technology`, DNS TCP/QUIC seeds for that edge,
the `burn-dragon-mainnet` / `nca-prepretraining` / `nca-r1` experiment ids, and
a storage root under `$XDG_DATA_HOME/burn_dragon_p2p/mainnet-native` or
`~/.local/share/burn_dragon_p2p/mainnet-native`. Override the storage root with
`BURN_DRAGON_P2P_NATIVE_STORAGE_ROOT` when running multiple peers on one host.

Install a narrower target when you need a backend-specific binary:

```bash
# CPU
cargo install --locked burn_dragon_p2p --version 0.21.0-pre.29 --bin burn_dragon_p2p_native --no-default-features --features native

# WGPU
cargo install --locked burn_dragon_p2p --version 0.21.0-pre.29 --bin burn_dragon_p2p_native --features native,wgpu

# CUDA
cargo install --locked burn_dragon_p2p --version 0.21.0-pre.29 --bin burn_dragon_p2p_native --no-default-features --features native,cuda

# ROCm
cargo install --locked burn_dragon_p2p --version 0.21.0-pre.29 --bin burn_dragon_p2p_native --no-default-features --features native,rocm
```

`--backend webgpu` is accepted as an alias for `--backend wgpu`. CUDA and ROCm
installs must be built with the matching feature on hosts that have the matching
driver and toolkit libraries available to the linker and runtime.

For custom networks, start from the example config in [deploy/native-peer.toml.example](deploy/native-peer.toml.example).

Resolve the config against a specific network before launching:

```bash
cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  resolve-config \
  --config path/to/peer.toml \
  --edge-url "$MAINNET_EDGE_URL" \
  --seed-node-url "/dnsaddr/seed-1.example/tcp/4001/p2p/..." \
  --seed-node-url "/dnsaddr/seed-2.example/tcp/4001/p2p/..."
```

That resolves the effective edge URL and seed node set. The same override
surface is used by `run-peer`.

If the selected directory entry includes Dragon profile metadata, native peers can leave `training_config_paths` empty and let the network-provided profile materialize the training config locally under the peer storage root.

Inspect the preflight capability decision before launching:

```bash
cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  assess-capability \
  --config path/to/peer.toml \
  --experiment-kind nca \
  --backend wgpu \
  --native-wgpu-memory-budget-mib 6144 \
  --output-format json
```

Useful override flags for both `resolve-config` and `assess-capability`:

- `--native-cpu-memory-budget-mib`
- `--native-wgpu-memory-budget-mib`
- `--native-cuda-memory-budget-mib`
- `--native-rocm-memory-budget-mib`
- `--browser-wgpu-memory-budget-mib`
- `--no-native-validator-fallback`
- `--no-browser-verifier-fallback`

Provision GitHub auth:

```bash
cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  login \
  --config path/to/peer.toml \
  --experiment-kind nca \
  --backend wgpu \
  --edge-url "$MAINNET_EDGE_URL" \
  --auth-bundle-out /var/lib/burn_dragon_p2p/auth-bundle.json
```

That launches the deployed browser callback bridge, completes GitHub SSO in the browser, relays the provider callback back into the local CLI over a loopback listener, and writes a refreshable auth bundle. The same bundle is also cached under the peer storage root, and `run-peer`, `run-head-mirror`, `run-validator-daemon`, and `train-window-once` now reuse that cache and attempt session refresh automatically before falling back to another browser login.

If the edge cannot infer the public Pages host for the native callback bridge,
set `BURN_DRAGON_P2P_BROWSER_APP_BASE_URL` to the deployed browser shell URL
before running `login`.

Use `train-window-once --require-head-advanced` as the native post-deploy smoke
when you need proof that the peer published a strictly newer experiment head.
The same `--require-head-advanced` flag is available on `deployment-diagnostics`
to make readiness fail while the matching edge head is still at global step `0`.

The manual two-step path remains available for headless or debugging workflows:

```bash
cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  begin-github-login \
  --config path/to/peer.toml \
  --experiment-kind nca \
  --backend wgpu \
  --edge-url "$MAINNET_EDGE_URL" \
  --pending-out /var/lib/burn_dragon_p2p/pending-login.json

cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  complete-github-login \
  --config path/to/peer.toml \
  --pending /var/lib/burn_dragon_p2p/pending-login.json \
  --provider-code "$GITHUB_PROVIDER_CODE" \
  --auth-bundle-out /var/lib/burn_dragon_p2p/auth-bundle.json
```

Run the long-lived peer:

```bash
cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  run-peer \
  --config path/to/peer.toml \
  --experiment-kind nca \
  --backend wgpu \
  --auth-bundle /var/lib/burn_dragon_p2p/auth-bundle.json \
  --status-interval-secs 30
```

`run-peer` restores the current experiment head at startup and resyncs it every
15 seconds by default. That keeps a later native peer aligned with canonical
work from earlier peers before it starts publishing new windows. It also
installs a Ctrl-C handler, requests upstream shutdown, and waits for the
runtime to exit cleanly instead of dropping detached background work.

There is also a deploy example systemd unit in [deploy/burn-dragon-p2p-native.service](deploy/burn-dragon-p2p-native.service).

If a native trainer failed at runtime and you want to inspect or override the persisted downgrade state, the helper binary also supports:

```bash
cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  mark-runtime-failure \
  --config path/to/peer.toml \
  --experiment-kind nca \
  --backend wgpu \
  --reason "out of memory allocating optimizer state"
```

```bash
cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  clear-downgrade \
  --config path/to/peer.toml \
  --experiment-kind nca \
  --backend wgpu
```

For downstream native launchers, the library still exposes the managed runtime seam that the operator binary itself uses:

- [spawn_prepared_native_peer](src/native_runtime.rs)
- [ManagedRunningNativePeer](src/native_runtime.rs)

## Dynamic Experiment Admin

The deployed bootstrap can publish updated Dragon experiment profiles without forcing peers to ship a new static config.

The secure admin path is:

1. deploy the network with explicit GitHub admin logins
2. authenticate through the normal edge login flow
3. use the session-backed browser operator UI or the native operator binary for admin actions
4. roll updated directory entries through `RolloutAuthPolicy`

The recommended day-to-day operator flow is now the browser shell:

1. open the deployed browser shell
2. click `Sign In (Admin)`
3. enter the study id, for example `burn-dragon-mainnet`
4. click `Load Directory`
5. click `Load Selected Entry` or paste a replacement entry JSON into the editor
6. click `Upsert Editor Entry` to update the local draft
7. click `Roll Out Directory`

The native operator binary remains the fallback path for scripted or headless rollout.

Generate a network-publishable Dragon profile from a local training config:

```bash
cargo run -p burn_dragon_p2p --features native --bin burn_dragon_p2p_native -- \
  build-profile \
  --training-config crates/burn_dragon_p2p/deploy/profiles/nca-r1.training.toml \
  --experiment-kind nca \
  --output /tmp/nca-r2.profile.json
```

Inspect the current network directory:

```bash
cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  admin-export-directory \
  --edge-url "$MAINNET_EDGE_URL"
```

Roll a replacement directory entry from a local Dragon config:

```bash
cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  admin-rollout-profile \
  --config path/to/native-peer.toml \
  --experiment-kind nca \
  --backend wgpu \
  --auth-bundle /var/lib/burn_dragon_p2p/auth-bundle.json
```

The rollout is session-authenticated. There is intentionally no deploy-time shared admin token in the production path.

## Build And Validation Harness

Install the local task runner:

```bash
cargo install --path xtask --force
```

Build coverage for the peer targets:

```bash
xtask build-native
xtask build-native-wgpu
xtask build-native-cuda
xtask build-native-rocm
xtask build-browser-cpu
xtask build-browser
xtask build-matrix
```

Validation ladder:

- `xtask smoke`
  - native WGPU smoke for:
    - NCA shard export + leased training windows
    - ClimbMix existing-shard multi-peer windows
    - browser/native manifest conformance on the same experiment net
  - real browser wasm smoke in headless Chrome/WebGPU via `wasm-bindgen-test-runner`
  - native CUDA build surface check
- `xtask mixed-fleet`
  - browser/native same-net mixed-fleet soak for:
    - NCA native windows plus browser trainer/verifier receipt cycles
    - ClimbMix multi-peer native windows plus browser trainer/verifier receipt cycles
  - ignored medium mixed-fleet rung for both experiments
- `xtask edge-drill`
  - local HTTP edge drill for both experiments
  - real native edge login + enrollment
  - real browser edge login + enrollment
  - session-gated directory access
  - browser training and validation receipt submission/ack against the same edge
- `xtask all`
  - build matrix
  - smoke
  - medium native scale rung
  - mixed-fleet smoke + scale rung
  - large native scale rung
  - edge-backed deployment rung

The wasm/browser smoke specifically covers:

- generated NCA training
- HTTP JSON shard training
- real Chrome + chromedriver execution with WebGPU flags
