# burn_dragon_p2p

`burn_dragon_p2p` integrates `burn_p2p` with burn_dragon language experiments.

Current supported experiment families:

- NCA pre-pre-training
- ClimbMix pre-training
- verifier-backed formal Ruliad pretraining

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
- [P2P production readiness](../../docs/p2p-production-readiness.md): trust,
  native/browser coherence, validation ladder, and remaining release gates

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

Preflight fallback policy:

- native peers: `trainer -> validator`
- browser peers: `browser_trainer_wgpu -> browser_verifier`

After a native trainer has already started, a probable OOM, allocation failure,
or device loss downgrades the live node to the read-only `viewer` role. The
current window may drain, but subsequent training windows are rejected.

This is still a heuristic fit model, not a portable exact VRAM probe. The important product behavior is that undersized peers should downgrade before training starts instead of crashing on first optimizer allocation.

Native and browser peers also persist downgrade state for a specific workload fingerprint:

- experiment kind
- backend
- model config
- batch size
- block size

If a trainer run fails with a probable local fit error like OOM / failed allocation / device loss, the live native node becomes read-only and the next startup remains observer-only; a browser reconnects as verifier/observer instead of retrying trainer blindly. Transient control-plane failures such as edge receipt submission 5xx responses are not treated as trainer fit failures, and stale records with those reasons do not bind. The downgrade record stops binding automatically if the configured trainer budget increases above the recorded failed footprint, and native peers can also clear it explicitly.

The browser app now renders the local capability decision directly:

- recommended role
- estimated training footprint
- trainer memory budget
- estimated tokens/sec
- checkpoint / shard / window budgets

Participation starts from a read-only posture. Capability transitions are
revisioned and run-scoped through `burn_ecs`, so stale asynchronous probe
results cannot overwrite a newer role decision. Observer and verifier peers do
not receive trainer authority merely because their hardware later reports
WebGPU support.

Native peers also recover in process. The default re-probe policy waits 30
seconds after downgrade, checks the configured training budget and 125% live
host-memory headroom, requires two successful probes, and retries with bounded
exponential backoff. A successful transition acknowledges the handled runtime
error, restores only the original CPU/GPU trainer role, and clears the
workload-specific downgrade record. Configure this under
`capability.native_reprobe`.

## Signed Revision And Genesis

Production peers should require one authority-signed
`RevisionContractBundle`. It binds:

- Dragon model, checkpoint, tokenizer, dataset, objective, optimizer,
  scheduler, recurrent-state, aggregation, and validation semantics
- the update codec used by native and browser contributors
- one complete, content-addressed model genesis artifact

Backend and local batch size remain capability choices and do not fork the
semantic revision.

Bootstrap loads contract files from `BURN_P2P_REVISION_CONTRACT_FILES`,
verifies both domain-separated Ed25519 signatures against the active trust
bundle, rejects conflicting contracts, and distributes verified bundles in the
browser-edge snapshot. Native nodes register the same bundle and use
`require_signed_revision_contracts(true)` for fail-closed startup.

Artifact hashing alone is not treated as model identity. Native peers decode
the signed genesis and stream a canonical tensor-pack digest without allocating
a second model-sized flat buffer. Browser peers decode the same record on the
CPU verification backend before WebGPU training, avoiding unsupported
synchronous WebGPU readback. Both compare names/layout and float32-normalized
values against the authority-signed `tensor_digest`.

The local deterministic initialization fallback is for tests and isolated
development. A public revision must provision and load the exact signed genesis
artifact; peers must not create private random initial weights.

## Convergence And Distributed State

The three-peer 1M-class release harness separates three questions:

- protocol parity: network candidates and promoted roots must match an
  independent tensor merge oracle exactly
- fair convergence: P2P is compared with centralized training over the same
  examples and the same number of optimizer updates
- upper-bound convergence: a sequential reference consumes the same examples
  with three times as many optimizer updates

Protocol parity passes. A historical pre-mask full-FedAvg run reached 66.45%
of centralized loss reduction with nine local steps and 64.32% with
three-step windows. Those values supervised repeated fixed-document EOS fill,
so they remain transport evidence but are not current quality evidence.

The artifact-window profile explicitly resets optimizer and scheduler state
per window. It is mechanically correct but is not the production
continual-pretraining default until it closes the convergence gate.
The protocol-aware DiLoCo path closes the corrected bounded local gate. Its
signed contract declares optimizer and scheduler state
`peer_local_persistent`, while artifact windows declare `reset_per_window`. A
six-round random-scaffold matrix with persistent peer-local AdamW state reached
91.63%-94.10% of matched synchronized trailing progress across seeds
1337-1339. Every batch had nonzero masked supervision, every network round
exactly matched the codec-aware protocol oracle, and hard DiLoCo request
failures remained zero. A matched three-seed FP16 matrix retained the
trajectory while halving parameter payload.

Run the release gate with
`BURN_DRAGON_P2P_PARITY_REQUIRE_CONVERGENCE=1` to make convergence parity a
hard assertion. Detailed methodology and remaining production gates are in
[P2P production readiness](../../docs/p2p-production-readiness.md).

## Protocol-Aware Native Training

The native operator separates network service from training:

- `run-peer` joins, synchronizes, serves, and reports a peer; it does not execute
  optimizer steps.
- `run-trainer-daemon` is the long-running trainer service. It reads the active
  signed revision and dispatches each step through `train_protocol_once`, so an
  ArtifactWindows revision cannot accidentally execute as DiLoCo or vice
  versa.
- `train-window-once` is the bounded ArtifactWindows operator primitive and
  intentionally rejects DiLoCo revisions.

The trainer daemon restores the canonical head before training and does not
initialize a private genesis unless explicitly requested for isolated
development. It pauses while its live role set is read-only, resumes after a
capability upgrade, backs off on disconnect/failure, and exports ECS ingress
pressure with its status. Use `--max-protocol-steps` for a finite deployment
smoke; the default of zero runs until shutdown.

```bash
cargo run -p burn_dragon_p2p --bin burn_dragon_p2p_native -- \
  run-trainer-daemon \
  --config ./crates/burn_dragon_p2p/deploy/native-peer.toml.example \
  --experiment-kind nca \
  --backend wgpu
```

Browser training currently implements ArtifactWindows only. A browser that
selects a DiLoCo revision is not offered a trainer configuration, and the
runtime also rejects stale/bypassed trainer configuration before loading
weights. It may still participate as an observer or verifier. Native DiLoCo is
therefore the convergence candidate today; mixed native/browser DiLoCo
training remains an explicit missing implementation rather than an implicit
fallback.

## Compact Browser Updates

The browser EGGROLL path publishes a `SeededFitness` compact update: shared
perturbation identity, contiguous generation numbers, exact batch digests, and
quantized or FP32 antithetic fitness values. It runs on a plain forward backend,
without enabling autodiff. Native peers regenerate the low-rank perturbations
and replay the update against the canonical base model.

Validators now independently recover content-verified microshards from the
authenticated assignment lease, reproduce the batch digest, replay the
contract-selected perturbation pairs, and compare fitness under signed
numerical tolerances. Replay-required updates fail admission unless that
evidence succeeds. Public arbitrary-peer promotion still requires a separate
multi-validator quorum and live quarantine/disagreement drill.

## Browser Data Sources

Browser-local training supports:

- inline token windows
- HTTP JSON token-window shards
- HTTP shard manifests with per-shard integrity verification
- generated NCA corpora
- generated formal Ruliad corpora

That covers:

- synthetic NCA pre-pre-training
- shard-backed ClimbMix pre-training
- structured-symbolic Ruliad proof training with verifier-compatible metadata

Generated Ruliad uses the same semantic contract, deterministic proof source,
272-token symbolic vocabulary, stream identity, and target-loss masks as the
native path. The real Chrome/WebGPU gate runs two train and two evaluation
batches and checks formal-family metadata plus block/TBPTT behavior. AdamW loss
scalars remain on the GPU through the browser window and are read once at the
window boundary, avoiding a per-step WebGPU synchronization.

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
cargo install --locked burn_dragon_p2p --version 0.21.0 --bin burn_dragon_p2p_native
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
cargo install --locked burn_dragon_p2p --version 0.21.0 --bin burn_dragon_p2p_native --no-default-features --features native

# WGPU
cargo install --locked burn_dragon_p2p --version 0.21.0 --bin burn_dragon_p2p_native --features native,wgpu

# CUDA
cargo install --locked burn_dragon_p2p --version 0.21.0 --bin burn_dragon_p2p_native --no-default-features --features native,cuda

# ROCm
cargo install --locked burn_dragon_p2p --version 0.21.0 --bin burn_dragon_p2p_native --no-default-features --features native,rocm
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

- `xtask local-browser-e2e`
  - fastest local production-parity gate for browser training changes
  - deployment config drift checks, including Pages seed/canary contracts
  - local HTTP edge/auth drill with one NCA browser trainer path and receipt ack
  - one real Chrome/WebGPU generated-NCA browser training smoke
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

### Native Convergence Parity

The ignored `ruliad_native_runtime_1m_convergence_matches_federated_oracle`
integration test is the release-mode learning and communication gate for a
small native fleet. It runs a 926,210-parameter Dragon model on three local
trainer peers and verifies:

- one shared training contract, content-bound dataset view, and canonical genesis
- disjoint, non-empty microshard leases for every peer and round
- candidate artifact transfer, update visibility, and three-receipt promotion
- exact candidate replay and exact promoted-tensor parity with an independent
  weighted-merge/root-EMA oracle
- validation parity on the materialized promoted artifact
- a compute-matched sequential comparator, reported separately from protocol
  parity

Run the full comparator:

```bash
RUSTC="$(rustup which rustc --toolchain stable)" \
RUSTFLAGS="-C target-cpu=native" \
BURN_DRAGON_P2P_PARITY_ROUNDS=2 \
BURN_DRAGON_P2P_PARITY_REPORT_ROOT=target/test-artifacts/p2p-convergence-parity \
"$(rustup which cargo --toolchain stable)" test --release \
  -p burn_dragon_p2p --test native_training \
  ruliad_native_runtime_1m_convergence_matches_federated_oracle -- \
  --ignored --exact --nocapture
```

For a multi-seed protocol matrix, set
`BURN_DRAGON_P2P_PARITY_REPLAY=false` and
`BURN_DRAGON_P2P_PARITY_SEQUENTIAL=false`. That mode still performs real
training, transport, promotion, artifact materialization, and independent
merge verification; it skips duplicate local candidate replay and sequential
training. Each seed writes a self-contained JSON report.

This local gate uses the development genesis path and records
`signed_revision_contract_exercised=false`. It does not replace the clean-store
staging canary for an authority-signed revision and provisioned genesis
artifact.

### Native DiLoCo Convergence Parity

The ignored `ruliad_native_runtime_1m_diloco_matches_protocol_oracle` test uses
the same 1M-class model and data accounting, but executes three local inner
loops followed by one codec-aware DiLoCo outer update. With random scaffolds
enabled, the model has 1,012,229 parameters and synchronizes 225,797 values
(22.31%). It checks the
matched cohort, rotating reducer, contribution commitments, aggregate and
parameter equality, deterministic peer identities, request-failure telemetry,
masked supervision, and convergence against synchronized AdamW over the same
microbatches.

Run one hard-gated release condition:

```bash
RUSTC="$(rustup which rustc --toolchain stable)" \
RUSTFLAGS="-C target-cpu=native" \
BURN_DRAGON_P2P_PARITY_SEED=1339 \
BURN_DRAGON_P2P_PARITY_ROUNDS=6 \
BURN_DRAGON_P2P_PARITY_LOCAL_STEPS=1 \
BURN_DRAGON_P2P_PARITY_RANDOM_SCAFFOLD=true \
BURN_DRAGON_P2P_PARITY_REQUIRE_CONVERGENCE=1 \
BURN_DRAGON_P2P_DILOCO_CODEC=fp32 \
BURN_DRAGON_P2P_DILOCO_REPORT_ROOT=target/test-artifacts/random-scaffold-diloco/release-default \
"$(rustup which cargo --toolchain stable)" test --release \
  -p burn_dragon_p2p --features native --test native_training \
  ruliad_native_runtime_1m_diloco_matches_protocol_oracle -- \
  --ignored --exact --nocapture
```

Repeat with distinct seeds for a promotion matrix. The corrected FP32 matrix
passes three seeds at 91.63%-94.10% of synchronized-reference trailing
learning progress in release profile, with monotonic validation and no hard
request failure. The untouched seed-1339 holdout achieved 92.89% trailing
progress. Across the matched three-seed matrix, FP16 changed mean final CE by
`+0.0000061`, reduced estimated wire payload from 21,676,512 to 10,838,256
bytes per run, and changed mean aggregate peer-step throughput from 1.031 to
1.014 steps/s on loopback. FP16 remains opt-in pending constrained-link and
browser measurements.

The complete local methodology and current production blockers are tracked in
[P2P production readiness](../../docs/p2p-production-readiness.md).
