# Browser Libp2p Roadmap

This document is the implementation handoff for moving browser peers from the
current edge-mediated runtime to a browser-capable libp2p runtime that behaves
much more like the native swarm.

It is written from the perspective of `burn_dragon`, but almost all transport
and swarm work belongs in `burn_p2p`.

## Executive Summary

The target architecture is:

1. browser authenticates and bootstraps through the edge
2. browser receives signed swarm bootstrap material from the edge
3. browser dials real bootstrap/seed multiaddrs directly
4. browser joins overlays and receives steady-state updates from the swarm
5. browser fetches artifacts from peers first and uses the edge only as a
   fallback

The implementation constraint is explicit:

- browser swarm logic should stay in Rust/wasm
- JS should be a thin transport bridge only where browser APIs force it
- `burn_dragon` should consume a finished browser swarm abstraction, not
  reimplement transport logic itself

## Problem Statement

Today the browser runtime is useful, but it is not a first-class libp2p swarm
peer in the same sense as the native runtime.

Current symptoms:

- browser state is still edge-mediated for the critical path
- transport status is partially synthetic
- peer counts are not fully real
- steady-state updates still depend on HTTP surfaces
- artifact fetch still depends heavily on edge routes
- deployment can look healthy while browser peer behavior is still degraded

That is good enough for a bootstrap product, but it is not the right long-term
shape if browser peers are expected to operate like real network participants.

## Goal

After browser auth/bootstrap completes, a browser peer should:

- dial real seed/bootstrap multiaddrs directly
- establish a real browser-capable libp2p transport
- subscribe to the same overlays a native peer needs for assignment/head sync
- receive steady-state directory/head/metrics/artifact updates from the swarm
- report real transport, real peers, and real sync state in diagnostics/UI

The target is not byte-for-byte equivalence with native nodes. Browsers cannot
use raw TCP or raw QUIC. The target is first-class swarm participation over
browser-capable transports.

## Non-Goals

- removing the edge entirely
- making browsers use native-only transports
- putting substantial swarm logic in JS
- forcing `burn_dragon` to own browser transport internals that belong in
  `burn_p2p`

## Design Principles

### Wasm First

Browser runtime logic should live in Rust/wasm:

- connection lifecycle
- peer state
- overlay joins
- artifact routing policy
- retry logic
- sync orchestration
- diagnostics/state exposure

JS is acceptable only for:

- exposing browser-only transport primitives that Rust libp2p cannot currently
  reach directly
- small bindings to browser APIs
- feature detection for browser transport support

The design should treat JS as a narrow adapter, not as the browser swarm
runtime.

### Edge For Bootstrap, Swarm For Steady State

The edge remains important, but its role should narrow to:

- auth start / callback / enroll
- trust/policy distribution
- signed bootstrap material
- recovery fallback
- diagnostics surfaces

The edge should not remain the normal steady-state source for:

- head propagation
- directory assignment
- metrics synchronization
- artifact fetch

### Truthful State

The browser UI and diagnostics must report:

- what is configured
- what was attempted
- what is actually connected
- what transport is actually active
- where the current data came from

No state should be inferred from “recommended transport” or “edge snapshot was
reachable.”

## Current State

Today the browser runtime is not a first-class libp2p swarm peer.

Evidence:

- `burn_p2p_swarm` keeps the real libp2p swarm builder and transport stack
  behind native-only cfg gates in
  [lib.rs](/home/mosure/repos/burn_p2p/crates/burn_p2p_swarm/src/lib.rs#L14)
- the browser control-plane client is HTTP-backed in
  [browser_edge.rs](/home/mosure/repos/burn_p2p/crates/burn_p2p_swarm/src/browser_edge.rs#L24)
- the Dragon browser runtime starts from the edge snapshot and then syncs
  runtime state through the edge client in
  [training.rs](/home/mosure/repos/burn_dragon/crates/burn_dragon_p2p/src/wasm/training.rs#L820)
- browser transport selection is currently a recommendation, not a live dialed
  transport, in
  [transport.rs](/home/mosure/repos/burn_p2p/crates/burn_p2p_browser/src/transport.rs#L115)
  and
  [worker.rs](/home/mosure/repos/burn_p2p/crates/burn_p2p_browser/src/worker.rs#L792)
- the browser app currently derives “direct peers” from `transport.active` in
  [app.rs](/home/mosure/repos/burn_p2p/crates/burn_p2p_browser/src/app.rs#L259)
- `seed_node_urls` exist in browser config, but the current browser connect
  path does not use them through a real browser swarm runtime

There is one precursor already:

- browser-side artifact fetch can optionally use a peer transport hook in
  [auth.rs](/home/mosure/repos/burn_p2p/crates/burn_p2p_browser/src/auth.rs#L1486)
  and
  [auth.rs](/home/mosure/repos/burn_p2p/crates/burn_p2p_browser/src/auth.rs#L2045)

That is useful, but it is only a narrow artifact path, not a browser swarm.

## Target Symmetry Model

Browser and native peers should become symmetric along the following axes.

### Symmetric

- both have a real peer identity
- both dial real seed/bootstrap addresses
- both join real overlays
- both learn heads from the swarm
- both learn directory/assignment from the swarm
- both prefer peer artifact transport
- both expose real connected peers and transport state

### Intentionally Asymmetric

- native peers may use TCP/QUIC; browser peers may not
- browser auth/bootstrap depends on browser-facing edge flows
- browser transport backend may need a thin adapter for WebRTC/WebTransport
- browser resource limits and background lifecycle are stricter

The target is not “identical implementation.” It is “same network role, browser
appropriate transport surface.”

## Target Architecture

The desired split is:

1. edge HTTP for auth/bootstrap/fallback only
2. browser swarm runtime for steady-state peer participation

### Edge Responsibilities

- login start / callback / enroll
- trust bundle and policy distribution
- signed browser seed advertisement
- initial signed directory/head bootstrap material
- diagnostics and operator fallback
- optional artifact fallback

### Browser Swarm Responsibilities

- direct seed dialing from signed/bootstrap seed set
- transport establishment
- overlay/topic subscription
- steady-state directory/head/metrics propagation
- peer artifact fetch and chunk fetch
- real peer presence and transport reporting

## Browser Bootstrap Source Of Truth

The roadmap should not rely only on static `seed_node_urls`.

That is too brittle because:

- Pages artifacts can outlive infra changes
- bootstrap transport addresses can change across deploys
- transport policy can change independently of the site artifact

Preferred model:

1. browser authenticates through the edge
2. browser fetches a signed browser seed advertisement from the edge
3. browser reconciles that with any statically baked `seed_node_urls`
4. browser dials the signed set first
5. static site-config seeds remain fallback only

Required output in `burn_p2p`:

- one signed seed advertisement payload for browser peers
- one deterministic merge policy between:
  - edge-published browser-dialable multiaddrs
  - site-config seed URLs
- one diagnostics field reporting which source was used:
  - `edge_signed`
  - `site_config_fallback`
  - `merged`

## Signed Browser Seed Advertisement

`burn_p2p` should define a stable signed payload similar to:

```json
{
  "network_id": "burn-dragon-mainnet",
  "issued_at": "2026-04-16T00:00:00Z",
  "expires_at": "2026-04-16T01:00:00Z",
  "transport_policy": {
    "preferred": ["webrtc-direct", "webtransport", "wss-fallback"],
    "allow_fallback_wss": true
  },
  "seeds": [
    {
      "peer_id": "12D3KooW...",
      "multiaddrs": [
        "/dns4/edge.dragon.aberration.technology/udp/4001/webrtc-direct",
        "/dns4/edge.dragon.aberration.technology/udp/443/webtransport",
        "/dns4/edge.dragon.aberration.technology/tcp/443/wss"
      ]
    }
  ],
  "signature": "..."
}
```

Required properties:

- scoped to one network
- short-lived
- signed by the deployment authority
- explicit about transport priority
- explicit about fallback policy

## Transport Strategy

Recommended order:

1. WebRTC direct
2. WebTransport
3. WSS fallback

Rationale:

- WebRTC direct is the closest browser-native peer path
- WebTransport is a real browser-capable transport and a good second option
- WSS fallback can preserve reachability, but should be treated as degraded

`burn_p2p` should represent transport state explicitly:

- `unresolved`
- `dialing_webrtc`
- `connected_webrtc`
- `dialing_webtransport`
- `connected_webtransport`
- `dialing_wss`
- `connected_wss`
- `failed`

The UI should not flatten that to a single “connected” label.

## Browser Runtime State Machine

The browser runtime should expose a first-class state machine:

1. `unauthenticated`
2. `auth_bootstrap`
3. `signed_seed_resolved`
4. `dialing_seed`
5. `transport_connected`
6. `overlay_joining`
7. `overlay_joined`
8. `directory_synced`
9. `head_synced`
10. `artifact_ready`
11. `training_ready`
12. `training_active`
13. `degraded_fallback`
14. `failed`

Each transition needs:

- cause
- timestamp
- active transport
- connected peer count
- last error

This must be library state in `burn_p2p`, not ad hoc UI state in `burn_dragon`.

## Bootstrap Sequence

Expected browser startup flow:

1. load site config
2. load durable browser session
3. if no session, remain `unauthenticated`
4. if session exists:
   - fetch edge snapshot
   - fetch signed seed advertisement
   - fetch trust bundle / policy
5. merge signed seeds with static seeds
6. dial seeds in preferred transport order
7. establish transport
8. join overlays
9. receive directory/head updates
10. fetch active checkpoint artifact if needed
11. enter `training_ready`

Failure handling:

- if seed advertisement fetch fails, attempt `site_config_fallback`
- if direct swarm join fails, enter `degraded_fallback`
- edge polling fallback should be explicit and visible, not silent

## Overlay Participation Requirements

The browser runtime should not stop at “connected transport.”

Required overlay participation:

- directory/assignment overlay
- head announcement overlay
- metrics live/catchup overlay if applicable
- artifact provider discovery overlay if separate

Required diagnostics:

- joined overlays
- failed overlays
- last subscription error
- current assignment source:
  - `swarm`
  - `edge_fallback`

## Artifact Transport Model

Artifact fetch should become symmetric too.

Preferred artifact order:

1. peer swarm direct provider fetch
2. browser transport relay/gateway if necessary
3. edge HTTP fallback

Required browser diagnostics:

- active head artifact source
- artifact provider peer ids
- chunk source counts
- fallback reason if edge was used

Required invariants:

- peer artifact transport verifies manifest and chunk integrity exactly like
  edge fallback
- edge fallback is visible in runtime state
- successful peer transport should populate real provider peer ids in browser
  state

## Proposed `burn_p2p` Integration Contract

The first engineer should not have to invent the runtime boundary from scratch.

Recommended library contract:

### `BrowserSwarmBootstrap`

- authenticated browser identity/session handle
- trust bundle
- selected experiment/revision
- browser-dialable signed seed advertisement
- site-config fallback seeds
- transport policy

### `BrowserSwarmStatus`

- bootstrap source
- desired transport
- connected transport
- connected peer ids
- joined overlays
- assignment source
- head sync state
- artifact transport mode
- last dial/sync error
- current phase in the runtime state machine

### `BrowserSwarmRuntime`

- `connect(bootstrap)`
- `disconnect()`
- `status()`
- `subscribe_directory()`
- `subscribe_heads()`
- `subscribe_metrics()`
- `fetch_artifact_manifest()`
- `fetch_artifact_chunk()`
- `force_edge_resync()`

`burn_p2p_browser` should consume this contract and stop synthesizing
connection state from HTTP snapshot state.

## Crate Ownership

### `burn_p2p_core`

Own:

- signed browser seed advertisement schema
- transport/status enums
- browser swarm diagnostics schema

### `burn_p2p_swarm`

Own:

- browser swarm runtime abstraction
- browser-capable transport backend
- overlay join logic
- seed dialing and retry policy

### `burn_p2p_browser`

Own:

- browser app wiring to the new runtime
- durable browser session to swarm bootstrap translation
- truthful UI/runtime state exposure
- edge fallback orchestration

### `burn_dragon`

Own:

- passes site config, edge URL, fallback seeds
- presents product-facing state
- deploys browser-capable listeners and seed advertisements
- adds deployment/browser canaries

## Protocol Surface Changes

The roadmap is not complete unless the protocol surfaces that have to change are
called out explicitly.

### New Or Changed Bootstrap Surfaces

`burn_p2p` should add or formalize:

- one signed browser seed advertisement response
- one browser transport policy response
- one browser-visible runtime diagnostics response that distinguishes:
  - edge bootstrap success
  - seed resolution success
  - transport success
  - overlay join success

These surfaces should be treated as bootstrap-only inputs. The browser should
not keep polling them as its steady-state data plane.

### New Browser Swarm Status Contract

The browser runtime needs a stable internal state contract for:

- selected transport family
- live connected transport family
- connected seed peer ids
- overlay join state by topic
- head source
- directory/assignment source
- artifact source
- last fatal and last recoverable error

This should exist in `burn_p2p_core` or another shared contract crate so
diagnostics, UI, and tests use the same schema.

### Artifact Transport Protocol Expectations

The swarm-side artifact path must support:

- manifest request
- chunk request
- provider discovery
- fallback signaling
- replay/resume when the browser tab reconnects

The browser should not have to infer “peer transport probably worked” from a
successful edge fallback.

## Native And Browser Parity Targets

This is the concrete parity target, not a vague aspiration.

### Connection Parity

Native today:

- dials seeds directly
- owns a real swarm
- exposes real peers and overlays

Browser target:

- dials seeds directly after auth bootstrap
- owns a real browser-capable swarm
- exposes real peers and overlays

### State Propagation Parity

Native today:

- receives head and directory state from the swarm

Browser target:

- receives head and directory state from the swarm after bootstrap
- may use the edge only for initial snapshot and explicit recovery

### Artifact Parity

Native today:

- gets artifacts from peers/CAS/publication surfaces as a real swarm participant

Browser target:

- gets artifacts from peers first
- uses the edge only as explicit fallback

### Diagnostics Parity

Native today:

- can describe real runtime state

Browser target:

- can describe real runtime state with no synthetic transport/peer labels

## Compatibility And Rollout Rules

The browser swarm rollout has to coexist with the current edge-mediated path
without breaking existing deployments.

### Feature Flag Strategy

Recommended rollout flags in `burn_p2p`:

- `browser_signed_seed_bootstrap`
- `browser_swarm_seed_dial`
- `browser_swarm_state_sync`
- `browser_swarm_artifact_transport`
- `browser_edge_fallback`

Each stage should be able to run with the later stages disabled.

### Backward Compatibility Rules

- old Pages artifacts must still be able to use edge-only bootstrap
- old bootstrap nodes must not advertise browser swarm support they cannot
  actually satisfy
- browser swarm transport should only be considered available when both:
  - deployment advertises it
  - runtime confirms it joined successfully

### Protocol Versioning

The signed browser seed advertisement and browser swarm diagnostics contract
should carry an explicit version.

At minimum:

- `schema_version`
- `network_id`
- `transport_policy_version`

Do not rely on “best effort JSON evolution” for the browser swarm bootstrap
contract.

## Failure And Degradation Model

Browser peers need explicit, bounded degradation instead of silently falling
back forever.

Required degradation states:

- `bootstrap_only`
- `seed_resolution_failed`
- `transport_unavailable`
- `overlay_join_failed`
- `artifact_peer_transport_failed`
- `edge_fallback_active`
- `recovery_required`

Required behavior:

- edge fallback should be explicit in diagnostics
- edge fallback should not masquerade as full browser swarm success
- deploy canaries should treat permanent fallback as degraded, not healthy

## Risk Register

These are the main technical risks that should be tracked up front.

### Risk 1: Rust wasm transport support is insufficient

Mitigation:

- keep the runtime boundary in Rust
- allow a thin JS transport adapter behind a Rust trait only if necessary
- do not move browser swarm orchestration into JS

### Risk 2: Deployment advertises transports that browsers cannot really use

Mitigation:

- browser-capable transport probes in deploy validation
- production diagnostics must prove joinability, not just advertise flags

### Risk 3: Browser background lifecycle breaks long-lived swarm assumptions

Mitigation:

- explicit suspend/resume behavior
- resumable artifact fetch
- reconnect path treated as first-class behavior in tests

### Risk 4: Edge fallback remains the accidental steady-state path

Mitigation:

- explicit source reporting for head/directory/artifact state
- canaries should fail if steady-state remains edge-backed where swarm is
  expected

## Definition Of Done By Milestone

The phase plan should map to concrete “done” definitions.

### Milestone 1 Done

- browser consumes signed seed advertisement
- browser dials bootstrap directly
- browser reports a real connected transport
- browser reports real seed peer ids
- browser can receive a head and assignment without repeated edge polling

### Milestone 2 Done

- browser remains current from swarm updates
- browser reconnect/suspend/resume is stable
- browser UI no longer depends on synthetic peer/transport state

### Milestone 3 Done

- browser artifact fetch prefers peer swarm
- edge artifact route is fallback only
- artifact diagnostics show source and fallback reason

### Production-Ready Done

- deploy canary proves browser joinability
- deploy diagnostics prove browser-capable transport health
- browser/native peers are functionally symmetric on:
  - seed dial
  - overlay join
  - head sync
  - assignment sync
  - artifact fetch preference

## Dependency Graph

The phases are not independent. The order below is the minimum dependency chain.

### Hard Dependencies

- truthful diagnostics must land before transport rollout
- signed browser seed advertisement must exist before real browser seed dial
- real browser seed dial must exist before swarm-based steady-state sync
- swarm-based steady-state sync must exist before artifact transport can be
  judged as symmetric
- deployment hardening must happen before production can claim browser/native
  symmetry

### Recommended Build Order

1. diagnostics contracts and browser-visible state taxonomy
2. signed seed advertisement schema and bootstrap fetch path
3. browser swarm runtime trait boundary
4. first real browser transport backend
5. overlay join and swarm-fed assignment/head state
6. peer-first artifact transport
7. deployment/browser canary and production hard gates

## Deliverables By Phase

Each phase should produce concrete code, not just behavior.

### Phase 0 Deliverables

- new diagnostics/status schema in `burn_p2p_core`
- browser runtime/UI wiring updated to expose truthful state
- tests proving peer/transport labels are not synthetic

### Phase 1 Deliverables

- signed browser seed advertisement schema
- edge endpoint or response field that returns the signed advertisement
- browser seed merge policy implementation
- diagnostics field showing the actual bootstrap source

### Phase 2 Deliverables

- `BrowserSwarmRuntime` trait and supporting status types
- one browser-capable transport backend
- browser seed dial orchestration in wasm
- tests covering successful and failed seed dial paths

### Phase 3 Deliverables

- swarm-fed head update path
- swarm-fed directory/assignment update path
- explicit edge recovery path
- tests proving the browser can stay current without repeated edge polling

### Phase 4 Deliverables

- browser swarm artifact manifest/chunk fetch path
- provider discovery wiring
- explicit fallback accounting for edge artifact fetch
- tests proving peer-first artifact preference

### Phase 5 Deliverables

- deployment/browser canary
- diagnostics gate for browser transport joinability
- production transport matrix validation
- operator-facing diagnostics for bootstrap seed advertisement and browser
  transport health

## Milestone Ownership Map

This section exists so work can be split cleanly across repos and crates.

### `burn_p2p_core`

- signed browser seed advertisement schema
- browser swarm status and diagnostics schema
- transport family/status enums
- protocol versioning for browser bootstrap contracts

### `burn_p2p_swarm`

- browser swarm runtime trait
- transport implementation
- seed dial orchestration
- overlay join logic
- reconnect/suspend-resume logic

### `burn_p2p_browser`

- browser session/bootstrap integration
- runtime/UI state projection
- explicit fallback orchestration
- artifact transport selection policy

### `burn_dragon`

- Pages config and product-facing browser state
- deployment/browser canary
- production transport policy exposure
- deployment diagnostics integration

## Suggested Issue Breakdown

If this roadmap is turned into issues, the first useful set should be:

1. add browser swarm status contract to `burn_p2p_core`
2. add signed browser seed advertisement schema and versioning
3. expose signed browser seed advertisement from bootstrap edge
4. introduce `BrowserSwarmRuntime` trait in `burn_p2p_swarm`
5. implement first real browser transport backend
6. wire `burn_p2p_browser` to real transport state
7. add swarm-fed directory/head sync
8. add peer-first artifact fetch in browser runtime
9. add deployment/browser canary in `burn_dragon`

Each issue should name:

- owning crate
- dependency issue ids
- acceptance test
- rollback/degradation behavior

## Concrete Rust API Sketch

The roadmap should include one plausible Rust shape so implementers do not have
to invent the API boundary during the first PR.

### Core Bootstrap Types

```rust
pub struct BrowserSeedAdvertisement {
    pub schema_version: u32,
    pub network_id: NetworkId,
    pub issued_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub transport_policy: BrowserTransportPolicy,
    pub seeds: Vec<BrowserSeedRecord>,
    pub signature: BrowserAdvertisementSignature,
}

pub struct BrowserSeedRecord {
    pub peer_id: PeerId,
    pub multiaddrs: Vec<Multiaddr>,
}

pub struct BrowserSwarmBootstrap {
    pub session: BrowserSessionBinding,
    pub trust_bundle: TrustBundle,
    pub transport_policy: BrowserTransportPolicy,
    pub seed_source: BrowserSeedSource,
    pub signed_advertisement: Option<BrowserSeedAdvertisement>,
    pub fallback_seed_urls: Vec<String>,
    pub selected_experiment: Option<ExperimentId>,
    pub selected_revision: Option<RevisionId>,
}
```

### Runtime Status Types

```rust
pub enum BrowserSwarmPhase {
    Unauthenticated,
    AuthBootstrap,
    SignedSeedResolved,
    DialingSeed,
    TransportConnected,
    OverlayJoining,
    OverlayJoined,
    DirectorySynced,
    HeadSynced,
    ArtifactReady,
    TrainingReady,
    TrainingActive,
    DegradedFallback,
    Failed,
}

pub enum BrowserTransportFamily {
    WebRtcDirect,
    WebTransport,
    WssFallback,
}

pub struct BrowserSwarmStatus {
    pub phase: BrowserSwarmPhase,
    pub bootstrap_source: BrowserBootstrapSource,
    pub desired_transport: Option<BrowserTransportFamily>,
    pub connected_transport: Option<BrowserTransportFamily>,
    pub connected_peer_ids: Vec<PeerId>,
    pub joined_overlays: Vec<OverlayTopic>,
    pub assignment_source: BrowserAssignmentSource,
    pub head_source: BrowserHeadSource,
    pub artifact_source: BrowserArtifactSource,
    pub last_error: Option<String>,
}
```

### Runtime Trait Boundary

```rust
#[async_trait(?Send)]
pub trait BrowserSwarmRuntime {
    async fn connect(&mut self, bootstrap: BrowserSwarmBootstrap) -> Result<()>;
    async fn disconnect(&mut self) -> Result<()>;
    fn status(&self) -> BrowserSwarmStatus;
    async fn subscribe_directory(&mut self) -> Result<()>;
    async fn subscribe_heads(&mut self) -> Result<()>;
    async fn subscribe_metrics(&mut self) -> Result<()>;
    async fn fetch_artifact_manifest(
        &mut self,
        request: BrowserArtifactRequest,
    ) -> Result<ArtifactManifest>;
    async fn fetch_artifact_chunk(
        &mut self,
        request: BrowserArtifactChunkRequest,
    ) -> Result<Vec<u8>>;
}
```

This is not mandatory syntax, but the actual implementation should be this
level of explicit.

## File And Module Touch Map

The roadmap should also show likely landing zones so work is not scattered
randomly.

### `burn_p2p_core`

Likely files/modules:

- new `browser_bootstrap.rs` or `schema/browser.rs`
- transport/status enums in a schema module
- signed advertisement schema + signature validation helpers

### `burn_p2p_swarm`

Likely files/modules:

- new wasm/browser swarm backend module
- transport backend abstraction
- seed dial orchestration
- overlay join state tracking

### `burn_p2p_browser`

Likely files/modules:

- browser auth/bootstrap integration
- UI binding/state projection
- artifact transport policy integration
- fallback orchestration

### `burn_dragon`

Likely files/modules:

- wasm UI state rendering
- deployment diagnostics/canary wiring
- Pages config for browser seed bootstrap expectations

## Test And Canary Scenario Catalog

The testing matrix is good, but implementers also need named scenarios.

### Scenario A: Browser Bootstrap Only

- browser signs in
- browser resolves signed seeds
- browser dials bootstrap directly
- browser receives head and assignment
- browser trains with bootstrap-only topology

### Scenario B: Browser + Native Trainer

- native trainer publishes head updates
- browser receives head changes over swarm
- browser stays current without edge steady-state polling

### Scenario C: Transport Downgrade

- WebRTC unavailable
- browser falls back to WebTransport
- if WebTransport unavailable, browser falls back to WSS
- diagnostics show actual family selected

### Scenario D: Suspend And Resume

- browser joins swarm
- tab is backgrounded or suspended
- browser resumes
- browser reconnects and resynchronizes without stale “connected” UI

### Scenario E: Artifact Provider Preference

- peer providers available
- browser fetches manifest/chunks from peers first
- edge fallback remains unused
- diagnostics prove peer-first behavior

### Scenario F: Recovery Fallback

- browser transport join fails
- browser enters explicit `degraded_fallback`
- edge bootstrap/recovery still works
- UI and diagnostics do not misreport this as full symmetry

## Performance And Resource Budgets

Browser/native symmetry is also a performance problem, not only an API problem.

The implementation should define budgets for:

- maximum bootstrap latency from sign-in completion to transport connected
- maximum artifact bootstrap latency to first checkpoint availability
- reconnect latency after tab resume
- acceptable edge fallback rate in staging and production

Without budgets, “works” is too vague and the browser path will regress silently.

## Operator Debug Requirements

The debugging story for browser-native symmetry should be explicit.

Operators should be able to answer:

- did the browser receive a signed advertisement?
- which transport family did it attempt?
- which transport family did it actually connect with?
- did it join overlays?
- is it using swarm-fed state or edge fallback?
- did artifact fetch come from peers or the edge?

That means the browser runtime needs structured debug state, not just UI text.

## Migration Rules

The browser path needs explicit migration semantics while both old and new
behavior coexist.

### Client Migration Rules

- browsers without signed seed advertisement support should continue to operate
  in edge-bootstrap mode
- browsers with signed seed advertisement support but no browser transport join
  should expose `degraded_fallback`, not “connected”
- browsers with active swarm join should prefer swarm-fed state and only use
  edge fallback when required

### Deployment Migration Rules

- production must not flip to “browser swarm required” until canaries prove the
  browser transport path works
- bootstrap nodes must not advertise browser transport support unless the
  transport is actually wired and reachable
- Pages config should not hard-depend on new swarm-only fields until the edge
  path is available

## Browser UX State Requirements

The roadmap should also constrain the browser UX, because misleading UI is part
of the current asymmetry.

Required public runtime states:

- `sign in`
- `connecting`
- `joined network`
- `syncing head`
- `ready to train`
- `training`
- `degraded fallback`
- `reconnect required`

The UI should never:

- label a browser “connected” when it only has a cached edge snapshot
- label peer count from a synthetic heuristic
- hide that it is operating in edge fallback mode

## Observability Requirements

The browser/native symmetry project needs better measurements than it has now.

Required browser metrics/events:

- seed advertisement fetch success/failure
- seed dial attempts by transport family
- transport join success/failure
- overlay join success/failure
- head source transitions
- assignment source transitions
- artifact source transitions
- reconnect/suspend-resume events

Required deployment diagnostics:

- signed seed advertisement validity
- browser-capable multiaddr availability
- active browser transport matrix
- real artifact-head route health
- assignment/head presence

## Rollback Criteria

The rollout should stop or roll back if any of these occur in staging or
production:

- browser peers cannot complete signed seed resolution
- browser peers join only through fallback for the majority of sessions
- browser peers lose head sync after direct transport join
- artifact fetch becomes slower or less reliable than the edge-only baseline
- deployment advertises browser-capable transport that cannot be joined

## Phase Plan

### Phase 0: Truthful Diagnostics

Before changing behavior, make the current model observable.

Required:

- add browser diagnostics fields for:
  - configured seed URLs
  - whether a real seed dial occurred
  - whether current state is edge-bootstrap only
  - transport source: `recommended`, `connected`, `fallback`
  - real connected peer count
  - overlay subscription status
  - artifact source: `peer_swarm`, `edge_http`
- stop presenting `transport.active` as proof of a live swarm connection
- stop deriving peer count from `transport.active.is_some()`

Exit criteria:

- signed and unsigned browser UI can distinguish:
  - edge bootstrap only
  - transport selected
  - transport connected
  - overlay joined
- deployment diagnostics can say whether browser swarm join is actually
  happening

### Phase 1: Browser Seed Advertisement And Bootstrap Contract

Required:

- define signed browser seed advertisement schema
- define merge policy with site-config fallback seeds
- expose signed browser seed fetch from the edge
- add transport-policy payload

Exit criteria:

- browser can consume a signed seed advertisement
- diagnostics report the seed source used

### Phase 2: Real Browser Seed Dial

Introduce a browser swarm runtime in `burn_p2p`.

Required:

- add a browser/wasm swarm backend in `burn_p2p_swarm`
- make browser peers dial bootstrap directly after auth/bootstrap
- support at least one real browser-capable transport end to end
- define connection lifecycle states

Design gate:

- decide whether the first transport backend is:
  - Rust libp2p on wasm directly
  - a JS/browser transport adapter behind a Rust trait boundary

Explicit preference:

- prefer Rust/wasm-first implementation
- only use JS where browser transport APIs require a bridge

Exit criteria:

- browser connects to bootstrap over a real browser transport
- transport status is based on real live connection
- seed dial failures are explicit

### Phase 3: Swarm-Based State Propagation

Replace steady-state edge polling for directory/head/metrics with swarm-fed
updates.

Required:

- keep edge snapshot only for bootstrap/recovery
- subscribe browser peers to directory/head/metrics overlays
- reconcile bootstrap snapshot with live swarm state
- add explicit fallback to edge snapshot if swarm join fails

Exit criteria:

- browser can stay current without repeated `/directory`, `/heads`, or metrics
  polling
- head updates from native peers reach the browser through the swarm

### Phase 4: Real Peer Artifact Transport

Turn the existing browser artifact hook into a real browser swarm transport.

Required:

- replace or formalize the ad hoc peer artifact bridge
- serve artifact manifests/chunks over the actual browser swarm transport
- keep edge download as fallback only
- record artifact source in runtime diagnostics

Exit criteria:

- browser artifact fetch prefers peer swarm when providers are available
- chunk fetch metrics distinguish peer swarm success from edge fallback

### Phase 5: Browser-First Deployment Hardening

Make deployment intentionally support browser peers as swarm participants.

Required:

- bootstrap nodes advertise browser-dialable multiaddrs
- production transport matrix is explicit:
  - WebRTC direct
  - WebTransport
  - fallback WSS if retained
- deploy diagnostics fail if browser-capable transports are advertised but not
  actually joinable

Exit criteria:

- deploy validation includes browser-capable transport probes
- production no longer reports transport support that browsers cannot actually
  use

## Testing Matrix

### Unit / Integration

`burn_p2p`

- transport selection only reports connected transport after a real join
- seed dial failure produces explicit diagnostics
- signed seed advertisement merge policy is deterministic
- overlay subscription state transitions are covered
- artifact transport falls back correctly from peer swarm to edge HTTP

### Local End-to-End

- browser + bootstrap only
- browser + one native trainer
- browser suspend/resume and reconnect
- transport downgrade order:
  - WebRTC direct unavailable
  - WebTransport unavailable
  - WSS fallback only
- browser receives assignment/head over swarm, not only edge bootstrap

### Deployment Validation

- bootstrap advertises browser-dialable multiaddrs
- browser can authenticate, dial bootstrap, receive head, and start training
- production diagnostics fail if:
  - no matching head
  - no matching directory entry
  - browser transport advertised but not joinable
  - no signed browser seed advertisement is available
  - head artifact route is broken

## Security Requirements

Browser swarm participation changes trust boundaries. Require:

- authenticated browser identity remains bound to issued session identity
- overlay join authorization remains scope-gated
- browser peers cannot join training overlays without enrollment
- transport-level connection does not bypass trust checks
- artifact peer transport verifies manifest/chunk integrity exactly like edge
  fallback
- signed seed advertisement is authority-scoped, time-bounded, and network-bound

## Operational Requirements

Deployment/diagnostics should report:

- whether browser swarm transport is actually joinable
- whether the signed browser seed advertisement is present and valid
- which transport family is currently enabled in production
- whether artifact peer transport and edge fallback are both healthy

Inspect/deploy tooling should capture:

- signed seed advertisement payload
- browser-capable multiaddrs
- transport flags exposed by the edge
- artifact route health
- assignment/head visibility

## Rollout Strategy

Do not cut directly from edge-mediated browser behavior to full browser swarm
parity.

Use staged rollout:

1. truthful diagnostics only
2. signed seed advertisement
3. real browser seed dial
4. dual-path state propagation
5. peer-first artifact transport
6. deployment/browser canary required for production

Each stage should be deployable without requiring the next one immediately.

## Acceptance Criteria

The project should not claim “browser peers behave like native peers” until all
of these are true:

- browser dials bootstrap directly after auth
- transport label reflects a real live transport connection
- peer count is real
- steady-state head/directory updates come from the swarm, not edge polling
- artifact fetch prefers peer providers over edge download
- deploy diagnostics can prove browser swarm join succeeded

## First Milestone Recommendation

The first useful milestone is not full parity. It is:

- browser authenticates through edge
- browser consumes a signed browser seed advertisement
- browser dials bootstrap directly
- browser receives head and directory assignment over a real browser transport
- browser can train with bootstrap-only topology

That milestone proves the architecture without requiring every advanced swarm
optimization immediately.

## Open Decisions

These decisions still belong in `burn_p2p` before implementation starts:

1. exact first transport backend:
   - Rust libp2p wasm directly
   - or thin JS adapter behind a Rust trait boundary
2. exact shape of the signed browser seed advertisement
3. whether WebTransport support is required for milestone 1 or phase 4
4. whether metrics propagation should be overlay-native immediately or may stay
   partially edge-backed for one milestone

## First PR Slices

If the work is handed to an engineer or agent, split it like this:

1. add signed browser seed advertisement schema to `burn_p2p_core`
2. add truthful browser transport diagnostics to `burn_p2p_browser`
3. add browser swarm runtime trait boundary to `burn_p2p_swarm`
4. implement one real seed dial backend
5. wire browser app to real runtime status
6. add swarm-fed head/directory synchronization
7. add peer-first artifact transport
8. add deployment/browser canary and hard fail production when browser swarm is
   not actually available

## Agent Handoff Checklist

For the engineer or agent implementing this in `burn_p2p`:

1. add a design note describing the chosen browser transport backend
2. define the signed browser seed advertisement format and merge policy
3. introduce a browser swarm runtime abstraction in `burn_p2p_swarm`
4. make `burn_p2p_browser` consume that abstraction instead of synthetic
   transport selection
5. replace UI transport/peer reporting with real swarm state
6. demote edge snapshot sync to bootstrap/fallback
7. integrate artifact peer transport with the same runtime
8. add deployment/browser canary tests
9. update `burn_dragon` to surface only real connection state

## Immediate Follow-Up In `burn_dragon`

Once `burn_p2p` has the first milestone:

- simplify the browser UI to show only:
  - auth state
  - seed connection state
  - sync state
  - training readiness
- remove remaining synthetic transport/peer messaging
- add a deployment check that fails if production claims browser transport
  support but the browser cannot actually join through it
