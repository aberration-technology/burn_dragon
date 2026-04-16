# Browser Libp2p Roadmap

This document is the implementation handoff for moving browser peers from the
current edge-mediated runtime to a real browser-capable libp2p participation
model.

It is written from the perspective of `burn_dragon`, but most of the required
work belongs in `burn_p2p`.

## Goal

After browser auth/bootstrap completes, a browser peer should:

- dial real seed/bootstrap multiaddrs directly
- establish a real browser-capable peer transport
- participate in steady-state directory/head/metrics/artifact sync through the
  swarm instead of edge polling
- expose real peer/transport status in the UI and diagnostics

The target is not strict equivalence with native nodes. Browsers cannot use the
same transport surface as native peers. The target is first-class swarm
participation over browser-capable transports.

## Current State

Today the browser runtime is not a first-class libp2p swarm peer.

Evidence:

- `burn_p2p_swarm` keeps the real libp2p swarm builder and transport stack
  behind native-only cfg gates in
  [lib.rs](/home/mosure/repos/burn_p2p/crates/burn_p2p_swarm/src/lib.rs#L14).
- the browser control-plane client is HTTP-backed in
  [browser_edge.rs](/home/mosure/repos/burn_p2p/crates/burn_p2p_swarm/src/browser_edge.rs#L24).
- the Dragon browser runtime starts from the edge snapshot and then syncs
  runtime state through the edge client in
  [training.rs](/home/mosure/repos/burn_dragon/crates/burn_dragon_p2p/src/wasm/training.rs#L820).
- browser transport selection is currently a recommendation, not a live dialed
  transport, in
  [transport.rs](/home/mosure/repos/burn_p2p/crates/burn_p2p_browser/src/transport.rs#L115)
  and
  [worker.rs](/home/mosure/repos/burn_p2p/crates/burn_p2p_browser/src/worker.rs#L792).
- the browser app currently derives "direct peers" from `transport.active` in
  [app.rs](/home/mosure/repos/burn_p2p/crates/burn_p2p_browser/src/app.rs#L259),
  which is not a real peer count.
- `seed_node_urls` exist in browser config in
  [site_config.rs](/home/mosure/repos/burn_p2p/crates/burn_p2p_browser/src/site_config.rs#L6)
  and Dragon config in
  [config.rs](/home/mosure/repos/burn_dragon/crates/burn_dragon_p2p/src/config.rs#L154),
  but the current browser connect path does not dial them through a browser
  swarm runtime.

There is one useful precursor already:

- browser-side artifact fetch can optionally use a peer-swarm JS bridge in
  [auth.rs](/home/mosure/repos/burn_p2p/crates/burn_p2p_browser/src/auth.rs#L1486)
  and
  [auth.rs](/home/mosure/repos/burn_p2p/crates/burn_p2p_browser/src/auth.rs#L2045)

That is not a full solution. It is only a narrow artifact-fetch hook.

## Live Production Constraint

The current production edge at `dragon.aberration.technology` advertises:

- `webrtc_direct = true`
- `webtransport_gateway = false`
- `wss_fallback = true`

So even the deployment surface is not yet configured for the desired
WebTransport-capable browser path.

## Target Architecture

The desired split is:

1. edge HTTP for auth/bootstrap/fallback only
2. browser swarm runtime for steady-state peer participation

Edge responsibilities:

- login start / callback / enroll
- trust bundle and policy distribution
- initial signed directory/head bootstrap material
- diagnostics and recovery fallback

Browser swarm responsibilities:

- direct seed dialing from `seed_node_urls`
- overlay/topic subscription
- steady-state directory/head/metrics propagation
- peer artifact fetch and chunk fetch
- actual peer presence and transport status

## Bootstrap Source Of Truth

The roadmap should not rely only on statically baked `seed_node_urls`.

That is too brittle for the real deployment model because:

- Pages artifacts can outlive infra changes
- bootstrap transport addresses can change across deploys
- the browser still needs an authenticated bootstrap path anyway

Preferred model:

1. browser authenticates through the edge
2. browser fetches a signed browser-dialable seed set from the edge bootstrap
   surface
3. browser reconciles that signed seed set with any statically baked
   `seed_node_urls`
4. browser dials the signed set first
5. static seed URLs remain fallback only

Required output in `burn_p2p`:

- one signed seed advertisement payload for browser peers
- one deterministic merge policy between:
  - edge-published browser-dialable multiaddrs
  - site-config seed URLs
- one diagnostics field reporting which source was used:
  - `edge_signed`
  - `site_config_fallback`
  - `merged`

## Non-Goals

- removing the edge completely
- making browsers use native-only transports like TCP or QUIC directly
- making `burn_dragon` own browser transport internals that belong in
  `burn_p2p`

## Repository Ownership

### `burn_p2p`

Primary ownership:

- browser swarm transport/runtime
- browser-capable transport policy
- browser control-plane/swarm sync model
- browser artifact peer transport
- browser transport diagnostics
- browser acceptance tests

Likely crates:

- `burn_p2p_swarm`
- `burn_p2p_browser`
- possibly `burn_p2p_core` for diagnostics/state contracts

### `burn_dragon`

Consumer ownership:

- passes edge URL and seed URLs
- renders real connection state
- deploys bootstrap nodes with the correct browser-capable listeners enabled
- adds product-facing diagnostics for browser users

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

Acceptance:

- unsigned and signed browser UI both distinguish:
  - edge bootstrap only
  - transport selected
  - transport connected
- deployment diagnostics can say whether browser swarm join is actually
  happening

### Phase 1: Real Browser Seed Dial

Introduce a browser swarm runtime in `burn_p2p`.

Required:

- add a browser/wasm swarm backend in `burn_p2p_swarm`
- make browser peers dial `seed_node_urls` directly after auth/bootstrap
- support at least one real browser-capable transport end to end
- define connection lifecycle states:
  - `bootstrap_sync`
  - `dialing_seed`
  - `transport_connected`
  - `overlay_joined`
  - `assignment_ready`

Design constraint:

- if the current Rust libp2p line cannot cleanly support the needed browser
  transports, use a thin runtime adapter boundary instead of burying JS glue in
  `burn_dragon`

Required decision gate before implementation:

- decide whether the first real browser transport backend is:
  - Rust libp2p on wasm directly
  - a JS/browser transport adapter behind a Rust trait boundary

Do not start wiring browser swarm behavior into `burn_dragon` before that
decision is made in `burn_p2p`.

Acceptance:

- browser connects to the bootstrap node using a real browser-capable peer
  transport
- transport status is based on a live connection, not recommendation
- seed dial failure is explicitly visible in diagnostics

### Phase 2: Swarm-Based State Propagation

Replace steady-state edge polling for directory/head/metrics with swarm-fed
updates.

Required:

- keep edge HTTP snapshot only for bootstrap/recovery
- subscribe browser peers to directory/head/metrics overlays
- reconcile bootstrap snapshot with live swarm state
- add recovery path back to edge snapshot if swarm join fails

Acceptance:

- browser can stay current without repeatedly polling `/directory`, `/heads`,
  and `/metrics/live/latest`
- head updates visible from native peers propagate to the browser over the
  swarm

### Phase 3: Real Peer Artifact Transport

Turn the existing browser artifact hook into a real browser swarm transport.

Required:

- replace or formalize the ad hoc `__burnP2PArtifactSwarm` bridge in
  [auth.rs](/home/mosure/repos/burn_p2p/crates/burn_p2p_browser/src/auth.rs#L2045)
- serve artifact manifests/chunks over the actual browser swarm transport
- keep edge download as fallback only
- record artifact source in runtime diagnostics

Acceptance:

- browser artifact fetch prefers peer swarm when providers are available
- chunk fetch metrics distinguish peer swarm success from edge fallback

### Phase 4: Browser-First Deployment Hardening

Make deployment intentionally support browser peers as swarm participants.

Required:

- bootstrap nodes advertise browser-dialable seed multiaddrs
- production transport matrix is explicit:
  - WebRTC direct
  - WebTransport
  - fallback WSS if retained
- deploy diagnostics fail if browser-capable transports are advertised but not
  actually joinable

Acceptance:

- deploy validation includes a browser-capable transport probe
- the runtime no longer reports browser transport support that production cannot
  actually use

## Proposed `burn_p2p` Integration Contract

The first engineer should not have to invent the runtime boundary from scratch.

Recommended library contract:

- `BrowserSwarmBootstrap`
  - authenticated browser identity/session handle
  - trust bundle
  - selected experiment/revision
  - browser-dialable seed addresses
  - transport policy
- `BrowserSwarmStatus`
  - bootstrap source
  - desired transport
  - connected transport
  - connected peer ids
  - overlay join state
  - head sync state
  - artifact transport mode
  - last dial/sync error
- `BrowserSwarmRuntime`
  - `connect(bootstrap)`
  - `disconnect()`
  - `status()`
  - `subscribe_directory()`
  - `subscribe_heads()`
  - `subscribe_metrics()`
  - `fetch_artifact_manifest()`
  - `fetch_artifact_chunk()`

`burn_p2p_browser` should consume this contract and stop synthesizing
connection state from HTTP snapshot state.

## Transport Strategy

Recommended order:

1. WebRTC direct
2. WebTransport
3. WSS fallback

Rationale:

- WebRTC direct is the closest browser-native peer path
- WebTransport is useful where direct browser-peer semantics are harder, but it
  still gives a browser-capable transport
- WSS fallback can preserve reachability, but it should be treated as degraded,
  not as the ideal browser swarm path

## Security Requirements

Browser swarm participation changes trust boundaries. Require:

- authenticated browser identity remains bound to issued cert/session identity
- overlay join authorization remains scope-gated
- browser peers cannot join training overlays without enrollment
- transport-level connection does not bypass session/trust checks
- artifact peer transport verifies manifest/chunk integrity the same way edge
  fallback does

## Testing Matrix

### Unit / Integration

`burn_p2p`

- browser transport selection only reports connected transport after a real join
- seed dial failure produces explicit diagnostics
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

### Deployment Validation

- bootstrap advertises browser-dialable multiaddrs
- browser can authenticate, dial bootstrap, receive head, and start training
- production diagnostics fail if:
  - no matching head
  - no matching directory entry
  - browser transport advertised but not joinable
  - no signed browser seed advertisement is available

## Acceptance Criteria

The project should not claim "browser peers behave like native peers" until all
of these are true:

- browser dials `seed_node_urls` directly after auth
- transport label reflects a real live transport connection
- peer count is real
- steady-state head/directory updates come from the swarm, not edge polling
- artifact fetch prefers peer providers over edge download
- deploy diagnostics can prove browser swarm join succeeded

## First Milestone Recommendation

The first useful milestone is not "full parity." It is:

- browser authenticates through edge
- browser dials bootstrap directly
- browser receives head and directory assignment over a real browser-capable
  transport
- browser can train with bootstrap-only topology

That milestone is enough to prove the architecture.

## Agent Handoff Checklist

For the agent or engineer implementing this in `burn_p2p`:

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

- simplify the browser UI to show:
  - auth state
  - seed connection state
  - sync state
  - training readiness
- remove any remaining synthetic transport/peer messaging
- add a deployment check that fails if the production edge claims browser
  transport support but the browser cannot join through it
