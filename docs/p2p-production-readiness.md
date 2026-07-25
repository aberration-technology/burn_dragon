# Dragon P2P production readiness

This document is the implementation and release ledger for a decentralized
Dragon training deployment. It covers the dependency stack:

```text
burn_ecs -> burn_p2p -> burn_dragon
burn_eggroll ----------------^
burn_pc ---------------------^
```

`burn_ecs` is the run-scoped orchestration layer. `burn_p2p` owns
architecture-neutral distributed contracts. Dragon owns the model, datasets,
objectives, checkpoint mapping, recurrent state, and optimizer-specific update
replay. `burn_eggroll` and `burn_pc` remain independent numerical crates.

## current status

The codebase has a production-oriented foundation, but it is not yet safe to
describe the public training network as adversarially production-ready.

| area | implemented and locally verified | remaining release gate |
| --- | --- | --- |
| revision identity | hardware-neutral signed contract, strict trust validation, atomic live rollout, and authority rotation tooling | execute the rotation and disaster-recovery drill against staging |
| initial weights | signed full-head genesis, content-addressed artifact verification, and canonical decoded-tensor verification on native and browser loaders | clean-storage staging canary using the production artifact |
| native training | one-shot/continuous windows, a protocol-aware managed trainer daemon, head reconcile, TBPTT state, capability downgrade/upgrade, exact three-peer candidate/root replay, and three-seed DiLoCo convergence parity | run a 24-hour restart and partition soak |
| browser training | ArtifactWindows WebGPU AdamW and forward-only seeded-fitness paths, WASM/WebGPU compile, lease-scoped Firefox execution, clean no-WebGPU downgrade, and fail-closed DiLoCo selection | deployed Chrome/Pages/edge/WebGPU training canary against a signed ArtifactWindows production contract; implement browser DiLoCo before claiming mixed-protocol trainer parity |
| compact updates | bounded seeded-fitness publication, deterministic reconstruction, authenticated lease recovery, and independent sampled fitness replay | untrusted multi-validator quorum and quarantine drill |
| capability policy | conservative preflight, persistent downgrade, revisioned live roles, memory-headroom re-probe, success hysteresis, and bounded backoff | device-loss/recreation drill on each supported GPU backend |
| read-only peers | observer/verifier scopes, projections, and role replacement rather than stale-role union | public UI authorization drill |
| ECS integration | run-scoped lifecycle, capability, window, reconcile, and bounded ingress state | sustained multi-run ingress soak |
| checkpoint scaling | deterministic GDN2 widening and versioned metadata | mixed-width network activation and rollback drill |
| bandwidth | measured canonical payload sizes, topology simulations, networked DiLoCo, persistent outer SGD/momentum state, and FP32/FP16/int8 DiLoCo quality matrix | live WAN and heterogeneous native measurements; browser DiLoCo quality is not implemented |

Unit tests prove contract mechanics and deterministic behavior. The release
matrix provides bounded local convergence evidence; it does not prove
long-horizon convergence, Byzantine robustness, or acceptable production
economics.

## one revision, heterogeneous execution

A Dragon revision is identified by:

- model program and tensor schema
- checkpoint and initialization algorithms
- tokenizer, preprocessing, dataset view, and objective
- optimizer, scheduler, aggregation, and validation semantics
- recurrent-state ownership
- update codec

It is not identified by CUDA versus WGPU, a device name, memory size, local
batch size, or calibrated population size. Those are peer capabilities.

That contract permits:

- a CUDA native trainer with a large local batch
- a WGPU native trainer with a smaller batch
- a WebGPU browser trainer using the same objective
- a CPU or browser verifier that never mutates canonical weights
- an observer that only receives metrics and directory state

Every contribution remains bound to an exact revision, base head, window,
lease, dataset view, and update codec.

## measured 1M convergence

### Artifact-window baseline

The release-only parity harness trains a 926,210-parameter Dragon model with
three native CPU peers. It uses two layers, embedding width 256, latent width
1,024, block size 64, batch size 4, disjoint shard leases, full federated
averaging, and exact candidate replay. Every promoted network tensor is
compared with an independent in-process merge oracle, and every peer evaluates
the same canonical validation stream.

The fair centralized reference consumes the same examples and makes the same
number of optimizer updates as one federated trajectory. It interleaves one
batch from each peer and uses gradient accumulation 3. The sequential reference
also consumes the same examples, but makes three times as many AdamW updates;
it is an upper bound, not a parity baseline.

Seed 1337 produced:

| local steps x rounds | peer updates | P2P final loss | matched central final loss | P2P/central progress | protocol duty |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 x 2 | 18 | 3.551498 | 2.472019 | 66.45% | 45.29% |
| 3 x 6 | 18 | 3.681065 | 2.566830 | 64.32% | 25.05% |

Both conditions start at validation loss 5.689760 and process 216 records. The
nine-step condition is better in both learning progress and protocol duty.
More frequent promotion did not recover the gap.

This result has two distinct conclusions:

- transport, artifact publication, merge replay, and validation are exact;
  candidate and canonical tensors match the oracle every round
- artifact-window AdamW does not have convergence parity with synchronized
  data-parallel AdamW at this horizon

The likely mechanism is independent adaptive optimizer normalization plus
non-IID shard-local trajectories before endpoint averaging. This is not
evidence of transport corruption: the network result and independent endpoint
merge are identical.

The report's default convergence promotion threshold is 90% of matched central
progress. Set `BURN_DRAGON_P2P_PARITY_REQUIRE_CONVERGENCE=1` to make that
threshold a hard test assertion. The current result fails that learning-quality
gate, so the artifact-window AdamW protocol is not the production
continual-pretraining default.

### DiLoCo parity

The protocol-aware DiLoCo path uses persistent outer SGD state, rotating
reducers, content-addressed cohort commitments, a two-stage ready barrier, exact
base-parameter checksums, and transport-decoded pseudo-gradients. The matched
reference applies AdamW to gradients accumulated over the same three peer
microbatches at each local-step index.

The release matrix uses four outer rounds, nine local AdamW steps per peer per
round, batch size 4 per peer, and outer SGD learning rate 1.0 without momentum.
Peer identities are deterministically derived from seed and role so lease
partitioning and batch order remain fixed across conditions.

| seed | genesis | P2P final | matched final | P2P/matched progress |
| ---: | ---: | ---: | ---: | ---: |
| 1337 | 5.710400 | 2.638148 | 2.585947 | 98.33% |
| 1338 | 5.687529 | 2.534715 | 2.357901 | 94.69% |
| 1339 | 5.670796 | 2.562618 | 2.457294 | 96.72% |

Mean progress parity is 96.58%; the minimum seed clears the 90% promotion
threshold. Across all 12 rounds, network parameters and aggregates exactly
match the independent codec-aware protocol oracle, every peer applies the same
parameter pack, all three contributions are committed, and hard DiLoCo request
failures remain zero. The convergence assertion was enabled for every row, so a
below-threshold seed could not emit a passing test result. Mean loopback
network-round time is 8.53 seconds, mean compute duty is 76.4%, and aggregate
peer-local throughput is 3.17 inner steps per network second.

The same-binary seed-1338 codec ablation used release executable SHA-256
`2633d3dd42da98ae9bf89c5cbc3e1a1af960ff8a698ca3b7dba6a4b19a780aa6`:

| codec | local gradient payload | vs FP32 | P2P final | progress parity | network seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 44,458,080 | 100.0% | 2.534715 | 94.690% | 34.70 |
| FP16 | 22,229,040 | 50.0% | 2.534802 | 94.687% | 34.42 |
| blockwise int8/256 | 11,375,088 | 25.6% | 2.535125 | 94.677% | 33.45 |

FP16 and int8 preserve the bounded-run learning trajectory while reducing
payload. They do not improve loopback wall time because model compute and
fixed control work dominate. WAN bandwidth/latency, mixed hardware, and longer
training remain separate required measurements.

The machine-readable reports for the final enforced matrix are under
`target/test-artifacts/p2p-diloco-release-enforced-final-v2/`. Every report uses
schema version 3, records the release build profile and ndarray CPU backend,
contains 17 enforced gate assertions, and was produced by the executable above.
The current-stack seed-1337 rerun independently reproduced `2.638148` P2P
loss, `2.585947` synchronized loss, `98.329%` progress parity, all 17 passing
gates, 108 peer-local inner updates, 3.030 aggregate inner steps per network
second, and 59,277,440 estimated wire payload bytes excluding control
overhead.

### Local transport and recovery gate

The convergence matrix runs above the complete local protocol suites:

- `burn_p2p_swarm`: 84 serial tests, including rendezvous registration,
  Kademlia recovery, relay reservation, 192 KiB relayed transfer, direct-route
  handoff, and post-transfer single-route reconciliation
- `burn_p2p`: 266 serial tests, including DiLoCo cohort/reducer behavior,
  runtime restart, security-state restoration, diffusion, and control-plane
  projection
- the four-peer rotating-reducer authority scenario passed eight consecutive
  process-level repetitions after startup dial failures were made immediately
  retryable

Observed addresses are now admitted only from inbound identify connections, so
relay servers can advertise reachable listeners without publishing outbound
ephemeral ports. Relay-to-direct reconciliation waits for in-flight control
requests and responses before pruning the old route. Initial connection
refusal releases the dial debounce immediately instead of suppressing
connectivity repair for 30 seconds. Security state is persisted for
security-relevant events and flushed at shutdown, while public control-plane
projection still refreshes for every pubsub message.

## trusted startup sequence

A production revision starts in this order:

1. build the Dragon config and materialize the model schema
2. produce one complete genesis checkpoint
3. compute the canonical tensor digest
4. build `TrainingContractManifest`
5. bind revision, training contract, and genesis in
   `RevisionContractBundle`
6. sign genesis and the complete bundle with the authority Ed25519 key
7. provision the genesis artifact in the artifact store
8. register the authority public key in the active trust bundle
9. provision the bundle file through
   `BURN_P2P_REVISION_CONTRACT_FILES`
10. start bootstrap and require contract validation
11. start native and browser peers in fail-closed contract mode
12. confirm every peer resolves the same contract, genesis payload, tensor
    digest, and canonical head before issuing leases

No peer should initialize a private random "genesis" on a public revision. The
local deterministic fallback exists only for isolated development.

## capability lifecycle

Participation is selected from measured and configured capability, not only
compile-time Cargo features:

- `observer`: directory, metrics, and head visibility only
- `validator`: read-only model/update verification
- `trainer`: lease execution and candidate publication
- `aggregator`: reduction work where topology enables it
- `archive`: artifact persistence and serving

The default lifecycle begins read-only. A successful capability assessment
emits a revisioned upgrade. OOM, failed allocation, or device loss emits a
revisioned downgrade and persists it against the workload fingerprint.
Increasing the memory budget or changing the workload invalidates the stale
downgrade.

Native runtime fit failures replace the live trainer advertisement with a
read-only viewer role. An in-flight window may finish, but new training windows
fail closed. The monitor then waits for the configured cooldown, checks the
budget and live `MemAvailable` headroom, requires consecutive successful
probes, acknowledges the handled runtime error, restores only the original
trainer role class, and clears the workload-specific downgrade record.
Failures use bounded exponential backoff. A stale probe cannot overwrite a
newer run-scoped capability revision.

This is role and resource-fit recovery, not portable GPU hot-plug. Backend
device loss/recreation still needs a staging drill for CUDA, ROCm, native WGPU,
and browser WebGPU before claiming transparent hardware migration.

## native and browser coherence

Native and browser execution share:

- `WorkloadTrainingPlan`, lease, contribution, and result schemas
- the signed Dragon training contract
- stream batch planning and logical stream identity
- TBPTT recurrent-state scoping
- objective and optimizer hashes
- compact seeded-fitness generation ordering
- artifact/update envelopes and receipt semantics

The browser AdamW path uses WebGPU autodiff. The browser EGGROLL path uses a
plain forward backend and does not retain reverse-mode graphs. Native
validators reconstruct browser compact updates from the canonical base model.

Browser CPU remains a development smoke path, not a production trainer.

Protocol compatibility is explicit rather than inferred from backend:

- `run-peer` joins and serves network state but performs no training.
- the managed `run-trainer-daemon` checks the live trainer role and dispatches
  every iteration through `train_protocol_once`.
- ArtifactWindows training waits for canonical advancement before opening the
  next local window; DiLoCo advances through its round cursor and persistent
  outer state.
- browser training configuration is exposed only for ArtifactWindows. Both
  profile resolution and the live browser runtime reject DiLoCo training.

This means native DiLoCo and browser ArtifactWindows are separately coherent
execution paths, not yet one mixed trainer cohort. A signed protocol revision
must choose one. Browsers can remain read-only metric/verifier peers on a
DiLoCo revision, but browser DiLoCo training is a remaining implementation and
quality gate.

## recurrent state contract

Dragon TBPTT state is run- and stream-local. Its identity includes revision,
base head, lease, and logical stream. It is reset when the canonical model
becomes incompatible and never lives in process-global mutable storage.

This matters for both correctness and multi-run hosting. Two Dragon pipelines
in one process must not share rho state, optimizer state, scheduler cursor,
checkpoint files, or event sinks merely because they use the same model type.

The current artifact-window parity profile declares both optimizer and
scheduler state as `reset_per_window`. This is implemented behavior, not merely
metadata. No optimizer moments are silently transferred between peers or
attached to the canonical head. It also means the current protocol cannot
claim optimizer-state equivalence with centralized AdamW. A stateful protocol
must choose and implement either peer-local state invalidated on reconcile,
canonical optimizer artifacts, or an explicit outer optimizer such as
DiLoCo/FedOpt; changing that choice creates a new signed training contract.

## shard partition and capacity

Dragon exports contiguous stream segments as indivisible groups so TBPTT rows
cannot be split across peers inside a segment. Groups are packed largest-first
into the least-loaded microshard, which avoids key-order skew while preserving
recurrent continuity. Window selection rotates over bounded micro-epochs
instead of replaying the first shard prefix. Dataset materialization caps the
physical shard count to the number of records, or to the number of indivisible
groups for grouped data, so the lease planner cannot assign a valid-but-empty
training shard.

The number of independently leasable microshards must cover the simultaneous
trainer cohort. When trainers outnumber available partitions, assignment fails
closed; peers do not steal another trainer's shard or silently read the full
dataset. Production scheduling should expose that state as idle/no-work
telemetry and provision enough partitions for the advertised concurrency.

## compact update path

The current browser EGGROLL path transmits:

- perturbation population, rank, seed, and generation
- parameter catalog and generator hashes
- optimizer-update semantics hash
- exact batch digest per generation
- compact antithetic fitness values

It does not transmit one dense value per model parameter. The native workload
regenerates perturbations and replays the optimizer update against the exact
canonical base model.

This is the forward-only analogue of low-dimensional affine update transport.
The generic `SubspaceLatent` codec also supports FLITE-style shared subspace
coordinates without introducing Dragon assumptions into `burn_p2p`.

Independent replay is now part of candidate admission. Validators require the
authenticated, unexpired assignment lease; recover only content-verified
leased microshards; reproduce the exact batch digest; regenerate
contract-selected plus/minus perturbation pairs; recompute fitness; and compare
under absolute/relative tolerances in the signed codec policy. Missing,
conflicting, wrong-owner, unleased, wrong-batch, and forged-fitness inputs fail
closed. Replay-required codecs cannot reach promotion with
`replay_verified=false`.

This closes the previous single-validator trust shortcut. It does not prove
Byzantine robustness by itself. Public anonymous participation still requires
separate validators, quorum greater than one, disagreement telemetry,
quarantine/rate policy, and a live adversarial drill.

## ECS and observability

`burn_p2p` exposes one non-blocking `TrainingWindowObserver` path for one-shot
and continuous native windows. Dragon binds it to
`P2pTrainingEcsObserver`, which feeds a bounded `burn_ecs` ingress.

One run entity owns:

- lifecycle and capability revisions
- experiment, revision, window, and head telemetry
- window child entities and P2P metadata
- event files and dashboard state
- model, optimizer, scheduler, device, data, and recurrent-state slots where a
  host chooses ECS ownership

Global resources are limited to true process concerns such as cancellation and
the bounded ingress handle. This preserves multiple independent pipelines in
one ECS world.

## release validation ladder

Every deployment candidate should pass these gates in order:

1. static
   - format, strict clippy, unit tests, schema snapshots, WASM compile
2. contract
   - signature mutation, untrusted signer, conflicting revision, missing
     genesis, wrong schema, stale base head, malformed compact payload
3. local E2E
   - three native trainers, bootstrap-only topology, diffusion/promotion
   - observer and verifier scope isolation
   - one-shot and continuous ECS event delivery
4. browser
   - Chrome/WebGPU execution from the built Pages artifact
   - signed contract and exact genesis received from the edge
   - trainer, verifier, and observer role-specific auth
5. recovery
   - trainer crash mid-window
   - bootstrap restart from durable state
   - late peer genesis/head sync
   - network partition, stale contribution, and canonical reconcile
6. adversarial
   - tampered signature and artifact
   - forged fitness, duplicate generation, batch digest mismatch
   - unauthorized train/archive receipt
   - validator disagreement and quorum loss
7. scale
   - bounded memory on native and browser peers
   - sustained ingress and artifact retention
   - measured bytes per accepted token and time to canonical promotion
8. learning
   - fixed-baseline convergence and verifier matrix
   - heterogeneous versus homogeneous fleet comparison
   - full update versus compact codec quality/throughput/bandwidth ablation

The first six are release correctness gates. The last two are promotion gates
for performance and model quality.

## local evidence snapshot

The following bounded checks have passed on the current ARM development host.
Unless explicitly identified as a learning matrix above, they are
wiring/correctness evidence rather than convergence claims:

- complete `burn_dragon_p2p` library suite: 72 tests
- all 18 non-ignored native integration tests, including shard-backed NCA,
  live-source Ruliad, mixed native/browser roles, persistence, and diffusion
- strict native and WASM package Clippy for the P2P surfaces
- real headless Chromium WebGPU execution of generated NCA browser training
  from the WASM package
- production Pages shell build with the WebGPU WASM bundle and auth callback
  routes
- real Firefox WASM execution for lease-scoped shard selection
- real Firefox no-WebGPU downgrade to a read-only role
- signed three-peer restart recovery in 3.008 seconds with persistent peer
  identity and exact head/tensor/validation recovery
- strict signature, conflicting-contract, exact-artifact, decoded-tensor,
  lease, wrong-owner, batch-digest, unleased-record, and forged-fitness checks
- partition, stale-head, slow-peer, churn, relay-loss, and rejoin recovery
  matrix in `burn_p2p_testkit`
- non-ignored NCA and ClimbMix mixed-fleet smokes plus the explicit ignored
  medium rung
- deployment workflow contract checks and WebGPU WASM target check
- strict 1M-parameter three-peer DiLoCo bulk exchange
- three-seed, four-round release DiLoCo convergence parity
- same-binary FP32/FP16/blockwise-int8 codec quality and payload ablation

The current Firefox runner did not expose WebGPU. Its successful downgrade is
valid role/capability evidence, not a browser GPU training result. A deployed
Chrome/WebGPU canary remains mandatory.

The canonical 100M-parameter bandwidth ablation measured:

| update | payload bytes | reduction vs dense fp32 |
| --- | ---: | ---: |
| dense fp32 | 400,000,000 | 1x |
| subspace, 1,280 fp32 coefficients | 5,394 | 74,157x |
| subspace, 1,280 int8 coefficients | 1,566 | 255,428x |
| subspace, 4,096 int8 coefficients | 4,382 | 91,283x |
| seeded fitness, population 256 int8 | 3,026 | 132,188x |
| seeded fitness, population 1,024 int8 | 3,794 | 105,430x |
| seeded fitness, population 4,096 int8 | 6,866 | 58,258x |

For 64 peers, the topology simulation reduced the 400 MB dense global
all-to-all estimate from 1.4884 TB/window to 5.83 MB for 1,280-dimensional
int8 subspace updates or 25.55 MB for population-4,096 seeded fitness. The
central-reducer and replicated-DAG totals for the 1,280-dimensional int8
condition were 95,526 and 200,448 bytes, respectively. These are canonical
wire-size and topology measurements. They do not establish equal learning
quality. The current `SubspaceLatent` implementation is a seeded
CountSketch-style affine map; it is compatible with the communication
principle of lightweight subspace fine-tuning, but it is not a learned
low-rank reconstructor.

## required production dashboards

At minimum, expose per revision and peer class:

- connected, observed, trainer, verifier, and archive peer counts
- capability upgrade/downgrade reason and revision
- canonical head, speculative head, and lag
- windows started/completed/failed and reconcile count
- lease, data-fetch, compute, publication, validation, and promotion latency
- artifact and update bytes
- decoded norm, non-finite, clipping, and replay-verification status
- tokens/samples processed and accepted
- browser WebGPU availability and device-loss rate
- auth denial, replay rejection, quarantine, and validator disagreement
- Dragon loss, verifier accuracy, output degeneracy, and ruliad difficulty

Read-only metric peers should consume these projections without receiving
`Train`, `Validate`, `Archive`, or `Admin` authority unless explicitly granted.

## promotion decision

The current implementation is appropriate for local, trusted-fleet, and staged
deployment experiments. Public arbitrary-peer training remains blocked on:

- a clean signed-genesis native/browser staging deployment
- separate untrusted validators, quorum greater than one, and quarantine drill
- 24-hour restart, partition, state-recovery, and bounded-ingress soak
- GPU device-loss/recreation drills across supported backends
- heterogeneous native/browser compact-versus-dense learning-quality and
  live-WAN evidence
- browser DiLoCo execution and convergence evidence before enabling browser
  trainers on the native DiLoCo revision

Those are concrete release gates, not optional follow-up polish.
