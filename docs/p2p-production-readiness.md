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
| native training | one-shot/continuous windows, a protocol-aware managed trainer daemon, head reconcile, TBPTT state, capability downgrade/upgrade, exact three-peer candidate/root replay, corrected formal-Ruliad convergence parity, and three-seed DiLoCo parity | run a 24-hour restart and partition soak |
| browser training | ArtifactWindows WebGPU AdamW and forward-only seeded-fitness paths, real Chrome/WebGPU generated NCA and formal-Ruliad windows, clean no-WebGPU downgrade, and fail-closed DiLoCo selection | deployed Pages/edge/WebGPU training canary against a signed production contract; implement browser DiLoCo before claiming mixed-protocol trainer parity |
| compact updates | bounded seeded-fitness publication, deterministic reconstruction, authenticated lease recovery, independent sampled fitness replay, and a generation-bound context-sparse codec | untrusted multi-validator quorum and quarantine drill; activate routed-PC only after its local Ruliad quality gate |
| capability policy | conservative preflight, persistent downgrade, revisioned live roles, memory-headroom re-probe, success hysteresis, and bounded backoff | device-loss/recreation drill on each supported GPU backend |
| read-only peers | observer/verifier scopes, projections, role replacement rather than stale-role union, retrospective diffusion evaluation, pre-promotion validator-quorum evaluation bound to the exact head/artifact/protocol/report, a real two-validator Ruliad quorum, a CUDA-trainer/CPU-validator promotion gate, and a concurrent 10M CUDA/CPU-peer duty stress | network-coupled large-model/WAN evaluator soak, untrusted disagreement/quarantine drill, and public UI authorization drill |
| ECS integration | run-scoped lifecycle, capability, window, reconcile, and bounded ingress state | sustained multi-run ingress soak |
| checkpoint scaling | deterministic GDN2 widening and versioned metadata | mixed-width network activation and rollback drill |
| bandwidth | measured canonical payload sizes, topology simulations, networked DiLoCo, persistent outer SGD/momentum state, and FP32/FP16/int8 DiLoCo quality matrix | live WAN and heterogeneous native measurements; browser DiLoCo quality is not implemented |

Unit tests prove contract mechanics and deterministic behavior. The release
matrix provides bounded local convergence evidence; it does not prove
long-horizon convergence, Byzantine robustness, or acceptable production
economics.

The context-sparse codec is currently a protocol capability, not an advertised
distributed learning mode. It signs the dynamic context family, slot,
generation, and parameter catalog; native and browser builds share the same
decoder and stale-generation rejection. Dragon's local routed fixed-prediction
pipeline now checkpoints its context bank, optimizer moments, recurrent state,
curriculum, and ECS gate state exactly. A three-seed 888,194-parameter CUDA
control reaches convergence/retention parity with matched routed AdamW while
retaining 78.3% throughput. A longer fixed-holdout Ruliad matrix, restart soak,
and measured wire-volume comparison are still required before a revision may
select this codec for decentralized training.

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

Read-only evaluation uses the same binding. The generic `burn_p2p` node API
materializes a requested head, checks network/study/experiment and revision
scope, evaluates the requested split, persists a content-addressed
`HeadEvalReport` and `EvalProtocolManifest`, and announces the resulting metric
cursor. In diffusion mode Dragon's validator daemon follows each newly
promoted head once. In validator-quorum mode validators instead evaluate the
merged candidate before promotion and attest to the exact head ID, artifact
ID, protocol ID, and report ID. Trainer and reducer roles do not run this
formal generation path.

The backend-neutral contract is exercised end to end, not only by manifest
comparison. An operator-invoked native gate starts a CUDA trainer and a CPU-only
validator from distinct signed release artifacts, transfers a real Ruliad
candidate through the swarm, evaluates it on the CPU peer, and promotes it with
exact decoded-tensor equality. The candidate was 12,918,148 bytes and the
content-addressed four-sample formal report was 2,595 JSON bytes. The report is
loaded by its complete head/artifact/protocol/report binding and its identity is
recomputed before the test passes. The CUDA window took 64.01 s in this tiny
model gate because it includes first-use kernel compilation; it is a portability
and protocol result, not a throughput claim. Network-coupled large-model duty
and live-WAN latency remain separate release gates.

A separate warm co-location stress ran the 10M-class release CUDA trainer while
the complete CPU peer/validator quorum gate executed on the same unified-memory
host. This is more contentious than the intended remote-validator placement
because it also includes a CPU trainer. CUDA throughput changed from 28,391 to
27,572 tokens/s (-2.89%), while sampled utilization remained continuously at
95-96%, model duty was 95.38%, foreground loader wait was 0.0013%, and local
validation remained zero. No sample fell below 80% utilization. This closes the
local ECS/P2P scheduling-stall hypothesis; the small throughput cost is shared
host-memory contention and does not substitute for a network-coupled WAN soak.

Dragon's formal evaluator also passes a real quorum-two gate. One trainer
published a 6,448,259-byte candidate to two read-only validators. The first
validator emitted a reduction but could not promote. After the second
independent evaluation, both peers converged on exactly one quorum certificate
and one promoted head. The two 2,542-byte reports have distinct content IDs but
bind the same head, artifact, and protocol; both cover four samples and agree on
verifier accuracy, partial credit, answer-field accuracy, and completion
quality. The promoted decoded-tensor digest equals the trainer candidate. This
closes trusted local quorum mechanics, not malicious-validator quarantine.

## measured 1M convergence

### Corrected formal-Ruliad artifact window

The current signed release gate uses a 926,210-parameter Dragon, three native
peers, two rounds, nine local steps per peer per round, and 54 aggregate
peer-local steps. Ruliad fixed-document padding is target-masked and each peer
receives a disjoint, nonempty, bounded stream-segment lease.

| genesis | P2P final | synchronized final | P2P/synchronized progress | restart recovery |
| ---: | ---: | ---: | ---: | ---: |
| 5.718210 | 2.258732 | 1.930078 | 91.324% | 3.176 s |

The synchronized reference consumes the same examples with the same 18
optimizer updates and gradient accumulation three. P2P clears the hard 90%
progress threshold. Every candidate tensor, promoted tensor, validation value,
and merge result matches the independent oracle exactly. The signed revision,
shared data/objective, disjoint lease, three-receipt merge, restart, and
convergence gates all pass.

The restart drill removes one trainer after round one, keeps it offline for two
seconds, then recovers the same peer identity, canonical head, tensor digest,
and validation loss in 3.176 seconds. The schema-6 report is
`target/experiments/p2p-ruliad-parity/release-r2-s9-signed-restart/seed-1337.json`.

This is a bounded local result. It does not establish long-duration stability,
WAN behavior, heterogeneous-device parity, or adversarial robustness.

### Historical artifact-window baseline

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

These figures predate target-loss-mask propagation for fixed-size Ruliad
documents. Repeated EOS fill was supervised, so they are retained only as
historical protocol evidence and must not be used as current quality evidence.
The artifact-window matrix needs a corrected-objective rerun.

The historical result had two distinct conclusions:

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
threshold a hard test assertion. Artifact-window AdamW is not the production
continual-pretraining default.

### DiLoCo parity

The protocol-aware DiLoCo path uses persistent peer-local AdamW and scheduler
records, outer SGD state, rotating reducers, content-addressed cohort
commitments, a two-stage ready barrier, exact base-parameter checksums, and
transport-decoded pseudo-gradients. The matched reference accumulates the same
three peer microbatch gradients into each AdamW update.

The corrected matrix uses a 1,012,229-parameter random-scaffold model, six
outer rounds, one local AdamW step per peer per round, batch size 4 per peer,
and outer SGD learning rate 1.20 without momentum. Only 225,797 mutable values
(22.31%) are synchronized. Fixed-document EOS fill is masked on native and
browser paths. Each promoted run consumed 18 unique nonempty-supervision
batches with no duplicates.

| seed | genesis | P2P final | matched final | endpoint progress | trailing progress |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1337 | 5.677486 | 3.689056 | 3.568502 | 94.284% | 94.098% |
| 1338 | 5.662974 | 3.655515 | 3.473025 | 91.667% | 91.635% |
| 1339 | 5.676928 | 3.539646 | 3.385779 | 93.284% | 92.890% |

Mean endpoint parity is 93.08%; mean trailing parity is 92.87%; and the minimum
seed clears the 90% promotion threshold. Across all 18 rounds, network
parameters and aggregates exactly match the independent codec-aware protocol
oracle, every peer applies the same parameter pack, all three contributions
are committed, validation is monotonic, and hard DiLoCo request failures
remain zero.

All three release-profile confirmations explicitly omitted the outer
learning-rate environment override. Mean aggregate throughput was 1.031 peer
inner steps per network second, and each condition completed in 71.19-72.64
seconds. Native peers permit two established routes during simultaneous dial
reconciliation; this removed the prior timeout-scale request stalls.

The matched three-seed codec matrix is:

| codec | estimated wire payload/run | vs FP32 | mean P2P final | mean trailing progress | mean peer steps/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 21,676,512 | 100.0% | 3.628072 | 92.874% | 1.0312 |
| FP16 | 10,838,256 | 50.0% | 3.628078 | 92.874% | 1.0138 |

FP16 preserves the bounded-run trajectory to a `+0.0000061` mean final-CE
difference while halving payload. Loopback throughput is 1.69% lower because
codec overhead is exposed when bandwidth is free, so FP16 is a bandwidth
option rather than a local speed optimization. It remains opt-in pending a
constrained real-network and browser measurement. WAN bandwidth/latency, mixed
hardware, and longer training remain separate required measurements.

A newer single-seed source-matched FP16 comparison separates the transport
benefit of random scaffolds from absolute model quality:

| model | total params | synchronized values | final P2P | synchronized final | trailing parity | wire bytes | wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 926,210 | 926,210 | 3.260373 | 3.066923 | 92.767% | 44,458,080 | 24.657 s |
| random scaffold rank 16 | 1,012,229 | 225,797 | 3.613970 | 3.495448 | 94.374% | 10,838,256 | 17.254 s |

The scaffold reduces wire bytes by 75.62% and network wall time by 30.0% at
fixed rounds. It has slightly better synchronized-progress parity but worse
absolute loss. Compactness is therefore promoted as a bandwidth mechanism,
not as an accuracy win.

Machine-readable reports are under
`target/test-artifacts/random-scaffold-diloco/masked-route2/` and
`target/test-artifacts/random-scaffold-diloco/release-contract-final/`. The
final release reports use schema version 5 and record the corrected objective,
supervision fingerprints, dual-route native transport, release build profile,
gates asserting the signed persistent optimizer/scheduler policies, and hard
convergence assertions.

### Local transport and recovery gate

The convergence matrix runs above the complete local protocol suites:

- `burn_p2p_swarm`: serial tests including rendezvous registration,
  Kademlia recovery, relay reservation, 192 KiB relayed transfer, direct-route
  handoff, and bounded connection reconciliation
- `burn_p2p`: tests including DiLoCo cohort/reducer behavior,
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

The shipped native experiment manifest now grants `Validator` and `Evaluator`
roles the `Validate` scope. In diffusion mode the validator daemon follows
`LatestPromoted`, evaluates each exact head once, and retries failed
evaluations without advancing its local evaluated-head cursor. This closes the
former condition where a validator could join and serve artifacts but never
produce model-quality reports.

Validator-quorum mode is a separate promotion path. Dragon selects
`microcohort_reduce_plus_validator_promotion`, advertises the configured
quorum, and leaves formal generation off the trainer. Each validator
materializes the candidate from a synchronized canonical base, runs the fixed
formal panel, and emits a reduction certificate bound to the exact head,
artifact, protocol, and report. Quorum observation verifies the visible
backing reductions rather than trusting a coordinator-supplied ID list; report
IDs must be distinct and authorized attesters must satisfy the configured
quorum. Missing or mismatched evidence fails closed.

The local Dragon gate trains one real Ruliad window, transfers the half-record
artifact to a non-training validator, promotes it with quorum one, verifies the
canonical decoded-tensor digest exactly, and resolves the attested report by
its content ID. A generic two-validator native gate requires two distinct
reports over the same head/artifact/protocol. A one-candidate weighted merge
now returns the candidate exactly instead of recomputing
`base + (candidate - base)`, which avoids cancellation and bit drift. JSON
metric persistence enables exact floating-point round trips and verifies the
canonical report identity before publication. The complete generic P2P library
suite passes 241 tests with these contracts. Heterogeneous validators over a
real WAN, quorum loss, disagreement, and adversarial authorization remain
staging gates.

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

Generated NCA and generated formal Ruliad are shared browser/native source
contracts. The real Chrome/WebGPU Ruliad gate checks formal-family metadata,
the structured 272-token vocabulary, target masks, and stream/TBPTT behavior
over two train and two evaluation batches. Browser AdamW aggregates detached
scalar losses on the GPU and performs one asynchronous read at the window
boundary rather than one synchronization per training step.

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

The artifact-window parity profile declares optimizer and scheduler state as
`reset_per_window`. The protocol-aware DiLoCo profile instead declares both as
`peer_local_persistent`: records survive reconciliation on the same peer but
are not transferred when that peer adopts another model. Both behaviors are
implemented and bound into distinct signed training contracts. No optimizer
moments are silently attached to a canonical head or transferred between
peers.

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

- complete `burn_dragon_p2p` suite: 96 library tests, 18 native-CLI tests, and
  19 non-ignored integration tests; 14 named release/scale tests remain
  operator-invoked and ignored by default
- native integration coverage includes shard-backed NCA, live-source Ruliad,
  mixed native/browser roles, persistence, and diffusion
- 158 `burn_dragon_universality` tests and 184 focused language Ruliad tests
- strict native P2P, language-training, universality, xtask, and WASM package
  Clippy for the exercised surfaces
- real headless Chrome/WebGPU execution of generated NCA and generated formal
  Ruliad browser training from the WASM package; both source tests pass
- production Pages shell build with the WebGPU WASM bundle and auth callback
  routes
- real Firefox WASM execution for lease-scoped shard selection
- real Firefox no-WebGPU downgrade to a read-only role
- signed three-peer restart recovery in 3.176 seconds with persistent peer
  identity and exact head/tensor/validation recovery
- strict signature, conflicting-contract, exact-artifact, decoded-tensor,
  lease, wrong-owner, batch-digest, unleased-record, and forged-fitness checks
- partition, stale-head, slow-peer, churn, relay-loss, and rejoin recovery
  matrix in `burn_p2p_testkit`
- non-ignored NCA and ClimbMix mixed-fleet smokes plus the explicit ignored
  medium rung
- operator-invoked CUDA-trainer/CPU-validator Ruliad promotion with an exact
  cross-backend tensor digest and content-addressed formal report
- real Ruliad quorum-two promotion requiring distinct exact-head reports from
  both read-only validators
- deployment workflow contract checks and WebGPU WASM target check
- strict 1M-parameter three-peer DiLoCo bulk exchange
- corrected-objective three-seed, six-round DiLoCo convergence parity
- matched FP32/FP16 random-scaffold codec quality and payload ablation

The local Chrome result is a real browser GPU training result, but it uses the
locally built WASM package and local source contracts. The current Firefox
runner did not expose WebGPU; its successful downgrade remains valid
role/capability evidence. A deployed Pages/edge Chrome/WebGPU canary against a
signed production contract is still mandatory.

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
- evaluator head/revision, report/protocol IDs, sample count, and evaluation latency
- artifact and update bytes
- decoded norm, non-finite, clipping, and replay-verification status
- tokens/samples processed and accepted
- browser WebGPU availability and device-loss rate
- auth denial, replay rejection, quarantine, and validator disagreement
- Dragon loss, verifier accuracy, output degeneracy, and ruliad difficulty
- trainer model duty and wall tokens/s, separated from evaluator compute duty

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
