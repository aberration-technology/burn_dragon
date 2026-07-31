# Random-Scaffold Dragon Report

Date: 2026-07-30

## Decision

Dragon's random-scaffold parameterization has cleared the bounded promotion
gates covered by this report:

- Three matched local CUDA seeds achieved quality parity with dense Dragon over
  2,048 AdamW steps. Rank-stabilized rank 16 improved mean validation
  cross-entropy by 7.18% and verifier rate by 7.29 percentage points.
- Mean training throughput was 95.58% of dense. A larger batch-32 trace kept
  the GPU compute phase dense: 90.97% mean active utilization versus 92.60%
  for dense, with no utilization or power stall pattern.
- Three native peers achieved at least 0.90 of synchronized-reference
  convergence on all three seeds. Mean endpoint progress ratio was 0.9308 and
  mean trailing progress ratio was 0.9287.
- The final release-profile six-round gate passed with exact protocol replay,
  monotonic validation, no transient drawup, no all-padding batches, and no
  hard request failures.
- A matched three-seed FP16 transport matrix preserved final CE within
  `6.2e-6` of FP32 while reducing mutable-update payload bytes by exactly 50%.

This promotes rank-stabilized rank 16 as the random-scaffold experiment
default. It does not replace dense Dragon as the general training default.
Internet/WAN convergence, browser-WebGPU convergence, long-horizon reasoning
quality, and large-model scaling remain outside the evidence in this report.

## Model Contract

For each selected shared Dragon projection, the effective matrix is

```text
W_effective = gain * W_seed + scale * A * B
```

`W_seed` is immutable and reconstructed from a versioned generator contract.
`A` is initialized from a deterministic Kaiming-uniform stream, `B` starts at
zero, and `gain` starts at one. Standard LoRA uses `scale = alpha / rank`;
rank-stabilized LoRA uses `scale = alpha / sqrt(rank)`.

The implementation follows the random-scaffold and rsLoRA ideas from
[A Little Rank Goes a Long Way: Random Scaffolds with LoRA Adapters Are All
You Need](https://arxiv.org/abs/2604.08749), with one important
architecture-specific correction: each scaffold tensor uses Dragon's native
projection initializer standard deviation. It is not blindly scaled by
`1 / sqrt(fan_in)`. That preserves the recurrent model's expected initial
signal scale.

The promoted experiment profile is:

```toml
[model.random_scaffold]
enabled = true
seed = 20260729
distribution = "gaussian_clt12"
rank = 16
alpha = 16.0
scaling = "rank_stabilized"
trainable_gain = true
```

Dragon shares its recurrent encoder, value encoder, and decoder across layers
and recurrent time. Scaffold mode therefore adds one adapter to each shared
matrix, not one adapter per unrolled layer or step. Split fast/slow
hierarchical models receive separate deterministic scaffold paths and adapter
sets.

`materialize_random_scaffold_for_inference` folds the trained effective
matrices once after conversion to a validation/inference backend. Token-by-token
generation does not recompute `A * B`.

## Ownership

The implementation preserves repository boundaries:

- `burn_eggroll` owns the generic, versioned scaffold generator, tensor
  manifest, distribution, adapter scaling, and artifact-size contracts.
- `burn_dragon_core` selects Dragon matrices, applies Dragon-specific
  initialization scale, computes effective weights, and folds inference
  weights.
- `burn_dragon_language` validates training compatibility, writes manifests,
  handles checkpoints/resume, and provides matched experiment profiles.
- `burn_p2p` owns generic deterministic-genesis, mutable-parameter subset, and
  persistent Burn optimizer/scheduler state contracts.
- `burn_dragon_p2p` binds Dragon catalogs, native/browser artifacts, masked
  token records, and the DiLoCo experiment gate.

Random-scaffold training remains AdamW-only. EGGROLL's current population
executor evolves dense shared projections, continual backprop replaces
features, and neuron widening changes the reconstruction shape. Those modes
are rejected instead of silently mutating an immutable scaffold contract.

## Local CUDA Quality

The matched matrix used:

- NVIDIA GB10 CUDA backend;
- seeds 29, 30, and 31;
- 4 layers, embedding 64, 4 heads, latent width 1,024;
- block 128, batch 8, and 2,048 optimizer steps;
- Ruliad answer-completion data;
- AdamW at `1e-3`, no dropout, no continual backprop, and no neuron scaling.

### Selected Comparison

| Seed | Dense valid CE | rs-rank16 valid CE | Dense verifier | rs-rank16 verifier |
|---:|---:|---:|---:|---:|
| 29 | 0.20366 | 0.19661 | 0.2656 | 0.2656 |
| 30 | 0.21905 | 0.20627 | 0.1875 | 0.2656 |
| 31 | 0.21482 | 0.18885 | 0.1719 | 0.3125 |
| Mean | 0.21251 | 0.19724 | 0.2083 | 0.2813 |

| Aggregate metric | Dense | rs-rank16 | Relative result |
|---|---:|---:|---:|
| Validation CE | 0.21251 | 0.19724 | 7.18% lower |
| Train loss | 0.20622 | 0.19895 | 3.52% lower |
| Verifier rate | 0.2083 | 0.2813 | +7.29 points |
| Completion health | 0.8469 | 0.9219 | +7.50 points |
| Steps/s | 72.831 | 69.613 | 4.42% lower |
| Mean active GPU utilization | 61.76% | 62.78% | 1.64% higher |

All three paired validation results favored rank 16. This is a small
three-seed screen, so it establishes an engineering parity gate, not a
publication-level estimate of superiority.

The rank screen also retained standard rank 8 and rank-stabilized ranks 16 and
32:

| Variant | Mean valid CE | Mean verifier | Mean steps/s |
|---|---:|---:|---:|
| Dense | 0.21251 | 0.2083 | 72.831 |
| Standard rank 8 | 0.20029 | 0.2292 | 68.633 |
| Rank-stabilized rank 16 | 0.19724 | 0.2813 | 69.613 |
| Rank-stabilized rank 32 | 0.22234 | 0.2448 | 70.921 |

Rank 32 did not improve the objective, so increasing adapter rank is not
monotonic. Rank 16 is selected by measured quality, not by assuming more rank
is better.

The machine-readable analyzer now derives completed steps from run events and
enforces:

- three identical seed sets;
- mean validation loss within 2% of dense;
- every-seed validation loss within 5% of dense;
- verifier rate no more than 2 points below dense;
- throughput and active utilization at least 90% of dense;
- completion health no more than 5 points below dense.

The current `quality_efficiency_gate_passed` value is `true`.

## GPU Efficiency

A second release CUDA screen increased the workload to embedding 128, latent
width 16,384, batch 32, block 128, and 64 steps. Active samples are those with
at least 50% GPU utilization, excluding startup, evaluation, and process exit.

| Metric | Dense | rs-rank16 |
|---|---:|---:|
| Mean active utilization | 92.600% | 90.969% |
| Minimum utilization inside active envelope | 77.000% | 75.000% |
| Peak active utilization | 95.000% | 95.000% |
| Mean active power | 40.287 W | 39.322 W |
| Peak active power | 43.120 W | 43.030 W |
| Samples below 50% inside active envelope | 0 | 0 |
| End-to-end wall time | 31.67 s | 35.56 s |
| Peak host RSS | 740,472 KiB | 866,500 KiB |

The training compute phase is dense and power-stable. The 12.28% end-to-end
penalty is larger than the 4.42% matrix-average training-throughput penalty
because validation, scaffold folding, and checkpoint work are included in wall
time. Adapter matmuls are not fused into Dragon's dense projection kernels, so
the current implementation should not be described as faster than dense.

## Corrected P2P Objective

The first P2P convergence experiment was invalid: fixed-size document storage
filled the remainder of short documents with repeated EOS tokens, and the
full-document objective supervised that fill. Peers could reduce loss by
learning padding while a synchronized reference consumed a different effective
objective.

The corrected contract:

- retains supervision on the first true EOS target;
- masks every repeated EOS fill target after document end;
- carries the optional loss mask through native shard records, native Burn
  batches, browser records, browser AdamW, and browser EGGROLL;
- keeps legacy records backward-compatible by treating a missing mask as all
  ones;
- fingerprints inputs, targets, and masks in the convergence report;
- rejects any promoted run containing an all-zero-supervision batch.

Across the promoted P2P matrix, every run consumed 18 unique batches with zero
duplicates and zero padding-only batches. Supervised-token fractions ranged
from 0.6753 to 0.6886.

## Persistent Inner Optimizer

The second convergence defect was optimizer discontinuity. The generic Burn
learner rebuilt AdamW and its scheduler at every DiLoCo window, discarding
moments and schedule progress.

`burn_p2p` now provides a generic persistent inner-loop contract:

- `from_stateful_components`;
- `from_stateful_loaders`;
- `run_persistent_inner_steps`;
- `BurnPersistentInnerLoopResult`.

Optimizer and scheduler records are serialized into peer-local state after a
window and restored before the next one. They are deliberately not transmitted
with model updates. A numerical unit test proves that two serialized
one-step Adam rounds match uninterrupted two-step Adam to `1e-6`.

This behavior is part of the signed training contract, not an implementation
convention. DiLoCo declares optimizer and scheduler policy as
`peer_local_persistent`; artifact-window training declares `reset_per_window`.
Adopting another peer's model never imports that peer's optimizer moments or
scheduler cursor.

The synchronized oracle also accumulates the same three peer microbatch
gradients into one AdamW update. It no longer compares one P2P outer round
against three sequential optimizer updates.

## Native Transport

Native training uses two established connections per peer:

- one steady route for bidirectional state/gradient request streams;
- one temporary reconciliation route while simultaneous dials settle.

The old single-route limit produced 35-second request timeouts under six-round
full-mesh traffic. With two routes, all promoted rounds completed without a
hard request failure. Browser peers retain a one-route limit and bootstrap
peers retain four.

## Three-Peer Convergence

The promoted release-profile matrix used a 1,012,229-parameter
random-scaffold Dragon model, three native NdArray CPU peers, six DiLoCo
rounds, one local AdamW step per peer per round, and an FP32 outer SGD update at
learning rate 1.20. The synchronized mutable subset contains 225,797 values, or
22.31% of model parameters.

| Seed | Genesis CE | P2P final CE | Sync final CE | Endpoint ratio | Trailing ratio | Peer steps/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1337 | 5.67749 | 3.68906 | 3.56850 | 0.94284 | 0.94098 | 1.0094 |
| 1338 | 5.66297 | 3.65551 | 3.47303 | 0.91667 | 0.91635 | 1.0567 |
| 1339 | 5.67693 | 3.53965 | 3.38578 | 0.93284 | 0.92890 | 1.0275 |
| Mean | 5.67246 | 3.62807 | 3.47577 | 0.93078 | 0.92874 | 1.0312 |

The hard gate is 0.90 trailing synchronized progress. The minimum observed
ratio was 0.91635. Every P2P validation curve was monotonic, every protocol
oracle comparison was exact, and hard request failure count was zero.

Seeds 1337 and 1338 participated in outer-learning-rate tuning. Seed 1339 was
the untouched holdout. Every release run explicitly removed the
outer-learning-rate environment override and confirmed the 1.20 default:

| Release matrix metric | Result |
|---|---:|
| Mean endpoint progress ratio | 0.93078 |
| Mean trailing progress ratio | 0.92874 |
| Minimum trailing progress ratio | 0.91635 |
| Maximum validation drawup | 0.00000 |
| Mean aggregate peer steps/network second | 1.0312 |
| Hard request failures | 0 |
| Per-seed wall time including setup/evaluation | 71.19-72.64 s |
| Peak host RSS range | 3,228,196-3,379,168 KiB |

## Bandwidth

Matched FP32 and FP16 mutable-update matrices both covered seeds 1337-1339:

| Mean metric | FP32 | FP16 |
|---|---:|---:|
| P2P final CE | 3.628072 | 3.628078 |
| Endpoint progress ratio | 0.930783 | 0.930781 |
| Trailing progress ratio | 0.928742 | 0.928740 |
| Estimated wire payload/run | 21,676,512 B | 10,838,256 B |
| Aggregate peer steps/s | 1.0312 | 1.0138 |

FP16 halved parameter payload with a mean final-CE difference of
`+0.0000061`; every convergence and protocol gate passed for all six runs.
Local-loopback throughput was 1.69% lower because codec work is visible when
bandwidth is effectively free. FP16 is therefore a validated bandwidth option,
not a local-compute speedup. It remains opt-in until measured on a constrained
real network and browser peers.

## Browser Alignment

The browser path now reconstructs signed compact scaffold genesis and promoted
heads, uses the same mutable catalog order, and consumes the same target loss
masks as native peers. Verified coverage includes:

- native/browser mutable ordering and complete-model tensor digest parity;
- legacy and masked token-record batching;
- malformed loss-mask rejection;
- headless Firefox tests proving browser batches preserve native masks;
- `wasm32-unknown-unknown` library compilation with `wasm-peer,wgpu`.

This is mechanics and objective parity, not browser convergence evidence. A
real browser-WebGPU peer has not yet run the six-round convergence matrix.

## Verification

The final source state passed:

```bash
cargo fmt --all -- --check

cargo clippy -p burn_p2p_swarm --all-targets -- -D warnings
cargo clippy -p burn_p2p --features burn --all-targets -- -D warnings
cargo clippy -p burn_dragon_p2p --features native \
  --all-targets --no-deps -- -D warnings

cargo check --target wasm32-unknown-unknown \
  -p burn_dragon_p2p --no-default-features \
  --features wasm-peer,wgpu --lib

cargo test --release -p burn_dragon_p2p --features native \
  --test native_training --no-run
```

Focused native, browser, scaffold, loss-mask, persistent-Adam, transport, and
protocol-oracle tests also pass. The full Dragon dependency-inclusive strict
clippy lane is not green: it currently exposes an older warning backlog in
`burn_dragon_language` (65 library warnings and 96 when tests are included).
The Dragon P2P crate itself is clean under `--no-deps`; this report does not
misrepresent the repository-wide lint state.

Generated evidence is retained under:

- `target/experiments/random-scaffold-parity/`;
- `target/experiments/random-scaffold-efficiency/`;
- `target/test-artifacts/random-scaffold-diloco/masked-route2/`;
- `target/test-artifacts/random-scaffold-diloco/release-default/`;
- `target/test-artifacts/random-scaffold-diloco/release-fp16/`;
- `target/test-artifacts/random-scaffold-diloco/release-contract-final/`.

## Remaining Production Gates

The demonstrated claim is narrow and reproducible: random-scaffold Dragon has
local small-model quality parity and localhost native P2P convergence parity
without a GPU utilization collapse.

The remaining production work is:

1. Repeat quality and convergence at 10M, 100M, and long continual-learning
   horizons with stronger absolute verifier performance.
2. Run multi-host WAN tests with delay, loss, churn, and constrained
   bandwidth.
3. Run the same objective and convergence oracle through real browser WebGPU
   trainers, including capability downgrade and reconnect/resume.
4. Fuse or cache adapter projection work further to close the remaining
   4.42% training-throughput and 12.28% end-to-end wall-time gaps.
5. Measure FP16 codec cost and end-to-end benefit on constrained WAN links and
   browser peers before making it the wire default.

Those gates constrain production deployment claims; they do not invalidate the
bounded parity result established here.
