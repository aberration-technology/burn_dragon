# Predictive Coding in Dragon Training

Date: 2026-08-06

## Status

This document records canonical local predictive-coding solvers and the older recurrent-state
correction auxiliary as separate mechanisms. AdamW backpropagation remains the default training
algorithm. Fixed prediction reaches backpropagation-equivalent convergence without a global
backward call but is slower. The new depth-batched layer-local prediction solver is substantially
more sample efficient than matched terminal-loss backpropagation on a bounded non-ceiling modular
stream while retaining 95-97% throughput. It is not a terminal-gradient approximation, and its
positive result requires sparse context routing plus context-scoped optimizer state.

The continual-learning result remains a complementary-system result. Task-ID-free context
selection, balanced sparse routing, and context-scoped optimizer state are responsible for
retention; layer-local prediction improves acquisition inside that system. A dense shared-state
control forgets more under the same local derivative. Production and Ruliad promotion remain open.

## 2026-08-06 depth-batched layer-local prediction

`solver = "layer_local_prediction"` attaches a next-token factor to every shared Dragon layer use.
Intermediate activities are detached. Layer depth is folded into the batch axis, producing one
batched head VJP, one batched shared-body VJP, and one embedding VJP per update. Shared-body credit
is summed over the physical uses of Dragon's shared weights; the shared readout is averaged over
the auxiliary local readouts. The terminal loss and supervised-token count remain exactly the
ordinary masked terminal cross entropy. Recurrent rho continuation is the ordinary feed-forward
terminal state, so TBPTT does not carry transient inferred activities.

The full 891,266-parameter CUDA geometry reports terminal-loss error below `5e-7`, gradient norm
ratio `1.012`, and global cosine `0.6222` against terminal backpropagation. The modest cosine is
expected: this is a layer-local semi-gradient with direct supervised factors, not an approximation
to the terminal derivative. A small CPU geometry reaches cosine `0.9531`, showing that similarity
depends on depth and operating point. The implementation makes zero global backward calls.

The release-CUDA continual matrix uses four shared layer uses, embedding 96, four heads, latent
width 3,072, batch 16, block 16, modulus 32, four sequential recurrence tasks, four holdout batches,
and paired seeds 17/29/43/59/71. Train and holdout initial conditions are disjoint. Both learners use
learning rate `0.003`, the same model initialization, token stream, sparse masks, and context-scoped
optimizer lifecycle. Intervals are two-sided paired or per-arm 95% Student-t intervals.

| Updates/task | Learner | Acq | Final accuracy | Mean forgetting | Tokens/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 128 | AdamW backprop | 5/5 | 0.4952 +/- 0.1037 | 0.0198 +/- 0.0225 | 23,843 +/- 590 |
| 128 | Layer-local PC | 5/5 | **0.7692 +/- 0.0626** | 0.0175 +/- 0.0232 | 22,758 +/- 275 |
| 256 | AdamW backprop | 5/5 | 0.7241 +/- 0.1348 | 0.0102 +/- 0.0159 | 23,242 +/- 485 |
| 256 | Layer-local PC | 5/5 | **0.9876 +/- 0.0107** | 0.0043 +/- 0.0038 | 22,495 +/- 691 |

At 128 updates, the paired layer-local accuracy delta is `+0.2740 +/- 0.0627`; at 256 it is
`+0.2635 +/- 0.1363`. Every paired seed has the same positive direction. Forgetting differences
cross zero at both budgets. Layer-local PC retains 95.5% and 96.8% of backpropagation throughput,
respectively, while making three local VJP launches per update and no global backward call.

The dense shared-state control prevents an intrinsic-PC claim. At 128 updates, dense AdamW and
dense layer-local PC reach `0.2786` and `0.2956` accuracy, a paired delta of only
`+0.0170 +/- 0.0603`. Layer-local mean forgetting is worse (`0.9296` versus `0.5318`). Sparse
context isolation is therefore part of the successful training system rather than an optional
evaluation convenience.

Long-budget testing exposed and resolved a selector failure. A fixed sequential novelty threshold
eventually treated a stationary hard sample as a new task; conservative calibration avoided that
false positive but under-discovered contexts at the shorter budget. The router now evaluates one
deterministic unallocated reserve subnetwork only after every existing expert rejects a prefix. A
new context is confirmed only if that reserve is within the best expert's calibrated loss scale.
With the responsive calibration, both 128- and 256-update five-seed matrices discover exactly four
contexts with selector accuracy 1.0 and no duplicate allocation. Rejected reserve tests update the
selected expert's calibration, and the production probe does not materialize checkpoint state.
Automatic least-recently-used replacement is no longer the default because a full bank has no true
unallocated control.

The machine-readable result is
`docs/experiments/predictive-coding-layer-local-20260806.json`. Raw reports are under
`target/pc-layer-local-20260806/`. This is strong bounded structured-recurrence evidence, not a
state-of-the-art claim. Text/Ruliad fixed-holdout convergence, longer context churn, larger models,
TBPTT quality, and decentralized synchronization remain required promotion gates.

## 2026-08-04 Lifelong Context-Routing Result

The prior status remains correct for **plain** predictive coding. A new controlled result supports
a narrower system-level claim: fixed-prediction local learning combined with task-ID-free context
selection, balanced sparse subnetworks, and context-scoped optimizer state is a materially stronger
continual learner than an ordinary dense AdamW model. It does not show that predictive-coding
derivatives outperform backpropagation when both use the same routing system.

This distinction matches the source research. [Lifelong Neural Predictive
Coding](https://proceedings.neurips.cc/paper_files/paper/2022/file/26f5a4e26c13d1e0a47f46790c999361-Paper-Conference.pdf)
attributes retention to a complementary system: a task selector drives lateral competition in the
generative circuit, reducing representational overlap. Its
[supplement](https://proceedings.neurips.cc/paper_files/paper/2022/file/26f5a4e26c13d1e0a47f46790c999361-Supplemental-Conference.pdf)
states explicitly that resistance to forgetting comes from the interaction between selector and
generative circuit, not from predictive coding in isolation.

The controlled benchmark is `pc_lifelong_stream`. It presents four context-identifiable modular
recurrence laws sequentially. Each law has disjoint train and holdout initial conditions, while all
laws use the same initial-condition seeds. There is no task marker. The context selector receives a
unit-normalized modular-transition consistency sketch computed from the first 16 observed tokens;
supervision starts only after that support prefix is causally available. The selector never receives
the hidden benchmark task enum. Cosine novelty creates contexts online, holdout selection must recover
the learned context, and a usage-balanced mask allocator minimizes channel reuse. At active fraction
0.25, four contexts partition residual and neuron channels without overlap. Every arm uses the same
model initialization, examples, token budget, and evaluation matrix.

The release-CUDA matrix used NVIDIA GB10, 888,194 parameters, four shared Dragon layer uses,
embedding 96, four heads, latent width 3,072, batch 32, block 128, 128 updates per task, 2,097,152
train tokens per run, four holdout batches per task, and five seeds. `Acq` is the number of seeds that
passed every task's acquisition gate. A gate first requires the matched backprop baseline itself to
reduce loss by at least 0.5 and gain at least 0.25 accuracy, then requires the candidate to retain at
least 90% of that loss reduction and stay within 0.05 accuracy gain. Across the matrix, the weakest
baseline task reduced loss by 2.414 and gained 0.802 accuracy.

| Learner | Routing / optimizer state | Acq | Final accuracy | BWT | Mean forgetting | Tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Backprop + AdamW | Dense / shared | 5/5 | 0.3849 +/- 0.0214 | -0.8198 +/- 0.0282 | 0.8198 +/- 0.0282 | 66,316 +/- 1,507 |
| Fixed-prediction PC | Dense / shared | 5/5 | 0.3963 +/- 0.0080 | -0.8050 +/- 0.0107 | 0.8050 +/- 0.0107 | 41,905 +/- 615 |
| Backprop + AdamW | Selected sparse / context-scoped | 5/5 | 0.9935 +/- 0.0134 | +0.0015 +/- 0.0034 | **0.0000 +/- 0.0000** | **70,375 +/- 1,541** |
| Fixed-prediction PC | Selected sparse / context-scoped | 5/5 | **0.9984 +/- 0.0027** | -0.0005 +/- 0.0012 | 0.0005 +/- 0.0012 | 40,624 +/- 665 |

Against dense AdamW, the complete fixed-PC system improves final accuracy by
`+0.6135 +/- 0.0235`, BWT by `+0.8193 +/- 0.0278`, and mean forgetting by
`-0.8193 +/- 0.0278`; every paired seed has the same direction. Selector accuracy is 1.0, and exactly
four contexts are created, in every routed run. Fixed-PC makes zero global backward calls. It retains
61.3% of dense-AdamW throughput and 57.7% of matched routed-AdamW throughput. Direct sampling during
the arms showed 90-91% GPU utilization and roughly 40-42 W, while host/shared memory remained about
7.6 GiB. The throughput gap is repeated local device work, not a host or data-generation stall.
The paired 95% confidence intervals are `[+0.5843, +0.6426]` for final accuracy,
`[+0.7848, +0.8537]` for BWT, and `[-0.8537, -0.7848]` for forgetting.

The matched routed-AdamW control is important. Routed fixed-PC minus routed backprop is
`+0.0048 +/- 0.0146` final accuracy, `-0.0021 +/- 0.0033` BWT, and
`+0.0005 +/- 0.0012` forgetting: statistical parity at this sample size and ceiling, not PC
superiority. Routed backprop is 1.73x faster. Fixed prediction is backpropagation-equivalent local
error transport, so parity is the expected result. Dense fixed-PC also has only a small,
high-variance advantage over dense backprop (`+0.0113 +/- 0.0182` accuracy). The benchmark supports
the paper's complementary-system claim, not a claim that its PC derivative is intrinsically better
than AdamW/backpropagation.
The matched routed 95% confidence intervals all cross zero: `[-0.0132, +0.0229]` accuracy,
`[-0.0062, +0.0021]` BWT, and `[-0.0009, +0.0020]` forgetting.

A prospective reverse-Gauss-Seidel smoke at half the token budget provides the negative control. It
kept energy monotone and used zero global backward calls, but failed acquisition: final accuracy was
0.7325 versus 0.9944 for matched backprop, at 11,416 versus 71,781 tokens/s. Prospective equilibrium
learning therefore remains experimental.

### 2026-08-05 feedback-transport and batching screen

Four follow-up implementations were screened against the same numerical and acquisition contracts.
None passed, and all corresponding solver/configuration branches were removed rather than retained
as dormant production surface.

An identity direct-feedback arm broadcast the terminal residual-stream error to every shared layer
use. On the full 937,154-parameter CUDA fidelity geometry, its global cosine with exact backprop was
only `0.6608`, with norm ratio `1.0411` and relative L2 error `0.8414`. The embedding cosine was
`0.2097`. Dragon's equal residual widths make the broadcast shape-valid, but they do not make the
sample-, token-, and activation-dependent layer Jacobians identity maps.

A bounded exact-transport arm then applied the true local activity Jacobian through the last `k`
layer uses and broadcast the resulting shared residual error below that boundary:

| Exact transport depth | Global cosine | PC/reference norm | Relative L2 |
| ---: | ---: | ---: | ---: |
| 0 | 0.6608 | 1.0411 | 0.8414 |
| 1 | 0.8704 | 0.9308 | 0.4959 |
| 2 | 0.9572 | 0.9456 | 0.2898 |
| 3 | 0.9893 | 0.9799 | 0.1461 |
| 4 | 0.9999997 | 1.0000 | 0.00083 |

The fidelity frontier is smooth, but the intermediate point is not an efficient learner. In a
matched one-seed, 128-update-per-task CUDA run, exact depth two reached final accuracy `0.3915`, BWT
`-0.8114`, and 29,800 tokens/s. Matched backprop reached `0.3970`, `-0.8040`, and 68,633 tokens/s;
fixed prediction reached `0.3996`, `-0.8006`, and 43,179 tokens/s. Depth two failed the acquisition
gate on task D with an accuracy-gain delta of `-0.0660`. Device sampling remained dense at 91-95%
SM utilization, so the deficit was extra local Jacobian work rather than a host stall.

A learned amortized-feedback arm maintained one residual-width matrix per shared layer use and
updated it from periodic exact-Jacobian calibration samples. Learning-rate screens at `1e-6`,
`0.01`, and `0.25` all failed acquisition on the bounded CPU benchmark, with final accuracy between
`0.153` and `0.174`. A single sample-independent linear map cannot track the changing attention,
ReLU, normalization, token, and context Jacobians in this Dragon factor. A future learned transport
must be sample-conditioned and nonlinear; retaining this matrix state would only create misleading
checkpoint and run-lifecycle complexity.

Finally, fixed prediction was rewritten as a reverse activity-only wave followed by one batched
shared-parameter VJP. CUDA fidelity remained exact (`0.9999997` cosine and `0.00083` relative L2),
but steady-state throughput fell to 25,463 tokens/s because the second factor pass duplicated most
of the local Jacobian work. Reduction-order drift also moved task-D acquisition just outside the
configured tolerance (`-0.0543`). The serial full-VJP reverse wave was restored.

The first context-selector follow-up also failed: a generic token-transition hash sketch did not
provide a stable novelty margin. That negative result is retained as evidence against fixed random
descriptors, but it has been superseded by the causal predictive-loss selector described below.

**Decision at the time of this screen:** the supported local-PC solver surface was synchronous
equilibrium, reverse-Gauss-Seidel, and fixed prediction. Fixed prediction was the
numerical/convergence control; the other two were research controls. The 2026-08-06 section records
the later layer-local solver and supersedes this historical surface decision.

Raw feedback-screen reports are under `target/pc-feedback-screen-20260805/`.

Raw reports are under `target/pc-lifelong-task-id-free-1m-cuda-20260804/analysis/`. This is controlled
benchmark evidence, not yet a production or Ruliad promotion. The benchmark uses a family-aware but
task-ID-free transition sketch and a fixed four-context capacity; it is retained only as the
descriptor control for the causal follow-up below. The final schema-v2 files are `dense-five-seed-v2.json`,
`routed-five-seed-v2.json`, and `reverse-seed-17-smoke-v2.json`.

### 2026-08-05 causal predictive context and recurrent-factor follow-up

`burn_pc` now provides a model-agnostic, serializable `PredictiveContextBank` and
`PredictiveContextNoveltyGate`. Dragon scores only the causally visible prefix under every known
subnetwork. Minimum absolute next-token loss chooses the expert. Per-expert EMA mean and variance
define novelty envelopes; calibrated z-scores are diagnostic rather than routing scores. A new
context requires three consecutive all-expert rejections, preventing one transient loss spike from
allocating permanent model capacity. Selection is read-only until the caller explicitly observes a
training decision, so holdout evaluation cannot mutate calibration or confirmation state.

The hot routing probe batches all expert losses into one backend-to-host read. With the default
eight-update cadence, routing remains a small periodic side path rather than part of every model
step. The benchmark buffers pending confirmation probes at a true boundary and performs no
parameter update against their fallback expert. Committed routing accuracy is therefore reported
separately from discovery delay. A host-side Criterion microbenchmark measures context-bank
selection at about 32 ns for eight experts and 199 ns for 64 experts; model prefix scoring and its
single scalar readback, not bank selection, dominate routing cost.

The release-CUDA follow-up uses the same GB10, 888,194-parameter, four-layer-use geometry as the
original controlled matrix: embedding 96, four heads, latent width 3,072, batch 32, block 128, 128
updates per task, four holdout batches, and 2,097,152 training tokens per arm. Four contexts use a
0.25 active fraction, giving zero pairwise overlap in both residual and neuron masks.

| Learner | Seeds | Acq | Contexts | Final accuracy | BWT | Mean forgetting | Tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Backprop + AdamW | 3 | 3/3 | 4/4 | 0.9849 +/- 0.0624 | +0.00004 +/- 0.00017 | 0.00238 +/- 0.01025 | 71,900 +/- 3,822 |
| Fixed-prediction PC | 3 | 3/3 | 4/4 | 0.9979 +/- 0.0089 | 0.00000 +/- 0.00000 | 0.00000 +/- 0.00000 | 42,574 +/- 660 |

Every arm discovers exactly four contexts, passes all four acquisition gates, and avoids duplicate
allocation. The legacy schema counted deferred discovery decisions as selector errors. The
finalized schema-6 rerun reports 123 committed decisions at `123/123 = 1.0` accuracy and seven
deferred probes per arm. Fixed prediction retains 59.2% of matched Backpropagation throughput.
During direct sampling both learners sustain roughly 89-91% GPU utilization at 40-43 W with
7.5-8.0 GiB host memory in use. The quality result supports causal context discovery and
sparse/context-scoped retention, not intrinsic PC superiority: the paired quality difference is at
a ceiling and Backpropagation remains 1.69x faster.

Local PC also now accepts one detached incoming rho per shared layer use. The analytic dense causal
attention VJP includes exact query, value, ALiBi-decay, and incoming-state terms. The terminal rho is
the ordinary feed-forward continuation state; activity inference remains transient. Production
`TrainStep` slices the block at `tbptt_chunk_size`, carries rho between factors and optionally across
batches, aggregates shared derivatives, and honors stream reset. Masked document chunks are weighted
by their raw supervised-token count entirely on device, including zero-supervision chunks. The
recurrent forward reuses the fused dense-attention kernel and adds only the incoming-rho context plus
the terminal-state reduction.

Numerical gates pass on NdArray: the stateful attention VJP matches autodiff for query, value, decay,
and incoming rho; recurrent fixed-prediction gradients have cosine above 0.99999 and relative L2
below `1e-4` against a detached-TBPTT reference; terminal rho matches ordinary Dragon state; and
chunked masked loss matches the full-block loss. The canonical profile composition is
`local-pc-smoke.toml + pc-fixed-prediction.overlay.toml + pc-recurrent-tbptt.overlay.toml`.

The release-CUDA production path also completed an eight-step recurrent smoke with a complete
model, optimizer, and scheduler checkpoint triplet. Every optimizer step reported four chunks corrected,
16 local VJPs, no global autodiff graph, and zero global backward calls; mean validation CE was
5.495 after the intentionally short run. In a matched 32-step wiring comparison on the same tiny
profile, stateless and recurrent runs completed in 3.75 s and 3.88 s wall time. Steady local-PC
correction time was 6.29 ms for one full-block factor versus 24.97 ms for four recurrent factors;
validation CE was 4.549 versus 4.581. This verifies production execution and quantifies chunking
cost, but is too short and too small to establish a recurrent-quality advantage.

Raw follow-up artifacts are
`target/pc-remaining-blockers/cuda-routed-confirmed-seed17.json` and
`target/pc-remaining-blockers/cuda-routed-confirmed-three-seed.json`. The corrected schema-6 control
is `target/pc-remaining-blockers/cuda-routed-confirmed-schema6-seed17.json`; production smoke events
are under `runs/one-steel`, `runs/roasted-slope`, and `runs/odd-push`.

At the time of that matrix, the remaining promotion boundary was architectural: the normal Ruliad loader did not yet own a
run-scoped context/subnetwork bank or context-scoped optimizer collection. Adding routing there
without those two pieces would reproduce neither the controlled retention mechanism nor its
checkpoint semantics. Until that integration receives a Ruliad holdout matrix, the predictive bank
was a tested generic primitive and benchmark integration, not a default production policy. The
follow-up below closes the lifecycle integration while retaining the quality promotion gate.

### 2026-08-05 production routing and exact-resume follow-up

The local language pipeline now owns the complete routed-learning state that was missing above.
The router, selected sparse masks, per-context AdamW moments, per-context recurrent TBPTT state,
source selector, stochastic schedule counter, stability controller, and `burn_ecs` run state are
all checkpointed. Validation uses the selected context without mutating calibration, reports both
source-weighted and stream-warm metrics, and keeps the same context-local recurrent state contract
as training. Gradient accumulation rejects routed execution rather than silently sharing one
context optimizer across an ambiguous accumulation window.

The bounded release-CUDA confirmation uses 888,194 parameters, four shared Dragon layer uses,
embedding 96, four heads, latent width 3,072, batch 16, block 16, 128 updates per task, four
holdout batches per task, and seeds 17/29/43. It is deliberately shorter than the earlier
batch-32/block-128 descriptor-control matrix: its purpose is to verify the production router and
lifecycle implementation under three independent trajectories.

| Learner | Seeds | Acq | Contexts | Selector | Final accuracy | Mean forgetting | Max forgetting | Tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Routed AdamW/backprop | 3 | 3/3 | 4 | 1.000 | 0.98134 | 0.00868 | 0.02257 | **23,127** |
| Routed fixed prediction | 3 | 3/3 | 4 | 1.000 | **0.98958** | **0.00540** | **0.01505** | 18,094 |

The paired fixed-prediction deltas are `+0.00825` final accuracy and `-0.00328` mean forgetting.
Every arm discovers exactly four contexts and passes every acquisition gate. Fixed prediction
makes zero global backward calls and 3,072 explicit local VJPs per run; backprop makes 512 global
backward calls. Fixed prediction retains 78.3% of matched throughput. The direction is promising,
but three ceiling-adjacent seeds on a modular recurrence stream are not evidence of intrinsic PC
superiority or long-horizon Ruliad promotion. A 25,730-parameter, 64-update negative control only
found three contexts and failed to learn durable validation behavior, establishing a minimum
capacity/horizon boundary rather than a favorable result.

Exact resume was tested independently with the same 888,194-parameter production profile. A fresh
four-step run was captured after epoch two, then epochs three and four were replayed in a separate
process. The final model, global optimizer, routing bank, context recurrent states, dynamics,
model/config contract, scheduler, stochastic runtime state, source selector, stability state, and
ECS gate/dashboard state all matched byte-for-byte. All 50 post-resume training events also matched
in content and order after removing only run identity and elapsed wall time. Burn's binary context-
optimizer recorder embeds process-local parameter IDs, so those bytes are not canonical across
processes; identical model updates after two resumed optimizer steps provide the semantic parity
check for that state.

The generic P2P stack now has a signed `ContextSparseDelta` codec that binds family, slot,
generation, and dynamic parameter catalog in both envelope and body, rejects stale-generation
aggregation, and compiles on native and WASM. This is protocol readiness, not decentralized-PC
promotion. Dragon will only enable that codec in a distributed training revision after a longer
fixed-holdout Ruliad matrix clears quality, retention, restart, and bandwidth gates.

The committed machine-readable summary is
`docs/experiments/predictive-context-routing-20260805.json`; the full local matrix is
`target/pc-context-routing-1m-three-seed-20260805.json`, and the exact replay pair is under
`target/resume-capture/context-resume-parity-1m-v5-20260805/` and
`target/context-resume-parity-1m-v5-20260805/`.

### 2026-08-04 canonical local-factor implementation

Set `training.algorithm = "predictive_coding"` to select the canonical local-learning path. The
outer learner retains Burn's `Autodiff<B>` model type only for checkpoint and optimizer
compatibility. Each train step immediately takes `model.valid()` and performs factor evaluation,
activity inference, analytic VJPs, and derivative accumulation on `B::InnerBackend`. It never calls
`.backward()` and does not retain a global parameter graph.

```toml
[training]
algorithm = "predictive_coding"

[training.validation]
sampling = "fixed_holdout"
seed = 3509215397

[training.local_predictive_coding]
solver = "synchronous_equilibrium"
learning_schedule = "equilibrium"
prediction_precision = 1.0
factor_reduction = "sum"

[training.local_predictive_coding.inference]
steps = 4
step_size = 0.05
gradient_norm_scope = "per_row"

[optimizer]
name = "adamw" # update transform over local derivatives only
```

The implemented equilibrium schedule is:

1. Run the current shared-weight Dragon layers to initialize one activity per layer.
2. Define a local squared prediction-error factor between every inferred activity and its current
   layer prediction, plus a terminal token cross-entropy factor.
3. Relax unclamped activities with analytic input VJPs for the configured number of local
   inference steps.
4. Evaluate parameter VJPs once at the settled activities, aggregate derivatives from every use of
   Dragon's shared weights, and normalize by the number of supervised tokens.
5. Pass those derivative tensors to Burn's AdamW transform. AdamW only updates parameters and its
   moments from supplied local derivatives; there is no preceding AdamW/backprop training pass.

`solver = "fixed_prediction"` selects a second, deliberately narrower control. It retains the
feed-forward prediction at every shared layer use, initializes the terminal token error, and sends
that error through one reverse sequence of exact local layer VJPs. Contributions from each use are
summed into the same shared encoder, value encoder, decoder, and normalization parameters. This
path still creates no global autodiff graph and never calls global backward. It is nevertheless a
backpropagation-equivalent triangular error solve, not a claim that depth dependencies have become
parallel. The activity-inference `steps`, `step_size`, clipping, and prediction precision are not
used by this solver; the report records one fixed-prediction error wave.

The fixed-prediction control can be applied to either canonical profile with
`config/language/experiments/predictive_coding/pc-fixed-prediction.overlay.toml`.

Activities and errors are batch-local transient state. Checkpoints continue to contain model
parameters and optimizer moments, not an equilibrium trajectory. Layer forwards and local VJPs are
tensorized over batch/token positions. Shared-weight layer uses are aggregated before the update.
The exported train-loss metric is the ordinary feed-forward token cross entropy measured before
activity relaxation; post-inference energy is reported separately when synchronized diagnostics are
enabled. This keeps train-loss comparisons with the backpropagation baseline meaningful.

This exact implementation is deliberately fail-closed. It supports the flat, untied standard
language head; vanilla residual stream; dense short-context linear attention with ALiBi; uniform
full latent fanout; one rollout; no dropout, random scaffold, hierarchy, slow memory, summary
memory, or latent-reasoning recurrence; and detached recurrent rho factors through TBPTT.
Unsupported combinations fail configuration validation instead of silently falling back to global
backprop.

The historical `[training.predictive_coding]` configuration below is a different mechanism: it
corrects recurrent state as an auxiliary inside ordinary global-backprop training. It remains
useful as a control, but is not the canonical local-factor contract. The former
`optimizer.name = "predictive_coding"` spelling has been retired and now returns a migration error;
optimizer transforms are not learning algorithms.

#### Validation contract

The matrix exposed that ordinary Ruliad validation had followed the evolving live source-selection
distribution. That made nominal validation losses incomparable across trajectories. Validation now
defaults to a deterministic `fixed_holdout` stream with its own seed, independent of both the
training seed and live source-selection weights. `live_source_selection` remains an explicit
validation mode. Source-weighted validation and the free-running verifier remain adaptive
diagnostics; they are not the fixed holdout.

An exact-repeat check at four inference steps and `step_size = 0.05` produced identical train loss
2.400160 and fixed-holdout loss 3.177092 in both runs. Mean throughput was 11,240 +/- 61 tokens/s.
The matched CUDA AdamW repeats were not bitwise deterministic: an early logged-loss difference of
about `4.5e-5` amplified to train losses 0.90773 and 1.00839 and validation losses 2.10056 and
2.07079. The holdout examples are fixed; this residual variation belongs to the CUDA global-
backprop trajectory and is why promotion comparisons use multiple seeds. In an earlier diagnostic,
otherwise identical PC `step_size = 0.5` repeats followed the same trajectory through step 95 and
then diverged sharply. The larger step is retained only as an instability ablation; 0.05 is the
profile default.

#### Fixed-token CUDA matrix

Each row is a release-CUDA, three-seed run on NVIDIA GB10: 937,154 shared parameters, 4 shared
Dragon layers, embedding 96, 4 heads, latent width 3,072, batch 32, block 128, and 128 updates
(524,288 train tokens per run).
Adaptive dynamics, continual backprop, and neuron scaling are disabled. Validation uses the same
fixed holdout seed in every arm.

| Arm | Valid loss | Last train loss | Verifier | Partial progress | Tokens/s | Local-PC time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AdamW | **2.078 +/- 0.0325** | **0.944 +/- 0.0919** | **0.0417 +/- 0.0540** | **0.2144 +/- 0.0493** | **56,620 +/- 1,600** | - |
| Local PC, 1 step | 3.235 +/- 0.129 | 2.497 +/- 0.284 | 0 | 0.0625 +/- 0.122 | 21,170 +/- 619 | 2,061 +/- 51 ms |
| Local PC, 2 steps | 3.214 +/- 0.135 | 2.466 +/- 0.236 | 0 | 0.0625 +/- 0.122 | 16,210 +/- 300 | 3,285 +/- 50 ms |
| Local PC, 4 steps | 3.136 +/- 0.0886 | 2.340 +/- 0.215 | 0 | 0.1111 +/- 0.111 | 11,003 +/- 65 | 5,163 +/- 31 ms |

The paired four-step-minus-AdamW delta is +1.058 +/- 0.108 validation loss and -45,613 +/- 1,610
tokens/s. All local-PC event rows report `learning_contract = "local_factor_vjp_v1"`,
`global_autodiff_graph = false`, and zero global backward calls. Median GPU utilization is 92-93%
for PC, versus 79-89.5% for the much shorter AdamW runs, so the PC path is not host-stalled. Its
lower throughput is added local inference/VJP work. At this scale the removal of global backward
does not compensate for repeated local sweeps.

Verifier performance remains near zero because these are short optimization screens, not
reasoning-quality runs. No arm crossed a fatal training gate. Peak host RAM was 8.9-10.9 GB with at
least 113 GB available. The raw analyzed tables are in
`target/local-pc-factor-fixed-final-metric-128x3-20260804/analysis/`; exact-repeat evidence is in
`target/local-pc-fixed-repeat-final-20260804/analysis/`.

**Decision:** retain canonical local PC as an experimental, mechanically verified algorithm; do
not promote it as the default. The next quality experiment must improve the local objective or
schedule and beat this fixed-holdout baseline before a longer continual-learning claim is warranted.

#### Gradient-fidelity and batch-scaling diagnosis

`pc_gradient_fidelity` is an offline diagnostic for the canonical local-factor executor. It runs
one local-PC step and exactly one reference backward pass over the same deterministic, optionally
masked next-token objective. It reports dot product, cosine, norm ratio, relative L2 error,
least-squares rescaling, and elementwise non-negative-product fraction globally and for all nine
PC parameter families. The reference backward is isolated from the training and telemetry paths;
the report records one reference backward while the nested PC step must continue to record zero.

```bash
cargo run -p burn_dragon_language --release \
  --example pc_gradient_fidelity --features train,cuda -- \
  --backend cuda --n-layer 4 --n-embd 96 --n-head 4 \
  --latent-total 3072 --vocab-size 272 --batch-size 32 --block-size 128 \
  --inference-steps 1,2,4,8,16,32,64,128 --step-sizes 0.05,0.1
```

On the 937,154-parameter CUDA geometry, the feed-forward PC and reference masked losses agree
exactly. The derivative comparison identifies finite-relaxation credit latency rather than a
numerical VJP failure: the language-head derivative is nearly exact immediately, while error
reaches the shared layers and embedding only after repeated synchronous activity updates.

| Inference | Global cosine | PC/reference norm | Embedding cosine | Embedding norm ratio | Encoder cosine | Local step ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 x 0.05 | 0.338 | 0.188 | 0.987 | 0.000006 | 0.611 | 318 |
| 32 x 0.10 | 0.700 | 0.403 | 0.734 | 0.0938 | 0.732 | 1,946 |
| 64 x 0.10 | 0.787 | 0.509 | 0.765 | 0.230 | 0.801 | 3,829 |
| 128 x 0.10 | 0.850 | 0.642 | 0.829 | 0.412 | 0.865 | 7,644 |

These synchronized diagnostic times include pre/post energy readbacks and are not production
throughput measurements. `step_size = 0.1` lowers joint factor energy monotonically over this
range. At 0.2, energy rises after 16 steps and directional fidelity regresses; 0.5 is clearly
unstable. More brute-force relaxation therefore improves derivative fidelity only on the slower
stable branch. Removing the default per-row gradient clip does not repair the solver: at 128 steps
the unclipped global cosine is 0.832 versus 0.850 clipped, with nearly identical energy.

A release-CUDA batch sweep used 16 updates per arm at block 128. Event-level throughput peaks at a
different batch for each executor, but larger batches do not recover PC solver cost:

| Batch | AdamW tok/s | PC-4 tok/s | PC-32 x 0.1 tok/s |
| ---: | ---: | ---: | ---: |
| 16 | 46,450 | 10,530 | **2,071** |
| 32 | 55,330 | **11,100** | 2,056 |
| 64 | **58,410** | 10,840 | 1,945 |
| 128 | 58,330 | 10,260 | 1,940 |

All trials stayed inside the host-memory guard; the largest batch-128 PC arm peaked at 17.4 GB
with more than 107 GB available. PC-32 sustained roughly 87% mean and 90% median GPU utilization
in the longer quality screen, so its throughput loss is repeated device work, not CPU starvation.

A matched three-seed, 128-update, batch-16 screen then compared the higher-fidelity PC-32 arm with
AdamW at equal token exposure (262,144 tokens per run):

| Arm | Valid loss | Verifier | Partial progress | Tok/s |
| --- | ---: | ---: | ---: | ---: |
| AdamW | 2.165 +/- 0.0445 | 0.0521 +/- 0.0204 | 0.2109 +/- 0.190 | **44,190 +/- 1,510** |
| PC-32 x 0.1 | 2.146 +/- 0.0402 | 0.1458 +/- 0.0540 | 0.2396 | 1,949 +/- 49 |

The paired PC-minus-AdamW validation delta is `-0.0188 +/- 0.0539`, verifier delta is
`+0.0938 +/- 0.0707`, and partial-progress delta is `+0.0287 +/- 0.190`. This is weak short-run
quality evidence and a 22.7x throughput regression, not a promotion result. Four-step PC remains a
cheap control and 32-step PC remains a fidelity control; neither should become the default. The
next algorithmic gate is a solver or inference schedule that propagates local credit across the
four shared layer uses without dozens of full factor sweeps. It must improve gradient fidelity and
fixed-holdout convergence while retaining materially more of AdamW throughput before any long-run
continual-learning test.

Raw artifacts are under `target/pc-gradient-fidelity-1m-cuda-20260804/`,
`target/pc-gradient-fidelity-1m-cuda-frontier-20260804/`,
`target/pc-gradient-fidelity-1m-cuda-unclipped-20260804/`,
`target/local-pc-batch-throughput-20260804/`, and
`target/local-pc-fidelity-quality-128-b16-3seed-analysis-20260804/`.

#### Fixed-prediction shared-weight control

The follow-up control directly tests whether Dragon's shared weights remove the finite-depth
failure above. They remove duplicated parameter sets, but not the activity graph: each layer use
has a different input, trace, and Jacobian. A synchronous Jacobi solver therefore still advances
terminal credit by roughly one factor per round. Once the error at every use is available, however,
all derivatives target the same shared parameter IDs and can be accumulated before one optimizer
update.

On the same 937,154-parameter, four-use CUDA geometry, fixed prediction matches the exact masked
reference derivative with global cosine `0.99999964`, norm ratio `1.00000285`, and relative L2
error `8.27e-4`. Every active parameter family has cosine above `0.9999992`; the local step reports
zero global backward calls. The default synchronous `4 x 0.05` control remains at cosine `0.338`
and relative L2 error `0.953`. CPU tests match to relative L2 below `1e-4`; the larger CUDA residual
comes from different reduction/aggregation order.

A 16-update release-CUDA batch sweep reports model-step throughput from the ECS event counters:

| Batch | AdamW tok/s | Fixed prediction tok/s | Synchronous PC-4 tok/s |
| ---: | ---: | ---: | ---: |
| 16 | 65,743 | 37,590 | 12,555 |
| 32 | 73,532 | 47,684 | **12,895** |
| 64 | 70,933 | **48,219** | 12,748 |
| 128 | **76,825** | 43,567 | 12,426 |

Fixed prediction retains 62.8% of AdamW's independently best tested throughput and is 3.7x faster
than the best synchronous arm. At matched batch 32 it retains 64.9%. Its peak host use was at most
11.7 GB across the sweep, versus 15.4 GB for AdamW and 20.7 GB for synchronous PC. This is a large
improvement over equilibrium relaxation, but not a throughput win over Burn's fused autodiff path.

The required matched three-seed screen used release CUDA, batch 32, block 128, 128 updates, the
same fixed holdout, and 524,288 train tokens per run:

| Arm | Valid loss | Last train loss | Verifier | Partial progress | Tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| AdamW | 2.095 +/- 0.0120 | 0.999 +/- 0.203 | 0.0104 +/- 0.0204 | 0.182 +/- 0.182 | **55,633 +/- 686** |
| Fixed prediction | **2.086 +/- 0.0100** | **0.994 +/- 0.197** | 0.0104 +/- 0.0204 | 0.0694 +/- 0.107 | 35,305 +/- 881 |
| Synchronous PC-4 | 3.138 +/- 0.0903 | 2.343 +/- 0.216 | 0 | 0.111 +/- 0.111 | 10,904 +/- 366 |

The paired fixed-prediction-minus-AdamW validation delta is `-0.00977 +/- 0.0216` and train-loss
delta is `-0.00516 +/- 0.0504`: convergence parity is achieved at this short horizon. Verifier and
partial-progress samples are too sparse/noisy to support a reasoning-quality claim. Median GPU
utilization was 90-91% for fixed prediction, 83.5-89% for AdamW, and 92.5-93% for synchronous PC;
the fixed path is device-work limited rather than host stalled.

**Decision:** fixed prediction is promoted as a numerical and convergence control for local VJPs,
not as the default learner. It proves that shared-weight gradient aggregation is correct and that
the synchronous activity solver caused the quality deficit. AdamW backpropagation remains the
default because it is about 1.58x faster at matched batch 32 and supports the full Dragon/TBPTT
architecture. A genuinely PC-native promotion still requires a causal/local schedule that avoids
the reverse depth barrier, supports recurrent rho/TBPTT state, and demonstrates long-horizon
continual-learning benefit rather than merely reproducing the backpropagation derivative.

Raw artifacts are under `target/pc-fixed-prediction-1m-cuda-20260804/`,
`target/pc-fixed-prediction-throughput-analysis-20260804/`, and
`target/pc-fixed-prediction-quality-128-b32-3seed-analysis-20260804/`.

### 2026-08 causal-contract correction

The historical runs in this report optimized recurrent state against the same chunk's next-token
targets before predicting that chunk. Those targets are unavailable at deployment, so the rows
below are now classified as an **oracle-target negative control**, not evidence for a deployable PC
inference mechanism. Reproducing that path requires both
`observation_contract = "oracle_next_token_negative_control"` and
`allow_oracle_target_leak = true`.

The default `observed_prefix` contract instead uses only transitions within tokens already observed.
It infers a detached corrected-state teacher, replays the observed chunk from that teacher, and
constrains the ordinary Dragon transition toward the result. The ordinary state, not the inferred
teacher, continues into later chunks. This makes training, validation, and deployment use the same
state-transition path while retaining PC as a causal training signal. The constraint uses a
scale-symmetric relative MSE, activates only outside the squared relative-RMS
`amortization_tolerance`, and samples at most
`amortization_max_state_slots` per recurrent axis.

The prior causal implementation replaced the continuing state with the inferred state. It also
returned the entering state when the first chunk had no recurrent latents, effectively dropping the
observed chunk at selected boundaries. Both behaviors are removed. Historical results below predate
the amortized contract and are not promotion evidence for it.

### 2026-08-04 kernel-path hardening

State inference now runs through a current-weight model view whose parameters have autodiff
disabled. Recurrent-state tensors remain differentiable, but correction no longer constructs
parameter adjoints that are discarded immediately afterward. One detached view is built per train
step and reused by every selected chunk. The `all` state scope is generated from one typed state
mapper and includes fast state, slow rho/sequence/Mamba state, hierarchical slow hidden state,
clocked slow state, and summary memory.

Gradient clipping now defaults to `gradient_norm_scope = "per_sample"`. Batch replication therefore
does not change an individual correction. `global` remains an explicit coupled ablation. Synced
diagnostics report clipping-group mean, maximum, delta RMS, and clipped fraction with one combined
readback per state tensor. Amortization slots use a deterministic rotating stratified sample and
reuse each index tensor across matching layer shapes within a constraint evaluation.

The first per-sample implementation reduced every non-batch axis separately and reached only
58,488 wall tokens/s on the fixed 1M-class screen. Flattening each tensor to
`[batch, features]` before one grouped reduction raises that to 66,571 tokens/s; the coupled-global
control reaches 70,014 tokens/s and AdamW reaches 81,279 tokens/s. The remaining per-sample cost is
small inside the PC correction itself, while the full correction forward/VJP/replay dominates the
end-to-end difference.

A matched 128-update 1M-class CUDA trace measures the hardened every-four path against AdamW:

| Metric | AdamW | PC every four | Increment |
| --- | ---: | ---: | ---: |
| CUDA launches | 426,540 | 501,509 | +17.6% |
| GPU kernel work | 2.552 s | 3.320 s | +30.1% |
| Kernel span | 6.641 s | 7.859 s | +18.3% |
| H2D copies | 290,589 | 343,022 | +18.0% |

The previous implementation added 33% launches and 44% kernel work on the corresponding screen.
Parameter detachment removes about 44% of incremental launches, but the remaining energy
forward/backward and exact corrected-state replay are full model traversals. A fused point update
cannot remove that dominant cost.

Corrected diagnostics also exposed an algorithmic failure hidden by the old RMS implementation.
The old metric computed `sqrt(mean(delta^2) + eps)`, which reported a false `1e-4` floor. The
correct `sqrt(mean(delta^2)).clamp_min(eps)` metric shows that the established
`step_size = 0.01` path changes state by only `1.0e-8` to `1.8e-8` RMS. Clipping is never active,
and energy deltas are around floating-point noise. Raw-step screens from 1 through 30,000 descend
the observed-prefix energy without instability, but this does not repair the learning contract:
with amortization tolerance set to zero, replay reduces the post-first-correction constraint to
`1.7e-18` or less. The recurrent transition washes out the corrected entry state before the
terminal-state teacher is compared to the student.

A three-seed, 128-update tolerance-zero screen confirms that larger corrections do not improve the
short-run learner:

| Arm | Wall tokens/s | Train loss | Valid loss | Verifier | Partial progress |
| --- | ---: | ---: | ---: | ---: | ---: |
| AdamW | **83,118** | **0.2799** | 0.4585 | **0.2188** | **0.4002** |
| PC every four, step 3,000 | 64,675 | 0.3000 | **0.4583** | 0.1667 | 0.3733 |
| PC every four, step 10,000 | 64,502 | 0.3000 | **0.4583** | 0.1667 | 0.3733 |

This short matrix is not a quality promotion test, but it is sufficient to reject step-size tuning
as the missing mechanism. The current observed-prefix implementation is a full-model recurrent
state smoother and clean-input replay auxiliary. It is not layer-local PC, does not parallelize
credit assignment across Dragon layers, and must not be described as a backprop replacement.

A bounded 10M-class fused-attention screen (`4x256`, latent 4096, chunk 512) applies one correction
every four chunks. At batch 16, AdamW reaches 1,499.3 tokens/s and AdamW+PC reaches 1,439.6 tokens/s,
retaining 96.0% of baseline throughput. A batch-48 recheck reaches 1,514.8 and 1,478.4 tokens/s,
respectively, retaining 97.6%. Both batch-48 arms have 96% median GPU utilization, so PC does not
introduce a device-duty stall at this cadence. Absolute throughput remains about 18 times below
this report's historical 28k-token/s fused-path evidence; AdamW backward alone consumes 101.5
seconds over the eight-update batch-48 screen. This is a separate shared Dragon backward-path
regression. These rows establish relative PC cost only and are not production throughput promotion.

## Historical Recurrent-State Abstract Draft

We evaluate causal recurrent-state correction as an amortized teacher for Dragon TBPTT language
training. In the deployable contract, correction uses only an already-observed prefix and the
ordinary Dragon state remains the continuation state used by validation and deployment. The
implementation is backend-resident, avoids discarded parameter adjoints, and adds 2.4% wall cost
at every-four cadence in the current 10M-class batch-48 screen.

Corrected state-delta telemetry changes the algorithmic conclusion. The established inference step
is effectively zero, while much larger stable steps are erased by exact prefix replay before the
terminal teacher state is compared to the student. A tolerance-zero three-seed screen does not
improve verifier or partial-progress metrics over AdamW. The current mechanism is therefore a
recurrent denoising/replay auxiliary, not layer-local predictive coding and not a backprop
replacement. It remains disabled and useful as a reproducible control. A genuine PC follow-up must
expose layer-local Dragon activities and prediction errors directly.

## Historical Recurrent-State Method

The training path under test is Dragon language modeling with TBPTT. In AdamW+PC mode, PC performs
one or more recurrent-state correction steps inside each selected TBPTT chunk, then normal gradient
training updates parameters. The established every-two-chunk state-correction arm uses:

```toml
[training]
tbptt_chunk_size = 64
batch_size = 64

[training.predictive_coding]
enabled = true
mode = "recurrent_state"
state_scope = "core"
backward_mode = "chunked"
parameter_update = "optimizer"
observation_contract = "observed_prefix"
steps = 1
step_size = 0.01
gradient_norm_scope = "per_sample"
apply_every_chunks = 2
amortization_tolerance = 0.05
amortization_max_state_slots = 128
sync_diagnostics = false
```

The every-four variant remains the least expensive recurrent-state control. It is not a promotion
candidate after the corrected step and replay-effectiveness diagnostics.

The state-only control keeps the same state correction but disables parameter mutation:

```toml
[training.predictive_coding]
enabled = true
parameter_update = "state_only_control"
```

The historical `optimizer.name = "predictive_coding"` parameter-transform control has been removed.
It consumed ordinary backpropagated gradients and therefore obscured the learning contract. Old
configurations now fail with a migration message. Use `training.algorithm = "predictive_coding"`
for local factor learning, or `training.algorithm = "backpropagation"` with an ordinary optimizer
for global backpropagation.

Paper-matrix overlays disable adaptive dynamics recovery, continual backprop, and neuron scaling.
Those systems are important continual-learning machinery, but they would confound an optimizer and
state-correction ablation by changing the run after collapse, plateau, or capacity events.

## Causal Amortized Evidence (2026-08-03)

Release CUDA artifacts:

- `target/pc-amortized-relative-mse-128/analysis/`
- `target/pc-amortized-global-cadence-512x3/analysis-every2/`
- `target/pc-amortized-global-cadence-512x3/analysis-every4/`
- `target/pc-amortized-global-cadence-512x3/analysis-every8/`

Matched 512-step conditions use the same profile, batch size 16, TBPTT chunk size 64, three seeds,
and 2,097,152 training tokens per run. Adaptive dynamics, continual backprop, and neuron scaling
are disabled in every arm.

| Arm | Seeds | Wall s | Tokens/s | Last train loss | Last valid loss | Verifier accuracy | Partial progress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AdamW | 3 | 20.71 +/- 0.50 | 101,315 +/- 2,500 | 0.2292 +/- 0.0089 | 0.1594 +/- 0.1044 | 0.2604 +/- 0.0408 | 0.3637 +/- 0.1421 |
| AdamW + PC every 2 chunks | 3 | 37.22 +/- 0.40 | 56,343 +/- 608 | 0.2359 +/- 0.0216 | 0.1021 +/- 0.0101 | 0.2604 +/- 0.0204 | 0.4132 +/- 0.0274 |
| AdamW + PC every 4 chunks | 3 | 28.99 +/- 0.27 | 72,344 +/- 681 | 0.2907 +/- 0.1196 | 0.1194 +/- 0.0460 | 0.3021 +/- 0.0204 | 0.4141 +/- 0.0589 |
| AdamW + PC every 8 chunks | 3 | 25.42 +/- 0.57 | 82,520 +/- 1,881 | 0.3231 +/- 0.1814 | 0.1507 +/- 0.1083 | 0.2083 +/- 0.1242 | 0.3168 +/- 0.1675 |

Paired every-four-PC-minus-AdamW deltas:

| Metric | Mean delta | Interpretation |
| --- | ---: | --- |
| Last valid loss | -0.0400 +/- 0.1239 | favorable mean, inconclusive with three seeds |
| Verifier accuracy | +0.0417 +/- 0.0540 | favorable mean, inconclusive with three seeds |
| Partial progress | +0.0503 +/- 0.1807 | favorable mean, inconclusive with three seeds |
| Wall time | +8.28 +/- 0.30 s | PC is consistently slower |
| Tokens/s | -28,971 +/- 1,940 | PC throughput is 71.4% of AdamW |

Every PC run reports `observation_contract=observed_prefix_amortized` and
`deployment_aligned=true`. The every-two, every-four, and every-eight arms apply 128, 64, and 32
amortization components per run respectively. The earlier local-chunk sparse-cadence artifact at
`target/pc-amortized-cadence-direct-512x3/` applied zero components for every-four and every-eight;
it exposed a cadence phase bug and is explicitly excluded from evidence. Cadence now uses a global
chunk ordinal and selects an observed chunk rather than resetting at each four-chunk block.

PC's lower throughput is additional model work rather than a host-stall signature. The every-four
arm reports 66-68% median GPU utilization, dataloader foreground wait below 0.05%, and zero host
synchronization points in the stage profiler. All 12 runs completed without a fatal gate. Peak host
memory was 6.58 GB, with at least 118.0 GB available throughout the matrix.

This screen does not establish adaptive curriculum behavior: all arms ended at source mean
difficulty zero and capability allowed maximum one. It tests the correction contract and early
optimization only, not whether PC improves an expanding Ruliad frontier or long-run continual
learning.

The one-seed 128-step chronology/control screen also completes without NaN or CUDA faults. It is
not promotion evidence, but confirms that `observed_prefix_amortized` reports
`deployment_aligned=true`, while the explicitly labeled oracle control reports
`deployment_aligned=false`.

## Historical Oracle-Target Evidence (Not Deployable)

Primary runtime profile:

- backend: CUDA
- dataset: `crates/burn_dragon_p2p/deploy/profiles/ruliad-1m.jepa.training.toml`
- model: 4 layers, 128 embedding dim, 4 heads, 512 latent total
- block size: 256
- batch size: 64
- TBPTT chunk size: 64
- optimizer baseline: AdamW
- PC: recurrent-state correction, core state, one correction step, step size 0.01, every other chunk

Raw local artifacts from the initial matrix:

- `target/pc-paper/pc_ablation_all_summary.csv`
- `target/pc-paper/pc_ablation_gpu_20260620123459.csv`
- `target/pc-paper/pc_ablation_2048_gpu_20260620124632.csv`

### 512-Step Three-Seed Matrix

Each run processes 8,388,608 tokens.

| Arm | Seeds | Wall s | Tokens/s | Last train loss | Last valid loss | Source loss | Mean difficulty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AdamW | 3 | 58.83 +/- 0.15 | 142,584 +/- 374 | 0.1410 +/- 0.0085 | 0.4233 +/- 0.1073 | 0.1662 +/- 0.0237 | 5.1479 +/- 0.0041 |
| AdamW+PC | 3 | 76.11 +/- 0.10 | 110,212 +/- 142 | 0.1361 +/- 0.0015 | 0.2915 +/- 0.0096 | 0.1448 +/- 0.0051 | 5.1511 +/- 0.0004 |
| State-only control | 3 | 75.79 +/- 0.05 | 110,687 +/- 79 | 2.3036 +/- 0.6521 | 2.1172 +/- 0.0160 | 2.8364 +/- 0.0356 | 4.9393 +/- 0.0306 |

At 512 steps, AdamW+PC has better mean validation loss and source-selection loss than AdamW, but
costs roughly 29% more wall time. The state-only control does not learn durable validation
behavior.

### 2048-Step Three-Seed Matrix

Each run processes 33,554,432 tokens.

| Arm | Seeds | Wall s | Tokens/s | Last train loss | Last valid loss | Source loss | Mean difficulty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AdamW | 3 | 232.42 +/- 0.49 | 144,368 +/- 306 | 1.0363 +/- 0.0339 | 0.1413 +/- 0.0447 | 1.0524 +/- 0.0131 | 5.6811 +/- 0.0143 |
| AdamW+PC | 3 | 300.32 +/- 0.19 | 111,728 +/- 69 | 1.0405 +/- 0.0142 | 0.1411 +/- 0.0381 | 1.0634 +/- 0.0099 | 5.6815 +/- 0.0141 |

Paired AdamW+PC minus AdamW deltas at 2048 steps:

| Metric | Mean delta | Interpretation |
| --- | ---: | --- |
| Last valid loss | -0.0002 +/- 0.0091 | no meaningful validation advantage |
| Source loss | +0.0109 +/- 0.0171 | no source-selection advantage |
| Last train loss | +0.0043 +/- 0.0201 | effectively tied |
| Wall time | +67.90 +/- 0.31 s | PC is consistently slower |
| Tokens/s | -32,641 +/- 239 | PC throughput is 77.4% of AdamW |

### Fixed-Small TBPTT-256 Ruliad/NextLat Matrix

The fixed-small latent objective screens need explicit multi-chunk TBPTT for PC
to be meaningful. A `block_size = 64`, `tbptt_chunk_size = 64` profile has no
chunk boundary for recurrent-state correction, so the PC follow-up uses
`block_size = 256` and `tbptt_chunk_size = 64`.

Artifacts:

- `target/ruliad-r1-tbptt256-pc-probe128-2048-latest/summary.csv`
- `target/ruliad-r1-tbptt256-nextlat-pc-followup-2048-latest/summary.csv`
- `target/ruliad-r1-tbptt256-nextlat-pc-warm1024-4096-latest/summary.csv`

At 2048 steps, plain JEPA TBPTT improves structurally when PC is enabled:
schema-wrong drops from 0.961 to 0.484 and health rises from 39k PPM to 516k
PPM with about 18% wall-time overhead. Every-chunk PC is much slower and is not
better enough to justify the cost.

On the stronger JEPA+NextLat h2 delayed1024 sparse16 baseline, the result is
mixed. Default PC can improve short-run CE and partial progress in one
2048-step follow-up, and a `warmup_steps = 1024` variant produced a short
verifier/semantic hit. The 4096-step warmup-PC gate did not preserve that hit:
validation CE was worse than the non-PC TBPTT baseline, partial progress tied,
and schema/health improved only slightly.

Current interpretation: PC is useful as a TBPTT recurrent-state stability
screen, especially for plain JEPA, but it is not yet additive enough with the
leading NextLat candidate to promote by default.

The reusable runner now exposes this as a preset:

```bash
scripts/pc_paper_experiments.sh \
  --matrix nextlat-tbptt \
  --out-dir target/pc-paper/nextlat-tbptt-$(date -u +%Y%m%dT%H%M%SZ)
```

## Publish-Grade Experiment Matrix

Use `scripts/pc_paper_experiments.sh` for new runs and `scripts/pc_paper_analyze.py` for
aggregation.

### Main Fixed-Token Matrix

```bash
scripts/pc_paper_experiments.sh \
  --matrix main-fixed-token \
  --out-dir target/pc-paper/main-fixed-token-$(date -u +%Y%m%dT%H%M%SZ)
```

Required arms:

- AdamW baseline
- AdamW+PC established: core state, chunked backward, one correction step, `step_size = 0.01`, every two global chunks
- AdamW+PC cadence candidate: same settings with `apply_every_chunks = 4`

Required seeds and horizons:

- seeds: `20260621,20260622,20260623,20260624,20260625`
- fixed-token horizons: `2048` and `8192` iterations

### Controls

```bash
scripts/pc_paper_experiments.sh --matrix controls
```

Required control:

- state-only PC at 512 and 2048 steps, three seeds

This control prevents misinterpreting transient latent correction as durable learning.

### Fixed-Wall-Clock Matrix

```bash
scripts/pc_paper_experiments.sh --matrix wall-clock --wall-clock-seconds 3600
```

Required arms:

- AdamW
- AdamW+PC every four global chunks

Required seeds:

- `20260621,20260622,20260623`

This matrix is mandatory because PC is slower at fixed tokens. A positive fixed-token result is not
practically meaningful if AdamW wins at equal wall clock by processing more tokens.

### Long Stability Probe

```bash
scripts/pc_paper_experiments.sh --matrix stability --wall-clock-seconds 21600
```

Required arms:

- AdamW
- AdamW+PC every four global chunks

Required seeds:

- two seeds minimum

Primary question: does PC reduce validation regression, output degeneracy, verifier correctness
regression, or source-selection collapse over longer continual training?

### PC Optimizer Appendix

```bash
scripts/pc_paper_experiments.sh --matrix pc-optimizer
```

Required transforms:

- `sgd`
- `momentum`
- `adamw`
- `diagonal_natural`

This section should stay in the appendix unless the optimizer path clearly beats AdamW in both
fixed-token and fixed-wall-clock comparisons.

## Analysis Protocol

Aggregate raw artifacts with:

```bash
scripts/pc_paper_analyze.py \
  target/pc-paper \
  --out-dir target/pc-paper/analysis \
  --baseline adamw \
  --compare adamwpc_every4
```

The analyzer writes:

- `normalized_summary.csv`: normalized legacy summary rows
- `summary_by_arm.csv`: mean and 95% CI by iteration count and arm
- `paired_deltas.csv`: paired seed deltas for AdamW+PC minus AdamW
- `event_run_summary.csv`: final event-stream metrics per generated run
- `source_bucket_summary.csv`: final source-selection bucket telemetry when present
- `gpu_summary.csv`: GPU utilization and power summaries
- `manifest_summary.csv`: trial metadata, command context, git SHA, status, and run directory
- `paper_tables.md`: compact Markdown tables for the manuscript

Primary outcome metrics:

- validation loss at fixed tokens
- validation loss at fixed wall clock
- ruliad source loss
- normalized and mean source difficulty
- verifier correctness and failure rate
- output degeneracy: entropy, max probability, distinct-2, repetition, dominant periodicity

Efficiency metrics:

- tokens/sec
- wall-clock seconds
- PC correction milliseconds
- GPU utilization and power
- energy/token when power samples are dense enough

Statistical rules:

- Use paired seed deltas for AdamW+PC vs AdamW.
- Report mean, standard deviation, and 95% CI.
- Keep fixed-token and fixed-wall-clock conclusions separate.
- Do not claim improvement unless quality and wall-clock results both support it.

## Claim Boundary

Supported by current evidence:

1. Recurrent-state correction can run without pathological CPU transfer on CUDA.
2. Detaching model parameters removes a substantial portion of discarded PC backward work.
3. Per-sample clipping is batch-replication invariant and has an explicit global control.
4. Every-four correction retains 97.6% of AdamW throughput in the current 10M-class batch-48
   screen, although the shared backward path has a separate absolute throughput regression.
5. Raw state correction can descend observed-prefix energy, but replay erases its terminal-state
   teaching signal to numerical noise after the first selected chunk.
6. State-only correction is not a viable substitute for parameter optimization.
7. In the controlled task-ID-free recurrence stream, fixed-prediction PC plus selected sparse routing
   and context-scoped optimizer state is a substantially stronger continual-learning system than
   ordinary dense AdamW across five paired seeds.
8. The same routing and optimizer-state isolation nearly eliminate forgetting under both
   fixed-prediction PC and backpropagation; their matched quality is at parity, while backpropagation
   remains substantially faster.
9. A causal predictive-loss selector with sequential novelty confirmation discovers all four
   contexts without duplicates across three paired CUDA seeds and preserves the routed retention
   result without a family-specific descriptor.
10. Exact local-PC factors carry detached linear-attention rho through TBPTT and match the
    corresponding stateful autodiff/numerical contracts.

Not yet supported:

1. PC derivatives intrinsically improve long-run continual learning over a matched backpropagation
   learner.
2. PC prevents output degeneracy or collapse.
3. PC is worth its throughput cost by default.
4. The first-class PC optimizer path is competitive with AdamW.
5. PC is additive with JEPA+NextLat beyond short-run or single-seed evidence.
6. The historical recurrent-state replay auxiliary is layer-local PC or parallelizes credit
   assignment across Dragon layers.
7. The predictive context bank is ready as a default Ruliad routing policy; production still lacks
   run-scoped subnetwork/optimizer checkpoint integration and a Ruliad holdout promotion matrix.
8. Fixed-prediction PC exceeds matched Backpropagation throughput or establishes a quality advantage
   away from the controlled benchmark's ceiling.

## Acceptance Gate For An arXiv Submission

The paper is publish-grade only after all of these are true:

- the 5-seed fixed-token matrix is complete
- the 3-seed fixed-wall-clock matrix is complete
- the long stability probe has at least two seeds per main arm
- every run has a manifest with git SHA, config overlay, command, seed, hardware/backend, and run directory
- source-selection buckets, verifier metrics, output degeneracy metrics, and gate events are included in the analysis tables
- the conclusion remains neutral unless both fixed-token and fixed-wall-clock results support a stronger claim

Until then, this document should be treated as a reproducible internal report and protocol, not as
a finished arXiv paper.
