# Predictive Coding State Correction in Dragon TBPTT Training

Date: 2026-06-20

## Status

This document is a preprint draft and experiment protocol. It is **not publish-grade yet** because
the current evidence is a small local ablation, not a complete controlled study.

The correct paper framing is a neutral result: predictive-coding (PC) recurrent-state correction is
technically viable in Dragon training, but the current evidence does not show a fixed-wall-clock
advantage over AdamW. The draft should not claim that PC prevents collapse, improves long-run
continual learning, or should be enabled by default until the experiment matrix below is complete.

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

## Abstract Draft

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

## Method

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

The implementation also exposes a coupled conventional parameter-transform control:

```toml
[optimizer]
name = "predictive_coding"
learning_rate = 0.001
weight_decay = 0.0

[optimizer.predictive_coding]
transform = "sgd" # sgd | momentum | adamw | diagonal_natural

[training.predictive_coding]
enabled = true
backward_mode = "chunked"
parameter_update = "optimizer"
```

Despite the historical `optimizer.name` spelling, these transforms consume ordinary backpropagated
parameter gradients; they are not layer-local predictive-coding weight rules. This path is
validated and smoke-tested, but it is appendix material until it has its own controlled matrix. The
main scientific claim remains about PC state correction plus AdamW.

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

Not yet supported:

1. PC improves long-run continual learning.
2. PC prevents output degeneracy or collapse.
3. PC is worth its throughput cost by default.
4. The first-class PC optimizer path is competitive with AdamW.
5. PC is additive with JEPA+NextLat beyond short-run or single-seed evidence.
6. The current recurrent-state replay path is layer-local PC or parallelizes credit assignment
   across Dragon layers.

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
