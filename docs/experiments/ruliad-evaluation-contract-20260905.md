# Ruliad Evaluation Contract And Local Pilot

This is an execution record for the prerequisite gates in
[the training roadmap](../dragon-training-roadmap-20260905.md). It is not a
production promotion or evidence of a state-of-the-art continual learner.

## Scope

- Local CUDA, GB10 shared memory, one process at a time. No live peer deployment,
  browser canary, long training job, auto-width growth or auto-batch probing.
- Rust 1.95.0, NVIDIA driver 580.142, release binaries. NVIDIA exposes neither
  dedicated memory capacity nor a power limit on this GB10; physical RAM is the
  shared-memory safety denominator, not an invented VRAM figure.
- Approximately 1M parameters: four shared-weight Dragon layers, four heads,
  embedding width 96, neuron width 3,072, linear attention, dropout zero.
- The saved-checkpoint comparison uses training seed 13 and the August 12
  decoder-coupled 1,024-update checkpoints. Evaluation panel seeds are independent
  holdouts, **not** additional training seeds.
- The fresh pilot uses 512 updates, batch eight, block 1,024, TBPTT chunk 64,
  streaming data, fixed curriculum feedback and the same conditional objectives.
  PC uses local learning gradients with AdamW parameter updates, not a global
  end-to-end backward pass. Ordinary AdamW is the matched reference.
- This corpus exercises formal proof-action selection. Results do not establish
  coverage of arbitrary mathematics or long-horizon continual acquisition.

## Implementation

1. Teacher-forcing v2 retains full prompts and answers, including targets beyond
   one block. It scores recurrent chunks and bounded row batches, reduces on the
   device, and reads three scalars per row rather than per-token logits.
2. Final chunks are right-padded to a stable shape. Padding is not a prefix reset;
   no future padded position contributes to the score. CPU numerical tests compare
   chunk sizes 1/2/4/32 and row batch sizes 1/2 against full-context scoring.
3. Report mean sequence NLL separately from token-averaged NLL, first-token
   accuracy, whole-sequence accuracy and mismatched-prompt NLL gain.
4. Fixed proof-action panels now honor the requested seed. Panel schema v4 also
   binds the corpus semantic identity. Existing incompatible caches fail closed;
   use a new panel path instead of overwriting historical evidence.
5. Checkpoint suite v7 verifies model tensor identity before and after evaluation.
   Its CLI records effective options, corpus identity and panel seed. The analyzer
   rejects mixed panel, evaluator, teacher-forcing or corpus contracts.
6. Startup model identity is now typed configuration under `training.provenance`,
   default-on for primary fresh/transfer launches. It is not a hot-loop operation.
   Provenance settings do not change the immutable training/resume contract.
7. The modular Python/TOML runner owns only sequencing, safety and evidence.
   Training behavior remains in Rust configs. It rejects unknown settings, removes
   legacy training environment overrides, records sources/inputs/binaries, copies
   executables into evidence bundles, rejects drift and kills process groups on
   interruption, timeout or memory failure. It does not claim to be a production
   scheduler or an allocation-level OOM guarantee.
8. Stateless masked streaming remains rejected unless mandatory, self-contained
   primary objectives cover every step. Coverage checks include startup offsets,
   cadence, algorithm and non-joint terminals. Prompt binding now has its own
   `require_scheduled_update` setting; missing rows emit telemetry and fail before
   the context-only zero-gradient path. Existing optional configs retain their
   behavior and serialized defaults.

## Completed Initial Checks

All four small checkpoint cases completed with source/input identities unchanged.
Each cell is 16 held-out items spanning four difficulty strata.

| Panel | Optimizer | Free token NLL | Mean answer NLL | Free verifier | Typed action verifier | Wall seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 71 | AdamW | 0.45808 | 6.7566 | 0% | 62.50% | 13.27 |
| 71 | PC | 0.45325 | 6.6854 | 0% | 68.75% | 12.27 |
| 72 | AdamW | 0.48281 | 7.4835 | 0% | 56.25% | 13.28 |
| 72 | PC | 0.48074 | 7.4514 | 0% | 68.75% | 13.28 |

Whole-answer teacher-forced accuracy and policy-context free-generation verifier
accuracy were also zero. Peak whole-host physical use was 13,883 MiB (11.1% of
124,611 MiB). These are inference measurements, not training throughput.

Artifacts: `target/experiments/ruliad-checkpoint-contract/`, including separate
`analysis-panel71/` and `analysis-panel72/` reports. They must not be pooled as
independent trained-model replications.

## Initial Training Pilot

The completed v2 pilot predates default-on startup fingerprints. Its parameters
were seeded and eagerly materialized, but the realized initial digest was not
recorded. It is exploratory evidence, not an identity-qualified confirmation.

| Metric | AdamW | PC |
| --- | ---: | ---: |
| Completed updates | 512 | 512 |
| Validation CE at updates 128/256/384/512 | 0.24749 / 0.09603 / 0.07360 / 0.07059 | 0.24937 / 0.09496 / 0.07447 / 0.06911 |
| Final free verifier, 64 items | 0% | 0% |
| Final typed verifier, 32 items | 68.75% | 68.75% |
| Wall seconds, including startup/evaluation/checkpoints | 185.51 | 234.86 |
| Mean sampled GPU utilization | 78.29% | 81.85% |
| Mean / maximum sampled GPU power | 44.97 / 56.90 W | 41.18 / 60.58 W |
| GPU samples below 50% utilization | 6.52% | 3.42% |
| Peak whole-host physical use | 38,964 MiB | 37,804 MiB |

GPU samples were taken every two seconds. They include all phases and are not
model-only CUDA timings. PC took 26.6% longer; this single-seed pilot does not show
a decisive quality benefit. The warm-stream scheduled token count is not a count
of tokens receiving gradients in these structured objectives.

Artifacts: `target/experiments/ruliad-training-contract-v2/`; runs
`runs/proud-zinc` and `runs/rightful-front`.

## Failed Or Interrupted Measurements

- `ruliad-training-contract/`: intentionally interrupted after its resolved probe
  config inherited completion-likelihood scoring instead of residual energy. The
  explicit settings and profile-parity regression test now prevent that mismatch.
- `ruliad-checkpoint-contract-expanded/`: the first 256-item case hit its 600-second
  watchdog before announcing the closed-loop stage. Peak whole-host use was
  13,832 MiB. No complete quality report exists for this case, and the PC arm was
  not launched. Do not present it as a completed 256-item comparison or attribute
  its timeout to closed-loop search without phase evidence.
- `ruliad-training-contract-v3/`: both ordinary 512-update arms completed with
  matching initial tensor identities. The first no-maintenance arm was rejected
  by configuration before GPU allocation; the fourth arm was not started. This
  is not a completed four-arm comparison. AdamW/PC final CE was 0.07017/0.06970,
  typed accuracy 23/32 versus 21/32, and free accuracy zero for both. A new output
  directory preserves this failed matrix while retesting the guarded exception.
- `ruliad-training-contract-v4/`: both maintained-state arms again completed
  (185.51/230.90 seconds, CE 0.07005/0.07007, typed 21/32 versus 20/32, free zero).
  The no-maintenance arm then hit a separate document-context audit during dataset
  construction, before GPU allocation. The coverage contract now lives in the
  training config module and is shared by runtime validation and dataset
  preparation. Real dataset-construction tests cover acceptance and rejection;
  the audit still reports document visibility rather than falsifying it to 100%.
  All 256 policy batch/panel identities in these two completed arms matched
  exactly as 64-bit integers, not float-rounded JSON values.

## Follow-up Matrix

The manifests split large decoder/policy evaluation from bounded search and add a
maintenance-state ablation. A manifest or directory alone is not completion evidence.

- `ruliad-checkpoint-contract-expanded.toml`: 256-item paired decoder/policy audit,
  with stable-shape teacher forcing and explicit per-phase progress.
- `ruliad-checkpoint-closed-loop.toml`: independently budgeted 16-item paired search.
- `ruliad-training-contract.toml`: same 512-update AdamW/PC pilot, now recording
  realized initial identity, plus each optimizer without unused stream-state
  maintenance. Streaming data and within-row recurrent attention remain enabled.

### Expanded Checkpoint Audit: Completed

Fixed panel 73, 256 items, four difficulty strata, evaluation row batch two.
Both reports passed before/after tensor-identity checks and source/input checks.

| Metric | AdamW | PC |
| --- | ---: | ---: |
| Full-context answer token NLL | 0.46365 | 0.46142 |
| Mean answer sequence NLL | 7.02720 | 6.99343 |
| Teacher-forced token accuracy | 77.55% | 77.99% |
| Teacher-forced whole-answer accuracy | 0/256 | 2/256 |
| Unconstrained action verifier, canonical prompt | 0/256 | 2/256 |
| Unconstrained action verifier, policy prompt | 0/256 | 2/256 |
| Typed candidate action verifier | 181/256 | 181/256 |
| Counterfactual-target top-1 change rate | 84.77% | 84.38% |
| Counterfactual-target equivalent probability gain | 0.37926 | 0.37801 |
| Context-swap equivalent probability drop | -0.01863 | -0.01447 |
| Wall seconds | 120.22 | 120.78 |
| Peak whole-host physical use, MiB | 13,045 | 13,034 |

The earlier 16-item panels missed the two PC successes. This is still a large
failure gap between candidate selection and unconstrained output. For example,
an AdamW completion was `g2|a:r1|f|0.0.0` where the verified target was
`g3|a:r0|f|1.0.0`. Fluent action syntax is not a correct grounded action.
Target intervention sensitivity is encouraging, but the negative context-swap
drop means this control did not reduce equivalent-candidate probability on
average. It does not establish robust use of the full proof context.

Artifacts: `target/experiments/ruliad-checkpoint-contract-expanded-v2/`, including
`analysis/`. The completed timing is not a controlled speedup measurement against
the earlier timeout: the invocation and execution conditions changed.

### Bounded Closed-Loop Audit: Completed

Fixed panel 74, 16 items, evaluated separately from the 256-item decoder audit.
The search selects among verifier-enumerated actions, so action validity is
guaranteed by the candidate mechanism, not learned unconstrained generation.

| Metric | AdamW | PC |
| --- | ---: | ---: |
| Complete proofs solved | 4/16 | 5/16 |
| Goal completion rate | 31.17% | 41.56% |
| Top-1 expert-equivalent action rate | 66.37% | 70.02% |
| Mean search steps | 106.88 | 116.75 |
| Repeated-state / backtrack rate | 0 / 0 | 0 / 0 |
| Wall seconds, full evaluation invocation | 61.13 | 83.66 |
| Peak whole-host physical use, MiB | 14,177 | 12,840 |

The PC difference is one solved proof on one trained seed, not promotion evidence.
For AdamW, the closed-loop phase alone took 48.61 seconds: 8.33 seconds CPU
preparation, 25.52 seconds model scoring, 14.75 seconds CPU transitions. This
search has meaningful CPU work; its duty cycle must not be labeled training duty.

Artifacts: `target/experiments/ruliad-checkpoint-closed-loop/`, including `analysis/`.

### Why Test Maintenance Separately?

The policy and full-completion objectives in this particular recipe run on
self-contained rows. They separately advance the streaming rho state without a
weight update; the next structured update does not consume that state. Thus
`tbptt_persist_across_steps=true` is not sufficient evidence that the recipe learns
through cross-chunk state. Removing that maintenance pass is a scoped performance
ablation, not a recommendation to discard rho or TBPTT in a state-consuming learner.

### Final Maintenance Matrix: Completed

`target/experiments/ruliad-training-contract-v5/results.json` reports all four
cases `ok`, `complete=true`, and `source_unchanged=true`. Same release binary,
seed 13, batch eight, 512 updates, block 1,024 and within-row recurrent attention.
Only training algorithm and cross-step state maintenance differ in the saved
training configs. Population sizing does not apply to these AdamW/local-PC runs.

| Algorithm | Maintain unused stream state | Wall seconds | Updates/s | Final validation CE | In-run typed verifier, 32 items | Free verifier, 64 items |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| AdamW | Yes | 182.44 | 2.806 | 0.06957 | 22/32 | 0/64 |
| PC | Yes | 231.93 | 2.208 | 0.07137 | 20/32 | 0/64 |
| AdamW | No | 138.87 | 3.687 | 0.07045 | 21/32 | 0/64 |
| PC | No | 184.75 | 2.771 | 0.06968 | 24/32 | 0/64 |

No-maintenance throughput is 31.37% higher for AdamW and 25.54% higher for PC in
this single paired pilot. Timings include startup, evaluation and checkpoints;
they are not model-only kernel timings or gradient-token throughput. In both
maintenance conditions PC remains slower than the corresponding AdamW reference.

| Algorithm / maintenance | Mean sampled GPU util | Mean / peak sampled power | Samples below 50% util | Peak whole-host MiB |
| --- | ---: | ---: | ---: | ---: |
| AdamW / yes | 75.80% | 44.76 / 57.62 W | 5.49% | 39,975 |
| PC / yes | 81.64% | 41.68 / 57.72 W | 4.35% | 41,783 |
| AdamW / no | 78.46% | 47.29 / 55.10 W | 12.86% | 38,861 |
| PC / no | 82.49% | 42.74 / 63.75 W | 6.52% | 34,106 |

These are two-second samples across all phases, not a CUDA duty-cycle trace.
The maximum physical use was 40.80 GiB, 33.53% of system RAM, with no memory guard
failure. The retained two-checkpoint directories were 95 MiB with maintenance
versus 23 MiB without it, including optimizer and runtime state.

Identity and delivery checks:

- All four realized initial model tensor digests were
  `061573b8ced2980150f8c915177e4b3c5248fa013764217b00a1139e2cb5783c`.
- All 256 ordered policy input/panel identity triples matched exactly. Canonical
  integer-preserving SHA-256, using `scripts.pc_paper_identity.stream_sha`:
  `409546746b374641ef452f2bec0545bff63c72debc6b6cd64a5b1b912cbfd7ff`.
- Each arm delivered 256 policy updates and 256 decoder updates, with no skips.
  Decoder work was exactly 2,048 rows and 30,842 supervised tokens in every arm.
  Decoder telemetry recorded 256 global backward calls for AdamW, zero for PC.
- Initial identity and input equality do **not** imply bitwise-equal final weights.
  Final model digests differ. Repeated maintained-state runs also vary; this turn
  did not isolate the source of that numerical variation.

Run directories in table order: `runs/zealous-animal`, `runs/unaccountable-beds`,
`runs/cautious-dog`, `runs/complex-stream`. Executables, source archives, declared
inputs, guard logs and GPU samples are retained in the matrix evidence bundle.

### Final Independent Checkpoint Holdout: Completed

Panel 75, 16 new items, row batch two, same before/after tensor-identity checks.
All four cases and the common-panel analyzer completed without source/input drift.
This is a different panel from the 32-item in-run typed probe above.

| Algorithm / maintenance | Full-context token NLL | Mean answer NLL | Typed verifier | Free verifier |
| --- | ---: | ---: | ---: | ---: |
| AdamW / yes | 0.51186 | 7.77393 | 9/16 | 0/16 |
| PC / yes | 0.51802 | 7.86746 | 9/16 | 0/16 |
| AdamW / no | 0.50934 | 7.73561 | 6/16 | 0/16 |
| PC / no | 0.51824 | 7.87075 | 9/16 | 0/16 |

The AdamW control loses three typed decisions despite similar token NLL. This
small, single-seed result neither proves a systematic regression nor establishes
quality neutrality. The speedup is not permission to switch production defaults.
The no-maintenance configuration remains an explicit experiment overlay.

Artifacts: `target/experiments/ruliad-maintenance-checkpoints/`, including
`analysis/checkpoint_eval_report.md`. Each case took 12.77-13.27 seconds and peaked
below 13,700 MiB of whole-host physical use. No training or evaluation job remains
running after these bounded matrices.

## Promotion Decision

No production model, optimizer, objective or browser default is promoted here.
Low CE and typed action selection do not establish correct autonomous proof
generation. Model growth, long continual-learning runs and mixed-peer promotion
remain behind the roadmap's complete-task success and state-continuity gates.
The next learning experiment must measure those endpoints, not optimize CE alone.
The [policy-control follow-up](ruliad-policy-controls-20260905.md) subsequently
completed chance, positional, heuristic, no-context, and scorer-component controls
on frozen checkpoints. They do not establish reliable context-conditioned learning.
Remaining gates include multi-seed/repeated quality comparisons, independent proof
checking, an actual state-consuming long-horizon objective, signed real-browser
accepted-work canaries, and matched multi-peer convergence.
The roadmap's browser, heterogeneous-protocol and long-soak phases are not complete.

## Verification

- Focused Rust tests passed for complete-context teacher forcing, seeded panels,
  tensor fingerprints, resume provenance, objective schedule coverage, missing
  required batches, ordinary streaming guards and data-loader boundaries.
- Prompt-value and full-completion local-PC numerical gradient comparisons against
  global backprop passed on the small CPU test models.
- `cargo check -p burn_dragon_p2p --tests --features cuda --locked -j 2` passed.
  This is a native compile check, not a multi-peer convergence result.
- Thirteen Python runner/analyzer tests, the analyzer self-test and
  `git diff --check` passed. Guard tests use mocked memory readings and bounded
  sleeping child processes, never real OOM allocations.
- Separately, all ten Node browser-override/canary-profile tests passed in the
  clean `main` worktree at `18da6001d3c768b1ebd2315169aad35b5cb632aa`. These
  validate routing and the distinction between local smoke and canonical receipt
  policy; they are not a real-WebGPU canary or tests of this research branch.
- Full workspace tests, browser runtime tests, accepted live peer receipts and
  long-horizon multi-seed promotion are not completed by these checks.
