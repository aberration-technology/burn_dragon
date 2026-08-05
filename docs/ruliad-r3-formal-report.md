# Ruliad R3 formal pretraining report

This report describes the formal Ruliad source implemented by
`burn_dragon_universality`, its Dragon training integration, the bounded
evidence collected so far, and the gates that still prevent a frontier-quality
reasoning claim.

## Status

Ruliad R3 is a coherent verifier-backed symbolic proof-training system. It is
no longer a collection of unrelated text templates: six mathematical
frontends lower into one proof IR, one deterministic transition kernel, one
compact wire format, and one family of supervision and evaluation contracts.

That is a useful platform result, not a claim that the model spans all
computable mathematics. Current experiments establish mechanics, local proof
policy improvement, browser/native source parity, and bounded P2P convergence.
They do not establish long-horizon theorem discovery, transfer to an external
proof assistant, or frontier model quality.

## One proof abstraction

The data path is organized around the following contract:

```text
equational   category   logic   automata   process   metagraph
      \          |        |        |          |          /
                    RuliadProofProblem
           terms + axioms + goals + dependencies
                              |
                 deterministic proof kernel
                              |
       problem / action / certificate supervision views
                              |
      structured tokens, TBPTT streams, verifier metrics
```

The frontends provide different algebras and contextual rewrite laws, while
the shared IR carries:

- interned terms and explicit equalities
- named, directed rewrite axioms
- dependency-ordered proof goals
- candidate proof steps with paths and substitutions
- complete or partial certificates
- a complexity vector used for curriculum telemetry

The kernel validates problem structure and replays every transition under
explicit limits. It does not accept a trace because its serialized text looks
plausible. The semantic contract versions the generator, kernel, source
selection, and wire encodings so native and browser peers cannot silently train
different objectives under one revision.

## Training views

R3 supports three complementary formal tasks:

- `advance_proof`: choose or emit a verifier-valid local transition
- `construct_proof`: compose transitions into a complete certificate
- `check_proof`: classify or diagnose a supplied certificate

The default corpus uses `trace_and_answer` token masking. Prompt material is
context; proof/certificate and answer fields carry the next-token objective.
Documents are variable-length streams, and block/TBPTT packing preserves
logical stream identity across chunks. Fixed-size export paths propagate a
target-loss mask, so EOS padding is never learned as useful content.

The action-policy auxiliary is not natural-language chain-of-thought
imitation. Static mode trains verifier actions on source-selected certificate
states. DAgger mode additionally visits states under the current model and
asks the deterministic kernel for the best progress-making candidate.
Holdout rollouts remain evaluation-only. No action-policy objective is a
promoted default yet.

## Difficulty and source selection

Live source selection is a low-cost curriculum over formal source buckets.
Each bucket is identified by family, task, and difficulty. Feedback includes
loss, verifier success, completion health, schema failures, entropy, hash-noise
probability, and per-domain/task mastery.

The production R3 profile starts at difficulty zero, requires measured mastery
before releasing cold start, and lazily appends another frontier level when the
policy concentrates near its current edge. `max_materialized_levels = 0`
means there is no configured curriculum ceiling. It does not allocate an
infinite bucket table: levels are generated only when selected.

Difficulty changes independent proof coordinates rather than merely adding
tokens. It increases rewrite depth, proof leaves, context nesting, dependency
structure, and distractor axioms. Far levels remain safe because every sample
has a finite kernel and document resource envelope. The current formal
generator caps a single proof at 4,096 leaves, cycles context depth, and grows
rewrite depth logarithmically. Consequently, the curriculum index is
unbounded, but the theorem language and every realized problem are bounded.
Calling this the complete mathematical Ruliad would be incorrect.

The source-selection hot path uses deterministic generation, parallel
preparation, bounded caches, and named batch requests/results. Training does
not scan or pre-materialize the frontier. Cold-start and frontier changes are
event-driven rather than per-token policy work.

## Evaluation contract

Cross entropy is reported, but it is not a promotion criterion by itself. R3
also records:

- exact, semantic, and verifier acceptance
- schema-valid-but-wrong, malformed, and missing completions
- partial proof progress and answer-field coverage
- proof-policy solve and goal-completion rates
- valid/invalid actions, repeated states, backtracks, and candidate top-1
- per-difficulty, family, task, domain, and reasoning-mode groups
- output repetition and answer-collapse diagnostics

The proof-policy scorer batches candidate rows across problems on CUDA. Host
threads prepare the next transition wave while a bounded number of GPU batches
are in flight; scalar results are read only at the deferred boundary. The
inline health probe is intentionally small. The larger promotion audit is an
explicit operator action because verifier-coupled transition search remains
sequential between waves.

## Bounded local results

### Historical proof-policy result

The historical seed-1337, 4,096-update CUDA comparison used the same 128-item
completion probe and the same 16-problem, beam-4 closed-loop rollout:

| objective | valid loss | verifier | solve | goal completion | candidate top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CE only | 0.337641 | 0.406250 | 0.187500 | 0.240260 | 0.477273 |
| CE + pre-v8 policy auxiliary 0.25 | 0.347236 | 0.453125 | 0.187500 | 0.344156 | 0.740396 |

This result was originally labelled DAgger, but the old row budget was filled
by the first certificate-state wave. No model-visited state reached the loss.
It is therefore evidence for a static policy auxiliary, not DAgger. The
quality deltas remain valid for that historical implementation, but they
cannot support an on-policy learning claim.

### Corrected policy-distribution smoke

Telemetry version 10 distributes the row budget across rollout depth and
records static, DAgger, and model-visited rows separately. The paired schedule
first learns static expert states, then retains expert rows while adding an
equal number of on-policy rows. Each post-transition update in the seed-1337
smoke contained 16 static rows, 16 DAgger rows, 12 later-depth model-visited
rows, and four batched scoring waves of four trajectories.

The matched 1,024-update CUDA matrix used batch 64, 512-token blocks, a
2-layer width-64 latent-256 Dragon, 128 structural-holdout items, and the same
closed-loop policy probe:

| objective | valid CE | verifier | partial | same-item top-1 | solve | goal completion | rollout top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CE only | 0.194060 | 0.328125 | 0.750000 | 0.328125 | 0.062500 | 0.101124 | 0.814199 |
| static balanced vocabulary marginal 0.25 | 0.180515 | 0.476562 | 0.773438 | 0.476562 | 0.000000 | 0.078571 | 0.525790 |
| static then paired DAgger 0.25 | 0.183553 | 0.351562 | 0.640625 | 0.351562 | 0.000000 | 0.100000 | 0.646859 |

Static supervision improves held-out verifier and same-item accuracy by
14.84 percentage points, but hurts closed-loop search. Paired DAgger recovers
some rollout top-1 and goal completion relative to static supervision while
losing most of its verifier gain. It does not beat CE on solve, goal
completion, or rollout top-1. This is a one-seed smoke and neither policy
objective passes promotion.

### Proof-policy GPU duty

Removing per-chunk host readback and batching rows across problems changed an
earlier matched probe as follows:

| implementation | policy time | total time | observed GPU duty | observed power |
| --- | ---: | ---: | ---: | ---: |
| serialized readback | 27.898 s | 39.554 s | low, visibly bursty | low/bursty |
| tensorized deferred readback | 21.539 s | 33.563 s | 90-93% | 42-45 W |

Policy time fell 22.8% and total probe time fell 15.1% with exact same-seed
quality parity. Pipeline depth two was retained; deeper queues increased
memory without a useful throughput gain.

The semantic-action evaluator subsequently exposed another launch-density
limit. Candidate presentations had ragged prompts, so the prefix-reuse scorer
grouped exact lengths and encoded one four-row cyclic orbit at a time. It now
advances all active rows to the shortest true prompt boundary, removes those
exact recurrent states, and continues the remaining rows. No padding token is
ever applied to recurrent state. A matched release-CUDA sweep used 512 updates,
batch 64, 512-token blocks, the 2-layer width-64 latent-256 Dragon, 16 rollout
problems, and a 32,768-token scoring cap:

| max rows | scoring batches | mean rows | policy time | scored states/s | peak host use |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 370 | 4.00 | 17.732 s | 20.87 | 24.5 GB |
| 16 | 143 | 10.35 | 13.907 s | 26.61 | 23.9 GB |
| 32 | 139 | 10.65 | **13.329 s** | **27.76** | 22.8 GB |

All three arms produced exactly identical rollout records: 0.9375 verifier,
0.92697 partial credit, 0.89730 candidate top-1, and 0.385448 validation CE.
During the selected 32-row probe, one-second GPU samples held 74-95%
utilization and 52-63 W. The profile therefore promotes 32 rows while retaining
the token budget and queue depth two as hard memory bounds.

Free correctness decoding has 104 distinct prompt positions among 128 rows,
with no exact-position group larger than three. Exact-length grouping therefore
cannot provide useful launch density. The recurrent decoder now batches ragged
rows without padding: every row advances at the same absolute position, rows
still in prefill consume their real prompt token, and rows past their prompt
consume the device-side greedy token. Stopped rows are removed from recurrent
state only after a bounded four-token device buffer is resolved.

An initial row-only implementation grouped probes in sample order. It improved
the tiny model but regressed the 10M-class shape because the shortest prompt
forced all longer prompts into hundreds of one-token forwards. The selected
scheduler sorts by `(prompt_length, original_index)` and bounds both cohort rows
and prompt-position span. Generated records are written back by original index.
The 32-token span preserves a large common-prefix forward and hard-bounds the
ragged token-at-a-time tail.

A one-update release-CUDA span sweep at the 4-layer width-256 latent-4096 shape,
batch 48, and 128 exact items produced identical records in every arm:

| decoder/span | mean rows | maximum rows | probe time |
| --- | ---: | ---: | ---: |
| independent, four in flight | 1.00 | 1 | 10.604 s |
| 16 tokens | 5.12 | 13 | **6.761 s** |
| 32 tokens | 8.00 | 20 | 6.776 s |
| 64 tokens | 16.00 | 35 | 7.415 s |
| 96 tokens | 18.29 | 35 | 8.919 s |
| 128 tokens | 25.60 | 40 | 8.933 s |

Span 32 is selected because its latency is within 0.2% of span 16 while it
provides larger cohorts and fewer low-duty samples. Two full matched checks then
confirmed the result:

| shape / decoder | correctness time | validation share | wall tok/s | GPU util | <30% samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| width 64, independent | 2.509 s | 27.48% | 174,504 | 82.54% | 3.0% |
| width 64, unsorted ragged 64 | 1.498 s | 23.92% | 182,898 | 84.21% | 1.1% |
| width 64, sorted span 32 | **1.055 s** | **22.83%** | **183,329** | 85.11% | 2.1% |
| 10M-class, independent | 9.398 s | 6.25% | 16,308 | 90.35% | 2.0% |
| 10M-class, unsorted ragged 48 | 16.886 s | 9.76% | 15,719 | 91.67% | 1.5% |
| 10M-class, sorted span 32 | **5.211 s** | **4.14%** | **16,769** | **92.81%** | **0.0%** |

The width-64 arms emitted 512 normalized records with SHA-256
`c95cc537ef8f4b77afc2ca3ccd08505666ffcdd7a2aba18a3b0e1a05e7c6258d`;
the 10M-class arms emitted 128 records with SHA-256
`fab8796bea7f03105a3881bcd4bcd33708b2c3eedf7c30a207e48aa5a4c42bc5`.
The profile promotes a 64-row ceiling and 32-token span. Cohorts are additionally
capped by training batch size. The highest observed host use was 59.6 GB with
65.0 GB still available, inside the harness's stricter 80% shared-memory guard.

The policy token budget was tested separately at 65,536 tokens and 64 rows. It
changed floating-point rollout decisions (392 versus 370 scored states) while
improving policy latency only 0.6% (12.591 versus 12.665 seconds), so it is
rejected. The exact 32,768-token, depth-two policy contract remains selected.
Its model-scoring phase sustains high accelerator duty; its residual latency is
sequential proof-search depth, not dataloader or optimizer starvation.

Training-side stalls had a separate cause. The loader generated 64 formal
policy metadata samples on every optimizer step even though the objective was
scheduled every fourth step after step 128. Validation generated the same
sidecar even though `ValidStep` never consumes it. Loaders now evaluate the
complete supervision cadence before materialization, and validation leaves
policy metadata disabled.

The same seed, binary configuration, and 1,024-update matrix before and after
the loader fix produced identical quality metrics:

| arm | loader CPU | loader wait | validation | GPU util | >=80% duty | wall tok/s | wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| static, before | 57.96% | 12.43% | 21.57% | 69.12% | 57.14% | 156,661 | 217 s |
| static, cadence-aware | 12.58% | 0.29% | 10.24% | 82.01% | 73.54% | 179,716 | 189 s |
| paired DAgger, before | 56.51% | 12.31% | 21.47% | 76.60% | 67.74% | 155,837 | 217 s |
| paired DAgger, cadence-aware | 12.61% | 0.30% | 10.41% | 83.24% | 75.66% | 180,329 | 189 s |

Foreground loader wait is now at the 0.3% CE-control level. Static wall
throughput improves 14.7% and paired throughput improves 15.7%; both finish
12.9% sooner. The remaining gap to CE is real auxiliary forward/backward and
closed-loop evaluation work. On this tiny model, active power remains about
35 W while active utilization is near 90%; board power alone is not evidence
of a stall.

Validation budgeting had a separate long-horizon scaling bug. Each logical
epoch previously ran `total_steps / log_frequency` teacher-forced validation
batches. A 4,096-update run split into 32 checkpoint epochs therefore ran 256
validation batches 32 times. Validation work grew with both run length and
epoch count instead of remaining proportional to training. The scheduler now
derives the budget from `train_steps_per_epoch / log_frequency`, caps it by the
validation dataset, and never permits an empty validation epoch.

A controlled release-CUDA rerun retained all 32 epoch boundaries, the seed,
training batches, checkpoints, 16 free-generation probes, and eight proof
policy probes:

| validation budget | validation batches | wall | wall tok/s | model duty | validation | loader wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| total-run budget on every epoch | 8,192 | 1,021 s | 132,104 | 41.9% | 44.9% | 1.52% |
| logical-epoch budget | 256 | 789 s | 170,971 | 55.4% | 27.0% | 0.10% |

Wall time falls 22.7% and useful wall throughput rises 29.4%. The 256 logged
training losses agree within `2.4e-7`; learning rates, all eight closed-loop
policy trajectories, free verifier/semantic/partial scores, source
probabilities, capability EMAs, and the materialized frontier are identical.
The final teacher-forced CE values are intentionally not compared because the
fixed validation panel is now smaller. Remaining validation time is dominated
by the explicitly scheduled free-generation and closed-loop proof probes, not
by redundant teacher-forced batches.

The structural-generalization harness now records training, validation, and
checkpoint wall fractions separately. Its bounded inline policy rollout runs
at the final checkpoint, while two free-generation checkpoints preserve the
minimum temporal-collapse signal. A one-step latent model is no longer
evaluated a second time through an identical `eval_step_sweep = [1]` pass.
On the seed-1337 value-balanced smoke, these scheduling changes preserved the
0.328125 verifier rate and 0.75 partial-credit rate while reducing measured
training-schedule wall time from 38.7 s to 20.0 s. Model duty rose from about
18.6% to 39.4%; the remaining 52.8% is explicitly attributed to validation,
not hidden as an apparent dataloader or CUDA stall.

### CUDA batch-density control

An audit-free equal-token control isolated the optimizer/model path from proof
search. Each row processed 8,388,608 tokens with the same 2-layer, width-64,
latent-256 Dragon and 512-token blocks:

| batch | wall tok/s | model duty | active GPU util | active power | >=80% util duty |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 108,940 | 94.2% | 60.3% | 23.5 W | 0.0% |
| 16 | 337,153 | 93.6% | 71.6% | 30.7 W | 0.0% |
| 32 | 439,893 | 94.7% | 79.3% | 35.0 W | 39.1% |
| 64 | 451,935 | 94.9% | 86.8% | 36.6 W | 90.0% |
| 128 | 391,691 | 93.8% | 91.1% | 35.8 W | 80.0% |

Batch 64 is the measured throughput knee for this deliberately tiny model;
batch 128 raises utilization while reducing useful throughput by 13.3%.
Accordingly, board power is not used as the optimization objective. A guarded
larger-shape check (`4x256`, latent 4,096, batch 8) held 87.4% active
utilization and 41.9 W with 92.3% model duty and no low-duty-cycle gaps.
These controls show that the trainer is dense when the workload is large
enough, while the batch-4 formal smoke is intentionally underfilled.

The end-to-end value-balanced formal run at batch 64 retained the verifier
audit and processed 1,024 updates. Active samples covered 96.9% of the trace,
active utilization averaged 93.1%, and 95.7% of all samples were at or above
80% utilization. Model duty was 73.3%, validation was 14.8%, active power was
36.7 W, and peak host use was 15.7 GB. The verifier and partial-credit rates
remained 0.328125 and 0.75, so this establishes execution density but does not
promote the value-balanced objective. The structural matrix selects batch 64
by default only for its calibrated tiny CUDA shape; custom shapes and CPU use
batch 4 unless explicitly overridden.

### Bounded exact-orbit follow-up

Candidate indices are presentation labels rather than proof semantics. Exact
cyclic-orbit supervision originally expanded each of 32 semantic proof states
into four presentations, silently materializing 128 auxiliary rows. The policy
config now has separate semantic and physical row budgets. The fixed profile
retains exact finite-group risk while admitting eight states times four
presentations, capped at 32 tensor rows. Telemetry schema v13 records both
budgets and the analyzer rejects an incomplete orbit or a physical overrun.

On the matched seed-1337, 1,024-update CUDA smoke, this changed exact-orbit
model throughput from 200,489 to 271,694 tokens/s and wall time from 239 to 183
seconds. The CE control completed in 179 seconds at 287,845 model tokens/s.
Closed-loop solve remained above CE (0.3750 versus 0.1875), as did goal
completion (0.3571 versus 0.1854). Active GPU utilization was 92.45% for orbit
and 92.55% for CE. The performance fix therefore removes the expansion cost
without reintroducing low-duty-cycle execution.

A three-seed matched matrix confirmed the direction:

| metric | structural CE | bounded exact orbit | matched delta |
| --- | ---: | ---: | ---: |
| validation CE | 0.3199 | 0.3076 | -0.0123 |
| same-item top-1 | 0.7526 | 0.8047 | +0.0521 |
| policy solve | 0.1875 | 0.4167 | +0.2292 |
| goal completion | 0.2060 | 0.3976 | +0.1916 |
| model tokens/s | 275,799 | 263,397 | -4.50% |
| active GPU utilization | 93.22% | 93.24% | +0.02 points |
| wall time | 183 s | 189 s | +3.28% |

This candidate is not promoted. Free generation still emits only two of four
held-out action labels, and orbit-averaged equivalent NLL regresses slightly
(+0.0079). The next policy experiment must report canonical, mean-orbit, and
worst-presentation correctness separately so group averaging cannot hide a
weak canonical decoder.

That presentation audit is now implemented. Every matrix arm uses the same
complete cyclic-orbit evaluator and reports orbit-average, canonical,
all-presentations (worst), per-presentation, Jensen-Shannon disagreement, and
top-1 consensus metrics. Promotion rejects incomplete orbits and canonical,
worst, or presentation-consistency regressions.

The audit exposed a real failure hidden by the old aggregate. On seed 1337,
structural CE reached 0.7578 orbit-average top-1 but 0.3281 canonical top-1 and
0.0000 all-presentations top-1. Exact mean-orbit supervision raised the
aggregate to 0.8750 and halved JS divergence from 0.2030 to 0.1096, but left
all-presentations top-1 at zero and reduced top-1 consensus. A coefficient-free
distributionally robust alternative now supports `presentation_risk =
"worst"`, minimizing the weakest verifier-equivalent log probability in each
finite orbit. It improves canonical and worst NLL but also leaves the strict
all-presentations gate at zero after 1,024 updates, so it remains ablation-only.

The primary data contract was the more important issue: proof-action CE had no
explicit presentation in its sample specification. Verifier schema v8 binds a
cyclic presentation rotation into each `FormalProof` action sample. Query,
target, oracle hash, expected answer, and verifier interpretation now share
that field, while old schema rows default to canonical rotation zero. The
infinite training stream covers all four rotations without adding another
model forward.

Matched one-seed CUDA smokes isolate the effects:

| condition | valid CE | verifier | canonical top-1 | worst top-1 | worst NLL | orbit JS | model tok/s | wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v14 structural CE, implicit presentation | 0.3158 | 0.3281 | 0.3281 | 0.0000 | 1.6152 | 0.2030 | 280,938 | 183 s |
| v15 structural CE, primary rotation | 0.2891 | 0.3906 | 0.3672 | 0.0000 | 1.5453 | 0.1958 | 281,155 | 183 s |
| v15 primary rotation + worst orbit | 0.2850 | 0.3906 | 0.3672 | 0.0000 | 1.4350 | 0.1714 | 273,538 | 181 s |

Primary presentation balancing improves CE, verifier accuracy, canonical
accuracy, and worst NLL at throughput parity, so it is retained as a semantic
data correction. The robust auxiliary still fails free-output coverage,
same-item NLL, consensus, and strict worst-presentation gates and is not a new
default. These are single-seed directional results, not promotion evidence.

The same bounded objective was measured on a larger 4-layer, width-256,
latent-4,096 Dragon for 256 updates, with the objective active for the final
128 updates:

| batch | wall tokens/s | model duty | active GPU util | active power | >=80% duty | peak host use |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 10,939 | 78.7% | 94.56% | 43.90 W | 90.91% | 31.2 GB |
| 16 | 13,200 | 81.9% | 94.88% | 44.50 W | 96.27% | 34.6 GB |
| 32 | 14,753 | 84.2% | 95.28% | 43.68 W | 97.57% | 40.7 GB |

Batch 32 is the measured throughput winner within the requested cap. Power no
longer rises with batch after the CUDA queue is full, so board watts are not a
valid proxy for useful throughput on this unified-memory GB10 workload.

A current-binary confirmation used the more frequent semantic proof objective
on the same 4-layer, width-256, latent-4,096 shape at batch 48. After its
90-second cold graph-compilation window, mean GPU utilization was 95.16%, the
median was 96%, 99.32% of samples were at or above 80%, and no sample was below
20%. Mean steady power was 51.71 W, peak power was 65.73 W, peak host use was
47.37 GiB, model throughput was 21,063 tokens/s, and foreground loader wait
was 0.013%. The step-128 objective graph transition did not introduce a
recurring low-duty interval.

A matched Nsight Systems trace at batch 32 confirms that this is not a launch
stall: the 77.11-second CUDA interval contains 74.07 seconds of kernel work
(96.06% device busy), no inter-kernel gap above 100 ms, and only 16 gaps above
10 ms. The trace instead exposes 113,571 launches over 64 updates. Elementwise
and reduction kernels dominate more GPU time than matrix multiplication, so
the remaining limit is recurrent-graph kernel granularity and memory traffic.

Enabling Dragon's opt-in custom fused kernels is not a solution on this shape.
It raises active board power from about 43 W to 65.9 W while reducing wall
throughput from 13,614 to 4,472 tokens/s, a 3.04x regression. The standard Burn
CUDA graph-fusion path is selected instead. Promotion analysis now rejects CUDA
trials below 85% active utilization, below 80% high-utilization samples, below
55% measured model duty, or above 2% foreground data wait. Power is reported
but is not a promotion gate.

### CUDA graph-fusion and precision gate

The CUDA backend dependency previously compiled without Burn's `fusion`
feature even though the workspace enabled the generic fusion API. A matched
seed-1337 FP32 comparison used the fixed validation panel, batch 64, 512-token
blocks, and the 2-layer width-64 latent-256 Dragon:

| horizon | backend | wall tok/s | model tok/s | model duty | steady util | steady power | valid CE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | raw CUDA | 241,758 | 285,833 | 84.6% | 94.4% | 34.3 W | 0.3127 |
| 256 | CUDA graph fusion | 258,867 | 314,997 | 82.2% | 93.7% | 37.1 W | 0.3072 |
| 1,024 | raw CUDA | 241,921 | 285,880 | 84.6% | 94.1% | 35.7 W | 0.3000 |
| 1,024 | CUDA graph fusion | 278,252 | 338,174 | 82.3% | 93.6% | 38.4 W | 0.3014 |

At 1,024 updates, graph fusion improves wall throughput by 15.0% and model
throughput by 18.3%. Its cold graph compilation is visible in very short runs,
but amortizes over the longer horizon. The fixed checkpoint audit retains
correctness verifier parity at 0.5000 and reaches 0.5625 proof-policy verifier
versus 0.5000 for raw CUDA. This is a one-seed execution-path gate, not evidence
that fusion improves the learning objective. Workspace CUDA now enables Burn
graph fusion by default.

The promoted default was also remeasured on the larger 4-layer, width-256,
latent-4,096 Dragon at batch 32 over the same 4,194,304 training tokens used by
the earlier density gate:

| backend | wall tok/s | model tok/s | model duty | active util | >=80% duty | active power | valid CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw CUDA | 14,753 | 17,528 | 84.2% | 95.1% | 98.9% | 43.7 W | 0.4652 |
| CUDA graph fusion | 18,450 | 21,294 | 86.6% | 94.4% | 98.6% | 49.2 W | 0.3499 |
| CUDA graph fusion, batch 48 | 19,475 | 22,286 | 87.4% | 95.0% | 98.0% | 49.8 W | 0.3107 |

Graph fusion improves wall throughput by 25.1% and model throughput by 21.5%
on this shape. The validation values are reported for completeness but are not
treated as a quality comparison: the older raw run predates intervening data
and objective changes even though the seed and materialized profile match.
The raw and batch-32 one-item completion probes were schema-valid but wrong;
the batch-48 probe accepted its one item. None of these one-item probes is a
capability promotion.

An equal-token graph-fusion sweep then measured the local batch-size knee with
the auxiliary policy disabled. Every arm processed about 2.1 million tokens:

| batch | wall tok/s | model tok/s | model duty | active util | active power | peak host use |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 19,885 | 21,995 | 90.4% | 95.1% | 49.4 W | 26.0 GiB |
| 40 | 20,118 | 22,017 | 91.4% | 95.2% | 50.2 W | 28.0 GiB |
| 48 | 20,136 | 22,182 | 90.8% | 95.1% | 50.7 W | 31.9 GiB |
| 56 | 20,077 | 21,843 | 91.9% | 95.6% | 51.0 W | 36.6 GiB |
| 64 | 19,963 | 22,063 | 90.5% | 95.3% | 50.7 W | 38.7 GiB |

Batch 48 is the narrow measured winner, but the important result is the broad
32-to-64 plateau: adding memory does not raise useful GPU work once this queue
is full. The full policy objective at batch 48 improves wall throughput 5.6%
over the batch-32 graph-fusion run and peaks at 46.1 GiB host use. Its only
interior sub-80% utilization intervals are 0.99 and 1.93 seconds around
scheduled graph changes; no low-duty interval recurs in steady execution.
The structural matrix therefore selects batch 48 automatically for the exact
CUDA `4,256,8,4096` shape while retaining conservative defaults elsewhere.

A matched 64-step Nsight structural-CE trace isolates execution overhead. Raw
CUDA launches 92,963 kernels with 60.43 seconds of kernel work over a
62.76-second kernel span. Graph fusion launches 58,372 kernels with 45.51
seconds of kernel work over a 48.50-second span: 37.2% fewer launches, 24.7%
less aggregate kernel time, and a 22.7% shorter span. Neither trace contains an
inter-kernel gap above 100 ms. In the full batch-32 graph-fusion run, 98.64% of
active 200 ms samples remain at or above 80% utilization. Its sole material dip
is a one-time 4.06-second compilation when the proof-policy graph first
activates at step 128; it does not recur during steady execution.

Enabling fusion also changed `burn_cuda::Cuda` from the raw Cube backend to a
fusion wrapper. Custom-kernel tests bind explicitly to the raw backend when
testing raw execution, while separate CUDA and WGPU fusion-autodiff contracts
verify forward values and query, value, and decay gradients through the fusion
wrapper. Capability dispatch recognizes both primitive forms.
That audit found and fixed a CUDA dense-causal kernel launch bug: the launch
arguments and grid axes disagreed with the kernel ABI. The corrected kernel has
a new cache identity so CubeCL cannot reuse the stale binary. The complete
kernel suite passes numerically on both CUDA and WGPU after the correction.

### Dense-score execution geometry gate

The short-context linear-attention executor previously split every 512-token
score matrix into two 256-row graphs. That reduced the size of individual
allocations but duplicated query-key work and produced substantially more
elementwise, slice, and launch traffic. The executor now exposes
`dense_score_row_chunk` as part of `SequenceKernelConfig`, defaults it to 512,
and precomputes immutable ALiBi score/state decay tensors for exact block-size
matches up to 1,024 tokens. Variable or larger shapes retain the formula-based
fallback. The cache is runtime state rather than checkpoint state.

A same-binary, equal-token CUDA matrix used the 4-layer, width-256,
latent-4,096 Dragon and about 2.1 million tokens per arm:

| row chunk | batch | wall tok/s | model tok/s | model duty | active util | active power | peak host use |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 48 | 19,843 | 21,684 | 91.5% | 95.59% | 50.79 W | 31.97 GiB |
| 512 + decay cache | 32 | 23,035 | 25,400 | 90.7% | 94.98% | 52.88 W | 26.86 GiB |
| 512 + decay cache | 40 | 23,246 | 25,716 | 90.4% | 95.40% | 53.03 W | 28.91 GiB |
| 512 + decay cache | 48 | **23,311** | **26,103** | 89.3% | 95.23% | 51.65 W | 32.30 GiB |
| 512 + decay cache | 56 | 23,140 | 25,761 | 89.8% | 95.63% | 53.03 W | 37.28 GiB |
| 512 + decay cache | 64 | 22,772 | 25,303 | 90.0% | 95.82% | 53.09 W | 44.71 GiB |

Batch 48 remains the narrow throughput winner. The broad 32-to-56 plateau and
the batch-64 regression again show why board watts and allocated memory are not
optimization targets by themselves. Relative to the exact 256-row control,
the selected arm improves wall throughput by 17.5% and model throughput by
20.4%. Query, value, and carried-rho gradients match the dynamic decay formula
within `2e-5` in the focused autodiff contract.

The full 256-update proof-policy workload confirms that the gain survives the
scheduled graph change:

| executor | wall tok/s | model tok/s | active util | >=80% duty | longest sub-80% interval | peak host use | valid CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 rows, dynamic decay | 19,475 | 22,286 | 95.0% | 98.0% | 1.93 s | 46.10 GiB | 0.3107 |
| 512 rows, cached decay | **22,302** | **26,016** | 94.79% | 98.78% | 1.32 s | 45.98 GiB | 0.3075 |

Both one-item completion probes happened to verify, but one item is not a
capability result. The useful conclusion is execution and loss parity: the new
path is 14.5% faster at the wall, uses no additional peak memory, and does not
regress the fixed validation panel in this matched directional run.

A same-binary, batch-48, 64-update Nsight Systems pair directly measures the
launch topology:

| executor | launches | kernel work | kernel span | gaps >10 ms | gaps >100 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256 rows, dynamic decay | 59,913 | 75.76 s | 79.56 s | 12 | 0 |
| 512 rows, cached decay | 47,633 | 63.84 s | 66.55 s | 9 | 0 |

The selected path removes 20.5% of launches, 15.7% of kernel work, and 16.4%
of the CUDA span. There is no host or data-pipeline stall in either trace; the
remaining power variation follows a workload that alternates large matmuls
with memory-bound elementwise and reduction kernels.

CubeCL's generic CUDA autotuner is deliberately not enabled. A balanced cold
trial reached only 16 updates after roughly two minutes of tuning and
compilation. A subsequent minimal warm-cache trial aborted after 22 seconds
when it selected an invalid `matmul_simple_tma_mma` configuration for mixed
broadcast dimensions. Burn graph fusion remains enabled; CUDA autotuning must
not enter the training happy path until that upstream TMA contract is fixed and
covered on this device.

An all-BF16 backend with FP32 AdamW master parameters was also tested and
rejected. It reached 358,726 wall tokens/s and 431,225 model tokens/s at 1,024
updates, but proof-policy verifier fell from 0.5000 to 0.1875 and correctness
verifier fell from 0.5000 to 0.4688. A mixed FP32/BF16 loss graph additionally
produced non-finite losses and was removed. The supported CUDA training path
therefore remains FP32 graph fusion; reduced precision requires a real
operation-aware AMP implementation and a repeated verifier-parity matrix.

### Decoder-tail geometry gate

The shared low-rank decoder previously evaluated one batched matrix multiply
per head and then reduced the head axis. The decoder is mathematically a single
linear map from the concatenated head-latent axis, so the selected path reshapes
`[batch, heads, time, latent]` to `[batch * time, heads * latent]` and executes
one matrix multiply against `[heads * latent, embedding]`. Population execution
uses the same grouped operation; no environment switch or legacy execution mode
remains. A direct numerical contract compares the result against the headwise
sum and checks activation/weight gradient parity. Population tests cover
single-member equivalence and cross-member isolation.

A matched batch-32, 64-update CUDA control isolated this change before it was
promoted:

| decoder | wall tok/s | model tok/s | active util | active power | >=80% duty | sub-30% active |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| headwise sum | 19,306 | 22,938 | 93.76% | 50.48 W | 95.85% | 0.46% |
| flat linear map | **20,714** | **25,032** | **94.06%** | **53.73 W** | **98.00%** | **0.00%** |

This is a 7.3% wall-throughput and 9.1% model-throughput improvement without an
objective or parameter-layout change. Flattening the encoder projections was
also numerically valid, but reduced wall throughput by 2.8%, so those projections
retain their head-batched geometry. CubeCL global autotuning was rejected after
a fully warmed run reached only 14,620 wall tokens/s. It also exposed an upstream
TMA cache-key defect for mixed broadcast batch dimensions; no local registry
patch or autotune feature remains in the supported build.

The release candidate was then measured over longer, warm samples:

| batch / updates | wall tok/s | model tok/s | model duty | active util | active power | >=80% duty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 / 128 | 20,341 | 24,450 | 83.19% | 94.74% | 53.63 W | 100.00% |
| 48 / 128 | **22,053** | 26,382 | 83.59% | **95.50%** | **54.77 W** | **100.00%** |
| 64 / 96, equal tokens | 22,241 | **26,508** | **83.90%** | 93.72% | 53.06 W | 97.16% |

Batch 64 is only 0.85% faster at the wall and reintroduces sampled utilization
dips, while batch 48 has continuous high-duty execution and higher useful
power. Batch 48 remains the stable operating point for this shape.

Finally, the actual `ruliad-r3.typed-policy.training.toml` schedule was run for
256 updates at batch 48 so that the proof-policy graph was active for the final
128 updates. It sustained 22,433 wall tokens/s and 26,323 model tokens/s with
85.22% model duty, 94.86% active utilization, 54.95 W active power, and 98.92%
of compute-span samples at or above 80% utilization. Only 0.36% of samples were
below 30%, at startup or the 64-step telemetry boundaries; the proof-policy
transition did not create a recurring stall.

The final 64-update Nsight trace contains 57,924 launches and 68.06 seconds of
kernel work over a 72.75-second kernel span. No inter-kernel gap exceeds 100 ms.
Matmul accounts for 40.8% of GPU time, fused elementwise kernels 31.7%, and
reductions about 9.6%. CUDA is queued hundreds of milliseconds ahead during the
main loop, so the remaining 51-58 W oscillation is the alternating compute- and
memory-bound kernel mix, not starvation. Further performance work must reduce
normalization/elementwise traffic or fuse recurrent graph regions while retaining
the current numerical and verifier gates; board watts alone are not a promotion
target.

### 10M phase-accounting confirmation

A later 4-layer, 256-embedding, 8-head, latent-4096 screen rechecked the same
claim at batch 48 and block 512. Across complete 1,024-update runs, the CE and
semantic-policy arms averaged 94.6% and 94.8% sampled GPU utilization. Only
0.5% and 0.3% of one-second samples were at or below 20% utilization, while
98.7% and 99.2% were at or above 80%. The semantic arm's lower reported
`model_duty_fraction` (72.9% versus 77.1%) is host-side stage attribution, not
GPU duty: validation, auxiliary graph construction, and asynchronously queued
work do not map one-to-one onto that timer.

A separate 128-update Nsight Systems run on the same model shape measured
90,916 CUDA launches, 109.169 seconds of kernel work, and a 113.861-second
first-to-last kernel span. Positive inter-kernel gaps totalled 4.693 seconds;
all gaps above 10 ms were in startup except one 18.8 ms interior gap. A 200 ms
external sample during the optimizer loop remained at 95-96% utilization with
stable 2.49 GHz SM clocks and 52-54 W power. The host spent most CUDA API time
in `cuEventSynchronize`, but the trace shows that those calls wait for queued
device work rather than starving the device.

The kernel mix explains why watts remain below a dense tensor-core benchmark:
46.1% of kernel time is generic FP32 matmul, 34.5% fused elementwise work, and
the remainder is primarily reductions, copies, and recurrent-state operations.
The supported interpretation is therefore:

- sampled GPU utilization and inter-kernel gaps measure duty;
- tokens per second measures useful performance;
- board power describes arithmetic intensity and is not itself a stall gate;
- `model_duty_fraction` is retained as train-stage wall accounting and must not
  be reported as device duty.

### Semantic-energy power-density recheck

A later same-binary check repeated the density measurement after adding the
scalar semantic-energy proof-policy head. The release-CUDA workload used four
layers, width 256, eight heads, latent width 4,096, batch 48, and 512-token
blocks. The 256-update run activated semantic-energy supervision at step 128.
It sustained 26,934 model tokens/s with 95.10% active GPU utilization and
53.68 W active mean power. Across the complete process lifetime, including
startup, three synchronous validation boundaries, and teardown, 96.52% of
samples were at or above 80% utilization and 1.74% were at or below 20%.
After removing only startup and teardown, 98.57% were at or above 80% and none
were at or below 20%. The only interior sub-80% samples were isolated validation
boundaries; a 200 ms optimizer-loop window remained continuously at 94-96%.

A matched 64-update control then tested whether the optional custom fused-kernel
path could turn the lower board watts into useful throughput:

| execution | wall tok/s | model tok/s | active util | active power | train loss | valid loss | peak host use |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| supported graph-fused FP32 | **27,199** | **30,460** | 94.19% | 53.85 W | 0.383636 | 0.378996 | 42.43 GiB |
| custom fused kernels | 14,572 | 16,329 | **95.53%** | **64.61 W** | 0.383633 | 0.378962 | **37.44 GiB** |

The custom path is numerically aligned and uses less memory, but it loses 46.4%
of model throughput while drawing 20.0% more active power. It remains disabled.
This ablation is also a direct counterexample to using watts as the optimization
target: the higher-power path performs substantially less useful training work.
Promotion continues to gate sampled duty, contiguous idle gaps, throughput, loss,
and verifier behavior instead.

### Target-conditioned semantic energy

The first semantic-energy scorer was not target-conditioned in practice. A
diagonal prompt/terminal interaction reached strong candidate-presentation
scores, but an exact intervention that held the current state and four actions
fixed while changing only the verifier-valid goal produced zero preference
change and effectively zero JS divergence. Increasing the auxiliary to 32 rows
on every update did not repair it and reduced tiny-model throughput to 34.5% of
CE. That full-rate arm remains a mechanistic diagnostic, not a training default.

Two data-contract faults were found while isolating the failure. Paired targets
were independently rotated by the balanced-presentation schedule, leaking a
menu-position cue. The compact query also discarded the rewrite location and
placed the destination before a long candidate menu; valid outcomes at different
paths could therefore serialize the same local target and leave no usable target
signal near the decode boundary. Counterfactual variants now inherit the base
rotation exactly, and proof queries end with the verifier-derived destination and
`at=` rewrite path. This byte-level change bumps the Ruliad semantic contract to
version 9 and the generator semantics identity to
`burn-dragon-ruliad-semantic-proof-action-generator-v8`.

The scorer is now a task-neutral rank-64 query/key compatibility head. Separate
prompt and candidate projections form a learned low-rank bilinear map before the
scalar score. A CPU numerical gate uses one real formal proof state, one unchanged
four-action menu, and two opposite verifier-valid goals; it requires the model to
overfit both decisions and verifies that the target difference remains near the
decode boundary.

A matched seed-1337 release-CUDA screen used 1,024 updates, batch 64, 512-token
blocks, and the 2-layer width-64 latent-256 Dragon. This is one-seed directional
evidence, not promotion evidence.

| metric | structural CE | sparse query/key | full-rate query/key |
| --- | ---: | ---: | ---: |
| alternate-target top-1 | 0.4219 | **0.6094** | **0.7344** |
| alternate-target probability gain | 0.0000 | **0.4241** | **0.5836** |
| alternate-target preference change | 0.6172 | **0.9531** | **0.9609** |
| alternate-target JS divergence | 0.0000 | **0.5739** | **0.6472** |
| worst-presentation top-1 | 0.0000 | **0.8438** | 0.8203 |
| closed-loop solve | 0.1250 | **0.6250** | **0.8750** |
| model tokens/s | **416,593** | 362,136 | 145,536 |
| active GPU utilization | 92.78% | 91.82% | **93.10%** |
| wall time | **132 s** | 142 s | 331 s |

The sparse schedule retains the causal target signal at 86.9% of baseline model
throughput and 92.9% of baseline wall-token throughput. The full-rate schedule
buys additional target accuracy at disproportionate cost and is rejected. The
sparse arm is not yet promoted: free generation remains weak after the revised
query schema, the broad context-swap probability drop is below threshold, and
the result has only one seed.

A separate 256-update 10M-class systems screen used four layers, width 256,
eight heads, latent width 4,096, batch 48, and 512-token blocks. The sparse arm
sustained 95.33% active utilization, 56.65 W active power, 96.31% of samples at
or above 80% utilization, 1.48% at or below 20%, and 0.02% foreground-loader
wait. It delivered 28,655 model tokens/s versus 32,198 for CE, peaked at
46.6 GB unified-memory use, and retained 78.0 GB available. These measurements
rule out a host/data stall in the active training path; lower watts on the tiny
screen reflect workload shape rather than missing GPU work.

In the earlier four-layer scalar-head screen, the structural CE arm reached
0.4062 free verifier and 0.4375 policy solve. Semantic static supervision improved candidate-orbit top-1 from
0.7422 to 0.8594 and policy solve from 0.4375 to 0.6875, but free verifier fell
to 0.0469 and model throughput fell from 30,184 to 23,179 tokens/s. Width alone
therefore does not repair the mismatch between whole-candidate energy scoring
and autoregressive serialization.

An atomic four-token action-pointer vocabulary was also tested as a scoped
counterfactual. At the matched tiny 1,024-update screen it produced valid
one-token actions, but verifier remained flat (0.3984 versus 0.4062), policy
solve regressed (0.2500 versus 0.3750), valid CE worsened by 0.2687, and model
throughput fell 13.9%. The tokenizer/profile experiment was removed rather than
retained as an unsupported mode. This rejects token granularity as the primary
failure mechanism and keeps the production vocabulary/checkpoint contract
unchanged.

### Corrected causal-policy and horizon decision

The initial semantic-energy screen was subsequently superseded by a stricter
corpus and evaluator. Semantic expert rows now use the same proof-action answer
contract as the policy probe, and the promotion gate uses an exact intervention
that changes only the verifier-valid target while retaining the laws, current
state, and candidate actions. A context swap remains a stress diagnostic because
the borrowed state need not make the retained action menu applicable.

A corrected three-seed release-CUDA matrix used 1,024 updates, batch 64,
512-token blocks, and the 2-layer width-64 latent-256 Dragon. Values below are
means over seeds 1337, 2027, and 9001.

| metric | semantic CE | sparse semantic energy |
| --- | ---: | ---: |
| valid CE | 0.2681 | **0.2629** |
| free verifier | **0.0391** | 0.0365 |
| same-target top-1 | 0.8099 | **0.8177** |
| closed-loop solve | 0.2083 | **0.4792** |
| goal completion | 0.3652 | **0.5581** |
| rollout expert top-1 | **0.7657** | 0.6900 |
| exact alternate-target top-1 | 0.1146 | **0.6719** |
| alternate-target probability gain | 0.0019 | **0.4378** |
| alternate-target preference change | 0.1068 | **0.9505** |
| alternate-target JS divergence | 0.0002 | **0.5852** |
| wall tokens/s | **274,636** | 217,457 |
| model tokens/s | **394,800** | 344,544 |
| active GPU utilization | 91.40% | **91.41%** |
| active power | 40.61 W | 40.71 W |

The energy head learns a real target-conditioned decision rule, but it does not
pass the runtime policy contract: one seed solves only 31.25% of closed-loop
problems, mean goal completion is 55.81%, and rollout expert agreement regresses.
Free generation remains collapsed at roughly 3.7% verifier accuracy, 2-3%
distinct answers, and about 80.5% dominant-answer frequency. The candidate is
therefore not promoted.

The exact intervention also caught a misleading shortcut. A static
completion-likelihood policy reached 81.25% solve and 89.89% goal completion in
a one-seed screen, but alternate-target top-1 remained 10.16%, probability gain
was only 0.28%, and JS divergence was effectively zero. Those apparent gains are
target-independent and are rejected by the causal gate.

A true paired-DAgger screen fixed a planner defect that had previously allocated
no model-visited state under the eight-row budget. The corrected planner emitted
two static rows and a depth-two on-policy trajectory, with verifier relabeling of
the visited state. Against static semantic energy at seed 1337, paired DAgger
improved solve from 56.25% to 62.50%, goal completion from 60.11% to 70.79%, and
rollout expert top-1 from 67.82% to 85.94%. It simultaneously regressed
same-target top-1 from 84.38% to 71.09% and exact alternate-target top-1 from
66.41% to 56.25%. Both arms retained about 91% active GPU utilization and
completed in 156-157 seconds. The dedicated semantic-energy DAgger profile was
removed after this rejection; the general paired-DAgger planner, telemetry, and
budget validation remain covered for other experiments.

Finally, a 4,096-update seed-1337 diagnostic crossed the 2,048-step live-source
hold and evaluated the same fixed panels at intermediate and final checkpoints.

| metric | semantic CE | sparse semantic energy |
| --- | ---: | ---: |
| valid CE | 0.2702 | **0.2687** |
| free verifier | **0.1016** | 0.0859 |
| same-target top-1 | 0.7812 | **0.8438** |
| closed-loop solve | 0.3125 | **0.5625** |
| goal completion | 0.4326 | **0.6067** |
| exact alternate-target top-1 | 0.1172 | **0.7578** |
| alternate-target probability gain | 0.0034 | **0.5501** |
| free answer dominant fraction | **0.4297** | 0.8047 |
| model tokens/s | **393,902** | 340,687 |
| wall tokens/s | **298,205** | 235,314 |
| active GPU utilization | 92.25% | 92.24% |
| samples at or above 80% utilization | 94.49% | **95.47%** |
| samples at or below 20% utilization | 1.32% | **0.70%** |
| foreground loader wait | 0.13% | **0.10%** |

More updates strengthen exact target causality but do not close the autoregressive
or closed-loop contract. The live scheduler correctly remains on difficulty zero:
the baseline ends with verifier EMA 0.64 and the energy arm only reaches 0.775,
below mastery. The arm is rejected as a production default rather than extended
with more scalar weighting or on-policy rows. The next objective must connect
semantic action selection to the deployed decoder without sacrificing
autoregressive coverage.

These runs also resolve the apparent power stall. During optimizer phases the
tiny model sustains 92% active utilization at a 2.5 GHz SM clock; low 39-41 W
board power is the workload's memory-traffic-heavy operating point. The visible
dips are bounded synchronous validation and ragged closed-loop proof search, not
data loading. Larger 10M-class screens raise active power to roughly 57 W while
retaining 95% active utilization. Promotion gates therefore use utilization,
idle-sample fraction, the longest contiguous low-duty and idle gaps, throughput,
loader wait, and quality rather than a raw watt threshold. Startup and teardown
are excluded from the contiguous-gap window; a CUDA trial fails if it remains
below 80% utilization for more than ten consecutive one-second samples or at or
below 20% for more than five.

### Detached score-head and GPU-duty closure

The semantic-energy comparison initially had two hidden reproducibility faults.
Constructing the optional score head consumed the backend RNG stream, and Burn's
autodiff-to-valid conversion also advanced that stream. Consequently, a nominally
matched CE arm could see a different dropout sequence even when the policy head
did not update the shared Dragon. The score head now uses deterministic host-side
initialization, and every optimizer step reseeds named main, proof-policy, and
verifier-policy stochastic streams independently. In `score_head_only` mode the
shared hidden trajectory is computed through the valid model and detached; only
the semantic score head receives gradients.

Focused numerical tests require a head-only optimizer update to change semantic
scores while leaving every language logit exactly unchanged. Repeated detached
scores must be identical, explicit reseeding must recover the expected stream,
and adding the optional head must preserve both shared parameters and the next
backend RNG sample.

A corrected seed-1337, 1,024-update release-CUDA screen used the 2-layer,
width-64, latent-256 Dragon, batch 64, 512-token blocks, and the fixed 128-item
structural holdout:

| arm | valid CE | free verifier | same-target top-1 | alternate-target top-1 | target probability gain | solve | goal completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| semantic CE | 0.260836 | 0.046875 | **0.8125** | 0.1328 | 0.0014 | 0.1875 | 0.3427 |
| sparse detached head | 0.260836 | 0.046875 | 0.6328 | **0.6094** | 0.2050 | 0.3750 | **0.5955** |
| full-rate detached head | 0.260836 | 0.046875 | 0.8047 | 0.5078 | **0.2139** | 0.3750 | 0.3041 |

Exact CE and free-generation parity demonstrate that detached training no longer
perturbs the deployed language model. Sparse exposure passes the one-seed 0.60
counterfactual-target floor but loses too much ordinary same-target ranking.
Full-rate exposure recovers same-target ranking but falls below the causal floor,
regresses goal completion, and takes 187 rather than 125 seconds. Both remain
mechanistic diagnostics; neither is promoted.

Stage profiling now records auxiliary-objective and proof-policy time separately.
The prior `model_tokens_per_second` denominator omitted those forwards and could
therefore make an auxiliary arm look faster than CE. The corrected objective
throughput includes the main forward, auxiliary construction/forwards, and the
combined backward; `main_model_tokens_per_second` is retained only as a diagnostic.

A matched 256-update systems run rechecked the sparse detached schedule on the
4-layer, width-256, latent-4096 shape at batch 48:

| arm | wall tok/s | objective tok/s | auxiliary wall | proof-policy wall | active util | active power | mean/min SM MHz | max C | peak host use |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| semantic CE | **26,891** | **30,460** | 2.27% | 0.00% | 95.10% | 54.42 W | 2476 / 2450 | 72 | **34.77 GB** |
| sparse detached head | 25,556 | 28,911 | 12.93% | 8.35% | 95.09% | 56.38 W | 2466 / 2444 | 74 | 36.41 GB |

Every active telemetry sample in both arms remained at P0. Each arm had only one
isolated in-window sample below 80% utilization and no in-window idle sample.
Foreground loader wait was 0.02%. The 5.0% wall-throughput cost is therefore real
auxiliary compute rather than a pipeline stall. GPU sidecars now retain P-state,
SM clock, memory-engine utilization, and temperature so later power complaints can
distinguish low occupancy, clock throttling, thermal throttling, and workload mix.
The GB10 reports zero memory-engine utilization even during dense training, so that
unsupported counter is telemetry only and is not a gate.

Finally, an equal-horizon 64-update batch sweep measured the useful throughput
knee rather than optimizing board watts:

| batch | wall tok/s | objective tok/s | active util | active power | min SM MHz | peak host use |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 48 | 27,005 | **30,146** | 94.26% | 53.71 W | 2463 | **33.63 GB** |
| 64 | 26,963 | 29,749 | 94.55% | 53.57 W | 2431 | 64.24 GB |
| 80 | 26,724 | 29,606 | 95.38% | 55.21 W | 2450 | 49.10 GB |
| 96 | **27,061** | 29,964 | 95.93% | 56.10 W | 2450 | 59.01 GB |

Batch 96 buys 1.67 utilization points and 2.39 W but loses 0.60% objective
throughput and uses 75.5% more host memory. Batch 48 remains the selected
operating point for this shape. The remaining opportunity is kernel arithmetic
intensity and fusion, not host queuing, batch inflation, or a higher power target.

### Longer presentation-risk diagnostic

A matched seed-1337 diagnostic extended CE, mean-orbit risk, and worst-orbit
risk to 4,096 updates. This is one-seed directional evidence, not promotion
evidence. All arms used the same final structural holdout panel and exact
cyclic-orbit evaluator.

| objective | valid CE | verifier | orbit top-1 | presentation top-1 | canonical top-1 | worst top-1 | worst NLL | policy solve | goal completion | model tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CE only | 0.3104 | 0.3594 | 0.8516 | 0.3262 | 0.3125 | 0.0000 | 1.9885 | 0.2500 | 0.1250 | 280,679 |
| mean orbit | 0.3145 | 0.3203 | 0.7031 | 0.2832 | 0.2891 | 0.0000 | 1.8343 | 0.5000 | 0.5062 | 266,263 |
| worst orbit | 0.3153 | 0.3516 | 0.7656 | 0.2832 | 0.2422 | 0.0000 | 1.5005 | 0.1875 | 0.1313 | 266,682 |

Mean-orbit risk improves the orbit-averaged closed-loop search while
regressing free decoding and every direct presentation metric. Worst-orbit
risk substantially improves worst-presentation NLL and reduces orbit JS
divergence, but collapses to one dominant free action and does not preserve
its epoch-4 closed-loop gain. Neither objective is promoted. The result argues
against hiding a positional action contract behind orbit averaging; the next
policy representation should bind output to action semantics directly.

The run sustained 92.5-92.8% mean GPU utilization, 93.3-93.5% active
utilization, and 97.6-98.6% samples at or above 80% utilization. Foreground
loader wait stayed at 0.51-0.57%. Mean active power remained about 37 W. This
is dense, memory-traffic-heavy recurrent execution rather than a host stall.

The diagnostic also exposed that older epoch-to-epoch reports sampled a new
deterministic validation subset each epoch. Final paired arm comparisons were
still valid, but those changing subsets were too weak for continual-regression
claims. Validation probes now retain a fixed sample panel per source bucket
while live source selection remains free to change the represented bucket
mixture. This adds no model work and makes per-bucket trajectories comparable.
A release CUDA smoke emitted the same ordered 32-item panel at epochs two and
four (matching SHA-256 fingerprint) while the measured verifier rate changed
from 0.5000 to 0.4375, confirming that subsequent deltas measure the model on
the same proofs rather than probe resampling.

### Typed semantic-policy gate

Semantic-action experiments exposed two distinct capabilities that must not be
collapsed into one verifier number. Unconstrained autoregressive generation
must choose and serialize an action from scratch. Typed proof-policy inference
instead scores the finite semantic action set enumerated by the proof kernel,
selects in semantic candidate order, and copies the selected candidate's exact
serialization. The verifier still checks and applies every transition; the
model is responsible for the decision rather than syntax generation.

`burn_dragon_language::api::formal_policy` now exposes this typed contract as
`select_ruliad_proof_actions_batch`. It accepts canonical, balanced, or complete
cyclic-orbit presentations, uses the same tensorized scorer as training and
evaluation, maps scores back to semantic order, and returns both the selected
semantic index and deterministic completion tokens. Row bounds alter only
launch geometry, not the decision; focused numerical tests cover serialized
versus tensorized parity and rotated-candidate rendering.

A preregistered release-CUDA matrix used three matched seeds, 1,024 updates,
batch 64, 512-token blocks, the 2-layer width-64 latent-256 Dragon, fixed
structural holdouts, and complete four-presentation orbits. The candidate used
eight static semantic states every second optimizer update after step 128:

| metric | structural CE | typed semantic policy | matched delta |
| --- | ---: | ---: | ---: |
| closed-loop solve | 0.3750 | **0.8750** | +0.5000 |
| goal completion | 0.2453 | **0.9101** | +0.6648 |
| rollout expert top-1 | 0.6741 | **0.8844** | +0.2103 |
| canonical top-1 | 0.4115 | **0.8411** | +0.4297 |
| worst-presentation top-1 | 0.0000 | **0.8255** | +0.8255 |
| presentation consensus | 0.4154 | **0.9831** | +0.5677 |
| free-generation verifier | **0.4062** | 0.0286 | -0.3776 |
| model tokens/s | 377,489 | 314,125 | -16.8% |
| active GPU utilization | 92.96% | 92.03% | -0.93 points |
| foreground loader wait | 0.12% | **0.10%** | -0.02 points |

Every candidate seed passes the configured closed-loop runtime gate and reaches
at least three times four-way chance on orbit, canonical, and worst-presentation
top-1. The typed-policy gate therefore passes. Free generation still fails
coverage, correctness non-inferiority, and the structural-verifier guard, so
the combined promotion gate remains false. This distinction is deliberate: it
promotes the verifier-guided proof tool without claiming a healthy free-text
proof decoder.

The selected long-run profile is
`ruliad-r3.typed-policy.training.toml`. It retains automatic batch sizing,
continual-learning controls, fixed validation panels, and runtime proof-policy
gates from the production stack. The experimental field-binding contrast is
not included: one-seed directional runs reduced free verifier accuracy to
3.9-7.0% without improving the already strong constrained policy, so it is
rejected rather than accumulated as another weighted auxiliary.

### Production validation duty cycle

The optimizer path was not the source of the reported low-power intervals. A
fresh 10M-class Nsight Systems trace recorded 25,132 kernels and 29.10 seconds
of kernel work over a 31.20-second launch span. After the first five seconds of
CUDA module and JIT warmup, kernel-stream duty was 95.42%, the longest gap was
23.01 ms, and there was no gap at or above 100 ms. Matmul accounted for 45.4%
of kernel time and fused elementwise work for 33.5%. Host-side
`cuEventSynchronize` time was therefore waiting for queued device work rather
than starving the GPU.

The visible long interval instead came from the synchronous verifier-backed
closed-loop policy audit. On the matched tiny release-CUDA control it scored
591 proof states in 15.20 seconds, of which 14.50 seconds was model scoring.
The policy-probe schedule now separates that production audit from the cheaper
same-item and counterfactual constrained-action diagnostics. Existing ablation
profiles retain their old matched cadence; the selected long-run typed-policy
profile runs constrained diagnostics every four validation epochs and the
closed-loop audit every sixteen.

A same-binary seed-1337 screen changed only the closed-loop cadence. Both arms
used 1,024 updates, batch 64, 512-token blocks, a two-layer width-64
latent-256 Dragon, and identical fixed validation panels.

| closed-loop cadence | instrumented wall | wall tok/s | validation wall | constrained scorer | active util | active power | >=80% samples | longest sub-80% streak |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 epochs | 128.77 s | 260,572 | 22.99 s | 2.531 s | 92.95% | 39.53 W | 86.47% | 8 s |
| 16 epochs | **116.08 s** | **289,055** | **8.25 s** | 2.539 s | **93.43%** | 39.68 W | **90.08%** | **6 s** |

The split removes 64.1% of validation wall time in this four-epoch screen and
improves whole-run throughput by 10.9%. It does not alter learning: final
validation CE differed by 2.8e-9, and both the counterfactual-target top-1 rate
(0.1250) and probability gain (0.005733) matched exactly. The remaining short
validation dips are expected because free generation and constrained scoring
are still synchronous. The external evaluator path below removes those audits
from the trainer instead of merely making them less frequent.

### External formal evaluator and trainer duty

Formal Ruliad evaluation is now a side-effect-free model operation, and the
P2P role boundary is explicit. Trainers and reducers emit teacher-forced
metrics only. A read-only validator follows `LatestPromoted`, materializes the
exact signed head, evaluates the fixed formal panel, persists the resulting
`HeadEvalReport` and `EvalProtocolManifest`, and publishes the metric cursor.
The evaluator rejects heads from a different network, study, experiment, or
revision. It never mutates model, optimizer, scheduler, source-selection, or
promotion state.

A matched release-CUDA screen compared synchronous trainer-side audits with
the trainer side of this external-evaluator configuration. Both arms used seed
1337, 128 updates, batch 48, 512-token blocks, 3,145,728 training tokens, and
the same four-layer width-256 latent-4,096 Dragon. Dynamics gates and width
scaling were disabled so that only evaluator placement differed.

| evaluator placement | wall | wall tok/s | model tok/s | model duty | validation wall | active GPU util | active power CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| synchronous trainer | 163.55 s | 19,234 | 29,948 | 64.23% | 53.57 s | 93.04% | 9.87% |
| external evaluator trainer arm | **113.20 s** | **27,790** | 29,852 | **93.09%** | **3.72 s** | **94.89%** | **5.27%** |

Removing formal audits from the trainer improved useful wall throughput by
44.5%, raised model duty by 28.87 percentage points, and removed 93.1% of
trainer validation wall. Hot-path model throughput remained within 0.32%,
which isolates the gain to scheduling rather than a changed kernel. The active
power coefficient of variation fell by 46.6%; mean board power was slightly
lower because verifier scoring is real GPU work but is no longer charged to
the trainer. Useful-token duty, not watts alone, is the operating target.

Learning parity passed. Final train CE (0.321621), validation CE (0.242781),
output entropy (0.943546 bits), dominant-period diagnostic (0.850622), and the
generated-output probe matched. Burn checkpoint files contain random
per-process parameter IDs; after normalizing only those IDs, the final model
records have the same SHA-256 digest. Source-selection capability feedback is
intentionally absent from the trainer-side state in the external-evaluator
arm and is emitted as run- and head-keyed validator telemetry instead.

The companion native two-peer gate now exercises pre-promotion
validator-quorum mode. It trains one real Ruliad window, pins the validator to
the canonical genesis, transfers the candidate to the non-training validator,
runs the fixed formal evaluator, and verifies exact decoded-tensor equality
after promotion. The reduction certificate binds the exact head, artifact,
evaluation protocol, and content-addressed report; the report is retrieved by
that complete binding and its identity is recomputed. Generic native coverage
also requires two validators to produce distinct reports over the same
head/artifact/protocol and rejects coordinator claims without visible backing
reductions. A cross-revision head is rejected. This is bounded local evidence
for role isolation and exact-head reporting; network-coupled heterogeneous/WAN
and untrusted-validator drills remain release gates.

An additional operator-invoked hardware gate replaces the trainer backend with
CUDA while retaining a CPU-only validator. The peers advertise distinct signed
release artifacts but resolve an identical revision and training contract. A
real 12,918,148-byte Ruliad candidate traversed the swarm, the CPU peer produced
a four-sample 2,595-byte JSON formal report, and promotion preserved the exact
decoded-tensor digest observed by the CUDA trainer. The report was retrieved by
the complete head/artifact/protocol/report binding and its content ID was
recomputed. The tiny-model CUDA window took 64.01 s because this gate includes
first-use kernel compilation; it is cross-backend correctness evidence, not a
throughput result. The release-profile duty measurements below remain the
authoritative performance evidence.

The formal evaluator now also runs through Dragon's real quorum-two path. A
small native trainer published one 6,448,259-byte candidate to two CPU-only
validators. The first reduction could not promote. The second evaluation
completed a two-attester certificate, after which both validators observed
exactly one merge and the promoted tensor digest matched the candidate. Each
validator persisted a distinct 2,542-byte content-addressed report over the
same head, artifact, protocol, and four-sample panel. Their verifier accuracy,
partial credit, answer-field accuracy, and completion quality agreed within
1e-6. This is trusted local quorum evidence; malicious disagreement and
quarantine remain separate adversarial gates.

A guarded fixed-step 10M-class batch sweep found 26.18k wall tok/s at batch 96
versus 25.49k at batch 48, but the larger batch doubled host use and diluted
the step-cadenced auxiliary rows per token. It therefore does not supersede the
equal-token batch-48 calibration. Useful token throughput and correctness,
not board watts or allocated memory, remain the operating-point gates.

The trainer now also has an explicit
`training.validation.execution = "external_evaluator"` contract. Local
validation remains the default. External mode persists model, optimizer,
scheduler, dynamics, and source-selection candidate state, but emits every
trainer checkpoint as unpromoted and executes no teacher-forced, degeneracy,
correctness, or proof-policy validation. Configuration validation rejects this
mode when local gates, dynamics recovery, neuron scaling, correctness probes,
source-weighted validation, proof-policy probes, or capability-gated latent
starts remain enabled. This makes evaluator ownership explicit instead of
silently dropping trainer safeguards.

A 384-update release-CUDA boundary check used batch 48, 512-token blocks, and
the same four-layer width-256 latent-4,096 Dragon. It crossed a complete
256-update checkpoint boundary and then completed a partial second window.
The comparison below uses each trace's interval from first to last sample at
or above 80% utilization, so process startup and shutdown are excluded.

| trainer boundary | active span | mean/min util | samples below 80% | samples below 50% | mean power | power CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| local teacher-forced validation | 194 s | 94.19% / 8% | 3 | 2 | 53.91 W | 7.88% |
| external evaluator | **436 s** | **95.39% / 82%** | **0** | **0** | 54.85 W | **3.77%** |

The external arm processed 9,437,184 training tokens at 21,554 wall tok/s.
Its stage profile reports 95.08% model duty, 97.16% total train-compute duty,
0% validation wall, 0.0012% foreground loader wait, and 0.0105% checkpoint
wall. The lower aggregate token rate than the earlier 128-update placement
screen is not a regression in validation placement: this longer arm crosses
the configured proof-policy start at update 128 and spends 9.67% of wall time
on proof-policy training. Loss remained finite and continuous across the
checkpoint. The event stream contains no validation event and both candidate
checkpoint events are unpromoted.

A current-binary equal-token recheck separated low board power from low duty.
Both arms processed 3,145,728 tokens with external evaluator ownership and no
local validation. Batch 48 used about 30 GiB total host memory; batch 96 used
about 54 GiB.

| batch / updates | process wall | mean/min active util | active power | power CV | sampled active segments |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 48 / 128 | **111 s** | 95.54% / 95% | 54.71 W | **3.46%** | 1 |
| 96 / 64 | 113 s | **95.80%** / 86% | 54.61 W | 3.51% | 1 |

Doubling batch did not raise power or improve equal-token wall time, while it
consumed another 24 GiB of unified memory. Batch 48 therefore remains the
supported operating point. A separate 64-update stage profile measured 28,004
wall tok/s, 97.26% model duty, 98.37% train-compute duty, zero validation wall,
0.0011% foreground loader wait, and zero host synchronization points. The
remaining 54-55 W level is the recurrent backward and memory-traffic mix, not
a launch or data-generation stall.

The release binary rebuilt after the exact-head validator changes was then
rechecked for 128 warm updates at batch 48. It processed 3,145,728 tokens in
110.80 s of instrumented training time (28,391 wall tokens/s), with 95.14%
model duty, 96.71% total train-compute duty, 0% validation wall, 0.0012%
foreground loader wait, and zero host synchronization points. Across the active
GPU span, mean utilization was 95.42% and mean power was 54.75 W. One sample at
the first checkpoint reached 77% utilization; the following three checkpoint
boundaries retained 95-96% utilization, and checkpoint work accounted for only
0.142% of wall time. This is a bounded checkpoint bubble, not the recurring
low-duty PWM-like execution that motivated evaluator offload.

The same release trainer was then repeated while the complete CPU peer and
validator-quorum gate ran concurrently on the unified-memory host. The stress
is conservative because the companion process also trains a small CPU model;
production validators are intended to run on separate peers.

| condition | wall tok/s | model duty | active GPU util | active power | sub-80% samples | loader wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUDA trainer alone | **28,391** | 95.14% | 95.42% | 54.75 W | 1 | 0.0012% |
| CUDA trainer + CPU peer gate | 27,572 | **95.38%** | **95.43%** | 54.18 W | **0** | 0.0013% |

Co-location reduced useful throughput by 2.89% but introduced no device-duty
stall, validation work, or host synchronization into the trainer. The CPU gate
itself slowed because both workloads share system memory; that is expected on
this integrated host and is not evidence of a GPU pipeline bubble. A real WAN
soak must still measure artifact transfer and promotion latency with the
validator on another machine.

This run exposed a partial-window metadata bug: the final 384-update candidate
was initially labeled with the full logical-window step 511. Checkpoint,
validation, and Ruliad probe metadata now use the number of steps actually
completed in the epoch, so that case resolves to step 383; a focused regression
test covers it. The metadata-only correction landed after the hardware trace
and does not affect its duty measurements. A final three-update release-CUDA
smoke with two-update logical windows emitted checkpoint steps 1 and 2, stored
source-selection offset 2, emitted no validation event, and left both
checkpoints unpromoted.

The next CUDA pass removed work rather than chasing board power. A nonpersistent,
unchunked dense-score step previously constructed the terminal linear-attention
rho state even when no configured objective, recurrent continuation, or
auxiliary memory could consume it. Dragon now distinguishes an ephemeral state
that still captures terminal rho from a stateless execution contract. Automatic
elision is limited to dense-score linear attention with one fast step, no
persistent or effective chunked TBPTT, no predictive coding, no rho/Dragon-state
auxiliary objective, no pipeline schedule, and no y-neuron, hierarchical,
clocked-slow, or summary memory. The explicit
`training.retain_ephemeral_terminal_sequence_state = true` setting restores the
old behavior for compatibility and same-binary ablations.

The context-only executor matches the stateful context and query/value/initial-
rho gradients within `1e-6`; state-policy tests cover the automatic and retained
constructors plus every exclusion above. Existing predictive-coding and
rho-SIGReg state tests continue to pass. A release-CUDA A/B then used the
same binary, seed, four-layer width-256 latent-4,096 model, batch 48, 512-token
blocks, 128 updates, and external evaluator boundary:

| terminal rho policy | wall tok/s | main-model tok/s | forward wall | loss/backward wall | active GPU util | active power | final logged loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| retained control | 28,060 | 30,177 | 23.19 s | 81.06 s | 95.41% | 54.52 W | 0.3216212094 |
| automatic elision, repeat | **29,057** | **32,286** | **20.37 s** | **77.07 s** | 94.67% | 54.77 W | 0.3216212094 |
| automatic elision, first run | **29,296** | **32,702** | **20.54 s** | **75.65 s** | 95.00% | 54.27 W | 0.3216212094 |

The repeated same-binary arm improves wall throughput by 3.55% and main-model
throughput by 6.99%; the two elided runs average 29,177 wall tok/s, 3.98% above
the retained control. Logged losses at steps 31, 63, 95, and 127 and the complete
source-selection telemetry are identical between the same-binary arms. Power is
unchanged because both arms keep a dense queue; the improvement is less useful
GPU work per token, not a higher-watt operating point. The repeat trace includes
two short checkpoint-aligned samples below 80% utilization, while checkpoint
wall remains 0.129% and foreground loader wait 0.0014%, so no recurring device
starvation was introduced.

A mixed-precision Flex32 experiment was rejected. Tiny debug forward/backward,
validation, verifier, and checkpoint smokes were finite, but the 10M-class CUDA
shape requested the same 2,041,618,944-byte CubeCL allocation at batches 48 and
24. Minimal autotuning avoided the immediate allocation but completed no update
in 180 seconds while emitting thousands of tiny initialization kernels. This is
a dtype/autotune backend failure rather than batch pressure. The temporary CLI
mode was removed; only the backend-generic constant dtype corrections remain.

### Native three-peer Ruliad gate

The signed restart run uses a 926,210-parameter Dragon, three native peers, two
rounds, nine local steps per peer per round, and 54 aggregate peer-local steps.

| genesis loss | P2P final | synchronized final | progress parity | restart |
| ---: | ---: | ---: | ---: | ---: |
| 5.718210 | 2.258732 | 1.930078 | 91.324% | 3.176 s |

Candidate tensors, promoted tensors, validation, and the independent merge
oracle match exactly. All signed-contract, shard, receipt, restart, and
convergence gates pass. This is a bounded local convergence gate, not a WAN or
continual-learning result.

### Browser source parity

Real headless Chrome/WebGPU executes both generated NCA and generated formal
Ruliad training. The Ruliad lane consumes two train and two evaluation batches,
checks formal-family metadata, the 272-token symbolic vocabulary,
trace/answer masking, and block/TBPTT behavior. Browser AdamW now accumulates
detached scalar losses on the GPU and performs one asynchronous read at the
window boundary rather than synchronizing after every step.

## Promotion gaps

The next model-quality matrix must be preregistered and run over multiple
seeds. It should include:

1. Held-out proof compositions, symbols, rewrite laws, and domain mixtures
   that cannot be solved by memorizing generator-local frequencies.
2. Difficulty-stratified completion and closed-loop policy metrics, including
   performance beyond the highest level seen during training.
3. CE-only versus static, true DAgger, and paired DAgger at longer horizons
   and at least two model scales, with throughput and peak-memory accounting.
4. Continual-learning retention after frontier expansion, including old-level
   regression and output-collapse gates.
5. Export into an independent proof environment or a separately implemented
   kernel to test semantic, not merely implementation, agreement.
6. A 24-hour native soak, public signed-genesis staging canary, heterogeneous
   WAN measurements, and adversarial validator/quorum drills.

Until those gates pass, the accurate claim is: R3 provides a portable,
verifier-backed formal pretraining and evaluation substrate with promising
local proof-policy signal. It is not yet a leading general reasoning model.
