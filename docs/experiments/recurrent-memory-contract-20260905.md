# Recurrent Memory Contract: Learning And Transfer

## Decision

There is now successful local proof-action learning with a large matched
memory-timescale improvement and a completed curriculum/retention follow-up.
The mastery-first continuation reaches 256/256 d0 and 251/256 d1 verification on
fresh same-grammar examples, retains d0 there, and improves zero-shot structural
d2/d3 transfer. The improvement uses explicit reference ALiBi, persisted rho,
answer-conditioned CE and AdamW, not a larger model or auxiliary objectives.

This is promising enough for gated follow-up scaling, not blanket wide-scale
promotion. Structural d0 falls from 229/256 to 214/256 after the shift, and
counterfactual target binding remains weak. These experiments do not establish
SotA, complete-proof construction, indefinite no-forgetting learning, or
decentralized/browser convergence. No global defaults were changed.

The earlier objective, kernel, scale, positional and two-seed screening results
are in [the NextLat experiment log](nextlat-v2-20260905.md). The final winning
condition here is answer-conditioned CE with AdamW. NextLat, JEPA, PC, CBP,
width scaling, recovery and P2P are off. An unused NextLat predictor is allocated
in all matched models to preserve initial parameter identity.

## Mechanism And Compatibility

For four heads, historical slopes were approximately `[1, .8409, .7071, .5946]`.
With the implemented decay `exp(-slope)` per token, the direct memory-write
half-lives are only 0.69-1.17 tokens. At 64 tokens, even the slowest head retains
approximately `3e-17` of a direct write. This is a direct-state attenuation
calculation, not proof that a nonlinear recurrent network cannot relay signals.

The explicit reference schedule `[.25, .0625, .015625, .00390625]` extends the
slowest half-life to approximately 177 tokens. Its retention over 256 tokens is
`exp(-1)`. Both schedules use the same compute and tensor shapes. CPU/CUDA pulse,
chunked-state and VJP tests verify the implemented recurrence analytically.

The reference schedule follows the original
[ALiBi implementation](https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py).
Public [BDH](https://github.com/pathwaycom/bdh/blob/main/bdh.py) uses a different
positional contract; this correction does not make Dragon identical to BDH.
Omitting `model.alibi_slopes` deliberately retains historical behavior for
checkpoint compatibility. Explicit slopes are validated, saved in model config,
and cannot silently change on exact resume. There is no global default promotion.

## Matched Long Experiment

Matrix: `config/experiments/nextlat-memory-timescale-long.toml`.
Evidence: `target/experiments/nextlat-memory-timescale-long/training-summary.json`.
Both cases completed; captured source and binaries remained unchanged, and
initial model fingerprints matched.

- Approximately 1M parameters, embedding 96, neuron dimension 3072, four heads,
  four shared-weight layers; CUDA release on GB10.
- Seed 29, AdamW LR 0.0003, weight decay 0.01, norm clip 1, dropout 0.
- Batch 8, block 1024, TBPTT chunks 256, credit window four chunks, persisted rho.
- 8,192 scheduled steps, 67,108,864 scheduled positions, 536,028 supervised
  answer positions. Both had 8,166 nonempty CE batches and 26 state-only batches.
- Fixed d0 training, freshly generated virtual epochs, not replay of 512 cached
  examples. This fixed level is an experimental control, not a production cap.
- Development free-run panel: 64 items, evaluated every 512 steps. Cold and
  stream-warm validation each request 128 batches, independently of log cadence.

| Final epoch 16 | Historical decay | Reference decay |
| --- | ---: | ---: |
| Verified /64 | 6 | 57 |
| Answer NLL | 0.323335 | 0.017613 |
| Answer token accuracy | 82.43% | 99.19% |
| Context-swap NLL gain | 0.184702 | 1.834738 |
| Paired warm NLL | 0.075498 | 0.009133 |
| Paired cold NLL | 0.089274 | 0.138664 |
| Carry NLL gain | 0.013776 | 0.129532 |
| Eligible carry pairs | 12 | 12 |
| Whole-case seconds | 1,824.21 | 1,796.89 |
| Scheduled positions/s, including evaluation | 36,788 | 37,347 |
| Mean sampled GPU utilization | 86.10% | 86.14% |
| Mean sampled GPU power | 46.07 W | 46.26 W |
| Peak whole-host usage | 10.66 GiB | 10.65 GiB |

Reference verifier counts at successive checkpoints:
`4,19,21,22,14,24,22,22,35,53,55,57,58,58,58,57`.
Control: `2,1,2,3,3,2,1,1,1,4,4,7,2,3,6,6`.
Learning is not monotonic: there is a transient reference regression at step
2,560 and an apparent plateau before step 4,608. It subsequently improves without
rollback, LR changes or parameter replacement. This argues against aggressively
stopping a run solely on a few flat 64-item panels. It is not a calibrated
production plateau policy.

Twelve carry pairs are better evidence than the earlier one-pair screen, but
still a small sample. Unconditional blank-prompt entropy is not a valid primary
competence gate for an answer-only model. Scheduled positions are not independent
proofs or exclusively supervised output tokens.

## Fresh Final-Checkpoint Confirmation

Matrix: `config/experiments/nextlat-memory-long-confirmation.toml`.
Evidence: `target/experiments/nextlat-memory-long-confirmation/evaluation-summary.json`.
Final epoch 16 was fixed before evaluation; no best-checkpoint selection. Unused
generator seed 8817302 and panel seed 9173402 differ from both development and
the previous short-run confirmation. Matched requests produced identical panel
fingerprints. Each d0 cell has 256 free-generated items and 16 separate typed
policy controls; CUDA batch 2. All generated answers terminated.

| Holdout | Historical verified | Reference verified | Historical NLL | Reference NLL |
| --- | ---: | ---: | ---: | ---: |
| Same grammar, fresh instances | 7/256 (2.73%) | 224/256 (87.50%) | 0.326833 | 0.017208 |
| Structural/new-law | 8/256 (3.13%) | 195/256 (76.17%) | 0.345451 | 0.061637 |

Reference dominant-answer share is 4.30% in both holdouts, versus 72.66% and
74.61% for historical decay. This is a semantic improvement, not merely a
termination or output-entropy improvement. The structural generalization gap
remains material. The short seed-13 experiment independently favored the same
schedule, but two training seeds with unequal budgets do not estimate full
training-seed variance at this longer budget.

### Zero-Shot Difficulty Transfer

A separate structural panel uses 64 items per difficulty, batch 1, and the same
new generator seed. It is not pooled with the 256-item d0 comparison.

| Unseen level | Historical verified /64 | Reference verified /64 |
| --- | ---: | ---: |
| d0 | 3 | 48 |
| d1 | 1 | 23 |
| d2 | 0 | 16 |
| d3 | 0 | 9 |

All 512 answers terminate. Reference transfers partially, but d0 training is not
enough for difficult prompts. Surrounding proof size grows with level; that alone
does not prove increased proof-search depth, arbitrary difficulty scaling, or
complete mathematical coverage.

## Remaining Task-Validity Limits

The answer `gN` refers to the positional goal index from `?:select;g=N`, not the
possibly shuffled named goal ID inside `P`. Direction is constant `f` in the
elementary training source. These are substantially copyable protocol fields.
Candidate order and path hints can also provide shortcuts. A readonly audit of
separate 64-item inspection panels found that selecting the first forward
candidate exactly matches 42/64 same-grammar validation answers and 39/64
structural answers. Those are different panels, not a matched statistical
comparison with the model results above.

A subsequent same-panel audit of the 64 final development items gives 49 exact
matches for first-forward, 39 for first-forward-at-path, and 30 for the majority
source plus hinted path. The model verifies 57. Thus most elementary examples
are already solvable by a surface heuristic; 89% does not mean 89% on a hard
proof-search benchmark. This is exploratory, reused-panel evidence, not a
preregistered confirmatory comparison. Per-item predictions are saved in
`target/experiments/nextlat-memory-timescale-long/surface-heuristic-audit.json`.

The action verifier checks the current local transition against the preferred
certificate outcome. It is not an independent Lean proof checker, and it need
not accept every alternative action that eventually completes a valid proof.
Prompt state/pattern abbreviation and canonical-expert bias require an explicit
observability and counterfactual-target audit before broad reasoning claims.
Typed-menu scoring is shortcut-sensitive; free generation is the evidence here.

## Retention Design And Prerequisites

`config/experiments/nextlat-memory-retention.toml` specifies a single-process
distribution-shift experiment with a stationary control. Both use the reference
recipe and identical two-bucket catalogs. For 8,192 steps both see only d0; then
one releases d1 with equal d0/d1 sampling while the control stays on d0, through
step 16,384. No optimizer reset, model reload, new LR, or width change occurs at
the boundary. Separate 64-item d0 and d1 development panels measure forgetting
and adaptation. Actual exposure and released probabilities must be verified,
not inferred from config. Fresh final-checkpoint confirmation follows only after
the training matrix finishes. Retention with ongoing d0 rehearsal is not a
no-replay continual-learning result.

Before this matrix, the bounded release/clock smoke and focused tests must pass.
Validation events now retain the completed training step instead of deriving it
from validation-batch counts. The separately stored `step_in_epoch` remains the
validation progress counter. Source-weighted sampling seeds are unchanged; its
logging clock is corrected separately from sampling coordinates. The non-streaming
Burn learner uses a separate metric adapter and is not covered by this change.

The CUDA prerequisite completed as `runs/colorful-boat`: 32 steps, two validation
passes with 128 batches each, 45.35 seconds, peak whole-host use 12.77 GiB. All
964 validation metric events retained their epoch-end training clocks (15 and
31); both rho and paired carry metrics were present. Recorded source probabilities
were d0=1/d1=0 at steps 0 and 15, then d0=d1=0.5 at step 31. Source/binary
identities remained unchanged. This is a mechanics smoke, not learning evidence.

Focused validation also passed the new CPU clock test (budgets 1 and 5 around a
two-step training interval), two cold-start tests, five existing scheduler
objective tests, and 20 Python experiment/analyzer tests. The rebuilt release
training and checkpoint-evaluation binaries completed successfully. Broader
non-streaming, browser, P2P and long-run gate qualification remains separate.

### Aborted Curriculum Matrix: Checkpoint Clock Mutation

The first retention arm (`runs/ratty-flight`, original output
`target/experiments/nextlat-memory-retention`) was stopped when telemetry showed
d1 release at step 4,128 instead of 8,192. The stationary arm was not launched.
Its result is not a valid comparison for the declared curriculum schedule.
There was no OOM: the runner was explicitly interrupted after the discrepancy
was diagnosed. The earlier fixed-d0 comparisons do not exercise this release
transition and remain valid for their stated fixed-source comparison.

Root cause: exporting a source-selection checkpoint stored the last training
step into the live runtime's additive step offset. Later calls added that offset
to already-global within-run steps. This both advanced the curriculum early and
made its behavior depend on checkpoint frequency. The first short smoke released
its second level just after its first checkpoint, so it did not discriminate
this bug; the revised smoke places three checkpoints before the release boundary.

Source-state contract v2 separates immutable `run_step_origin` from the exclusive
`completed_run_steps` checkpoint counter. Export is read-only. Exact resume
retains the origin; a new optimizer phase starts at the next global step once.
Training document generation also uses that origin so a new phase does not
silently restart the data stream; fixed validation coordinates remain unchanged.
Older v1 source-state files are rejected with an explicit weights-only migration
message rather than guessing their clock semantics. Model-only evaluation of
the old checkpoints remains supported.

The revised smoke and long-matrix configs write new `*-clock-v2` output roots;
the aborted experiment and its captured original configs remain preserved.

The fix passed all 101 dataset/loader tests, eight manifest tests and the
validation-clock regression test. Document-coordinate tests cover both ordinary
generation and consolidation replay across a new-phase handoff. Consolidation
uses a global generation coordinate for replay identity and the current global
training step for new-source eligibility. Stratified policy sidecars use the
same generation coordinate. Masked-out buckets now retain exactly zero weight;
sampling uses the standard `rand::WeightedIndex` distribution instead of restoring
an epsilon probability to forbidden buckets. Both release binaries rebuilt.

### Clock V2 CUDA Qualification

The revised 64-step smoke (`runs/scary-earth`) and 32-step prefix
(`runs/alcoholic-metal`) passed. Source probabilities remained d0=1 through step
47 despite checkpoints at 16, 32 and 48 completed steps; release occurred at the
declared step 48. All 708 validation events in the full run and 354 in the prefix
used the correct training clock. Every exported origin remained zero.

An exact resume of a hash-verified private copy of the prefix reproduced the
uninterrupted source schedule and checkpoint clocks through step 64. Original
prefix artifacts were preserved. The CUDA trajectories were not bit-identical:

| Final metric | Uninterrupted | Exact resume | Fresh uninterrupted repeat |
| --- | ---: | ---: | ---: |
| Validation objective loss | 0.87485037 | 0.87505573 | 0.87479733 |
| Answer-panel NLL | 2.34278500 | 2.34335882 | 2.34249930 |

The resume difference is small but larger than this single fresh-repeat
difference, so this does not establish bitwise or long-horizon numerical resume
equivalence. Greedy outputs at 64 steps are undertrained and not a useful
learning-parity gate. Prefix/full/resume/repeat wall times were 16.28/25.82/15.79/
26.56 seconds; whole-host peaks remained below 13.1 GiB. Source and binary
identities stayed unchanged within each experiment. Evidence roots are
`nextlat-memory-retention-smoke-clock-v2`, `nextlat-memory-resume-smoke-clock-v2`
and `nextlat-memory-repeat-smoke-clock-v2` under `target/experiments/`.

## Corrected Retention Results

Both clock-v2 arms completed 16,384 scheduled steps with identical initial
weights and unchanged captured sources/binaries. The curriculum released d1 at
exactly step 8,192; the stationary control never released it. All 30,848
validation metric clocks and 39,040 finite metric values passed the post-run
audit. Development panel inputs matched across arms and all epochs. Pre-shift
per-epoch supervised-token counts matched, though CUDA trajectories were not
bit-identical and d0 learning speeds differed before release.

| Final epoch 32 | Stationary d0 | d0 then d0/d1 |
| --- | ---: | ---: |
| Development d0 verified /64 | 64 | 52 |
| Development d1 verified /64 | 30 | 44 |
| Answer NLL, mixed panel | 0.383224 | 0.033607 |
| Paired warm NLL | 0.425524 | 0.031336 |
| Paired cold NLL | 0.479953 | 0.120508 |
| Eligible carry pairs | 38 | 38 |
| Final rho RMS | 13.3269 | 17.7968 |
| Wall seconds | 3,998.46 | 3,716.78 |
| Supervised answer positions | 1,071,046 | 948,334 |
| Nonempty CE batches | 16,330 | 14,548 |
| State-only batches | 54 | 1,836 |
| Mean sampled GPU utilization | 84.73% | 84.33% |
| Mean sampled GPU power | 45.29 W | 45.47 W |
| Peak whole-host memory | 12.75 GiB | 12.72 GiB |

Both scheduled 134,217,728 positions. The shorter curriculum wall time is not
a kernel-throughput improvement: longer d1 documents create more state-only
windows without CE/backward updates. Supervised answer throughput is about 268
versus 255 positions/second. Equal d0/d1 probability applies at document-batch
selection, not to tokens or training steps. GPU utilization below 25% occurred
in under 0.5% of samples for either arm. Resource samples include evaluation.

### Fresh Matched Midpoint/Final Confirmation

`config/experiments/nextlat-memory-retention-confirmation.toml` completed all
eight cells with unchanged source/binary identities. Final epoch 32 and midpoint
epoch 16 were predetermined, not chosen by validation rank. Midpoints were
hash-verified copies made before checkpoint pruning. Generator seed 8817303,
panel seed 9173403, and a 1,024-index validation space were unused by the earlier
experiments. Each cell contains 256 free-generated items per difficulty, CUDA
batch 2, and 16 separate typed-policy controls. Matched panel fingerprints pass.

| Training condition | Same-grammar d0 /256 | Same-grammar d1 /256 | Structural d0 /256 | Structural d1 /256 |
| --- | ---: | ---: | ---: | ---: |
| Curriculum midpoint | 182 | 59 | 126 | 18 |
| Curriculum final | 197 | 166 | 194 | 146 |
| Stationary midpoint | 199 | 57 | 163 | 37 |
| Stationary final | 256 | 98 | 235 | 94 |

The curriculum adapts to harder material while improving its own d0 scores,
including structural transfer. It does not match the d0 specialization achieved
by training only d0 longer. A single matched initialization and repeated panels
do not establish training-seed robustness or indefinite retention. All 4,096
answers terminate; stationary-final d1 has 10 malformed same-grammar and three
malformed structural completions, while both curriculum-final strata have zero.
Each evaluation takes 141-146 seconds; peak whole-host use stays below 10.4 GiB.

The preceding first-forward heuristic still matches 49/64 answers at each level
of the development panel, versus curriculum-final 52/64 and 44/64. Typed-menu
accuracy also overstates reasoning: on the fresh same-grammar panel, the
stationary final model and its no-context control both score 16/16. Curriculum
final scores 16/16 versus 13/16 without context, but changes its top choice on
only 3/16 counterfactual targets (4/16 structural). These are small, diagnostic
panels. Full proof construction and shortcut-resistant reasoning remain unproven.

### Exploratory Mastery-First Continuation

The fresh stationary-final checkpoint reached 256/256 d0 and 235/256 structural
d0, whereas the early-release arm had not mastered d0 before its shift. The next
bounded test extends a private exact-resume copy of that stationary run to
24,576 steps. Its already-declared cold-start schedule releases d1 at step
16,385. Only the horizon changes: model, AdamW moments, scheduler, source state
and recurrent runtime are restored without an LR or optimizer reset. This is an
exploratory follow-up selected after the matrix, not a preregistered equal-total-
compute comparison with the 16,384-step arms. New final confirmation is required.

## Mastery-First Outcome

The continuation completed all 8,192 additional steps, reaching epoch 48 and
24,576 total steps. Source and binary identities stayed unchanged. The original
run is preserved; its private copy contains the resumed training and final
checkpoint. This is the same approximately 1M model, not a width-scaled model.

- Additional wall time: 1,792.44 seconds; combined training-case wall time with
  its 16,384-step precursor: 5,790.90 seconds (96.5 minutes, including validation).
- Additional scheduled positions: 67,108,864; supervised answer positions:
  413,320; 6,407 nonempty CE batches and 1,785 state-only batches. Combined
  supervised answer exposure is 1,484,366 positions, not 201M supervised targets.
- Additional-phase mean sampled GPU utilization: 84.11%; mean power: 44.53 W;
  peak whole-host memory: 12.81 GiB. Utilization below 25% occupies 0.68% of
  samples. The same 90%-of-shared-memory guard and conservative admission
  estimate remained enabled; no capacity/OOM probing was performed.
- All 7,712 additional validation metric clocks and 9,760 finite metric values
  pass the audit. Final source clock is origin 0, completed steps 24,576.
- Post-release d0 development verification never falls below 62/64 and ends
  64/64. d1 trajectory across the 16 checkpoints is
  `47,54,53,57,57,59,58,60,59,60,60,63,61,63,63,64`.
- Final mixed-panel answer NLL: 0.000263; paired warm/cold NLL:
  0.000243/0.088396 over 38 eligible pairs; rho RMS: 9.3938.

The paired carry result supports useful through-time state in this task; it is
not a proof of arbitrary-context or infinite-horizon stability. The simultaneous
blank-prompt entropy is only 0.766 bits despite perfect development task
verification. It must not be interpreted as output collapse without a matching
conditional task probe. No recovery, width scaling, CBP, JEPA, NextLat, or PC
objective is responsible for this result.

### Fresh Confirmation And Transfer

`config/experiments/nextlat-memory-mastery-confirmation.toml` completed all six
cells. Generator seed 8817304 and panel seed 9173404 were reserved until this
continuation completed. Model identities and matched panel fingerprints pass;
evaluation does not mutate model weights. The main panel has 256 items per
level; the separate harder panel has 64 per level and is not pooled with it.

| Fresh free generation | Before, epoch 32 | After, epoch 48 |
| --- | ---: | ---: |
| Same-grammar d0 | 256/256 (100.00%) | 256/256 (100.00%) |
| Same-grammar d1 | 97/256 (37.89%) | 251/256 (98.05%) |
| Structural d0 | 229/256 (89.45%) | 214/256 (83.59%) |
| Structural d1 | 98/256 (38.28%) | 163/256 (63.67%) |
| Separate structural d2, unseen in training | 23/64 (35.94%) | 41/64 (64.06%) |
| Separate structural d3, unseen in training | 12/64 (18.75%) | 36/64 (56.25%) |

The same-grammar mixed-panel NLL improves from 0.459658 to 0.002354;
structural mixed-panel NLL improves from 0.450530 to 0.108390. All 2,560
before/after free-run answers terminate; all final-checkpoint cells have zero
malformed outputs. Evaluation takes 129-151 seconds per cell, with peak
whole-host memory below 11.9 GiB.

There is a real observed structural-d0 retention cost: 15 fewer correct answers
out of 256 (5.86 percentage points). The 64-item harder panel happens to retain
54/64 on its different d0 subset; that does not erase the larger-panel decrease.
This supports adaptation with in-distribution retention, not no forgetting
across every distribution. These are finite generator holdouts, not external
mathematical benchmarks or an independently implemented proof checker.

Counterfactual controls still fail to justify broad reasoning claims. On the
main fresh same-grammar panel, final typed-menu accuracy is 16/16 versus 15/16
without context, with only 1/16 target changes. On the structural panel, model
accuracy is 13/16 versus 14/16 without context, with 2/16 target changes. The
corresponding counterfactual probability gains are positive, but those small
gains are not reliable discrete goal-conditioned reasoning. Free proof-action
generation, typed oracle-menu selection and full proof construction must remain
separate evaluation contracts.

## Qualification And Next Experimental Gate

The achieved scope is a resource-safe, measurably learning local recipe and a
diagnosed curriculum failure, with reproducible configs and captured evidence.
The strongest recipe is an experimental anchor, not a new automatic production
policy. In particular, the later release was chosen after measuring mastery;
an automatic verifier-driven release controller was not trained/qualified here.

Before broad scaling, the next explicit matrix should combine:

1. Three independent training seeds, matched 1M model and total token budgets:
   reference decay versus historical decay, and fixed-time versus verifier-
   mastery release. Keep source/answer exposure and context lengths explicit.
2. Frozen-model evaluations with fully observable proof states, balanced rule
   orientation, candidate-order transformations and verifier-valid alternative
   targets. Include first-forward, no-context and structural-distance controls;
   require gains beyond those controls before claiming stronger reasoning.
3. Repeated difficulty shifts with rehearsal, separate structural-d0 retention
   panels, and a same-budget stationary comparator. Use per-bucket verification
   and exposure, not blank-prompt entropy or aggregate teacher-forced CE alone.
4. Only after those gates, a bounded 1M/10M reference-memory scaling comparison
   with the same tasks, budgets, resource guard and CUDA efficiency accounting.
   Re-ablate NextLat/JEPA/PC against this corrected CE baseline before promotion.

No experimental local position/source-state contract is automatically qualified
for P2P or browser peers. Source-state v1 exact resume remains fail-closed;
weights-only migration and evaluation are distinct from exact continuation.
Non-streaming metric adapters, source-weighted validation, and zero-CE-batch
auxiliary objectives remain outside this recipe's qualification. The vendored
CUDA fusion fix also needs upstream integration and broader backend coverage.

Final focused revalidation passed 101 dataset/loader tests, eight manifest tests,
the validation-clock regression test, and 20 Python runner/analyzer tests.
Earlier CPU/CUDA pulse and VJP tests and both CUDA release binaries passed.
Existing vendored autotune-disabled unused-code warnings remain; this is not a
warning-free full-workspace/wasm CI claim. All bounded training/evaluation jobs
are finished. Evidence summaries are retained under the named experiment roots.
