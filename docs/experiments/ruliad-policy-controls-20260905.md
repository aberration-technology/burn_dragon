# Ruliad Policy Controls And Scorer Ablation

Date: 2026-09-05. Status: both matrices completed, 12/12 CUDA evaluations passed.
No learner, maintenance setting, scorer, or deployment default is promoted.

## Executive Result

The evaluated models learn useful statistical preferences over supplied actions,
but the current reference-action score does not establish problem-conditioned
reasoning. All six frozen checkpoints scored higher without a problem prompt than
with it. Removing the semantic energy component restores a small contextual
advantage, but lowers overall reference-action accuracy. Removing the language
likelihood component does not resolve the negative contextual advantage.

This is a diagnostic result, not evidence that context is universally harmful or
that PC cannot learn reasoning. In particular, a valid alternative proof action
can disagree with the reference certificate. Actual goal completion, independent
verification, and repeated training seeds remain necessary.

## Implemented Work

- Checkpoint-only controls live in
  `crates/burn_dragon_language/src/train/schedule/ruliad_policy_controls.rs`, with
  tests in its own submodule. Normal training explicitly selects `Disabled` and
  performs no additional control forwards or certificate audits.
- Controls include exact uniform chance, first canonical/presented action,
  shortest serialized semantic action, one-step structural-distance search, and
  the same frozen model with only the `!:` answer delimiter as prompt.
- All controls retain the same candidates and presentation rotations. Heuristic
  ties use uniform expected credit, never reference-label tie breaking.
- Each item's complete reference certificate and prefix are replayed. All
  candidate outcomes, distances, and equivalence labels are checked against
  fresh kernel execution. Cached-label/outcome corruption fails evaluation.
  This uses the existing kernel; it is not a second independent proof checker.
- Suite v8 exports item identities, counts, paired outcomes/probabilities, and
  difficulty/source aggregates. The analyzer independently checks coverage,
  identities, chance, model/no-context aggregates, and verifier agreement. It
  rejects dropped items, duplicate identities, and incomplete kernel audits.
- The archived masked-stream reset profiles remain negative fixtures. Their
  launcher now rejects them before compiling, creating artifacts, or launching
  training. The default TBPTT sweep is carry-only at chunk sizes 512/128/64;
  it is no longer advertised as a runnable reset/carry factorial experiment.

## Experimental Contract

- NVIDIA GB10, driver 580.142, CUDA f32, release build.
- 1M-class Dragon: four tied-weight layers, four heads, embedding 96, neuron
  width 3,072, linear attention, ALiBi, dense short-context score executor.
- Existing training checkpoints: seed 13, batch 8, block 1,024, TBPTT chunk 64,
  alternating required policy/full-completion updates. These evaluations do not
  perform new training. PC here is local credit assignment with AdamW parameter
  updates, not an optimizer-free procedure.
- Main matrix: panel seed 73, 256 items per panel, four difficulty strata with
  64 policy items each, inference batch 2, four candidate actions, no closed-loop
  rollout. Each model is fingerprinted before/after evaluation.
- Component matrix: the same 256 policy items, but 16 items in each repeated
  full-document generation panel. Policy identities, candidates, labels, and
  heuristic controls were checked equal across both matrices. Whole-suite panel
  hashes differ because the full-document panel sizes differ; analyses are kept
  separate. All arms within each analysis share their whole-suite fingerprint.
- Sources, executable, checkpoint/config inputs, and sibling repository states
  are archived by the typed experiment runner. Both matrices report
  `complete=true` and `source_unchanged=true`; all declared inputs stayed unchanged.
- The 512-update and 1,024-update groups are separate experiments, not consecutive
  checkpoints of one trajectory. One training seed is not a multi-seed result.
  This repeatedly inspected panel is exploratory, not a fresh confirmation set.

Reproduction manifests:

- `config/experiments/ruliad-policy-controls.toml`
- `config/experiments/ruliad-policy-scoring-controls.toml`

Run with `python3 -m scripts.experiments <manifest>`. Existing output directories
are not overwritten; choose a new output path for replication.

## Frozen-Checkpoint Matrix

Every accuracy below uses the same 256 policy items, except the free-generation
column, which uses its separate matched 256-item document panel. "Typed" means
reference-equivalent selection from a verifier-enumerated oracle menu, followed
by deterministic rendering. It does not mean the model proposed a proof action.

| Checkpoint | Typed | No context | Context delta | Free verified | Full-answer token NLL | Sequence NLL | Eval seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AdamW, 1,024 updates | 181/256 (70.70%) | 188/256 (73.44%) | -2.73 pp | 0/256 | 0.463651 | 7.027203 | 128.50 |
| PC, 1,024 updates | 181/256 (70.70%) | 189/256 (73.83%) | -3.12 pp | 2/256 | 0.461422 | 6.993431 | 129.48 |
| AdamW, 512, carry | 186/256 (72.66%) | 195/256 (76.17%) | -3.52 pp | 0/256 | 0.513441 | 7.781836 | 126.02 |
| PC, 512, carry | 160/256 (62.50%) | 182/256 (71.09%) | -8.59 pp | 0/256 | 0.525602 | 7.966157 | 129.96 |
| AdamW, 512, no maintenance | 158/256 (61.72%) | 191/256 (74.61%) | -12.89 pp | 0/256 | 0.513464 | 7.782183 | 128.77 |
| PC, 512, no maintenance | 177/256 (69.14%) | 182/256 (71.09%) | -1.95 pp | 0/256 | 0.513499 | 7.782723 | 125.96 |

Shared non-model baselines:

| Control | Expected reference-action accuracy |
| --- | ---: |
| Uniform random action | 25.00% |
| First canonical action | 23.05% |
| First presented action | 23.05% |
| Shortest semantic action, uniform ties | 60.58% |
| Minimum executed distance to goal, uniform ties | 43.36% |

Chance is computed per item as `equivalent_candidates / candidates`. This panel
happens to have exactly one equivalent action among four on every item; tests
also cover multiple equivalent actions. Fractional heuristic scores are expected
tie-breaking credit, not integer counts of executed random trials.

The paired context comparison is not just a difference between unrelated samples:

| Checkpoint | Context helps | Context hurts | Equivalent-probability gain |
| --- | ---: | ---: | ---: |
| AdamW 1,024 | 6 | 13 | -0.04295 |
| PC 1,024 | 2 | 10 | -0.03179 |
| AdamW 512 carry | 6 | 15 | -0.04365 |
| PC 512 carry | 32 | 54 | -0.12309 |
| AdamW 512 no maintenance | 4 | 37 | -0.07340 |
| PC 512 no maintenance | 4 | 9 | -0.02326 |

For example, difficulty stratum 3 gives AdamW 1,024 **56.25% with context versus
70.31% without it**, and PC 512 carry **45.31% versus 68.75%**. Each stratum has
only 64 items. These are exact evaluated-panel outcomes, not seed-level confidence
intervals or generalization guarantees across mathematics.

The maintenance comparison remains unresolved as an optimization promotion:
AdamW loses 28 reference decisions when maintenance is removed, while PC gains
17. The prior pilot demonstrated a 31.4%/25.5% update-throughput improvement, but
it did not demonstrate quality neutrality. The stored stream state is not consumed
by these self-contained primary objectives, so these results must not be presented
as proof that rho carry caused the differences. Realized decoder-batch identity,
numerical nondeterminism, and repeated seeds still need isolation.

## Frozen Scorer-Component Ablation

The residual scorer is `mean language log likelihood + semantic energy`. This
matrix changes only the existing inference scorer on the same trained weights;
it does not retrain each scorer as a standalone learner. All rows use the same
256 policy items and candidate-conditional normalization.

| Learner | Scorer | Typed | No context | Context delta | Counterfactual target probability gain |
| --- | --- | ---: | ---: | ---: | ---: |
| AdamW | Residual combination | 70.70% | 73.44% | -2.73 pp | +0.37926 |
| AdamW | Language likelihood | 62.50% | 58.20% | +4.30 pp | +0.01145 |
| AdamW | Semantic energy | 69.92% | 75.00% | -5.08 pp | +0.37593 |
| PC | Residual combination | 70.70% | 73.83% | -3.12 pp | +0.37801 |
| PC | Language likelihood | 65.23% | 64.45% | +0.78 pp | +0.00926 |
| PC | Semantic energy | 69.92% | 73.05% | -3.12 pp | +0.37593 |

The energy component carries most of the reference-menu score and target-change
sensitivity. Its standalone result still has a negative natural-panel context
advantage. The LM prior is therefore not, by itself, the explanation for that
negative advantage. Conversely, the positive LM-only context delta is small and
comes with lower total accuracy; it is not grounds to replace the deployed scorer.

Strong counterfactual target sensitivity and weak natural-panel context benefit can
coexist. Retargeting a goal to a supplied candidate's outcome differs from choosing
a step on an arbitrary reference proof. A model may also prefer a valid alternative
action that fails reference equivalence. Neither measurement alone establishes
successful autonomous proof search.

## Metric Interpretation

- The earlier approximately 0.07 validation values are the dashboard's
  `Random Cold Loss`, not the complete-answer likelihood measured here. Do not
  plot them as equivalent measures of complete proof correctness.
- Mean complete-answer sequence NLL near 7 is materially different from an
  apparently small per-token loss. Teacher forcing supplies earlier correct
  answer tokens; greedy generation does not.
- Oracle-menu validity is partly supplied by construction. The candidate builder
  retains the reference step; selecting a valid presented action is not discovering
  one without assistance.
- Closed-loop `top1_expert_rate` currently compares against the action set's
  selected index. When the runtime builds a menu without a preferred certificate
  step, that index is the structural-distance minimizer. It is not the same label
  contract as this offline reference-certificate panel. Use actual solve rate as
  the primary closed-loop outcome. No new closed-loop comparison was run here.

## Efficiency And Safety

The full audit took 125.96-129.96 seconds per checkpoint; the reduced-document
component audit took 35.56-36.32 seconds. Total child-process wall time was about
16.4 minutes. The prior two-checkpoint audit took 120.22/120.78 seconds: the added
controls cost approximately 7% in these unreplicated whole-evaluator measurements.
There is no new training-throughput benchmark in this report.

GPU samples were taken every two seconds. Full audits averaged 61.1-64.1% reported
GPU utilization and 32.3-33.0 W; component audits averaged 58.3-59.4% and 28.9-30.9 W.
These include teacher forcing, incremental generation, CPU preparation, and kernel
verification. They are not CUDA duty traces, model-only throughput, or evidence
that dense training utilization has been solved.

Peak total host-used memory across all cases was 13,826 MiB, approximately 13.50
GiB or 11.1% of physical RAM. The guard used a 90% limit, 4 GiB headroom, conservative
16 GiB additional-memory admission estimates, and 250 ms host sampling. GB10 RAM
was counted once; separate VRAM capacity and GPU power limit report N/A. No OOM
probe, training restart, or automatic width/batch expansion was attempted.

## Verification And Remaining Gates

Passed:

- Release `evaluate_ruliad_checkpoint` build with `train,cuda`, locked dependencies.
- 278 focused Ruliad Rust tests, including six new control tests. The broader run
  found the stale reset-profile expectation; it passed after aligning the fixtures
  and launcher with the existing safety contract.
- 21 Python runner/analyzer/TBPTT tests; analyzer self-test; shell syntax check.
- Nine browser routing/canary-profile Node tests in this research checkout. These
  are not real-WebGPU or accepted-receipt tests of the separate main deployment.
- Twelve checkpoint suites with before/after tensor identity, source/input identity,
  full panel coverage, and per-candidate kernel audits. Separate generated analyses
  exist for the 1,024-update pair, maintenance quartet, and scorer components.

Next experiments must address the demonstrated limitations:

1. Build matched goal/premise counterfactual panels with label-free candidate
   generation and score actual goal completion, not only a reference's next step.
   Include the same no-context and symbolic-search controls. Separate syntax-only
   decoding from verifier-enumerated choices.
2. Compare the existing residual baseline, standalone semantic scorer, and a
   context-discriminative training condition with identical initialization,
   realized policy **and decoder** rows/masks, and three training seeds. Confirm
   candidates on a new holdout and five seeds before promotion. This inference
   ablation is not a substitute for that learning experiment.
3. Exercise an objective that actually consumes carried state on documents whose
   answers depend on earlier chunks. Pair correct carry with deliberate reset and
   wrong-document state at evaluation; prevent context-only optimizer updates.
4. Finish the independent verifier, signed real-WebGPU accepted-work canary, and
   matched multi-peer convergence gates before claiming end-to-end decentralized
   reasoning training. These deployment and long-horizon phases remain incomplete.

Artifacts: `target/experiments/ruliad-policy-controls/` and
`target/experiments/ruliad-policy-scoring-controls/`. Both jobs exited; no training
or evaluation process was left running.
