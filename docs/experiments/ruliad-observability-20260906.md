# Ruliad Observation And Counterfactual Audit

Date: 2026-09-06. All four CUDA evaluations completed. No training or deployment
default is promoted. This is a frozen-checkpoint evaluation ablation, not a new
training/convergence experiment.

## Decision

The corrected recurrent-memory recipe learns useful answer and action patterns,
but it is not yet a robust, target-dependent reasoning learner. Giving the
existing model a complete local observation is **not** a drop-in fix. Typed-menu
accuracy can approach 90% while joint correctness on an original problem and a
verifier-valid changed target remains zero. The evaluation must retain both
measurements rather than rewarding any answer change as successful reasoning.

Continuation improves harder structural examples and counterfactual performance
under the familiar prompt, while forgetting some easier structural examples.
Aggregate answer NLL hides that tradeoff. These results do not establish
indefinite continual learning, full-proof generation, PC superiority, or SotA.

## Implemented Contracts

- `oracles/action_prompt.rs` owns shared candidate rendering and a new exact
  action observation. It contains complete current/target terms and oriented
  rewrite rules, without truncation, hashes, or a first-difference hint. It does
  not expose reference labels or cached candidate outcomes/distances.
- Historical query serialization is preserved. Its documentation now correctly
  describes a focused, potentially lossy observation rather than claiming that
  it is verifier-sufficient. The old first-difference hint is computed from
  public state and target; it is not itself a leaked certificate label.
- `RuliadProofPolicyPromptContext::ExactActionState` is an explicit option.
  Fixed-block training encoding rejects insufficient context instead of cropping
  an exact observation. Checkpoint scoring retains complete recurrent prefixes.
  The new checkpoint CLI option is `--policy-prompt-context exact_action_state`.
- Suite v9 exports counterfactual item count, target accuracy, target NLL, and
  joint pair accuracy, as well as the existing change/probability-gain metrics.
  Pair accuracy requires both original and retargeted answers to be correct.
  The current state and candidate menu are preserved and target labels must be
  disjoint. The same metric is emitted through the training event path.
- Counterfactual validity means a kernel-checked one-step transition reaches the
  changed target. It does not mean the original whole-problem certificate remains
  valid after retargeting. Original accuracy is reference-action equivalence,
  not acceptance of every alternative mathematical proof.
- Analysis separates prompt contracts, checks matched identities, and rejects
  impossible counterfactual coverage, rates, pair counts, or nonfinite NLL.
  Additional controls remain checkpoint-only; no new training forward pass was
  added. Historical training formats, optimizer settings, and defaults remain
  unchanged.
- A stale supervision-audit test was corrected: 8/12 fixture observations have
  their query and first answer prediction in the same 128-token block. Short
  query-to-answer span alone does not imply aligned-block visibility.

## Experiment

- Local NVIDIA GB10, release CUDA f32, direct inference backend without autodiff.
- Approximately 1M parameters: embedding 96, neuron width 3072, four heads and
  four tied-weight layers; linear attention with reference ALiBi timescales.
- Training seed 29. The before checkpoint is `runs/terrific-friend`, epoch 32,
  after 16,384 stationary d0 steps. The after checkpoint is the exact-resume
  continuation under `target/experiments/nextlat-memory-mastery-work/terrific-friend`,
  epoch 48, after 8,192 additional mixed d0/d1 steps, 24,576 total.
- Existing recipe: AdamW, batch 8, block 1024, TBPTT chunk 256, credit window 4,
  persisted rho, answer-conditioned CE. PC, JEPA/NextLat losses, CBP, recovery,
  scaling, and P2P were disabled. The allocated unused NextLat head is unchanged.
- Fresh structural holdout: generator seed 8817305, panel seed 9173405,
  1024-index validation space. Each cell uses 128 canonical free-generation
  items and 128 policy items, balanced across d0/d1. Training-serialization and
  policy-context generation are evaluated separately. Inference batch bound 2,
  four policy candidates, scoring batch bound 4, balanced presentation rotations.
  This is not a complete permutation-orbit or balanced-rule-orientation ablation.
- No closed-loop proof rollout and no optimizer updates occur in this matrix.
  The before/after contrast includes more training, not equal total compute.
- One sequential GPU job, 32 GiB conservative additional admission estimate,
  90% physical shared-memory ceiling, 4 GiB headroom, 250 ms host monitoring,
  and 2 s GPU sampling. No OOM probing or automatic batch growth.

## Results

All correctness counts below have denominator 128. "Policy free" means actual
autoregressive generation using that policy observation. "Typed" is ranking
verifier-enumerated candidates and rendering the selected action.

| Checkpoint | Policy observation | Canonical free | Policy free | Typed | No context | Changed target | Both targets |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Before, epoch 32 | Historical full problem | 78 | 80 | 110 | 92 | 8 | 1 |
| After, epoch 48 | Historical full problem | 82 | 82 | 97 | 88 | 27 | 9 |
| Before, epoch 32 | Exact local state | 78 | 5 | 115 | 92 | 6 | 0 |
| After, epoch 48 | Exact local state | 82 | 5 | 112 | 88 | 8 | 0 |

The shared original-menu baselines are 25% uniform, 28.125% first canonical,
25% first presented, 61.068% shortest action with unbiased ties, and 50%
one-step structural-distance search. These are menu controls, not unconstrained
proof-generation baselines. No-context decisions match exactly between prompt
variants for each checkpoint.

Under the historical prompt, context helps/hurts 24/6 items before continuation
and 20/11 afterward. Under the exact prompt it helps/hurts 27/4 and 31/7. Thus
the model does use contextual information; the problem is not complete context
blindness. Those preferences do not translate into reliable adaptation to a
changed target. Exact-state pair accuracy is 0/128, with a descriptive Wilson
95% upper bound of 2.91%. Historical after-continuation pair accuracy is 9/128,
7.03%, with interval 3.74%-12.82%. These item-level intervals do not estimate
variation across training seeds and are not multiplicity-adjusted claims.

The exact observation changes several input properties and is out of distribution
for these checkpoints. Its 5/128 free-generation score demonstrates a format
transfer gap, not that training on exact observations cannot work. Its high typed
score is also not sufficient reason to promote it. Training representation,
candidate-distribution bias, target dependence, and action rendering must be
tested independently.

### Structural Retention

Canonical generation does not depend on the policy-prompt override, and is
identical across both prompt arms for a given checkpoint.

| Metric | Before | After |
| --- | ---: | ---: |
| Structural d0 verified, /64 | 58 | 50 |
| Structural d1 verified, /64 | 20 | 32 |
| Total verified, /128 | 78 | 82 |
| Canonical answer-token NLL | 0.501950 | 0.142371 |

This fresh panel again shows the harder-task gain/easier-structural-task loss
tradeoff. Do not pool it with the different earlier confirmation panels or claim
that a four-answer aggregate gain resolves forgetting. See the larger prior
confirmation in [the memory-contract report](recurrent-memory-contract-20260905.md).

### Cost And Safety

| Cell | Suite wall seconds | Typed/control stage seconds | Mean GPU utilization | Mean power | Peak whole-host GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before, historical | 65.62 | 7.38 | 63.38% | 33.31 W | 9.23 |
| After, historical | 66.86 | 7.15 | 59.00% | 34.31 W | 9.23 |
| Before, exact | 57.85 | 3.58 | 56.33% | 30.98 W | 10.39 |
| After, exact | 56.60 | 3.26 | 55.97% | 31.07 W | 10.45 |

The matrix took 246.93 measured process seconds, excluding compilation. All
declared inputs and model fingerprints passed; source state stayed unchanged
throughout the matrix. Peak whole-host use was about 8.6% of physical RAM. CUDA
reports no separate VRAM capacity on this shared-memory machine, so RAM is counted
once. GPU means include startup, recurrent generation, controls and CPU verifier
work; they are not dense-training throughput measurements. The shorter exact
observation reduces inference work, but lower accuracy in its free-generation
contract prevents a quality-equivalent speedup claim.

## Evidence And Verification

Reproduction: `config/experiments/nextlat-memory-observability.toml`, run with
`python3 -m scripts.experiments <manifest>` after building the CUDA release
`evaluate_ruliad_checkpoint` example. Choose a fresh output path for replication.

Evidence root: `target/experiments/nextlat-memory-observability/`:

- `results.json`: four successful cells, `complete=true`, `source_unchanged=true`.
- `evaluation-summary.json`: model/panel identities and full control results.
- `observability-summary.json`: compact quality/resource table.
- `matched-contract-audit.json`: identical policy item identities, unchanged
  canonical/training-serialization outputs, and unchanged no-context decisions
  across prompt interventions. Each cell retains its binary, declared input
  hashes, logs, full evaluation, RAM trace and GPU samples.

Local checks passed: 203 universality tests (one pre-existing ignored), 130
training-schedule tests, 35 policy tests, 101 dataset tests, and 26 Python
experiment/analyzer tests. CPU policy tests include batched/dense numerical
parity, exact-context overflow, config serialization and paired-target scoring.
The CUDA release build and all four e2e evaluations passed.

Packaging checks also passed: workspace formatting, universality all-target
strict Clippy, `cargo check --locked -p xtask`, workflow YAML parsing, and
`bootstrap_stack.py --verify`. Dragon's public stack lock now pins the PC/ECS
commits used by the experiments and the already-published EGGROLL revision;
P2P's revision is unchanged. CI includes the bounded-experiment contract tests
and is triggered by stack-lock/bootstrap changes. These local checks are not a
claim that the full native/wasm deployment CI matrix is green.

The related PC commit also fixes tiny-positive-target normalization and the
clipped set-NLL/exact-VJP mismatch; 87 PC tests, all-target benchmark smokes and
strict Clippy passed. Those changes are not exercised as a learning algorithm
in this AdamW-checkpoint matrix and do not establish a new PC convergence result.

## Next Learning Gate

1. Separate observation and training-distribution interventions in a 2x2 matrix:
   historical/exact observations versus original/target-balanced action data.
   Use three training seeds, the corrected memory recipe, matched supervised
   answer and scheduled-token budgets, and the same no-context/heuristic controls.
   Include valid retargeted examples and rule-orientation transformations, not
   just additional traces or loss coefficients. Train the decoder on the same
   observation contract used for free evaluation.
2. Pre-register free-answer, paired-target, per-orientation and closed-loop proof
   metrics. Require improvement beyond blind controls, not merely a higher
   typed-menu score, answer-change rate, or lower aggregate CE. Keep original
   reference-action agreement distinct from acceptance of alternative proofs.
3. Test repeated shifts with structural rehearsal against a same-budget stationary
   control. Report d0 retention separately from harder-task acquisition and keep
   held-out confirmation panels separate from curriculum feedback panels.
4. Only then re-ablate PC and JEPA/NextLat against this corrected baseline and
   qualify larger local runs. Browser/P2P deployment is not established by these
   local frozen-model results.
