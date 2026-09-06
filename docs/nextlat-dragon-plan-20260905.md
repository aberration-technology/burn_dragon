# NextLat For Stateful Dragon Training

Date: 2026-09-05. Status: reviewed design and proposed experiments, not an
implemented correction or a promoted training recipe. No training was launched
for this review. Existing research changes in the worktree are left intact.

Implementation follow-up: this document preserves the original review. Corrected
contracts, completed matrices, negative results, and the memory-timescale finding
are recorded in [the experiment report](experiments/nextlat-v2-20260905.md).

## Decision

Revisit NextLat as a precisely specified auxiliary objective on Dragon's actual
stateful language forward. Do not introduce a replacement architecture, another
latent-objective crate, or a new production default yet.

The immediate work is a correction and causal evaluation of the existing
implementation. Its current loss contract differs materially from the reference,
and prior results do not demonstrate reliable proof generation. The intended
benefit is better use and preservation of task-relevant information through time,
not simply smaller hidden-state error or more diverse output.

Implementation order:

1. Fix objective semantics, masking, gradient ownership, and reporting.
2. Establish a genuinely state-consuming CE baseline and NextLat comparison.
3. Test transfer, proof correctness, and retention, separately from curriculum.
4. Add explicit local-PC factors, then measure their interaction with NextLat.
5. Qualify checkpoint migration, P2P participation, and browser execution.
6. Evaluate optional latent drafting and semantic-action lookahead separately.

Each stage has a concrete matrix and an exit criterion below. Failure is a valid
experimental result; none of these mechanisms guarantees frontier capability.

## Paper Review

Reviewed [arXiv:2511.05963v4](https://arxiv.org/pdf/2511.05963v4), dated June 15,
2026, and the [official implementation](https://github.com/JaydenTeoh/NextLat).
The repository's main revision observed during review was
`3770be6009cea2b3c455a9ce7f2ca88b504bb955`.

NextLat predicts the next pre-logit state from the current state and next token,
alongside CE. Its belief-state theorem assumes exact token and transition
consistency; it is not a convergence or continual-learning guarantee. The reported
1.3B/100B-token comparison gives average accuracy 58.82 versus 59.21, but
FineWeb-Edu perplexity 10.52 versus 10.88. Single-B200 training steps/s are 3.09
for GPT and 3.09/2.79/1.73 for NextLat horizons 1/2/8. The maximum 3.3x speedup
concerns speculative inference, not training. Draft length is fixed per decode.
Appendix E reports rising latent error despite improving draft acceptance.
[Paper, Theorem 3.2, Tables 2-4, Appendix E](https://arxiv.org/pdf/2511.05963v4).

My assessment: the controlled state-tracking motivation is more relevant here
than the small aggregate language-benchmark gain. The paper supports a testable
representation-learning hypothesis, not a claim that an auxiliary loss repairs
our data, verifier, optimizer, or state-routing contracts.

The [reference loss implementation](https://github.com/JaydenTeoh/NextLat/blob/3770be6009cea2b3c455a9ce7f2ca88b504bb955/models/model_nextlat.py)
detaches target latents and KL decoder weights, but propagates accumulated source
hidden-state and action-embedding gradients into the backbone. The apparent
temporary detach in `compute_loss` is not a permanently frozen encoder. Its
separate loss families are averaged over horizon before weighting. Its residual
three-layer GELU predictor uses learned input normalization and ordinary random
output initialization. These details distinguish the reference from our defaults.

The [authors' reproducibility notes](https://github.com/JaydenTeoh/NextLat#reproducibility)
also identify compiler-sensitive results and an extra CE computation used only
for logging. We should reproduce mathematical outputs and gradients, not copy
unnecessary diagnostic compute or assume another framework's speed transfers.

## Current Dragon Audit

Paths below are relative to this repository; line numbers refer to the reviewed
worktree. These are static findings, not newly reproduced numerical failures.

| Area | Current implementation | Consequence and planned disposition |
| --- | --- | --- |
| Predictor | `crates/burn_dragon_core/src/model/dragon/latent.rs:195`, `:212`; three linear layers allocated in `dragon.rs:985` | Already conditions a residual prediction on hidden state plus token embedding. Reuse it. Parameter-free input normalization and zero output initialization are deliberate Dragon variants, not reference parity. |
| Token alignment | `crates/burn_dragon_language/src/train/steps/latent_objectives.rs:260` | Horizon loop correctly shifts action tokens and recursively feeds predictions. Preserve this, with direct alignment and recursive-gradient fixtures. |
| KL gradient | `latent_objectives.rs:303`; `crates/burn_dragon_core/src/model/dragon/latent.rs:272` | Student logits use the ordinary differentiable head. Detaching only teacher logits does not freeze decoder parameters on the KL branch. Add a parameter-detached projection that still differentiates its hidden input. |
| Loss scaling | `latent_objectives.rs:329`, `:1023` | Sum is divided by the number of enabled loss components, then again by outer auxiliary components. Turning KL or JEPA on changes the effective NextLat regression coefficient. Replace implicit component-count coupling with independently normalized, explicitly weighted terms. |
| Masks | `latent_objectives.rs:235`; `crates/burn_dragon_language/src/train/next_latent.rs:7` | NextLat receives no document/validity mask, and ignores the parent token-loss mask. It cannot distinguish a valid transition from padding or a packed-document boundary. Hidden and token losses need distinct masks. |
| Target state | `latent_objectives.rs:6` | Detached-student mode can reuse the current hidden tensor. EMA mode calls `forward_hidden(clean_inputs)` without the carried state; pipeline mode changes back to detached student. Do not silently change state or teacher semantics. |
| Architecture scope | `crates/burn_dragon_core/src/model/dragon/forward.rs:93` | `forward_hidden` includes the optional final refiner. Initial experiments must disable refinement or explicitly name which latent is supervised. NextLat does not require an inference-time thinking head. |
| Local PC | `crates/burn_dragon_language/src/config/train/validate.rs:525` | Analytic local PC rejects global latent auxiliaries. Historic state-correction-PC experiments are not evidence that current `training.algorithm=predictive_coding` supports NextLat. Implement a local factor; retain the rejection until covered. |
| State routing | `crates/burn_dragon_language/src/config/train/structured_schedule.rs:6`; `train/steps/train_step.rs` | Required self-contained policy/completion schedules can replace every streamed update. Maintaining unused rho on the side does not establish stateful learning. Verify actual forward inputs per objective. |
| Scheduling | `latent_objectives.rs:719`; `config/train/schema.rs:1082` | Existing start/cadence controls are useful. `normalized_aux_scale` here is a static multiplier plus ramp, not a measured gradient normalizer. Log the resolved coefficient and exposure count. |
| Unsupported head | `latent_objectives.rs:303` | Requested token KL is silently omitted for factorized heads. Reject that configuration until an exact compatible KL implementation exists. |
| Historical profiles | `crates/burn_dragon_p2p/deploy/profiles/ruliad-r1.nextlat-fixed-ablation.toml` | Uses `[9999]` to suppress JEPA, detached action embeddings, cadence 8, and a 0.01 outer multiplier. Use an explicit empty JEPA offset list and local experiment overlays, not sentinel offsets or new deployment defaults. |

The KL and scaling findings invalidate a simple inference that prior token-KL
losses were intrinsically unhelpful. They do not prove that fixing them will
improve verifier results. Both propositions require new controlled measurements.

### Existing Results, Not A New Ablation

The [historical latent report](ruliad-r1-latent-ablation-report.md) contains
mixed evidence. Its 4,096-step, three-seed comparison reports:

| Historical arm | Validation mean | Verifier rate | Schema wrong |
| --- | ---: | ---: | ---: |
| JEPA auxiliary | 0.6163 | 0.0208 | 0.3333 |
| JEPA + NextLat | 0.6186 | 0.0208 | 0.4167 |
| JEPA + weak delayed/sparse NextLat | 0.6111 | 0.0208 | 0.6146 |

Its 16,384-step single-run endpoints are more favorable to delayed NextLat:
validation 0.5444 versus 0.5308/0.5264; verifier 0 versus 0.016/0.016; wall time
1,021 versus 1,017/1,019 seconds. These are suggestive archived results, not
confirmation: few verified outputs, old corpus/probe contracts, and no new
source-matched rerun here. The report's initial scope is CPU; later sections mix
execution settings without a uniform per-row backend/batch manifest. Recover
those manifests before using these timings as GPU throughput evidence. Do not
reinterpret old `Valid` values as current full-answer NLL.

The current [policy-controls report](experiments/ruliad-policy-controls-20260905.md)
is a different experiment: frozen 1M-class, seed-13 checkpoints, CUDA f32,
inference batch 2, training batch 8, 256 policy and 256 generation items:

| Checkpoint | Reference-menu accuracy | Same model, no context | Free verified |
| --- | ---: | ---: | ---: |
| AdamW, 1,024 updates | 70.70% | 73.44% | 0/256 |
| Local PC, 1,024 updates | 70.70% | 73.83% | 2/256 |

Each evaluation took approximately 129 seconds; this is not training throughput.
Neither row is a corrected-NextLat run. The finding motivates stronger causal
panels: selecting a reference action from an oracle menu is insufficient evidence
of context-dependent reasoning. All claims below must survive those controls.

## Proposed Objective Contract

Use `D` for embedding width, `N` for neuron width, and `s_t` for the complete
recurrent state. Keep these distinct:

```text
(s_t, h_t) = Dragon_theta(s_(t-1), x_t)
prediction_0(t) = h_t
prediction_k(t) = F_psi(prediction_(k-1)(t), Embed_theta(x_(t+k)))
target_k(t) = stop_gradient(h_(t+k))
```

`h_t` is the final pre-logit tensor used by CE, with latent refinement disabled
in the initial reference arm. `s_t` includes each layer's memory, normalization
state, positions and any kernel-specific state. The compact readout is not
assumed to encode everything in `s_t`.

This differs from the current JEPA auxiliary: its future prediction does not
receive the intervening tokens, whereas NextLat is a token-conditioned state
transition. Neither is the same operation as refining a latent repeatedly while
holding token time fixed. Keep prediction horizon, TBPTT credit length and
inference-time thinking steps as separate configuration dimensions.

For each valid horizon, define independently normalized terms:

```text
H_k = sum(valid_transition * SmoothL1(prediction_k, target_k))
      / (D * count(valid_transition))
K_k = sum(valid_decode * KL(teacher_probability || predicted_probability))
      / count(valid_decode)
L = CE + lambda_h * mean_k(H_k) + lambda_kl * mean_k(K_k)
```

Teacher probability uses detached actual target logits. Predicted probability
uses the same decoder values, detached as parameters but not as a function of
the predicted hidden input. Average only horizons with valid support; export
support counts. Empty support produces a finite zero and no optimizer update
unless another objective has signal. Keep CE's own supervision denominator.

### Gradient Ownership

| Path | Backbone/source hidden | Predictor | Action embedding | Decoder parameters | Target branch |
| --- | --- | --- | --- | --- | --- |
| Ordinary CE | Yes | No | Through ordinary input use | Yes | Labels only |
| NextLat regression | Yes, within declared credit window | Yes | Yes in reference arm | No direct path | Detached |
| NextLat KL | Yes through predicted hidden | Yes | Yes in reference arm | Detached on this branch | Detached |
| Frozen-backbone predictor diagnostic | No | Yes | Detached | Detached | Detached |

With tied input/output embeddings, the shared tensor may receive legitimate
input/action gradients despite a detached decoder projection. Test contributions
by path; do not assert that the shared tensor's total gradient must be zero.
Detach all decoder-side conditioning parameters too if a later experiment enables
step-conditioned decoding.

Do not detach predicted states between horizons in the reference objective.
Do not supervise `x_(t+1)` from a predictor that has already received that token.
Prediction of `h_(t+1)` is evaluated against the distribution for `x_(t+2)`.

### Coefficients And Initialization

Start the corrected reference objective with `lambda_h=1`, `lambda_kl=0`,
beta 1, horizon 1, no extra global multiplier, no EMA, and no delayed start.
The KL reference comparison uses `lambda_kl=1`. Those are declared reference
settings, not supposedly optimal Dragon coefficients.

The initial reference disables input corruption, dropout, refinement, EMA,
state-consistency losses, continual-backprop replacement and automatic width
changes. Keep tokenization, learning-rate schedule and primary CE mask fixed.
Noisy-view prediction is a separate contract: do not accidentally use corrupted
source states with clean action inputs while calling the target a same-pass
reference.

First distinguish objective corrections from architecture initialization changes:
hold the existing residual predictor fixed for the contract audit, then compare
learned input normalization/random output initialization in one matched arm.
All baseline backbone tensors are identical; auxiliary allocation must not
advance the backbone initialization RNG. Export backbone and auxiliary parameter
counts separately. With the current hidden multiplier 2, its three linear layers
contain `10*D*D + 5*D` parameters including biases: 92,640 at D=96 and 656,640
at D=256, before any learned normalization. This is not negligible for a 1M model.

Measure gradient RMS and cosine conflict for CE versus auxiliary objectives on
bounded diagnostic steps, not every training update. If coefficients need tuning,
use a declared development-only grid `lambda_h={0.1,1}`,
`lambda_kl={0,0.1,1}`, with equal selection budget for controls. Freeze the choice
before confirmation. Reject hidden automatic rescaling or post-hoc composite
scores as an explanation of success. Sparse cadence also changes cumulative
objective exposure; report it, and do not silently multiply sparse losses by the
inverse sampling frequency.

## TBPTT And Rho

### Stateful Primary Forward

The learning forward, not a separate maintenance pass, must load and return the
same run/stream-keyed state. Every batch records document identity, stream slot,
start position, resets, token validity, and token-supervision masks. Preserve
state across physical windows of one document and reset at true document edges.
Continuation rows must not secretly repeat the complete original prompt.

Retain separate counts for input tokens, CE-supervised tokens, latent transition
pairs, optimizer updates, and processed documents. A context-only chunk can
produce a real latent update when valid transition pairs exist; enabling a
NextLat flag is not enough to bypass the existing zero-signal guard. Validate
schedule coverage, horizon support and state availability first. Update the
guard only alongside gradient tests proving the new supported path.

### Boundaries And Credit

For horizon k, a valid transition must remain within one uninterrupted document
for every intermediate step, not merely have a valid destination token. Token
KL additionally respects the supervision mask of the decoded future token.
Prompt tokens may participate in hidden regression without becoming CE targets.

Begin with within-chunk latent pairs and real carried rho. Export omitted
boundary-pair counts. This already tests whether the objective improves the
information in the memory that actually conditions later chunks.

The next boundary arm retains at most k source hidden vectors per stream as
detached anchors inside one optimizer revision. These can train the predictor
and action embedding against the next chunk, but cannot retrospectively train
the previous chunk's encoder. Name this contract `detached_boundary`; do not
describe it as extended TBPTT credit. Discard anchors at optimizer updates,
document resets, stream reassignment, checkpoint load, or architecture change.

Compare credit-window lengths 1 and 2 separately using the existing bounded
temporal-credit mechanisms. Keep weights unchanged during each credit window.
Never retain an entire growing document graph or compare stale latent coordinates
across optimizer revisions as if they were current targets. If exact window
support is unavailable for a selected executor, fail configuration validation.

EMA is a separate later arm. Its teacher must consume the same document history
through its own bounded state, using a specified update cadence. Do not feed a
warm student a cold teacher target, or substitute detached student silently.
Measure teacher cost and target lag; do not cache historical latent targets
indefinitely. First reference experiments avoid this extra encoder entirely.

### Memory Supervision Is A Separate Hypothesis

For Dragon, the complete recurrent state is already updated recursively, but
its D-dimensional output need not be sufficient to advance every layer's memory.
Shared weights do not eliminate layer-specific state or its temporal credit.
Therefore distinguish compact-output sufficiency from complete-memory stability.

Do not force raw rho toward an isotropic Gaussian, or force distinct proof states
to coincide. The linear memory kernel combines query/value outer products with
decay; sign, normalization and geometry depend on both features and the selected
kernel. ReLU on one input alone does not imply a positive-semidefinite memory.
Mamba/GDN2 states also cannot inherit an assumed linear-attention rho geometry.

After output NextLat clears its gate, test a bounded memory-readout variant using
existing per-layer memory read operations at sampled positions. Predict future
readouts or held-out executable state queries, not a dense flattened rho tensor.
Keep dimensions and sampling cost independent of N where possible. Start with
offline probes before adding any new training loss. Do not allocate a covariance
matrix over all neurons or materialize rho at every token.

Low effective rank can represent useful compression. Diagnose collapse with
state discrimination, task performance, conditional output diversity and probe
quality together. Raw latent loss, marginal entropy and rank alone are neither
sufficient promotion gates nor reliable automatic stop conditions.

### Stability Without Erasing Memory

Our working error model for a learned transition is
`e_(k+1) <= L_k * e_k + epsilon_k`, where `epsilon_k` is one-step approximation
error and `L_k` is sensitivity around the visited trajectory. NextLat can reduce
one-step error without bounding accumulated sensitivity. Conversely, forcing
every state direction to be strongly contractive can erase delayed facts.

Measure finite perturbation growth along actual and predicted trajectories,
separately for relevant-state changes and nuisance changes. Report raw latent
distance together with decoded distributions and executable query accuracy.
Do not equate increasing latent distance with task failure. Prefer a state that
retains useful distinctions and rejects irrelevant perturbations over one that
merely contracts all inputs to the same point.

There is also a capacity tradeoff: requiring a small D-dimensional output to
summarize all useful information in much larger rho can constrain Dragon's memory
capacity. If NextLat helps local loss but harms delayed-query performance, test
a bounded memory-conditioned predictor against the hidden-only one before
increasing loss strength. That variant no longer tests sufficiency of h alone;
declare its additional state inputs and compute explicitly.

## Ruliad Tasks And Evaluation

NextLat cannot manufacture semantics absent from the data. Build the new panels
on the existing executable proof/program structure in `burn_dragon_universality`,
not another collection of arbitrary text formats.

1. **State tracking:** execute compositions, substitutions or rewrites, then ask
   a state-dependent question after a delayed query. Include distractor histories
   with matched token statistics but different semantic states.
2. **Equivalent histories:** construct multiple certified paths to the same
   state and goal, compare future distributions and solve outcomes. Use cases
   where equivalence is decidable; preserve variable-renaming maps.
3. **Legal detours:** start from an off-reference but valid proof prefix, then
   measure model-generated valid continuation and goal completion.
4. **Equivariance:** consistently rename symbols, reorder independent premises,
   and transport predictions back before comparison. Do not require coordinate
   equality of raw latent vectors under renaming.
5. **Long documents:** put necessary facts 2, 4, 8 and 16 chunks before the
   answer. Compare correct state, cold state and another example's state.
6. **Structural difficulty:** independently vary dependency depth, branching,
   expression nesting, delayed recall and composition novelty. Token count is
   one axis, not a substitute for all others.

Select an existing finite, independently checkable subset for an initial success
criterion. Audit its state transitions against a structurally independent small
interpreter; replay through the same production kernel is not independent proof.
Use malformed certificates and verifier-mutation fixtures to test rejection.

Do not force a particular reference proof as the only correct reasoning trace.
Score executable validity, goal completion, and resource cost; allow multiple
valid solutions. Hidden targets come from current executions, not a canned
chain-of-thought string. Any use of canonical successful trajectories, proof
labels, or additional semantic observations must be separately disclosed and
identical across compared learners.

Primary metrics: independent solve rate, valid generated-step rate, solve rate
at fixed token/executor budgets, full-answer sequence NLL, and held-out
length/composition generalization. Report teacher forcing, oracle-menu selection,
grammar-constrained generation, free generation and closed-loop proof search in
separate columns. Budget failures count as failures, not dropped examples.

Mandatory controls: no-context, context swap, state reset, wrong-state injection,
candidate-length heuristic, executable one-step heuristic, and candidate support
coverage. Reset controls are inference-only unless a legal training contract is
explicitly provided. Do not revive invalid masked-stream reset profiles.

Latent diagnostics: horizon-specific raw and target-RMS-relative error, identity
and token-only predictor baselines, decoded KL, predictive probes at held-out
offsets, and decoded sensitivity to bounded state perturbations. Train frozen
linear probes with their own disjoint fit/evaluation splits. A low prediction
error that also occurs with shuffled targets or no context is not useful evidence.

For continual learning use a predefined A/B/C/A stream, plus a stationary stream
control. Report acquisition curves, worst old-task regression, retention AUC,
backward transfer and recovery tokens. Fixed panels must stay fixed while live
difficulty telemetry separately reports probability, loss and verified mastery
by structural bucket. A materialized frontier is not a demonstrated ability level
or a maximum permitted generator complexity.

## Modular Implementation

Extend existing boundaries instead of introducing a second training framework:

| Owner | Work |
| --- | --- |
| `burn_dragon_core` | Reuse transition parameters; isolate transition forward helpers under `model/dragon/next_latent.rs`; add a typed frozen-parameter decoder view; numerical fixtures beside these modules. |
| `burn_dragon_language` config | Move NextLat-specific schema/validation into dedicated modules, re-exporting the existing public types. Keep `model.next_latent_transition` and `training.latent_reasoning.next_latent` as the canonical namespaces. |
| `train/next_latent/` | Refactor the existing helper file into `loss`, `mask`, `schedule`, `telemetry`, `tests`; own independent reductions, pair eligibility and diagnostics. Keep `steps` as orchestration. |
| Dragon training plugins | Add a typed objective plugin with config, counters and boundary cache attached to each run entity. Publish run-keyed aggregate events through existing ECS metrics/sinks. No boxed setup closures, global teacher, or per-token ECS entities. |
| `burn_pc` | Reuse generic factor/criterion contracts; add generic residual/VJP composition only where genuinely reusable. No Ruliad or Dragon model configuration in this crate. |
| Dragon local PC | Implement the token-conditioned latent factor and parameter ownership in `train/local_predictive_coding/next_latent.rs`, using the same masks/reductions as the backprop reference. |
| `burn_dragon_universality` | Verified state-equivalence, detour and composition panels built from existing programs/proofs; expose typed metadata independent of optimizer. |
| Experiment tooling | Typed local overlays in `config/language/experiments/next_latent/`; explicit matrices in `config/experiments/`; use `scripts.experiments` for archival, sequential execution and safety. |

Leave existing JEPA/energy/state-consistency behavior unchanged except explicit
normalization migration covered by fixtures. Eliminate sentinel offsets in new
profiles. Avoid a broad rename or refactor of unrelated training features.

Persist a resolved objective-contract version in the run manifest/checkpoint,
along with loss units, masks, target source, initialization, schedules, tokenizer,
state/credit policy and auxiliary optimizer slots. Do not silently resume an old
checkpoint under changed loss normalization. A deliberate weights-only branch
may adopt the corrected objective, with its ancestor and reset state recorded.
Provide one-way config migration, not permanent legacy execution modes.

Width changes retain D and compatible predictor parameters; invalidate boundary
caches, rebuild appropriately sized memory, and preserve optimizer slots by
parameter identity. An embedding-width change is a different architecture
migration and must not be treated as ordinary appended neuron growth.

## Local PC Contract

First validate exact local VJPs against a tiny backprop reference for the same
semi-gradient objective. The target is detached, not an independently optimized
activity seeking an easier label. Clamp reference target latents for one outer
parameter revision and do not update them during a solver sweep.

Represent the auxiliary transition as a typed factor over source hidden,
action embedding and predicted hidden. Sum its source errors with the CE
terminal contribution, and route local errors through the existing factor graph.
Accumulate tied-weight contributions across uses and apply one outer parameter
update. Predictor parameters receive exactly their own declared contributions.

Analytic GELU/normalization/linear/SmoothL1/KL VJPs or bounded factor-local
autodiff are implementation choices; neither permits a hidden end-to-end
backward call. Test and export `global_backward_calls=0`. Do not call the
procedure optimizer-free: AdamW can still apply locally obtained parameter
gradients. Approximate PC solvers are tested after exact-gradient controls.

Use a 2x2 comparison: backprop versus local PC, each with and without corrected
NextLat. Hold the objective, data order, state contract, optimizer and parameter
initialization fixed. Keep dropout, hierarchy, model scaling and unrelated
auxiliaries disabled until their separate local factors exist.

## GPU Efficiency And Safety

The default comparison must reuse the primary forward's hidden states and token
embeddings. Do not add a second backbone forward for each horizon. Batch all
valid starts, keep the small horizon loop on-device, and fuse pointwise loss and
mask reductions where measurements justify it. Teacher targets are detached
views of the same pass in the initial arm.

KL may dominate memory at large vocabulary sizes. Start with hidden-only loss;
for KL, use bounded vocabulary/position tiles and stable FP32 log-sum-exp and
reductions, numerically checked against the dense implementation. Tiling forward
alone is insufficient if backward retains all intermediates: measure retained
activations and implement recomputation/custom local VJPs if needed. Top-k KL is
a different objective, not an exact memory optimization.

Use static supported shapes, bounded prefetch, stable stream batches, no scalar
readbacks in the inner loss loop, and asynchronous aggregate telemetry. Separate
data assembly, forward, auxiliary, backward/local solver, update, evaluation and
checkpoint timing. Report CUDA event timing and profiler-derived host gaps,
alongside end-to-end tokens/s and joules per processed token. High power alone
does not prove useful work; low whole-evaluation power is not a training trace.

Begin with the previously safe 1M-class dimensions (D=96, N=3072, four tied
layers, four heads, flat head, linear attention), release CUDA f32. Start batch
2 for mechanics; benchmark 2/4/8 only after projected bounds pass. Use the same
effective batch for learning comparisons; population count is not applicable.
Record actual unique trainable parameters instead of relying on a size label.

Use the existing external memory guard and sequential runner. Apply the user's
90% ceiling to the smaller applicable physical RAM/VRAM budget, counting unified
memory once. Reserve launch/shutdown headroom, account for current non-job use,
and reject unsafe predicted peaks before allocation. A watcher cannot prevent
every instantaneous allocation spike, so pair it with conservative static bounds
and isolated limits where supported. No exhaustive batch search or OOM probing.

Initial engineering budgets: at least 90% of baseline training throughput for
horizon 1 and 80% for horizon 2, at equal batch and data, with all logging costs
included. These are proposed acceptance targets, not measured outcomes. A slower
candidate needs a demonstrated equal-wall-time quality advantage. Validation
and startup costs are separately reported, never hidden outside the comparison.

## Experiment Program

All rows below are planned, not completed. Stage budgets are ceilings; mechanics
or quality failures stop expansion. Archive sources, sibling revisions, binary,
resolved configuration, initial tensors and every realized batch/objective hash.
No changes to active experiment sources, no concurrent GPU learners.

After E1, derive per-arm wall-time ceilings from measured end-to-end update cost
plus the scheduled evaluator cost, with a declared 50% execution margin. Stop
and investigate timeouts instead of silently extending a run. Publish projected
matrix GPU-hours before E2; the table is a sequential, gated program, not an
instruction to launch every combination at once.

| Stage | Conditions and budget | Output and gate |
| --- | --- | --- |
| E0: numerical contract | CPU tiny arrays and bounded CUDA fixtures; horizons 1/2/4, ragged/packed/tiny documents, tied/untied head | Loss/VJP parity; zero forbidden gradients; correct masks; unchanged disabled forward. No learning claim. |
| E1: throughput | Safe 1M geometry, batches 2/4/8; CE, hidden h1/h2, h1/h2 + KL; 64 warmup + 256 measured updates, three repetitions | Phase timings, memory peaks, host gaps, tokens/s, parameter counts; meet budgets or optimize before expanding. |
| E2: useful state | CE; CE + hidden h1; CE + hidden h2; CE + hidden/KL h1; frozen-backbone predictor control; 4,096 updates x 3 seeds | Same state-consuming stream, correctness and retention panels; improvement must exceed identity/token-only/no-context explanations. |
| E3: reference fidelity | Best hidden arm versus action-embedding detach and predictor initialization/normalization controls; 4,096 updates x 3 seeds | Separate objective benefit from architecture/RNG changes. Do not combine every control into one changed arm. |
| E4: historical challengers | Matched CE baseline, CE + JEPA, CE + corrected NextLat, CE + JEPA/NextLat; 4,096 updates x 3 seeds | Recover value of historical delayed/sparse schedules on current data, with independent coefficients and identical cadence accounting. |
| E5: state/credit | Baseline and best NextLat; chunk 64/256, credit window 1/2, then detached-boundary control; 4,096 updates x 3 seeds, staged not full Cartesian product | State intervention effects and delayed-query accuracy. Export all omitted/cross-boundary pair counts. |
| E6: local PC | Backprop/local PC x CE/best NextLat; 4,096 updates x 3 seeds, only after local VJP gate | Same objective and effective batch; compare solver residual, gradients, verifier, throughput; global backward count zero for PC. |
| E7: continual confirmation | Baseline and at most two finalists; 16,384 updates x 5 seeds on A/B/C/A and stationary controls | Fresh confirmation panels, retention/acquisition curves, equal-token and equal-wall-time frontiers. No production promotion from one seed. |
| E8: scale | Winning baseline/challenger, measured 10M class; 32,768 updates x 3 seeds after safe calibration | Quality/resource Pareto result, long-document generalization, stable memory. Test 16K+ neuron widths only if their measured bounds pass. |
| E9: deployment | Deterministic two-native-peer replay, then one-native/one-WebGPU peer and an observer; bounded 256-update parity followed by 4,096-update paired runs | Revision agreement, initial tensor equality, update weighting, bandwidth/token, convergence and native/browser phase timing. |

Use seeds 13/29/47 for screening and predeclare five confirmation seeds in the
manifest. Seed reuse across arms gives paired comparisons, not independent
samples. Use a development panel distinct from repeatedly inspected panel 73;
freeze a fresh confirmation panel before finalist selection. Initial offline
generation panels have at least 256 items; confirmation uses at least 1,024
stratified problems where practical. Confidence intervals must account for both
training seed and problem grouping, not treat tokens as independent trials.

For E2-E6 first use one executable state-tracking subset and fixed difficulty
mixtures, plus a bounded natural-text control. Expand to held-out proof programs
only after mechanics and useful-state tests pass. Report difficulty distributions
and useful supervised tokens, not just equal update counts. Record all early
terminations in the result table.

Predeclared decision rule: a finalist must improve independently verified
completion or retention, have positive paired context/state benefit on tasks
constructed to need those inputs, and preserve acquisition on new tasks at an
equal compute budget. For noninferiority, use a proposed margin of 1 percentage
point on old-task solve rate and 2% on full-answer NLL, with confidence bounds;
freeze margins before confirmation. When success is near zero, noninferiority
is vacuous: require an absolute useful solve target (initially 50% on the small
decidable state-tracking subset) before any reasoning-pipeline promotion.

That target is an engineering gate on a named finite subset, not a claim about
arbitrary mathematics. Broad Ruliad claims require positive held-out
composition/length results; continual claims require A/B/C/A retention and
stationary-control stability. Never promote solely on latent error, CE,
reference-menu accuracy, entropy, or a composite dashboard scalar.

## Tests And Verification

- Finite differences away from SmoothL1 kinks for source hidden, predictor,
  action embeddings, and decoder paths; stable KL at extreme finite logits.
- Hand-computed h1/h2 fixtures proving recursive predictions, not repeated
  independent one-step predictions or a token off-by-one.
- Enabling KL does not rescale regression; enabling JEPA does not rescale
  NextLat; disabled auxiliaries preserve original outputs and gradients.
- Padding, document resets, all-invalid rows, single-token chunks, and horizon
  longer than a document yield correct masks/counts and no NaNs.
- Full versus chunked causal forwards agree at frozen weights; tests separately
  assert expected TBPTT gradient truncation instead of demanding full-BPTT parity.
- Same-document state carry and mixed-length batch slot reassignment never leak
  another document's state. Boundary caches invalidate on every revision/reset.
- EMA targets use matched histories and cannot fall back to another target mode.
- Local-PC exact factor gradients match the reference, tied-parameter reductions
  are correct, and approximate solvers expose their deviations.
- Checkpoint round trips preserve predictor and optimizer slots; old contracts
  require explicit migration; neuron widening preserves compatible parameters.
- Two run entities with different schedules and data do not share caches,
  teachers, metrics or control actions. Observer-only execution allocates no model.
- CUDA and WebGPU loss/VJP fixtures use declared tolerances, FP32 reductions,
  and rejection tests for unsupported combinations. No backend-specific silent
  change to loss weights, horizon, masks or parameter ownership.

Run scoped Rust tests serially using rustup toolchain binaries, followed by
release CUDA numerical fixtures and E1. Tests prove mechanics; only E2 onward
can support claims about learning behavior.

## P2P, Browser And Optional Inference

Keep objective semantics in Dragon. `burn_p2p` transports a generic signed
training/revision contract containing an objective fingerprint, parameter schema,
sample/transition counts and negotiated capabilities. NextLat predictor weights
are trained parameters and must enter initial synchronization, aggregation and
checkpoint identity; they are not untracked peer-local adapters.

Recurrent state is stream-local, not an averaged parameter. On a new canonical
weight revision, use a specified document boundary or bounded replay protocol
to reestablish state; discard latent anchor caches. Reduce per-objective gradient
sums by their own global valid counts. Do not average already-normalized peer
means when document lengths or transition supports differ.

Changing batch size is acceptable under matched effective-batch accounting.
Changing horizon, objective, mask or precision contract is not an automatic
capability downgrade. Negotiate an equivalent implementation or become an
observer/verifier; do not silently train a different objective. Test duplicate
and stale updates, reconnect, heterogeneous microbatches, and multiple run IDs.
Observer peers subscribe to aggregated telemetry without receiving private
training examples or allocating model buffers.

After local promotion, inspect synchronized model loss and verifier curves on
two native peers, then native plus browser WebGPU. Report bytes per useful token,
predictor update payload, revision latency and actual optimizer count. No expected
P2P bandwidth reduction follows just from adding this objective.

Two optional research directions remain separately gated:

1. **Latent drafting:** reuse the trained predictor to propose tokens, then
   validate with canonical Dragon. Exact stochastic acceptance and rollback must
   reconcile every layer's rho, counters and positions; greedy-prefix acceptance
   alone does not preserve a sampling distribution. Benchmark prefill, draft,
   verification and state restoration. Do not replace canonical rho with a
   predicted compact hidden vector. This is inference acceleration, not better
   reasoning by itself.
2. **Semantic-action dynamics:** condition on typed executable proof actions
   instead of individual byte/token steps, with intermediate program states as
   evaluation targets. This changes temporal resolution and is a new experiment,
   not the token-level reference method. Generate and verify actions without
   injecting the preferred reference into inference menus; compare actual solve
   rate at equal search/verification cost against existing action scoring.

Neither direction requires an EBM, SIGReg loss, adaptive halting, EGGROLL, or
model-width growth in the first comparison. Add one interaction only when its
individual baseline is already measured. If these extensions remain unhelpful,
retain the corrected objective as an opt-in experiment and keep the simpler
stateful baseline; do not interpret inconclusive results as a reason to stack
more mechanisms.

## Deliverables

The first implementation deliverable is corrected masking/gradients/reductions,
resolved objective manifests, numerical tests, and the E1/E2 tables. The next
decision is based on verified task behavior and state interventions, not another
long unattended run. Complete the later matrices only as their stated gates pass.
All training and deployment behavior remains unchanged by this planning document.
