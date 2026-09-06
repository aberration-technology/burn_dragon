# Dragon Training Roadmap

Date: 2026-09-05

Status: the evaluation-contract and bounded maintenance pilot are implemented and
measured; remaining learning and delivery promotion phases are not complete. See the
[execution record](experiments/ruliad-evaluation-contract-20260905.md) for completed,
interrupted and timed-out measurements. This roadmap is not a promotion report.

The [policy-control follow-up](experiments/ruliad-policy-controls-20260905.md)
completed 12 frozen-checkpoint CUDA evaluations, per-item chance/heuristic and
no-context controls, kernel cache audits, and a scorer-component ablation. All six
checkpoint conditions scored higher without context on reference-action accuracy.
The evidence does not support recipe promotion. The archived masked-stream reset
arms are now rejected by the launcher; the supported TBPTT sweep is carry-only.

## Executive decision

Dragon has substantial reusable infrastructure: tied-weight recurrent models, local
learners, a verifier-backed synthetic source, run-scoped ECS orchestration, signed
P2P revisions, and an executable WebGPU trainer. The evidence does not yet establish
a state-of-the-art reasoning model, durable continual learning, or an adversarially
production-ready public training network.

The next milestone should be a reproducible, efficient, stateful reasoning learner
whose held-out competence survives distribution changes and whose learning contract
is preserved when distributed. More objectives, more difficulty indices, and more
peers are not substitutes for that result.

Run two coordinated tracks:

1. **Learning:** repair the gap between semantic decisions and complete solutions;
   establish continual acquisition and retention; then test architectural scaling.
2. **Delivery:** complete hardware-browser participation and lifecycle accounting;
   establish heterogeneous convergence; then optimize communication and admit
   increasingly untrusted contributions.

AdamW remains the reference, not an architectural commitment. Exact local PC,
JEPA/NextLat, and random scaffolds remain named challengers. Promote a challenger
when its measured quality/cost tradeoff wins, not because its implementation exists.

## Evidence and provenance

This review examined the working research checkout at `2073b90` on
`agent/layer-local-pc`, its uncommitted changes and local experiment artifacts, and
the separate main worktree at `18da600`. These are different code states. The
research worktree has about 15,577 added lines in tracked diffs, plus untracked
implementation files; it is not an immutable release artifact.

No new training, benchmark, deployment, or test run was performed for this planning
review. Results below are inspected existing evidence. A merged browser PR is not
proof that its deployed training contribution has been accepted.

| Evidence | Configuration | Recorded outcome | What it establishes |
| --- | --- | --- | --- |
| Clean PC replication, August 8 | 937,154 parameters, CUDA, batch 32, 512 updates, five seeds, persistent rho, TBPTT 64, exogenous source | AdamW/PC cold CE 1.0889/1.0962; constrained solve 0.4625/0.4813; free verifier 0/0; model throughput 55,310/41,704 tok/s | Bounded PC quality parity, approximately 24.6% model-throughput deficit; no free-proof or continual-learning promotion |
| Decoder-coupled screen, August 12 | 1M-class profile, shared weights, 4 layers, embedding 96, neuron width 3,072; CUDA, batch 8, block 1,024, TBPTT 64, 1,024 updates, one seed | AdamW/PC CE 0.06058/0.05979; structured verifier 0.6562/0.6875; free document verifier 0/0; model throughput 33,860/23,480 tok/s | Strong decision-versus-generation gap remains on the newer objective |
| Same decoder-coupled screen | Same executable hash, dirty source tree | Wall times 349/477 seconds; wall throughput 24,440/17,860 tok/s; host peak-used telemetry 57,052/56,629 MB | Bounded exploratory comparison, not a clean replication or model-only memory measurement |
| Stateful Ruliad screen | 4 layers, embedding 128, neuron width 512; CUDA, batch 32, block 512, TBPTT 128, 2,048 updates, one seed | Balanced reset/carry CE 0.6782/0.6452; verifier 1/32 versus 3/32; carry model throughput 166,250 tok/s | Carry is promising; all three carry successes were difficulty-0 advance actions, not general theorem construction |
| Corrected native artifact-window gate | 926,210 parameters, three native peers, two rounds, nine local steps per peer per round | CE 5.718210 -> 2.258732; matched synchronized final 1.930078; 91.324% progress ratio; exact replay and restart passed | Short, trusted local transport and convergence evidence |
| Browser performance PR 34 | NCA R4, 1,617,922 parameters, real Chromium/WebGPU, batch 6, 64 train batches | Three fresh runs mean 6,720 training tok/s, 78.2% in-window duty; controlled 8-to-64-batch sweep 4,473 -> 6,939 tok/s | Meaningful browser improvement; not Ruliad quality, session-wide useful throughput, or a live accepted receipt |
| Random scaffold report | Small local CUDA model, rank 16, 2,048 updates, three seeds | CE 7.18% lower, verifier +7.29 percentage points, training throughput 95.58% of dense; native P2P mean progress ratio 0.9308 | Promising bandwidth candidate; selection and confirmation are not independent, and browser/WAN/large-scale quality remains open |

Evidence locations:

- [Clean PC report](predictive-coding-dragon-report.md) and
  [machine-readable capability contract](experiments/predictive-coding-capability-contract-20260808.json).
- Local `target/pc-paper/decoder-coupled-screen1024-20260812/analysis/paper_tables.md`
  and its `manifests/` directory. Its source commit is `1b4ec1c`, marked dirty.
- Local `target/pc-paper/structured-policy-static-parity512-materialized-3seed-20260812/`:
  three-seed 512-update structured accuracy is 0.6667 AdamW and 0.5938 PC, while
  both free document verifier rates remain zero. The larger closed-loop directory
  has only one completed arm in its status summary; its name is not completion evidence.
- [Stateful TBPTT report](stateful-tbptt-ruliad-report.md),
  [P2P readiness ledger](p2p-production-readiness.md), and
  [random scaffold report](random-scaffold-dragon-report.md).
- [Merged browser performance PR](https://github.com/aberration-technology/burn_dragon/pull/34).

These rows use different corpora, objectives, sequence lengths, and evaluation
contracts. Do not compare their absolute CE or throughput as a longitudinal model
improvement curve. The older JEPA reports also contain historical R1 results and
recurrent-state-correction PC, which is distinct from canonical local-PC learning.

## 1. Define the claims before changing the model

Separate four products and their acceptance evidence:

| Product | Primary evidence | Insufficient substitutes |
| --- | --- | --- |
| Recurrent language learner | Held-out text likelihood, long-context use, coherent generated continuations, matched compute baselines | Synthetic-template CE alone |
| Verifiable reasoning learner | Complete held-out task success, composition/renaming transfer, proof cost, independent certificate checking | Selecting an oracle-provided action, grammar validity, or teacher-forced accuracy alone |
| Continual learner | Repeated novel-task acquisition plus retained earlier capability at bounded state/compute | A single low-loss checkpoint, growing difficulty IDs, or repeated rollback |
| Decentralized training platform | Accepted contributions, heterogeneous convergence, useful work per second/byte/joule, recovery and trust tests | Connected peer count, falling local loss, or exact message replay alone |

The plausible near-term SotA target is a specific Pareto result: independently
verified reasoning/continual-learning quality per training compute and network
byte on a documented hardware envelope. A 1M or 10M model result cannot establish
frontier general intelligence or broad LLM superiority.

Use a quality vector rather than a weighted composite score. For each condition,
publish exact successes/attempts, uncertainty, task strata, CE denominators, useful
tokens/s, solver calls, network bytes, and energy. Require a preregistered primary
endpoint and report all other endpoints even when they disagree.

An optimizer comparison inside Dragon is not an architecture-SotA comparison.
Include the original BDH fidelity configuration, a small conventional Transformer,
and an existing recurrent-kernel baseline at matched parameters and separately at
matched wall time. Evaluate external held-out text/code and executable reasoning
tasks in addition to Ruliad. Record corpus licenses, provenance and structural
decontamination. Test text-only, Ruliad-only, and a declared mixed-token budget;
select the mixture on acquisition/retention tradeoffs, not a hidden auxiliary weight.

## 2. Close evaluation and experiment-integrity gaps first

### One evaluation ladder

Every checkpoint evaluation must explicitly identify its interface:

1. Teacher-forced document loss: syntax, semantic action fields, final values,
   answer length, supervised-token count, and complete-document NLL separately.
2. Constrained action ranking: all presented candidates and their provenance;
   equivalence-set accuracy, candidate coverage, and context sensitivity.
3. Typed action generation: construct an action without an oracle-selected menu.
4. Closed-loop execution: model-visited states, bounded checker/search work, and
   complete certificates, including legal but unproductive or looping actions.
5. Raw token generation: the model serializes its own output; no renderer supplies
   correctness, delimiters, or a certificate after the decision.
6. External transfer: independently specified problems and, for a supported
   formal subset, a second proof checker.

The current structured decoder deliberately renders a selected semantic action
through a deterministic surface formatter. That is a legitimate tool interface,
but its verifier score is not independent of action-ranking accuracy. Keep it;
label it. Never merge it with free-generation or full-proof success.

The newer semantic-action corpus trains `select_proof_action` exclusively. A high
score there does not establish `construct_proof` or `check_proof` competence.
Evaluation must show task support as well as task performance.

### Diagnose the very-low-CE/zero-verifier result

Use one immutable checkpoint and byte-identical held-out problems to compare:

- Full document versus local-policy prompt; cold state versus correctly warmed
  document state; teacher forcing versus incremental decode.
- Free decode versus syntax-only constrained decode versus oracle-menu selection.
- Sum NLL of the complete answer versus token-average CE, and the first wrong
  semantic decision versus later errors propagated from it.
- Identical model parameters before every evaluation: force lazy initialization
  at a defined RNG boundary before cloning; assert no evaluator parameter update.
- Exact tokenizer IDs, answer delimiters, EOS policy, loss masks, truncation,
  padding, position indices, and streaming row identities.

Syntax-only constraints may remove illegal grammar productions. They must not
select only actions known to make progress. If grammar constraints recover quality,
fix the output interface; if oracle menus alone recover it, the search/action
representation still carries much of the task. If changing the prompt fixes it,
train/eval context alignment takes priority over optimizer research.

Use the likelihood accounting identity `NLL(answer) = sum_t -log p(y_t | prefix)`.
A low average CE over many predictable syntax tokens can obscure a few decisive
semantic errors. Teacher-forced answer likelihood and greedy exact success are
different quantities; do not infer either from the other's average. Log the
distribution of semantic-decision errors and sequence NLL by answer length.

### Negative controls and independent checking

- Shuffle candidate order, alpha-rename symbols, permute irrelevant axioms, and
  alter whitespace/token presentation independently of mathematical structure.
- Swap a goal or premise so the answer must change; preserve the candidate set
  where possible. Include matched changes that should leave the answer unchanged.
- Evaluate no-context, nearest-template, random-action, majority-action, and
  deterministic bounded-search baselines. Compute chance from each valid action
  set; it is not necessarily `1 / candidate_count`.
- Mutate a certificate's rule, substitution, path, dependency, or final equality.
  Reject invalid proofs and accept independently constructed alternate valid ones.
- Keep expert certificates and progress labels outside the model-visible problem
  and deployable proposal generator. Log candidate generation separately from
  model scoring and final verification.
- Freeze promotion holdouts. Live curriculum uses a separate feedback panel;
  repeated testing on a promotion panel must not train the source selector.

### Experiment contracts

Extend the existing manifests, do not create a parallel identity mechanism.
Require clean source and sibling `stack.lock.toml` revisions for confirmation;
retain dirty runs as exploration with their patch digest and binary hash.

Bind model/schema, eager initial tensor digest, tokenizer, generator/kernel,
objective normalization, sampler state, masks, exact training data IDs,
evaluation panel, recurrent-state policy, optimizer state, solver, decoder, and
backend/dtype execution details. Separate portable mathematical identity from
backend capability. Cross-backend numerical agreement is tolerance-based;
transport/reconstruction of the same canonical artifact is byte/digest exact.

Use paired seeds and the same exogenous sample schedule to isolate learners.
Compare both equal supervised information and equal wall time. Report auxiliary
rows, repeated candidates, teacher work, extra recurrent steps, and oracle calls
instead of crediting all compute to the base token count.

Screen with 256 stratified tasks; confirm with at least 1,024 and enough samples
per reported stratum. Use task-level intervals for a checkpoint and paired
seed-level intervals for learner differences. Three seeds are screening evidence;
use five independent confirmation seeds and a separate selection holdout. If
intervals are too wide to resolve the promotion margin, collect more evidence or
report an unresolved result, not parity.

## 3. Make Ruliad difficulty semantic and measurable

### Preserve the shared abstraction

Keep `burn_dragon_universality` as the owner of proof IR, generators, kernel,
oracles, portable source descriptors, and distribution tests. The existing
equational/category/logic/automata/process/metagraph frontends should lower into
that contract, not grow separate textual datasets or unrelated reward functions.

Current source inspection matters: `formal.rs::for_difficulty` grows coordinates
by the bit length of a machine-sized integer, caps leaves at 4,096, and constructs
proofs from a small set of wrapping/rewrite laws. Context depth now grows without
cycling; the R3 report's statement that it cycles is stale. Nevertheless, a finite
machine integer and bounded theorem templates do not supply indefinitely growing
mathematical expressivity. Large intervals of difficulty IDs share the same
generator coordinates.

### Replace a scalar claim with a complexity profile

Record requested and realized complexity separately:

- Dependency DAG depth/width, distinct assumptions, lemma reuse, and context nesting.
- Number of simultaneously live variable bindings and substitution scope.
- Independent constraints, required compositions, distractor similarity, and
  branching among valid actions that do not all solve the goal.
- Shortest known certificate length, checker work, bounded reference-search
  expansions, and whether those quantities are measured, upper bounds, or unknown.
- Input/output length, TBPTT chunk count, resource-budget rejection, and generation cost.

Known certificate length is an upper bound, not a proof of intrinsic difficulty.
Verifier runtime is not the model's reasoning difficulty. Match token length while
varying dependency/branching structure, and match structure while varying length.

Report a capability surface, not percentage of an unbounded scale: for example,
`composition depth 8, 12 live bindings, 512 independent tasks, 81% solved [CI]`.
Distinguish sampled difficulty, released curriculum support, highest generated
level, and independently demonstrated mastery. A frontier index is an allocator
coordinate, never a mastery score.

### Extend actual proof structure

1. Add compositional generators with multiple valid proof paths, branching search,
   reversible rules, hard legal distractors, and reusable dependency DAGs. Avoid
   generating only repeated simplifications of a wrapped term.
2. Introduce typed terms and scoped binders where needed, with capture-avoiding
   substitution and explicit proof rules. Category-theoretic generators should
   test typed composition, diagram equivalence, and universal-property witnesses,
   not merely rename a rewrite template as category theory.
3. Add a bounded executable process/metagraph frontend only through shared IR and
   certificates. A metta/rho-calculus-style language is a useful program source,
   not itself a guarantee of proof coverage or learnable difficulty.
4. Maintain a content-addressed library of independently checked lemmas and
   programs. Generate new tasks by composing and reusing them across documents.
   Separate training-library ancestry from held-out theorem ancestry.
5. Stream arbitrarily long finite derivations through continuation descriptors
   and proof chunks. Version an extensible structural address rather than relying
   on overflow or saturation of `usize` difficulty. Enforce finite per-sample and
   per-step resource budgets without silently relabeling a smaller task as harder.

Use a small independently checked Lean subset first. Kernel acceptance means
validity relative to specified assumptions and rules, not truth under arbitrary
invented axioms. Audit the trusted base, prohibit unsafe proof escape hatches in
certification, and track which IR subset has an independent translation.
The [miniF2F repository](https://github.com/facebookresearch/miniF2F) provides an
external formal benchmark direction; it is not an initial expected capability of
a tiny synthetic-only model.

No finite test can show coverage of all computable mathematics, and no general
procedure can promise to solve every theorem. The attainable engineering contract
is extensible proof languages, fair exploration, finite verified samples, and
measured growth on held-out structures.

### Curriculum and efficient data delivery

Retain live source selection for ordinary training. Use the exogenous mode only
for controlled learner comparisons. Maintain an exploration allocation, replay
of mastered structures, and learning-progress estimates from independent feedback.
Apply confidence-aware expansion; avoid interpreting CE-only ease as proof mastery.

Use a bounded active working set and hierarchical sampling over family/task/
complexity so an ever-growing frontier does not create an ever-growing hot-path
table. Snapshot the policy per lease; generate from deterministic portable sample
IDs. Record rejection/resampling probabilities and detect length/resource filters
that distort the intended distribution. Check the same identifiers on native and
wasm32, including large coordinates and overflow boundaries.

Audit adjacent-seed autocorrelation, symbol/answer-position frequencies, structural
deduplication, rename invariance, valid-action multiplicity, and length-conditioned
task balance. Random-looking token bytes are not evidence of semantic diversity.
Publish inspected sample sets at elementary, intermediate and far-out structural
settings with replay certificates, realized complexity and chunk boundaries. Cache
keys include the generator/kernel/split contract so stale samples cannot silently
cross a corpus revision or a validation boundary.

Benchmark generation plus validation plus packing, not generation alone. Target
p95 foreground data wait below 2% of model-step time and producer throughput at
least 1.5 times measured consumer demand on representative admitted profiles.
Use bounded prefetch and caches; never materialize a whole frontier. A document
that exceeds the resource envelope becomes an explicit unsupported/retry event,
not truncation of its target or proof.

## 4. Establish the local stateful baseline

### Architecture invariants

Retain tied Dragon weights, large neuron dimension distinct from embedding/rank,
ALiBi positioning, and persistent rho as the baseline. The
[original BDH implementation](https://github.com/pathwaycom/bdh) is the independent
reference for an explicit fidelity configuration; additional memory kernels and
reasoning modules should be labeled architectural variants.

Test full-sequence versus chunked recurrent forward equivalence, and gradient
equivalence where the same credit horizon permits it. Then test the intentional
TBPTT detach separately. An indefinitely usable recurrent state is not an
indefinitely long differentiable credit path or lossless infinite memory.

The linear reference currently accepts a shape-mismatched rho by creating zeros.
Audit callers so auto-batching and growth cannot silently discard context. Runtime
state must be keyed by document/stream, position, layer, model revision, and row
mapping. Batch resize requires explicit re-packing or named reset/warm replay.
Validation state must never mutate training state or cross independent documents.

### Baseline scale and batch selection

Start at approximately 1M actual parameters, then 10M. Report `num_params`,
trainable parameters, persistent state bytes, and optimizer bytes, not just profile
names. Shared weights mean layer count does not multiply all parameter matrices.

At 1M, compare a width-heavy and an embedding-heavy configuration at matched
parameter count. Separately sweep neuron widths 4k/16k/32k/64k at embedding
128/256 where admitted; these are resource/performance scaling measurements, not
automatically matched-parameter quality comparisons. Every configuration must
pass measured memory admission before execution.

Test block lengths 512/1,024 and TBPTT horizons 64/128/256 sequentially, not as one
large Cartesian product. Preserve complete documents across multiple chunks.
Select batch size from measured useful throughput subject to memory and
optimization constraints. Compare fixed effective batch via accumulation before
claiming an optimizer improvement from larger physical batches.

### Learning objective

Keep CE as the reference proper token-likelihood objective. Fix denominators and
semantic supervision before adding losses. For actions, use likelihood of the set
of valid equivalent decisions, rather than punishing alternate correct proofs.
Declare how decision-level and token-level examples are sampled; alternating
objectives also implies a weighting through their cadence.

Do not require natural-language reasoning-trace imitation. Compare verified final
answers and actions, optional certificate pretraining, and model-visited proof
states. A model-generated intermediate step earns credit for a verified state
transition, not for sounding plausible. Legal action accuracy is auxiliary to
complete-task success because indefinitely cycling legal actions is not reasoning.

Keep compute cost separate from correctness in the promotion criterion. Train a
budget-conditioned policy or use a constrained compute budget instead of inventing
an unexplained weighted sum of CE, proof progress, entropy, and step penalties.

### Continual learning protocol

Run stationary IID controls, gradual structural shifts, abrupt A->B->C->A shifts,
and interleaved mixtures. Keep model width fixed for the primary retention study.
At regular token budgets evaluate frozen old tasks, fresh tasks, and next-frontier
tasks, with both cold and causal warm state.

Track acquisition area under the learning curve, updates/time to a declared
competence threshold, backward transfer, retained solve rate, final-minus-best
loss, and external text retention. Count rollbacks, reinitializations, neuron
replacement, and curriculum retreats; a reset-based run cannot be advertised as
an uninterrupted stable learner.

Use causal diagnostics: update/parameter norm, gradient norms, clipping rate,
optimizer moment scale, active-neuron fraction, sampled hidden effective rank,
rho energy/drift, perturbation amplification, and paired reset/carry quality.
Output entropy alone is not a collapse gate: a deterministic correct task can
have low entropy. Diagnose loss of input dependence and verifier/retention decline.

Compare bounded replay, conservative learning-rate controls, continual backprop,
and sparse/context-routed learning one at a time. Continual backprop must reset
the associated optimizer moments and preserve tied-weight accounting; useful
unit replacement is distinct from indiscriminate model reset. The original
[plasticity study](https://www.nature.com/articles/s41586-024-07711-7) motivates
these controls, but does not establish that the same intervention fixes Dragon.
Separate forgetting old tasks from inability to learn new tasks.

## 5. A focused architecture and local-learning research program

### Memory versus reasoning workspace

The recent [BDH-CQ paper](https://arxiv.org/html/2608.09888v1) distinguishes
contextual memory updated by observations from a workspace iteratively refined
to answer a query. It reports an ARC cost/accuracy result, not general continual
training superiority, and explicitly withholds exact updates and the full recipe.
Treat it as motivation for our own controlled implementation, not a reproducible
implementation specification or evidence that current Dragon already matches it.

Proposed Dragon interface, using existing state and latent-reasoning modules:

```text
rho_next = observe(theta, rho, observed_document_chunk)
z_0      = initialize_query(theta, rho_next, query)
z_k+1    = refine(theta, z_k, rho_next, query, remaining_compute_budget)
answer   = decode(theta, z_K)
```

During the initial experiment, refinement reads a fixed contextual rho and changes
only query-local workspace. This prevents a repeated thought step from counting
the same observation as new evidence. A write-back variant requires a separately
trained and evaluated commit rule. Keep the existing single-stream model as an
exact baseline; shared rho, split rho, and split weights answer different questions.

Train with bounded randomized refinement counts, then evaluate 1/2/4/8 steps on
the same held-out tasks and checkpoint. Compare with an equal-compute wider model
and equal-compute sampling/search. Train an adaptive stop rule only after extra
fixed steps improve quality; its features may not use the future answer or oracle.
Publish solve-versus-compute curves including failures, not only best-of-step results.

### JEPA, NextLat, and rho

Re-test the strongest existing JEPA and delayed/sparse NextLat schedules on the
current R3 objective; old R1 comparisons cannot establish current defaults.
[NextLat](https://arxiv.org/abs/2511.05963) predicts future latent state conditioned
on the next token. That is distinct from forcing every raw recurrent memory entry
to follow a centered Gaussian distribution.

Separate observation-state prediction, query-workspace prediction, and decoder
supervision in configuration and metrics. Use causal student inputs and detached
teacher targets; mask/reset both consistently at document boundaries. A target
may use observed future training data for an auxiliary loss, but it must not enter
the student's earlier inference or the evaluation state.

If isotropic regularization is tested, first apply it to a learned, centered
projection/readout. The linear memory stores query/value associations, so positivity
depends on the actual operands and positional transforms, not merely the presence
of a ReLU somewhere upstream. Do not force raw rho into an incompatible prior.
Compare regularized and unregularized retrieval/retention, not rank alone.

Likewise, global contraction is not automatically desirable: contracting a memory
can erase long-lived evidence. Test controlled query-workspace stability and
bounded state writes separately from memory time constants. Log sampled local
amplification, not an unproven global Lipschitz certificate.

### Predictive coding

Preserve three explicit labels: global-backprop baseline, exact local-PC execution,
and approximate/non-exact PC learner. The older state-correction auxiliary remains
an ablation and must not be described as the no-global-backward contract.

[Exact-PC research](https://arxiv.org/abs/2103.03725) provides a gradient-fidelity
control. If objective, derivative, optimizer, data, and state are identical, an
exact derivative implementation is not expected to invent a better learning rule.
Shared Dragon weights reduce parameter duplication, but do not remove state- and
depth-dependent credit propagation.

First optimize reusable local derivatives, shared-weight reductions, clipping,
and shape-specialized batched factors with all diagnostics device-resident.
Numerical gates include finite differences, directional derivatives, BP-gradient
agreement, sum over tied uses, multi-chunk credit, zero-signal updates, masked
terminals, and long-run optimizer/resume parity. Measure absolute and relative
errors across magnitude ranges; cosine alone misses scaling errors.

Then compare at most two non-exact challengers selected from current evidence:
bounded parallel/Jacobi inference and a validated temporal-credit/adjoint scheme.
Re-run teacher-refresh overhead and teacher-free quality explicitly. Error energy
decrease is solver evidence, not final-answer improvement. A learned adjoint that
saves local VJPs but damages credit is rejected before a long run.

Do not implement browser PC simply to complete a feature grid. Implement the
winning supported PC contract after local semantic quality justifies it; meanwhile
keep incompatible signed revisions observer/verifier-only, never silent AdamW.

### Capacity growth

Only classify a capacity plateau after stable held-out task mastery stops growing,
the difficulty producer still supplies genuinely harder tasks, optimization is
healthy, and an equal-budget wider control demonstrates headroom. Plateau does
not uniquely diagnose capacity exhaustion.

Keep in-process growth with preserved trained blocks, controlled contribution
from new neurons, optimizer migration, explicit rho migration, and re-calibrated
batch size. Compare immediate versus gated new-neuron contribution and fixed-width
controls. Record source config, resolved width history, checkpoint schema, and
validation before/after every event. Respect configured bounds, including the
existing default 8,192 growth limit; larger experimental widths require explicit
profiles, not an implicit change of the production bound.

Network growth is an atomic revision transition, not independent per-peer width
selection. Prepare the new genesis/migration, obtain validator approval, reassess
capabilities, and activate at a common boundary. Smaller peers can change batch
size, select a compatible specialized role, or become read-only; they cannot
average differently shaped arbitrary models into the same revision.

## 6. GPU efficiency and memory safety

Profile both a dense region and a complete session. Attribute data wait, CPU
submission, GPU queue/drain, model work, objective/teacher work, decode, verifier,
checkpoint encoding, transport, lease wait, and canonical acceptance. Time GPU work
with fences/timestamps at measured boundaries, not only around asynchronous dispatch.

Report model tokens/s, supervised tokens/s, complete problems/s, accepted training
tokens/s, bytes/accepted token, and joules/solved held-out problem. GPU power and SM
utilization are diagnostics, not optimization objectives. A memory-bound kernel can
be efficient below peak power, and high utilization can still execute redundant work.

Prioritize large observed costs: repeated prompt evaluation, sequential candidate
scoring, per-token readback, host-based clipping/reduction, repeated initialization,
and checkpoint materialization. Reuse causal prefixes with numerical equivalence
tests; batch active decode trajectories and kernel transitions; overlap bounded
CPU proof work with device work. Budget periodic expensive evaluation explicitly
and isolate it by checkpoint, rather than hiding its cost or co-locating unbounded
validator jobs on the same accelerator.

Memory admission is a hard contract:

- Cap at 90% physical RAM and 90% dedicated VRAM independently; the tighter
  applicable allocation budget wins. Count shared CPU/GPU physical memory once.
  Account for existing machine use, not just the new process's RSS.
- Reserve headroom below the cap for the largest in-flight allocation, compiler
  workspace, checkpoint encoding, verifier, optimizer migration, and other peers.
  A polling watchdog alone cannot prevent a sudden allocation from crossing a cap.
- Probe one candidate at a time, warm to steady state, drain queues, validate
  cleanup, and stop with OS memory-pressure signals before exhaustion. Never
  binary-search by repeatedly causing OOM. Use an external guardian and allocation
  admission limits; do not promise a mathematical no-crash guarantee for drivers.
- Measure cold peak, warm peak, and steady state separately. Avoid simultaneously
  retaining initialized random weights, loaded weights, host copies, and old/new
  optimizer generations. Stream/digest artifacts tensor by tensor where possible.
- Choose the smallest batch within measurement uncertainty of the best admissible
  throughput; using every free GB is not a learning or efficiency objective.
- Browser APIs do not reliably expose total/free device or system memory. Use
  adapter limits, conservative tracked allocations, a bounded user/device budget,
  and no destructive capacity probing. Native hardware-browser benches need the
  same external host guardian. Unknown capability is not unlimited memory.

## 7. A coherent heterogeneous training contract

### Preserve the dependency boundaries

| Owner | Responsibility |
| --- | --- |
| `burn_ecs` | Run entities; typed lifecycle/metric/control messages; bounded ingress; generic gates, policies, sinks, and shared resource admission |
| `burn_p2p` | Signed revisions, leases, capabilities, canonical heads, aggregation, codecs, validator decisions, transport and recovery; no Dragon/data assumptions |
| `burn_pc` | Generic factor/schedule/derivative contracts and numerical kernels; no task-specific reward or corpus policy |
| `burn_eggroll` | Deterministic low-rank/scaffold primitives and ES/update contracts; random-scaffold adapter training is not synonymous with EGGROLL ES |
| Dragon core | Tied weights, recurrent states, latent workspace, forward/local derivatives, model migration and kernel dispatch |
| Dragon universality | Typed tasks, proof kernel, generator, curriculum descriptors, sample identity and independent evaluation interfaces |
| Dragon language | Dataset plugins, learner/objective adapters, stream continuity, checkpoints, controlled experiments and model-quality decisions |
| Dragon P2P | Dragon workload plugin, portable profile mapping, native/browser executors, UI projections and deployed canaries |

Build on existing modules. Do not move the local trainer into `burn_p2p` or make
local mode initialize networking. CPU/CUDA/WebGPU dispatch belongs at a thin
executor boundary; shared data/objective/mask/recurrent logic must not fork.

Use normal Bevy plugins. Attach resolved config, learner identity, curriculum,
stream ownership, telemetry sinks, and checkpoint/growth state to each run entity.
Heavy generic model/optimizer state stays in typed backend-owned executors with
run-scoped ownership; do not force WebGPU handles across invalid thread boundaries
or expose tensors through the metric bus. Process cancellation and physical-device
resource admission are genuinely shared resources.

### Portable semantics versus local capability

A signed revision binds objective and normalization, optimizer/scheduler,
recurrent-state policy, model shape, tokenizer/data contract, codec, and aggregation.
Capabilities describe backend/dtype support, physical microbatch, measured rate,
resource budget, and implemented execution contracts. Hardware adaptation may
change execution shape only within semantics declared by the revision.

Specify state scope explicitly:

- Weights are canonical network state.
- Inner optimizer and scheduler state follow the selected protocol, not a guessed
  default. ArtifactWindows currently recreates the optimizer per window; DiLoCo
  has persistent inner/outer semantics. Optimizing away reset changes the algorithm
  unless the revision declares the new behavior.
- Rho belongs to a peer-local logical document stream, never to an arbitrary batch
  slot or to the network mean. After reconciling new weights, use the declared
  reset/replay or compatibility policy. Test drift from stale rho with new weights.
- Curriculum is revisioned/checkpointed. A peer may not silently replace a signed
  source contract with easier data to complete more leases.

Automatic upgrade/downgrade is a tested state machine: probe, admit, train,
degrade, back off, re-probe with hysteresis. Treat software adapters, device loss,
thermal pressure, hidden tabs, deadline failure, and unsupported objectives as
named reasons. A read-only metric observer needs no model; a verifier has its own
inference/checking capability and must not be mistaken for a full training peer.

## 8. Browser training and UX delivery

The merged browser work already keeps progress panels bounded and persistent,
defers most readbacks, caches generated NCA data, and increases useful window work.
Do not reimplement those fixes. Complete their missing deployment and session tests.

1. Start from clean browser storage against a signed production-shaped revision.
   Trace profile -> genesis tensor digest -> lease -> actual GPU work -> candidate
   upload -> accepted receipt -> promoted head -> next-window reconciliation.
   Repeat reload, expired lease, tab suspension, reconnect, and device recreation.
2. Define session-wide duty and accepted-work throughput, including inter-window
   waits. The current `tokens_per_second` is based on training wall time, while
   `training_duty_percent` divides by the current window's total time; neither alone
   establishes useful session-level training throughput.
3. Profile initialization/load, dense train, final drain, encoding, publication,
   receipt wait, and next lease independently. Reuse device/pipeline/buffer state
   where safe. Never retain stale weights or optimizer state just to remove a stall.
4. Move CPU-heavy generation/serialization and, where supported, training runtime
   work to a dedicated worker behind typed commands/events. Preserve a supported
   main-thread fallback and test actual worker-WebGPU availability. Bound queues
   and dispatch duration so cancel/pause and lease expiry are respected.
5. Preserve last completed window/loss and distinguish submitted, GPU-completed,
   receipt-accepted, and canonical-promoted work. Show stale metric age and role/
   downgrade reason. Coalesce render-only telemetry; never drop lifecycle events.
6. Use fixed-height panels and bounded activity history, with desktop/mobile visual
   tests across training, waiting, error, retry and role changes. Validate keyboard
   focus, reduced motion, no page jumping, and pause feedback independently of GPU
   queue completion. Do not expose unsupported power/memory precision as fact.

Hardware matrix: Chromium WebGPU on the available GB10, at least one discrete
Windows GPU and one integrated/unified-memory device when available. Browser/OS
versions and adapter limits are recorded. Firefox/Safari and no-WebGPU are explicit
capability rows, not inferred passes. Software-adapter CI validates downgrade and
protocol mechanics; scheduled real-hardware jobs validate training. No unavailable
hardware may be labeled verified.

## 9. P2P optimization, convergence, and trust

### First compare mathematically compatible learners

Run the same data IDs, masks, model initialization, local-update count, optimizer
policy, and canonical validation on single-peer, synchronized-central, and P2P
conditions. A sequential baseline with more optimizer updates is not a fair
synchronized reference. For variable-length documents, declare aggregation mass
in supervised information, not padded tokens or a peer-supplied unverified count.

Test 3 and then 8 peers, first CPU loopback for exactness, then CUDA/native WGPU/
real browser where the same contract is implemented. Inject heterogeneity,
non-IID shards, slower peers, churn, duplicate leases, and partitions. Measure
staleness and the fraction of computed contributions discarded or rejected.

Browser DiLoCo remains a missing execution contract; implement persistent inner
optimizer state, round cursors, outer-step application, resume, and signed
capability declaration before claiming mixed DiLoCo convergence. A browser PC
observer and an AdamW trainer are valid heterogeneous roles, not PC trainer parity.

### Communication roadmap

Use full/dense updates as the correctness oracle, then compare existing DiLoCo
FP32, FP16 and blockwise INT8 with the same local-step budget. Measure all wire
traffic: genesis, canonical downloads, update upload, protocol metadata, retries,
validation reports, and checkpoint recovery. Report amortization break-even time.

After the non-overlapped reference passes, add bounded blockwise communication
overlap with versioned snapshots and explicit staleness. The
[DiLoCo](https://arxiv.org/abs/2311.08105) and
[Streaming DiLoCo](https://arxiv.org/abs/2501.18512) papers motivate this direction;
their results do not establish equivalence on Dragon's recurrent, non-IID stream.
Test local steps 8/32/128 first, then 512 only if drift and deadline gates pass.

Re-test scaffold rank 8/16/32 at 1M and then 10M on the new task contract. Keep
frozen scaffold identity and initialization portable. The
[random-scaffold paper](https://arxiv.org/abs/2604.08749) supports testing adapter-only
training; it is not evidence of arbitrary-scale continual-learning equivalence.
Adapter factor averaging is not generally averaging effective weight updates:
`mean(A) * mean(B) != mean(A * B)`. Specify the actual aggregation algorithm,
test independent effective-weight reconstruction, and measure its convergence.

For a lower-bandwidth experimental channel, a fixed affine subspace
`theta = theta_base + P z` gives exact latent-averaging equivalence to weights in
that shared subspace. [FLITE](https://arxiv.org/abs/2607.18343) studies a related
low-rank, seed-regenerable channel around a pretrained base. Its evidence is
federated vision fine-tuning, not general from-scratch LLM pretraining. Compare
subspace capacity, drift, numerical reconstruction, and held-out retention before
admitting it to a revision. Re-basing/subspace expansion must be versioned.

EGGROLL ES is a separate candidate, not a default bridge into that codec. Evaluate
it on equal forward FLOPs, population/batch memory, and accepted task improvement,
including reconstruction/validation cost. No hybrid switch or huge population run
is justified solely by the existence of cheap seed payloads.

### Public network threat model

Authentication proves who submitted bytes, not that useful training occurred.
Small verifier panels cannot prove an arbitrary model update honest. Define the
initial release as permissioned validators with untrusted candidate trainers.

Require exact revision/base-head/lease binding, replay protection, bounded payload
decode, non-finite rejection, update-norm limits, independent candidate evaluation,
and quarantine/revocation. Test malicious updates, crafted compressed payloads,
equivocating reducers, conflicting reports, validator outages, replayed receipts,
and Sybil concentration. Quorum size must follow an explicit fault/admission model;
two agreeing local validators are not a Byzantine-resilience proof.

Keep private rotating promotion panels separate from public metrics. Bound ingress,
artifact retention, decompression ratios, and evaluator work to prevent resource
exhaustion. Document authority bootstrap/rotation, quorum membership, partition
behavior, and disaster recovery. A decentralized transport can still have an
authority-signed control plane; describe that trust boundary honestly.

## 10. Configuration and file organization

Consolidate around existing ownership rather than another general framework:

```text
burn_dragon_core/src/model/dragon/
  forward, sequence_dispatch, latent, predictive_coding, state/migration
burn_dragon_universality/src/ruliad/
  ir, kernel, formal, policy, source_selection, evaluation contracts
burn_dragon_language/src/train/
  steps, local_predictive_coding, objectives, schedule, manifest
burn_dragon_language/src/dataset/
  universality, scheduler, validation_panel, stream ownership
burn_dragon_p2p/src/
  profile, capability, native executor, wasm/{session,executor,progress,ui}
xtask/src/experiments/
  manifest, matrix, admission, runner, analysis, promotion
```

Paths above are a responsibility map, not a claim that all proposed submodules
already exist. Keep shared backend-generic learner logic below native/wasm hosts.

The PC experiment shell currently has roughly 4,631 lines and many environment
overrides. Replace experiment behavior with typed TOML matrix definitions and a
single CLI resolver; keep environment variables only for process/deployment
integration. Snapshot the fully resolved config and reject unknown settings and
unsupported combinations. Sparse timing or hidden inherited values must not alter
the scientific condition without entering its identity.

Main's browser `training.rs` is roughly 4,794 lines including substantial tests;
split session lifecycle, AdamW/ES executors, batch materialization, telemetry, and
publication along real ownership boundaries. Research `ruliad_policy.rs` has
roughly 2,600 production lines and 1,800 test lines: prioritize splitting scoring,
candidate representation, and evaluation rather than mechanically moving tests
to conceal size. Other apparent large files have extensive tests and should not
be classified solely by total line count.

Use a production-code review threshold around 2,000 lines, not a rigid tiny-file
rule. Maintain benchmark/numerical baselines through extraction. Do not combine
cleanup, a new loss, a new corpus, and a new distributed optimizer into one result.

Replace contradictory historical status prose with a generated capability index:
implemented, tested backend, experimental quality, local-promoted, network-promoted,
and release-verified. Preserve negative experiments with scope/provenance; do not
allow a later heading or filename to imply an unfinished matrix completed.

## 11. Staged experiment matrix

These are planned runs, not results. Execute sequentially on the shared-memory
machine. Each stage admits only survivors to the next; do not run the full cross
product of objectives, solvers, memory kernels, widths, and protocols.

| ID | Conditions | Budget and backend | Decision/output |
| --- | --- | --- | --- |
| E0: evaluator audit | Same checkpoint, teacher forcing/raw/syntax-only/menu/closed-loop; context swaps and corrupted certificates | CPU checker plus bounded CUDA decode; 256 then 1,024 tasks; no training | Determine where information or semantics change; exact fixtures and disjoint metric contracts |
| E1: clean local replication | Matched decoder-coupled AdamW and PC; then ordinary CE control on the same source | 1M class, CUDA, batch 8, block 1,024, TBPTT 64; 512 updates x 3 seeds; 1,024 x 5 confirmation | Reproduce the August 12 decision/decode gap under immutable provenance; report every objective cost |
| E2: task/interface learning | Final answer/action likelihood; syntax-only typed interface; model-visited action training; trace baseline as a separate control | 1M, same stream and budget; 512 x 3 screen, up to 4,096 x 5 survivors | Complete held-out success over shortcut controls; candidate coverage and oracle cost explicit |
| E3: recurrence and capacity | Reset/carry/causal warm replay; short/long credit; width-heavy versus embedding-heavy | 1M then 10M, CUDA; 2,048 x 3; at least four chunks/document; admitted batches only | Positive held-out carry benefit and correct reset/resize behavior, not just faster stateless training |
| E4: continual recipe | Best baseline; bounded replay; JEPA; sparse/delayed NextLat; CBP; selected PC, screened separately | 1M, 4,096 x 3; top two versus baseline 16,384 x 5; fixed tokens and wall budgets | New-task acquisition plus old-task retention; learner wins independent of curriculum feedback |
| E5: latent compute | Existing recurrence; separate workspace; split-memory shared-weight hierarchy | Train 1/2/4 sampled steps, evaluate 1/2/4/8; 1M screen then 10M; matched wider/search controls | Quality-compute curve improves; no repeated observation writes or target-dependent halting |
| E6: GPU and batch | Fixed effective batch and admissible physical batch; reference versus optimized kernels; dense versus all-in timeline | CUDA first, native WGPU and hardware browser; cold and warm repetitions | Numerical parity, best useful throughput, bounded memory, no readback-induced stalls |
| E7: native P2P | Central synchronized, artifact windows, DiLoCo; FP32/FP16/INT8; 3 then 8 peers; IID/non-IID/churn | 1M, 32 rounds x 3 seeds; 8/32/128 local steps screened in stages | Same-data convergence, accepted-work throughput, total wire bytes and staleness |
| E8: mixed/browser | CPU and CUDA/native WGPU plus a real browser under one supported contract; DiLoCo after implementation | 32 consecutive live windows; 3 repeat sessions; controlled disconnect/reload/device-loss drills | Signed genesis, accepted receipts, promotion/reconcile, UI stability and mixed convergence |
| E9: compression | Dense DiLoCo; scaffold ranks 8/16/32; affine subspace only after local success | 1M local 2,048 x 3 then network 32 rounds x 3; 10M confirmation | Quality-bandwidth frontier including head download and validation, not payload size alone |
| E10: release soak | Promoted 10M learner, fixed-width reference; local then mixed network | 24h then 48h, at least 20 structural shifts and held-out revisits; resource-admitted peers | Durable acquisition/retention, bounded resources, restart and partition recovery; 100M only after success |

An 8-peer co-location test is admitted by total resource consumption, not assumed
safe because each peer fits individually. Simulations remain labeled simulations.
Real Windows/integrated-GPU/WAN evidence requires those environments; localhost
traffic shaping is an intermediate gate, not a substitute.

### Proposed initial acceptance policies

Thresholds below are engineering promotion policies, not mathematical constants or
claims of achieved performance. Freeze them before confirmation; refine them from
pilot variance and the actual deployment SLO rather than after seeing a winner.

- **Semantic learning:** the lower confidence bound beats the strongest shortcut
  baseline on complete task success in each claimed supported task, with no hidden
  oracle menu advantage. A first bounded-task product target is 80% exact success
  and at most 1% malformed outputs on its advertised elementary competence set;
  harder held-out composition remains a separately reported frontier.
- **Continual quality:** on previously qualified tasks, retention remains within
  a preregistered 5-point absolute margin; late-task acquisition is at least 90%
  of matched early-task learning efficiency across the shift protocol. Confidence
  intervals must resolve those margins; no claim of infinite lifetime follows.
- **Optimizer promotion:** either verified quality improves at matched wall time,
  or throughput/energy improves while quality is non-inferior. Exact-PC internal
  energy or a favorable CE minimum cannot override failed reasoning endpoints.
- **Performance:** local model duty target at least 85%, p95 data wait below 2%,
  control/telemetry overhead below 2%, and dense optimized-kernel throughput within
  5% of its best matched reference unless quality justifies the cost. Report
  unavoidable evaluation separately and in all-in throughput.
- **P2P convergence:** retain the existing 90% synchronized-progress diagnostic,
  but also require endpoint CE and verifier non-inferiority. Initially use at most
  5% relative CE and 3-point absolute verifier loss as acceptance margins. Progress
  ratios are invalid when the reference makes negligible or negative progress.
- **Browser delivery:** no silent objective/backend substitution; accepted receipts
  in every eligible clean session; 32 consecutive valid windows without unexplained
  failure, stale-head reuse or layout jumps. Target at least 75% session duty on
  the reference LAN profile and report WAN separately; receipt/canonical latency
  gets its own SLO. A lease-expiry test expects correct rejection, not acceptance.
- **Safety:** no process knowingly admitted beyond the memory policy, no unbounded
  queues/caches, no non-finite accepted update, and no unsigned/mismatched artifact
  accepted by a canonical participant. Driver failure is contained and reported.

## 12. Implementation sequence and deliverables

| Phase | Work | Exit artifact | Dependencies |
| --- | --- | --- | --- |
| P0 | Freeze source and evidence; correct metric names/denominators; complete checkpoint evaluation audit | Immutable manifest bundle, E0/E1 tables, shortcut/leakage tests | First; no architecture changes |
| P1a | Strengthen proof task generation, split policy/render/generation evaluation, preserve streamed document state | Distribution audit, realized-complexity samples, E2/E3 tables | P0 |
| P1b | Complete signed hardware-browser canary and session accounting; modularize hot lifecycle | Accepted-receipt evidence, timeline, desktop/mobile state captures, E6/E8 baseline | P0; can progress alongside local learning |
| P2 | Establish continual baseline; screen existing JEPA/NextLat/PC/CBP without stacking changes | E4 quality/retention/throughput matrix and selected recipe | P1a |
| P3 | Add separated query workspace and justified state/growth changes | E5 compute curves, numerical/state migration checks | P2, except isolated small mechanistic tests |
| P4 | Implement missing mixed-peer protocol executor, matched aggregation and efficient codecs | E7/E8/E9 convergence and complete bandwidth report | P1b plus a qualified local learner |
| P5 | Trust hardening, authority/recovery drills, long local/network soaks and external transfer | E10 results, release capability matrix, model/data/system cards | P2/P4; P3 optional unless it wins |

The first implementation batch should produce: a clean matched local table,
an audited decision-versus-generation breakdown, a fresh-generation complexity
audit, and one real browser contribution traced through canonical acceptance.
It should not launch another 48-hour run merely because local loss falls.

Every subsequent work batch ends with a concrete numerical comparison or an
explicitly incomplete matrix with failed/skipped arms and reasons. Unit tests,
CI, and browser smokes remain necessary, but are not learning-quality evidence.
Poll CI sparingly; preserve compute and attention for experiments and code review.

The final release claim is bounded and useful: a reproducible Dragon learner with
measured reasoning and continual-learning capability, an honest hardware support
matrix, and decentralized contributions that preserve that capability at a known
cost. Broader SotA claims require external, independently reproducible comparisons.
