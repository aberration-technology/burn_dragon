# Bounded Local Experiments

Run from the Dragon workspace root (Python 3.11+, Linux):

```bash
python3 -m scripts.experiments config/experiments/ruliad-checkpoint-contract.toml
python3 -m unittest scripts.test_experiments scripts.test_ruliad_checkpoint_eval_analyze
```

The checkpoint manifest expects paired epoch-8 runs staged at
`target/experiment-inputs/evaluation-contract/{adamw,pc}`. Stage directories or
symlinks to existing run directories, not individual checkpoint tensors. Build
`evaluate_ruliad_checkpoint` with `train,cuda` in release mode first. Change
`output` for a new execution; existing evidence is never overwritten.

## Contract

- Rust training config owns model, optimizer, source selection and objectives.
  This runner only owns experiment ordering, command arguments, provenance,
  resource monitoring and timeouts. It adds no training implementation.
- Manifests reject unknown fields. Commands are argument arrays, not shell text.
  `{output}` expands to the case's artifact directory. Paths resolve from the
  workspace, including declared `inputs` and sibling `repositories`.
- Cases run sequentially and fail fast. Timeout, interruption, guard failure and
  nonzero exit are failures, not completed experiments. Process groups are killed
  on every exit path, including descendants left by an exited group leader.
- Each case records binary and declared input SHA-256 identities, arguments,
  stdout/stderr, host memory samples, optional GPU samples, elapsed wall time and
  status. Declare checkpoints, configs and corpus/proof inputs explicitly.
  Source archives include HEAD, tracked binary patch and bounded untracked files
  for every listed repository. Source or input drift invalidates completion.
  Executables are copied once per hash and executed from the evidence bundle, so
  a later Cargo rebuild cannot erase the exact binary used for a measurement.
- `BURN_*` and `DragonModel_*` environment overrides are removed from children. CUDA/CubeCL/WGPU
  overrides are recorded; no environment secrets are dumped.

## Memory Safety

`expected_peak_mib` is a conservative **additional physical-memory estimate** for
admission, not a measured process peak or a cgroup allocation limit. Admission
requires current used RAM + expected peak + headroom below `system_fraction`.
The fraction must be <=0.90. Runtime checks retain the headroom and use
`MemTotal - MemAvailable`, not process RSS, to account for concurrent workloads.

Set `shared_gpu_memory=true` only on unified-memory hardware such as GB10. Unified
memory is counted once; NVIDIA's unavailable memory counter is not treated as
infinite VRAM. On discrete GPUs the RAM and VRAM checks must **both** pass; missing
or stale GPU memory telemetry fails closed. NVIDIA queries run off the fast host
watchdog loop. GPU power samples are observations, not a training-duty metric.

Polling is a secondary safety mechanism, not protection against instantaneous
unbounded allocations or a kernel that cannot be interrupted. Keep workloads
bounded and estimates conservative; use OS/cgroup limits where supported. Never
calibrate by provoking an OOM. This runner is not a production job scheduler.

## Evidence

Two held-out panel seeds for the same checkpoint are not two training seeds.
Analyze each fixed panel separately; the checkpoint analyzer refuses mismatched
panels, evaluation versions, teacher-forcing versions and corpus identities.
Teacher-forcing v2 retains complete prompts/answers in recurrent chunks and
reports both per-token NLL and mean sequence NLL. It is not free generation.
Suite v8 verifies model parameter identity before/after its probes. A successful
execution means the measurement completed, not that a model passed promotion.

Suite v8 also reports checkpoint-only policy controls: exact per-item uniform
chance (equivalent actions / all actions), fixed menu positions, shortest semantic
action, one-step structural-distance search, and the same model with only the
answer delimiter as context. Tied heuristics use uniform expected credit, never
expert-based tie breaking. All controls share the oracle menu with the model;
none establishes unassisted proof generation. The kernel replays the reference
certificate and every cached candidate outcome/label before accepting a report.
This checks caches against execution, not an independently implemented verifier.
Per-item paired results and difficulty/source summaries remain in the JSON.
The in-training probe explicitly disables these additional controls.

`config/experiments/ruliad-policy-controls.toml` compares six existing frozen
checkpoints on the same 256-item panel. The 512- and 1024-update groups are
different training experiments, not a causal checkpoint learning curve. Compare
optimizers and maintenance settings within their matched group.

`config/experiments/ruliad-policy-scoring-controls.toml` separates the existing
residual, language-likelihood, and semantic-energy scorers on the same frozen
weights and policy panel. It reduces only the full-document generation panel to
16 items; analyze its reports separately because whole-suite panel hashes include
those document items. This is an inference ablation, not retraining each scorer.

Fresh/transfer runs capture initial tensor identity by default through
`[training.provenance] initial_model_fingerprint = true`, without an environment
switch. This is startup-only and primary-peer-only. Changing provenance capture
does not change the immutable training contract or invalidate an exact resume.
