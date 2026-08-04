# burn_dragon 🔥🐉

`burn_dragon` is the dragon model + training workspace.

it pairs the dragon model stack with [`burn_p2p`](https://github.com/aberration-technology/burn_p2p) for native + browser p2p training, deployment, and live network operation.

the model shape follows the [dragon hatchling / bdh paper](https://arxiv.org/abs/2509.26507).

## what is here

- `crates/burn_dragon_core`: core model, state, and config
- `crates/burn_dragon_language`: language training + inference adapters
- `crates/burn_dragon_universality`: verifier-backed formal Ruliad source
- `crates/burn_dragon_p2p`: p2p runtime, browser ui, deployment, and integration tests
- `xtask`: build, smoke, deploy, and release helpers

## common paths

- model + language code: [crates/burn_dragon_core](crates/burn_dragon_core), [crates/burn_dragon_language](crates/burn_dragon_language)
- p2p + deployment: [crates/burn_dragon_p2p](crates/burn_dragon_p2p), [crates/burn_dragon_p2p/deploy/README.md](crates/burn_dragon_p2p/deploy/README.md)
- formal Ruliad design and evidence: [docs/ruliad-r3-formal-report.md](docs/ruliad-r3-formal-report.md)
- protocol/runtime layer: [`burn_p2p`](https://github.com/aberration-technology/burn_p2p)

## random scaffold adapters

Dragon can parameterize its three shared recurrent projections as immutable,
seeded random scaffolds plus trainable low-rank adapters. This preserves the
architecture's across-layer and through-time weight sharing: there is one
adapter for each shared encoder, value encoder, and decoder, not one adapter per
unrolled recurrent step.

```toml
[model.random_scaffold]
enabled = true
seed = 20260729
distribution = "gaussian_clt12"
rank = 16
alpha = 16.0
scaling = "rank_stabilized"
trainable_gain = true
```

The portable generator and adapter manifest live in `burn_eggroll`; Dragon owns
the projection selection and training behavior. A run writes
`random_scaffold_manifest.json`, and resume rejects a different seed,
generator, shape, rank, or gain contract. Matched local experiment profiles are
under
[`config/language/experiments/random_scaffold`](config/language/experiments/random_scaffold).
Random-scaffold experiments currently use AdamW; EGGROLL's existing population
executor evolves dense shared projections and is deliberately rejected rather
than silently mutating the immutable scaffold.

The selected rank-stabilized rank-16 profile clears the matched three-seed
local CUDA quality/efficiency gate and the three-peer native
synchronized-convergence gate. The implementation, corrected masked objective,
compact P2P protocol, bandwidth matrix, GPU traces, and remaining WAN/browser
production gates are documented in the
[`random-scaffold Dragon report`](docs/random-scaffold-dragon-report.md).

## formal ruliad pretraining

Ruliad R3 lowers equational, category, logic, automata, process-calculus, and
metagraph problems into one proof IR and deterministic transition kernel. The
same compact source, target masks, verifier contracts, and curriculum semantics
run in local, native P2P, and browser-WebGPU training. Difficulty levels are
materialized lazily without a configured frontier cap, while each realized
proof remains bounded by explicit resource limits.

The current evidence shows verifier and partial-proof-policy gains, exact
three-peer protocol replay, and native/browser source parity. It does not yet
show general mathematical reasoning or long-horizon production readiness. See
the [formal Ruliad report](docs/ruliad-r3-formal-report.md) for the architecture,
ablation tables, and remaining promotion gates.

## quick start

```bash
python3 scripts/bootstrap_stack.py
cargo run -p xtask -- local-browser-e2e
cargo run -p xtask -- smoke
cargo run -p xtask -- deploy-check
```

Dragon intentionally develops against sibling path dependencies. The exact
`burn_ecs -> burn_p2p -> burn_dragon` stack plus `burn_eggroll` and `burn_pc`
is pinned in `stack.lock.toml`. Run `scripts/bootstrap_stack.py` to clone
missing siblings, `--verify` to reject revision/remote drift, or
`--repair-existing` to move only clean existing siblings to the locked
revisions. CI uses the same lock through the shared bootstrap action.
All locked providers are public and clone over HTTPS, so stack bootstrap
requires no cross-repository credential.

Use `local-browser-e2e` as the first browser/p2p production-parity gate. It runs
the deployment config drift checks, a local edge/auth/browser training receipt
e2e, and the smallest real Chrome/WebGPU browser training smoke without forcing
the full CI build matrix.

For the slow browser peer loop, run the lane you need instead of waiting for a
Pages deploy. The offline default remains:

```bash
cargo run -p xtask -- local-browser-e2e --lane all
```

To test the exact browser artifact locally against a live or staging edge, set
the browser canary edge/principal/callback environment variables and run:

```bash
cargo run -p xtask -- local-browser-e2e --lane canary-webrtc-direct-training --build-site
```

Canary artifacts are written under `target/test-artifacts/browser-peer-e2e/`.
If the local `../burn_p2p` checkout is on an in-flight branch that does not
match Dragon's pinned CI version, use `cargo run -p xtask -- local-browser-e2e-ci-sibling`
with the same lane flags. It runs the command in a temporary Dragon worktree
paired with the CI-pinned `burn_p2p` sibling and applies the current Dragon diff.
