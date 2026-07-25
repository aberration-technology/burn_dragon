# burn_dragon 🔥🐉

`burn_dragon` is the dragon model + training workspace.

it pairs the dragon model stack with [`burn_p2p`](https://github.com/aberration-technology/burn_p2p) for native + browser p2p training, deployment, and live network operation.

the model shape follows the [dragon hatchling / bdh paper](https://arxiv.org/abs/2509.26507).

## what is here

- `crates/burn_dragon_core`: core model, state, and config
- `crates/burn_dragon_language`: language training + inference adapters
- `crates/burn_dragon_p2p`: p2p runtime, browser ui, deployment, and integration tests
- `xtask`: build, smoke, deploy, and release helpers

## common paths

- model + language code: [crates/burn_dragon_core](crates/burn_dragon_core), [crates/burn_dragon_language](crates/burn_dragon_language)
- p2p + deployment: [crates/burn_dragon_p2p](crates/burn_dragon_p2p), [crates/burn_dragon_p2p/deploy/README.md](crates/burn_dragon_p2p/deploy/README.md)
- protocol/runtime layer: [`burn_p2p`](https://github.com/aberration-technology/burn_p2p)

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
For private siblings, configure the repository secret `BURN_STACK_TOKEN` with
read-only contents access to all four repositories. GitHub's repository-scoped
job token does not provide cross-repository access and is intentionally not
used as a fallback.

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
