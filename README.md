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

- model + language code: [crates/burn_dragon_core](/home/mosure/repos/burn_dragon/crates/burn_dragon_core), [crates/burn_dragon_language](/home/mosure/repos/burn_dragon/crates/burn_dragon_language)
- p2p + deployment: [crates/burn_dragon_p2p](/home/mosure/repos/burn_dragon/crates/burn_dragon_p2p), [crates/burn_dragon_p2p/deploy/README.md](/home/mosure/repos/burn_dragon/crates/burn_dragon_p2p/deploy/README.md)
- protocol/runtime layer: [`burn_p2p`](https://github.com/aberration-technology/burn_p2p)

## quick start

```bash
cargo run -p xtask -- smoke
cargo run -p xtask -- deploy-check
```
