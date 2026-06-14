# AGENTS.md instructions for burn_dragon

## Local Tooling

- Use the rustup toolchain binaries directly when running Rust commands in this workspace. Prefer the installed stable toolchain path from `rustup which cargo` / `rustup which rustc`, or the matching files under `~/.rustup/toolchains/stable-*/bin/`.
- Set `RUSTC` to the matching rustup `rustc` when invoking Cargo. For wrappers such as `cargo fmt`, also set `CARGO` to the matching rustup `cargo` when needed.
- Avoid relying on `/snap/bin/cargo`; snap confinement can fail before repository logic runs.
- The workspace dependencies for `burn_p2p` crates intentionally use local sibling paths under `../burn_p2p/crates/` so Dragon can exercise in-flight p2p changes during development.

## Dependency Layering

- Preserve the stack direction: `burn_ecs -> burn_p2p -> burn_dragon`.
- Dragon is the domain application and training pipeline consumer. Do not move Dragon-specific ruliad, NCA, language-model, CLI, or config assumptions into `burn_ecs` or `burn_p2p`.
- Use `burn_ecs` for training app/plugin/runtime orchestration and `burn_p2p` only for P2P mode integration.
- In local-only training mode, Dragon should not require P2P plugins or network state. In P2P training mode, Dragon should add the P2P plugin explicitly.

## Training ECS Principles

- Prefer canonical Bevy app/plugin flows. Use plugins with config resources/components rather than setup closures or boxed functions.
- Run-local training state should be entity-scoped in `burn_ecs`. Dragon code should emit run-keyed messages and attach Dragon-specific state through plugins rather than assuming singleton runtime state.
- Preserve support for multiple model/pipeline training runs in one ECS world. Do not introduce global mutable training state unless it is truly process-wide.
- Keep local control handles, interrupters, and process-level cancellation as global resources where appropriate.
- Keep event/log/dashboards attached to the relevant run through `burn_ecs` abstractions rather than writing ad hoc Dragon-only sinks.

## Ruliad And Dataset Principles

- Ruliad synthetic data should be a coherent categorical/reasoning abstraction, not a loose pile of unrelated families.
- Generators should expose verifiable structure, reasoning traces, and task metadata suitable for both pretraining-style trace learning and later RL/evaluation.
- Live source-selection must remain cheap relative to batch construction and model steps. Data generation should not become the bottleneck for train/eval batch assembly.
- Avoid random/seed functions that create obvious periodic or moire artifacts. Validate distribution coverage and keep tests/smokes for sample sanity.
- Metrics should surface difficulty, entropy, hash-noise, verifier failures, and learning progress in local runs, not only in P2P paths.

## Browser P2P Deployment Notes

- Browser Pages config is not driven only by `BURN_DRAGON_P2P_PAGES_SEED_NODE_URLS`. The Pages build also derives browser seeds from the live edge, so seed bugs often require checking both the environment variable and `xtask/src/deploy_settings.rs`.
- Prefer DNS multiaddrs with certhash for browser seeds rather than raw `ip4` literals.
- The real browser training path is WebGPU. Browser CPU is smoke/dev only. Firefox may connect successfully but still downgrade to observer/verifier when WebGPU is unavailable.
- Native CLI auth is split across hosts: the Pages app serves the minimal callback bridge page, while the edge serves auth/API. Debug login failures by separating bridge-page routing from edge reachability.
- The Pages workflow can spend a while on the browser-site build artifact after a clean build; do not classify it as hung too early.
