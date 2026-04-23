# AGENTS.md instructions for /home/mosure/repos/burn_dragon

## Local Tooling

- Use the rustup toolchain binaries directly when running Rust commands in this workspace:
  - `/home/mosure/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/cargo`
  - `/home/mosure/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/rustc`
- Set `RUSTC=/home/mosure/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/rustc` when invoking Cargo. For wrappers such as `cargo-fmt`, also set `CARGO=/home/mosure/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/cargo`.
- Avoid relying on `/snap/bin/cargo`; snap confinement can fail before repository logic runs.
- The workspace dependencies for `burn_p2p` crates intentionally use local sibling paths under `../burn_p2p/crates/` so Dragon can exercise in-flight p2p changes during development.

## Browser P2P Deployment Notes

- Browser Pages config is not driven only by `BURN_DRAGON_P2P_PAGES_SEED_NODE_URLS`. The Pages build also derives browser seeds from the live edge, so seed bugs often require checking both the environment variable and `xtask/src/deploy_settings.rs`.
- Prefer DNS multiaddrs with certhash for browser seeds rather than raw `ip4` literals.
- The real browser training path is WebGPU. Browser CPU is smoke/dev only. Firefox may connect successfully but still downgrade to observer/verifier when WebGPU is unavailable.
- Native CLI auth is split across hosts: the Pages app serves the minimal callback bridge page, while the edge serves auth/API. Debug login failures by separating bridge-page routing from edge reachability.
- The Pages workflow can spend a while on the browser-site build artifact after a clean build; do not classify it as hung too early.
