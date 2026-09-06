# Dependency Backports

## burn-cubecl-fusion 0.21.0

Source: the unmodified crates.io 0.21.0 package, except the changes below.
Package VCS revision: `546cacb55fe00168854d19bdf0a5d79bd8060e03`.
Upstream: https://github.com/tracel-ai/burn/tree/v0.21.0/crates/burn-cubecl-fusion
License: MIT OR Apache-2.0 (upstream license files included).

The fusion planner selects its I/O width from the smallest input/output dtype.
A u8 comparison mask can therefore force a vector of 16 into a floating-point
matmul whose selected tile supports only 8. CUDA then aborts the training process
with `Invalid vector size. Got 16 which should not be >8`.

The local backport adds a routine-specific minimum element width. Only matmul
overrides it, using the maximum storage width of its three numeric operands.
Elementwise and reduction fusion are unchanged. Quantized vector selection keeps
its separate path. No host synchronization or architecture change is introduced.

Changed upstream files:

- `src/engine/launch/runner.rs`: default routine constraint.
- `src/engine/launch/vectorization/planner.rs`: respect that constraint for ordinary I/O.
- `src/optim/matmul/optimization.rs`: numeric matmul storage constraint.

Remove this patch when an upstream release passes the CUDA mixed-mask regression
and the Dragon capacity training smokes. Do not edit the global Cargo registry.

Regression: `cuda_fused_matmul_byte_mask_matches_direct_values_and_gradients` in
`burn_dragon_kernel`. It compares ordinary CUDA with fused CUDA for values,
boolean masks, and both parameter/input gradients at 256-wide and 4096-wide
projection shapes. This does not claim exhaustive coverage of quantized kernels.
