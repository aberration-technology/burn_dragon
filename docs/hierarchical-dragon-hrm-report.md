# Hierarchical Dragon HRM Notes

This feature adds an HRM-Text-inspired fast/slow hierarchy to Dragon without changing the
default architecture. It is opt-in through `model.hierarchical_dragon` and is intended for
ruliad and language continual-learning ablations where a single recurrent rho stream may be
too shallow to separate rapid token dynamics from slower document-level state.

## Architecture

- Fast branch: the ordinary Dragon shared low-rank step over the current token span.
- Slow branch: a one-token summary state derived from the fast branch mean and recurrently
  updated after each fast cycle group.
- Rho sharing: `shared` reuses the ordinary sequence memory; `split` gives the slow branch
  its own rho/Mamba/GDN state slot on each layer.
- Weight sharing: `shared` reuses Dragon low-rank encoder/value/decoder weights; `split`
  adds slow encoder/value/decoder params. The split slow decoder is zero-initialized so
  the branch starts as a stable adapter rather than a random residual injector.
- Scope: `last_layers` applies hierarchy only to the top N layers, matching existing
  clocked/summary memory scoping patterns.

The implementation uses the existing sequence dispatcher for linear attention, Mamba3, and
GDN2. Split slow rho works by temporarily swapping the layer's fast and slow sequence-state
fields before dispatch.

## Supported Matrix

Current fixed ablation overlays:

- `ruliad-r1.hdragon-shared-rho-shared-weights-probe128-fixed-ablation.toml`
- `ruliad-r1.hdragon-split-rho-shared-weights-probe128-fixed-ablation.toml`
- `ruliad-r1.hdragon-split-rho-split-weights-probe128-fixed-ablation.toml`

Recommended first-pass comparison:

1. Baseline JEPA/NextLat fixed profile.
2. Shared rho + shared weights, to isolate cycle depth from extra state/params.
3. Split rho + shared weights, to test whether memory separation improves verifier stability.
4. Split rho + split weights, to test whether separate slow dynamics justify extra params.

## Guardrails

- Disabled by default.
- Rejected with `parallel.pipeline.enabled`; layer partitioning needs explicit slow-state
  routing before it is safe.
- Rejected with `clocked_slow_memory` and `y_neuron_recurrence`; these are separate slow
  recurrence mechanisms and should not be stacked until their interactions are designed.
- Excluded from shared low-rank population forward and shared low-rank continual backprop
  fast paths. Those paths can get dedicated hierarchical kernels later.
- Checkpoint metadata records hierarchy enabled state, sharing modes, and cycle counts.

## Metrics To Watch

Use the normal CE, verifier/schema correctness, ruliad bucket telemetry, degeneracy probes,
and throughput metrics. The hierarchy should only be promoted if verifier/schema stability
improves at comparable CE and acceptable throughput.
