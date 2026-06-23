# Next-Latent Dragon Auxiliary Report

## Scope

This branch adds a NextLat-style latent transition objective for Dragon language
training and extends it with a Dragon-specific recurrent-state consistency
objective. The implementation follows the paper at the level needed for Dragon:
the model learns an auxiliary transition from current final hidden state and the
next-token embedding to the next final hidden state, with stop-gradient targets.
Dragon also carries important through-time state in `rho`, so the stronger
profile adds an EMA-teacher consistency loss over sampled rho memory rows and
their row energy. The standard Dragon forward path is unchanged unless an
experiment separately enables `model.latent_reasoning`.

## Implementation

- `model.next_latent_transition` allocates a training-only transition MLP.
- `training.latent_reasoning.next_latent` enables the auxiliary objective under
  the existing latent auxiliary cadence and warmup controls.
- The transition head receives `[h_t, embed(x_{t+1})]`, optionally normalizes the
  concatenated input, predicts a residual delta, and is zero-initialized by
  default so it starts as an identity delta.
- The loss uses SmoothL1 against detached target hidden states. Optional token
  KL is available for flat language heads and is disabled by default.
- `training.latent_reasoning.dragon_state` enables EMA-teacher rho-state
  consistency. It separately compares normalized rho row direction and rho row
  RMS energy, with deterministic slot sampling when memories are large.
- Telemetry now emits `Latent Reasoning NextLat Components` and
  `Latent Reasoning Dragon State Components` alongside JEPA and SIGReg
  component counters.

## Profiles

- `ruliad-r1.adamw-fixed-ablation.toml`: complete fixed-size baseline profile.
- `ruliad-r1.latent-jepa-fixed-ablation.toml`: fixed JEPA comparison.
- `ruliad-r1.nextlat-fixed-ablation.toml`: fixed NextLat comparison.
- `ruliad-r1.nextlat-smoke.toml`: short fixed-batch CPU smoke profile.
- `ruliad-r1.state-nextlat-fixed-ablation.toml`: NextLat plus EMA rho-state
  consistency.
- `ruliad-r1.state-nextlat-smoke.toml`: short fixed-batch CPU smoke profile for
  the combined objective.

## Verification

Commands run:

```bash
CARGO=$(rustup which cargo) RUSTC=$(rustup which rustc) \
  $(rustup which cargo) test -p burn_dragon_core next_latent -- --nocapture

CARGO=$(rustup which cargo) RUSTC=$(rustup which rustc) \
  $(rustup which cargo) test -p burn_dragon_language --features train next_latent -- --nocapture

CARGO=$(rustup which cargo) RUSTC=$(rustup which rustc) \
  $(rustup which cargo) test -p burn_dragon_language --features train latent_reasoning -- --nocapture

CARGO=$(rustup which cargo) RUSTC=$(rustup which rustc) \
  $(rustup which cargo) run -p burn_dragon_language --example train_language --features train -- \
  --backend cpu --config crates/burn_dragon_p2p/deploy/profiles/ruliad-r1.nextlat-smoke.toml

CARGO=$(rustup which cargo) RUSTC=$(rustup which rustc) \
  $(rustup which cargo) test -p burn_dragon_language --features train dragon_state -- --nocapture

CARGO=$(rustup which cargo) RUSTC=$(rustup which rustc) \
  $(rustup which cargo) run -p burn_dragon_language --example train_language --features train -- \
  --backend cpu --config crates/burn_dragon_p2p/deploy/profiles/ruliad-r1.state-nextlat-smoke.toml
```

All passed. The NextLat smoke run produced
`Latent Reasoning NextLat Components = 2`, with JEPA/SIGReg at zero.
The combined state/NextLat smoke run `runs/old-base` produced
`Latent Reasoning NextLat Components = 2`,
`Latent Reasoning Dragon State Components = 4`, and JEPA/SIGReg at zero.

## Ablation Results

The 64-step CPU matrix used the fixed 2-layer/64-embed/256-latent profiles.
These are smoke-scale dynamics checks, not long-run stability evidence.

| Variant | Run | Wall Time | Final Train Loss | Final Valid CE | Ruliad Composite | Verifier |
|---|---:|---:|---:|---:|---:|---:|
| AdamW baseline | `runs/feeble-answer` | 74.9s | 2.9636 | 2.8049 | 500000 | 0.0 |
| JEPA auxiliary | `runs/repulsive-scarecrow` | 75.2s | 2.6996 | 2.5184 | 750000 | 0.0 |
| NextLat auxiliary | `runs/huge-swing` | 104.8s | 3.1086 | 2.8719 | 1000000 | 0.0 |
| NextLat + rho-state | `runs/shivering-kick` | 115.2s | 3.0692 | 2.9142 | 1000000 | 0.0 |

Initial read:

- The NextLat objective is wired and stable at smoke scale.
- The horizon-2 transition objective is about 1.4x slower than baseline/JEPA on
  this CPU smoke. CUDA throughput still needs separate measurement.
- At 64 steps, NextLat did not improve cross-entropy versus AdamW or JEPA. Its
  coarse ruliad composite was higher, but all variants remained verifier-zero,
  so this is not evidence of solved reasoning.
- The combined state/NextLat profile keeps the objective active through the full
  train loop and slightly lowers train loss versus NextLat-only at this tiny
  horizon, but teacher-forced valid CE is worse. This should be treated as a
  long-run stability candidate, not as a short-run convergence win.
- JEPA remains the stronger short-horizon CE baseline in this tiny matrix.
- The Dragon-state consistency tests prove the loss is zero for matching rho
  state and positive for rho row drift. This is a dynamics-preservation check,
  not yet a convergence win.

## Practical Next Steps

- Run CUDA timing at the same fixed config to measure actual overhead.
- Add a longer, small-model continual run that tracks verifier, bucketed ruliad
  mastery, output entropy, and collapse metrics across many epochs.
- Sweep `horizon`, `normalized_aux_scale`, `detach_action_embedding`, and
  `token_kl_weight`; the current defaults prioritize stability over strength.
- Sweep `dragon_state.rho_weight`, `rho_energy_weight`, and `teacher_update_rate`
  against verifier and output-degeneracy metrics. The default combined profile is
  intentionally mild: it should stabilize through-time rho dynamics before it is
  asked to dominate cross-entropy.
- Consider a lower-cost transition head or sharing the existing latent JEPA
  predictor when throughput matters more than paper-faithful conditioning.
