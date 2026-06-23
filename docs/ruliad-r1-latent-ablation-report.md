# Ruliad R1 Latent Objective Ablation

## Scope

This ablation evaluates the fixed small ruliad-r1 Dragon training setup across
latent auxiliary objectives. It is intentionally bounded: CPU backend, fixed
2-layer/64-embed/256-latent model, no auto-batch, no continual backprop, no
neuron scaling, and no long production run. The goal is to decide whether the
new NextLat and Dragon rho-state objectives are worth carrying forward as
pretraining objectives for stronger through-time dynamics.

The matrix used the fixed profiles under
`crates/burn_dragon_p2p/deploy/profiles/`. Raw run mapping is in
`target/ruliad_r1_latent_ablation_runs.tsv`,
`target/ruliad_r1_latent_ablation_128_runs.tsv`, and the follow-up
`target/ruliad_r1_latent_ablation_256_runs.tsv`. Parsed summaries are in
`target/ruliad_r1_latent_ablation_summary.csv`,
`target/ruliad_r1_latent_ablation_128_summary.csv`, and
`target/ruliad_r1_latent_ablation_256_summary.csv`. The top-candidate
1024-step follow-up is in `target/ruliad_r1_latent_ablation_1024_runs.tsv` and
`target/ruliad_r1_latent_ablation_1024_summary.csv`.
The supervision-target follow-up is in
`target/ruliad_r1_latent_supervision_ablation_256_summary.csv`,
`target/ruliad_r1_latent_supervision_ablation_1024_summary.csv`, and
`target/ruliad_r1_latent_supervision_ablation_4096_summary.csv`.
The delayed/sparse NextLat follow-up is in
`target/ruliad_r1_nextlat_schedule_ablation_4096_summary.csv`,
`target/ruliad_r1_nextlat_schedule_ablation_4096b_summary.csv`, and
`target/ruliad_r1_nextlat_schedule_latent_counts.csv`.
The 4096-step three-seed schedule aggregate is in
`target/ruliad-r1-nextlat-multiseed-4096-latest/combined_3seed_summary.csv`;
the two additional seed runs are in
`target/ruliad-r1-nextlat-multiseed-4096-latest/summary.csv`.
The decoupled NextLat schedule pass, where JEPA and NextLat have separate
cadence/start controls, is in
`target/ruliad-r1-nextlat-decoupled-4096-latest/summary.csv`.

The tables report the ruliad competence scalar as `Composite`. That metric is
only a coarse lexicographic dashboard encoding of verifier, semantic, partial,
certificate, and completion-health PPM. It is not a smooth reasoning score. With
the tiny 4-item validation probe used in these bounded ablations, completion
health changes in 250k increments and can dominate the displayed value while
verifier accuracy remains exactly zero. Promotion decisions should therefore
lean on verifier/schema behavior, validation CE, output-degeneracy metrics, and
source-bucket learning telemetry rather than the composite alone.

## 64-Step Matrix

| Variant | Run | Time | Train | Teacher CE | Valid | Composite | Verifier | Schema Wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AdamW | `runs/dramatic-song` | 74.4s | 2.9577 | 2.7898 | 2.6673 | 500000 | 0.0 | 0.50 |
| JEPA | `runs/literate-authority` | 84.4s | 2.6996 | 2.5184 | 2.2794 | 750000 | 0.0 | 0.25 |
| Dragon state only | `runs/scattered-spring` | 104.7s | 3.0584 | 2.9057 | 2.7163 | 250000 | 0.0 | 0.75 |
| NextLat h1 | `runs/nutritious-egg` | 94.1s | 3.1135 | 2.8853 | 2.7137 | 1000000 | 0.0 | 0.00 |
| NextLat h2 | `runs/numberless-flavor` | 93.5s | 3.1086 | 2.8719 | 2.7084 | 1000000 | 0.0 | 0.00 |
| NextLat h4 | `runs/ripe-smoke` | 104.2s | 3.0882 | 2.8914 | 2.7200 | 1000000 | 0.0 | 0.00 |
| NextLat h2 + token KL | `runs/relieved-cover` | 93.4s | 3.1138 | 2.8856 | 2.7139 | 1000000 | 0.0 | 0.00 |
| State+NextLat weak | `runs/illustrious-fowl` | 104.7s | 3.0940 | 2.8929 | 2.6979 | 1000000 | 0.0 | 0.00 |
| State+NextLat default | `runs/phobic-flight` | 115.4s | 3.0649 | 2.8826 | 2.6876 | 1000000 | 0.0 | 0.00 |
| State+NextLat strong | `runs/spotted-care` | 105.2s | 3.0957 | 2.9109 | 2.7109 | 0 | 0.0 | 1.00 |
| JEPA+state | `runs/dispensable-weather` | 105.1s | 2.8750 | 2.5479 | 2.2451 | 750000 | 0.0 | 0.25 |

## 128-Step Follow-Up

| Variant | Run | Time | Train | Teacher CE | Valid | Composite | Verifier | Schema Wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AdamW | `runs/disgusted-muscle` | 110.0s | 1.8217 | 1.5253 | 1.7245 | 500000 | 0.0 | 0.50 |
| JEPA | `runs/smiling-front` | 114.8s | 1.6309 | 1.3941 | 1.5711 | 750000 | 0.0 | 0.25 |
| NextLat h2 | `runs/greasy-bell` | 116.7s | 1.9063 | 1.6045 | 1.8512 | 0 | 0.0 | 1.00 |
| State+NextLat default | `runs/handy-push` | 135.6s | 1.8349 | 1.5143 | 1.8133 | 0 | 0.0 | 1.00 |
| JEPA+state | `runs/female-structure` | 136.0s | 1.6373 | 1.2963 | 1.6075 | 750000 | 0.0 | 0.25 |

## 256-Step Follow-Up

| Variant | Run | Time | Train | Teacher CE | Valid | Composite | Verifier | Schema Wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AdamW | `runs/tremendous-dirt` | 13.7s | 1.1436 | 0.9503 | 1.2478 | 500000 | 0.0 | 0.50 |
| JEPA | `runs/nappy-fuel` | 13.5s | 0.9844 | 0.8576 | 1.1882 | 750000 | 0.0 | 0.25 |
| JEPA+state | `runs/staking-tray` | 18.7s | 1.2287 | 0.9183 | 1.1779 | 500000 | 0.0 | 0.50 |
| NextLat h2 | `runs/terrific-skin` | 13.8s | 1.0623 | 0.9913 | 1.2278 | 0 | 0.0 | 1.00 |
| State+NextLat default | `runs/overrated-insurance` | 18.5s | 1.0781 | 0.9936 | 1.2460 | 250000 | 0.0 | 0.75 |

## 1024-Step Top-Candidate Follow-Up

| Variant | Run | Time | Train | Teacher CE | Valid | Composite | Verifier | Schema Wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AdamW | `runs/receptive-boats` | 50.7s | 1.0641 | 0.6252 | 0.6627 | 500000 | 0.0 | 0.50 |
| JEPA | `runs/ready-doctor` | 53.2s | 1.2602 | 0.6615 | 0.7016 | 750000 | 0.0 | 0.25 |
| JEPA+state | `runs/conscious-friction` | 70.3s | 1.1013 | 0.6131 | 0.6547 | 750000 | 0.0 | 0.25 |

## Supervision Target Semantics

The current latent objectives are separate mechanisms:

- JEPA predicts future hidden states from current hidden states with an energy
  contrastive loss. It does not supervise rho memory or continual plasticity.
- NextLat hidden regression uses `model.next_latent_transition` to predict a
  future hidden state from `(current_hidden, future_token_embedding)`, then
  applies smooth-L1 against the target hidden state. This is latent hidden-state
  supervision.
- NextLat token supervision is optional `token_kl_weight`: the predicted hidden
  state is decoded through the language head and matched to target-hidden logits
  with KL. This is decoded-logit supervision, not direct answer CE.
- Dragon state consistency is the separate `training.latent_reasoning.dragon_state`
  objective. It compares EMA-teacher and student rho rows plus rho energy.
- Continual backprop is not a latent supervision loss. It is optimizer/runtime
  plasticity that replaces low-utility shared low-rank latent features.
- Delayed latent supervision is controlled through
  `training.latent_reasoning.constraint_balancer.start_after_steps`. This blocks
  all latent auxiliary losses until the configured number of optimizer steps has
  completed, then applies the normal cadence and warmup ramp.

## 256-Step Supervision Matrix

| Variant | Time | Train | Valid | Verifier | Schema Wrong | Partial | CBP Events |
|---|---:|---:|---:|---:|---:|---:|---:|
| JEPA | 14.0s | 1.0534 | 1.2183 | 0.25 | 0.25 | 0.00 | 0 |
| NextLat hidden | 14.1s | 1.1311 | 1.2442 | 0.50 | 0.50 | 0.00 | 0 |
| NextLat token only | 14.0s | 1.1098 | 1.2306 | 0.25 | 0.75 | 0.00 | 0 |
| NextLat hidden+token | 13.5s | 1.1376 | 1.2449 | 0.25 | 0.25 | 0.00 | 0 |
| State only | 18.4s | 0.9949 | 1.2391 | 0.00 | 0.75 | 0.00 | 0 |
| State+NextLat | 18.1s | 1.1172 | 1.2324 | 0.00 | 0.00 | 0.00 | 0 |
| JEPA+NextLat | 13.7s | 1.2830 | 1.1616 | 0.00 | 0.50 | 0.00 | 0 |
| JEPA+NextLat token only | 13.8s | 1.3480 | 1.1511 | 0.00 | 0.00 | 0.00 | 0 |
| JEPA+NextLat hidden+token | 14.3s | 1.2790 | 1.1561 | 0.00 | 0.00 | 0.00 | 0 |
| JEPA+state | 18.6s | 1.1111 | 1.1802 | 0.25 | 0.75 | 0.00 | 0 |
| JEPA+state+NextLat | 18.3s | 1.0050 | 1.2108 | 0.00 | 0.00 | 0.00 | 0 |
| JEPA+CBP | 14.0s | 0.9554 | 1.2084 | 0.00 | 0.00 | 0.00 | 8 |

## 1024-Step Supervision Follow-Up

| Variant | Time | Train | Valid | Verifier | Schema Wrong | Partial | CBP Events |
|---|---:|---:|---:|---:|---:|---:|---:|
| JEPA | 51.3s | 1.2800 | 0.7025 | 0.00 | 0.25 | 0.00 | 0 |
| NextLat hidden | 52.7s | 1.2370 | 0.6938 | 0.00 | 1.00 | 0.00 | 0 |
| JEPA+NextLat | 50.4s | 0.9673 | 0.6548 | 0.00 | 0.00 | 0.083 | 0 |
| JEPA+NextLat token only | 52.7s | 0.9593 | 0.6526 | 0.00 | 0.75 | 0.083 | 0 |
| JEPA+NextLat hidden+token | 51.0s | 0.9983 | 0.6590 | 0.00 | 0.50 | 0.50 | 0 |
| JEPA+state+NextLat | 72.2s | 1.0566 | 0.6890 | 0.00 | 0.50 | 0.25 | 0 |
| JEPA+CBP | 51.5s | 1.4530 | 0.7372 | 0.25 | 0.00 | 0.25 | 32 |

## 4096-Step Candidate Follow-Up

| Variant | Time | Train | Valid | Verifier | Semantic | Schema Wrong | Partial | CBP Events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| JEPA | 205.7s | 1.3000 | 0.6387 | 0.50 | 0.50 | 0.25 | 0.50 | 0 |
| JEPA+NextLat | 213.6s | 1.3285 | 0.6057 | 0.00 | 0.00 | 0.50 | 0.00 | 0 |
| JEPA+CBP | 207.6s | 1.2537 | 0.6199 | 0.00 | 0.00 | 0.50 | 0.25 | 128 |

## 4096-Step Probe32 Schedule Matrix

This matrix repeats the 4096-step fixed-small setup with 32 ruliad correctness
probe items instead of the older 4-item smoke probe.

| Variant | Aux Calls | Time | Train | Valid | Verifier | Semantic | Schema Wrong | Malformed | Partial |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| JEPA | 512 | 207.3s | 1.3367 | 0.6343 | 0.00 | 0.00 | 0.219 | 0.000 | 0.177 |
| JEPA+NextLat | 512 | 216.4s | 1.2678 | 0.6157 | 0.00 | 0.00 | 0.375 | 0.094 | 0.052 |
| JEPA+NextLat sparse16 | 256 | 207.1s | 1.2696 | 0.6069 | 0.00 | 0.00 | 0.281 | 0.000 | 0.000 |
| JEPA+NextLat sparse32 | 128 | 205.2s | 1.3379 | 0.6111 | 0.00 | 0.00 | 0.406 | 0.000 | 0.000 |
| JEPA+NextLat weak | 512 | 216.1s | 1.2964 | 0.6061 | 0.094 | 0.094 | 0.406 | 0.000 | 0.094 |
| JEPA+NextLat delayed1024 | 384 | 205.8s | 1.2733 | 0.6079 | 0.00 | 0.00 | 0.250 | 0.000 | 0.000 |
| JEPA+NextLat delayed2048 | 256 | 213.8s | 1.2740 | 0.6085 | 0.00 | 0.00 | 0.344 | 0.000 | 0.010 |
| JEPA+NextLat delayed1024 sparse16 | 192 | 214.4s | 1.2948 | 0.6165 | 0.00 | 0.00 | 0.281 | 0.000 | 0.000 |
| JEPA+CBP mild | 512 | 217.1s | 1.3177 | 0.6418 | 0.00 | 0.00 | 0.375 | 0.000 | 0.250 |

## 4096-Step Weak Schedule Follow-Up

This follow-up checks whether the weak-scale verifier signal survives when
paired with sparse/delayed schedules.

| Variant | Aux Calls | Time | Train | Valid | Verifier | Semantic | Schema Wrong | Partial |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| JEPA+NextLat weak | 512 | 207.8s | 1.2518 | 0.6070 | 0.00 | 0.00 | 0.531 | 0.042 |
| JEPA+NextLat weak sparse16 | 256 | 213.8s | 1.2841 | 0.6145 | 0.00 | 0.00 | 0.375 | 0.000 |
| JEPA+NextLat weak delayed1024 | 384 | 214.6s | 1.3038 | 0.6117 | 0.00 | 0.00 | 0.375 | 0.000 |
| JEPA+NextLat weak delayed1024 sparse16 | 192 | 214.5s | 1.2670 | 0.6013 | 0.00 | 0.00 | 0.406 | 0.000 |
| JEPA+NextLat weak001 | 512 | 207.9s | 1.2755 | 0.6109 | 0.00 | 0.00 | 0.469 | 0.000 |
| JEPA+NextLat sparse16 rerun | 256 | 206.7s | 1.2608 | 0.6108 | 0.00 | 0.00 | 0.438 | 0.000 |
| JEPA+NextLat delayed1024 rerun | 384 | 215.2s | 1.2979 | 0.6057 | 0.00 | 0.00 | 0.500 | 0.000 |

## 4096-Step Three-Seed Schedule Aggregate

This aggregate combines the default-seed 4096-step probe32 runs with two
additional seed overlays. It is the current best read on whether NextLat
scheduling is a robust improvement rather than a single-run fluctuation.

| Variant | N | Valid Mean | Valid Std | Verifier | Semantic | Partial | Schema Wrong | Malformed | Health PPM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| JEPA | 3 | 0.6163 | 0.0230 | 0.0208 | 0.0208 | 0.2882 | 0.3333 | 0.0000 | 666667 |
| JEPA+NextLat | 3 | 0.6186 | 0.0168 | 0.0208 | 0.0208 | 0.0486 | 0.4167 | 0.0313 | 552083 |
| JEPA+NextLat sparse16 | 3 | 0.6157 | 0.0120 | 0.0000 | 0.0000 | 0.0729 | 0.4688 | 0.0104 | 520833 |
| JEPA+NextLat delayed1024 | 3 | 0.6148 | 0.0149 | 0.0104 | 0.0104 | 0.0104 | 0.4896 | 0.0000 | 510417 |
| JEPA+NextLat weak delayed1024 sparse16 | 3 | 0.6111 | 0.0197 | 0.0208 | 0.0208 | 0.0208 | 0.6146 | 0.0000 | 385417 |

## 4096-Step Decoupled Schedule Pass

This pass uses per-objective cadence/start controls: JEPA keeps the base
cadence while only NextLat is sparse and/or delayed.

| Variant | Time | Train | Valid | Verifier | Semantic | Partial | Schema Wrong | Malformed | Health PPM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Decoupled sparse16 | 206s | 1.2954 | 0.6097 | 0.00 | 0.00 | 0.031 | 0.219 | 0.156 | 625000 |
| Decoupled delayed1024 | 216s | 1.2996 | 0.6058 | 0.00 | 0.00 | 0.063 | 0.688 | 0.000 | 312500 |
| Decoupled delayed1024 sparse16 | 207s | 1.2940 | 0.5986 | 0.00 | 0.00 | 0.000 | 0.406 | 0.000 | 593750 |
| Decoupled weak delayed1024 sparse16 | 204s | 1.2537 | 0.6104 | 0.00 | 0.00 | 0.000 | 0.438 | 0.000 | 562500 |

## Readout

- JEPA is still the best practical objective family in this ablation, but the
  1024-step result is not a clean "JEPA-only wins everything" story. JEPA wins
  64/128/256-step validation and completion health, but AdamW slightly beats
  JEPA-only on 1024-step validation CE. JEPA+state wins 1024-step validation CE
  and keeps the JEPA completion-health advantage, but adds about 31-38% wall
  time overhead in the 256/1024-step release runs.
- Dragon rho-state consistency by itself is not useful here. It is slower than
  JEPA and worse than AdamW on teacher CE, validation aggregate, and ruliad
  completion health.
- NextLat alone improves shallow completion-health style metrics at 64 steps,
  but it does not improve CE and collapses to schema-valid wrong completions by
  128 steps in this setup. Horizon 2 is slightly better than horizon 1/4 at 64
  steps, but none of the horizons improve verifier.
- Adding a small token KL to NextLat does not help in this matrix.
- State+NextLat default is a mild 64-step improvement over NextLat-only on
  aggregate validation, but the advantage does not survive to 128 steps. It also
  adds runtime overhead from the EMA teacher/state pass.
- Strong rho-state weighting is clearly too much: it produces the worst 64-step
  ruliad score and all schema-valid-wrong completions.
- JEPA+state is the most interesting hybrid: it has the best 128-step
  teacher-forced CE, the best 1024-step validation CE, and the same 1024-step
  completion-health composite as JEPA. Its cost means it should be promoted as a
  second candidate, not silently defaulted over JEPA-only.
- NextLat hidden regression becomes interesting only when paired with JEPA. At
  1024 steps, JEPA+NextLat improves validation and clean completion health. At
  4096 steps it still improves validation average, but the tiny correctness
  probe regresses versus JEPA, so it is not ready to replace JEPA as the default.
- Strong decoded-logit/token-KL NextLat is not stable enough yet. The token-only
  and hidden+token JEPA variants look good at 256 steps, but by 1024 steps they
  have worse schema health than JEPA+NextLat hidden regression.
- The aggressive JEPA+CBP arm proves plasticity telemetry and replacement are
  active, but it does not beat JEPA at 4096 steps. Its 1024-step verifier hit is
  not durable enough to promote without a multi-seed/larger-probe follow-up.
- With a 32-item probe, sparse/delayed JEPA+NextLat consistently improves
  validation CE versus JEPA, but the correctness signal is still weaker. The
  best validation number in this pass is weak+delayed1024+sparse16, but it has
  no verifier/semantic/partial progress and worse schema health than JEPA.
- The one nonzero verifier result in the schedule matrix came from always-on
  weak JEPA+NextLat, but it was not reproduced in the focused weak follow-up.
  Treat this as sample noise until it survives multiple seeds.
- The three-seed schedule aggregate does not promote NextLat yet. The best mean
  validation loss is the weak delayed+sparse variant, but it pays for that with
  much worse schema-wrong rate and completion health. JEPA-only has the best
  partial progress and health, and no malformed completions.
- Decoupling JEPA and NextLat cadence/start is mechanically useful and should
  replace the older coupled schedule profiles for future experiments. The first
  decoupled delayed+sparse run achieves the best single-run validation loss in
  this family, but it still has zero verifier/semantic/partial progress. The
  sparse-only decoupled arm preserves JEPA-like schema-wrong rate, but introduces
  malformed completions, so it is not a promotion candidate either.
- Mild CBP did not improve validation or verifier behavior, although it did
  raise partial progress. It should remain a plasticity experiment rather than a
  default training component.

## Recommendation

Promote the JEPA family to the default auxiliary objective for ruliad continual
pretraining. The checked-in default profiles are:

- `crates/burn_dragon_p2p/deploy/profiles/ruliad-r1.jepa.training.toml`
- `crates/burn_dragon_p2p/deploy/profiles/ruliad-1m.jepa.training.toml`
- `crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-16k.jepa.training.toml`
- `crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-32k.jepa.training.toml`
- `crates/burn_dragon_p2p/deploy/profiles/ruliad-1m-la-64k.jepa.training.toml`

Do not default to NextLat or Dragon-state consistency yet. Keep the
implementation and profiles for longer stability experiments, but use JEPA as
the first auxiliary candidate. JEPA+state is now strong enough to keep in the
promotion lane through `ruliad-r1.jepa-state.training.toml`: it should displace
JEPA-only only if multi-seed, longer-horizon runs show better collapse
resistance or verifier/schema progress despite its extra runtime cost.

Treat JEPA+NextLat hidden regression as the next experimental candidate, not as
the default. It appears to improve validation CE, but current evidence says it
can trade away verifier/schema behavior. The likely next useful variant is
delayed or sparse JEPA+NextLat, not always-on token-KL decoding.

The current best practical recipe remains JEPA-only. The current best research
candidate is JEPA+NextLat hidden regression with sparse or delayed scheduling,
because it sometimes improves validation CE without runtime cost explosion.
However, the three-seed aggregate shows that the current schedules are still
trading away verifiable output structure, so they should not be promoted until
schema/correctness metrics improve alongside CE.

Use the decoupled NextLat schedules for future NextLat work. They are cleaner
than the older coupled profiles because JEPA can remain active while only
NextLat is sparse/delayed.

The current rho-state objective is mechanically correct and covered by unit
tests, but this ablation does not show it is a better training objective at this
scale.

## Next Ablation Steps

- Run the next promotion gate on longer 16k-32k step windows for JEPA-only,
  JEPA+state, and the best sparse/delayed JEPA+NextLat candidates. Require
  verifier/schema parity with JEPA before considering NextLat promotion.
- Separate cadence and weight controls for JEPA, NextLat, and rho-state losses.
  The current single `training.latent_reasoning.every_steps` setting makes
  low-cost JEPA and expensive state consistency move together, which is too
  blunt for production training.
- Add explicit rho-dynamics telemetry: student/teacher rho drift, rho energy
  drift, slot redundancy, hidden-state variance, and chunk-boundary prediction
  error. This is the missing signal for whether rho is stabilized through time.
- Rework NextLat as a delayed state-prediction curriculum: enable it only after
  CE and schema health are stable, start with horizon 1, then sweep horizons
  2/4/8 against verifier/schema/degen metrics. The 64/128-step evidence says
  always-on NextLat is not safe enough.
- Avoid treating composite ruliad score as a promotion target until the probe is
  larger and verifier accuracy is nonzero. For now, composite is dashboard
  context, not a capability objective.

Concrete next matrix:

| Gate | Arms | Purpose | Promotion Rule |
|---|---|---|---|
| Telemetry-only | AdamW, JEPA, JEPA+state | Establish rho drift/redundancy and output-degen baselines without changing objectives. | No objective promotion from this gate; it validates the diagnostic signals. |
| Sparse state | JEPA, JEPA+state every 16/32/64 aux steps | Check whether state consistency helps when decoupled from JEPA cadence. | Keep only arms that improve validation or degen metrics without worse schema health and with less than 15% overhead. |
| Delayed state | JEPA+state activated after CE/schema-health thresholds | Test whether state consistency is harmful before the token model has a stable manifold. | Promote over JEPA-only only if it improves 4096-step validation and does not regress completion health. |
| Delayed NextLat | JEPA+NextLat h1, h2, h4 after stability threshold | Test next-latent prediction as continuation supervision rather than always-on regularization. | Continue only if schema-wrong stays below JEPA-only and rho/chunk-boundary prediction improves. |
| Larger-profile check | AdamW, JEPA, JEPA+best-state on 1M/16k profile | Verify that the candidate scales beyond the toy fixed profile. | Default promotion requires no throughput collapse and no verifier/schema regression. |

Immediate follow-up:

- Add fixed profiles with at least 32 or 128 ruliad correctness probe items. The
  4-item probe is too coarse to distinguish real verifier progress from sample
  noise.
- Add sparse and delayed JEPA+NextLat hidden regression profiles. Candidate
  settings: every 16/32 aux steps, or activation after validation CE and schema
  health are stable.
- Keep token-KL decode supervision behind an experimental flag until it stops
  worsening schema health at 1024+ steps.
- Re-run 4096-step JEPA, sparse16 JEPA+NextLat, delayed1024 JEPA+NextLat, and
  weak+delayed1024+sparse16 JEPA+NextLat across at least three seeds.
- Add per-objective cadence controls if we want JEPA frequent but NextLat sparse.
  The current `every_steps` cadence gates all latent objectives together, so
  sparse NextLat also makes JEPA sparse.

Required metrics before the next promotion gate:

- `rho_drift_l1`, `rho_drift_cosine`, and `rho_energy_drift` for EMA-teacher
  state targets.
- `rho_slot_redundancy` and `rho_slot_variance` so memory collapse is visible
  before output collapse.
- `chunk_boundary_hidden_prediction_error` and
  `chunk_boundary_rho_prediction_error` for TBPTT continuity.
- Per-family/per-difficulty verifier, schema-wrong, and completion-health rates
  with at least 32 ruliad correctness probe items. Four-item probes are useful
  for smoke tests only.
