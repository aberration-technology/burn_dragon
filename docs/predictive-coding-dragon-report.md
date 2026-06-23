# Predictive Coding State Correction in Dragon TBPTT Training

Date: 2026-06-20

## Status

This document is a preprint draft and experiment protocol. It is **not publish-grade yet** because
the current evidence is a small local ablation, not a complete controlled study.

The correct paper framing is a neutral result: predictive-coding (PC) recurrent-state correction is
technically viable in Dragon training, but the current evidence does not show a fixed-wall-clock
advantage over AdamW. The draft should not claim that PC prevents collapse, improves long-run
continual learning, or should be enabled by default until the experiment matrix below is complete.

## Abstract Draft

We evaluate predictive coding as a recurrent-state correction mechanism for Dragon TBPTT language
training. The implementation corrects recurrent latent state before the optimizer step while AdamW
continues to update model parameters. On the local GB10 CUDA path, the implementation runs in a
dense regime: the 1M ruliad batch-64 profile sustains roughly 85-89% active GPU utilization at
about 40-41 W, avoiding the earlier host-synchronization pathology.

The learning result is mixed. In a three-seed 512-step matrix, AdamW+PC improves mean validation
loss versus AdamW, but the advantage disappears by 2048 steps while the throughput cost remains
roughly 29%. A state-only control, where PC corrects recurrent state but parameters are not
updated, does not learn durable validation behavior. These results support PC as a benchmarkable
state-inference ablation, not yet as a practical replacement or default companion for AdamW.

## Method

The training path under test is Dragon language modeling with TBPTT. In AdamW+PC mode, PC performs
one or more recurrent-state correction steps inside each selected TBPTT chunk, then normal gradient
training updates parameters. The recommended state-correction ablation uses:

```toml
[training]
tbptt_chunk_size = 64
batch_size = 64

[training.predictive_coding]
enabled = true
mode = "recurrent_state"
state_scope = "core"
backward_mode = "chunked"
parameter_update = "optimizer"
steps = 1
step_size = 0.01
apply_every_chunks = 2
sync_diagnostics = false
```

The state-only control keeps the same state correction but disables parameter mutation:

```toml
[training.predictive_coding]
enabled = true
parameter_update = "state_only_control"
```

The implementation also exposes a first-class predictive-coding optimizer path:

```toml
[optimizer]
name = "predictive_coding"
learning_rate = 0.001
weight_decay = 0.0

[optimizer.predictive_coding]
transform = "sgd" # sgd | momentum | adamw | diagonal_natural

[training.predictive_coding]
enabled = true
backward_mode = "chunked"
parameter_update = "optimizer"
```

That optimizer path is validated and smoke-tested, but it is appendix material until it has its own
controlled matrix. The main scientific claim remains about PC state correction plus AdamW.

Paper-matrix overlays disable adaptive dynamics recovery, continual backprop, and neuron scaling.
Those systems are important continual-learning machinery, but they would confound an optimizer and
state-correction ablation by changing the run after collapse, plateau, or capacity events.

## Existing Evidence

Primary runtime profile:

- backend: CUDA
- dataset: `crates/burn_dragon_p2p/deploy/profiles/ruliad-1m.jepa.training.toml`
- model: 4 layers, 128 embedding dim, 4 heads, 512 latent total
- block size: 256
- batch size: 64
- TBPTT chunk size: 64
- optimizer baseline: AdamW
- PC: recurrent-state correction, core state, one correction step, step size 0.01, every other chunk

Raw local artifacts from the initial matrix:

- `target/pc-paper/pc_ablation_all_summary.csv`
- `target/pc-paper/pc_ablation_gpu_20260620123459.csv`
- `target/pc-paper/pc_ablation_2048_gpu_20260620124632.csv`

### 512-Step Three-Seed Matrix

Each run processes 8,388,608 tokens.

| Arm | Seeds | Wall s | Tokens/s | Last train loss | Last valid loss | Source loss | Mean difficulty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AdamW | 3 | 58.83 +/- 0.15 | 142,584 +/- 374 | 0.1410 +/- 0.0085 | 0.4233 +/- 0.1073 | 0.1662 +/- 0.0237 | 5.1479 +/- 0.0041 |
| AdamW+PC | 3 | 76.11 +/- 0.10 | 110,212 +/- 142 | 0.1361 +/- 0.0015 | 0.2915 +/- 0.0096 | 0.1448 +/- 0.0051 | 5.1511 +/- 0.0004 |
| State-only control | 3 | 75.79 +/- 0.05 | 110,687 +/- 79 | 2.3036 +/- 0.6521 | 2.1172 +/- 0.0160 | 2.8364 +/- 0.0356 | 4.9393 +/- 0.0306 |

At 512 steps, AdamW+PC has better mean validation loss and source-selection loss than AdamW, but
costs roughly 29% more wall time. The state-only control does not learn durable validation
behavior.

### 2048-Step Three-Seed Matrix

Each run processes 33,554,432 tokens.

| Arm | Seeds | Wall s | Tokens/s | Last train loss | Last valid loss | Source loss | Mean difficulty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AdamW | 3 | 232.42 +/- 0.49 | 144,368 +/- 306 | 1.0363 +/- 0.0339 | 0.1413 +/- 0.0447 | 1.0524 +/- 0.0131 | 5.6811 +/- 0.0143 |
| AdamW+PC | 3 | 300.32 +/- 0.19 | 111,728 +/- 69 | 1.0405 +/- 0.0142 | 0.1411 +/- 0.0381 | 1.0634 +/- 0.0099 | 5.6815 +/- 0.0141 |

Paired AdamW+PC minus AdamW deltas at 2048 steps:

| Metric | Mean delta | Interpretation |
| --- | ---: | --- |
| Last valid loss | -0.0002 +/- 0.0091 | no meaningful validation advantage |
| Source loss | +0.0109 +/- 0.0171 | no source-selection advantage |
| Last train loss | +0.0043 +/- 0.0201 | effectively tied |
| Wall time | +67.90 +/- 0.31 s | PC is consistently slower |
| Tokens/s | -32,641 +/- 239 | PC throughput is 77.4% of AdamW |

## Publish-Grade Experiment Matrix

Use `scripts/pc_paper_experiments.sh` for new runs and `scripts/pc_paper_analyze.py` for
aggregation.

### Main Fixed-Token Matrix

```bash
scripts/pc_paper_experiments.sh \
  --matrix main-fixed-token \
  --out-dir target/pc-paper/main-fixed-token-$(date -u +%Y%m%dT%H%M%SZ)
```

Required arms:

- AdamW baseline
- AdamW+PC recommended: core state, chunked backward, one correction step, `step_size = 0.01`, every other chunk
- AdamW+PC every chunk: same as recommended but `apply_every_chunks = 1`

Required seeds and horizons:

- seeds: `20260621,20260622,20260623,20260624,20260625`
- fixed-token horizons: `2048` and `8192` iterations

### Controls

```bash
scripts/pc_paper_experiments.sh --matrix controls
```

Required control:

- state-only PC at 512 and 2048 steps, three seeds

This control prevents misinterpreting transient latent correction as durable learning.

### Fixed-Wall-Clock Matrix

```bash
scripts/pc_paper_experiments.sh --matrix wall-clock --wall-clock-seconds 3600
```

Required arms:

- AdamW
- AdamW+PC recommended

Required seeds:

- `20260621,20260622,20260623`

This matrix is mandatory because PC is slower at fixed tokens. A positive fixed-token result is not
practically meaningful if AdamW wins at equal wall clock by processing more tokens.

### Long Stability Probe

```bash
scripts/pc_paper_experiments.sh --matrix stability --wall-clock-seconds 21600
```

Required arms:

- AdamW
- AdamW+PC recommended

Required seeds:

- two seeds minimum

Primary question: does PC reduce validation regression, output degeneracy, verifier correctness
regression, or source-selection collapse over longer continual training?

### PC Optimizer Appendix

```bash
scripts/pc_paper_experiments.sh --matrix pc-optimizer
```

Required transforms:

- `sgd`
- `momentum`
- `adamw`
- `diagonal_natural`

This section should stay in the appendix unless the optimizer path clearly beats AdamW in both
fixed-token and fixed-wall-clock comparisons.

## Analysis Protocol

Aggregate raw artifacts with:

```bash
scripts/pc_paper_analyze.py \
  target/pc-paper \
  --out-dir target/pc-paper/analysis \
  --baseline adamw \
  --compare adamwpc
```

The analyzer writes:

- `normalized_summary.csv`: normalized legacy summary rows
- `summary_by_arm.csv`: mean and 95% CI by iteration count and arm
- `paired_deltas.csv`: paired seed deltas for AdamW+PC minus AdamW
- `event_run_summary.csv`: final event-stream metrics per generated run
- `source_bucket_summary.csv`: final source-selection bucket telemetry when present
- `gpu_summary.csv`: GPU utilization and power summaries
- `manifest_summary.csv`: trial metadata, command context, git SHA, status, and run directory
- `paper_tables.md`: compact Markdown tables for the manuscript

Primary outcome metrics:

- validation loss at fixed tokens
- validation loss at fixed wall clock
- ruliad source loss
- normalized and mean source difficulty
- verifier correctness and failure rate
- output degeneracy: entropy, max probability, distinct-2, repetition, dominant periodicity

Efficiency metrics:

- tokens/sec
- wall-clock seconds
- PC correction milliseconds
- GPU utilization and power
- energy/token when power samples are dense enough

Statistical rules:

- Use paired seed deltas for AdamW+PC vs AdamW.
- Report mean, standard deviation, and 95% CI.
- Keep fixed-token and fixed-wall-clock conclusions separate.
- Do not claim improvement unless quality and wall-clock results both support it.

## Claim Boundary

Supported by current evidence:

1. PC recurrent-state correction can run without pathological CPU transfer on the 1M batch-64 CUDA profile.
2. Core-state, chunked-backward PC is the best current implementation mode.
3. State-only PC correction is not a viable substitute for parameter optimization.
4. AdamW+PC gives an early 512-step validation improvement but no replicated 2048-step advantage.

Not yet supported:

1. PC improves long-run continual learning.
2. PC prevents output degeneracy or collapse.
3. PC is worth its throughput cost by default.
4. The first-class PC optimizer path is competitive with AdamW.

## Acceptance Gate For An arXiv Submission

The paper is publish-grade only after all of these are true:

- the 5-seed fixed-token matrix is complete
- the 3-seed fixed-wall-clock matrix is complete
- the long stability probe has at least two seeds per main arm
- every run has a manifest with git SHA, config overlay, command, seed, hardware/backend, and run directory
- source-selection buckets, verifier metrics, output degeneracy metrics, and gate events are included in the analysis tables
- the conclusion remains neutral unless both fixed-token and fixed-wall-clock results support a stronger claim

Until then, this document should be treated as a reproducible internal report and protocol, not as
a finished arXiv paper.
