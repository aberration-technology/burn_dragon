# Stateful TBPTT Ruliad Report

Status: screening evidence, not promoted

Date: 2026-08-03

## Scope

This report covers the matched local CUDA screen for ordered multi-chunk Ruliad documents,
reset-versus-persistent recurrent state, exact free-run verification, and segment-balanced
trace/answer supervision. The screened model is intentionally small: 4 layers, embedding width
128, 4 heads, latent total 512, block size 512, TBPTT chunk size 128, and batch size 32.

All runs used the same static source plan and canonical validation panel. They do not establish
large-model convergence, unbounded-difficulty mastery, browser parity, or P2P convergence.

## GPU Diagnosis

The dense training loop is not CPU data-bound on the GB10. In the balanced 2,048-step screen,
model duty was 85.6% for reset and 83.7% for carry. Data-loader foreground wait was 1.35% of reset
wall time. A live 30-second sample remained at 78-84% SM utilization and 38-40 W throughout the
dense phase.

The visible low-duty intervals came from synchronous autoregressive correctness probes. The
verifier consumed 11.6% of reset wall time and 12.9% of carry wall time. GPU statistics in matrix
reports cover both training and evaluation, so they must be read together with model duty and
validation fraction.

### Device-buffer screen

One reset arm, seed 4242, 64 updates, batch 32, 32 verifier items, and a 128-token generation
budget was used to select the accelerator-resident greedy decode buffer.

| buffered tokens | host syncs | validation s | wall tok/s | model tok/s | mean GPU % |
|---:|---:|---:|---:|---:|---:|
| 16 | 325 | 14.26 | 51,025 | 173,596 | 44.3 |
| 32 | 165 | 12.32 | 56,856 | 178,230 | 46.3 |
| 64 | 85 | 11.95 | 57,644 | 175,097 | 47.8 |
| 128 | 45 | 13.48 | 53,350 | 177,158 | 40.9 |

The profile therefore uses 64 buffered tokens. The 128-token arm reduced synchronization further
but lost throughput to unnecessary speculative recurrent work.

## Supervision Screen

`balance_trace_answer_mass` derives one answer scale from the observed trace and answer mask mass
of each full document. The full mask is built before the streaming loader slices TBPTT chunks, so
the answer is balanced against the complete multi-chunk trace. No family-specific or task-specific
coefficient is used.

The audit covered 96 canonical difficulty-0 documents. Relative to unbalanced trace-and-answer
supervision, realized mask mass increased by 1.98x for `advance_proof`, 1.97x for `check_proof`, and
1.34x for `construct_proof`; the difference follows from each task's observed answer length.

### Matched 2,048-step result

One seed, CUDA, batch 32. This is a screening comparison and has no confidence interval.

| objective/state | train loss | valid loss | paired warm | paired cold | verifier | partial | model tok/s | model duty | validation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| unbalanced/reset | 0.2044 | 0.7505 | 0.5207 | 0.5313 | 0/32 | 7/32 | 174,706 | 80.9% | 16.4% |
| unbalanced/carry | 0.2237 | 0.7422 | 0.4977 | 0.5300 | 0/32 | 7/32 | 167,685 | 80.4% | 16.4% |
| balanced/reset | 0.2235 | 0.6782 | 0.4974 | 0.5101 | 1/32 | 10/32 | 175,318 | 85.6% | 11.6% |
| balanced/carry | 0.2538 | 0.6452 | 0.4503 | 0.4697 | 3/32 | 10/32 | 166,250 | 83.7% | 12.9% |

Balanced carry produced three distinct exact answers, but all were difficulty-0 `advance_proof`
actions. At the final epoch:

- `advance_proof`: 3/13 verifier matches
- `check_proof`: 0/9 verifier matches, 7/9 partial credit
- `construct_proof`: 0/10 verifier matches, 10/10 malformed

Output entropy also fell to 0.277 bits. The result therefore supports further local study of
balanced supervision but does not qualify it as a general reasoning objective or justify P2P
promotion.

## Acceptance Gates

The next promotion matrix must use at least three seeds and retain the same data, model, optimizer,
batch, verifier panel, and generation settings. A candidate must:

1. Preserve dense model throughput within 10% of the matched unbalanced arm.
2. Produce nonzero verifier accuracy across more than one task contract.
3. Avoid malformed `construct_proof` collapse and preserve output diversity.
4. Show a positive state-carry effect with a paired confidence interval.
5. Pass native/P2P/browser serialization and initial-state parity only after the local quality
   gates pass.

Artifacts are under `target/experiments/stateful-tbptt/`, notably
`balanced-segment-2048-seed4242-v1` and `chunk128-convergence-2048-seed4242-v1`.
