# Ghostline Universal Policy - Model Card

## Status

Pipeline implemented and smoke-tested. The fair-teacher gate has passed; neural champion training and final neural evaluation are still pending. No neural acceptance or superhuman claim is made.

## Model

- Inputs: player-equivalent `GhostlineEnv-v2` dictionary observation, including the explicit objective vector.
- Actions: masked `Discrete(36)` movement, dash, and pulse combinations.
- Encoders: local-grid convolution, ego/objective/ray MLPs, and masked-attention target/entity pooling.
- Memory: configurable 256- or 384-unit GRU over recurrent sequences; 384 is the default candidate.
- Heads: separate 256-unit policy/value decoders plus goal-bearing and visible-danger auxiliaries.
- Learning: pure in-project PyTorch behavior cloning from the observation-only teacher, DAgger recovery aggregation, recurrent clipped PPO, generalized advantage estimation, decaying RND, and adaptive procedural curriculum; no external RL framework is required.

## Data and fairness

No human demonstrations, human trajectories, hidden generator state, privileged critic state, or unseen enemy state are used. The automated teacher receives the exact public v2 observation and action mask. Training seeds are below 1,000,000, validation uses 1,000,000-1,049,999, and final evaluation begins at 2,000,000. All attempted final-test slices are immutable: 2M-6M are historical pre-freeze evidence, while the tracked 7M slice remains reserved and unopened for the selected frozen-distribution neural champion.

The failed 2.98-million-decision pure-PPO attempt is retained as historical negative evidence only; it predates the final environment fingerprint and is not represented as an equal-budget current-distribution ablation. The hybrid pipeline is not presented as successful until final held-out evaluation passes.

## Current fair-teacher qualification

The final observation-only teacher passed two disjoint current-fingerprint validation gates covering 200 seeds per tier. Tier 1-6 success was `100/100/100/100/100/95%` at offset 6,000 and `100/99.5/99.5/100/100/94%` at offset 7,000. The machine-readable reports are [`teacher-current-validation-c-200.json`](../benchmarks/teacher/teacher-current-validation-c-200.json) and [`teacher-current-validation-d-200.json`](../benchmarks/teacher/teacher-current-validation-d-200.json). These gates qualify trajectory collection; they are not neural or final-test results.

## Historical pre-freeze teacher result

The following 6M result used the same public observation/action contract, but predates the final route, security, patrol, and shared-perception freeze. It is retained as transparent historical evidence, not represented as the current release gate.

| Tier | Teacher success | Wilson 95% interval | Mean damage | Dominant failure |
|---|---:|---:|---:|---|
| 1 - Orientation | 500/500 (100.0%) | 99.24%-100.00% | 0.000 | None |
| 2 - Surveillance | 500/500 (100.0%) | 99.24%-100.00% | 0.000 | None |
| 3 - Patrol | 479/500 (95.8%) | 93.66%-97.24% | 0.528 | Integrity loss (21) |
| 4 - Countermeasure | 481/500 (96.2%) | 94.14%-97.55% | 0.568 | Integrity loss (19) |
| 5 - Lockdown | 486/500 (97.2%) | 95.36%-98.32% | 0.418 | Integrity loss (14) |
| 6 - Ghostline | 444/500 (88.8%) | 85.73%-91.27% | 0.948 | Integrity loss (56) |

The aggregate report is tracked at [`benchmarks/teacher/teacher-release-gate-6m-500.json`](../benchmarks/teacher/teacher-release-gate-6m-500.json) with a [CSV export](../benchmarks/teacher/teacher-release-gate-6m-500.csv). The [audit history](../benchmarks/teacher/README.md) records why every 2M-6M result is now historical.

The immutable retired audits measured tiers 1-6 at `100.0/100.0/97.0/94.8/96.8/80.8` on 3M, `100.0/100.0/96.6/95.2/97.2/83.8` on 4M, and `100.0/100.0/97.6/93.6/97.4/85.6` on 5M. They are reported as negative generalization evidence and were never reopened after inspection.

Tier-6 directional inertia was selected only on two disjoint validation slices, never on a final-test slice. No code or setting changed while the 6M gate was open; later player-facing mechanics and perception changes deliberately retired that evidence and required fresh current-fingerprint teacher qualification.

These results validate teacher trajectory quality; they are not neural-policy results. The selected neural checkpoint, 500-seed neural benchmark, Python/ONNX parity, and matched-seed human cohort remain pending.

## Acceptance results

| Tier | Target | Measured neural success | Wilson 95% interval |
|---|---:|---:|---:|
| 1 - Orientation | 95% | Pending | Pending |
| 2 - Surveillance | 95% | Pending | Pending |
| 3 - Patrol | 95% | Pending | Pending |
| 4 - Countermeasure | 95% | Pending | Pending |
| 5 - Lockdown | 95% | Pending | Pending |
| 6 - Ghostline | 85% | Pending | Pending |

## Intended use and limitations

This is a portfolio/research policy for procedural, partially observed game RL. Results are tied to the frozen v2 mechanics and observation contract. Deterministic success does not imply optimal trace, optional-data collection, or route efficiency, and no comparison with real players is valid until the planned matched-seed human cohort is complete.

## Deployment precision gate

The release exporter retains a canonical FP32 ONNX graph and may derive a
dynamic-INT8 deployment candidate. Each graph is replayed independently for
1,000 player-equivalent recurrent transitions against deterministic PyTorch
actions. INT8 is selected only at zero action mismatches; any mismatch or
quantizer/runtime error falls back to the already verified FP32 graph. The
sibling parity report records artifact sizes, SHA-256 hashes, mismatch counts,
the first mismatch index, recurrent width, observation contract, and selected
deployment precision. Final champion sizes and parity hashes remain pending
until the neural checkpoint passes the held-out acceptance gate.
