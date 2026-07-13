# Ghostline Universal Policy - Model Card

## Status

Release candidate selected and independently audited. The frozen neural policy
passes the six-tier acceptance gate, matches its FP32 ONNX export on 1,000
recurrent transitions, runs in the Windows and web builds, and is tied to the
environment fingerprint below. No superhuman claim is made because the planned
matched-seed human cohort has not yet been collected.

## Model

- Inputs: player-equivalent `GhostlineEnv-v2` dictionary observation, including the explicit objective vector.
- Actions: masked `Discrete(36)` movement, dash, and pulse combinations.
- Encoders: local-grid convolution, ego/objective/ray MLPs, and masked-attention target/entity pooling.
- Memory: configurable 256- or 384-unit GRU over recurrent sequences; 384 is the default candidate.
- Heads: separate 256-unit policy/value decoders plus goal-bearing and visible-danger auxiliaries.
- Learning: pure in-project PyTorch behavior cloning from the observation-only teacher, DAgger recovery aggregation, recurrent clipped PPO, generalized advantage estimation, decaying RND, and adaptive procedural curriculum; no external RL framework is required.

## Data and fairness

No human demonstrations, human trajectories, hidden generator state, privileged critic state, or unseen enemy state are used. The automated teacher receives the exact public v2 observation and action mask. Training seeds are below 1,000,000, validation uses 1,000,000-1,049,999, and final evaluation begins at 2,000,000. All attempted final-test slices are immutable: 2M-6M are historical pre-freeze evidence, and the tracked 7M slice was consumed exactly once by the selected frozen-distribution neural champion. Its report and artifact hashes are bound in [`benchmarks/final-test-slices.json`](../benchmarks/final-test-slices.json).

The failed 2.98-million-decision pure-PPO attempt is retained as historical negative evidence only; it predates the final environment fingerprint and is not represented as an equal-budget current-distribution ablation. A current-fingerprint conservative PPO pilot was also rejected: it scored `84/94/98/92/90/86%`, while the immutable DAgger rollback scored `96/96/90/98/92/94%` on the same seeds. The release therefore uses the better BC+DAgger checkpoint and does not mislabel it as PPO-trained.

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

These results validate teacher trajectory quality; they are not neural-policy results. The selected neural checkpoint and its independent final benchmark are reported below. The matched-seed human cohort remains pending.

## Acceptance results

The frozen checkpoint has SHA-256
`458aa28d14b0829481a56c96dcc97a9ab9af2c463c15beef94d4c3e86ab59deb`
and environment fingerprint
`17d8617fd92015dc5a00b5314558fc7c0ff957685b12966efa5253806463739b`.
The one-time 7M audit ran 500 deterministic, unseen seeds per tier:

| Tier | Target | Measured neural success | Wilson 95% interval | Mean damage | Median time |
|---|---:|---:|---:|---:|---:|
| 1 - Orientation | 95% | 490/500 (98.0%) | 96.36%-98.91% | 0.000 | 13.31 s |
| 2 - Surveillance | 95% | 491/500 (98.2%) | 96.61%-99.05% | 0.000 | 12.14 s |
| 3 - Patrol | 95% | 496/500 (99.2%) | 97.96%-99.69% | 0.236 | 25.61 s |
| 4 - Countermeasure | 95% | 485/500 (97.0%) | 95.11%-98.17% | 0.204 | 24.79 s |
| 5 - Lockdown | 95% | 479/500 (95.8%) | 93.66%-97.24% | 0.238 | 29.40 s |
| 6 - Ghostline | 85% | 474/500 (94.8%) | 92.49%-96.43% | 0.554 | 40.51 s |

The complete [JSON report](../benchmarks/neural/champion-final-7m-500.json),
[aggregate CSV](../benchmarks/neural/champion-final-7m-500.csv), and
[episode CSV](../benchmarks/neural/champion-final-7m-500.episodes.csv) expose
all failures and per-episode evidence. Tier 6 had 18 clock expiries and eight
integrity failures; those 26 failures are retained rather than filtered.

## Training lineage

Training ran on an RTX 5080 Laptop GPU (16 GB) with a 24-logical-core host.
The fresh base corpus contains 1,800 successful teacher episodes and 467,853
transitions. Four policy-induced DAgger recovery rounds add 1,200 episodes and
406,932 transitions, for 3,000 episodes and 874,785 transitions in the complete
lineage. GRU-256 and GRU-384 candidates each received the same 10,000-update BC
budget; GRU-384 won the closed-loop gate and then received four 3,000-update
DAgger rounds. Two disjoint 200-seed-per-tier confirmation gates passed before
the final slice was opened. Exact reports and lineage are indexed in
[`benchmarks/neural/README.md`](../benchmarks/neural/README.md).

## Intended use and limitations

This is a portfolio/research policy for procedural, partially observed game RL. Results are tied to the frozen v2 mechanics and observation contract. Deterministic success does not imply optimal trace, optional-data collection, or route efficiency, and no comparison with real players is valid until the planned matched-seed human cohort is complete.

## Deployment precision gate

The canonical FP32 ONNX graph is 5,831,535 bytes with SHA-256
`cd90659286310f75766fe8ab6dc1dd22f36cb40e0e72ca7c0b9ac9a00a694b14`.
It matched PyTorch deterministic actions on all 1,000 player-equivalent
recurrent transitions. Dynamic INT8 was 28.3% smaller but changed three actions
(first mismatch at transition 575), so it was rejected and the verified FP32
graph was selected for desktop and web deployment. The full audit is
[`champion-onnx-parity.json`](../benchmarks/neural/champion-onnx-parity.json).
