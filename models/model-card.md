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
- Learning: pure in-project PyTorch behavior cloning from the observation-only teacher, four DAgger recovery rounds, and low-rate consolidation. PPO/GAE/RND are implemented and tested, but the current release checkpoint does not claim a PPO improvement.

## Optional adaptive-security policy

Env-v3 also bundles a distinct 256-unit parameter-shared GRU policy for up to
five security operatives. Actors receive only local/perception-gated records;
the fixed 64-value global state is used by the centralized critic during
training only. Strategic observation-only imitation is followed by recurrent
MAPPO with adaptive weakest-tier replay and a 50% scripted-opponent curriculum.
The frozen checkpoint hash is
`c7d717d16b6a60c580e3d909043bf9dd107a6a1c6cf009dd77d3c0804308c839`.
On the untouched 13M final slice it stopped the Env-v2 neural runner on
`4/0/8/16%` of tier 3-6 contracts (25 per tier). This is an optional adversarial
research result; tier 4 remains unsolved, and lightweight builds fall back to
the same deterministic observation-only tactical controller when PyTorch is
not included.

## Data and fairness

No human demonstrations, human trajectories, hidden generator state, or privileged critic state are used. The automated teacher receives the same public v2 observation and action mask as the neural policy. Live facility telemetry is deliberately shared with both the human HUD and policy; physical detection remains occlusion-based. Training seeds are below 1,000,000, validation uses 1,000,000-1,049,999, and final evaluation begins at 2,000,000. All attempted final-test slices are immutable: 2M-7M are historical evidence, and the tracked 8M slice was consumed exactly once by the selected current-fingerprint neural champion. Its report and artifact hashes are bound in [`benchmarks/final-test-slices.json`](../benchmarks/final-test-slices.json).

The failed 2.98-million-decision pure-PPO attempt is retained as historical negative evidence only; it predates the final environment fingerprint and is not represented as an equal-budget current-distribution ablation. A current-fingerprint conservative PPO pilot was also rejected: it scored `84/94/98/92/90/86%`, while the immutable DAgger rollback scored `96/96/90/98/92/94%` on the same seeds. The release therefore uses the better BC+DAgger checkpoint and does not mislabel it as PPO-trained.

## Current fair-teacher qualification

The final observation-only teacher passed two disjoint current-fingerprint validation gates covering 100 seeds per tier. Tier 1-6 success was `100/100/99/99/99/86%` and `100/100/99/99/99/89%`. The machine-readable reports are [`teacher-fast-ops-validation-a-100.json`](../benchmarks/teacher/teacher-fast-ops-validation-a-100.json) and [`teacher-fast-ops-validation-b-100.json`](../benchmarks/teacher/teacher-fast-ops-validation-b-100.json). These gates qualify trajectory collection; they are not neural or final-test results.

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
`76baa30af55cdaa2e71bb6ba06672bd9203455552358017505685827240b2e47`
and environment fingerprint
`521c449a8bd9a540977a918f5b094dd3aeff44cc579a55f75e22a74bab20e129`.
The one-time 8M audit ran 500 deterministic, unseen seeds per tier:

| Tier | Target | Measured neural success | Wilson 95% interval | Mean damage | Median time |
|---|---:|---:|---:|---:|---:|
| 1 - Orientation | 95% | 499/500 (99.8%) | 98.88%-99.96% | 0.000 | 12.98 s |
| 2 - Surveillance | 95% | 500/500 (100.0%) | 99.24%-100.00% | 0.000 | 12.73 s |
| 3 - Patrol | 95% | 482/500 (96.4%) | 94.38%-97.71% | 0.746 | 21.90 s |
| 4 - Countermeasure | 95% | 490/500 (98.0%) | 96.36%-98.91% | 0.622 | 23.17 s |
| 5 - Lockdown | 95% | 495/500 (99.0%) | 97.68%-99.57% | 0.560 | 27.86 s |
| 6 - Ghostline | 85% | 448/500 (89.6%) | 86.62%-91.98% | 1.222 | 31.07 s |

The complete [JSON report](../benchmarks/neural/champion-final-8m-500.json),
[aggregate CSV](../benchmarks/neural/champion-final-8m-500.csv), and
[episode CSV](../benchmarks/neural/champion-final-8m-500.episodes.csv) expose
all failures and per-episode evidence. Tier 6 had two clock expiries and 50
integrity failures; those 52 failures are retained rather than filtered.

## Training lineage

Training ran on an RTX 5080 Laptop GPU (16 GB) with a 24-logical-core host.
The current base corpus contains 1,800 successful teacher episodes and 412,483
transitions. Four policy-induced DAgger recovery rounds add 2,100 episodes and
529,401 transitions, for 3,900 episodes and 941,884 transitions in the complete
current-fingerprint lineage. A historical GRU checkpoint was used only as a
declared weight initialization; all selection evidence, labels, recovery data,
and optimizer state are current-fingerprint. The 384-unit model received a
5,000-update initial clone, 6,000 DAgger updates, and a 2,000-update low-rate
consolidation pass. Two disjoint 100-seed-per-tier confirmation gates passed
before the final slice was opened. Exact reports and lineage are indexed in
[`benchmarks/neural/README.md`](../benchmarks/neural/README.md).

## Intended use and limitations

This is a portfolio/research policy for procedural, partially observed game RL. Results are tied to the frozen v2 mechanics and observation contract. Deterministic success does not imply optimal trace, optional-data collection, or route efficiency, and no comparison with real players is valid until the planned matched-seed human cohort is complete.

## Deployment precision gate

The canonical FP32 ONNX graph is 5,831,535 bytes with SHA-256
`d27ff0aed1f9578b21fd490aa9071abc417c8b1a3e662832e988a96bbc7c031a`.
It matched PyTorch deterministic actions on all 1,000 player-equivalent
recurrent transitions. Dynamic INT8 was 28.3% smaller but changed five actions
(first mismatch at transition 279), so it was rejected and the verified FP32
graph was selected for desktop and web deployment. The full audit is
[`champion-onnx-parity.json`](../benchmarks/neural/champion-onnx-parity.json).
