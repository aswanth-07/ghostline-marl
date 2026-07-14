# Historical teacher benchmark

The observation-only `ObservationTeacherPolicy` passed the then-current
Ghostline curve on the previously untouched `6,000,000+` final-test slice. The
evaluation covers 500 deterministic episodes per tier and uses the same public
`GhostlineEnv-v2` observation and action mask available to the neural actor.
It predates the final route/security/patrol freeze and is retained as historical
evidence rather than represented as the current release gate.

| Slice | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Tier 5 | Tier 6 | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| 3M | 100.0% | 100.0% | 97.0% | 94.8% | 96.8% | 80.8% | Retired - failed |
| 4M | 100.0% | 100.0% | 96.6% | 95.2% | 97.2% | 83.8% | Retired - failed |
| 5M | 100.0% | 100.0% | 97.6% | 93.6% | 97.4% | 85.6% | Retired - failed |
| 6M | 100.0% | 100.0% | 95.8% | 96.2% | 97.2% | 88.8% | Passed prior curve; now historical |

The 3M, 4M, and 5M slices remain visible as negative evidence but became
immutable immediately after inspection. A failure may motivate a correction,
but that slice can never be reused to judge or select the correction; only a
later untouched slice can do so.
Tier-6 directional inertia was selected only on two disjoint validation slices.
The final Countermeasure curve uses three guards to remove a non-monotonic
integrity-loss spike while preserving its cameras, doors, terminals, and pulse
lesson. No code or controller setting changed while the 6M gate was open; the
subsequent portfolio mechanics pass deliberately created a new distribution.

- [`teacher-release-gate-6m-500.json`](teacher-release-gate-6m-500.json): complete metrics and Wilson intervals.
- [`teacher-release-gate-6m-500.csv`](teacher-release-gate-6m-500.csv): compact per-tier export.
- [`audit-history.json`](audit-history.json): retired-slice and calibration provenance.

This is historical teacher-quality evidence, not the initialization-corpus
release gate. The final current-fingerprint teacher subsequently passed two
disjoint 100-seed-per-tier validation gates at
`100/100/99/99/99/86%` and `100/100/99/99/99/89%`; see
[`teacher-fast-ops-validation-a-100.json`](teacher-fast-ops-validation-a-100.json)
and [`teacher-fast-ops-validation-b-100.json`](teacher-fast-ops-validation-b-100.json).
The neural champion later passed the locked 8M final test and 1,000-transition
ONNX parity gate. Human comparison remains pending, so no superhuman claim is
made.
