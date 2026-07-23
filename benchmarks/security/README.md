# Adaptive-security evidence

This directory is reserved for `GhostlineSecurityParallel-v0` evaluation
artifacts. Adaptive security is an optional Env-v3 research track and does not
replace or revise the frozen Env-v2 runner result.

The security team is evaluated against the published Env-v2 recurrent runner
checkpoint (`models/ghostline-policy.pt`, SHA-256
`76baa30af55cdaa2e71bb6ba06672bd9203455552358017505685827240b2e47`).
The runner retains its original 36 actions and receives only its original
player-equivalent observation fields. Temporary v3 locks enter its normal
occupancy grid; it is not given the v3 decoy action.

Seed namespaces are disjoint:

- training starts at `10,000,000`;
- checkpoint validation starts at `11,000,000`;
- the one-time final test starts at `12,000,000`.

`ghostline evaluate-security` writes three sibling artifacts: the complete
JSON report, an aggregate per-tier CSV, and a per-episode CSV. Reports include
95% Wilson intervals and bind the opponent hash, observation contract, and
security-environment fingerprint. A missing `--model` is explicitly labeled
as the deterministic observation-only tactical baseline.

## Selected learned-security result

The selected mixed-opponent recurrent MAPPO checkpoint is bundled as
`models/ghostline-security.pt` (SHA-256
`c7d717d16b6a60c580e3d909043bf9dd107a6a1c6cf009dd77d3c0804308c839`).
It uses security fingerprint
`96275bac09bd6fb321510e1bd23d0e025d157b4cdeeb919aded9bb38b850721b`
and was selected only against the frozen Env-v2 runner above.

Two disjoint validation gates measured `4/0/4/12%` over 25 contracts per tier
and `0/0/10/10%` over 10 contracts per tier. The first 12M final candidate was
honestly retired after scoring zero stops. After the opponent-curriculum model
was selected from both validation windows, the untouched 13M slice was opened once:

| Tier | Stops | Wilson 95% interval | Mean damage | Mean detections |
|---|---:|---:|---:|---:|
| 3 | 1/25 (4%) | 0.7%-19.5% | 0.92 | 16.88 |
| 4 | 0/25 (0%) | 0.0%-13.3% | 0.44 | 20.96 |
| 5 | 2/25 (8%) | 2.2%-25.0% | 1.16 | 46.96 |
| 6 | 4/25 (16%) | 6.4%-34.7% | 1.60 | 60.32 |

This is a measured 7% mean containment rate against an already strong runner,
not a claim that every tier is solved. Tier 4 is the clearest limitation. The
canonical report is [`adaptive-security-final-13m-25.json`](adaptive-security-final-13m-25.json),
with aggregate and episode CSV siblings. The failed 12M report remains tracked
as negative evidence and is never reused.

Training uses an observation-only tactical behavior warm-up followed by
recurrent MAPPO. Warm-up accuracy and entropy are written to
`behavior-warmup.json`; rollout entropy, throughput, reward, and episode outcome
are written to `training-metrics.jsonl`. Security `info` contains an exact
reward-component ledger, and changing the reward implementation changes the
environment fingerprint so stale checkpoints cannot resume.
After a held-out gate, the default curriculum assigns 70% of new contracts to
the weakest tier set and preserves 30% uniform replay. The checkpoint and each
rollout record include the resulting probability vector; `--uniform-curriculum`
is reserved for an equal-budget ablation.
Repeated radio messages cannot farm shaping: positive radio credit is exhausted
after the first possible teammate broadcasts in an episode, and this bound has
a dedicated regression test.
Each `validation-<steps>.json` has a corresponding immutable
`policy-<steps>.pt`; later training can update `latest.pt` and `champion.pt` but
cannot erase the policy that produced an earlier report.
Opponent curricula may mix the easier scripted runner into training, but every
validation and final report remains exclusively against the provenance-bound
Env-v2 neural checkpoint. The mix fraction is stored in training arguments and
is never inferred from validation outcomes.
