# Adaptive-security evidence

This directory preserves immutable evidence from the retired
`GhostlineSecurityParallel-v0` research contract. Those reports were produced
before the public version migration, when the shipped single-agent game was
called internal `GhostlineEnv-v2` and the experimental multi-agent track was
called Env-v3.

The public names are now unambiguous:

- `GhostlineEnv-v1` is the published single-agent game and frozen runner.
- `GhostlineEnv-v2` is the current developmental runner, map, and multi-agent
  contract.
- `GhostlineSecurityParallel-v2` is the current PettingZoo security contract.
- There is no public Env-v3.

The JSON and CSV files already tracked here are historical records. Their
contract strings, hashes, fingerprints, outcomes, and seed ranges must not be
edited or relabeled. The current verifier accepts them only as internally
consistent historical evidence; it never promotes them to v2 release evidence.

## Historical pre-migration result

The retired experiment evaluated security against the immutable published
runner checkpoint `models/ghostline-policy.pt`, SHA-256
`76baa30af55cdaa2e71bb6ba06672bd9203455552358017505685827240b2e47`.
That runner used the same 36-action, player-equivalent contract now exposed as
public v1; its immutable metadata retains the historical internal
`GhostlineEnv-v2` name.

Seed namespaces were disjoint:

- training started at `10,000,000`;
- checkpoint validation started at `11,000,000`;
- one-time final tests started at `12,000,000`.

The selected historical checkpoint was `models/ghostline-security.pt`,
SHA-256
`c7d717d16b6a60c580e3d909043bf9dd107a6a1c6cf009dd77d3c0804308c839`,
with security fingerprint
`96275bac09bd6fb321510e1bd23d0e025d157b4cdeeb919aded9bb38b850721b`.
Two validation windows measured `4/0/4/12%` over 25 contracts per tier and
`0/0/10/10%` over 10 contracts per tier. The first 12M candidate report was
retained after scoring zero stops. After opponent-curriculum selection, the
untouched 13M slice was opened once:

| Tier | Stops | Wilson 95% interval | Mean damage | Mean detections |
|---|---:|---:|---:|---:|
| 3 | 1/25 (4%) | 0.7%-19.5% | 0.92 | 16.88 |
| 4 | 0/25 (0%) | 0.0%-13.3% | 0.44 | 20.96 |
| 5 | 2/25 (8%) | 2.2%-25.0% | 1.16 | 46.96 |
| 6 | 4/25 (16%) | 6.4%-34.7% | 1.60 | 60.32 |

This was a measured 7% mean containment rate, not a solved benchmark. The
canonical historical report is
[`adaptive-security-final-13m-25.json`](adaptive-security-final-13m-25.json),
with aggregate and episode CSV siblings. The failed 12M report remains
negative evidence and its seeds are never reused.

## Why the old checkpoint is invalid for v2

The current `GhostlineSecurityParallel-v2` contract changes the learning
problem materially:

- ten semantic intents and ten tactical target slots use an explicit
  intent-by-target legality mask;
- actor observations include the revised public target, field-target, radio,
  teammate, and readiness records;
- the centralized critic receives a 72-value state with agent-specific
  operative blocks and an explicit presence mask;
- generation includes the developmental v2 maps, route guarantees, doors,
  vents, field tools, and runner mechanics;
- per-agent formation and progress shaping is capped and exactly accounted;
- the model contract and environment fingerprint bind all of the above.

Loading the retired checkpoint into this contract would produce neither a
valid resume nor a fair comparison. Runtime policy loading therefore fails
closed and the game uses its deterministic observation-only tactical
controller until a compatible v2 checkpoint passes the release gates.

## Current v2 protocol

Training uses a parameter-shared recurrent actor with an agent-specific
centralized critic, active-agent masks, generalized advantage estimation, and
recurrent MAPPO. Behavior warm-up may initialize the actor, after which the
curriculum can mix the frozen published-v1 runner, native v2 runner snapshots,
and scripted opponents. Validation and final reports must identify one exact
opponent, runner hash, security fingerprint, model contract, seed slice, and
curriculum configuration.

The seed namespaces remain:

- training: `10,000,000+`;
- validation: `11,000,000+`;
- current-v2 final: the reserved `14,000,000` slice.

Each validation report must have a corresponding immutable checkpoint.
Selection uses held-out results only; final slices are opened once. Reports
must include per-tier Wilson intervals, failure and reward-component
accounting, policy entropy, throughput, opponent provenance, and the exact
joint action-mask contract.

The tracked [`v2-final-test-slices.json`](v2-final-test-slices.json) ledger is
still `reserved_unopened`. `evaluate-security` validates the model and frozen
runner first, locks the ledger before the first episode, and then marks the
slice consumed or aborted-retired. It has no overwrite or reopen option. The
historical 12M/13M files above remain immutable under their retired fingerprint.

`scripts/verify_security_release_evidence.py` is intentionally fail-closed but
version-aware. It checks the pre-migration files against their own retired
contract and reports `historical: true`; a green result means the archive is
unaltered, not that it qualifies for v2. A separate current-v2 candidate needs
a newly trained `GhostlineSecurityParallel-v2` checkpoint and canonical
validation/final reports. The historical files themselves remain untouched.

No current v2 learned-security result is claimed in this repository yet.
