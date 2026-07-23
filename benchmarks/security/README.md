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

No learned-security result belongs here until the CUDA campaign has selected a
checkpoint using validation seeds. The 12M final slice must not be inspected
during tuning or reused after inspection.
