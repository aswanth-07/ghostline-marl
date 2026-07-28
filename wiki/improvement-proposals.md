---
title: Ghostline Readiness Audit
updated: 2026-07-28
status: resolved
---

# Readiness audit

This page records how the 2026-07-27 architecture audit was resolved. It is not
a work order. Current contracts live in [implementation.md](implementation.md)
and [training.md](training.md).

## Immutable boundary

The published-v1 mechanics files remain byte-identical:

- `config.py`
- `env.py`
- `generation.py`
- `policies.py`
- `simulation.py`
- `types.py`

They are bound to fingerprint
`521c449a8bd9a540977a918f5b094dd3aeff44cc579a55f75e22a74bab20e129`.
Historical artifacts call that contract `GhostlineEnv-v2`; the public wrapper
calls the exact released game `GhostlineEnv-v1`.

All new work is isolated in current v2 modules such as `config_v2.py`,
`types_v2.py`, `generation_v2.py`, `simulation_v2.py`, `env_v2.py`, and
`model_v2.py`, plus the security environment/model/trainer.

## Findings and resolutions

### Published runner races rather than plays stealth

Evidence: published median maximum trace reached 100 on several upper tiers.

Resolution: do not rewrite the published v1 result. V2 introduces real
exposure, first-detection, and quiet-data objective terms, plus crouch, audible
walking, dash trace cost, cover cooling, field hacks, and vents. The new
distribution requires a new campaign and may not borrow v1 acceptance.

### Security could see but not convert contact

Evidence: many detections, short episodes, and strictly slower pursuers.

Resolution: retain the intentional 95/97/99% chase-speed ceiling. Replace
Euclidean tail proximity with geodesic escape-route coverage, public cutoff
targets, PINCER, and graph-safe SEAL actions. Episode duration remains a useful
leading diagnostic, but held-out stop rate remains the selection objective.

### Identical team reward caused lazy-agent credit

Resolution: terminal and physical outcomes remain shared; bounded
discount-matched route-coverage/awareness shaping is attributed per operative.
The critic emits a separate value for every active agent. Presence masks remove
padded slots from GAE and all losses.

### Security observations disagreed with mechanics

Resolution:

- vision gates use the shared simulation constants and darkness scaling;
- target kinds have distinct one-hot codes, including explicit escape routes;
- heard contact is quantized/perception-gated rather than an invertible offset
  from true runner position;
- central state contains mission clock, alert, link progress, and an explicit
  operative presence mask;
- PINCER/SEAL use public target rows rather than hidden route state.

### Runner action contract was partially unreachable

Evidence: the environment exposed 288 actions but clipped values above 71,
making crouch/interact combinations unreachable through Gymnasium.

Resolution: all values 0-287 decode uniquely and pass through the wrapper.
Masks cover unavailable dash, pulse, decoy, crouch, and context-sensitive
interact combinations. API tests enumerate the full space.

### Detection and radio rewards were farmable

Resolution:

- direct sight cannot be demoted by HOLD;
- contact reward is issued once per operative per episode;
- radio assist is capped by possible teammates and first-information transfer;
- every named component is clipped and summed exactly once.

### Field observations leaked global state

Resolution: runner field records include only owned status, explored static
content, directly visible dynamic sensors/projectiles, and player-readable
route information. Hidden guards are never matched back to audio or memory by
nearest true state.

### Facility interactions overlapped or did nothing

Resolution:

- one tile-reservation contract covers structures, field content, and decor;
- reciprocal vents connect distinct rooms and meet a geodesic separation;
- decorative fake vent grates were removed;
- panels are reachable and effectful;
- each door panel binds to one exact generated security door;
- content-readiness validation runs after reshape and rejects a seed after
  eight deterministic failed attempts.

### Vent transit froze the wrong state

Resolution: only runner control is committed during transit. The clock,
cameras, guards, drones, trace, and termination continue to tick. The runner
cannot take damage, hack, extract, or trigger a field sensor while inside.

### Critic/optimizer contract was inconsistent

Resolution:

- agent-specific centralized value targets match per-agent rewards;
- inactive agents are excluded from advantages and losses;
- checkpoints store complete optimizer/curriculum/random state;
- source, observation, action, reward, and opponent fingerprints fail closed;
- rollout collection batches policy work where possible rather than
  serializing one actor call per operative.

## Readiness gate result

The implementation gate is complete:

- the Gymnasium and PettingZoo contracts, all 288 runner actions, factorized
  operative masks, deterministic replay, and stale-checkpoint rejection are
  covered by the full regression suite;
- 10,000 developmental-v2 facilities passed connectivity and content
  readiness with zero rejected final contracts;
- recurrent runner PPO completed a finite fresh run and exact resume;
- recurrent MAPPO completed two updates after resume against a frozen pool
  containing both published-v1 and native-v2 runners;
- both trainers emit immutable experiment manifests and restore optimizer,
  curriculum, environment, recurrent, and random state;
- the v2 runner export matched deterministic PyTorch actions on 1,000/1,000
  recurrent transitions;
- the isolated base wheel steps v1 and v2, renders a frame, and runs adaptive
  tactical security without Torch, ONNX Runtime, or PettingZoo;
- the human-only web archive is 24,165,062 bytes and the static JavaScript
  bridge tests pass.

The measured 22-worker Windows CPU result is 2,039 aggregate policy decisions
per second (12,234 simulation ticks per second). That is sufficient for smoke
and moderate campaigns but misses the aspirational 5,000-decision target.
Profiling attributes the remaining cost primarily to simulation ticks, guard
updates, movement, and route features. A long campaign should calibrate worker
count on the actual training host; this limitation must not be hidden behind a
synthetic scaling claim.

## Remaining campaign evidence

The new runner and security policies still require a long alternating
frozen-opponent campaign, disjoint validation, equal-budget ablations, and
held-out evaluation. Until that evidence exists:

- the published v1 runner result remains valid;
- the old adaptive-security checkpoint/result is historical only;
- no current v2 learned-security or superhuman claim is made;
- player/web builds use deterministic v2 tactical security when no compatible
  checkpoint exists.

## Deferred, evidence-dependent options

- multi-agent transformer or COMA after equal-budget recurrent MAPPO;
- simultaneous self-play after alternating frozen opponents establish a stable
  baseline;
- smaller GRU/quantized v2 web policy after parity and held-out success;
- vectorized observation planes if per-worker profiling shows Python grid
  assembly remains the dominant throughput cost.
