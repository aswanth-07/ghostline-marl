---
title: Ghostline Wiki
updated: 2026-07-28
status: active
---

# Ghostline

Ghostline is a procedural stealth-infiltration game and adversarial RL
benchmark with a deterministic 60 Hz simulation, validated furnished
facilities, a scrolling pixel-art presentation, six contract tiers, and
player-equivalent recurrent policies.

Public versioning is intentionally simple: v1 is the published single-agent
game and champion; v2 is the current multi-agent/new-map development contract.
There is no public v3. Published evidence retains a historical internal
`GhostlineEnv-v2` label because its exact bytes and source fingerprint are
immutable.

## Frozen product decisions

- Quota-based data theft followed by extraction.
- Three integrity points, tier-scaled mission clock, escalating recoverable trace, dash, and disruption pulse.
- Cameras, human guards, and late-tier response drones; no player weapon combat.
- Keyboard play plus Agent Lab and a static Chrome-first web showcase; no gamepad or multiplayer scope.
- Player-equivalent structured observations; no renderer-only or hidden live
  enemy state.
- Published-v1 recurrent policy trained through fair-teacher imitation, DAgger
  recovery data, and PPO fine-tuning.
- Developmental-v2 parameter-shared recurrent MAPPO security team with
  decentralized actors and an agent-specific centralized critic.

## Pages

- `implementation.md`: simulation, generator, presentation, and public contracts.
- `training.md`: policy, curriculum, rewards, seed namespaces, and evaluation.
- `setup.md`: install, play, verify, train, record, and package.
- `assets.md`: visual/audio workflow and disclosure.
- `web-deployment.md`: static Pygbag/ONNX Runtime Web architecture, build, Chrome QA, and Vercel release.
- `rl-architecture-proposals.md`: decision record for MAPPO, public-target
  coordination, credit assignment, and rejected first-campaign alternatives.
- `improvement-proposals.md`: resolved readiness audit and remaining empirical
  gates.
- `log.md`: newest-first project memory and verified results.
