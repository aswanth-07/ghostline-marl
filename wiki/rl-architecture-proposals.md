---
title: RL Architecture Decision Record
updated: 2026-07-28
status: superseded
---

# RL architecture decision record

This page preserves the outcome of the earlier proposal pass. The detailed
current contracts are in [implementation.md](implementation.md) and the
training protocol is in [training.md](training.md).

## Version resolution

The earlier working branch temporarily used a third-version label for its
experimental files. That label is retired. Public versioning is:

- v1: published single-agent game, champion policy, and immutable evidence;
- v2: new facility distribution, field-systems runner, and multi-agent
  adversarial security;
- no public v3.

The published artifacts retain their historical internal `GhostlineEnv-v2`
metadata because provenance-bound bytes are not rewritten.

## Decisions adopted

### Security learner

Use parameter-shared recurrent MAPPO before considering more complex
multi-agent algorithms.

Reasons:

- the team is cooperative and homogeneous enough to share actor weights;
- recurrent local actors match partial observation and deployment;
- a centralized critic improves training without privileged actor inputs;
- factorized semantic actions and exact masks fit PPO directly;
- the implementation and evidence burden is lower than a transformer joint
  policy, COMA counterfactual critic, or value-decomposition stack.

The critic is agent-specific rather than one scalar team critic because rewards
combine shared outcomes with bounded attributed shaping. A shared operative
encoder, presence-masked pooling, and per-operative heads preserve permutation
structure while producing one value per active agent.

### Coordination

Use explicit public tactical targets for PINCER and SEAL:

- escape-route cutoff records are derived from known facility geometry and
  perceived contact;
- PINCER assigns complementary public cutoffs;
- SEAL chooses a graph-safe door or cutoff;
- neither action reads the runner's hidden objective, heading, or exact unseen
  state.

This replaces oracle route projection. A learned joint-attention coordinator
can be an ablation later, but is not required for the first credible campaign.

### Credit assignment

Use a bounded shared outcome plus small, discount-matched per-agent potential:

- terminal containment/extraction, damage, runner data, radio, invalid action,
  and formation remain team terms;
- geodesic route coverage and awareness are attributed per operative;
- contact acquisition is credited once per operative per episode;
- active masks remove nonexistent agents from all objectives.

This is simpler and easier to audit than COMA and directly addresses the
lazy-agent failure of identical scalar rewards.

### Runner learner

Use a recurrent structured policy with:

- local-grid convolution;
- masked terminal/entity/field-target attention;
- ego/objective/ray/field MLPs;
- directive FiLM;
- GRU memory;
- separate actor/value decoders;
- objective and danger auxiliary targets;
- exact 288-action masking.

The first training baseline is recurrent PPO/GAE with frozen-opponent
curriculum. Behavior cloning and DAgger remain valid additions only when the
teacher uses the exact v2 observation/action contract.

### Opponent schedule

Use alternating frozen opponents rather than simultaneous unconstrained
self-play:

1. train one side against scripted and frozen snapshots;
2. select it on held-out seeds;
3. freeze it while training the other side;
4. retain earlier snapshots in the opponent mixture;
5. repeat only after regression gates pass.

This gives reproducible opponent identities and reduces cycling. Simultaneous
self-play remains a later ablation.

## Approaches rejected for the first campaign

- **Oracle pincer/seal targets:** strong but invalid under player-equivalent
  information.
- **One shared team value for per-agent rewards:** mismatched targets and poor
  credit assignment.
- **COMA first:** adds a counterfactual critic and joint-action complexity
  before the simpler credit fix is measured.
- **VDN/QMIX first:** designed around value factorization and discrete
  joint-action learning; less direct for recurrent factorized PPO actors.
- **Multi-agent transformer first:** promising but higher compute,
  implementation, and ablation burden. Revisit after MAPPO establishes a
  trustworthy baseline.
- **Unbounded detection/radio bonuses:** farmable and weakly tied to outcomes.
- **Simultaneous continuously changing opponents:** hard to reproduce and prone
  to non-stationary cycling.

## Promotion condition

Alternative architectures are considered only after the current v2 baseline
passes correctness, 10,000-seed generation, optimizer smoke, throughput,
resume, and held-out evaluation gates. A more complex model must improve
worst-tier held-out results at an equal environment-step budget and report its
inference cost.
