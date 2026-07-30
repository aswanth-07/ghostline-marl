---
title: RL Architecture Decision Record
updated: 2026-07-30
status: current
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

## 2026-07-29 runner recovery decision

DAgger is an online imitation-learning procedure, not a replacement neural
backbone. The existing v2 structured recurrent policy remains appropriate:
its critic reached roughly `0.90-0.98` explained variance through the stalled
Ghost tier while deterministic success stayed near zero. The network could
model the returns of its behavior; the missing ingredient was successful
stealth behavior to reinforce.

The first 5.66-million-decision Ghost specialist is therefore not resumed
unchanged. It learned tiers 1-2, reached the tier-3 objective reliably, and
usually extracted after exhausting the trace budget. On 50 unused training
seeds, 46 runs extracted without satisfying Ghost and none succeeded. This is a
hard-exploration and constraint-discovery failure, not evidence that a wider
GRU will help.

DAgger is deferred until a v2 supervisor passes a held-out qualification gate.
The published-v1 observation teacher emits only the old 36 action semantics.
Observation-only v2 wrappers reached `0-15%` tier-3 Ghost success over a
20-training-seed diagnostic, which is not credible supervision. Aggregating
those labels would teach the same trace-saturated behavior.

The accepted recovery path is:

1. keep the 384-unit recurrent actor-critic and exact 288-action mask;
2. pretrain Ghost on declared, training-only security rosters between
   camera-only tier 2 and full tier 3;
3. select each stage on disjoint validation windows clearly marked as
   training-only, never as release evidence;
4. return to the unmodified full distribution for PPO and acceptance;
5. use positive-advantage self-imitation once the staged policy supplies enough
   complete successful recovery trajectories;
6. add BC/DAgger only after a fair v2 teacher is independently qualified.

The Stage-1 run supplied enough successful trajectories to activate step five.
Its PPO critic remained healthy while held-out success plateaued near 31%, and
the final policy discarded crouching while continuing to extract loudly. The
v2.4 auxiliary objective therefore operates on recurrent on-policy rollouts:
it retains only complete successful episodes and applies supervised policy
loss only where the unnormalized GAE advantage is positive. It does not replay
failed actions, incomplete rollout prefixes, privileged state, or external
demonstrations. A coefficient calibration rejected `0.02` after no matched
improvement and selected `0.2` after two diagnostic windows improved from
`28/34%` to `34/36%` without KL or entropy instability.

The first deterministic probe supports this direction. The stalled checkpoint
scored `35%` on one guard, `35%` on one guard plus one camera, `10%` on two
guards plus one camera, and `0%` on the full tier-3 roster over the same 20
unused training seeds. That creates a learnable bridge without privileged
state, weakened release evaluation, or a new public environment.

## 2026-07-30 recoverable Ghost contract

The maximum-trace rule is retired for v2. It contradicted the game's core
promise that maximum trace creates lockdown pressure rather than immediate,
irreversible failure. Empirical calibration also showed a pathological cliff:
maximum-trace limits between `75` and `95` barely changed outcome rate because
most loud policies saturated exactly at `100`, while accepting `100` removed
nearly all stealth selectivity.

Ghost now requires zero damage and live trace below `95%` at extraction. This
preserves the requirement to disengage and cool the network, permits recovery
after a mistake, and is readable from the existing HUD. The trace portion of
the discount-matched potential follows the same live budget. Exposure,
detection, dash, awareness, and damage remain genuine costs, so the relaxed
terminal rule does not reward a permanently loud route.

This environment change invalidates every earlier v2 runner and security
checkpoint. The new campaign starts from the immutable published-v1 overlap
transplant rather than bypassing fingerprint checks or relabeling stale v2
weights.

This decision follows DAgger's requirement for an expert queried on the
learner's state distribution, the demonstration result for hard exploration
under partial observability, and constrained-RL's separation of task reward
from behavioral constraints:

- https://proceedings.mlr.press/v15/ross11a/ross11a.pdf
- https://arxiv.org/abs/1909.01387
- https://proceedings.mlr.press/v80/oh18b.html
- https://arxiv.org/abs/1705.10528
- https://arxiv.org/abs/1805.11074

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
