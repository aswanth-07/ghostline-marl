---
title: Ghostline RL Architecture Proposals
updated: 2026-07-28
status: proposals
---

# RL architecture and coordination proposals

Written by `claude` on 2026-07-28 for `codex` to take as far as it judges
worthwhile. **These are suggestions, not a work order.** Nothing here is a
requirement or an acceptance criterion. This page is a project record like any
other wiki page; `AGENTS.md` and the user remain the only instruction sources.
Disagree, reorder, drop, or replace any of it — just record the call in
`wiki/log.md` per the usual turn rules.

Terminology, because it has caused confusion: the user's **"project v1"** is the
code's `GhostlineEnv-v2` single-agent track (frozen champion, immutable 8M
audit, portfolio demo). Their **"project v2"** is the code's `GhostlineEnv-v3`
plus `GhostlineSecurityParallel-v0`. Everything below targets the latter.
`GhostlineEnv-v2` should stay byte-identical.

## Where things actually stand

Implemented and tested, but **untrained**: nothing below has a campaign behind
it. The reward budgets, the layout distribution, and the architecture choices
are all principled and unvalidated.

- Runner: crouch/cover stealth, vent network, environmental hacking, upgraded
  decoy lures. Env-v3 is 288 masked actions.
- Ops: predictive chokepoint seals, `PINCER` and `SEAL` intents, non-lethal
  field sensors on the shared ability bit.
- Networks: `model_v3.RunnerPolicyV3` (new; there was no v3 runner network at
  all) and `SharedSecurityActorCritic`, both orthogonally initialised, plus
  running value-target normalisation in `marl_train`.

## 1. Coordinated pincers — the part worth the most thought

`SecurityIntent.PINCER` currently assigns each operative a complementary arc
around the contact, spread deterministically by `guard_id`. That is a
*mechanism*, not coordination: the spread is fixed regardless of geometry, and
nothing makes an operative prefer the arc that actually closes an escape.

The underlying difficulty is worth stating plainly, because it shapes every
option below. Guards move at 95–99% of runner speed. **A tail chase can never
close.** The only way the team ever connects is if someone is already standing
where the runner is going. So coordination is fundamentally an *assignment*
problem over escape routes, not a pursuit problem.

Some directions, roughly in increasing ambition:

**a. Make the arc assignment geometric rather than positional.** Instead of
spreading by `guard_id`, compute the runner's escape bearings (which doorways
lead onward from its current room) and assign operatives to bearings by travel
time. This is a Hungarian-style assignment over a handful of options and is
cheap. It would likely be the single biggest behavioural improvement per line of
code, and it needs no learning at all — which also makes it a strong tactical
baseline to measure a learned policy against.

**b. Let the policy choose the assignment, and give it the vocabulary.** The
observation currently exposes eight tactical targets per operative. Adding
"cut-off point for escape route *k*" as explicit target kinds would let the
learned policy express a pincer directly rather than having to rediscover it as
a sequence of flank offsets. Target kinds are cheap; the one-hot already has
room.

**c. Autoregressive action heads.** The factorised action is currently four
independent heads. Target choice genuinely depends on intent — "flank left"
only makes sense with a flanking intent — so conditioning the target head on the
sampled intent embedding is well-motivated. It is a real refactor of
`_heads`/`factorized_log_prob` and worth doing only if independence is measurably
hurting; the action histograms already logged per factor would show that.

**d. An explicit team-level coordinator.** A single small policy that emits a
role assignment each tactical tick, with the shared actor conditioning on its
assigned role. This is the most faithful model of what a security team does and
the largest change; it also partly re-centralises what CTDE deliberately
decentralised, so it needs care to keep the actor honest.

I would try (a) first, measure, then (b). (c) and (d) are worth having on the
list but are not where the first win is.

## 2. Credit assignment beyond the current split

Rewards are currently shared for team outcomes and per-agent for containment
shaping, with per-operative GAE against the shared team value. That fixed the
lazy-agent problem in principle. Two follow-ups if it proves insufficient:

- **Counterfactual baselines (COMA).** Marginalise each operative's action
  against the centralised critic. Expensive with a factorised action space, so
  worth it only if per-agent shaping turns out to be too coarse.
- **Value decomposition.** Keep a global reward and learn a monotonic mixing
  network. Cleaner theory than hand-split shaping, but it gives up the explicit
  containment signal that currently makes the shaping interpretable.

A cheap diagnostic first: log per-operative action histograms and check whether
the five agents have actually differentiated. If they are still statistically
identical after a real campaign, credit assignment is still the bottleneck.

## 3. Reward system

The stealth economy is new and unvalidated. Worth re-checking after a campaign:

- Exposure and detection budgets were set from an explicit table, targeting
  8–11% of the positive budget for a fully hot mission. If runs come back with
  `median_max_trace` still pinned at 100, the coefficients are too soft.
- Episode length is the leading indicator on the security side. If containment
  is being learned, **runs get longer before stop rate moves.** Watch duration
  first; stop rate is a lagging and very noisy signal at 25 episodes per tier.
- The survival term becomes significant only if episodes start reaching the full
  window. At current ~30 s episodes it contributes about 1.2; at a full tier-6
  window it would approach 11 and start competing with the ±20 terminal.
- Now that the runner has vents, hacking and crouch, the Env-v3 reward has no
  term for *using* them well. That is deliberate for now — better to see what
  the agent does with free tools than to prescribe usage — but if they go
  unused entirely, a small per-first-use bonus is the least distorting nudge.

## 4. Runner architecture

`RunnerPolicyV3` is deliberately conservative: per-source encoders, attention
pooling over entities and targets, FiLM directive conditioning, a GRU core, and
separate actor/critic decoders. Things I considered and did not do:

- **Wider recurrent core.** 384 already exports to a 5.83 MB ONNX graph on the
  v2 side, which dominates web transfer. Worth revisiting only if the v3 track
  never ships to the browser.
- **A transformer core.** Probably unnecessary at this observation size, and it
  would complicate the ONNX export path that the release depends on.
- **Frame stacking.** Likely redundant with the GRU; cheap to ablate if
  temporal credit looks like the problem.

The one gap worth flagging: there is still **no training entry point for the
Env-v3 runner**. `torchrl_train.py` is Env-v2 only. A first campaign needs that
wired up, and it is the obvious next piece of work regardless of anything else
on this page.

## 5. What I would sequence first

1. Wire an Env-v3 runner training entry point. Nothing here is measurable
   without it.
2. Geometric pincer assignment (1a) as a tactical baseline.
3. A first Env-v3 campaign, watching episode duration and per-operative action
   histograms rather than stop rate.
4. Revisit reward coefficients against what that campaign actually shows.

Architecture changes beyond that are speculative until there are results to
point at.
