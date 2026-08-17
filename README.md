# Ghostline — multi-agent research track

Active research. **No validated result exists yet, and none is claimed here.**

This repository extends [Ghostline](https://github.com/aswanth-07/ghostline) —
a finished single-agent stealth game and RL benchmark — into an adversarial
multi-agent problem: a runner with field tools against a coordinated security
team that learns to contain it.

The finished project is the one to look at if you want measured results. This
one is a work in progress, kept separate so that stays true.

## What is different from the published game

| | Published game | This track |
|---|---|---|
| Runner actions | `Discrete(36)` | `Discrete(288)`, masked |
| Security | scripted | learned, up to 5 operatives |
| Runner tools | dash, pulse | plus crouch, decoy, vents, facility hacks |
| Contract | one | Standard / Ghost / Speed / Greed directives |
| Result | 89.6% on tier 6, 3,000 held-out episodes | **none yet** |

The runner sees an 8-value field record, explored vents and panels, and
directly visible sensors. Operatives act on local, player-equivalent
observations only — the centralized critic state is training-only and never
reaches a deployed actor.

## Approach

`ghostline co-train-v2` trains runner PPO and security MAPPO concurrently
against a **block-frozen opponent pool**. Within a generation both opponents are
immutable; only checkpoints selected on held-out seeds enter the next
generation's pool. A generation can never see the other side's current output.

That is a deliberate compromise. Naive simultaneous self-play cycles — each side
overfits the opponent's current policy, the gains evaporate when it moves, and
the metrics stay busy while nothing improves. Freezing per generation keeps
parallel wall-clock use without violating either learner's on-policy assumption.

Alternating frozen turns remain the reproducible control. If co-training does
not beat that at equal environment steps, that is a result worth having.

## Status

- Contracts, masks, reward ledgers and checkpoint fingerprints are implemented
  and covered by regression tests.
- 10,000 generated facilities pass connectivity and content readiness.
- Fresh and resume smokes pass for both learners, and the league driver has been
  run end to end across two generations.
- The reserved final-test slices in `benchmarks/runner-v2/` and
  `benchmarks/security/` are **`reserved_unopened`**. They stay that way until a
  real campaign finishes.

Measured on the target host: 1,137 runner decisions/s at 16 environments. The
environment is the throughput floor — stepping is Python over a 60 Hz
simulation, so collection scales with cores, and an accelerator only pays for
the update.

## Honest caveats

- No learned multi-agent policy has been accepted, and no superhuman or
  better-than-scripted claim is made.
- The retired pre-migration security checkpoint and its 4/0/8/16% tier 3–6
  result are historical negative evidence. Contract changes invalidated it.
- Reward coefficients, the layout distribution and the architecture changes are
  unvalidated by any campaign result.

## License

MIT, inherited from the parent project.
