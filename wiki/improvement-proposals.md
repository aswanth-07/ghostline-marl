---
title: Ghostline Improvement Proposals
updated: 2026-07-27
status: proposals
---

# Improvement proposals for the next agent/training pass

Raised by `claude` on 2026-07-27 for `codex` to take as far as it judges
worthwhile. **These are suggestions, not a work order.** Nothing here is a
requirement, an acceptance criterion, or an instruction: this page is a project
record like every other wiki page, and `AGENTS.md` plus the user's own request
remain the only instruction sources. Disagree with any of it, reorder it, drop
it, or replace it with something better — just record the call in `wiki/log.md`
per the usual turn rules.

Each item states the evidence it came from, why it matters, a suggested
direction, and how you would know it worked. The evidence is quoted so you can
re-derive it rather than trust this page.

Two hard constraints worth restating, because most of these touch them:

- `config.py`, `env.py`, `generation.py`, `policies.py`, `simulation.py`, and
  `types.py` are hashed into `ENVIRONMENT_FINGERPRINT_FILES`. Touching any of
  them invalidates every Env-v2 checkpoint and the frozen 8M evidence.
- `config_v3.py`, `types_v3.py`, `simulation_v3.py`, `security_baselines.py`,
  and `security_env.py` are hashed into the security contract identity. Most of
  section 2 lands inside that set, so it is a deliberate re-baseline with a new
  validation campaign, not a patch.

---

## 1. Runner policy (Env-v2)

Current champion: 384-unit GRU, `76baa30a...b2e47`, final 8M audit
`99.8 / 100.0 / 96.4 / 98.0 / 99.0 / 89.6%` over 500 seeds per tier.

### 1.1 The success curve is non-monotonic, and tier 6 is not the only dip

From `benchmarks/neural/champion-final-8m-500.json`:

| tier | success | mean guard damage |
|------|---------|-------------------|
| 3    | 96.4%   | 0.746             |
| 4    | 98.0%   | 0.622             |
| 5    | 99.0%   | 0.560             |
| 6    | 89.6%   | 1.222             |

Tier 3 is the second-worst tier despite being the easiest of the four, and it
takes more damage than tiers 4 and 5. That is a curriculum artefact rather than
a difficulty artefact — tier 3 is where roaming patrols first appear, and the
policy appears under-trained on the first-contact case specifically.

*Suggested direction:* look at the curriculum sampler's tier weights during the
PPO phase before touching the network. A per-tier replay buffer weighted by
recent failure rate is cheaper than any architecture change and would show up
quickly. *Signal:* tier 3 damage falling toward the tier-4/5 band without tier 6
regressing.

### 1.2 The policy is not playing stealth, it is racing

`median_max_trace` is exactly `100.0` on tiers 3, 4, and 6, and `97.7` on tier
5. The median run saturates the trace meter. The policy has learned that trace
has no terminal cost it cannot outrun, so it takes the loud route every time.

This is arguably fine for a success-rate benchmark and clearly wrong for a
stealth game — and it is why tier 6 collapses, because tier 6 is the tier where
saturated trace actually deploys a drone (threshold 72).

*Suggested direction:* this is a reward-shaping question, not a network one. A
modest trace-integral penalty, or making the terminal reward scale with
remaining integrity, would change the behaviour class. Worth deciding
deliberately with the user first, because "should the benchmark policy play
stealthily or optimally?" is a product question and the answer changes what the
portfolio demo shows. *Signal:* median max trace dropping below the 75
escalation step while success holds.

### 1.3 Architecture observations (low confidence, low priority)

`UniversalGhostlinePolicy` is a sound design: per-source encoders, two
attention-pooled masked set encoders, a 384-wide fusion into a GRU, separate
policy/value decoders, plus `objective_head` and `danger_head` auxiliaries.

Nothing here looks like the bottleneck. If you want to spend capacity anyway,
the ordering I would suggest is: (a) the local grid is 8 channels at 15x15 down
to 4x4 through two strided convs — that is a lot of spatial compression for a
15x15 input, and a dilated or full-resolution variant is cheap to try;
(b) frame-stacking or an explicit velocity channel may be redundant with the
GRU and worth ablating out for throughput. I would not widen the GRU: 384 units
already exports to a 5.83 MB ONNX graph that dominates the web payload.

---

## 2. Security MARL (`GhostlineSecurityParallel-v0`)

Current result over the untouched 13M slice, 25 episodes per tier:
`4 / 0 / 8 / 16%` stop rate for tiers 3-6.

### 2.1 Read the failure before changing the model

Mean episode duration is **26.7-36.9 s** against mission windows of 160-225 s,
so the runner nearly always extracts long before the timer. That means a "stop"
is in practice always a damage kill, never a successful containment.

Meanwhile the team registers **16.9 to 60.3 mean detections per episode**. The
operatives see the runner constantly. Perception is not the problem.

The problem is that they cannot convert sight into contact:

```
PLAYER_SPEED             126.0 px/s
GUARD_CHASE_SPEED_RATIOS (0.95, 0.97, 0.99)  ->  119.7 / 122.2 / 124.7 px/s
```

**A guard is strictly slower than the runner, so pursuit can never close.** The
only paths to damage are interception, cornering, or catching the runner
stalled at a terminal. This is a deliberate game-design decision (the wiki is
explicit that dash should remain the decisive escape tool), so the fix belongs
in what the policy is taught to do, not in the speed table.

And the shaping actively teaches the losing behaviour. `_security_potential`
rewards `proximity = 1 - nearest_guard_distance / diagonal`, i.e. being *near*
the runner — which for a slower pursuer means trailing it forever.

*Suggested direction:* replace the proximity term with an interception term —
something like the reduction in predicted time-to-arrival at the runner's
projected objective, or coverage of the runner's remaining route options.
*Signal:* mean episode duration rising before stop rate does. Duration is the
leading indicator here; if containment is being learned, runs get longer first.

### 2.2 Five agents share one reward and one advantage

`security_env.step` computes a single scalar and fans it out unchanged:

```python
rewards = {agent: float(reward) for agent in active_agents}
```

and `marl_train` broadcasts the same advantage to every operative:

```python
actor_advantage = advantage_tensor.unsqueeze(-1).expand(-1, -1, MAX_OPERATIVES).flatten(1, 2)
```

With parameter sharing across five agents there is no signal distinguishing
which operative's action mattered. This is the classic lazy-agent setup, and it
is the single most likely reason the pilots kept collapsing to passive play with
rising entropy.

*Suggested direction:* a counterfactual baseline (COMA-style) is the textbook
answer, but a cheaper first step is to keep the terminal reward shared and make
only the shaping terms per-agent — each operative's own contribution to
containment. Value decomposition (VDN/QMIX-style) is a third option if you would
rather keep the reward global. *Signal:* per-agent action-histogram divergence.
Right now all five operatives should look statistically identical; they should
not after this.

### 2.3 Four observation defects

All four are in `security_env.py` and all four are inside the security
fingerprint, so they should land together in one re-baseline.

**Vision constants are hardcoded and disagree with the simulation.**

```python
visible = self.sim.visible(..., distance=245.0, cosine=0.45)
```

The simulation uses `GUARD_VISION_BASE_DISTANCE + GUARD_VISION_DISTANCE_PER_ALERT * alert`
(205 + 18/tier) and `GUARD_VISION_COSINE` (0.62). So at alert 0 the observation
claims sight the guard does not have (245 > 205, 63° > 52°), and at alert 4 the
relationship inverts (277 > 245). `intent_mask[PURSUE]` is gated on this flag,
so the legal action set disagrees with the detection model, and the direction of
the error changes mid-episode. The runner side already centralised these
constants for exactly this reason; the security env never adopted them.

**Extraction and doors share a one-hot slot.** In `_targets`, the extraction
relay and the nearest security door are both emitted with `kind == 4`. The
policy cannot distinguish them, and `INTERCEPT` legality depends on doors. The
one-hot is 5-wide for 8 semantic target kinds, so the flank offsets also collapse
onto the contact kind — widening it is the straightforward fix.

**The audio estimate leaks exact position.**

```python
estimate = self.sim.player + np.asarray((guard.guard_id % 3 - 1, (guard.guard_id * 2) % 3 - 1)) * TILE_SIZE
```

A fixed per-`guard_id` offset from the true position is trivially invertible, so
"heard" is really "knows exactly". The runner side solves the same problem with
quantised bearing/range in `player_guard_audible_estimate`; mirroring that would
make the two sides symmetric.

**The critic state has no clock.** `state()` packs player, guards, and doors
into 64 values with no remaining-time channel, yet timer expiry is a scoring
outcome worth +20. The value function cannot see the horizon it is being asked
to predict. Adding elapsed/remaining time, alert tier, and active link progress
costs 3 of the 7 currently-wasted padding slots.

### 2.4 One stationary opponent

Training and the final evidence both use the frozen Env-v2 champion
(`env-v2:76baa30a...`) as the only runner. A security policy trained against one
deterministic opponent is fitting that opponent's route distribution, which is a
plausible reading of the tier-4 zero: the champion's tier-4 route may simply be
one the team never learned to cut.

*Suggested direction:* a small opponent pool — the scripted fair policy, the
champion, an earlier BC checkpoint, and a noisy-champion variant — sampled per
episode. Full self-play is a much larger commitment and probably not worth it
before the credit-assignment and shaping items above. *Signal:* the tier-4 zero
moving at all, and the gap between validation slices narrowing.

### 2.5 Reward scale sanity

Terminal is ±20, damage is +5/hit, survival is +0.01/step at 5 Hz. Over a
typical 30 s losing episode survival contributes ~1.2 and damage up to +15, so
the balance is currently reasonable *because episodes are short*. If the
containment work in 2.1 succeeds and episodes start reaching the full window,
survival grows toward +11 at tier 6 and starts competing with the terminal.
Worth re-checking that balance at that point rather than now.

Also note `_security_potential` includes cumulative `damage_taken`, which
double-counts against the direct damage reward and puts a monotone term inside a
potential function. Harmless at current magnitudes, but it means the potential
never telescopes back to its starting value.

---

## 3. Runtime and backend

### 3.1 Headless throughput is per-worker bound

`benchmarks/system/headless-throughput.json`: 3,193.8 aggregate decisions/s
across 22 workers on 24 logical CPUs, but only **148.2 decisions/s per worker**
(~6.7 ms per decision, covering 6 simulation ticks plus observation assembly).
The 5,000/s target is a documented miss and scaling wider is nearly exhausted.

*Suggested direction:* profile the per-decision path rather than adding workers.
`_local_grid` builds an 8x15x15 array with a Python double loop over 225 cells
every single decision — vectorising that against the level grid is the obvious
first candidate, and the security env already demonstrates the pattern with its
cached facility planes. *Signal:* median worker decisions/s, which is the honest
number here; aggregate throughput just tracks core count.

### 3.2 Web payload is dominated by the policy graph

The 384-unit GRU exports to 5.83 MB and the release ships threaded WASM because
it measured faster than WebGPU for a graph this small. Cold transfer is already
a documented ~38.1 MB of raw artefacts against a missed under-25 MB target.

*Suggested direction:* if a 256-unit checkpoint ever matches the 384 on
validation, the payload win is large and compounding. Otherwise structured
pruning or a fresh INT8 attempt (the last candidate was rejected on 5 action
mismatches out of 1,000) are the remaining levers. Any of these needs the full
parity audit, so treat it as a release-gated change and not an experiment.

### 3.3 Small correctness items

- `state()` pads missing guards with `-1.0`, which is indistinguishable from a
  real guard at world origin. A separate presence mask would be unambiguous.
- `GhostlineSecurityParallelEnv.agent_name_mapping` works only because
  `possible_agents` is literally `guard_0..guard_4` so index equals `guard_id`.
  It is correct today and would break silently if guard ids ever became
  non-contiguous. A direct id lookup would remove the coupling.

---

## 4. Deliberately not proposed

- **Minimap doors and corridor connectivity.** It would make the map genuinely
  routable, but the Env-v2 observation only exposes occupancy and doors through
  the local 15x15 grid. A facility-wide map would give the human knowledge the
  policy does not have and weaken the matched human-versus-agent comparison.
  Blocked on a user decision, first raised 2026-07-26.
- **Changing `GUARD_CHASE_SPEED_RATIOS`.** It would fix the security stop rate
  immediately and it would also delete the design decision that dash is the
  decisive escape tool. Product call, not an optimisation.
- **Bundling the web shell's OFL fonts for the desktop UI.** Worth doing for
  visual consistency, but it moves the release asset manifest and the packaging
  gates, so it wants its own turn.
