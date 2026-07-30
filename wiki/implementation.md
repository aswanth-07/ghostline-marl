---
title: Ghostline Implementation
updated: 2026-07-28
status: active
---

# Architecture

Ghostline has two deliberate public contracts and no public v3:

- `GhostlineEnv-v1` is the released single-agent game and policy benchmark.
- `GhostlineEnv-v2` is the in-development multi-agent game, new facility
  distribution, field-systems runner, and adversarial security benchmark.
- `GhostlineLegacyEnv-v0` is a compatibility-only predecessor.

The released source and artifacts historically called their contract
`GhostlineEnv-v2`. Their immutable checkpoint, ONNX graph, 3,000-episode audit,
parity report, and throughput evidence retain that internal label. The
zero-mechanics `env_v1.py` wrapper gives the exact released environment its
stable public v1 name without changing the fingerprint
`521c449a8bd9a540977a918f5b094dd3aeff44cc579a55f75e22a74bab20e129`.

## Layer boundaries

- `simulation.py`, `generation.py`, `types.py`, and `config.py` are the frozen
  published-v1 simulation and facility contract.
- `env_v1.py` is the public-name adapter for the exact published environment.
- `simulation_v2.py`, `generation_v2.py`, `types_v2.py`, and `config_v2.py`
  own all new v2 mechanics and content. They inherit stable v1 rules without
  modifying the fingerprinted files.
- `env_v2.py` is the player-equivalent Gymnasium runner contract.
- `security_env.py` is the PettingZoo Parallel API contract for simultaneous
  security control.
- `security_types.py` contains the dependency-light tactical target enum used
  by the player fallback. This prevents importing PettingZoo merely to run
  deterministic adaptive security in the base wheel; a parity test keeps its
  values synchronized with the training environment.
- `model_v2.py` owns the v2 runner network. `security_model.py` owns the shared
  recurrent operative actor and centralized training critic.
- `presentation.py`, `app.py`, and `audio.py` consume simulation state and event
  streams. Pygame never enters simulation or generation imports.
- Human, scripted, and neural controllers emit the same semantic actions.

Simulation runs at 60 Hz. Runner policies decide at 10 Hz through six-tick
action repeat. Security policies decide at 5 Hz through twelve-tick tactical
repeat. Replays are deterministic from contract, tier, seed, and action
sequence.

## Published v1 contract

`GhostlineEnv-v1` exposes `Discrete(36)`:

`9 movement x 2 dash x 2 pulse`

Its observation dictionary contains the released ego, objective, local-grid,
terminal, security-entity, ray, and action-mask records. Information shared
with the runner is also readable in the human HUD, minimap, or facility
telemetry. The actor and critic receive no hidden live enemy coordinates.

The published v1 game remains the source of the bundled runner checkpoint and
web takeover policy. The wrapper reports:

- `contract: GhostlineEnv-v1`
- `historical_internal_contract: GhostlineEnv-v2`

Evidence validators continue to check the historical internal label because
rewriting those records would break provenance.

## Developmental v2 runner contract

`GhostlineEnv-v2` exposes masked `Discrete(288)`:

`9 movement x 2 dash x 2 pulse x 2 decoy x 2 crouch x 2 interact`

The context-sensitive `interact` bit enters a vent or operates a nearby field
panel. Both actions can never be legal on the same tile; the action mask
selects the meaningful interpretation. Every integer from 0 through 287 has a
unique semantic decode.

The v2 observation is:

| Key | Shape | Meaning |
|---|---:|---|
| `ego` | `27` | runner resources, integrity, trace, timer, quota, alert |
| `objective` | `8` | public phase, bearing, range, next door, link state |
| `directive` | `6` | Standard, Ghost, Speed, or Greed contract context |
| `field` | `8` | owned charges and directly readable interaction state |
| `field_targets` | `16 x 13` | explored vents/panels and directly visible sensors |
| `local_grid` | `15 x 15 x 15` | occupancy, doors, objectives, hazards, field content |
| `targets` | `5 x 10` | known terminal records |
| `entities` | `12 x 16` | visible or audibly inferred security with confidence |
| `rays` | `24 x 4` | geometry, occlusion, and danger distances |
| `action_mask` | `288` | legal flattened actions |

Field observations never expose global unseen devices, sensors, projectiles,
or operative coordinates. Static field content appears after map exploration;
dynamic field sensors require direct visibility. Audio supplies quantized
bearing/range rather than a hidden-state lookup.

## V2 field and stealth systems

- Ghost succeeds after a zero-damage extraction with live trace below `95%`.
  A prior lockdown is recoverable; lifetime maximum trace remains debrief and
  evaluation telemetry. The HUD exposes current trace, and the debrief reports
  both exit and maximum trace so the rule never depends on hidden state.
- Crouch moves at 52% speed, reduces footstep radius, scales awareness gain,
  and improves trace cooling. It does not make the runner invisible and cannot
  silence a dash.
- Walking emits deterministic cadence-based footsteps. Dashing is fast and
  loud and carries a trace cost.
- Vents are reciprocal, runner-only, cross-room pairs with a 1.15-second
  committed transit. The runner cannot steer, hack, extract, take damage, or
  trigger a field sensor in transit, while the mission clock and full security
  world continue to advance.
- Field hacks share limited charges. Camera panels disable their camera, light
  panels darken their own room, and a door panel opens only its bound security
  door. No panel can silently affect a random or global target.
- Darkness scales the shared physical visibility predicate for observers and
  targets in the affected room. Detection, policy observations, and rendered
  cones therefore use the same range.
- Decoys create a bounded lure window at their landing point. A crouched throw
  is shorter and quieter.
- Patrol and Interceptor operatives can deploy one non-damaging sensor.
  Sensors require line of sight to trip. Suppressors retain a telegraphed,
  nonlethal projectile instead.
- PINCER and SEAL consume explicit public escape-route or door targets. They do
  not infer the runner's hidden objective, heading, or unseen exact position.
- Direct visual pursuit is a motor-level safety reflex and cannot be demoted by
  a repeated HOLD order. This removes the old detection-reward loop.

## V2 facility generation

`FacilityLayoutV2` starts from a valid published layout, reshapes it
deterministically, adds field content, and validates the final result. It does
not reimplement base reachability rules.

Geometry passes add alcoves, corner recesses, role-specific aisles, pillars,
partitions, and nonblocking environmental detail. Placement uses one reserved
tile contract: essential vents and panels claim space before decorative props,
so a cosmetic object cannot hide or overlap an interaction.

Every accepted seed must satisfy both base validity and v2 readiness:

- reachable quota and extraction, safe spawn, valid patrols, and route loops;
- exact reciprocal vent counts with distinct-room, geodesically separated,
  reachable endpoints;
- exact device counts, valid target type and id, and reachable panel tiles;
- door panels bound to the exact generated security-door tile;
- no overlap among doors, objectives, vents, panels, props, and protected
  doorway throats;
- graph-safe security doors and a positive directive speed par.

Generation makes eight deterministic reshape attempts and then fails loudly.
There is no feature-silent fallback. The release gate is a 10,000-seed audit
through `scripts/fuzz_ghostline_levels.py --adaptive`.

## V2 runner network

`RunnerPolicyV2` uses:

- a convolutional encoder for the 15-channel local grid;
- masked attention pooling for terminals, perceived entities, and field
  targets;
- MLP encoders for ego, objective, rays, and field status;
- FiLM conditioning for the contract directive;
- a 256-, 384-, or 512-unit GRU;
- separate policy and value decoders;
- objective-bearing and danger heads reserved for an auxiliary-loss ablation;
- exact action masking before sampling or argmax.

Hidden layers use orthogonal initialization, action logits use gain `0.01`,
and value heads use gain `1.0`. The v2 checkpoint stores observation contract,
action count, recurrent width, and a normalized source fingerprint covering
all inherited and v2 transition/observation files. Loading fails closed on any
mismatch. No v1 runner checkpoint can load as a v2 runner.

## V2 multi-agent security contract

`GhostlineSecurityParallel-v2` exposes up to five simultaneous operatives.
Each actor action is factorized as:

1. ten semantic intents: patrol, investigate, search, pursue, intercept,
   flank-left, flank-right, hold, pincer, or seal;
2. one of ten public tactical target rows;
3. a discrete radio message;
4. a role-gated ability bit.

The target set includes patrol, perceived contact, heard contact, terminal,
extraction, door, flank, and public escape-route cutoff records. PINCER accepts
an escape-route target. SEAL accepts a door or escape-route target. Invalid
factor combinations fall back safely and are penalized once. An explicit
intent-by-target conditional mask prevents sampling an individually legal
intent and individually legal target that are illegal together.

Actors receive an 8-channel local grid, their own state, perception-gated
runner information, masked teammates, masked targets, and recent radio. The
training-only central state is 72 values: mission context, five fixed-width
operative blocks, facility/door context, and an explicit presence mask.

## Security network and credit assignment

The default learner is parameter-shared recurrent MAPPO:

- one decentralized GRU actor is shared across roles;
- the target head is a pointer over the current target rows rather than a fixed
  index classifier;
- the centralized critic uses a shared operative encoder and masked pooling;
- the critic emits one value per operative, conditioned on global context,
  unordered team context, and that operative's own block;
- active masks exclude padded, absent, and terminated agents from GAE, policy,
  entropy, and value losses;
- rollout hidden state resets exactly at episode boundaries.

The reward ledger separates shared outcomes from attributed shaping. Shared
terms include damage, first contact acquisition, runner data, bounded radio
assist, invalid actions, formation, discount-matched team potential, and
terminal containment/extraction. A small bounded, discount-matched
per-operative potential attributes route coverage and awareness to the agent
that earned it. Components are clipped individually, summed exactly, and
reported in `info`.

The team potential uses geodesic escape-route coverage, awareness, trace, and
runner mission progress. It is not Euclidean tail-chasing. Detection credit is
awarded once per operative per episode, so issuing HOLD cannot manufacture new
contact bonuses.

## Checkpoint compatibility

The following artifact families are intentionally separate:

- published v1 runner checkpoint/ONNX/evidence: valid and immutable, with
  historical internal `GhostlineEnv-v2` metadata;
- pre-migration adaptive security checkpoint and 13M evidence: historical
  only, invalid for the current v2 fingerprint;
- developmental v2 runner checkpoints: must carry the current
  `runner-recurrent-field-policy-v2` fingerprint;
- developmental v2 security checkpoints: must carry the current observation,
  mechanics, generation, reward, and model fingerprint.

No result from an invalidated checkpoint may be described as a current v2
result. Until new training and held-out evaluation finish, v2 uses the
deterministic tactical fallback in player builds.

## Presentation contract

The renderer presents a 640x360 world with native-resolution UI, integer
desktop scaling, smooth physical cones, eight-direction locomotion, persistent
security readability, compact HUD, minimap, captions, and touch controls.
Darkness is composited after world props but before objectives, prompts,
sensors, and actors so the mechanical state remains readable. Rendered cone
lengths call the same darkness scaling used by simulation visibility.

### Frame budget

The browser build interprets Python, so the frame cost is dominated by the
number of Python-level draw calls rather than by pixels. Two caches keep that
count low:

- **Terrain** is painted once per level into a floor surface and a wall surface
  and then blitted. The simulation never writes to the level grid, so terrain is
  a pure function of the grid, room roles and seed. Tile coordinates are exact
  multiples of the tile size and adding an integer commutes with rounding, so
  the blit lands on the same pixels the per-tile path produced. Floor and walls
  stay separate surfaces because vision cones composite between them, and the
  wall layer carries per-pixel alpha so it cannot erase a cone.
- **Vision-cone fans** are memoised by quantised observer pose. The cast depends
  only on static geometry, so a guard that holds, aims or waits re-uses its rays
  while screen projection still runs every frame. The cache is bounded and
  cleared wholesale rather than tracking recency.

Both caches invalidate on `(seed, tier, level identity)`. A cached frame is
pixel-identical to one drawn with a cold cache; cone-pose quantisation is the
only approximation and is bounded by regression at well under 0.5% of pixels.
Isolated gameplay draw cost on the reference desktop is 2.14 ms per frame
against a 16.67 ms budget, measured across tiers 1, 3, 4 and 6.

Desktop v2 controls add `Q` decoy, `Left Ctrl` crouch, and `E` interact.
Touch adds dedicated sneak, decoy, and use buttons. The web shell labels the
released game as v1 and the multi-agent game as v2, and refuses to hand the
published v1 ONNX policy a v2 observation.

## Verification gates

- Gymnasium checker, PettingZoo parallel API, observation bounds, and masks.
- Deterministic replay, collision, LOS/cone parity, hacking, vents, trace,
  damage, guard orders, sensors, reward sums, and termination tests.
- Exact 288-action round-trip through the Gym wrapper.
- 10,000 v2 seeds with zero readiness failures.
- Runner PPO and MAPPO smoke runs with finite losses and resume checks.
- Headless throughput measured after the distribution freezes.
- Held-out runner and security evaluation from disjoint namespaces.
- Browser and Vercel QA in Chrome only; no in-app browser.

The frozen developmental runner fingerprint is
`be4a280a0d629cadabec08d038497eef331a14650c3e5fd23e97d4afca61efdd`.
Its reserved 20M final-test entry remains unopened. The 2026-07-28 readiness
pass completed 10,000 generated facilities, runner and security fresh/resume
smokes, an isolated clean install, source-archive audits, and 1,000/1,000 v2
runner ONNX action parity. No long v2 campaign or learned-policy acceptance
claim has been made.
