---
title: Ghostline Implementation
updated: 2026-07-13
status: active
---

# Architecture

Ghostline is a clean break from the legacy `neon_arena` package.

## Layers

- `simulation.py`: deterministic fixed-timestep movement, hacking, trace, integrity, guard/camera/drone AI, pulse, timer, and event stream.
- `generation.py`: modular room graph, authored furniture arrangements, security placement, tile collision, and validation.
- `presentation.py` and `app.py`: scrolling 640x360 pixel canvas, integer scaling, cinematic menus, camera, lighting, props, HUD, minimap, effects, accessibility, run telemetry, and game flow.
- `env.py`, `model.py`, and the training pipeline: versioned Gymnasium contract, entity-aware recurrent policy, action masking, rewards, curriculum, imitation learning, and PPO integration.

Simulation and generation do not import Pygame. Human and agent controllers both produce `Action(move, dash, pulse)`. Replays are deterministic from seed, tier, and action sequence.

## Fairness and interaction contract

- Terminal linking continues while the runner moves anywhere inside its visible interaction ring; leaving pauses progress without erasing it. The outer ring uses the literal 40 px simulation radius, while the smaller inner pulse remains decorative.
- Damage preserves partial terminal progress and has a 1.35-second global recovery window.
- Guard and camera sight accumulates awareness before confirmation and decays when line of sight breaks.
- Guard tackles require a visible 0.42-second strike wind-up and disengage into search after impact; response-drone contact requires a 0.55-second charge cue followed by one second of recoil. A 1.35-second global damage-recovery window prevents contact dogpiles while preserving chase pressure.
- Security generation reserves space around spawn, extraction, and terminals; validation rejects objective/security overlaps.
- Every terminal now has at least three collision-safe tile-centre link positions and at least one position outside the complete sweep envelope of every camera. Camera mounts stay five Manhattan tiles from objectives, point along useful room sightlines while explicitly minimizing objective exposure, and never sweep the spawn or extraction relay. Guards keep a four-tile objective buffer and all security stays two tiles away from doorway throats.
- Facility graphs guarantee alternate-route loops as difficulty grows: one loop on Surveillance/Patrol, two on Countermeasure/Lockdown, and three on Ghostline. This prevents a single cone or patrol from turning the only route into a forced detection check while preserving deterministic seed composition.
- Terminals are distributed across distinct rooms with open authored interaction pockets. Server, security, and vault terminals carry predictable higher-value potential; quota repair deterministically promotes specialist targets instead of rejecting seeds until random values happen to add up.
- Countermeasure uses three human guards plus its networked cameras, doors, and guarded terminals. The earlier four-guard curve created a non-monotonic integrity-loss spike above the drone-backed Lockdown tier; the corrected curve preserves the lesson while restoring a readable escape window.
- Tier 5 emphasizes systemic lockdown with four cameras, three roaming patrols, four pulses, and a response drone at full trace (100); tier 6 raises this to five cameras and a five-operative threat pyramid while reducing the runner to three pulses and deploying its drone at elevated trace (72), preserving a readable step into the full-system contract.
- Guards have explicit Standard, Interceptor, and Elite grades. Tier 3 introduces Standard patrols, tier 4 promotes only its final patrol to Interceptor, tier 5 deliberately mixes all three, and tier 6 fields three Standard, one Interceptor, and one Elite. Chase speeds are 95/97/99% of the runner's normal 126 px/s speed, so an Elite can maintain pressure without overtaking an undashed runner; dash remains the decisive escape tool. Patrol and search retain the `1.04x/1.10x/1.16x` grade curve and deterministic `0.78/0.52/0.36` second scan pauses. Small `I/II/III` floor badges communicate grade without relying on color; `EYE`, `SOUND`, and `RADIO` labels communicate the current stimulus.
- The recalibrated fair teacher uses a Tier-6 directional-inertia scale of 1.8x with objective/clearance/escape/sight weights `20/1/0.8/0.8`. It was selected only on validation seeds and changes controller decisions, not observations, simulation rules, or security behavior.
- The observation-only teacher treats the projected runner footprint as the authoritative local collision test, with adjacent blocked cells used only as soft clearance preferences. It interprets live facility telemetry, `RETURN` as recovery rather than maximum alert, public guard grade as a risk signal, and entity distances using the shared 390 px normalization constant. Two disjoint current-fingerprint 100-seed-per-tier gates cleared the teacher thresholds at `100/100/99/99/99/86%` and `100/100/99/99/99/89%`.
- Guard navigation uses radius-checked five-cell path look-ahead, combined/axis fallback movement, choke-aware separation, close-waypoint invalidation, smooth collision recovery, and stuck detection. Authored patrols now contain distinct local, doorway, and neighbouring-room points. A 1,146-traversal directed doorway audit found zero jamb stalls.
- Exploration reveals the exposed face of blocking furniture and walls as soon as LOS reaches a nearer adjacent floor tile. The persistent explored mask is retained for player-equivalent policy sensing, reward accounting, and map knowledge, but it no longer dims world art; furniture, floors, and walls always render at authored brightness.
- Partial guard sightings enter a visible suspicious pause before confirmation. Search guards scan the last-known position, and one successful tackle disperses nearby chasers into a short search window so a three-guard doorway dogpile cannot chain unavoidable integrity loss.
- Simulation owns a deterministic, renderer-free facility-telemetry cache. The breached security network publishes current guard, camera, and drone positions to the renderer and all Env-v2 controllers, preventing operatives from disappearing at arbitrary wall boundaries. This is a shared tactical-game rule, not a renderer-only wallhack: human play, the minimap, Python policies, ONNX takeover, recording, and replay receive the same live entity rows. Physical detection still uses true occlusion, distance, facing, and awareness.
- Presentation keeps guards, cameras, drones, and their current state readable at all times. Hostile sight envelopes remain physical rather than tile-based: cameras use 220 px and `acos(0.72)`, while guards use `205 + 18 * alert tier` px and `acos(0.62)`. Sixty-five occlusion-refined rays create a smooth continuous fan and stop it at the first wall or blocking prop; live operative telemetry does not let a cone see through that occluder.
- Dash emits one restrained ring showing its literal 185 px sound radius. Only one wave exists per continuous dash, dash-trail particles are emitted four times less often, particles are capped at 72, and the decorative twelve-spoke burst was removed. Detection feedback exposes the recoverable loop as `SPOTTED -> LINE BROKEN -> SEARCHING -> CLEAR`; trace bars mark the 25/50/75 escalation steps and drone-backed tiers show their deployment threshold before arrival.
- Acquire-phase objective selection is sticky: once a terminal is selected it remains the shared HUD/policy target until completed, quota is met, or the runner deliberately enters another terminal's link ring. This hysteresis prevents equidistant terminals from flipping the route bearing at tile boundaries.
- The objective observation's next-waypoint bearing follows the distance-map gradient for up to six line-of-sight tiles. The look-ahead remains player-equivalent to the HUD route hint while avoiding one-tile left/right oscillation around furniture and doors.

## Procedural facilities and presentation

Facilities use 11x9-tile room modules on a connected graph. Roles include office, lounge, lab, server, security, vault, utility, corridor, and extraction. Roles are dealt without replacement before repeating so compact maps do not collapse into duplicate room themes. Each furnished role now selects one of three authored arrangements containing desks, tables, chairs, sofas, TVs, monitors, racks, consoles, lockers, plants, crates, generators, or vault cases. Modular server and locker banks plus incompatible vertical sofa and thin console runs are emitted as deterministic one-tile visual modules: the union of blocking cells is unchanged, but every blocked tile receives visible art instead of relying on one bottom-pivoted atlas crop at the end of a long footprint. Presentation adds deterministic role-specific materials, decals, wall trim, and animated equipment without entering generation or simulation imports.

The post-freeze procedural audit validated 10,000/10,000 seeds under these stronger rules at 99.0 levels/second, with exact security counts, reachable quota/extraction, unobstructed doors, valid multi-point patrols, minimum route redundancy, and zero camera-locked terminals.

The renderer builds a cached tile-to-room-role lookup for each level. It uses original runtime eight-direction actors, smooth occlusion-clipped cones, terminal/extraction glow, directional threat indicators, event particles, captions, and cinematic effects. The old square explored-space fog layer was removed because it obscured furniture and made vision feel tile-based. Surveillance has three deliberately distinct visual states: a faint amber true sight envelope, amber-to-danger acquire feedback, and a high-contrast confirmed-detection state. Camera cones carry a dashed centre beam and square glyph; guard cones use boundary notches and a triangle glyph. Four-segment entity badges, an eight-segment top-centre acquire meter, and labeled edge arrows communicate the same escalation through color, shape, and text. Menus use the same flat code-native facility schematic as the 2D game; the retired pseudo-isometric key art is provenance-only and never packaged.

## Game flow, accessibility, and local data

The executable includes title, main menu, contract selection, briefing, play, pause, field manual, grouped settings, credits, debrief, Agent Lab selection, and Agent Lab playback. Agent Lab exposes deterministic seed controls, runtime identity, action, latency, recurrent-state norm, objective phase, and matched local human/agent summaries.

The versioned profile at `%LOCALAPPDATA%/Ghostline/progression-v1.json` stores unlocks, scores, audio mix, display/accessibility settings, and keyboard bindings. Every gameplay/menu action is remappable; conflicting assignments swap instead of silently disabling an action. Full benchmark records append to `%LOCALAPPDATA%/Ghostline/runs-v1.jsonl`, including sampled position/trace curves, actions, idle rate, distance, efficiency, outcomes, and agent latency.

Accessibility includes independent master/music/SFX volume, sound captions, high contrast, color-safe cues, reduced motion, reduced flashes, three HUD scales, an opt-in 35% human timer assist, timer warnings, tutorial hints, screen shake, fullscreen, and remappable keyboard controls. Assisted runs are explicitly tagged in telemetry and do not alter the default environment or agent contract. Exact integer scale is used where the window permits; non-native ratios are letterboxed rather than stretched.

## Public contract

- Current Gym id: `GhostlineEnv-v2`; `GhostlineEnv-v1` is the documented baseline.
- Action: `Discrete(36)` = `9 movement x 2 dash x 2 pulse`.
- Simulation: 60 Hz; policy control: 10 Hz through six-tick repeat.
- Observation: ego, explicit objective, local structured grid, known targets, live shared facility-security telemetry, directional rays, masks, and legal actions. The 12 entity rows contain 13 values: kind one-hot; relative x/y/distance; velocity; facing; alert; confidence/status; and explicit guard grade. The human HUD/minimap and policy receive the same current operative records with confidence one.
- Facility occupancy, doors, known terminal locations, and extraction geometry match the always-visible minimap; the separate explored channel remains a route-memory and reward signal. Pulse count is normalized across the literal zero-to-four charge range with no clipping.
- Objective vector: phase, goal dx/dy, route distance, next-waypoint dx/dy, link progress, and target value.
- Terminal telemetry: success/reason, tier/seed, quota/data, duration, trace, detections, guard/drone damage attribution, pulse use, path distance, action histogram, idle decisions, efficiency, and exact reward components.

Training-only lessons simplify mechanics in seven reverse-curriculum stages; final validation never applies those modifications. Human play writes a corresponding local telemetry schema for the later locked matched-seed benchmark.

The fair teacher uses only this public contract. Its final tier-5 movement curve increases mission commitment against the persistent Lockdown drone while leaving electronic/pursuit relief to pulse timing; it does not add privileged state. BC, DAgger, RND, recurrent PPO, optimizer state, and curriculum checkpointing are implemented directly in PyTorch rather than through an external RL framework.

The live-operative telemetry, faster chase curve, five-operative tier-6 pyramid, and recalibrated teacher changed the training-environment fingerprint again on 2026-07-14. Every earlier BC/DAgger corpus and neural checkpoint is rejected for final selection; only fresh trajectories whose manifest matches fingerprint `521c449a...e129` may enter current training.

Final-test audit slices are immutable after inspection. The 3M, 4M, and 5M slices are retained as failed generalization evidence; the teacher passed the then-current curve on the untouched 6M slice. The later route/security/patrol freeze deliberately changed the procedural distribution, so all 2M–6M results are now historical and a later untouched slice is reserved for the selected frozen-distribution champion. None of these results establishes neural-policy acceptance.
