"""Deterministic simulation extensions for the multi-agent v2 contract."""

from __future__ import annotations

from collections.abc import Mapping
import itertools
import math

import numpy as np

from ghostline.config import (
    DETECTION_GRACE_SECONDS,
    GUARD_CHASE_SPEED_RATIOS,
    GUARD_GRADE_SPEED_MULTIPLIERS,
    GUARD_PATROL_DWELL_SECONDS,
    GUARD_SEARCH_DURATION_MULTIPLIERS,
    GUARD_STRIKE_WINDUP_SECONDS,
    GUARD_VISION_BASE_DISTANCE,
    GUARD_VISION_COSINE,
    GUARD_VISION_DISTANCE_PER_ALERT,
    TRACE_MAX,
    PLAYER_GUARD_AUDIBLE_DISTANCE,
    PLAYER_RADIUS,
    PLAYER_SPEED,
    PULSE_RADIUS,
    SIM_HZ,
    TILE_SIZE,
)
from ghostline.config_v2 import (
    DECOY_LIFETIME_SECONDS,
    COVER_TRACE_DECAY_BONUS,
    CROUCH_AWARENESS_SCALE,
    CROUCH_FOOTSTEP_RADIUS,
    CROUCH_SPEED_SCALE,
    CROUCH_TRACE_DECAY_BONUS,
    CHOKEPOINT_MIN_RUNNER_DISTANCE,
    CHOKEPOINT_TEAM_COOLDOWN_SECONDS,
    DECOY_CROUCH_THROW_SCALE,
    DECOY_LURE_RADIUS,
    DECOY_LURE_SECONDS,
    FIELD_SENSOR_ARM_SECONDS,
    FIELD_SENSOR_CHARGES,
    FIELD_SENSOR_LIFETIME_SECONDS,
    FIELD_SENSOR_RADIUS,
    HACK_CAMERA_DISABLE_SECONDS,
    HACK_CHARGES_PER_TIER,
    HACK_COOLDOWN_SECONDS,
    HACK_DOOR_OVERRIDE_SECONDS,
    HACK_LIGHTS_SECONDS,
    HACK_LIGHTS_VISION_SCALE,
    HACK_RANGE,
    VENT_TRANSIT_SECONDS,
    WALK_FOOTSTEP_RADIUS,
    DASH_TRACE_COST_PER_SECOND,
    DECOY_NOISE_RADIUS,
    DECOY_PULSE_INTERVAL_SECONDS,
    DECOY_THROW_DISTANCE,
    SECURITY_DOOR_FORCED_OPEN_SECONDS,
    SECURITY_DOOR_LOCK_SECONDS,
    SECURITY_DOOR_TEAM_COOLDOWN_SECONDS,
    SECURITY_DOOR_WARNING_SECONDS,
    SUPPRESSOR_AIM_SECONDS,
    SUPPRESSOR_COOLDOWN_SECONDS,
    SUPPRESSOR_MAX_RANGE,
    SUPPRESSOR_MIN_RANGE,
    SUPPRESSOR_PROJECTILE_LIFETIME_SECONDS,
    SUPPRESSOR_PROJECTILE_RADIUS,
    SUPPRESSOR_PROJECTILE_SPEED,
)
from ghostline.generation import tile_center, world_to_tile
from ghostline.generation_v2 import FacilityLayoutV2
from ghostline.simulation import GhostlineSimulation, MOVE_DIRECTIONS, norm, unit
from ghostline.types import Guard, GuardMode, SimEvent, Tile
from ghostline.types_v2 import (
    RUNNER_ACTION_COUNT_V2,
    FieldSensor,
    ContractDirective,
    Decoy,
    GuardRole,
    OperativeState,
    RadioMessage,
    RadioTransmission,
    RunnerActionV2,
    SecurityDoor,
    SecurityIntent,
    SecurityOrder,
    ShockProjectile,
)


_ACTION_VALUES = np.arange(RUNNER_ACTION_COUNT_V2, dtype=np.int16)
_ACTION_MOVE = _ACTION_VALUES % 9
_ACTION_DASH = (_ACTION_VALUES // 9) % 2 == 1
_ACTION_PULSE = (_ACTION_VALUES // 18) % 2 == 1
_ACTION_DECOY = (_ACTION_VALUES // 36) % 2 == 1
_ACTION_CROUCH = (_ACTION_VALUES // 72) % 2 == 1
_ACTION_INTERACT = (_ACTION_VALUES // 144) % 2 == 1
_STATIC_ACTION_MASK = np.ones(RUNNER_ACTION_COUNT_V2, dtype=np.int8)
_STATIC_ACTION_MASK[_ACTION_DASH & (_ACTION_MOVE == 0)] = 0
_STATIC_ACTION_MASK[_ACTION_CROUCH & _ACTION_DASH] = 0


class GhostlineSimulationV2(GhostlineSimulation):
    """V2 deterministic extension with semantic security control.

    Classic simulation remains untouched.  This subclass owns every new state
    object so a v2 rollout can never silently change a published-v1 replay.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        tier: int = 1,
        directive: ContractDirective | str | int = ContractDirective.STANDARD,
        external_security: bool = False,
    ):
        self.directive = ContractDirective.parse(directive)
        self.external_security = bool(external_security)
        super().__init__(seed=seed, tier=tier)

    crouching: bool = False

    def reset(self, *, seed: int | None = None, tier: int | None = None) -> None:
        # The multi-agent track builds its own reshaped facilities. Swapping the
        # generator here rather than adding a seam to the base class keeps
        # simulation.py byte-identical, which the published-v1 fingerprint
        # requires. ``__init__`` assigns the default generator before calling
        # reset, so this override always runs first.
        if not isinstance(self.generator, FacilityLayoutV2):
            self.generator = FacilityLayoutV2()
        super().reset(seed=seed, tier=tier)
        self.decoy_charges = 0 if self.tier <= 2 else (1 if self.tier <= 4 else 2)
        self.decoys_used = 0
        self.decoy_cooldown = 0.0
        self._decoy_latched = False
        self.decoys: list[Decoy] = []
        self.projectiles: list[ShockProjectile] = []
        self._next_decoy_id = 0
        self._next_projectile_id = 0
        self.operative_states = {
            guard.guard_id: OperativeState(role=self._role_for_guard(guard))
            for guard in self.level.guards
        }
        self.radio_log: list[RadioTransmission] = []
        self._pending_security_orders: dict[int, SecurityOrder] = {}
        self.security_doors = self._build_security_doors()
        self.security_door_cooldown = 0.0
        self._base_blocked_tiles = set(self._blocked_tiles)
        self._v2_locked_tiles: set[tuple[int, int]] = set()
        self._refresh_navigation_blocks()
        self.vents = list(getattr(self.level, "vents", []))
        self.hackable = self._bind_hackable_devices(
            list(getattr(self.level, "hackable", []))
        )
        self.hack_charges = HACK_CHARGES_PER_TIER.get(self.tier, 2)
        self.hack_cooldown = 0.0
        self.hacks_used = 0
        self._interact_latched = False
        self.vent_transit = 0.0
        self._vent_destination: np.ndarray | None = None
        self.vent_uses = 0
        self.darkened_rooms: dict[int, float] = {}
        self.field_sensors: list[FieldSensor] = []
        self._next_sensor_id = 0
        self.sensor_charges = {guard.guard_id: FIELD_SENSOR_CHARGES for guard in self.level.guards}
        self.directive_par_seconds = self._speed_directive_par_seconds()

    def _bind_hackable_devices(self, devices):
        """Bind every control pedestal to one real, functional facility system."""

        bound = []
        security_by_tile = {door.tile: door for door in self.security_doors}
        used_door_ids: set[int] = set()
        room_ids = {int(room.room_id) for room in self.level.rooms}
        camera_ids = {int(camera.camera_id) for camera in self.level.cameras}
        for device in sorted(devices, key=lambda item: item.device_id):
            if device.kind == "camera":
                if int(device.target_id) not in camera_ids:
                    continue
            elif device.kind == "lights":
                if int(device.target_id) not in room_ids:
                    continue
            elif device.kind == "door":
                door = security_by_tile.get(device.target_tile)
                if door is None:
                    door = next(
                        (
                            candidate
                            for candidate in sorted(
                                self.security_doors, key=lambda item: item.door_id
                            )
                            if candidate.door_id not in used_door_ids
                        ),
                        None,
                    )
                if door is None:
                    continue
                used_door_ids.add(int(door.door_id))
                device.target_id = int(door.door_id)
                device.target_tile = tuple(door.tile)
            else:
                continue
            bound.append(device)
        return bound

    def _speed_directive_par_seconds(self) -> float:
        """Deterministic, seed-specific par shared by game and RL wrapper."""

        best = math.inf
        terminals = self.level.terminals
        for count in range(1, len(terminals) + 1):
            for subset in itertools.permutations(terminals, count):
                if sum(terminal.value for terminal in subset) < self.level.quota:
                    continue
                points = [self.level.spawn, *(terminal.position for terminal in subset), self.level.extraction]
                travel = sum(norm(second - first) for first, second in zip(points, points[1:]))
                link = sum(terminal.hack_seconds for terminal in subset)
                best = min(best, travel / PLAYER_SPEED + link + 8.0)
            if math.isfinite(best):
                break
        return max(12.0, float(best if math.isfinite(best) else self.level.mission_seconds * 0.65))

    @property
    def directive_completed(self) -> bool:
        if not self.extracted:
            return False
        if self.directive == ContractDirective.GHOST:
            return self.damage_taken == 0 and self.max_trace < 75.0
        if self.directive == ContractDirective.SPEED:
            return self.elapsed_seconds <= self.directive_par_seconds
        if self.directive == ContractDirective.GREED:
            return all(terminal.completed for terminal in self.level.terminals)
        return True

    def _role_for_guard(self, guard: Guard) -> GuardRole:
        count = len(self.level.guards)
        if self.tier >= 5 and guard.guard_id == count - 1:
            return GuardRole.SUPPRESSOR
        if self.tier >= 4 and guard.guard_id == max(0, count - 2):
            return GuardRole.INTERCEPTOR
        return GuardRole.PATROL

    def _build_security_doors(self) -> list[SecurityDoor]:
        requested = {4: 1, 5: 2, 6: 3}.get(self.tier, 0)
        candidates = [door for door in self.level.doors if self._door_edge_is_redundant(door.room_a, door.room_b)]
        candidates.sort(key=lambda door: (door.tile[1], door.tile[0], door.room_a, door.room_b))
        if candidates:
            offset = int(np.random.SeedSequence([self.seed, self.tier, 3001]).generate_state(1)[0]) % len(candidates)
            candidates = candidates[offset:] + candidates[:offset]
        return [SecurityDoor(index, door.tile) for index, door in enumerate(candidates[:requested])]

    def _door_edge_is_redundant(self, room_a: int, room_b: int) -> bool:
        visited = {room_a}
        pending = [room_a]
        while pending:
            room = pending.pop()
            for neighbour in self.level.adjacency[room]:
                if {room, neighbour} == {room_a, room_b}:
                    continue
                if neighbour not in visited:
                    visited.add(neighbour)
                    pending.append(neighbour)
        return room_b in visited

    def action_mask(self) -> np.ndarray:
        dash_available = self.dash_energy > 1.0
        pulse_available = self.pulse_charges > 0 and self.pulse_cooldown <= 0.0
        decoy_available = self.decoy_charges > 0 and self.decoy_cooldown <= 0.0
        interact_available = self.can_interact()
        signature = (
            dash_available,
            pulse_available,
            decoy_available,
            interact_available,
        )
        if getattr(self, "_action_mask_signature", None) == signature:
            return self._action_mask_cache.copy()
        mask = _STATIC_ACTION_MASK.copy()
        if not dash_available:
            mask[_ACTION_DASH] = 0
        if not pulse_available:
            mask[_ACTION_PULSE] = 0
        if not decoy_available:
            mask[_ACTION_DECOY] = 0
        if not interact_available:
            mask[_ACTION_INTERACT] = 0
        self._action_mask_signature = signature
        self._action_mask_cache = mask
        return mask.copy()

    def set_security_orders(self, orders: Mapping[int, SecurityOrder]) -> None:
        self._pending_security_orders = {
            int(guard_id): order
            for guard_id, order in orders.items()
            if int(guard_id) in self.operative_states
        }

    def _tick(self, action: RunnerActionV2, *, allow_pulse: bool) -> None:
        dt = 1.0 / SIM_HZ
        # Crouching is only honoured while actually sneaking: it cannot be used
        # to cancel a dash mid-frame, which would make the loud route free.
        self.crouching = bool(action.crouch) and not (
            action.dash and action.move != 0 and self.dash_energy > 0.0
        )
        self.decoy_cooldown = max(0.0, self.decoy_cooldown - dt)
        self.hack_cooldown = max(0.0, self.hack_cooldown - dt)
        self._update_security_doors(dt)
        self._update_hacks(dt)
        self._update_field_sensors(dt)
        self._update_decoys(dt)

        # Vent transit freezes only the runner.  The facility clock, cameras,
        # guards, drones, trace, cooldowns, doors, and termination checks keep
        # advancing through the ordinary deterministic world tick.
        if self.vent_transit > 0.0:
            self.vent_transit = max(0.0, self.vent_transit - dt)
            self.velocity[:] = 0.0
            if self.vent_transit <= 0.0 and self._vent_destination is not None:
                self.player[:] = self._vent_destination
                self._vent_destination = None
                self.events.append(SimEvent("vent_exit", tuple(self.player)))
            super()._tick(RunnerActionV2(), allow_pulse=False)
            self._update_operative_state(dt)
            self._apply_field_abilities()
            self._update_suppressors(dt)
            self._update_projectiles(dt)
            return

        if action.interact and not self._interact_latched:
            self._activate_interact()
            self._interact_latched = True
        if not action.interact:
            self._interact_latched = False

        if action.decoy and not self._decoy_latched:
            self._activate_decoy(action)
            self._decoy_latched = True
        if not action.decoy:
            self._decoy_latched = False
        super()._tick(action, allow_pulse=allow_pulse)
        self._emit_footsteps(action)
        self._update_operative_state(dt)
        self._apply_field_abilities()
        self._update_suppressors(dt)
        self._update_projectiles(dt)

    def _emit_footsteps(self, action: RunnerActionV2) -> None:
        """Make ordinary movement audible so quiet play is a real choice.

        The frozen simulation only broadcasts a dash wave, which is why walking
        everywhere was free. Footsteps are emitted on a fixed cadence rather
        than every tick so the cue stays readable and the cost stays bounded.
        """

        if action.move == 0 or self._dash_latched or norm(self.velocity) < 12.0:
            return
        if self.elapsed_ticks % 20 != 0:
            return
        radius = CROUCH_FOOTSTEP_RADIUS if self.crouching else WALK_FOOTSTEP_RADIUS
        self._broadcast_noise(radius=radius)
        self.events.append(SimEvent("footstep", tuple(self.player), radius))

    def _move_player(self, delta: np.ndarray) -> None:
        """Crouching trades speed for silence."""

        if self.crouching:
            delta = delta * CROUCH_SPEED_SCALE
        super()._move_player(delta)

    def _broadcast_noise(self, *, radius: float) -> None:
        """Footsteps carry a stealth-dependent radius.

        The frozen simulation broadcasts one 185 px dash wave and nothing for
        ordinary movement. V2 makes movement itself audible so that quiet
        and loud are genuinely different routes rather than the same route at
        different speeds.
        """

        if self.crouching:
            radius = min(radius, CROUCH_FOOTSTEP_RADIUS)
        super()._broadcast_noise(radius=radius)

    def _update_trace(self, dt: float) -> None:
        """Loud is viable but expensive; quiet cools the network faster."""

        super()._update_trace(dt)
        if self._dash_latched:
            self.trace = min(TRACE_MAX, self.trace + DASH_TRACE_COST_PER_SECOND * dt)
        elif not self._was_seen:
            relief = CROUCH_TRACE_DECAY_BONUS if self.crouching else 0.0
            if self.in_cover:
                relief += COVER_TRACE_DECAY_BONUS
            if relief:
                self.trace = max(self.trace_floor, self.trace - relief * dt)
        self.max_trace = max(self.max_trace, self.trace)

    @property
    def in_cover(self) -> bool:
        """True when the runner is against a blocking prop or wall.

        Cover is read from the existing collision grid rather than authored
        volumes, so every generated facility supports it without changing
        generation or adding a second source of truth.
        """

        tx, ty = world_to_tile(self.player)
        for offset_x, offset_y in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = tx + offset_x, ty + offset_y
            if not (0 <= ny < self.level.grid.shape[0] and 0 <= nx < self.level.grid.shape[1]):
                return True
            if (nx, ny) in self._blocked_tiles or self.level.grid[ny, nx] == Tile.WALL:
                return True
        return False

    # -- runner field systems ----------------------------------------------

    def nearest_vent(self):
        """Return the vent under the runner, if any."""

        tile = world_to_tile(self.player)
        return next((vent for vent in self.vents if vent.tile == tile), None)

    def nearest_hackable(self):
        """Return the closest ready device inside hack range."""

        ready = [
            device
            for device in self.hackable
            if device.cooldown <= 0.0 and norm(device.position - self.player) <= HACK_RANGE
        ]
        if not ready:
            return None
        return min(ready, key=lambda device: norm(device.position - self.player))

    def can_interact(self) -> bool:
        if self.vent_transit > 0.0 or self._interact_latched:
            return False
        if self.nearest_vent() is not None:
            return True
        return self.hack_charges > 0 and self.hack_cooldown <= 0.0 and self.nearest_hackable() is not None

    def _activate_interact(self) -> None:
        """One context-sensitive verb: enter a vent, or hack a device."""

        vent = self.nearest_vent()
        if vent is not None:
            self.vent_transit = VENT_TRANSIT_SECONDS
            self._vent_destination = vent.exit_position.copy()
            self.vent_uses += 1
            self.velocity[:] = 0.0
            self.events.append(SimEvent("vent_enter", tuple(self.player)))
            return
        if self.hack_charges <= 0 or self.hack_cooldown > 0.0:
            return
        device = self.nearest_hackable()
        if device is None:
            return
        self.hack_charges -= 1
        self.hacks_used += 1
        self.hack_cooldown = HACK_COOLDOWN_SECONDS
        if device.kind == "camera":
            device.hacked_for = HACK_CAMERA_DISABLE_SECONDS
            for camera in self.level.cameras:
                if int(camera.camera_id) == device.target_id:
                    camera.disabled_for = max(camera.disabled_for, HACK_CAMERA_DISABLE_SECONDS)
            self.events.append(
                SimEvent("hack_camera", tuple(device.position), float(device.target_id))
            )
        elif device.kind == "door":
            device.hacked_for = HACK_DOOR_OVERRIDE_SECONDS
            # A pedestal controls one readable door; it is not a facility-wide
            # magic switch.  The target tile is authored and validated.
            door = next(
                (
                    candidate
                    for candidate in self.security_doors
                    if candidate.door_id == device.target_id
                    and (
                        device.target_tile is None
                        or candidate.tile == device.target_tile
                    )
                ),
                None,
            )
            if door is not None:
                door.lock_remaining = 0.0
                door.warning_remaining = 0.0
                door.forced_open_remaining = max(
                    door.forced_open_remaining, HACK_DOOR_OVERRIDE_SECONDS
                )
                self.events.append(
                    SimEvent("hack_door", tuple(device.position), float(door.door_id))
                )
            self._refresh_navigation_blocks()
        else:
            device.hacked_for = HACK_LIGHTS_SECONDS
            self.darkened_rooms[device.target_id] = HACK_LIGHTS_SECONDS
            self.events.append(
                SimEvent("hack_lights", tuple(device.position), float(device.target_id))
            )
        device.cooldown = device.hacked_for + 4.0
        self.events.append(SimEvent("hack", tuple(device.position), float(device.hacked_for)))

    def _update_hacks(self, dt: float) -> None:
        for device in self.hackable:
            device.hacked_for = max(0.0, device.hacked_for - dt)
            device.cooldown = max(0.0, device.cooldown - dt)
        for room_id in list(self.darkened_rooms):
            self.darkened_rooms[room_id] = max(0.0, self.darkened_rooms[room_id] - dt)
            if self.darkened_rooms[room_id] <= 0.0:
                del self.darkened_rooms[room_id]

    def _update_hacking(self, dt: float) -> None:
        """Terminal links pause while the runner is physically inside a duct."""

        if self.vent_transit > 0.0:
            self._active_hack = None
            return
        super()._update_hacking(dt)

    def _check_extraction(self) -> None:
        if self.vent_transit <= 0.0:
            super()._check_extraction()

    def _room_at(self, position: np.ndarray) -> int:
        tile_x, tile_y = world_to_tile(position)
        for room in self.level.rooms:
            if room.x <= tile_x < room.x + room.width and room.y <= tile_y < room.y + room.height:
                return int(room.room_id)
        return -1

    def visible(self, origin, facing, target, *, distance: float, cosine: float) -> bool:
        """Shared sight predicate for darkness and runner duct transit.

        Overriding here rather than in ``simulation.py`` keeps the frozen
        Env-v2 contract byte-identical. Every caller benefits automatically:
        guard awareness, the operative observation, and the rendered cone all
        route through this one predicate, so a hacked room cannot look bright
        while behaving dark, or the reverse.
        """

        target_array = np.asarray(target)
        is_runner = target is self.player or np.shares_memory(target_array, self.player)
        if is_runner and self.vent_transit > 0.0:
            return False
        if self.darkened_rooms and (
            self._room_at(np.asarray(origin)) in self.darkened_rooms
            or self._room_at(target_array) in self.darkened_rooms
        ):
            distance = distance * HACK_LIGHTS_VISION_SCALE
        return super().visible(origin, facing, target, distance=distance, cosine=cosine)

    def guard_vision_scale(self, guard: Guard) -> float:
        """Darkened rooms shorten a guard effective sight envelope."""

        return self.security_vision_scale(guard.position)

    def security_vision_scale(self, origin: np.ndarray) -> float:
        """Presentation-equivalent scale for any guard or camera cone."""

        return (
            HACK_LIGHTS_VISION_SCALE
            if (
                self._room_at(np.asarray(origin)) in self.darkened_rooms
                or self._room_at(self.player) in self.darkened_rooms
            )
            else 1.0
        )

    # -- operative field systems -------------------------------------------

    def deploy_field_sensor(self, guard: Guard) -> bool:
        """Place a non-lethal sensor that reports a crossing.

        Sensors never damage. They convert operative time into map knowledge,
        which is the coverage a strictly slower team needs.
        """

        if self.sensor_charges.get(guard.guard_id, 0) <= 0:
            return False
        self.sensor_charges[guard.guard_id] -= 1
        self.field_sensors.append(
            FieldSensor(
                self._next_sensor_id,
                guard.guard_id,
                guard.position.copy(),
                FIELD_SENSOR_ARM_SECONDS,
                FIELD_SENSOR_LIFETIME_SECONDS,
            )
        )
        self._next_sensor_id += 1
        self.events.append(SimEvent("sensor_deployed", tuple(guard.position), FIELD_SENSOR_RADIUS))
        return True

    def _update_field_sensors(self, dt: float) -> None:
        for sensor in list(self.field_sensors):
            sensor.armed_in = max(0.0, sensor.armed_in - dt)
            sensor.lifetime -= dt
            if sensor.lifetime <= 0.0:
                self.field_sensors.remove(sensor)
                continue
            if sensor.armed_in > 0.0 or sensor.triggered:
                continue
            if (
                self.vent_transit <= 0.0
                and norm(sensor.position - self.player) <= FIELD_SENSOR_RADIUS
                and self.line_of_sight(sensor.position, self.player)
            ):
                sensor.triggered = True
                self.events.append(SimEvent("sensor_trip", tuple(sensor.position), FIELD_SENSOR_RADIUS))
                # A trip is information, not damage: it seeds the heard estimate.
                for guard in self.level.guards:
                    state = self.operative_states[guard.guard_id]
                    state.heard_position = sensor.position.copy()
                    state.heard_confidence = max(state.heard_confidence, 0.7)

    def _activate_decoy(self, action: RunnerActionV2) -> None:
        if self.decoy_charges <= 0 or self.decoy_cooldown > 0.0:
            return
        direction = MOVE_DIRECTIONS[action.move] if action.move else self.heading
        direction = unit(direction)
        landing = self.player.copy()
        # A crouched throw is quieter but does not carry as far: the quiet route
        # trades reach for concealment here exactly as it does everywhere else.
        throw = DECOY_THROW_DISTANCE * (DECOY_CROUCH_THROW_SCALE if self.crouching else 1.0)
        for distance in np.arange(throw, -0.1, -TILE_SIZE / 2):
            candidate = self.player + direction * distance
            if self._can_occupy(candidate, 5.0):
                landing = candidate.astype(np.float32)
                break
        self.decoy_charges -= 1
        self.decoys_used += 1
        self.decoy_cooldown = 0.8
        self.decoys.append(Decoy(self._next_decoy_id, landing, DECOY_LIFETIME_SECONDS, 0.0))
        self._next_decoy_id += 1
        self.events.append(SimEvent("decoy_deployed", tuple(landing), DECOY_NOISE_RADIUS))

    def _update_decoys(self, dt: float) -> None:
        active: list[Decoy] = []
        for decoy in self.decoys:
            decoy.lifetime -= dt
            decoy.pulse_cooldown -= dt
            if decoy.pulse_cooldown <= 0.0:
                self._broadcast_noise_at(decoy.position, DECOY_NOISE_RADIUS, source="decoy")
                # Lure: hold the estimate at the decoy so operatives commit to
                # it, instead of a single ping they immediately discard.
                for guard in self.level.guards:
                    if norm(guard.position - decoy.position) <= DECOY_LURE_RADIUS:
                        state = self.operative_states[guard.guard_id]
                        state.heard_position = decoy.position.copy()
                        state.heard_confidence = max(state.heard_confidence, 0.85)
                        state.lure_remaining = DECOY_LURE_SECONDS
                decoy.pulse_cooldown = DECOY_PULSE_INTERVAL_SECONDS
                self.events.append(SimEvent("decoy_pulse", tuple(decoy.position), DECOY_NOISE_RADIUS))
            if decoy.lifetime > 0.0:
                active.append(decoy)
        self.decoys = active

    def _broadcast_noise_at(self, position: np.ndarray, radius: float, *, source: str) -> None:
        for guard in self.level.guards:
            distance = norm(guard.position - position)
            if distance > radius:
                continue
            state = self.operative_states[guard.guard_id]
            state.heard_position = position.copy()
            state.heard_confidence = max(state.heard_confidence, 1.0 - distance / max(1.0, radius))
            if not self.external_security and guard.mode != GuardMode.CHASE:
                guard.mode = GuardMode.INVESTIGATE
                guard.mode_seconds = 2.6
                guard.last_known = position.copy()
                guard.patrol_pause_seconds = 0.0
                guard.stimulus = source

    def _update_guards(self, dt: float) -> None:
        """Apply orders and crouch scaling before any detection transition."""

        if self.external_security and self._pending_security_orders:
            self._apply_pending_orders()
        self._seen_by_guard = False
        for guard in self.level.guards:
            guard.hit_cooldown = max(0.0, guard.hit_cooldown - dt)
            guard.radio_jammed_for = max(0.0, guard.radio_jammed_for - dt)
            has_sight = self.visible(
                guard.position,
                guard.facing,
                self.player,
                distance=GUARD_VISION_BASE_DISTANCE
                + GUARD_VISION_DISTANCE_PER_ALERT * self.alert_tier,
                cosine=GUARD_VISION_COSINE,
            )
            if has_sight:
                grace = max(0.38, DETECTION_GRACE_SECONDS - 0.05 * self.alert_tier)
                gain = dt / grace
                if self.crouching:
                    gain *= CROUCH_AWARENESS_SCALE
                guard.awareness = min(1.0, guard.awareness + gain)
                guard.last_known = self.player.copy()
                if guard.awareness >= 0.18 and guard.mode in (
                    GuardMode.PATROL,
                    GuardMode.RETURN,
                ):
                    guard.mode = GuardMode.SUSPICIOUS
                    guard.mode_seconds = max(guard.mode_seconds, 0.7)
                    guard.stimulus = "eye"
            else:
                guard.awareness = max(0.0, guard.awareness - 1.8 * dt)

            sees_player = guard.awareness >= 1.0 and has_sight
            if sees_player:
                self._seen_by_guard = True
                if guard.mode != GuardMode.CHASE:
                    self.detections += 1
                    self.events.append(SimEvent("detected", tuple(self.player)))
                guard.mode = GuardMode.CHASE
                guard.mode_seconds = (
                    2.6 * GUARD_SEARCH_DURATION_MULTIPLIERS[int(guard.grade)]
                )
                guard.last_known = self.player.copy()
                guard.stimulus = "eye"
                if guard.radio_jammed_for <= 0.0:
                    self._share_alert(guard)
            elif guard.mode == GuardMode.CHASE:
                guard.mode_seconds -= dt
                if guard.mode_seconds <= 0.0:
                    guard.mode = GuardMode.SEARCH
                    guard.mode_seconds = (
                        3.5 * GUARD_SEARCH_DURATION_MULTIPLIERS[int(guard.grade)]
                    )
            elif guard.mode in (
                GuardMode.SUSPICIOUS,
                GuardMode.INVESTIGATE,
                GuardMode.SEARCH,
            ):
                guard.mode_seconds -= dt
                if guard.mode_seconds <= 0.0:
                    guard.mode = GuardMode.RETURN
                    guard.stimulus = "patrol"
                    self.events.append(SimEvent("guard_clear", tuple(guard.position)))

            if guard.mode == GuardMode.CHASE:
                target = self.player
                speed = PLAYER_SPEED * GUARD_CHASE_SPEED_RATIOS[int(guard.grade)]
            elif guard.mode == GuardMode.SUSPICIOUS:
                target, speed = guard.last_known, 0.0
            elif guard.mode in (GuardMode.INVESTIGATE, GuardMode.SEARCH):
                target, speed = guard.last_known, 69.0
            else:
                target, speed = guard.patrol[guard.patrol_index], 54.0
                if guard.patrol_pause_seconds > 0.0:
                    guard.patrol_pause_seconds = max(
                        0.0, guard.patrol_pause_seconds - dt
                    )
                    speed = 0.0
                    scan_direction = (
                        1.0
                        if (guard.guard_id + guard.patrol_index) % 2 == 0
                        else -1.0
                    )
                    guard.facing += scan_direction * 0.72 * dt
                    if guard.patrol_pause_seconds <= 0.0:
                        guard.patrol_index = (guard.patrol_index + 1) % len(
                            guard.patrol
                        )
                        target = guard.patrol[guard.patrol_index]
                elif norm(target - guard.position) < 14.0:
                    guard.patrol_pause_seconds = GUARD_PATROL_DWELL_SECONDS[
                        int(guard.grade)
                    ]
                    speed = 0.0

            if guard.mode != GuardMode.CHASE:
                speed *= GUARD_GRADE_SPEED_MULTIPLIERS[int(guard.grade)]
            self._move_agent(guard, target, speed, dt)
            if (
                guard.mode == GuardMode.SEARCH
                and norm(guard.last_known - guard.position) < 18.0
            ):
                turn = 1.0 if guard.guard_id % 2 == 0 else -1.0
                guard.facing += turn * 1.45 * dt
            in_tackle_range = (
                guard.mode == GuardMode.CHASE
                and self.vent_transit <= 0.0
                and norm(guard.position - self.player) <= 20.0
            )
            if (
                in_tackle_range
                and guard.hit_cooldown <= 0.0
                and self.damage_cooldown <= 0.0
            ):
                guard.attack_windup += dt
                if guard.attack_windup >= GUARD_STRIKE_WINDUP_SECONDS:
                    guard.hit_cooldown = 1.8
                    guard.attack_windup = 0.0
                    self._damage(guard.position, source_kind="guard")
                    guard.mode = GuardMode.SEARCH
                    guard.mode_seconds = (
                        1.8 * GUARD_SEARCH_DURATION_MULTIPLIERS[int(guard.grade)]
                    )
                    guard.awareness = 0.0
                    guard.last_known = self.player.copy()
                    guard.stimulus = "eye"
            else:
                guard.attack_windup = max(0.0, guard.attack_windup - 2.5 * dt)

    def _apply_pending_orders(self) -> None:
        orders, self._pending_security_orders = self._pending_security_orders, {}
        for guard in self.level.guards:
            order = orders.get(guard.guard_id)
            if order is None:
                continue
            state = self.operative_states[guard.guard_id]
            direct_sight = self.visible(
                guard.position,
                guard.facing,
                self.player,
                distance=GUARD_VISION_BASE_DISTANCE
                + GUARD_VISION_DISTANCE_PER_ALERT * self.alert_tier,
                cosine=GUARD_VISION_COSINE,
            )
            target = self._valid_security_target(
                order.target.copy() if order.target is not None else guard.last_known.copy()
            )
            intent = order.intent
            use_ability = order.use_ability
            message = order.message
            # Decoys remain binding for externally controlled security.  A
            # policy cannot inspect a lure and immediately overwrite it with a
            # hidden-state-perfect pursuit order.
            if state.lure_remaining > 0.0 and not direct_sight:
                intent = SecurityIntent.INVESTIGATE
                target = self._valid_security_target(state.heard_position.copy())
                use_ability = False
                message = RadioMessage.NONE

            if intent == SecurityIntent.PINCER:
                # The selected target is already a public escape-route cutoff.
                # Never rotate it around the hidden live runner heading.
                target = self.pincer_station(guard, target)
            elif intent == SecurityIntent.SEAL:
                # Seal the explicitly selected public door/cutoff rather than a
                # route derived from the hidden mission objective.
                self.request_predictive_seal(target)

            state.current_order = SecurityOrder(
                intent, target.copy(), message, use_ability
            )
            if direct_sight:
                # Motor reflexes own confirmed sight.  In particular, HOLD may
                # not demote CHASE every tactical step and create repeatable
                # "new" detections for reward or telemetry.
                guard.last_known = self.player.copy()
            elif intent == SecurityIntent.PATROL:
                guard.mode = GuardMode.RETURN
            elif intent == SecurityIntent.HOLD:
                guard.mode = GuardMode.SUSPICIOUS
                guard.mode_seconds = 0.3
                guard.last_known = target
            elif intent == SecurityIntent.PURSUE and guard.awareness >= 1.0:
                guard.mode = GuardMode.CHASE
                guard.mode_seconds = max(guard.mode_seconds, 1.0)
                guard.last_known = target
            elif intent == SecurityIntent.PINCER:
                guard.mode = GuardMode.INVESTIGATE
                guard.mode_seconds = max(guard.mode_seconds, 0.9)
                guard.last_known = target
            else:
                guard.mode = (
                    GuardMode.SEARCH
                    if intent == SecurityIntent.SEARCH
                    else GuardMode.INVESTIGATE
                )
                guard.mode_seconds = 0.45
                guard.last_known = target
            guard.patrol_pause_seconds = 0.0
            guard.stimulus = "policy"
            if message != RadioMessage.NONE:
                self._transmit_radio(guard, message, target)
            if intent == SecurityIntent.INTERCEPT:
                self._request_nearest_door_lock(target)

    # -- coordinated security ----------------------------------------------

    def escape_route_cutoffs(
        self, contact: np.ndarray, *, limit: int = 4
    ) -> list[np.ndarray]:
        """Return public doorway cutoffs around a perceived contact.

        Facility topology is known to security.  The contact may be direct,
        heard, remembered, or radio-shared; this routine never reads the
        runner's hidden objective, velocity, or heading.
        """

        room_id = self._room_at(np.asarray(contact, dtype=np.float32))
        doors = [
            door
            for door in self.level.doors
            if room_id < 0 or room_id in (int(door.room_a), int(door.room_b))
        ]
        if not doors:
            doors = list(self.level.doors)
        candidates: list[tuple[float, int, int, np.ndarray]] = []
        rooms = {int(room.room_id): room for room in self.level.rooms}
        for door in doors:
            other_room = (
                int(door.room_b)
                if int(door.room_a) == room_id
                else int(door.room_a)
            )
            center = tile_center(door.tile)
            room = rooms.get(other_room)
            if room is not None:
                room_center = tile_center(
                    (
                        int(room.x + room.width // 2),
                        int(room.y + room.height // 2),
                    )
                )
                point = center + unit(room_center - center) * TILE_SIZE
            else:
                point = center
            point = self._valid_security_target(point.astype(np.float32))
            candidates.append(
                (norm(point - contact), int(door.tile[1]), int(door.tile[0]), point)
            )
        candidates.sort(key=lambda item: item[:3])
        return [point.copy() for *_order, point in candidates[: max(1, int(limit))]]

    def request_predictive_seal(
        self, selected_target: np.ndarray | None = None
    ) -> bool:
        """Seal the eligible door nearest a policy-selected public target.

        Every guarantee from the reactive lock still holds, because this only
        chooses *which* eligible door to ask for and then defers to the same
        request path: only redundant room-graph edges are eligible, the door
        telegraphs before it closes, an occupied door never closes, and the
        runner keeps both a pulse override and a door hack.
        """

        if self.security_door_cooldown > 0.0 or not self.security_doors:
            return False
        # Never seal a door the runner is already standing in or beside; that is
        # the case that feels arbitrary rather than outplayed.
        eligible = [
            door
            for door in self.security_doors
            if norm(tile_center(door.tile) - self.player) >= CHOKEPOINT_MIN_RUNNER_DISTANCE
        ]
        if not eligible:
            return False
        if selected_target is None:
            selected_target = tile_center(eligible[0].tile)
        selected_target = self._valid_security_target(
            np.asarray(selected_target, dtype=np.float32)
        )
        target = min(
            eligible,
            key=lambda door: (
                norm(tile_center(door.tile) - selected_target),
                door.door_id,
            ),
        )
        if self._request_nearest_door_lock(tile_center(target.tile)):
            self.security_door_cooldown = max(
                self.security_door_cooldown, CHOKEPOINT_TEAM_COOLDOWN_SECONDS
            )
            self.events.append(SimEvent("chokepoint_seal", tuple(tile_center(target.tile))))
            return True
        return False

    def pincer_station(self, guard: Guard, contact: np.ndarray) -> np.ndarray:
        """Validate an explicit public cutoff without consulting hidden state."""

        del guard
        return self._valid_security_target(np.asarray(contact, dtype=np.float32))

    def _valid_security_target(self, target: np.ndarray) -> np.ndarray:
        """Project policy waypoints into the navigable world deterministically."""

        maximum = np.asarray(
            (self.level.world_width - TILE_SIZE * 0.5, self.level.world_height - TILE_SIZE * 0.5),
            dtype=np.float32,
        )
        clipped = np.clip(
            np.nan_to_num(target, nan=TILE_SIZE * 0.5, posinf=maximum, neginf=TILE_SIZE * 0.5),
            TILE_SIZE * 0.5,
            maximum,
        ).astype(np.float32)
        if self._can_occupy(clipped, 7.0):
            return clipped
        origin = world_to_tile(clipped)
        candidates: list[tuple[int, int]] = []
        for radius in range(1, 6):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    tile = (origin[0] + dx, origin[1] + dy)
                    if not (0 <= tile[1] < self.level.grid.shape[0] and 0 <= tile[0] < self.level.grid.shape[1]):
                        continue
                    candidate = tile_center(tile)
                    if self._can_occupy(candidate, 7.0):
                        candidates.append(tile)
            if candidates:
                candidates.sort(key=lambda tile: ((tile[0] - origin[0]) ** 2 + (tile[1] - origin[1]) ** 2, tile[1], tile[0]))
                return tile_center(candidates[0])
        return self.level.spawn.copy()

    def _share_alert(self, source: Guard) -> None:
        if not self.external_security:
            super()._share_alert(source)

    def _transmit_radio(self, sender: Guard, message: RadioMessage, position: np.ndarray) -> None:
        if sender.radio_jammed_for > 0.0:
            return
        transmission = RadioTransmission(sender.guard_id, message, position.copy(), self.elapsed_ticks)
        self.radio_log.append(transmission)
        self.radio_log = self.radio_log[-32:]
        self.events.append(SimEvent("radio_message", tuple(sender.position), float(message)))
        for guard in self.level.guards:
            if guard is sender or guard.radio_jammed_for > 0.0 or norm(guard.position - sender.position) > 380.0:
                continue
            state = self.operative_states[guard.guard_id]
            state.heard_position = position.copy()
            state.heard_confidence = max(state.heard_confidence, 0.9)
            state.radio_assists += 1

    def _update_operative_state(self, dt: float) -> None:
        for state in self.operative_states.values():
            state.heard_confidence = max(0.0, state.heard_confidence - 0.24 * dt)
            state.lure_remaining = max(0.0, state.lure_remaining - dt)
            state.weapon_cooldown = max(0.0, state.weapon_cooldown - dt)

    def _apply_field_abilities(self) -> None:
        """Route the shared ability bit by role.

        Suppressors already own the telegraphed shock round. Patrol and
        Interceptor operatives now spend the same bit on a non-lethal sensor,
        so the ability slot is meaningful for the whole team rather than dead
        for three of five members.
        """

        for guard in self.level.guards:
            state = self.operative_states[guard.guard_id]
            if not state.current_order.use_ability:
                continue
            if state.role == GuardRole.SUPPRESSOR:
                continue
            if self.deploy_field_sensor(guard):
                state.current_order = SecurityOrder(
                    state.current_order.intent,
                    state.current_order.target,
                    state.current_order.message,
                    False,
                )

    def _update_suppressors(self, dt: float) -> None:
        for guard in self.level.guards:
            state = self.operative_states[guard.guard_id]
            if state.role != GuardRole.SUPPRESSOR:
                continue
            order = state.current_order
            distance = norm(self.player - guard.position)
            legal = (
                order.use_ability
                and state.weapon_cooldown <= 0.0
                and SUPPRESSOR_MIN_RANGE <= distance <= SUPPRESSOR_MAX_RANGE
                and self.visible(guard.position, guard.facing, self.player, distance=SUPPRESSOR_MAX_RANGE, cosine=-0.15)
            )
            if not legal:
                state.aim_progress = max(0.0, state.aim_progress - 2.5 * dt)
                if state.aim_progress <= 0.0:
                    state.aim_target = None
                continue
            if state.aim_target is None:
                state.aim_target = self.player.copy()
                self.events.append(SimEvent("suppressor_aim", tuple(guard.position), guard.guard_id))
            state.aim_progress += dt
            if state.aim_progress < SUPPRESSOR_AIM_SECONDS:
                continue
            direction = unit(state.aim_target - guard.position)
            if norm(direction) > 0.0 and self._shot_clear(guard, state.aim_target):
                origin = guard.position + direction * 11.0
                self.projectiles.append(
                    ShockProjectile(
                        self._next_projectile_id,
                        origin.astype(np.float32),
                        direction * SUPPRESSOR_PROJECTILE_SPEED,
                        guard.guard_id,
                        SUPPRESSOR_PROJECTILE_LIFETIME_SECONDS,
                    )
                )
                self._next_projectile_id += 1
                self.events.append(SimEvent("suppressor_fire", tuple(origin), guard.guard_id))
            state.aim_progress = 0.0
            state.aim_target = None
            state.weapon_cooldown = SUPPRESSOR_COOLDOWN_SECONDS

    def _shot_clear(self, source: Guard, target: np.ndarray) -> bool:
        if not self.line_of_sight(source.position, target):
            return False
        segment = target - source.position
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 1e-6:
            return False
        for guard in self.level.guards:
            if guard is source:
                continue
            t = float(np.clip(np.dot(guard.position - source.position, segment) / length_sq, 0.0, 1.0))
            closest = source.position + segment * t
            if norm(guard.position - closest) < 10.0:
                return False
        return True

    def _update_projectiles(self, dt: float) -> None:
        active: list[ShockProjectile] = []
        for projectile in self.projectiles:
            projectile.lifetime -= dt
            candidate = projectile.position + projectile.velocity * dt
            if projectile.lifetime <= 0.0 or not self._can_occupy(candidate, SUPPRESSOR_PROJECTILE_RADIUS):
                self.events.append(SimEvent("projectile_impact", tuple(projectile.position)))
                continue
            projectile.position = candidate.astype(np.float32)
            if (
                self.vent_transit <= 0.0
                and norm(projectile.position - self.player)
                <= PLAYER_RADIUS + SUPPRESSOR_PROJECTILE_RADIUS
            ):
                self._damage(projectile.position, source_kind="guard")
                self.events.append(SimEvent("projectile_hit", tuple(self.player), projectile.source_guard_id))
                continue
            active.append(projectile)
        self.projectiles = active

    def _damage(self, source: np.ndarray, *, source_kind: str = "unknown") -> None:
        """Duct transit removes the runner from the playable floor."""

        if self.vent_transit > 0.0:
            return
        super()._damage(source, source_kind=source_kind)

    def _request_nearest_door_lock(self, target: np.ndarray) -> bool:
        if self.security_door_cooldown > 0.0 or any(door.locked or door.warning_remaining > 0.0 for door in self.security_doors):
            return False
        candidates = [door for door in self.security_doors if door.forced_open_remaining <= 0.0]
        candidates.sort(key=lambda door: norm(tile_center(door.tile) - target))
        for door in candidates:
            center = tile_center(door.tile)
            occupied = norm(self.player - center) < 28.0 or any(norm(guard.position - center) < 24.0 for guard in self.level.guards)
            if occupied:
                continue
            door.warning_remaining = SECURITY_DOOR_WARNING_SECONDS
            self.security_door_cooldown = SECURITY_DOOR_TEAM_COOLDOWN_SECONDS
            self.events.append(SimEvent("door_warning", tuple(center), door.door_id))
            return True
        return False

    def _update_security_doors(self, dt: float) -> None:
        self.security_door_cooldown = max(0.0, self.security_door_cooldown - dt)
        changed = False
        for door in self.security_doors:
            was_locked = door.locked
            door.forced_open_remaining = max(0.0, door.forced_open_remaining - dt)
            if door.warning_remaining > 0.0:
                door.warning_remaining = max(0.0, door.warning_remaining - dt)
                if door.warning_remaining <= 0.0:
                    center = tile_center(door.tile)
                    if norm(self.player - center) >= 28.0:
                        door.lock_remaining = SECURITY_DOOR_LOCK_SECONDS
                        self.events.append(SimEvent("door_locked", tuple(center), door.door_id))
            else:
                door.lock_remaining = max(0.0, door.lock_remaining - dt)
            if was_locked != door.locked:
                changed = True
                if not door.locked:
                    self.events.append(SimEvent("door_opened", tuple(tile_center(door.tile)), door.door_id))
        if changed:
            self._refresh_navigation_blocks()

    def _refresh_navigation_blocks(self) -> None:
        locked = {door.tile for door in getattr(self, "security_doors", ()) if door.locked}
        if locked == getattr(self, "_v2_locked_tiles", set()) and hasattr(self, "_base_blocked_tiles"):
            return
        self._v2_locked_tiles = locked
        base = getattr(self, "_base_blocked_tiles", set(self._blocked_tiles))
        self._blocked_tiles = set(base) | locked
        self._nav_maps.clear()
        self._guard_waypoints.clear()
        self._drone_waypoints.clear()

    def _activate_pulse(self) -> None:
        before = self.pulse_charges
        super()._activate_pulse()
        if self.pulse_charges == before:
            return
        for door in self.security_doors:
            if norm(tile_center(door.tile) - self.player) <= PULSE_RADIUS:
                door.forced_open_remaining = SECURITY_DOOR_FORCED_OPEN_SECONDS
                door.warning_remaining = 0.0
                door.lock_remaining = 0.0
                self.events.append(SimEvent("door_forced_open", tuple(tile_center(door.tile)), door.door_id))
        self._refresh_navigation_blocks()

    @property
    def incoming_projectile_pressure(self) -> float:
        if not self.projectiles:
            return 0.0
        nearest = min(norm(projectile.position - self.player) for projectile in self.projectiles)
        return float(np.clip(1.0 - nearest / SUPPRESSOR_MAX_RANGE, 0.0, 1.0))

    def terminal_info(self) -> dict[str, float | int | bool | str]:
        info = super().terminal_info()
        info.update(
            {
                "contract": "GhostlineEnv-v2",
                "directive": self.directive.name.lower(),
                "directive_success": self.directive_completed,
                "directive_par_seconds": self.directive_par_seconds,
                "decoy_charges": self.decoy_charges,
                "decoys_used": self.decoys_used,
                "security_doors_locked": sum(door.locked for door in self.security_doors),
                "projectiles_active": len(self.projectiles),
            }
        )
        return info
