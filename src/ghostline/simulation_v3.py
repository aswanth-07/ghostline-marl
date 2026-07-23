from __future__ import annotations

from collections.abc import Mapping
import itertools
import math

import numpy as np

from ghostline.config import (
    PLAYER_GUARD_AUDIBLE_DISTANCE,
    PLAYER_RADIUS,
    PLAYER_SPEED,
    PULSE_RADIUS,
    SIM_HZ,
    TILE_SIZE,
)
from ghostline.config_v3 import (
    DECOY_LIFETIME_SECONDS,
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
from ghostline.simulation import GhostlineSimulation, MOVE_DIRECTIONS, norm, unit
from ghostline.types import Guard, GuardMode, SimEvent
from ghostline.types_v3 import (
    ContractDirective,
    Decoy,
    GuardRole,
    OperativeState,
    RadioMessage,
    RadioTransmission,
    RunnerActionV3,
    SecurityDoor,
    SecurityIntent,
    SecurityOrder,
    ShockProjectile,
)


class GhostlineSimulationV3(GhostlineSimulation):
    """Env-v3 deterministic extension with semantic security control.

    Classic simulation remains untouched.  This subclass owns every new state
    object so a v3 rollout can never silently change an Env-v2 replay.
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

    def reset(self, *, seed: int | None = None, tier: int | None = None) -> None:
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
        self._v3_locked_tiles: set[tuple[int, int]] = set()
        self._refresh_navigation_blocks()
        self.directive_par_seconds = self._speed_directive_par_seconds()

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
        mask = np.ones(72, dtype=np.int8)
        dash_available = self.dash_energy > 1.0
        pulse_available = self.pulse_charges > 0 and self.pulse_cooldown <= 0.0
        decoy_available = self.decoy_charges > 0 and self.decoy_cooldown <= 0.0
        for value in range(72):
            action = RunnerActionV3.decode(value)
            if action.dash and (not dash_available or action.move == 0):
                mask[value] = 0
            if action.pulse and not pulse_available:
                mask[value] = 0
            if action.decoy and not decoy_available:
                mask[value] = 0
        return mask

    def set_security_orders(self, orders: Mapping[int, SecurityOrder]) -> None:
        self._pending_security_orders = {
            int(guard_id): order
            for guard_id, order in orders.items()
            if int(guard_id) in self.operative_states
        }

    def _tick(self, action: RunnerActionV3, *, allow_pulse: bool) -> None:
        dt = 1.0 / SIM_HZ
        self.decoy_cooldown = max(0.0, self.decoy_cooldown - dt)
        self._update_security_doors(dt)
        self._update_decoys(dt)
        if action.decoy and not self._decoy_latched:
            self._activate_decoy(action)
            self._decoy_latched = True
        if not action.decoy:
            self._decoy_latched = False
        super()._tick(action, allow_pulse=allow_pulse)
        self._update_operative_state(dt)
        self._update_suppressors(dt)
        self._update_projectiles(dt)

    def _activate_decoy(self, action: RunnerActionV3) -> None:
        if self.decoy_charges <= 0 or self.decoy_cooldown > 0.0:
            return
        direction = MOVE_DIRECTIONS[action.move] if action.move else self.heading
        direction = unit(direction)
        landing = self.player.copy()
        for distance in np.arange(DECOY_THROW_DISTANCE, -0.1, -TILE_SIZE / 2):
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
        if self.external_security and self._pending_security_orders:
            self._apply_pending_orders()
        super()._update_guards(dt)

    def _apply_pending_orders(self) -> None:
        orders, self._pending_security_orders = self._pending_security_orders, {}
        for guard in self.level.guards:
            order = orders.get(guard.guard_id)
            if order is None:
                continue
            state = self.operative_states[guard.guard_id]
            target = self._valid_security_target(
                order.target.copy() if order.target is not None else guard.last_known.copy()
            )
            state.current_order = SecurityOrder(order.intent, target.copy(), order.message, order.use_ability)
            if order.intent == SecurityIntent.PATROL:
                guard.mode = GuardMode.RETURN
            elif order.intent == SecurityIntent.HOLD:
                guard.mode = GuardMode.SUSPICIOUS
                guard.mode_seconds = 0.3
                guard.last_known = target
            elif order.intent == SecurityIntent.PURSUE and guard.awareness >= 1.0:
                guard.mode = GuardMode.CHASE
                guard.mode_seconds = max(guard.mode_seconds, 1.0)
                guard.last_known = target
            else:
                guard.mode = GuardMode.SEARCH if order.intent == SecurityIntent.SEARCH else GuardMode.INVESTIGATE
                guard.mode_seconds = 0.45
                guard.last_known = target
            guard.patrol_pause_seconds = 0.0
            guard.stimulus = "policy"
            if order.message != RadioMessage.NONE:
                self._transmit_radio(guard, order.message, target)
            if order.intent == SecurityIntent.INTERCEPT:
                self._request_nearest_door_lock(target)

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
            state.weapon_cooldown = max(0.0, state.weapon_cooldown - dt)

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
            if norm(projectile.position - self.player) <= PLAYER_RADIUS + SUPPRESSOR_PROJECTILE_RADIUS:
                self._damage(projectile.position, source_kind="guard")
                self.events.append(SimEvent("projectile_hit", tuple(self.player), projectile.source_guard_id))
                continue
            active.append(projectile)
        self.projectiles = active

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
        if locked == getattr(self, "_v3_locked_tiles", set()) and hasattr(self, "_base_blocked_tiles"):
            return
        self._v3_locked_tiles = locked
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
                "contract": "GhostlineEnv-v3",
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
