from __future__ import annotations

from collections import deque
import math

import numpy as np

from ghostline.config import (
    CAMERA_VISION_COSINE,
    CAMERA_VISION_DISTANCE,
    DASH_DRAIN_PER_SECOND,
    DASH_ENERGY_MAX,
    DASH_RECHARGE_PER_SECOND,
    DASH_SPEED,
    DAMAGE_INVULNERABILITY_SECONDS,
    DETECTION_GRACE_SECONDS,
    DRONE_STRIKE_WINDUP_SECONDS,
    GUARD_GRADE_SPEED_MULTIPLIERS,
    GUARD_PATROL_DWELL_SECONDS,
    GUARD_SEARCH_DURATION_MULTIPLIERS,
    GUARD_STRIKE_WINDUP_SECONDS,
    GUARD_VISION_BASE_DISTANCE,
    GUARD_VISION_COSINE,
    GUARD_VISION_DISTANCE_PER_ALERT,
    HACK_RADIUS,
    PLAYER_RADIUS,
    PLAYER_FOOTSTEP_AUDIBLE_DISTANCE,
    PLAYER_GUARD_AUDIBLE_DISTANCE,
    PLAYER_PERCEPTION_DISTANCE,
    PLAYER_SPEED,
    PULSE_DISABLE_SECONDS,
    PULSE_RADIUS,
    SIM_HZ,
    TILE_SIZE,
    TRACE_MAX,
)
from ghostline.generation import LevelGenerator, tile_center, world_to_tile
from ghostline.types import Action, Drone, GeneratedLevel, Guard, GuardMode, SecurityIntel, SimEvent, Tile

MOVE_DIRECTIONS = np.asarray(
    (
        (0.0, 0.0),
        (0.0, -1.0),
        (0.70710677, -0.70710677),
        (1.0, 0.0),
        (0.70710677, 0.70710677),
        (0.0, 1.0),
        (-0.70710677, 0.70710677),
        (-1.0, 0.0),
        (-0.70710677, -0.70710677),
    ),
    dtype=np.float32,
)


def norm(vector: np.ndarray) -> float:
    # Every simulation vector is 2-D; avoiding np.linalg dispatch is material at
    # millions of guard/path calls during parallel training.
    return math.hypot(float(vector[0]), float(vector[1]))


def unit(vector: np.ndarray) -> np.ndarray:
    magnitude = norm(vector)
    return vector / magnitude if magnitude > 1e-6 else np.zeros(2, dtype=np.float32)


def angle_vector(angle: float) -> np.ndarray:
    return np.asarray((math.cos(angle), math.sin(angle)), dtype=np.float32)


class GhostlineSimulation:
    """Deterministic, renderer-free fixed-timestep Ghostline simulation."""

    def __init__(self, *, seed: int = 0, tier: int = 1):
        self.generator = LevelGenerator()
        self.seed = int(seed)
        self.tier = int(tier)
        self.level: GeneratedLevel
        self.reset(seed=seed, tier=tier)

    def reset(self, *, seed: int | None = None, tier: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        if tier is not None:
            self.tier = int(tier)
        self.level = self.generator.generate(seed=self.seed, tier=self.tier)
        self.player = self.level.spawn.copy()
        self.velocity = np.zeros(2, dtype=np.float32)
        self.heading = np.asarray((1.0, 0.0), dtype=np.float32)
        self.integrity = 3
        self.trace = 0.0
        self.trace_floor = 0.0
        self.max_trace = 0.0
        self.dash_energy = DASH_ENERGY_MAX
        self.pulse_charges = self.level.pulse_charges
        self.pulse_cooldown = 0.0
        self.damage_cooldown = 0.0
        self.data = 0
        self.elapsed_ticks = 0
        self.extracted = False
        self.terminated = False
        self.truncated = False
        self.fail_reason = "none"
        self.lockdown = False
        self.detections = 0
        self.damage_taken = 0
        self.damage_by_guard = 0
        self.damage_by_drone = 0
        self.pulses_used = 0
        self.hacks_completed = 0
        self.optional_data = 0
        self.distance_travelled = 0.0
        self.events: list[SimEvent] = []
        self.drones: list[Drone] = []
        self._blocked_tiles = {
            (prop.tile_x + dx, prop.tile_y + dy)
            for prop in self.level.props
            if prop.blocking
            for dx in range(prop.width)
            for dy in range(prop.height)
        }
        self._active_hack: int | None = None
        self.objective_terminal_id: int | None = None
        self._was_seen = False
        self._pulse_latched = False
        self._dash_latched = False
        self._drone_warning_emitted = False
        self._last_action = Action()
        self.explored = np.zeros_like(self.level.grid, dtype=bool)
        self._last_reveal_tile: tuple[int, int] | None = None
        self._seen_by_guard = False
        self._guard_waypoints: dict[int, tuple[tuple[int, int], np.ndarray]] = {}
        self._drone_waypoints: dict[int, tuple[tuple[int, int], np.ndarray]] = {}
        self._nav_maps: dict[tuple[int, int], np.ndarray] = {}
        self.security_intel: dict[tuple[str, int], SecurityIntel] = {}
        self._reveal_near_player()
        self._update_player_intel()

    @property
    def elapsed_seconds(self) -> float:
        return self.elapsed_ticks / SIM_HZ

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.level.mission_seconds - self.elapsed_seconds)

    @property
    def quota_met(self) -> bool:
        return self.data >= self.level.quota

    @property
    def alert_tier(self) -> int:
        return min(4, int(self.trace // 25.0))

    @property
    def active_hack_progress(self) -> float:
        if self._active_hack is None:
            return 0.0
        terminal = self.level.terminals[self._active_hack]
        return float(np.clip(terminal.progress / terminal.hack_seconds, 0.0, 1.0))

    def objective_terminal(self) -> Terminal | None:
        """Return the stable acquire-phase objective shared by HUD and policy.

        The objective is sticky to prevent equally attractive terminals from
        making the route marker oscillate as the runner crosses a tile boundary.
        Entering another terminal's link ring deliberately retargets it.
        """
        if self.quota_met:
            self.objective_terminal_id = None
            return None
        if self.objective_terminal_id is not None:
            for terminal in self.level.terminals:
                if terminal.terminal_id == self.objective_terminal_id and not terminal.completed:
                    return terminal
        remaining = [terminal for terminal in self.level.terminals if not terminal.completed]
        if not remaining:
            self.objective_terminal_id = None
            return None
        selected = min(
            remaining,
            key=lambda terminal: (norm(terminal.position - self.player) / max(1, terminal.value), terminal.terminal_id),
        )
        self.objective_terminal_id = selected.terminal_id
        return selected

    @property
    def context_hint(self) -> str:
        if self.quota_met:
            return "QUOTA SECURED  //  REACH THE GREEN EXTRACTION RELAY"
        if self._active_hack is not None:
            return "LINK ACTIVE  //  MOVE FREELY INSIDE THE RING"
        if self.damage_cooldown > 0.0:
            return "RECOVERING  //  BREAK LINE OF SIGHT"
        awareness = max(
            [camera.awareness for camera in self.level.cameras]
            + [guard.awareness for guard in self.level.guards]
            + [0.0]
        )
        if awareness > 0.05:
            return "SURVEILLANCE RISING  //  CROSS A WALL OR LEAVE THE CONE"
        if self.hacks_completed == 0 and self.elapsed_seconds < 22.0:
            return "ENTER AN AMBER RING TO LINK DATA"
        return ""

    def pop_events(self) -> list[SimEvent]:
        events = list(self.events)
        self.events.clear()
        return events

    def action_mask(self) -> np.ndarray:
        mask = np.ones(36, dtype=np.int8)
        dash_available = self.dash_energy >= 4.0
        pulse_available = self.pulse_charges > 0 and self.pulse_cooldown <= 0.0
        for value in range(36):
            move = value % 9
            dash = bool((value // 9) % 2)
            pulse = bool(value // 18)
            if dash and (not dash_available or move == 0):
                mask[value] = 0
            if pulse and not pulse_available:
                mask[value] = 0
        return mask

    def advance(self, action: Action, *, ticks: int = 1) -> None:
        if self.terminated or self.truncated:
            return
        for index in range(ticks):
            if self.terminated or self.truncated:
                break
            self._tick(action, allow_pulse=index == 0)
        self._last_action = action

    def _tick(self, action: Action, *, allow_pulse: bool) -> None:
        dt = 1.0 / SIM_HZ
        self.elapsed_ticks += 1
        if self.pulse_cooldown > 0.0:
            self.pulse_cooldown = max(0.0, self.pulse_cooldown - dt)
        self.damage_cooldown = max(0.0, self.damage_cooldown - dt)

        movement = MOVE_DIRECTIONS[action.move]
        dashing = action.dash and action.move != 0 and self.dash_energy > 0.0
        target_speed = DASH_SPEED if dashing else PLAYER_SPEED
        if action.move:
            self.heading = movement.copy()
            desired = movement * target_speed
            blend = 0.32 if dashing else 0.24
            self.velocity += (desired - self.velocity) * blend
        else:
            self.velocity *= 0.68
            if norm(self.velocity) < 1.0:
                self.velocity[:] = 0.0

        if dashing:
            self.dash_energy = max(0.0, self.dash_energy - DASH_DRAIN_PER_SECOND * dt)
            if not self._dash_latched:
                self.events.append(SimEvent("dash_noise", tuple(self.player), 185.0))
            self._dash_latched = True
            if self.elapsed_ticks % 8 == 0:
                self.events.append(SimEvent("dash", tuple(self.player)))
            self._broadcast_noise(radius=185.0)
        else:
            self._dash_latched = False
            self.dash_energy = min(DASH_ENERGY_MAX, self.dash_energy + DASH_RECHARGE_PER_SECOND * dt)

        previous = self.player.copy()
        self._move_player(self.velocity * dt)
        self.distance_travelled += norm(self.player - previous)
        player_tile = world_to_tile(self.player)
        # Refresh within 100 ms so furniture loses fog as soon as a moving
        # sightline reaches its exposed face, even before crossing a tile.
        if player_tile != self._last_reveal_tile or self.elapsed_ticks % 6 == 0:
            self._reveal_near_player()

        if allow_pulse and action.pulse and not self._pulse_latched:
            self._activate_pulse()
            self._pulse_latched = True
        if not action.pulse:
            self._pulse_latched = False

        self._update_cameras(dt)
        self._update_guards(dt)
        self._update_drones(dt)
        self._update_player_intel()
        self._update_hacking(dt)
        self._update_trace(dt)
        self._check_extraction()

        if self.remaining_seconds <= 0.0 and not self.extracted:
            self.truncated = True
            self.fail_reason = "contract_expired"
            self.events.append(SimEvent("failure", tuple(self.player)))
        if self.integrity <= 0:
            self.terminated = True
            self.fail_reason = "integrity_lost"
            self.events.append(SimEvent("failure", tuple(self.player)))

    def _move_player(self, delta: np.ndarray) -> None:
        candidate = self.player.copy()
        candidate[0] += delta[0]
        if self._can_occupy(candidate, PLAYER_RADIUS):
            self.player[0] = candidate[0]
        else:
            self.velocity[0] = 0.0
        candidate = self.player.copy()
        candidate[1] += delta[1]
        if self._can_occupy(candidate, PLAYER_RADIUS):
            self.player[1] = candidate[1]
        else:
            self.velocity[1] = 0.0

    def _can_occupy(self, position: np.ndarray, radius: float) -> bool:
        if not (radius <= position[0] < self.level.world_width - radius and radius <= position[1] < self.level.world_height - radius):
            return False
        for ox, oy in ((-radius, -radius), (radius, -radius), (-radius, radius), (radius, radius), (0.0, 0.0)):
            tile = (int((position[0] + ox) // TILE_SIZE), int((position[1] + oy) // TILE_SIZE))
            if tile in self._blocked_tiles:
                return False
            if not (0 <= tile[1] < self.level.grid.shape[0] and 0 <= tile[0] < self.level.grid.shape[1]):
                return False
            if int(self.level.grid[tile[1], tile[0]]) == 1:
                return False
        return True

    def _reveal_near_player(self) -> None:
        px, py = world_to_tile(self.player)
        self._last_reveal_tile = (px, py)
        for y in range(max(0, py - 7), min(self.level.grid.shape[0], py + 8)):
            for x in range(max(0, px - 7), min(self.level.grid.shape[1], px + 8)):
                if self.explored[y, x]:
                    continue
                if (x - px) ** 2 + (y - py) ** 2 > 49:
                    continue
                if self._tile_reached_by_player_sight(x, y):
                    self.explored[y, x] = True

    def _tile_reached_by_player_sight(self, x: int, y: int) -> bool:
        """Reveal an occluder's visible face without seeing through it.

        A centre-point ray correctly blocks detection, but it also terminates
        inside furniture and walls.  That made their own tiles permanently
        unexplored and the presentation fog greyed them out at every angle.
        For an occluding tile, a visible, nearer adjacent floor tile proves
        that the ray reached an exposed face.  Tiles behind the obstacle still
        fail the ordinary LOS check and remain hidden.
        """

        target = tile_center((x, y))
        if self.line_of_sight(self.player, target):
            return True
        occluding = (x, y) in self._blocked_tiles or int(self.level.grid[y, x]) == int(Tile.WALL)
        if not occluding:
            return False

        target_distance = norm(target - self.player)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            adjacent_x, adjacent_y = x + dx, y + dy
            if not (
                0 <= adjacent_y < self.level.grid.shape[0]
                and 0 <= adjacent_x < self.level.grid.shape[1]
            ):
                continue
            if (adjacent_x, adjacent_y) in self._blocked_tiles:
                continue
            if int(self.level.grid[adjacent_y, adjacent_x]) == int(Tile.WALL):
                continue
            adjacent = tile_center((adjacent_x, adjacent_y))
            if norm(adjacent - self.player) > target_distance + TILE_SIZE * 0.25:
                continue
            if self.line_of_sight(self.player, adjacent):
                return True
        return False

    def line_of_sight(self, origin: np.ndarray, target: np.ndarray) -> bool:
        ox, oy = float(origin[0]), float(origin[1])
        dx, dy = float(target[0]) - ox, float(target[1]) - oy
        distance = math.hypot(dx, dy)
        if distance <= 1e-6:
            return True
        inv = 1.0 / distance
        ux, uy = dx * inv, dy * inv
        steps = int(distance // 12.0)
        for index in range(1, steps + 1):
            sample_distance = index * 12.0
            tx = int((ox + ux * sample_distance) // TILE_SIZE)
            ty = int((oy + uy * sample_distance) // TILE_SIZE)
            if (tx, ty) in self._blocked_tiles or int(self.level.grid[ty, tx]) == 1:
                return False
        return True

    def player_can_see(
        self,
        position: np.ndarray,
        *,
        distance: float = PLAYER_PERCEPTION_DISTANCE,
    ) -> bool:
        """Use one literal player sight gate for UI and RL observations."""

        return norm(position - self.player) <= distance and self.line_of_sight(self.player, position)

    def player_can_hear_guard(self, guard: Guard) -> bool:
        """Return whether a guard earns an audibility-gated inference cue."""

        distance = norm(guard.position - self.player)
        alerted = guard.mode != GuardMode.PATROL and distance <= PLAYER_GUARD_AUDIBLE_DISTANCE
        footsteps = norm(guard.velocity) > 18.0 and distance <= PLAYER_FOOTSTEP_AUDIBLE_DISTANCE
        return alerted or footsteps

    def player_guard_audible_estimate(self, guard: Guard) -> tuple[np.ndarray, float] | None:
        """Return a deliberately coarse sound estimate, never hidden exact state."""

        if not self.player_can_hear_guard(guard):
            return None
        delta = guard.position - self.player
        distance = norm(delta)
        # Twelve bearing sectors and 48 px range bands communicate useful
        # direction without turning footsteps/radio calls into a wallhack.
        sector_width = math.tau / 12.0
        angle = round(math.atan2(float(delta[1]), float(delta[0])) / sector_width) * sector_width
        range_band = max(24.0, round(distance / 48.0) * 48.0)
        estimate = self.player + angle_vector(angle) * range_band
        audible_limit = (
            PLAYER_GUARD_AUDIBLE_DISTANCE
            if guard.mode != GuardMode.PATROL
            else PLAYER_FOOTSTEP_AUDIBLE_DISTANCE
        )
        strength = 1.0 - min(1.0, distance / max(1.0, audible_limit))
        return estimate.astype(np.float32), 0.10 + 0.30 * strength

    def _update_player_intel(self) -> None:
        """Snapshot only security actors currently inside true player LOS.

        Moving actors retain their last-seen snapshot after they leave sight;
        fixed cameras remain mapped.  This small deterministic intelligence
        layer prevents jarring disappearance while avoiding through-wall live
        tracking, and gives human/agent takeover an identical starting state.
        """

        tick = int(self.elapsed_ticks)
        for camera in self.level.cameras:
            if self.player_can_see(camera.position):
                self.security_intel[("camera", camera.camera_id)] = SecurityIntel(
                    "camera",
                    camera.camera_id,
                    camera.position.astype(np.float32).copy(),
                    np.zeros(2, dtype=np.float32),
                    float(camera.angle),
                    float(camera.awareness),
                    tick,
                )
        for guard in self.level.guards:
            if self.player_can_see(guard.position):
                self.security_intel[("guard", guard.guard_id)] = SecurityIntel(
                    "guard",
                    guard.guard_id,
                    guard.position.astype(np.float32).copy(),
                    guard.velocity.astype(np.float32).copy(),
                    float(guard.facing),
                    float(guard.mode) / float(max(GuardMode)),
                    tick,
                    guard.grade,
                )
        for drone in self.drones:
            if self.player_can_see(drone.position):
                self.security_intel[("drone", drone.drone_id)] = SecurityIntel(
                    "drone",
                    drone.drone_id,
                    drone.position.astype(np.float32).copy(),
                    np.zeros(2, dtype=np.float32),
                    float(drone.facing),
                    1.0,
                    tick,
                )

    def visible(self, origin: np.ndarray, facing: float, target: np.ndarray, *, distance: float, cosine: float) -> bool:
        delta = target - origin
        magnitude = norm(delta)
        if magnitude > distance or magnitude <= 1e-6:
            return False
        facing_x, facing_y = math.cos(facing), math.sin(facing)
        if (facing_x * float(delta[0]) + facing_y * float(delta[1])) / magnitude < cosine:
            return False
        return self.line_of_sight(origin, target)

    def _update_cameras(self, dt: float) -> None:
        for camera in self.level.cameras:
            if camera.disabled_for > 0.0:
                camera.disabled_for = max(0.0, camera.disabled_for - dt)
                camera.detected = False
                camera.awareness = max(0.0, camera.awareness - 3.0 * dt)
                continue
            camera.angle = camera.base_angle + math.sin(self.elapsed_seconds * camera.speed + camera.camera_id) * camera.sweep
            has_sight = self.visible(
                camera.position,
                camera.angle,
                self.player,
                distance=CAMERA_VISION_DISTANCE,
                cosine=CAMERA_VISION_COSINE,
            )
            camera.awareness = min(1.0, camera.awareness + dt / 0.45) if has_sight else max(0.0, camera.awareness - 2.8 * dt)
            camera.detected = camera.awareness >= 1.0

    def _update_guards(self, dt: float) -> None:
        self._seen_by_guard = False
        for guard in self.level.guards:
            guard.hit_cooldown = max(0.0, guard.hit_cooldown - dt)
            guard.radio_jammed_for = max(0.0, guard.radio_jammed_for - dt)
            has_sight = self.visible(
                guard.position,
                guard.facing,
                self.player,
                distance=GUARD_VISION_BASE_DISTANCE + GUARD_VISION_DISTANCE_PER_ALERT * self.alert_tier,
                cosine=GUARD_VISION_COSINE,
            )
            if has_sight:
                grace = max(0.38, DETECTION_GRACE_SECONDS - 0.05 * self.alert_tier)
                guard.awareness = min(1.0, guard.awareness + dt / grace)
                guard.last_known = self.player.copy()
                if guard.awareness >= 0.18 and guard.mode in (GuardMode.PATROL, GuardMode.RETURN):
                    # The guard visibly checks a partial sighting instead of
                    # walking its route while an awareness marker fills behind
                    # it. This also gives the runner a readable reaction beat.
                    guard.mode = GuardMode.SUSPICIOUS
                    guard.mode_seconds = max(guard.mode_seconds, 0.7)
                    guard.stimulus = "eye"
            else:
                guard.awareness = max(0.0, guard.awareness - 1.8 * dt)
            sees_player = guard.awareness >= 1.0
            if sees_player:
                self._seen_by_guard = True
                if guard.mode != GuardMode.CHASE:
                    self.detections += 1
                    self.events.append(SimEvent("detected", tuple(self.player)))
                guard.mode = GuardMode.CHASE
                guard.mode_seconds = 2.6 * GUARD_SEARCH_DURATION_MULTIPLIERS[int(guard.grade)]
                guard.last_known = self.player.copy()
                guard.stimulus = "eye"
                if guard.radio_jammed_for <= 0.0:
                    self._share_alert(guard)
            elif guard.mode == GuardMode.CHASE:
                guard.mode_seconds -= dt
                if guard.mode_seconds <= 0.0:
                    guard.mode = GuardMode.SEARCH
                    guard.mode_seconds = 3.5 * GUARD_SEARCH_DURATION_MULTIPLIERS[int(guard.grade)]
            elif guard.mode in (GuardMode.SUSPICIOUS, GuardMode.INVESTIGATE, GuardMode.SEARCH):
                guard.mode_seconds -= dt
                if guard.mode_seconds <= 0.0:
                    guard.mode = GuardMode.RETURN
                    guard.stimulus = "patrol"
                    self.events.append(SimEvent("guard_clear", tuple(guard.position)))

            if guard.mode == GuardMode.CHASE:
                target, speed = self.player, 84.0
            elif guard.mode == GuardMode.SUSPICIOUS:
                target, speed = guard.last_known, 0.0
            elif guard.mode in (GuardMode.INVESTIGATE, GuardMode.SEARCH):
                target, speed = guard.last_known, 69.0
            else:
                target, speed = guard.patrol[guard.patrol_index], 54.0
                if guard.patrol_pause_seconds > 0.0:
                    guard.patrol_pause_seconds = max(0.0, guard.patrol_pause_seconds - dt)
                    speed = 0.0
                    scan_direction = 1.0 if (guard.guard_id + guard.patrol_index) % 2 == 0 else -1.0
                    guard.facing += scan_direction * 0.72 * dt
                    if guard.patrol_pause_seconds <= 0.0:
                        guard.patrol_index = (guard.patrol_index + 1) % len(guard.patrol)
                        target = guard.patrol[guard.patrol_index]
                elif norm(target - guard.position) < 14.0:
                    guard.patrol_pause_seconds = GUARD_PATROL_DWELL_SECONDS[int(guard.grade)]
                    speed = 0.0

            speed *= GUARD_GRADE_SPEED_MULTIPLIERS[int(guard.grade)]
            self._move_agent(guard, target, speed, dt)
            if guard.mode == GuardMode.SEARCH and norm(guard.last_known - guard.position) < 18.0:
                turn = 1.0 if guard.guard_id % 2 == 0 else -1.0
                guard.facing += turn * 1.45 * dt
            in_tackle_range = guard.mode == GuardMode.CHASE and norm(guard.position - self.player) <= 20.0
            if in_tackle_range and guard.hit_cooldown <= 0.0 and self.damage_cooldown <= 0.0:
                guard.attack_windup += dt
                if guard.attack_windup >= GUARD_STRIKE_WINDUP_SECONDS:
                    guard.hit_cooldown = 1.8
                    guard.attack_windup = 0.0
                    self._damage(guard.position, source_kind="guard")
                    # A successful tackle creates a genuine escape window. The
                    # guard searches the impact point instead of remaining glued
                    # to the runner throughout the global damage grace period.
                    guard.mode = GuardMode.SEARCH
                    guard.mode_seconds = 1.8 * GUARD_SEARCH_DURATION_MULTIPLIERS[int(guard.grade)]
                    guard.awareness = 0.0
                    guard.last_known = self.player.copy()
                    guard.stimulus = "eye"
            else:
                guard.attack_windup = max(0.0, guard.attack_windup - 2.5 * dt)

    def _move_agent(self, guard: Guard, target: np.ndarray, speed: float, dt: float) -> None:
        goal_tile = world_to_tile(target)
        cached = self._guard_waypoints.get(guard.guard_id)
        if cached is None or cached[0] != goal_tile:
            next_target = self._next_path_point(guard.position, target)
            self._guard_waypoints[guard.guard_id] = (goal_tile, next_target.copy())
        else:
            next_target = cached[1]
        direction = unit(next_target - guard.position)
        separation = np.zeros(2, dtype=np.float32)
        # Separation looks natural in rooms but is harmful at one-tile doors:
        # the lateral force can pin two otherwise valid paths to the jamb.
        if not self._is_navigation_choke(world_to_tile(guard.position)) and not self._is_navigation_choke(world_to_tile(next_target)):
            for other in self.level.guards:
                if other is guard:
                    continue
                offset = guard.position - other.position
                distance = norm(offset)
                if 0.01 < distance < 25.0:
                    separation += unit(offset) * ((25.0 - distance) / 25.0)
        if norm(separation) > 0.0:
            direction = unit(direction + separation * 0.55)
        if norm(direction) > 0.0:
            guard.facing = math.atan2(float(direction[1]), float(direction[0]))
        guard.velocity = direction * speed
        previous = guard.position.copy()
        self._move_mobile(guard.position, guard.velocity * dt, radius=7.0)
        moved = norm(guard.position - previous)
        expected = speed * dt
        guard.stuck_seconds = guard.stuck_seconds + dt if expected > 0.0 and moved < expected * 0.2 else 0.0
        if norm(next_target - guard.position) < 5.0 or guard.stuck_seconds > 0.22:
            self._guard_waypoints.pop(guard.guard_id, None)
            if guard.stuck_seconds > 0.45:
                # Re-centre smoothly inside the current navigation cell. This
                # clears corner drift without a visible teleport or a change to
                # the deterministic replay contract.
                center = tile_center(world_to_tile(guard.position))
                correction = unit(center - guard.position) * min(3.0, norm(center - guard.position))
                self._move_mobile(guard.position, correction, radius=7.0)
                guard.stuck_seconds = 0.0

    def _next_path_point(self, start: np.ndarray, target: np.ndarray) -> np.ndarray:
        start_tile, target_tile = world_to_tile(start), world_to_tile(target)
        if start_tile == target_tile or self._segment_navigable(start, target, radius=7.0):
            return target
        distance_map = self._nav_maps.get(target_tile)
        if distance_map is None:
            distance_map = np.full(self.level.grid.shape, -1, dtype=np.int16)
            distance_map[target_tile[1], target_tile[0]] = 0
            queue = deque([target_tile])
            while queue:
                x, y = queue.popleft()
                distance = int(distance_map[y, x])
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nxt = (x + dx, y + dy)
                    if nxt in self._blocked_tiles or not (0 <= nxt[1] < self.level.grid.shape[0] and 0 <= nxt[0] < self.level.grid.shape[1]):
                        continue
                    if distance_map[nxt[1], nxt[0]] >= 0 or int(self.level.grid[nxt[1], nxt[0]]) == 1:
                        continue
                    distance_map[nxt[1], nxt[0]] = distance + 1
                    queue.append(nxt)
            self._nav_maps[target_tile] = distance_map
        if distance_map[start_tile[1], start_tile[0]] < 0:
            return target
        current = start_tile
        best = tile_center(start_tile)
        visited = {start_tile}
        # Look ahead across a few descending cells. Agents commit through a
        # doorway instead of stopping at each tile centre, while the radius
        # sweep prevents corner cutting through furniture.
        for _ in range(4):
            candidates = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (current[0] + dx, current[1] + dy)
                if (
                    nxt not in visited
                    and 0 <= nxt[1] < distance_map.shape[0]
                    and 0 <= nxt[0] < distance_map.shape[1]
                    and 0 <= distance_map[nxt[1], nxt[0]] < distance_map[current[1], current[0]]
                ):
                    candidates.append(nxt)
            if not candidates:
                break
            current = min(
                candidates,
                key=lambda tile: (
                    distance_map[tile[1], tile[0]],
                    (tile[0] - target_tile[0]) ** 2 + (tile[1] - target_tile[1]) ** 2,
                    tile[1],
                    tile[0],
                ),
            )
            visited.add(current)
            point = tile_center(current)
            if not self._segment_navigable(start, point, radius=7.0):
                break
            best = point
        return best

    def _segment_navigable(self, start: np.ndarray, target: np.ndarray, *, radius: float) -> bool:
        delta = target - start
        distance = norm(delta)
        if distance <= 1e-6:
            return True
        direction = delta / distance
        for sample_distance in np.arange(12.0, distance + 0.1, 12.0):
            if not self._can_occupy(start + direction * sample_distance, radius):
                return False
        return self._can_occupy(target, radius)

    def _move_mobile(self, position: np.ndarray, delta: np.ndarray, *, radius: float) -> bool:
        candidate = position + delta
        if self._can_occupy(candidate, radius):
            position[:] = candidate
            return True
        moved = False
        axes = (0, 1) if abs(float(delta[0])) >= abs(float(delta[1])) else (1, 0)
        for axis in axes:
            candidate = position.copy()
            candidate[axis] += delta[axis]
            if self._can_occupy(candidate, radius):
                position[axis] = candidate[axis]
                moved = True
        return moved

    def _is_navigation_choke(self, tile: tuple[int, int]) -> bool:
        x, y = tile
        if not (0 <= y < self.level.grid.shape[0] and 0 <= x < self.level.grid.shape[1]):
            return True
        if self.level.grid[y, x] == Tile.DOOR:
            return True
        neighbours = 0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if (
                0 <= ny < self.level.grid.shape[0]
                and 0 <= nx < self.level.grid.shape[1]
                and (nx, ny) not in self._blocked_tiles
                and self.level.grid[ny, nx] != Tile.WALL
            ):
                neighbours += 1
        return neighbours <= 2

    def _room_id_at(self, tile: tuple[int, int]) -> int:
        x, y = tile
        from ghostline.config import ROOM_HEIGHT_TILES, ROOM_WIDTH_TILES

        column = min(len({room.column for room in self.level.rooms}) - 1, max(0, x // ROOM_WIDTH_TILES))
        row = min(len({room.row for room in self.level.rooms}) - 1, max(0, y // ROOM_HEIGHT_TILES))
        columns = max(room.column for room in self.level.rooms) + 1
        return min(len(self.level.rooms) - 1, row * columns + column)

    def _update_drones(self, dt: float) -> None:
        warning_threshold = self.level.drone_trace_threshold - 10.0
        if (
            self.level.response_drones
            and not self._drone_warning_emitted
            and self.trace >= warning_threshold
            and not self.drones
        ):
            self._drone_warning_emitted = True
            self.events.append(
                SimEvent("drone_warning", tuple(self.level.extraction), self.level.drone_trace_threshold)
            )
        if self.level.response_drones and self.trace >= self.level.drone_trace_threshold and not self.drones:
            position = self.level.extraction.copy()
            self.drones.append(Drone(0, position))
            self.events.append(SimEvent("drone_deployed", tuple(position)))
        for drone in self.drones:
            drone.hit_cooldown = max(0.0, drone.hit_cooldown - dt)
            if drone.disabled_for > 0.0:
                drone.disabled_for = max(0.0, drone.disabled_for - dt)
                drone.attack_windup = 0.0
                continue
            goal_tile = world_to_tile(self.player)
            cached = self._drone_waypoints.get(drone.drone_id)
            if cached is None or cached[0] != goal_tile:
                target = self._next_path_point(drone.position, self.player)
                self._drone_waypoints[drone.drone_id] = (goal_tile, target.copy())
            else:
                target = cached[1]
            direction = unit(target - drone.position)
            if norm(direction) > 0.0:
                drone.facing = math.atan2(float(direction[1]), float(direction[0]))
            self._move_mobile(
                drone.position,
                direction * (102.0 + 6.0 * self.alert_tier) * dt,
                radius=8.0,
            )
            in_strike_range = norm(drone.position - self.player) <= 18.0
            if in_strike_range and drone.hit_cooldown <= 0.0:
                drone.attack_windup += dt
                if drone.attack_windup >= DRONE_STRIKE_WINDUP_SECONDS:
                    drone.hit_cooldown = 1.8
                    drone.attack_windup = 0.0
                    self._damage(drone.position, source_kind="drone")
                    # Response drones are pressure tools, not unavoidable contact
                    # damage. Recoil gives both human and policy controllers time
                    # to react to the telegraphed strike and break pursuit.
                    drone.disabled_for = max(drone.disabled_for, 1.0)
            else:
                drone.attack_windup = max(0.0, drone.attack_windup - 2.5 * dt)

    def _damage(self, source: np.ndarray, *, source_kind: str = "unknown") -> None:
        if self.damage_cooldown > 0.0:
            return
        self.damage_cooldown = DAMAGE_INVULNERABILITY_SECONDS
        self.integrity -= 1
        self.damage_taken += 1
        self.damage_by_guard += int(source_kind == "guard")
        self.damage_by_drone += int(source_kind == "drone")
        self.trace = min(TRACE_MAX, self.trace + 12.0)
        push = unit(self.player - source) * 18.0
        candidate = self.player + push
        if self._can_occupy(candidate, PLAYER_RADIUS):
            self.player[:] = candidate
        self.velocity[:] = 0.0
        self._active_hack = None
        # One confirmed hit is a failure of stealth, not permission for a
        # doorway dogpile. Nearby pursuers search the impact point through the
        # global grace window, leaving a short but earned route to escape.
        for guard in self.level.guards:
            if norm(guard.position - self.player) <= 92.0 and guard.mode == GuardMode.CHASE:
                guard.mode = GuardMode.SEARCH
                guard.mode_seconds = max(
                    guard.mode_seconds,
                    1.55 * GUARD_SEARCH_DURATION_MULTIPLIERS[int(guard.grade)],
                )
                guard.last_known = self.player.copy()
                guard.awareness = 0.0
                guard.attack_windup = 0.0
                guard.hit_cooldown = max(guard.hit_cooldown, 1.25)
                guard.stimulus = "eye"
        self.events.append(SimEvent("damage", tuple(self.player), 1.0))

    def _broadcast_noise(self, *, radius: float) -> None:
        for guard in self.level.guards:
            if guard.mode == GuardMode.CHASE or norm(guard.position - self.player) > radius:
                continue
            guard.mode = GuardMode.INVESTIGATE
            guard.mode_seconds = 2.6 * GUARD_SEARCH_DURATION_MULTIPLIERS[int(guard.grade)]
            guard.last_known = self.player.copy()
            guard.patrol_pause_seconds = 0.0
            guard.stimulus = "sound"

    def _share_alert(self, source: Guard) -> None:
        eligible = [
            guard for guard in self.level.guards
            if guard is not source
            and guard.radio_jammed_for <= 0.0
            and guard.mode != GuardMode.CHASE
            and norm(guard.position - source.position) <= 380.0
        ]
        eligible.sort(key=lambda guard: norm(guard.position - source.position))
        for guard in eligible[:2]:
            guard.mode = GuardMode.INVESTIGATE
            guard.mode_seconds = 3.0 * GUARD_SEARCH_DURATION_MULTIPLIERS[int(guard.grade)]
            guard.last_known = source.last_known.copy()
            guard.patrol_pause_seconds = 0.0
            guard.stimulus = "radio"

    def _activate_pulse(self) -> None:
        if self.pulse_charges <= 0 or self.pulse_cooldown > 0.0:
            return
        self.pulse_charges -= 1
        self.pulses_used += 1
        self.pulse_cooldown = 0.8
        affected = 0
        for camera in self.level.cameras:
            if norm(camera.position - self.player) <= PULSE_RADIUS:
                camera.disabled_for = PULSE_DISABLE_SECONDS
                affected += 1
        for drone in self.drones:
            if norm(drone.position - self.player) <= PULSE_RADIUS:
                drone.disabled_for = PULSE_DISABLE_SECONDS
                drone.attack_windup = 0.0
                affected += 1
        for guard in self.level.guards:
            if norm(guard.position - self.player) <= PULSE_RADIUS:
                guard.radio_jammed_for = PULSE_DISABLE_SECONDS
                if guard.mode == GuardMode.CHASE:
                    guard.mode = GuardMode.SEARCH
                    guard.mode_seconds = 2.5
                affected += 1
        self.trace = max(self.trace_floor, self.trace - 14.0)
        self.events.append(SimEvent("pulse", tuple(self.player), float(affected)))

    def _update_hacking(self, dt: float) -> None:
        nearest = None
        nearest_distance = float("inf")
        for terminal in self.level.terminals:
            if terminal.completed:
                continue
            distance = norm(terminal.position - self.player)
            if distance < nearest_distance:
                nearest, nearest_distance = terminal, distance
        if nearest is None or nearest_distance > HACK_RADIUS:
            self._active_hack = None
            return
        self._active_hack = nearest.terminal_id
        self.objective_terminal_id = nearest.terminal_id
        nearest.progress += dt
        if self.elapsed_ticks % 20 == 0:
            self.events.append(SimEvent("hack_tick", tuple(nearest.position), nearest.progress / nearest.hack_seconds))
        if nearest.progress + 1e-6 < nearest.hack_seconds:
            return
        nearest.completed = True
        nearest.progress = nearest.hack_seconds
        previous_data = self.data
        self.data += nearest.value
        self.hacks_completed += 1
        self.optional_data = max(0, self.data - self.level.quota)
        quota_ratio = min(1.0, self.data / max(1, self.level.quota))
        self.trace_floor = min(62.0, 8.0 + 48.0 * quota_ratio)
        self.trace = max(self.trace, self.trace_floor + 4.0)
        self.events.append(SimEvent("hack_complete", tuple(nearest.position), float(nearest.value)))
        self.objective_terminal_id = None
        if previous_data < self.level.quota <= self.data:
            self.events.append(SimEvent("quota_met", tuple(self.level.extraction), float(self.data)))
        self._active_hack = None

    def _update_trace(self, dt: float) -> None:
        seen_by_camera = any(camera.detected for camera in self.level.cameras)
        seen = seen_by_camera or self._seen_by_guard
        if seen:
            gain = 11.5 + 3.0 * self.alert_tier
            self.trace = min(TRACE_MAX, self.trace + gain * dt)
            if not self._was_seen:
                self.detections += int(seen_by_camera)
        else:
            self.trace = max(self.trace_floor, self.trace - 5.2 * dt)
        self._was_seen = seen
        self.max_trace = max(self.max_trace, self.trace)
        if self.trace >= TRACE_MAX and not self.lockdown:
            self.lockdown = True
            self.events.append(SimEvent("lockdown", tuple(self.player)))

    def _check_extraction(self) -> None:
        if self.quota_met and norm(self.level.extraction - self.player) <= 25.0:
            self.extracted = True
            self.terminated = True
            self.fail_reason = "none"
            self.events.append(SimEvent("extracted", tuple(self.player), float(self.data)))

    def terminal_info(self) -> dict[str, float | int | bool | str]:
        return {
            "is_success": self.extracted,
            "fail_reason": self.fail_reason,
            "tier": self.tier,
            "seed": self.seed,
            "quota": self.level.quota,
            "data": self.data,
            "optional_data": self.optional_data,
            "duration_seconds": self.elapsed_seconds,
            "remaining_seconds": self.remaining_seconds,
            "trace": self.trace,
            "max_trace": self.max_trace,
            "detections": self.detections,
            "integrity": self.integrity,
            "damage": self.damage_taken,
            "damage_by_guard": self.damage_by_guard,
            "damage_by_drone": self.damage_by_drone,
            "pulse_uses": self.pulses_used,
            "hacks_completed": self.hacks_completed,
            "distance_travelled": self.distance_travelled,
            "lockdown": self.lockdown,
        }
