"""Gymnasium runner interface for the multi-agent v2 contract."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from gymnasium import spaces

from ghostline.config import (
    LOCAL_GRID_SIZE,
    MAX_ENTITIES,
    PLAYER_PERCEPTION_DISTANCE,
    POLICY_REPEAT,
    RAY_COUNT,
    TILE_SIZE,
    TIERS,
    TRACE_MAX,
)
from ghostline.config_v2 import (
    DETECTION_COST,
    EXPOSURE_COST_PER_DECISION,
    FIELD_TARGET_FEATURES,
    MAX_FIELD_TARGETS,
    QUIET_DATA_BONUS,
    QUIET_TRACE_CEILING,
    VENT_TRANSIT_SECONDS,
)
from ghostline.env import (
    GhostlineEnv,
    PerceivedEntity,
    PROGRESS_POTENTIAL_SCALE,
    REWARD_DISCOUNT,
)
from ghostline.generation import tile_center, world_to_tile
from ghostline.simulation import norm
from ghostline.simulation_v2 import GhostlineSimulationV2
from ghostline.types import Tile
from ghostline.types_v2 import RUNNER_ACTION_COUNT_V2, ContractDirective, GuardRole, RunnerActionV2


_RAY_ANGLES = np.linspace(0.0, math.tau, RAY_COUNT, endpoint=False)
_RAY_DIRECTIONS = np.stack(
    (np.cos(_RAY_ANGLES), np.sin(_RAY_ANGLES)),
    axis=1,
).astype(np.float32)
_RAY_SAMPLE_DISTANCES = np.arange(1, 41, dtype=np.float32) * 8.0


def runner_potential_progress_reward(
    previous: float,
    current: float,
    *,
    gamma: float,
    terminal: bool = False,
) -> float:
    """Discount-matched mission shaping with a zero terminal potential."""

    next_potential = 0.0 if terminal else float(current)
    return float(
        PROGRESS_POTENTIAL_SCALE
        * (float(gamma) * next_potential - float(previous))
    )


class GhostlineEnvV2(GhostlineEnv):
    """Player-equivalent v2 contract with directives and field systems."""

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        seed: int = 0,
        tier: int = 1,
        directive: ContractDirective | str | int = ContractDirective.STANDARD,
        external_security: bool = False,
        reward_gamma: float = REWARD_DISCOUNT,
    ):
        self.directive = ContractDirective.parse(directive)
        self.external_security = bool(external_security)
        self.reward_gamma = float(reward_gamma)
        if not 0.0 < self.reward_gamma <= 1.0:
            raise ValueError("reward_gamma must lie in (0, 1]")
        self._directive_par_seconds = 1.0
        super().__init__(render_mode=render_mode, seed=seed, tier=tier)
        self.action_space = spaces.Discrete(RUNNER_ACTION_COUNT_V2)
        self.observation_space = spaces.Dict(
            {
                "ego": spaces.Box(-1.0, 1.0, shape=(27,), dtype=np.float32),
                "objective": spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32),
                "directive": spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32),
                # Runner-owned status and public, map-equivalent field records.
                "field": spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32),
                "field_targets": spaces.Box(
                    -1.0,
                    1.0,
                    shape=(MAX_FIELD_TARGETS, FIELD_TARGET_FEATURES),
                    dtype=np.float32,
                ),
                "field_target_mask": spaces.Box(0, 1, shape=(MAX_FIELD_TARGETS,), dtype=np.int8),
                "local_grid": spaces.Box(0.0, 1.0, shape=(15, LOCAL_GRID_SIZE, LOCAL_GRID_SIZE), dtype=np.float32),
                "targets": spaces.Box(-1.0, 1.0, shape=(5, 10), dtype=np.float32),
                "target_mask": spaces.Box(0, 1, shape=(5,), dtype=np.int8),
                "entities": spaces.Box(-1.0, 1.0, shape=(MAX_ENTITIES, 16), dtype=np.float32),
                "entity_mask": spaces.Box(0, 1, shape=(MAX_ENTITIES,), dtype=np.int8),
                "rays": spaces.Box(0.0, 1.0, shape=(RAY_COUNT, 4), dtype=np.float32),
                "action_mask": spaces.Box(0, 1, shape=(RUNNER_ACTION_COUNT_V2,), dtype=np.int8),
            }
        )
        self.sim = GhostlineSimulationV2(
            seed=self.initial_seed,
            tier=self.tier,
            directive=self.directive,
            external_security=self.external_security,
        )
        self._reset_episode_metrics()

    @property
    def unwrapped_sim(self) -> GhostlineSimulationV2:
        return self.sim

    def _reset_episode_metrics(self) -> None:
        self._distance_cache.clear()
        self._action_history = []
        self._trace_history = [self.sim.trace]
        self._idle_decisions = 0
        self._route_lower_bound = self._mission_route_lower_bound()
        self._directive_par_seconds = self.sim.directive_par_seconds
        self.reward_components = self._empty_rewards()
        self._previous_potential = self._mission_potential()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        options = dict(options or {})
        if "directive" in options:
            self.directive = ContractDirective.parse(options["directive"])
        if "external_security" in options:
            self.external_security = bool(options["external_security"])
        self.sim.directive = self.directive
        self.sim.external_security = self.external_security
        observation, info = super().reset(seed=seed, options=options)
        self._reset_episode_metrics()
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        decoded, invalid = self._validated_action(action)
        self._action_history.append(decoded.encode())
        self._idle_decisions += int(decoded.move == 0 and self.sim.active_hack_progress <= 0.0)

        before_data = self.sim.data
        before_optional = self.sim.optional_data
        before_integrity = self.sim.integrity
        before_trace = self.sim.trace
        before_detections = self.sim.detections
        before_explored = int(np.count_nonzero(self.sim.explored))
        self.sim.advance(decoded, ticks=POLICY_REPEAT)
        self._trace_history.append(self.sim.trace)
        terminal = bool(self.sim.terminated or self.sim.truncated)
        potential = self._mission_potential()

        components = self._empty_rewards()
        components["extraction"] = 20.0 if self.sim.extracted else 0.0
        components["data"] = min(6.0, max(0, self.sim.data - before_data) * 1.5)
        components["progress"] = runner_potential_progress_reward(
            self._previous_potential,
            potential,
            gamma=self.reward_gamma,
            terminal=terminal,
        )
        newly_explored = max(0, int(np.count_nonzero(self.sim.explored)) - before_explored)
        components["exploration"] = min(0.08, newly_explored * 0.008)
        components["trace"] = -0.006 * max(0.0, self.sim.trace - before_trace)
        # Exposure is a continuous state cost, not shaping: carrying a hot trace
        # is expensive every decision it persists. The old gain-only term made
        # the entire stealth budget 4.7% of a successful run, so going loud for
        # a whole mission cost less than 75 extra seconds of mission time.
        components["exposure"] = -EXPOSURE_COST_PER_DECISION * (self.sim.trace / TRACE_MAX)
        components["detection"] = -DETECTION_COST * max(0, self.sim.detections - before_detections)
        # Securing data while the network is still cold is the reward for the
        # quiet route, so stealth is actively worth something rather than merely
        # less bad than being loud.
        secured = max(0, self.sim.data - before_data)
        if secured and self.sim.trace <= QUIET_TRACE_CEILING:
            components["stealth"] = QUIET_DATA_BONUS * secured
        components["damage"] = -3.0 * max(0, before_integrity - self.sim.integrity)
        components["time"] = -0.002
        # Holding still is now a legitimate tactic: crouched, in cover and
        # unseen is waiting out a patrol, not stalling. The time cost still
        # applies, so it cannot be farmed indefinitely.
        holding_cover = bool(decoded.crouch) and self.sim.in_cover and not self.sim._was_seen
        idling = decoded.move == 0 and self.sim.active_hack_progress <= 0.0
        components["idle"] = -0.006 if idling and not holding_cover else 0.0
        components["invalid"] = -0.02 if invalid else 0.0
        if self.directive == ContractDirective.GHOST:
            components["directive"] -= 0.008 * max(0.0, self.sim.trace - before_trace)
            components["directive"] -= 0.12 * max(0, self.sim.detections - before_detections)
            components["directive"] -= 0.8 * max(0, before_integrity - self.sim.integrity)
        elif self.directive == ContractDirective.SPEED:
            components["directive"] -= 0.003
        elif self.directive == ContractDirective.GREED:
            components["directive"] += 0.75 * max(0, self.sim.optional_data - before_optional)
        if (self.sim.terminated or self.sim.truncated) and not self.sim.extracted:
            components["failure"] = -10.0 if self.sim.fail_reason == "integrity_lost" else -6.0
        if self.sim.extracted and self.directive_completed:
            components["directive"] += 4.0

        reward = float(sum(components.values()))
        for key, value in components.items():
            self.reward_components[key] += value
        self._previous_potential = 0.0 if terminal else potential
        info = self._info()
        if self.sim.terminated or self.sim.truncated:
            for key, value in self.reward_components.items():
                info[f"reward_{key}"] = float(value)
            info["reward_components"] = {
                key: float(value)
                for key, value in self.reward_components.items()
            }
            info["reward_total"] = float(sum(self.reward_components.values()))
            info["episode_extra_stats"] = {
                "tier": int(self.tier),
                "success": float(self.sim.extracted),
                "directive_success": float(self.directive_completed),
                "data": float(self.sim.data),
                "damage": float(self.sim.damage_taken),
                "max_trace": float(self.sim.max_trace),
            }
            info["telemetry"] = self.telemetry()
        return self._observation(), reward, self.sim.terminated, self.sim.truncated, info

    def _validated_action(self, action: int) -> tuple[RunnerActionV2, bool]:
        """Decode exactly the legal action whose log-probability was sampled.

        The former wrapper clipped every value to ``0..71`` even though the
        action space contains 288 codes.  That made PPO update the probability
        of one action from a transition generated by another.  This boundary is
        intentionally small and exhaustively testable.
        """

        try:
            value = int(action)
        except (TypeError, ValueError, OverflowError):
            return RunnerActionV2(), True
        in_bounds = 0 <= value < RUNNER_ACTION_COUNT_V2
        decoded = RunnerActionV2.decode(value if in_bounds else 0)
        legal = in_bounds and bool(self.sim.action_mask()[value])
        if legal:
            return decoded, False
        # Preserve a requested movement direction when a resource bit is
        # unavailable, but never execute a different ability behind PPO's back.
        return RunnerActionV2(move=decoded.move), True

    @property
    def directive_completed(self) -> bool:
        return self.sim.directive_completed

    def _field_systems(self) -> np.ndarray:
        """Eight values describing the runner's own field options.

        Everything here is state the player can see on their HUD: how many
        charges remain, whether something is in reach, and whether a transit or
        a darkened room is currently active. No hidden security state leaks in.
        """

        sim = self.sim
        vent = sim.nearest_vent()
        device = sim.nearest_hackable()
        ready_kinds = {
            kind
            for kind in ("camera", "door", "lights")
            if any(
                candidate.kind == kind
                and candidate.cooldown <= 0.0
                and norm(candidate.position - sim.player) <= 46.0
                for candidate in sim.hackable
            )
        }
        values = np.asarray(
            [
                min(1.0, sim.hack_charges / 3.0) * 2.0 - 1.0,
                float(sim.can_interact()) * 2.0 - 1.0,
                float(vent is not None) * 2.0 - 1.0,
                float("camera" in ready_kinds and device is not None) * 2.0 - 1.0,
                float("door" in ready_kinds and device is not None) * 2.0 - 1.0,
                float("lights" in ready_kinds and device is not None) * 2.0 - 1.0,
                min(1.0, sim.vent_transit / max(1e-6, VENT_TRANSIT_SECONDS)) * 2.0 - 1.0,
                float(sim._room_at(sim.player) in sim.darkened_rooms) * 2.0 - 1.0,
            ],
            dtype=np.float32,
        )
        return np.clip(values, -1.0, 1.0)

    def _field_targets(self) -> tuple[np.ndarray, np.ndarray]:
        """Return only field objects a human can see or has explored.

        Static vents and control pedestals become map knowledge once their tile
        has been explored.  Deployed sensors require direct player sight.  No
        global sensor count, unseen nearest-object vector, or hidden cooldown is
        exposed.
        """

        sim = self.sim
        entries: list[tuple[int, float, int, np.ndarray]] = []
        kind_codes = {"vent": 0, "camera": 1, "door": 2, "lights": 3, "sensor": 4}

        def explored(tile: tuple[int, int]) -> bool:
            x, y = tile
            return (
                0 <= y < sim.explored.shape[0]
                and 0 <= x < sim.explored.shape[1]
                and bool(sim.explored[y, x])
            )

        for vent in sim.vents:
            if not explored(vent.tile):
                continue
            position = tile_center(vent.tile)
            delta = position - sim.player
            exit_known = explored(vent.exit_tile)
            exit_delta = vent.exit_position - sim.player if exit_known else np.zeros(2, dtype=np.float32)
            ready = vent.tile == world_to_tile(sim.player) and sim.vent_transit <= 0.0
            record = self._field_target_record(
                kind_codes["vent"],
                delta,
                ready=ready,
                active=sim.vent_transit > 0.0 and ready,
                destination=exit_delta,
                destination_known=exit_known,
            )
            entries.append((0 if ready else 3, norm(delta), int(vent.vent_id), record))

        for device in sim.hackable:
            if not explored(device.tile):
                continue
            delta = device.position - sim.player
            ready = (
                sim.hack_charges > 0
                and sim.hack_cooldown <= 0.0
                and device.cooldown <= 0.0
                and norm(delta) <= 46.0
            )
            record = self._field_target_record(
                kind_codes[device.kind],
                delta,
                ready=ready,
                active=device.hacked_for > 0.0,
            )
            entries.append((0 if ready else 2, norm(delta), 1000 + int(device.device_id), record))

        for sensor in sim.field_sensors:
            if not sim.player_can_see(sensor.position):
                continue
            delta = sensor.position - sim.player
            record = self._field_target_record(
                kind_codes["sensor"],
                delta,
                ready=False,
                active=sensor.armed_in <= 0.0 and not sensor.triggered,
            )
            entries.append((1, norm(delta), 2000 + int(sensor.sensor_id), record))

        entries.sort(key=lambda item: (item[0], item[1], item[2]))
        values = np.zeros((MAX_FIELD_TARGETS, FIELD_TARGET_FEATURES), dtype=np.float32)
        mask = np.zeros(MAX_FIELD_TARGETS, dtype=np.int8)
        for index, (_priority, _distance, _stable_id, record) in enumerate(
            entries[:MAX_FIELD_TARGETS]
        ):
            values[index] = record
            mask[index] = 1
        return values, mask

    def _field_target_record(
        self,
        kind: int,
        delta: np.ndarray,
        *,
        ready: bool,
        active: bool,
        destination: np.ndarray | None = None,
        destination_known: bool = False,
    ) -> np.ndarray:
        kinds = np.full(5, -1.0, dtype=np.float32)
        kinds[kind] = 1.0
        destination = (
            np.asarray(destination, dtype=np.float32)
            if destination is not None and destination_known
            else np.zeros(2, dtype=np.float32)
        )
        record = np.asarray(
            [
                *kinds,
                np.clip(delta[0] / self.sim.level.world_width, -1.0, 1.0),
                np.clip(delta[1] / self.sim.level.world_height, -1.0, 1.0),
                min(1.0, norm(delta) / 900.0) * 2.0 - 1.0,
                float(ready) * 2.0 - 1.0,
                float(active) * 2.0 - 1.0,
                np.clip(destination[0] / self.sim.level.world_width, -1.0, 1.0),
                np.clip(destination[1] / self.sim.level.world_height, -1.0, 1.0),
                (
                    min(1.0, norm(destination) / 900.0) * 2.0 - 1.0
                    if destination_known
                    else -1.0
                ),
            ],
            dtype=np.float32,
        )
        return np.clip(record, -1.0, 1.0)

    def _empty_rewards(self) -> dict[str, float]:
        result = super()._empty_rewards()
        # V2 adds the directive term plus the stealth economy. Keeping
        # them declared here keeps exact component accounting intact.
        for key in ("directive", "exposure", "detection", "stealth", "field"):
            result[key] = 0.0
        return result

    def _observation(self) -> dict[str, np.ndarray]:
        observation = super()._observation()
        observation["directive"] = self._directive()
        observation["field"] = self._field_systems()
        field_targets, field_target_mask = self._field_targets()
        observation["field_targets"] = field_targets
        observation["field_target_mask"] = field_target_mask
        return observation

    def _ego(self) -> np.ndarray:
        base = super()._ego()
        maximum = 2.0
        extra = np.asarray(
            [
                self.sim.decoy_charges / maximum * 2.0 - 1.0,
                min(1.0, self.sim.decoy_cooldown) * 2.0 - 1.0,
                self.sim.incoming_projectile_pressure * 2.0 - 1.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate((base, extra)).astype(np.float32)

    def _directive(self) -> np.ndarray:
        directive_flags = [
            1.0 if self.directive == ContractDirective.GHOST else -1.0,
            1.0 if self.directive == ContractDirective.SPEED else -1.0,
            1.0 if self.directive == ContractDirective.GREED else -1.0,
        ]
        par_margin = np.clip((self._directive_par_seconds - self.sim.elapsed_seconds) / max(1.0, self._directive_par_seconds), -1.0, 1.0)
        all_data = sum(terminal.value for terminal in self.sim.level.terminals)
        greed_progress = min(1.0, self.sim.data / max(1, all_data)) * 2.0 - 1.0
        stealth_quality = 1.0 - 2.0 * max(self.sim.max_trace / 100.0, self.sim.damage_taken / 3.0)
        return np.clip(
            np.asarray([*directive_flags, par_margin, greed_progress, stealth_quality], dtype=np.float32),
            -1.0,
            1.0,
        )

    def _local_grid(self, visible_positions: list[tuple[np.ndarray, float]]) -> np.ndarray:
        sim = self.sim
        size = LOCAL_GRID_SIZE
        half = size // 2
        base = np.zeros((8, size, size), dtype=np.float32)
        px, py = world_to_tile(sim.player)
        height, width = sim.level.grid.shape
        world_x0, world_y0 = px - half, py - half
        source_x0, source_y0 = max(0, world_x0), max(0, world_y0)
        source_x1 = min(width, world_x0 + size)
        source_y1 = min(height, world_y0 + size)
        destination_x0 = source_x0 - world_x0
        destination_y0 = source_y0 - world_y0
        destination_x1 = destination_x0 + max(0, source_x1 - source_x0)
        destination_y1 = destination_y0 + max(0, source_y1 - source_y0)
        if source_x1 > source_x0 and source_y1 > source_y0:
            destination = (
                slice(destination_y0, destination_y1),
                slice(destination_x0, destination_x1),
            )
            window = sim.level.grid[
                source_y0:source_y1,
                source_x0:source_x1,
            ]
            base[0][destination] = window != Tile.WALL
            base[1][destination] = window == Tile.WALL
            base[2][destination] = window == Tile.DOOR
            base[3][destination] = sim.explored[
                source_y0:source_y1,
                source_x0:source_x1,
            ]
        if destination_y0 > 0:
            base[1, :destination_y0, :] = 1.0
        if destination_y1 < size:
            base[1, destination_y1:, :] = 1.0
        if destination_x0 > 0:
            base[1, :, :destination_x0] = 1.0
        if destination_x1 < size:
            base[1, :, destination_x1:] = 1.0

        def local(tile: tuple[int, int]) -> tuple[int, int]:
            return tile[0] - px + half, tile[1] - py + half

        for tile in sim._blocked_tiles:
            lx, ly = local(tile)
            if 0 <= lx < size and 0 <= ly < size:
                base[1, ly, lx] = 1.0
        for terminal in sim.level.terminals:
            if terminal.completed:
                continue
            lx, ly = local(world_to_tile(terminal.position))
            if 0 <= lx < size and 0 <= ly < size:
                base[4, ly, lx] = 1.0
        extraction_x, extraction_y = local(world_to_tile(sim.level.extraction))
        if (
            sim.quota_met
            and 0 <= extraction_x < size
            and 0 <= extraction_y < size
        ):
            base[5, extraction_y, extraction_x] = 1.0
        for entity_position, dangerous in visible_positions:
            lx, ly = local(world_to_tile(entity_position))
            if 0 <= lx < size and 0 <= ly < size:
                base[6, ly, lx] = max(base[6, ly, lx], dangerous)
        base[7, half, half] = 1.0

        # Locked/warning door, decoy, projectile, known vent, known control
        # pedestal, visible sensor, and explored darkness.
        extra = np.zeros((7, LOCAL_GRID_SIZE, LOCAL_GRID_SIZE), dtype=np.float32)

        def mark_tile(channel: int, tile: tuple[int, int], value: float = 1.0) -> None:
            tx, ty = tile
            lx, ly = tx - px + half, ty - py + half
            if 0 <= lx < LOCAL_GRID_SIZE and 0 <= ly < LOCAL_GRID_SIZE:
                extra[channel, ly, lx] = max(extra[channel, ly, lx], value)

        for door in self.sim.security_doors:
            if door.locked or door.warning_remaining > 0.0:
                mark_tile(0, door.tile, 1.0 if door.locked else 0.5)
        for decoy in self.sim.decoys:
            mark_tile(1, world_to_tile(decoy.position), min(1.0, decoy.lifetime / 2.0))
        for projectile in self.sim.projectiles:
            if self.sim.player_can_see(projectile.position):
                mark_tile(2, world_to_tile(projectile.position))
        for vent in self.sim.vents:
            if self.sim.explored[vent.tile[1], vent.tile[0]]:
                mark_tile(3, vent.tile)
        for device in self.sim.hackable:
            if self.sim.explored[device.tile[1], device.tile[0]]:
                mark_tile(4, device.tile, 1.0 if device.cooldown <= 0.0 else 0.45)
        for sensor in self.sim.field_sensors:
            if self.sim.player_can_see(sensor.position):
                mark_tile(5, world_to_tile(sensor.position), 0.55 if sensor.armed_in > 0.0 else 1.0)
        for room in self.sim.level.rooms:
            if int(room.room_id) not in self.sim.darkened_rooms:
                continue
            room_x0 = max(source_x0, int(room.x))
            room_y0 = max(source_y0, int(room.y))
            room_x1 = min(source_x1, int(room.x + room.width))
            room_y1 = min(source_y1, int(room.y + room.height))
            if room_x1 <= room_x0 or room_y1 <= room_y0:
                continue
            local_x0, local_y0 = room_x0 - world_x0, room_y0 - world_y0
            local_x1 = local_x0 + room_x1 - room_x0
            local_y1 = local_y0 + room_y1 - room_y0
            explored = self.sim.explored[
                room_y0:room_y1,
                room_x0:room_x1,
            ]
            extra[
                6,
                local_y0:local_y1,
                local_x0:local_x1,
            ] = explored.astype(np.float32)
        return np.concatenate((base, extra), axis=0)

    def _entities(
        self,
        percepts: list[PerceivedEntity] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        percepts = list(percepts if percepts is not None else self._security_percepts())
        records = sorted(
            percepts,
            key=lambda percept: (percept.priority, norm(percept.position - self.sim.player)),
        )
        result = np.full((MAX_ENTITIES, 16), -1.0, dtype=np.float32)
        mask = np.zeros(MAX_ENTITIES, dtype=np.int8)
        for index, percept in enumerate(records[:MAX_ENTITIES]):
            delta = percept.position - self.sim.player
            result[index, :13] = self._entity_record(
                percept.kind,
                delta,
                percept.velocity,
                percept.facing,
                percept.alert,
                percept.confidence,
                percept.grade,
            )
            # Role badges are public presentation state.  Attach them only to
            # an exact live guard percept; never associate an audio/memory row
            # with the nearest hidden live guard.
            if percept.kind == 0 and percept.priority == 0 and percept.confidence >= 0.999:
                guard = next(
                    (
                        candidate
                        for candidate in self.sim.level.guards
                        if norm(candidate.position - percept.position) <= 1e-3
                    ),
                    None,
                )
                if guard is not None:
                    role = int(self.sim.operative_states[guard.guard_id].role)
                    result[index, 13:16] = -1.0
                    result[index, 13 + role] = 1.0
            mask[index] = 1
        return result, mask

    def _rays(self, visible_positions: list[tuple[np.ndarray, float]]) -> np.ndarray:
        sim = self.sim
        points = (
            sim.player[None, None, :]
            + _RAY_DIRECTIONS[:, None, :]
            * _RAY_SAMPLE_DISTANCES[None, :, None]
        )
        tiles = np.floor(points / TILE_SIZE).astype(np.int32)
        tile_x = tiles[..., 0]
        tile_y = tiles[..., 1]
        height, width = sim.level.grid.shape
        in_bounds = (
            (tile_x >= 0)
            & (tile_y >= 0)
            & (tile_x < width)
            & (tile_y < height)
        )
        clipped_x = np.clip(tile_x, 0, width - 1)
        clipped_y = np.clip(tile_y, 0, height - 1)
        blocked = sim.level.grid == Tile.WALL
        if sim._blocked_tiles:
            blocked = blocked.copy()
            for tx, ty in sim._blocked_tiles:
                if 0 <= tx < width and 0 <= ty < height:
                    blocked[ty, tx] = True
        hits = ~in_bounds | blocked[clipped_y, clipped_x]
        has_hit = hits.any(axis=1)
        first_hit = np.argmax(hits, axis=1)
        geometry = np.where(
            has_hit,
            _RAY_SAMPLE_DISTANCES[first_hit],
            PLAYER_PERCEPTION_DISTANCE,
        )
        sample_indices = np.arange(
            _RAY_SAMPLE_DISTANCES.shape[0],
            dtype=np.int32,
        )[None, :]
        before_geometry_hit = (
            ~has_hit[:, None]
            | (sample_indices <= first_hit[:, None])
        )
        unexplored = (
            in_bounds
            & before_geometry_hit
            & ~sim.explored[clipped_y, clipped_x]
        )
        has_unexplored = unexplored.any(axis=1)
        first_unexplored = np.argmax(unexplored, axis=1)
        explored_fraction = np.where(
            has_unexplored,
            _RAY_SAMPLE_DISTANCES[first_unexplored]
            / PLAYER_PERCEPTION_DISTANCE,
            1.0,
        )
        danger = np.zeros(RAY_COUNT, dtype=np.float32)
        if visible_positions:
            positions = np.stack(
                [position for position, _ in visible_positions]
            ).astype(np.float32)
            pressure = np.asarray(
                [value for _, value in visible_positions],
                dtype=np.float32,
            )
            delta = positions - sim.player
            distance = np.linalg.norm(delta, axis=1)
            valid = distance > 1e-6
            directions = np.zeros_like(delta)
            directions[valid] = delta[valid] / distance[valid, None]
            alignment = _RAY_DIRECTIONS @ directions.T
            contribution = pressure * (
                1.0
                - np.minimum(
                    1.0,
                    distance / PLAYER_PERCEPTION_DISTANCE,
                )
            )
            danger = np.max(
                np.where(
                    (alignment > 0.94) & valid[None, :],
                    contribution[None, :],
                    0.0,
                ),
                axis=1,
            ).astype(np.float32)

        projectile = np.zeros(RAY_COUNT, dtype=np.float32)
        visible_projectiles = [
            shot
            for shot in sim.projectiles
            if sim.player_can_see(shot.position)
        ]
        if visible_projectiles:
            delta = np.stack(
                [shot.position - sim.player for shot in visible_projectiles]
            ).astype(np.float32)
            distance = np.linalg.norm(delta, axis=1)
            zero = distance <= 1e-6
            if zero.any():
                projectile[:] = 1.0
            valid = ~zero
            if valid.any():
                directions = delta[valid] / distance[valid, None]
                alignment = _RAY_DIRECTIONS @ directions.T
                contribution = 1.0 - np.minimum(
                    1.0,
                    distance[valid] / PLAYER_PERCEPTION_DISTANCE,
                )
                projectile = np.maximum(
                    projectile,
                    np.max(
                        np.where(
                            alignment >= math.cos(math.pi / RAY_COUNT),
                            contribution[None, :],
                            0.0,
                        ),
                        axis=1,
                    ),
                ).astype(np.float32)
        return np.stack(
            (
                geometry / PLAYER_PERCEPTION_DISTANCE,
                danger,
                explored_fraction,
                projectile,
            ),
            axis=1,
        ).astype(np.float32)

    def telemetry(self) -> dict[str, Any]:
        telemetry = super().telemetry()
        counts = np.bincount(np.asarray(self._action_history, dtype=np.int64), minlength=RUNNER_ACTION_COUNT_V2)
        telemetry.update(
            {
                "contract": "GhostlineEnv-v2",
                "directive": self.directive.name.lower(),
                "directive_success": self.directive_completed,
                "directive_par_seconds": self._directive_par_seconds,
                "decoys_used": self.sim.decoys_used,
                "action_counts": counts.tolist(),
            }
        )
        return telemetry

    def _info(self) -> dict[str, Any]:
        info = super()._info()
        info.update(
            {
                "contract": "GhostlineEnv-v2",
                "directive": self.directive.name.lower(),
                "directive_success": self.directive_completed,
                "directive_par_seconds": self._directive_par_seconds,
                "campaign_tier_name": TIERS[self.tier].name,
            }
        )
        return info
