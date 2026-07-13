from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ghostline.config import (
    DASH_ENERGY_MAX,
    LOCAL_GRID_SIZE,
    MAX_ENTITIES,
    MAX_PULSE_CHARGES,
    MAX_TARGETS,
    PLAYER_PERCEPTION_DISTANCE,
    POLICY_REPEAT,
    RAY_COUNT,
    TILE_SIZE,
    TRACE_MAX,
    TIERS,
)
from ghostline.generation import tile_center, world_to_tile
from ghostline.simulation import GhostlineSimulation, angle_vector, norm
from ghostline.types import Action, GuardGrade, GuardMode, SecurityIntel, Tile

OBJECTIVE_FEATURES = (
    "phase_extract", "goal_dx", "goal_dy", "goal_distance",
    "next_waypoint_dx", "next_waypoint_dy", "link_progress", "target_value",
)
REWARD_DISCOUNT = 0.995
PROGRESS_POTENTIAL_SCALE = 0.35


@dataclass(frozen=True)
class PerceivedEntity:
    kind: int
    position: np.ndarray
    velocity: np.ndarray
    facing: float | None
    alert: float
    confidence: float
    grade: GuardGrade
    priority: int


def potential_progress_reward(previous: float, current: float) -> float:
    """Discount-consistent potential shaping with no profitable closed cycles."""
    return PROGRESS_POTENTIAL_SCALE * (REWARD_DISCOUNT * current - previous)


class GhostlineEnv(gym.Env[dict[str, np.ndarray], int]):
    """Player-equivalent structured-observation RL interface for Ghostline."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, *, render_mode: str | None = None, seed: int = 0, tier: int = 1):
        super().__init__()
        self.render_mode = render_mode
        self.initial_seed = int(seed)
        self.tier = int(tier)
        self.action_space = spaces.Discrete(36)
        self.observation_space = spaces.Dict(
            {
                "ego": spaces.Box(-1.0, 1.0, shape=(24,), dtype=np.float32),
                "objective": spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32),
                "local_grid": spaces.Box(0.0, 1.0, shape=(8, LOCAL_GRID_SIZE, LOCAL_GRID_SIZE), dtype=np.float32),
                "targets": spaces.Box(-1.0, 1.0, shape=(MAX_TARGETS, 10), dtype=np.float32),
                "target_mask": spaces.Box(0, 1, shape=(MAX_TARGETS,), dtype=np.int8),
                "entities": spaces.Box(-1.0, 1.0, shape=(MAX_ENTITIES, 13), dtype=np.float32),
                "entity_mask": spaces.Box(0, 1, shape=(MAX_ENTITIES,), dtype=np.int8),
                "rays": spaces.Box(0.0, 1.0, shape=(RAY_COUNT, 3), dtype=np.float32),
                "action_mask": spaces.Box(0, 1, shape=(36,), dtype=np.int8),
            }
        )
        self.sim = GhostlineSimulation(seed=self.initial_seed, tier=self.tier)
        self._seeded_once = False
        self._renderer = None
        self.reward_components: dict[str, float] = {}
        self._distance_cache: dict[tuple[int, int], np.ndarray] = {}
        self._action_history: list[int] = []
        self._trace_history: list[float] = [self.sim.trace]
        self._idle_decisions = 0
        self._route_lower_bound = 1.0
        self._previous_potential = self._mission_potential()

    @property
    def unwrapped_sim(self) -> GhostlineSimulation:
        return self.sim

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        training_lesson = int((options or {}).get("training_lesson", 0))
        requested_tier = int((options or {}).get("tier", self.tier))
        lesson_tiers = {1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 4}
        tier = lesson_tiers.get(training_lesson, requested_tier)
        if seed is not None:
            episode_seed = int(seed)
        elif not self._seeded_once:
            episode_seed = self.initial_seed
        else:
            episode_seed = int(self.np_random.integers(0, 1_000_000))
        self._seeded_once = True
        self.tier = tier
        self.sim.reset(seed=episode_seed, tier=tier)
        if training_lesson:
            self._apply_training_lesson(training_lesson)
        self._distance_cache.clear()
        self._action_history: list[int] = []
        self._trace_history: list[float] = [self.sim.trace]
        self._idle_decisions = 0
        self._route_lower_bound = self._mission_route_lower_bound()
        self.reward_components = self._empty_rewards()
        self._previous_potential = self._mission_potential()
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action_value = int(action)
        action_mask = self.sim.action_mask()
        invalid = not 0 <= action_value < 36 or action_mask[int(np.clip(action_value, 0, 35))] == 0
        decoded = Action.decode(action_value)
        if invalid:
            decoded = Action(move=decoded.move)
        self._action_history.append(decoded.encode())
        self._idle_decisions += int(decoded.move == 0 and self.sim.active_hack_progress <= 0.0)

        before_data = self.sim.data
        before_integrity = self.sim.integrity
        before_trace = self.sim.trace
        before_explored = int(np.count_nonzero(self.sim.explored))
        self.sim.advance(decoded, ticks=POLICY_REPEAT)
        self._trace_history.append(self.sim.trace)
        potential = self._mission_potential()

        components = self._empty_rewards()
        components["extraction"] = 20.0 if self.sim.extracted else 0.0
        components["data"] = min(6.0, max(0, self.sim.data - before_data) * 1.5)
        components["progress"] = potential_progress_reward(self._previous_potential, potential)
        newly_explored = max(0, int(np.count_nonzero(self.sim.explored)) - before_explored)
        components["exploration"] = min(0.08, newly_explored * 0.008)
        components["trace"] = -0.006 * max(0.0, self.sim.trace - before_trace)
        components["damage"] = -3.0 * max(0, before_integrity - self.sim.integrity)
        components["time"] = -0.002
        components["idle"] = -0.006 if decoded.move == 0 and self.sim.active_hack_progress <= 0.0 else 0.0
        components["invalid"] = -0.02 if invalid else 0.0
        if (self.sim.terminated or self.sim.truncated) and not self.sim.extracted:
            components["failure"] = -10.0 if self.sim.fail_reason == "integrity_lost" else -6.0

        reward = float(sum(components.values()))
        for key, value in components.items():
            self.reward_components[key] += value
        self._previous_potential = potential
        info = self._info()
        if self.sim.terminated or self.sim.truncated:
            for key, value in self.reward_components.items():
                info[f"reward_{key}"] = float(value)
            info["reward_total"] = float(sum(self.reward_components.values()))
            info["episode_extra_stats"] = {
                "tier": int(self.tier),
                "success": float(self.sim.extracted),
                "data": float(self.sim.data),
                "damage": float(self.sim.damage_taken),
                "max_trace": float(self.sim.max_trace),
            }
            info["telemetry"] = self.telemetry()
        return self._observation(), reward, self.sim.terminated, self.sim.truncated, info

    def render(self) -> np.ndarray | bool | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from ghostline.presentation import GhostlineRenderer

            self._renderer = GhostlineRenderer(self.sim, visible=self.render_mode == "human")
        self._renderer.sim = self.sim
        return self._renderer.draw(return_array=self.render_mode == "rgb_array")

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def action_masks(self) -> np.ndarray:
        return self.sim.action_mask().astype(bool)

    def _empty_rewards(self) -> dict[str, float]:
        return {key: 0.0 for key in ("extraction", "data", "progress", "exploration", "trace", "damage", "time", "idle", "invalid", "failure")}

    def _mission_potential(self) -> float:
        selected = self.sim.objective_terminal()
        partial_data = max(
            (terminal.value * terminal.progress / terminal.hack_seconds for terminal in self.sim.level.terminals if not terminal.completed),
            default=0.0,
        )
        phase = min(1.0, (self.sim.data + partial_data) / max(1, self.sim.level.quota))
        goal = self.sim.level.extraction if selected is None else selected.position
        path_distance, _ = self._path_features(goal)
        diagonal = math.hypot(self.sim.level.world_width, self.sim.level.world_height)
        return 2.0 * phase + 1.0 - min(1.0, path_distance / max(1.0, diagonal))

    def _observation(self) -> dict[str, np.ndarray]:
        perceived = self._security_percepts()
        visible_positions = self._perceived_entity_positions(perceived)
        targets, target_mask = self._targets()
        entities, entity_mask = self._entities(perceived)
        return {
            "ego": self._ego(),
            "objective": self._objective(),
            "local_grid": self._local_grid(visible_positions),
            "targets": targets,
            "target_mask": target_mask,
            "entities": entities,
            "entity_mask": entity_mask,
            "rays": self._rays(visible_positions),
            "action_mask": self.sim.action_mask(),
        }

    def _ego(self) -> np.ndarray:
        sim = self.sim
        nearest_progress = sim.active_hack_progress
        values = np.asarray(
            [
                sim.velocity[0] / 230.0,
                sim.velocity[1] / 230.0,
                sim.heading[0],
                sim.heading[1],
                sim.integrity / 3.0 * 2.0 - 1.0,
                sim.trace / TRACE_MAX * 2.0 - 1.0,
                sim.trace_floor / TRACE_MAX * 2.0 - 1.0,
                sim.max_trace / TRACE_MAX * 2.0 - 1.0,
                sim.dash_energy / DASH_ENERGY_MAX * 2.0 - 1.0,
                sim.pulse_charges / MAX_PULSE_CHARGES * 2.0 - 1.0,
                min(1.0, sim.pulse_cooldown) * 2.0 - 1.0,
                min(1.0, sim.data / max(1, sim.level.quota)) * 2.0 - 1.0,
                min(1.0, sim.optional_data / 4.0) * 2.0 - 1.0,
                sim.remaining_seconds / sim.level.mission_seconds * 2.0 - 1.0,
                sim.tier / 6.0 * 2.0 - 1.0,
                sim.alert_tier / 4.0 * 2.0 - 1.0,
                float(sim.lockdown) * 2.0 - 1.0,
                float(sim.quota_met) * 2.0 - 1.0,
                nearest_progress * 2.0 - 1.0,
                sim.player[0] / sim.level.world_width * 2.0 - 1.0,
                sim.player[1] / sim.level.world_height * 2.0 - 1.0,
                sim.level.extraction[0] / sim.level.world_width * 2.0 - 1.0,
                sim.level.extraction[1] / sim.level.world_height * 2.0 - 1.0,
                min(1.0, sim.damage_taken / 3.0) * 2.0 - 1.0,
            ],
            dtype=np.float32,
        )
        return np.clip(values, -1.0, 1.0)

    def _objective(self) -> np.ndarray:
        sim = self.sim
        selected = sim.objective_terminal()
        goal = sim.level.extraction if selected is None else selected.position
        path_distance, waypoint = self._path_features(goal)
        delta = goal - sim.player
        diagonal = max(1.0, math.hypot(sim.level.world_width, sim.level.world_height))
        values = np.asarray(
            [
                1.0 if selected is None else -1.0,
                np.clip(delta[0] / sim.level.world_width, -1.0, 1.0),
                np.clip(delta[1] / sim.level.world_height, -1.0, 1.0),
                min(1.0, path_distance / diagonal) * 2.0 - 1.0,
                np.clip(waypoint[0] / (TILE_SIZE * 2), -1.0, 1.0),
                np.clip(waypoint[1] / (TILE_SIZE * 2), -1.0, 1.0),
                sim.active_hack_progress,
                ((selected.value / 3.0) * 2.0 - 1.0) if selected is not None else 1.0,
            ],
            dtype=np.float32,
        )
        return np.clip(values, -1.0, 1.0)

    def _apply_training_lesson(self, lesson: int) -> None:
        """Reverse-curriculum contracts used only in the training seed namespace."""
        terminals = self.sim.level.terminals
        if lesson == 1:
            terminals[0].position = self.sim.player.copy()
            terminals[0].value = 1
            terminals[0].hack_seconds = 0.45
            self.sim.level.terminals = terminals[:1]
            self.sim.level.quota = 1
            room = self.sim.level.rooms[0]
            candidate = tile_center((room.x + room.width - 2, room.y + room.height - 2))
            if self.sim._can_occupy(candidate, 9.0):
                self.sim.level.extraction = candidate
            self.sim.level.mission_seconds = 45
        elif lesson == 2:
            terminal = min(terminals, key=lambda item: norm(item.position - self.sim.player))
            terminal.terminal_id = 0
            terminal.value = max(1, terminal.value)
            terminal.hack_seconds = 0.75
            self.sim.level.terminals = [terminal]
            self.sim.level.quota = terminal.value
            self.sim.level.mission_seconds = 75
            self.sim.level.cameras = []
            self.sim.level.guards = []
            self.sim.level.response_drones = False
        elif lesson == 3:
            self.sim.level.cameras = []
            self.sim.level.guards = []
            self.sim.level.response_drones = False
        elif lesson == 4:
            self.sim.level.guards = []
            self.sim.level.response_drones = False
        elif lesson == 5:
            self.sim.level.response_drones = False
        elif lesson == 6:
            self.sim.level.response_drones = False

    def _mission_route_lower_bound(self) -> float:
        position = self.sim.level.spawn
        total = 0.0
        data = 0
        remaining = list(self.sim.level.terminals)
        while data < self.sim.level.quota and remaining:
            terminal = min(remaining, key=lambda item: norm(item.position - position) / max(1, item.value))
            total += norm(terminal.position - position)
            position = terminal.position
            data += terminal.value
            remaining.remove(terminal)
        total += norm(self.sim.level.extraction - position)
        return max(1.0, total)

    def telemetry(self) -> dict[str, Any]:
        distance = max(1.0, self.sim.distance_travelled)
        counts = np.bincount(np.asarray(self._action_history, dtype=np.int64), minlength=36)
        return {
            "seed": self.sim.seed,
            "tier": self.tier,
            "success": self.sim.extracted,
            "duration_seconds": self.sim.elapsed_seconds,
            "mean_trace": float(np.mean(self._trace_history)),
            "max_trace": self.sim.max_trace,
            "detections": self.sim.detections,
            "damage": self.sim.damage_taken,
            "distance_travelled": self.sim.distance_travelled,
            "path_efficiency": min(1.0, self._route_lower_bound / distance),
            "idle_decisions": self._idle_decisions,
            "decision_count": len(self._action_history),
            "action_counts": counts.tolist(),
            "trace_curve": [round(value, 3) for value in self._trace_history],
        }

    def _local_grid(self, visible_positions: list[tuple[np.ndarray, float]]) -> np.ndarray:
        sim = self.sim
        size = LOCAL_GRID_SIZE
        half = size // 2
        result = np.zeros((8, size, size), dtype=np.float32)
        px, py = world_to_tile(sim.player)
        terminal_tiles = {world_to_tile(t.position): t for t in sim.level.terminals if not t.completed}
        extraction_tile = world_to_tile(sim.level.extraction)
        for ly in range(size):
            for lx in range(size):
                tx, ty = px + lx - half, py + ly - half
                if not (0 <= ty < sim.level.grid.shape[0] and 0 <= tx < sim.level.grid.shape[1]):
                    result[1, ly, lx] = 1.0
                    continue
                explored = bool(sim.explored[ty, tx])
                result[3, ly, lx] = float(explored)
                tile = sim.level.grid[ty, tx]
                result[0, ly, lx] = float(tile != Tile.WALL)
                result[1, ly, lx] = float(tile == Tile.WALL or (tx, ty) in sim._blocked_tiles)
                result[2, ly, lx] = float(tile == Tile.DOOR)
                result[4, ly, lx] = float((tx, ty) in terminal_tiles)
                result[5, ly, lx] = float((tx, ty) == extraction_tile and sim.quota_met)
        result[7, half, half] = 1.0
        for entity_pos, dangerous in visible_positions:
            tx, ty = world_to_tile(entity_pos)
            lx, ly = tx - px + half, ty - py + half
            if 0 <= lx < size and 0 <= ly < size:
                result[6, ly, lx] = max(result[6, ly, lx], dangerous)
        return result

    def _targets(self) -> tuple[np.ndarray, np.ndarray]:
        result = np.zeros((MAX_TARGETS, 10), dtype=np.float32)
        mask = np.zeros(MAX_TARGETS, dtype=np.int8)
        for index, terminal in enumerate(self.sim.level.terminals[:MAX_TARGETS]):
            delta = terminal.position - self.sim.player
            path_distance, next_delta = self._path_features(terminal.position)
            result[index] = np.asarray(
                [
                    terminal.value / 3.0 * 2.0 - 1.0,
                    float(terminal.completed) * 2.0 - 1.0,
                    min(1.0, terminal.progress / terminal.hack_seconds) * 2.0 - 1.0,
                    np.clip(delta[0] / self.sim.level.world_width, -1.0, 1.0),
                    np.clip(delta[1] / self.sim.level.world_height, -1.0, 1.0),
                    min(1.0, norm(delta) / 900.0) * 2.0 - 1.0,
                    min(1.0, path_distance / 1400.0) * 2.0 - 1.0,
                    np.clip(next_delta[0] / (TILE_SIZE * 2), -1.0, 1.0),
                    np.clip(next_delta[1] / (TILE_SIZE * 2), -1.0, 1.0),
                    min(1.0, terminal.hack_seconds / 3.0) * 2.0 - 1.0,
                ],
                dtype=np.float32,
            )
            result[index] = np.clip(result[index], -1.0, 1.0)
            mask[index] = 1
        return result, mask

    def _security_percepts(self) -> list[PerceivedEntity]:
        """Build one current-best row for every piece of earned security intel."""

        sim = self.sim
        percepts: list[PerceivedEntity] = []
        for guard in sim.level.guards:
            key = ("guard", guard.guard_id)
            if sim.player_can_see(guard.position):
                percepts.append(
                    PerceivedEntity(
                        0,
                        guard.position,
                        guard.velocity,
                        guard.facing,
                        float(guard.mode) / float(max(GuardMode)),
                        1.0,
                        guard.grade,
                        0,
                    )
                )
                continue
            audible = sim.player_guard_audible_estimate(guard)
            if audible is not None:
                estimate, confidence = audible
                percepts.append(
                    PerceivedEntity(
                        0,
                        estimate,
                        np.zeros(2, dtype=np.float32),
                        None,
                        float(guard.mode) / float(max(GuardMode)),
                        confidence,
                        guard.grade,
                        1,
                    )
                )
                continue
            memory = sim.security_intel.get(key)
            if memory is not None:
                percepts.append(self._memory_percept(memory))

        for camera in sim.level.cameras:
            key = ("camera", camera.camera_id)
            if sim.player_can_see(camera.position):
                percepts.append(
                    PerceivedEntity(
                        1,
                        camera.position,
                        np.zeros(2, dtype=np.float32),
                        camera.angle,
                        float(camera.detected),
                        1.0,
                        GuardGrade.STANDARD,
                        0,
                    )
                )
            elif key in sim.security_intel:
                percepts.append(self._memory_percept(sim.security_intel[key]))

        for drone in sim.drones:
            key = ("drone", drone.drone_id)
            if sim.player_can_see(drone.position):
                percepts.append(
                    PerceivedEntity(
                        2,
                        drone.position,
                        np.zeros(2, dtype=np.float32),
                        drone.facing,
                        1.0,
                        1.0,
                        GuardGrade.STANDARD,
                        0,
                    )
                )
            elif key in sim.security_intel:
                percepts.append(self._memory_percept(sim.security_intel[key]))
        return percepts

    def _memory_percept(self, memory: SecurityIntel) -> PerceivedEntity:
        kind = {"guard": 0, "camera": 1, "drone": 2}[memory.kind]
        if memory.kind == "camera":
            confidence = 1.0
        else:
            age = max(0.0, (self.sim.elapsed_ticks - memory.last_seen_tick) / 60.0)
            confidence = max(0.51, 0.90 - age * 0.025)
        return PerceivedEntity(
            kind,
            memory.position,
            memory.velocity,
            memory.facing,
            memory.alert,
            confidence,
            memory.grade,
            2,
        )

    @staticmethod
    def _perceived_entity_positions(percepts: list[PerceivedEntity]) -> list[tuple[np.ndarray, float]]:
        entries: list[tuple[np.ndarray, float]] = []
        for percept in percepts:
            base = (1.0, 0.8, 1.0)[percept.kind]
            pressure = max(base * 0.45, percept.alert) * percept.confidence
            entries.append((percept.position, pressure))
        return entries

    def _entities(self, percepts: list[PerceivedEntity] | None = None) -> tuple[np.ndarray, np.ndarray]:
        records: list[tuple[int, float, np.ndarray]] = []
        sim = self.sim
        for percept in percepts if percepts is not None else self._security_percepts():
            delta = percept.position - sim.player
            record = self._entity_record(
                percept.kind,
                delta,
                percept.velocity,
                percept.facing,
                percept.alert,
                percept.confidence,
                percept.grade,
            )
            records.append((percept.priority, norm(delta), record))
        records.sort(key=lambda item: (item[0], item[1]))
        output = np.zeros((MAX_ENTITIES, 13), dtype=np.float32)
        mask = np.zeros(MAX_ENTITIES, dtype=np.int8)
        for index, (_, _, record) in enumerate(records[:MAX_ENTITIES]):
            output[index] = record
            mask[index] = 1
        return output, mask

    def _entity_record(
        self,
        kind: int,
        delta: np.ndarray,
        velocity: np.ndarray,
        facing: float | None,
        alert: float,
        confidence: float,
        grade: GuardGrade,
    ) -> np.ndarray:
        types = [0.0, 0.0, 0.0]
        types[kind] = 1.0
        facing_sin = 0.0 if facing is None else math.sin(facing)
        facing_cos = 0.0 if facing is None else math.cos(facing)
        grade_value = float(grade) - 1.0 if kind == 0 else -1.0
        raw = np.asarray(
            [
                *(value * 2.0 - 1.0 for value in types),
                np.clip(delta[0] / PLAYER_PERCEPTION_DISTANCE, -1.0, 1.0),
                np.clip(delta[1] / PLAYER_PERCEPTION_DISTANCE, -1.0, 1.0),
                min(1.0, norm(delta) / PLAYER_PERCEPTION_DISTANCE) * 2.0 - 1.0,
                np.clip(velocity[0] / 120.0, -1.0, 1.0),
                np.clip(velocity[1] / 120.0, -1.0, 1.0),
                facing_sin,
                facing_cos,
                np.clip(alert, 0.0, 1.0) * 2.0 - 1.0,
                confidence * 2.0 - 1.0,
                grade_value,
            ],
            dtype=np.float32,
        )
        return np.clip(raw, -1.0, 1.0)

    def _rays(self, visible_positions: list[tuple[np.ndarray, float]]) -> np.ndarray:
        result = np.zeros((RAY_COUNT, 3), dtype=np.float32)
        max_distance = PLAYER_PERCEPTION_DISTANCE
        for index, angle in enumerate(np.linspace(0.0, math.tau, RAY_COUNT, endpoint=False)):
            direction = angle_vector(float(angle))
            geometry_distance = max_distance
            explored_fraction = 1.0
            for step in range(1, 41):
                distance = step * 8.0
                point = self.sim.player + direction * distance
                tile = world_to_tile(point)
                if not (0 <= tile[1] < self.sim.level.grid.shape[0] and 0 <= tile[0] < self.sim.level.grid.shape[1]):
                    geometry_distance = distance
                    break
                if not self.sim.explored[tile[1], tile[0]]:
                    explored_fraction = min(explored_fraction, distance / max_distance)
                if tile in self.sim._blocked_tiles or int(self.sim.level.grid[tile[1], tile[0]]) == 1:
                    geometry_distance = distance
                    break
            danger = 0.0
            for entity_pos, pressure in visible_positions:
                delta = entity_pos - self.sim.player
                distance = norm(delta)
                if distance and float(np.dot(delta / distance, direction)) > 0.94:
                    danger = max(danger, pressure * (1.0 - min(1.0, distance / max_distance)))
            result[index] = (geometry_distance / max_distance, danger, explored_fraction)
        return result

    def _path_features(self, goal: np.ndarray) -> tuple[float, np.ndarray]:
        start = world_to_tile(self.sim.player)
        target = world_to_tile(goal)
        if start == target:
            return norm(goal - self.sim.player), goal - self.sim.player
        if self.sim.line_of_sight(self.sim.player, goal):
            return norm(goal - self.sim.player), goal - self.sim.player
        distance_map = self._distance_cache.get(target)
        if distance_map is None:
            distance_map = np.full(self.sim.level.grid.shape, -1, dtype=np.int32)
            distance_map[target[1], target[0]] = 0
            queue = deque([target])
            while queue:
                x, y = queue.popleft()
                distance = int(distance_map[y, x])
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nxt = (x + dx, y + dy)
                    if nxt in self.sim._blocked_tiles:
                        continue
                    if not (0 <= nxt[1] < self.sim.level.grid.shape[0] and 0 <= nxt[0] < self.sim.level.grid.shape[1]):
                        continue
                    if distance_map[nxt[1], nxt[0]] >= 0 or self.sim.level.grid[nxt[1], nxt[0]] == Tile.WALL:
                        continue
                    distance_map[nxt[1], nxt[0]] = distance + 1
                    queue.append(nxt)
            self._distance_cache[target] = distance_map
        start_distance = int(distance_map[start[1], start[0]])
        if start_distance < 0:
            return norm(goal - self.sim.player), goal - self.sim.player
        # Use a visible multi-tile look-ahead rather than the center of the next
        # tile. A one-tile waypoint can flip behind a fast-moving runner before
        # velocity settles, producing left/right oscillation at furniture and
        # doors. The farther visible gradient point is both a smoother policy
        # signal and a better approximation of the HUD's next-door bearing.
        current = start
        current_distance = start_distance
        waypoint = tile_center(start)
        for _ in range(6):
            candidates = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (current[0] + dx, current[1] + dy)
                if not (0 <= nxt[1] < distance_map.shape[0] and 0 <= nxt[0] < distance_map.shape[1]):
                    continue
                distance = int(distance_map[nxt[1], nxt[0]])
                if 0 <= distance < current_distance:
                    candidates.append(nxt)
            if not candidates:
                break
            following = min(candidates, key=lambda tile: (distance_map[tile[1], tile[0]], tile[1], tile[0]))
            candidate = tile_center(following)
            if not self.sim.line_of_sight(self.sim.player, candidate):
                break
            waypoint = candidate
            current = following
            current_distance = int(distance_map[current[1], current[0]])
            if current == target:
                waypoint = goal
                break
        return float(start_distance * TILE_SIZE), waypoint - self.sim.player

    def _info(self) -> dict[str, Any]:
        info = self.sim.terminal_info()
        info["action_mask"] = self.sim.action_mask()
        info["campaign_tier_name"] = TIERS[self.tier].name
        return info


class GhostlineEnvV1(GhostlineEnv):
    """Shape-compatible v1 baseline retained for legacy evaluation."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        spaces_v1 = dict(self.observation_space.spaces)
        spaces_v1.pop("objective")
        self.observation_space = spaces.Dict(spaces_v1)

    def _observation(self) -> dict[str, np.ndarray]:
        observation = super()._observation()
        observation.pop("objective")
        return observation
