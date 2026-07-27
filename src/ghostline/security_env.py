from __future__ import annotations

from enum import IntEnum
import math
from typing import Any, Callable

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from ghostline.config import (
    GUARD_VISION_BASE_DISTANCE,
    GUARD_VISION_COSINE,
    GUARD_VISION_DISTANCE_PER_ALERT,
    LOCAL_GRID_SIZE,
    PLAYER_PERCEPTION_DISTANCE,
    TILE_SIZE,
)
from ghostline.config_v3 import (
    MAX_RADIO_MESSAGES,
    MAX_SECURITY_TARGETS,
    MAX_TEAMMATES,
    SECURITY_CENTRAL_STATE_SIZE,
    SECURITY_TACTICAL_TICKS,
    SECURITY_TARGET_FEATURES,
    SECURITY_TARGET_KINDS,
)
from ghostline.generation import tile_center, world_to_tile
from ghostline.policies import FairScriptedPolicy
from ghostline.simulation import norm, unit
from ghostline.simulation_v3 import GhostlineSimulationV3
from ghostline.types import Guard, GuardMode, Tile
from ghostline.types_v3 import GuardRole, RadioMessage, RunnerActionV3, SecurityIntent, SecurityOrder

RunnerController = Callable[[GhostlineSimulationV3], int]


class TargetKind(IntEnum):
    """One code per tactical target slot exposed to the operative policy."""

    PATROL = 0
    CONTACT = 1
    HEARD = 2
    TERMINAL = 3
    EXTRACTION = 4
    DOOR = 5
    FLANK_LEFT = 6
    FLANK_RIGHT = 7


assert len(TargetKind) == SECURITY_TARGET_KINDS
TARGET_FEATURES = SECURITY_TARGET_FEATURES
CENTRAL_STATE_SIZE = SECURITY_CENTRAL_STATE_SIZE


def _capped_radio_credit(before: int, after: int, team_size: int) -> float:
    """Credit only the first information broadcast to each possible teammate."""

    cap = max(0, int(team_size) - 1)
    return 0.005 * max(0, min(int(after), cap) - min(int(before), cap))


class GhostlineSecurityParallelEnv(ParallelEnv):
    """Simultaneous, partially observed operative-control benchmark."""

    metadata = {"name": "GhostlineSecurityParallel-v0", "render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        *,
        tier: int = 6,
        seed: int = 20_000_000,
        runner: RunnerController | None = None,
        render_mode: str | None = None,
    ):
        self.tier = int(tier)
        self.initial_seed = int(seed)
        self.render_mode = render_mode
        self.possible_agents = [f"guard_{index}" for index in range(5)]
        self.agent_name_mapping = {agent: index for index, agent in enumerate(self.possible_agents)}
        self._observation_space = spaces.Dict(
            {
                "ego": spaces.Box(-1.0, 1.0, shape=(18,), dtype=np.float32),
                "local_grid": spaces.Box(0.0, 1.0, shape=(8, LOCAL_GRID_SIZE, LOCAL_GRID_SIZE), dtype=np.float32),
                "runner": spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32),
                "teammates": spaces.Box(-1.0, 1.0, shape=(MAX_TEAMMATES, 12), dtype=np.float32),
                "teammate_mask": spaces.Box(0, 1, shape=(MAX_TEAMMATES,), dtype=np.int8),
                "targets": spaces.Box(-1.0, 1.0, shape=(MAX_SECURITY_TARGETS, TARGET_FEATURES), dtype=np.float32),
                "target_mask": spaces.Box(0, 1, shape=(MAX_SECURITY_TARGETS,), dtype=np.int8),
                "radio": spaces.Box(-1.0, 1.0, shape=(MAX_RADIO_MESSAGES, 8), dtype=np.float32),
                "radio_mask": spaces.Box(0, 1, shape=(MAX_RADIO_MESSAGES,), dtype=np.int8),
                "intent_mask": spaces.Box(0, 1, shape=(8,), dtype=np.int8),
                "message_mask": spaces.Box(0, 1, shape=(5,), dtype=np.int8),
                "ability_mask": spaces.Box(0, 1, shape=(2,), dtype=np.int8),
            }
        )
        self._action_space = spaces.MultiDiscrete([8, MAX_SECURITY_TARGETS, 5, 2])
        self.observation_spaces = {agent: self._observation_space for agent in self.possible_agents}
        self.action_spaces = {agent: self._action_space for agent in self.possible_agents}
        self.state_space = spaces.Box(-1.0, 1.0, shape=(CENTRAL_STATE_SIZE,), dtype=np.float32)
        self._scripted_runner = FairScriptedPolicy()
        self.runner = runner or self._scripted_runner.act
        self.sim = GhostlineSimulationV3(seed=self.initial_seed, tier=self.tier, external_security=True)
        self.agents: list[str] = []
        self._renderer = None
        self._target_cache: dict[int, list[np.ndarray]] = {}
        self._last_runner_action = 0
        self._last_seed = self.initial_seed
        self._invalid_actions = 0
        self._last_reward_components: dict[str, float] = {}
        self._last_agent_shaping: dict[str, float] = {}
        self._current_observations: dict[str, dict[str, np.ndarray]] = {}
        self._plane_signature: tuple[Any, ...] | None = None
        self._plane_cache: np.ndarray | None = None

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        options = dict(options or {})
        if seed is not None:
            self._last_seed = int(seed)
        elif "seed" in options:
            self._last_seed = int(options["seed"])
        self.tier = int(options.get("tier", self.tier))
        self.sim.external_security = True
        self.sim.reset(seed=self._last_seed, tier=self.tier)
        reset_runner = getattr(self.runner, "reset", None)
        if callable(reset_runner):
            reset_runner(self.sim)
        self.agents = [f"guard_{guard.guard_id}" for guard in self.sim.level.guards]
        self._target_cache.clear()
        self._last_runner_action = 0
        self._invalid_actions = 0
        self._last_reward_components = {}
        self._last_agent_shaping = {}
        self._plane_signature = None
        self._plane_cache = None
        observations = {agent: self._observation(agent) for agent in self.agents}
        self._current_observations = observations
        infos = {agent: self._info(agent) for agent in self.agents}
        return observations, infos

    def step(
        self,
        actions: dict[str, np.ndarray | list[int] | tuple[int, ...]],
    ) -> tuple[
        dict[str, dict[str, np.ndarray]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        active_agents = list(self.agents)
        before_damage = self.sim.damage_taken
        before_detections = self.sim.detections
        before_data = self.sim.data
        before_radio = sum(state.radio_assists for state in self.sim.operative_states.values())
        before_potential = self._security_potential()
        before_agent_potentials = self._agent_potentials()
        orders, invalid = self.orders_from_actions(actions, observations=self._current_observations)
        self._invalid_actions += invalid
        self.sim.set_security_orders(orders)
        for tick in range(SECURITY_TACTICAL_TICKS):
            if tick % 6 == 0:
                self._last_runner_action = int(self.runner(self.sim))
            self.sim.advance(RunnerActionV3.decode(self._last_runner_action), ticks=1)
            if self.sim.terminated or self.sim.truncated:
                break

        terminal = bool(self.sim.terminated or self.sim.truncated)
        after_potential = 0.0 if terminal else self._security_potential()
        after_radio = sum(state.radio_assists for state in self.sim.operative_states.values())
        reward_components = {
            "damage": 5.0 * max(0, self.sim.damage_taken - before_damage),
            "detection": 0.08 * max(0, self.sim.detections - before_detections),
            "runner_data": -0.50 * max(0, self.sim.data - before_data),
            "survival": 0.01,
            "radio_assist": _capped_radio_credit(before_radio, after_radio, len(active_agents)),
            "invalid_action": -0.02 * invalid,
            "formation": -self._formation_penalty(),
            "potential": 0.995 * after_potential - before_potential,
            "terminal": -20.0 if self.sim.extracted else 20.0 if terminal else 0.0,
        }
        # Discount-matched potential shaping supplies pursuit/containment signal
        # without changing the optimal terminal objective.
        reward = float(sum(reward_components.values()))
        reward_components["total"] = reward
        self._last_reward_components = reward_components

        # Shared outcome, individual shaping.  Every operative previously
        # received the identical scalar, so with parameter sharing across five
        # agents nothing distinguished whose action mattered -- the classic
        # lazy-agent setup.  The terminal, damage, and data terms stay global
        # because they are genuinely team outcomes; only the discount-matched
        # containment shaping is attributed to the operative that earned it.
        after_agent_potentials = {} if terminal else self._agent_potentials()
        self._last_agent_shaping = {}
        rewards: dict[str, float] = {}
        for agent in active_agents:
            guard_id = self.agent_name_mapping[agent]
            before_value = before_agent_potentials.get(guard_id, 0.0)
            after_value = after_agent_potentials.get(guard_id, 0.0)
            shaping = 0.995 * after_value - before_value
            self._last_agent_shaping[agent] = float(shaping)
            rewards[agent] = float(reward + shaping)

        terminated = bool(self.sim.terminated)
        truncated = bool(self.sim.truncated)
        terminations = {agent: terminated for agent in active_agents}
        truncations = {agent: truncated for agent in active_agents}
        infos = {agent: self._info(agent) for agent in active_agents}
        if terminated or truncated:
            # Preserve each operative's final local frame alongside its
            # terminal flag.  This follows the Parallel API convention and is
            # useful for recurrent diagnostics without reviving the agents.
            observations = {agent: self._observation(agent) for agent in active_agents}
            self.agents = []
        else:
            observations = {agent: self._observation(agent) for agent in active_agents}
        self._current_observations = observations
        return observations, rewards, terminations, truncations, infos

    def _runner_goal(self) -> np.ndarray:
        """Where the runner is actually trying to reach right now."""

        if self.sim.quota_met:
            return self.sim.level.extraction
        terminal = self.sim.objective_terminal()
        return terminal.position if terminal is not None else self.sim.level.extraction

    def _interception_score(self, guard: Guard) -> float:
        """How well this operative cuts the runner off, in [0, 1].

        Raw proximity was the wrong objective.  A guard moves at 95-99% of the
        runner's speed, so it can never win a tail chase; rewarding closeness
        trained exactly the losing behaviour.  What actually converts sight into
        contact is standing between the runner and where it is going, so this
        scores progress along the runner's own route instead of distance to it.
        """

        goal = self._runner_goal()
        runner_to_goal = goal - self.sim.player
        route_length = float(norm(runner_to_goal))
        if route_length < 1.0:
            return 1.0
        direction = runner_to_goal / route_length
        guard_offset = guard.position - self.sim.player
        # Distance travelled along the runner's route, clamped to the segment.
        along = float(np.dot(guard_offset, direction))
        lateral = float(norm(guard_offset - direction * along))
        if along <= 0.0:
            return 0.0  # behind the runner: a pure tail chase, worth nothing
        ahead = min(1.0, along / route_length)
        # Being far off the route line is worth little even when ahead.
        corridor = 1.0 / (1.0 + (lateral / (TILE_SIZE * 3.0)) ** 2)
        return float(np.clip(ahead * corridor, 0.0, 1.0))

    def _security_potential(self) -> float:
        """Team potential.  Containment and pressure, not proximity."""

        interception = max((self._interception_score(guard) for guard in self.sim.level.guards), default=0.0)
        awareness = max((guard.awareness for guard in self.sim.level.guards), default=0.0)
        partial_link = self.sim.active_hack_progress
        mission_progress = min(1.0, (self.sim.data + partial_link) / max(1.0, self.sim.level.quota))
        # ``damage_taken`` is deliberately absent: it is already paid directly by
        # the damage component, and a monotone counter inside a potential never
        # telescopes back to its starting value.
        return (
            0.90 * interception
            + 0.35 * awareness
            + 0.25 * self.sim.trace / 100.0
            - 2.00 * mission_progress
        )

    def _agent_potentials(self) -> dict[int, float]:
        """Per-operative potential used for credit assignment."""

        return {
            guard.guard_id: 0.60 * self._interception_score(guard) + 0.30 * guard.awareness
            for guard in self.sim.level.guards
        }

    def orders_from_actions(
        self,
        actions: dict[str, np.ndarray | list[int] | tuple[int, ...]],
        *,
        observations: dict[str, dict[str, np.ndarray]] | None = None,
    ) -> tuple[dict[int, SecurityOrder], int]:
        """Validate semantic actions against the same masks used by training."""

        invalid = 0
        orders: dict[int, SecurityOrder] = {}
        for agent in self.agents:
            guard_id = self.agent_name_mapping[agent]
            observation = observations[agent] if observations is not None and agent in observations else self._observation(agent)
            raw = np.asarray(actions.get(agent, (0, 0, 0, 0)), dtype=np.int64).reshape(-1)
            if raw.shape != (4,):
                raw = np.zeros(4, dtype=np.int64)
                invalid += 1
            intent_value = int(np.clip(raw[0], 0, 7))
            target_value = int(np.clip(raw[1], 0, MAX_SECURITY_TARGETS - 1))
            message_value = int(np.clip(raw[2], 0, 4))
            ability_value = int(np.clip(raw[3], 0, 1))
            if (
                observation["intent_mask"][intent_value] == 0
                or observation["target_mask"][target_value] == 0
                or observation["message_mask"][message_value] == 0
                or observation["ability_mask"][ability_value] == 0
            ):
                invalid += 1
                intent_value, message_value, ability_value = 0, 0, 0
                target_value = int(np.flatnonzero(observation["target_mask"])[0])
            targets = self._target_cache[guard_id]
            orders[guard_id] = SecurityOrder(
                SecurityIntent(intent_value),
                targets[target_value].copy(),
                RadioMessage(message_value),
                bool(ability_value),
            )
        return orders, invalid

    def _guard(self, agent: str) -> Guard:
        guard_id = self.agent_name_mapping[agent]
        return next(guard for guard in self.sim.level.guards if guard.guard_id == guard_id)

    def _observation(self, agent: str) -> dict[str, np.ndarray]:
        guard = self._guard(agent)
        runner, runner_visible, runner_audible, confidence, contact = self._runner_contact(guard)
        teammates, teammate_mask = self._teammates(guard)
        targets, target_mask, positions = self._targets(guard, contact, confidence)
        self._target_cache[guard.guard_id] = positions
        radio, radio_mask = self._radio(guard)
        intent_mask = np.ones(8, dtype=np.int8)
        intent_mask[int(SecurityIntent.PURSUE)] = int(runner_visible)
        intent_mask[int(SecurityIntent.INTERCEPT)] = int(bool(self.sim.security_doors))
        intent_mask[int(SecurityIntent.FLANK_LEFT)] = int(confidence > 0.0)
        intent_mask[int(SecurityIntent.FLANK_RIGHT)] = int(confidence > 0.0)
        message_mask = np.ones(5, dtype=np.int8)
        if guard.radio_jammed_for > 0.0:
            message_mask[1:] = 0
        state = self.sim.operative_states[guard.guard_id]
        distance = norm(self.sim.player - guard.position)
        fire_legal = (
            state.role == GuardRole.SUPPRESSOR
            and state.weapon_cooldown <= 0.0
            and runner_visible
            and 96.0 <= distance <= 240.0
        )
        return {
            "ego": self._ego(guard),
            "local_grid": self._local_grid(guard, runner if confidence > 0.0 else None),
            "runner": self._runner_record(guard, runner, runner_visible, runner_audible, confidence),
            "teammates": teammates,
            "teammate_mask": teammate_mask,
            "targets": targets,
            "target_mask": target_mask,
            "radio": radio,
            "radio_mask": radio_mask,
            "intent_mask": intent_mask,
            "message_mask": message_mask,
            "ability_mask": np.asarray((1, int(fire_legal)), dtype=np.int8),
        }

    def _ego(self, guard: Guard) -> np.ndarray:
        state = self.sim.operative_states[guard.guard_id]
        role = np.zeros(3, dtype=np.float32)
        role[int(state.role)] = 1.0
        grade = np.zeros(3, dtype=np.float32)
        grade[int(guard.grade)] = 1.0
        values = np.asarray(
            [
                *role,
                *grade,
                guard.position[0] / self.sim.level.world_width * 2.0 - 1.0,
                guard.position[1] / self.sim.level.world_height * 2.0 - 1.0,
                math.sin(guard.facing),
                math.cos(guard.facing),
                np.clip(guard.velocity[0] / 126.0, -1.0, 1.0),
                np.clip(guard.velocity[1] / 126.0, -1.0, 1.0),
                float(guard.mode) / 5.0 * 2.0 - 1.0,
                guard.awareness * 2.0 - 1.0,
                min(1.0, state.weapon_cooldown / 2.4) * 2.0 - 1.0,
                min(1.0, state.aim_progress / 0.7) * 2.0 - 1.0,
                min(1.0, guard.radio_jammed_for / 5.0) * 2.0 - 1.0,
                self.sim.alert_tier / 4.0 * 2.0 - 1.0,
            ],
            dtype=np.float32,
        )
        return np.clip(values, -1.0, 1.0)

    def _runner_contact(self, guard: Guard) -> tuple[np.ndarray, bool, bool, float, np.ndarray]:
        # Sight uses the simulation's own guard envelope.  A hardcoded
        # 245 px / cos 0.45 cone previously reported the runner visible when the
        # guard could not detect it, and the sign of that error flipped at high
        # alert where the real envelope grows past 245 px.  ``PURSUE`` legality
        # is gated on this flag, so the legal action set has to agree with the
        # detection model that produces awareness.
        sight_distance = GUARD_VISION_BASE_DISTANCE + GUARD_VISION_DISTANCE_PER_ALERT * self.sim.alert_tier
        visible = self.sim.visible(
            guard.position,
            guard.facing,
            self.sim.player,
            distance=sight_distance,
            cosine=GUARD_VISION_COSINE,
        )
        state = self.sim.operative_states[guard.guard_id]
        audible = norm(guard.position - self.sim.player) <= PLAYER_PERCEPTION_DISTANCE * 0.48 and norm(self.sim.velocity) > 42.0
        if visible:
            return self.sim.player.copy(), True, audible, 1.0, self.sim.player.copy()
        if audible:
            strength = 1.0 - norm(guard.position - self.sim.player) / (PLAYER_PERCEPTION_DISTANCE * 0.48)
            estimate = self._quantized_audio_estimate(guard)
            return estimate, False, True, max(0.15, strength), estimate
        if state.heard_confidence > 0.02:
            return state.heard_position.copy(), False, True, state.heard_confidence, state.heard_position.copy()
        if guard.mode in (GuardMode.INVESTIGATE, GuardMode.SEARCH, GuardMode.CHASE):
            return guard.last_known.copy(), False, False, 0.45, guard.last_known.copy()
        return guard.position.copy(), False, False, 0.0, guard.position.copy()

    def _quantized_audio_estimate(self, guard: Guard) -> np.ndarray:
        """Return a coarse heard position, symmetric with the runner's own cue.

        The previous estimate was ``player + fixed_offset(guard_id)``.  Because
        the offset was constant per operative it was trivially invertible, so
        "heard" was really "knows exactly".  This mirrors
        ``GhostlineSimulation.player_guard_audible_estimate``: twelve bearing
        sectors and 48 px range bands, measured from the listener outward, so
        both sides of the match hear with the same fidelity.  It stays
        deterministic, which replay and evaluation both depend on.
        """

        delta = self.sim.player - guard.position
        distance = norm(delta)
        sector_width = math.tau / 12.0
        angle = round(math.atan2(float(delta[1]), float(delta[0])) / sector_width) * sector_width
        range_band = max(24.0, round(distance / 48.0) * 48.0)
        estimate = guard.position + np.asarray((math.cos(angle), math.sin(angle)), dtype=np.float32) * range_band
        return estimate.astype(np.float32)

    def _runner_record(self, guard: Guard, position: np.ndarray, visible: bool, audible: bool, confidence: float) -> np.ndarray:
        delta = position - guard.position
        velocity = self.sim.velocity if visible else np.zeros(2, dtype=np.float32)
        record = np.asarray(
            [
                np.clip(delta[0] / PLAYER_PERCEPTION_DISTANCE, -1.0, 1.0),
                np.clip(delta[1] / PLAYER_PERCEPTION_DISTANCE, -1.0, 1.0),
                min(1.0, norm(delta) / PLAYER_PERCEPTION_DISTANCE) * 2.0 - 1.0,
                np.clip(velocity[0] / 230.0, -1.0, 1.0),
                np.clip(velocity[1] / 230.0, -1.0, 1.0),
                float(visible) * 2.0 - 1.0,
                float(audible) * 2.0 - 1.0,
                confidence * 2.0 - 1.0,
                self.sim.heading[0] if visible else 0.0,
                self.sim.heading[1] if visible else 0.0,
                self.sim.trace / 100.0 * 2.0 - 1.0,
                float(self.sim.quota_met) * 2.0 - 1.0,
            ],
            dtype=np.float32,
        )
        return np.clip(record, -1.0, 1.0)

    def _local_grid(self, guard: Guard, contact: np.ndarray | None) -> np.ndarray:
        result = np.zeros((8, LOCAL_GRID_SIZE, LOCAL_GRID_SIZE), dtype=np.float32)
        gx, gy = world_to_tile(guard.position)
        half = LOCAL_GRID_SIZE // 2
        planes = self._shared_spatial_planes()
        height, width = self.sim.level.grid.shape
        world_x0, world_y0 = gx - half, gy - half
        source_x0, source_y0 = max(0, world_x0), max(0, world_y0)
        source_x1, source_y1 = min(width, world_x0 + LOCAL_GRID_SIZE), min(height, world_y0 + LOCAL_GRID_SIZE)
        destination_x0, destination_y0 = source_x0 - world_x0, source_y0 - world_y0
        destination_x1 = destination_x0 + max(0, source_x1 - source_x0)
        destination_y1 = destination_y0 + max(0, source_y1 - source_y0)
        if source_x1 > source_x0 and source_y1 > source_y0:
            result[:6, destination_y0:destination_y1, destination_x0:destination_x1] = planes[
                :, source_y0:source_y1, source_x0:source_x1
            ]
        # Off-map cells are solid even though the cached world planes have no
        # representation outside the generated grid.
        if destination_y0 > 0:
            result[1, :destination_y0, :] = 1.0
        if destination_y1 < LOCAL_GRID_SIZE:
            result[1, destination_y1:, :] = 1.0
        if destination_x0 > 0:
            result[1, :, :destination_x0] = 1.0
        if destination_x1 < LOCAL_GRID_SIZE:
            result[1, :, destination_x1:] = 1.0
        for teammate in self.sim.level.guards:
            if teammate is guard:
                continue
            tx, ty = world_to_tile(teammate.position)
            lx, ly = tx - gx + half, ty - gy + half
            if 0 <= lx < LOCAL_GRID_SIZE and 0 <= ly < LOCAL_GRID_SIZE:
                result[6, ly, lx] = 1.0
        if contact is not None:
            tx, ty = world_to_tile(contact)
            lx, ly = tx - gx + half, ty - gy + half
            if 0 <= lx < LOCAL_GRID_SIZE and 0 <= ly < LOCAL_GRID_SIZE:
                result[7, ly, lx] = 1.0
        return result

    def _shared_spatial_planes(self) -> np.ndarray:
        """Cache the six facility layers shared by every operative this tick."""

        signature = (
            id(self.sim),
            tuple(bool(terminal.completed) for terminal in self.sim.level.terminals),
            tuple((door.tile, bool(door.locked)) for door in self.sim.security_doors),
        )
        if signature == self._plane_signature and self._plane_cache is not None:
            return self._plane_cache
        grid = self.sim.level.grid
        result = np.zeros((6, *grid.shape), dtype=np.float32)
        result[0] = grid != Tile.WALL
        result[1] = grid == Tile.WALL
        for tx, ty in self.sim._blocked_tiles:
            if 0 <= ty < grid.shape[0] and 0 <= tx < grid.shape[1]:
                result[1, ty, tx] = 1.0
        result[2] = grid == Tile.DOOR
        for door in self.sim.security_doors:
            if door.locked:
                result[3, door.tile[1], door.tile[0]] = 1.0
        for terminal in self.sim.level.terminals:
            if not terminal.completed:
                tx, ty = world_to_tile(terminal.position)
                result[4, ty, tx] = 1.0
        extraction_x, extraction_y = world_to_tile(self.sim.level.extraction)
        result[5, extraction_y, extraction_x] = 1.0
        self._plane_signature = signature
        self._plane_cache = result
        return result

    def _teammates(self, guard: Guard) -> tuple[np.ndarray, np.ndarray]:
        result = np.zeros((MAX_TEAMMATES, 12), dtype=np.float32)
        mask = np.zeros(MAX_TEAMMATES, dtype=np.int8)
        others = sorted((other for other in self.sim.level.guards if other is not guard), key=lambda other: norm(other.position - guard.position))
        for index, other in enumerate(others[:MAX_TEAMMATES]):
            delta = other.position - guard.position
            state = self.sim.operative_states[other.guard_id]
            result[index] = np.asarray(
                [
                    np.clip(delta[0] / PLAYER_PERCEPTION_DISTANCE, -1.0, 1.0),
                    np.clip(delta[1] / PLAYER_PERCEPTION_DISTANCE, -1.0, 1.0),
                    min(1.0, norm(delta) / PLAYER_PERCEPTION_DISTANCE) * 2.0 - 1.0,
                    np.clip(other.velocity[0] / 126.0, -1.0, 1.0),
                    np.clip(other.velocity[1] / 126.0, -1.0, 1.0),
                    math.sin(other.facing),
                    math.cos(other.facing),
                    float(other.mode) / 5.0 * 2.0 - 1.0,
                    float(state.role) - 1.0,
                    other.awareness * 2.0 - 1.0,
                    min(1.0, other.radio_jammed_for / 5.0) * 2.0 - 1.0,
                    float(state.current_order.intent) / 7.0 * 2.0 - 1.0,
                ],
                dtype=np.float32,
            )
            mask[index] = 1
        return result, mask

    def _targets(self, guard: Guard, contact: np.ndarray, confidence: float) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        # One kind per semantic slot.  The extraction relay and the nearest
        # security door previously shared kind 4, so the policy could not tell
        # an exit from a chokepoint even though INTERCEPT legality depends on
        # doors; the two flank offsets also collapsed onto the contact kind.
        candidates: list[tuple[np.ndarray, int, bool]] = [
            (guard.patrol[guard.patrol_index].copy(), TargetKind.PATROL, True),
            (contact.copy(), TargetKind.CONTACT, confidence > 0.0),
            (
                self.sim.operative_states[guard.guard_id].heard_position.copy(),
                TargetKind.HEARD,
                self.sim.operative_states[guard.guard_id].heard_confidence > 0.02,
            ),
        ]
        unfinished = [terminal for terminal in self.sim.level.terminals if not terminal.completed]
        nearest_terminal = min(unfinished, key=lambda terminal: norm(terminal.position - guard.position), default=None)
        candidates.append(
            ((nearest_terminal.position if nearest_terminal else guard.position).copy(), TargetKind.TERMINAL, nearest_terminal is not None)
        )
        candidates.append((self.sim.level.extraction.copy(), TargetKind.EXTRACTION, self.sim.quota_met))
        nearest_door = min(self.sim.security_doors, key=lambda door: norm(tile_center(door.tile) - guard.position), default=None)
        candidates.append(
            ((tile_center(nearest_door.tile) if nearest_door else guard.position).copy(), TargetKind.DOOR, nearest_door is not None)
        )
        offset = unit(contact - guard.position)
        perpendicular = np.asarray((-offset[1], offset[0]), dtype=np.float32) * TILE_SIZE * 2.0
        candidates.append(((contact + perpendicular).astype(np.float32), TargetKind.FLANK_LEFT, confidence > 0.0))
        candidates.append(((contact - perpendicular).astype(np.float32), TargetKind.FLANK_RIGHT, confidence > 0.0))

        result = np.zeros((MAX_SECURITY_TARGETS, TARGET_FEATURES), dtype=np.float32)
        mask = np.zeros(MAX_SECURITY_TARGETS, dtype=np.int8)
        positions: list[np.ndarray] = []
        for index, (position, kind, valid) in enumerate(candidates[:MAX_SECURITY_TARGETS]):
            delta = position - guard.position
            one_hot = np.zeros(len(TargetKind), dtype=np.float32)
            one_hot[int(kind)] = 1.0
            result[index] = np.asarray(
                [
                    np.clip(delta[0] / self.sim.level.world_width, -1.0, 1.0),
                    np.clip(delta[1] / self.sim.level.world_height, -1.0, 1.0),
                    min(1.0, norm(delta) / 900.0) * 2.0 - 1.0,
                    *one_hot,
                ],
                dtype=np.float32,
            )
            mask[index] = int(valid)
            positions.append(position.astype(np.float32))
        if not np.any(mask):
            mask[0] = 1
        return result, mask, positions

    def _radio(self, guard: Guard) -> tuple[np.ndarray, np.ndarray]:
        result = np.zeros((MAX_RADIO_MESSAGES, 8), dtype=np.float32)
        mask = np.zeros(MAX_RADIO_MESSAGES, dtype=np.int8)
        messages = [message for message in reversed(self.sim.radio_log) if message.sender_id != guard.guard_id]
        for index, message in enumerate(messages[:MAX_RADIO_MESSAGES]):
            delta = message.position - guard.position
            age = (self.sim.elapsed_ticks - message.tick) / 60.0
            result[index] = np.asarray(
                [
                    float(message.message) / 4.0 * 2.0 - 1.0,
                    np.clip(delta[0] / PLAYER_PERCEPTION_DISTANCE, -1.0, 1.0),
                    np.clip(delta[1] / PLAYER_PERCEPTION_DISTANCE, -1.0, 1.0),
                    min(1.0, norm(delta) / PLAYER_PERCEPTION_DISTANCE) * 2.0 - 1.0,
                    min(1.0, age / 5.0) * 2.0 - 1.0,
                    message.sender_id / 4.0 * 2.0 - 1.0,
                    float(guard.radio_jammed_for <= 0.0) * 2.0 - 1.0,
                    1.0,
                ],
                dtype=np.float32,
            )
            mask[index] = 1
        return result, mask

    def _formation_penalty(self) -> float:
        penalty = 0.0
        guards = self.sim.level.guards
        for index, first in enumerate(guards):
            for second in guards[index + 1 :]:
                distance = norm(first.position - second.position)
                if distance < 30.0:
                    penalty += (30.0 - distance) / 300.0
        return penalty

    def state(self) -> np.ndarray:
        # Remaining time, alert tier, and live link progress are scoring inputs:
        # timer expiry is worth the +20 containment terminal, so a critic
        # without a clock cannot predict the dominant reward it is asked to
        # value.  A separate presence mask replaces -1.0 padding, which was
        # indistinguishable from a real operative at the world origin.
        remaining = self.sim.remaining_seconds / max(1.0, float(self.sim.level.mission_seconds))
        values = [
            self.sim.player[0] / self.sim.level.world_width * 2.0 - 1.0,
            self.sim.player[1] / self.sim.level.world_height * 2.0 - 1.0,
            self.sim.velocity[0] / 230.0,
            self.sim.velocity[1] / 230.0,
            self.sim.integrity / 3.0 * 2.0 - 1.0,
            self.sim.trace / 100.0 * 2.0 - 1.0,
            self.sim.data / max(1, self.sim.level.quota) * 2.0 - 1.0,
            float(self.sim.quota_met) * 2.0 - 1.0,
            float(np.clip(remaining, 0.0, 1.0)) * 2.0 - 1.0,
            self.sim.alert_tier / 4.0 * 2.0 - 1.0,
            float(np.clip(self.sim.active_hack_progress, 0.0, 1.0)) * 2.0 - 1.0,
            len([t for t in self.sim.level.terminals if t.completed]) / max(1, len(self.sim.level.terminals)) * 2.0 - 1.0,
        ]
        presence = [-1.0] * 5
        for guard_id in range(5):
            guard = next((item for item in self.sim.level.guards if item.guard_id == guard_id), None)
            if guard is None:
                values.extend([0.0] * 8)
                continue
            presence[guard_id] = 1.0
            state = self.sim.operative_states[guard.guard_id]
            values.extend(
                [
                    guard.position[0] / self.sim.level.world_width * 2.0 - 1.0,
                    guard.position[1] / self.sim.level.world_height * 2.0 - 1.0,
                    guard.velocity[0] / 126.0,
                    guard.velocity[1] / 126.0,
                    float(guard.mode) / 5.0 * 2.0 - 1.0,
                    guard.awareness * 2.0 - 1.0,
                    float(state.role) - 1.0,
                    float(state.current_order.intent) / 7.0 * 2.0 - 1.0,
                ]
            )
        for door_id in range(3):
            door = next((item for item in self.sim.security_doors if item.door_id == door_id), None)
            if door is None:
                values.extend([0.0] * 3)
            else:
                values.extend(
                    [
                        door.tile[0] / self.sim.level.grid.shape[1] * 2.0 - 1.0,
                        door.tile[1] / self.sim.level.grid.shape[0] * 2.0 - 1.0,
                        float(door.locked) * 2.0 - 1.0,
                    ]
                )
        # Pad first so the presence mask is always the trailing block.
        values.extend([0.0] * max(0, CENTRAL_STATE_SIZE - len(presence) - len(values)))
        values = values[: CENTRAL_STATE_SIZE - len(presence)]
        values.extend(presence)
        return np.clip(np.asarray(values[:CENTRAL_STATE_SIZE], dtype=np.float32), -1.0, 1.0)

    def _info(self, agent: str) -> dict[str, Any]:
        guard = self._guard(agent)
        state = self.sim.operative_states[guard.guard_id]
        return {
            "contract": "GhostlineSecurityParallel-v0",
            "tier": self.tier,
            "seed": self.sim.seed,
            "guard_id": guard.guard_id,
            "role": state.role.name.lower(),
            "intent": state.current_order.intent.name.lower(),
            "runner_success": self.sim.extracted,
            "runner_damage": self.sim.damage_taken,
            "detections": self.sim.detections,
            "invalid_actions": self._invalid_actions,
            "reward_components": dict(self._last_reward_components),
            "agent_shaping": float(self._last_agent_shaping.get(agent, 0.0)),
            "interception": float(self._interception_score(guard)),
        }

    def render(self):
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
        close_runner = getattr(self.runner, "close", None)
        if callable(close_runner):
            close_runner()


def parallel_env(**kwargs: Any) -> GhostlineSecurityParallelEnv:
    return GhostlineSecurityParallelEnv(**kwargs)
