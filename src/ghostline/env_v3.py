from __future__ import annotations

import math
from typing import Any

import numpy as np
from gymnasium import spaces

from ghostline.config import LOCAL_GRID_SIZE, MAX_ENTITIES, PLAYER_PERCEPTION_DISTANCE, POLICY_REPEAT, RAY_COUNT, TIERS
from ghostline.env import GhostlineEnv, potential_progress_reward
from ghostline.generation import world_to_tile
from ghostline.simulation import angle_vector, norm
from ghostline.simulation_v3 import GhostlineSimulationV3
from ghostline.types_v3 import ContractDirective, GuardRole, RunnerActionV3


class GhostlineEnvV3(GhostlineEnv):
    """Player-equivalent Env-v3 contract with directives and deception tools."""

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        seed: int = 0,
        tier: int = 1,
        directive: ContractDirective | str | int = ContractDirective.STANDARD,
        external_security: bool = False,
    ):
        self.directive = ContractDirective.parse(directive)
        self.external_security = bool(external_security)
        self._directive_par_seconds = 1.0
        super().__init__(render_mode=render_mode, seed=seed, tier=tier)
        self.action_space = spaces.Discrete(72)
        self.observation_space = spaces.Dict(
            {
                "ego": spaces.Box(-1.0, 1.0, shape=(27,), dtype=np.float32),
                "objective": spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32),
                "directive": spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32),
                "local_grid": spaces.Box(0.0, 1.0, shape=(11, LOCAL_GRID_SIZE, LOCAL_GRID_SIZE), dtype=np.float32),
                "targets": spaces.Box(-1.0, 1.0, shape=(5, 10), dtype=np.float32),
                "target_mask": spaces.Box(0, 1, shape=(5,), dtype=np.int8),
                "entities": spaces.Box(-1.0, 1.0, shape=(MAX_ENTITIES, 16), dtype=np.float32),
                "entity_mask": spaces.Box(0, 1, shape=(MAX_ENTITIES,), dtype=np.int8),
                "rays": spaces.Box(0.0, 1.0, shape=(RAY_COUNT, 4), dtype=np.float32),
                "action_mask": spaces.Box(0, 1, shape=(72,), dtype=np.int8),
            }
        )
        self.sim = GhostlineSimulationV3(
            seed=self.initial_seed,
            tier=self.tier,
            directive=self.directive,
            external_security=self.external_security,
        )
        self._reset_episode_metrics()

    @property
    def unwrapped_sim(self) -> GhostlineSimulationV3:
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
        action_value = int(action)
        action_mask = self.sim.action_mask()
        clipped = int(np.clip(action_value, 0, 71))
        invalid = not 0 <= action_value < 72 or action_mask[clipped] == 0
        decoded = RunnerActionV3.decode(clipped)
        if invalid:
            decoded = RunnerActionV3(move=decoded.move)
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
        self._previous_potential = potential
        info = self._info()
        if self.sim.terminated or self.sim.truncated:
            for key, value in self.reward_components.items():
                info[f"reward_{key}"] = float(value)
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

    @property
    def directive_completed(self) -> bool:
        return self.sim.directive_completed

    def _empty_rewards(self) -> dict[str, float]:
        result = super()._empty_rewards()
        result["directive"] = 0.0
        return result

    def _observation(self) -> dict[str, np.ndarray]:
        observation = super()._observation()
        observation["directive"] = self._directive()
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
        base = super()._local_grid(visible_positions)
        extra = np.zeros((3, LOCAL_GRID_SIZE, LOCAL_GRID_SIZE), dtype=np.float32)
        px, py = world_to_tile(self.sim.player)
        half = LOCAL_GRID_SIZE // 2

        def mark(channel: int, position: np.ndarray, value: float = 1.0) -> None:
            tx, ty = world_to_tile(position)
            lx, ly = tx - px + half, ty - py + half
            if 0 <= lx < LOCAL_GRID_SIZE and 0 <= ly < LOCAL_GRID_SIZE:
                extra[channel, ly, lx] = max(extra[channel, ly, lx], value)

        for door in self.sim.security_doors:
            if door.locked or door.warning_remaining > 0.0:
                mark(0, np.asarray(((door.tile[0] + 0.5) * 32.0, (door.tile[1] + 0.5) * 32.0), dtype=np.float32), 1.0 if door.locked else 0.5)
        for decoy in self.sim.decoys:
            mark(1, decoy.position, min(1.0, decoy.lifetime / 2.0))
        for projectile in self.sim.projectiles:
            mark(2, projectile.position)
        return np.concatenate((base, extra), axis=0)

    def _entities(self, percepts=None) -> tuple[np.ndarray, np.ndarray]:
        base, mask = super()._entities(percepts)
        result = np.full((MAX_ENTITIES, 16), -1.0, dtype=np.float32)
        result[:, :13] = base
        for index in np.flatnonzero(mask):
            if base[index, 0] < 0.5:
                continue
            estimated = self.sim.player + base[index, 3:5] * PLAYER_PERCEPTION_DISTANCE
            guard = min(self.sim.level.guards, key=lambda candidate: norm(candidate.position - estimated), default=None)
            if guard is None:
                continue
            state = self.sim.operative_states[guard.guard_id]
            result[index, 13] = float(state.role) - 1.0
            result[index, 14] = 1.0 if state.weapon_cooldown <= 0.0 else -1.0
            result[index, 15] = np.clip(state.aim_progress / 0.7, 0.0, 1.0) * 2.0 - 1.0
        return result, mask

    def _rays(self, visible_positions: list[tuple[np.ndarray, float]]) -> np.ndarray:
        base = super()._rays(visible_positions)
        projectile = np.zeros((RAY_COUNT, 1), dtype=np.float32)
        for index in range(RAY_COUNT):
            direction = angle_vector(index / RAY_COUNT * math.tau)
            for shot in self.sim.projectiles:
                delta = shot.position - self.sim.player
                distance = norm(delta)
                if distance <= 1e-6:
                    projectile[index, 0] = 1.0
                    continue
                if float(np.dot(direction, delta / distance)) >= math.cos(math.pi / RAY_COUNT):
                    projectile[index, 0] = max(projectile[index, 0], 1.0 - min(1.0, distance / PLAYER_PERCEPTION_DISTANCE))
        return np.concatenate((base, projectile), axis=1)

    def telemetry(self) -> dict[str, Any]:
        telemetry = super().telemetry()
        counts = np.bincount(np.asarray(self._action_history, dtype=np.int64), minlength=72)
        telemetry.update(
            {
                "contract": "GhostlineEnv-v3",
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
                "contract": "GhostlineEnv-v3",
                "directive": self.directive.name.lower(),
                "directive_success": self.directive_completed,
                "directive_par_seconds": self._directive_par_seconds,
                "campaign_tier_name": TIERS[self.tier].name,
            }
        )
        return info
