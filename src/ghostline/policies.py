from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ghostline.config import PLAYER_PERCEPTION_DISTANCE, ROOM_HEIGHT_TILES, ROOM_WIDTH_TILES, TILE_SIZE, TIERS
from ghostline.generation import tile_center, world_to_tile
from ghostline.simulation import GhostlineSimulation, norm, unit
from ghostline.types import Action, GuardMode, Tile


MOVE_VECTORS = np.asarray(
    (
        (0.0, 0.0),
        (0.0, -1.0),
        (1.0, -1.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 0.0),
        (-1.0, -1.0),
    ),
    dtype=np.float32,
)
MOVE_VECTORS[2::2] /= np.sqrt(2.0)


@dataclass(frozen=True)
class TeacherConfig:
    """Weights for the deterministic, player-equivalent expert controller."""

    objective_weight: float = 15.0
    clearance_weight: float = 2.0
    escape_weight: float = 1.2
    sight_weight: float = 1.2
    inertia_weight: float = 0.8
    tier6_inertia_scale: float = 3.5
    dash_energy_threshold: float = 0.10
    pulse_trace_threshold: float = 0.75


class ObservationTeacherPolicy:
    """Risk-aware expert that consumes only the public ``GhostlineEnv-v2`` observation.

    The controller deliberately does not receive a simulation instance, global map,
    guard state, or generator metadata.  It uses the same objective/minimap records
    and visibility-gated entity records available to a human player and neural agent.
    This makes its trajectories suitable for fair behavior-cloning and DAgger data.
    """

    def __init__(self, config: TeacherConfig | None = None) -> None:
        self.config = config or TeacherConfig()
        self.last_move = 0
        self.last_position: np.ndarray | None = None
        self.stalled_steps = 0

    def reset(self) -> None:
        self.last_move = 0
        self.last_position = None
        self.stalled_steps = 0

    def observe_executed_action(self, action: int) -> None:
        """Synchronize controller memory with the action that reached the next state.

        During DAgger the teacher labels a state, but the behavior policy may be
        selected to act instead.  Inertia and collision-stall detection on the
        following state must therefore use the executed movement, not the
        teacher's counterfactual recommendation.
        """
        self.last_move = Action.decode(action).move

    def act(self, observation: Mapping[str, np.ndarray]) -> int:
        objective = np.asarray(observation.get("objective", np.zeros(8)), dtype=np.float32)
        ego = np.asarray(observation["ego"], dtype=np.float32)
        action_mask = np.asarray(observation["action_mask"], dtype=np.int8)
        desired = self._desired_vector(objective, ego)
        goal_distance = float((objective[3] + 1.0) * 0.5) if objective.size >= 4 else 1.0
        self._update_stall_state(ego)

        active_link = bool(objective.size >= 7 and objective[6] > 0.001 and objective[0] < 0.0)
        extracting = bool(objective.size >= 1 and objective[0] > 0.0)
        threat_vectors, immediate_danger, electronics_danger, pursuit_danger = self._threats(observation)
        scores = self._movement_scores(observation, desired, threat_vectors, active_link, extracting)
        if active_link and immediate_danger < 0.14:
            move = 0
        else:
            # Idling is a semantic link action, not a navigation fallback.
            # Keeping its fixed -0.9 score in the navigation argmax can beat a
            # valid temporary detour with a negative objective score. The
            # controller then repeats idle forever because idle intentionally
            # clears collision-stall accumulation. Choose the best occupiable
            # movement outside an active link, even when it briefly points away
            # from the objective; wait only if every movement is invalid.
            navigation_scores = scores.copy()
            navigation_scores[0] = -np.inf
            move = int(np.argmax(navigation_scores))
            if navigation_scores[move] <= -99.0:
                move = 0
        if self.stalled_steps >= 8 and move == self.last_move:
            alternatives = np.argsort(scores)[::-1]
            move = next((int(item) for item in alternatives if int(item) not in (0, self.last_move)), move)
            self.stalled_steps = 0
        self.last_move = move

        trace = float((ego[5] + 1.0) * 0.5)
        dash_energy = float((ego[8] + 1.0) * 0.5)
        pulse = (
            electronics_danger > 0.22
            or pursuit_danger > 0.14
            or trace > self.config.pulse_trace_threshold
        )
        tier = int(round((float(ego[14]) + 1.0) * 3.0))
        safe_cruise = tier <= 2 and goal_distance > 0.10 and trace < 0.20
        urgent_extract = extracting and goal_distance > 0.025
        dash = move != 0 and not active_link and dash_energy > self.config.dash_energy_threshold and (
            immediate_danger > 0.18 or safe_cruise or urgent_extract
        )
        return self._legal_action(move, dash, pulse, action_mask)

    def _update_stall_state(self, ego: np.ndarray) -> None:
        """Track genuine collision stalls using only public player kinematics."""
        position = np.asarray(ego[19:21], dtype=np.float32)
        if self.last_position is not None and self.last_move != 0:
            displacement = float(np.linalg.norm(position - self.last_position))
            normalized_speed = float(np.linalg.norm(ego[:2]))
            if displacement < 0.001 and normalized_speed < 0.04:
                self.stalled_steps += 1
            else:
                self.stalled_steps = max(0, self.stalled_steps - 2)
        elif self.last_move == 0:
            self.stalled_steps = max(0, self.stalled_steps - 1)
        self.last_position = position.copy()

    @staticmethod
    def _desired_vector(objective: np.ndarray, ego: np.ndarray) -> np.ndarray:
        if objective.size >= 6:
            waypoint = objective[4:6]
            goal = objective[1:3]
        else:
            waypoint = ego[19:21]
            goal = ego[21:23]
        desired = waypoint if float(np.linalg.norm(waypoint)) > 0.045 else goal
        magnitude = float(np.linalg.norm(desired))
        return desired / magnitude if magnitude > 1e-6 else np.zeros(2, dtype=np.float32)

    def _movement_scores(
        self,
        observation: Mapping[str, np.ndarray],
        desired: np.ndarray,
        threats: list[tuple[np.ndarray, float, float, bool]],
        active_link: bool,
        extracting: bool,
    ) -> np.ndarray:
        blocked = np.asarray(observation["local_grid"], dtype=np.float32)[1]
        center = blocked.shape[0] // 2
        ego = np.asarray(observation["ego"], dtype=np.float32)
        tier = int(np.clip(round((float(ego[14]) + 1.0) * 3.0), 1, 6))
        # Generation places a one-tile outer border around the authored room
        # lattice. Ego position is normalized by that full level extent, so the
        # inverse projection must include the border as well. Omitting it shifts
        # the reconstructed sub-tile offset and makes valid doorway movement look
        # blocked to the observation-only clearance test.
        width = (TIERS[tier].room_columns * ROOM_WIDTH_TILES + 1) * TILE_SIZE
        height = (TIERS[tier].room_rows * ROOM_HEIGHT_TILES + 1) * TILE_SIZE
        player = np.asarray(
            (((float(ego[19]) + 1.0) * 0.5 * width), ((float(ego[20]) + 1.0) * 0.5 * height)),
            dtype=np.float32,
        )
        fractional = np.mod(player, TILE_SIZE)
        rays = np.asarray(observation["rays"], dtype=np.float32)
        objective_weight = self.config.objective_weight
        escape_weight = self.config.escape_weight
        sight_weight = self.config.sight_weight
        inertia_weight = self.config.inertia_weight
        if tier == 3:
            # Patrol is the first human-security lesson: retain enough lateral
            # avoidance to learn clean breaks rather than brute-force contact.
            objective_weight *= 0.80
            escape_weight *= 5.0 / 3.0
            sight_weight *= 5.0 / 3.0
        elif tier == 5:
            # Lockdown adds a persistent response drone. Finishing a short route
            # decisively is safer than orbiting persistent pursuers; pulse timing
            # handles contact pressure while movement stays mission-directed.
            objective_weight *= 1.20
            escape_weight = 0.0
            sight_weight = 0.0
        elif tier == 6:
            # Full-system contracts punish indecisive heading switches: a
            # validation-only calibration found that stronger directional
            # commitment reduced guard contacts without adding timeouts.
            inertia_weight *= self.config.tier6_inertia_scale
        ray_by_move = {1: 18, 2: 21, 3: 0, 4: 3, 5: 6, 6: 9, 7: 12, 8: 15}
        scores = np.full(9, -100.0, dtype=np.float32)
        for move, direction in enumerate(MOVE_VECTORS):
            if move == 0:
                scores[move] = 0.7 if active_link else -0.9
                continue
            clearance = self._clearance(blocked, center, direction, fractional)
            ray = rays[ray_by_move[move]]
            geometry_clearance = float(ray[0])
            # Twenty-five pixels of center-ray clearance covers the runner's
            # radius plus one 10 Hz movement decision. Rejecting tighter moves
            # makes the controller align with a doorway instead of repeatedly
            # accelerating into its jamb.
            if clearance <= 0.0 or geometry_clearance < 0.078:
                continue
            local_danger = max(
                (pressure * max(0.0, 1.0 - float(np.linalg.norm(relative)) / 235.0) for relative, pressure, _, _ in threats),
                default=0.0,
            )
            objective_scale = max(0.25, 1.0 - 1.35 * max(0.0, local_danger - 0.12))
            mission_weight = objective_weight * (1.65 if extracting else objective_scale)
            threat_scale = 0.30 if extracting else 1.0
            score = mission_weight * float(np.dot(direction, desired))
            score += self.config.clearance_weight * clearance
            score += 1.8 * min(1.0, geometry_clearance * 3.0)
            score -= 5.0 * float(ray[1])
            score += inertia_weight * float(move == self.last_move)
            for relative, pressure, exposure, electronic in threats:
                distance = max(1.0, float(np.linalg.norm(relative)))
                away = -relative / distance
                proximity = max(0.0, 1.0 - distance / (250.0 if electronic else 205.0))
                score += threat_scale * escape_weight * pressure * proximity * float(np.dot(direction, away))
                score -= threat_scale * sight_weight * exposure * max(0.0, float(np.dot(direction, relative / distance)))
            scores[move] = score
        return scores

    @staticmethod
    def _clearance(
        blocked: np.ndarray,
        center: int,
        direction: np.ndarray,
        fractional: np.ndarray,
    ) -> float:
        projected = fractional + direction * 18.0
        for corner_x, corner_y in ((-9.5, -9.5), (9.5, -9.5), (-9.5, 9.5), (9.5, 9.5), (0.0, 0.0)):
            offset_x = int(np.floor((projected[0] + corner_x) / TILE_SIZE))
            offset_y = int(np.floor((projected[1] + corner_y) / TILE_SIZE))
            x, y = center + offset_x, center + offset_y
            if not (0 <= x < blocked.shape[1] and 0 <= y < blocked.shape[0]) or blocked[y, x] > 0.5:
                return 0.0
        # The projected footprint above is the authoritative occupiability
        # check.  Do not reject a direction merely because the *centre tile*
        # one cell ahead is blocked: near a wall corner the runner can safely
        # slide within its current tile even though that coarse look-ahead
        # cell is occupied.  The former hard rejection could declare all eight
        # moves illegal and idle for the rest of an otherwise reachable run.
        # Nearby blockers remain a soft clearance preference.
        clearance = 1.0
        for radius, multiplier in ((1, 0.72), (2, 0.86)):
            x = int(np.clip(center + round(float(direction[0]) * radius), 0, blocked.shape[1] - 1))
            y = int(np.clip(center + round(float(direction[1]) * radius), 0, blocked.shape[0] - 1))
            if blocked[y, x] > 0.5:
                clearance *= multiplier
            else:
                clearance += 0.18
        return clearance

    @staticmethod
    def _threats(
        observation: Mapping[str, np.ndarray],
    ) -> tuple[list[tuple[np.ndarray, float, float, bool]], float, float, float]:
        entities = np.asarray(observation["entities"], dtype=np.float32)
        mask = np.asarray(observation["entity_mask"], dtype=np.int8)
        threats: list[tuple[np.ndarray, float, float, bool]] = []
        immediate = 0.0
        electronics = 0.0
        pursuit = 0.0
        for record, present in zip(entities, mask):
            if not present:
                continue
            kind = int(np.argmax(record[:3]))
            relative = record[3:5] * PLAYER_PERCEPTION_DISTANCE
            distance = max(1.0, float(np.linalg.norm(relative)))
            raw_alert = float((record[10] + 1.0) * 0.5)
            confidence = float((record[11] + 1.0) * 0.5)
            facing = np.asarray((record[9], record[8]), dtype=np.float32)
            toward_player = -relative / distance
            facing_score = max(0.0, float(np.dot(facing, toward_player)))
            electronic = kind in (1, 2)
            if kind == 0:
                # Guard alert is the public normalized GuardMode value, not an
                # ordinal danger score: RETURN happens to be enum value five
                # and must not be interpreted as more dangerous than CHASE.
                mode = int(np.clip(round(raw_alert * 5.0), 0, 5))
                alert = (0.0, 0.25, 0.50, 0.62, 1.0, 0.12)[mode]
                grade = float(record[12]) if record.size >= 13 else -1.0
                grade_scale = 1.0 + 0.22 * np.clip((grade + 1.0) * 0.5, 0.0, 1.0)

                # Confidence bands have explicit public semantics.  Exact LOS
                # is 1.0; frozen last-seen snapshots occupy 0.51..0.90; coarse
                # current audio occupies 0.10..0.40.  A last-seen marker at its
                # 0.51 floor is presentation memory, not evidence that a guard
                # still occupies that coordinate, so its tactical pressure
                # must decay to zero instead of repelling the teacher forever.
                live = confidence >= 0.99
                remembered = 0.50 <= confidence < 0.99
                audible = confidence < 0.50
                if live:
                    certainty = 1.0
                    facing_certainty = 1.0
                elif remembered:
                    recency = float(np.clip((confidence - 0.51) / 0.39, 0.0, 1.0))
                    certainty = recency**1.35
                    facing_certainty = certainty
                else:
                    # Audio is current but deliberately sector/range quantized;
                    # use it for escape/pursuit pressure without inventing a
                    # precise viewing cone from the zeroed facing fields.
                    certainty = 0.45 + 0.55 * float(np.clip(confidence / 0.40, 0.0, 1.0))
                    facing_certainty = 0.0
                exposure = (
                    grade_scale
                    * facing_certainty
                    * (0.8 + 0.5 * alert)
                    * max(0.0, (facing_score - 0.38) / 0.62)
                )
                pressure = grade_scale * certainty * (0.08 + 1.1 * alert) + 0.8 * exposure
                if audible:
                    pressure = max(pressure, grade_scale * certainty * (0.18 + 0.92 * alert))
                active_pressure = pressure if alert >= 0.35 else exposure
                immediate = max(immediate, active_pressure * max(0.0, 1.0 - distance / 210.0))
                if mode == 4:
                    pursuit = max(pursuit, pressure * max(0.0, 1.0 - distance / 220.0))
            elif kind == 1:
                exposure = confidence * (0.75 + 0.5 * raw_alert) * max(0.0, (facing_score - 0.58) / 0.42)
                pressure = 0.10 + 0.55 * raw_alert + exposure
                electronics = max(electronics, max(raw_alert, exposure) * max(0.0, 1.0 - distance / 230.0))
            else:
                if confidence >= 0.99:
                    certainty = 1.0
                else:
                    recency = float(np.clip((confidence - 0.51) / 0.39, 0.0, 1.0))
                    certainty = recency**1.35
                pressure = certainty * 1.25
                exposure = pressure * max(0.0, 1.0 - distance / 260.0)
                electronics = max(electronics, pressure * max(0.0, 1.0 - distance / 260.0))
                pursuit = max(pursuit, pressure * max(0.0, 1.0 - distance / 270.0))
            threats.append((relative, pressure, exposure, electronic))
        return threats, immediate, electronics, pursuit

    @staticmethod
    def _legal_action(move: int, dash: bool, pulse: bool, mask: np.ndarray) -> int:
        candidates = (
            Action(move, dash, pulse).encode(),
            Action(move, False, pulse).encode(),
            Action(move, dash, False).encode(),
            Action(move, False, False).encode(),
        )
        return next((value for value in candidates if 0 <= value < len(mask) and mask[value]), int(move))


class FairScriptedPolicy:
    """Non-learning baseline using only known objectives and currently visible threats."""

    def act(self, sim: GhostlineSimulation) -> int:
        targets = [terminal for terminal in sim.level.terminals if not terminal.completed]
        if sim.quota_met or not targets:
            target = sim.level.extraction
        else:
            target = min(targets, key=lambda terminal: norm(terminal.position - sim.player) / max(1, terminal.value)).position

        if norm(target - sim.player) <= 26.0 and not sim.quota_met:
            movement = 0
        else:
            waypoint = self._next_waypoint(sim, target)
            desired = waypoint - sim.player
            desired = self._avoid_visible_threats(sim, desired)
            movement = self._direction_index(desired)

        visible_threat = any(
            norm(guard.position - sim.player) < 165.0
            and sim.line_of_sight(sim.player, guard.position)
            and guard.mode in (GuardMode.CHASE, GuardMode.INVESTIGATE)
            for guard in sim.level.guards
        )
        pulse = sim.pulse_charges > 0 and sim.pulse_cooldown <= 0.0 and (sim.trace > 61.0 or visible_threat)
        dash = movement != 0 and sim.dash_energy > 34.0 and not visible_threat and sim.trace < 70.0
        return Action(movement, dash, pulse).encode()

    def _next_waypoint(self, sim: GhostlineSimulation, target: np.ndarray) -> np.ndarray:
        return sim._next_path_point(sim.player, target)

    @staticmethod
    def _avoid_visible_threats(sim: GhostlineSimulation, desired: np.ndarray) -> np.ndarray:
        adjusted = desired.astype(np.float32).copy()
        for guard in sim.level.guards:
            distance = norm(guard.position - sim.player)
            if distance < 115.0 and sim.line_of_sight(sim.player, guard.position):
                adjusted += unit(sim.player - guard.position) * (160.0 - distance) * 2.2
        for camera in sim.level.cameras:
            distance = norm(camera.position - sim.player)
            if camera.detected and distance < 190.0:
                adjusted += unit(sim.player - camera.position) * (210.0 - distance)
        return adjusted

    @staticmethod
    def _direction_index(vector: np.ndarray) -> int:
        if norm(vector) < 2.0:
            return 0
        x, y = float(vector[0]), float(vector[1])
        sx, sy = int(np.sign(x)), int(np.sign(y))
        if abs(y) <= max(3.0, abs(x) * 0.08):
            sy = 0
        if abs(x) <= max(3.0, abs(y) * 0.08):
            sx = 0
        return {
            (0, -1): 1,
            (1, -1): 2,
            (1, 0): 3,
            (1, 1): 4,
            (0, 1): 5,
            (-1, 1): 6,
            (-1, 0): 7,
            (-1, -1): 8,
        }.get((sx, sy), 0)
