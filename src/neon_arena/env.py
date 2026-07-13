from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from neon_arena.config import (
    ARENA_TEMPLATES,
    ArenaTemplate,
    BOOST_ACCELERATION_MULTIPLIER,
    BOOST_MAX_SPEED,
    CONE_EXIT_REWARD,
    CONE_EXIT_REWARD_COOLDOWN,
    CORE_RADIUS,
    DASH_ENERGY_DRAIN,
    DASH_ENERGY_MAX,
    DASH_ENERGY_RECHARGE,
    DASH_COOLDOWN_STEPS,
    DASH_SPEED,
    DRONE_ROLES,
    DRONE_RADIUS,
    EMP_DURATION_STEPS,
    EMP_RADIUS,
    HEAT_DASH_GAIN,
    HEAT_DASH_NOISE_RANGE,
    HEAT_DECAY,
    HEAT_DRONE_HIT_GAIN,
    HEAT_EMP_DECAY,
    HEAT_EMP_DROP,
    HEAT_EXPOSURE_GAIN,
    HEAT_MAX,
    HEAT_WALL_GAIN,
    HUNTER_ALARM_RANGE_BONUS,
    HUNTER_LOCK_STEPS,
    HUNTER_LOCK_RANGE,
    HUNTER_LOST_LOCK_DECAY,
    HUNTER_RANGE_PER_CORE,
    HUNTER_VISION_COSINE,
    MAX_STEPS,
    PATROL_LOCK_RANGE,
    PATROL_LOCK_STEPS,
    PATROL_LOST_LOCK_DECAY,
    PLAYER_ACCELERATION,
    PLAYER_FRICTION,
    PLAYER_MAX_SPEED,
    PLAYER_RADIUS,
    PORTAL_RADIUS,
    RAY_COUNT,
    RAY_LENGTH,
    Rect,
    SENTRY_LOCK_RANGE,
    SENTRY_PROJECTILE_COOLDOWN,
    SENTRY_PROJECTILE_HIT_GAIN,
    SENTRY_PROJECTILE_LIFETIME,
    SENTRY_PROJECTILE_RADIUS,
    SENTRY_PROJECTILE_SPEED,
    SENTRY_TURN_RATE,
    STALL_GRACE_STEPS,
    STALL_TERMINATE_STEPS,
    WORLD_HEIGHT,
    WORLD_WIDTH,
)
from neon_arena.geometry import (
    circle_rect_overlap,
    clamp_magnitude,
    length,
    normalized,
    ray_rect_distance,
)


@dataclass
class Drone:
    route: tuple[tuple[float, float], ...]
    position: np.ndarray
    velocity: np.ndarray
    route_index: int
    base_speed: float
    role: str
    lock_steps: int = 0
    facing_angle: float = 0.0
    fire_cooldown: int = 0
    state: str = "PATROL"
    target_pos: np.ndarray | None = None
    state_timer: int = 0
    planned_path: list[np.ndarray] = field(default_factory=list)
    path_recalc_cooldown: int = 0


@dataclass
class Projectile:
    position: np.ndarray
    velocity: np.ndarray
    life: int


@dataclass
class Gate:
    rect: Rect
    type: str  # "blue", "red", "security"
    is_open: bool = False


@dataclass
class Camera:
    position: np.ndarray
    base_angle: float
    sweep_amplitude: float
    sweep_speed: float
    fov: float
    current_angle: float = 0.0
    sweep_direction: int = 1


@dataclass
class Loot:
    position: np.ndarray
    type: str  # "data_shard", "gold_cache", "black_box"
    collected: bool = False





@dataclass
class RoomNode:
    id: int
    rect: Rect
    role: str  # SPAWN, MARKET, PLAZA, SECURITY, SEC-HUB, VAULT, EXTRACT, CONNECT
    center: tuple[float, float]
    col: int
    row: int


@dataclass
class DoorEdge:
    room_a: int
    room_b: int
    center: tuple[float, float]
    width: float
    is_locked: bool = False
    is_open: bool = True


@dataclass
class ShockTile:
    rect: Rect
    state: str = "SAFE"
    timer: int = 0


@dataclass
class CoolantSpill:
    rect: Rect


class BlacklineHeistEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        seed: int | None = None,
        action_mode: str = "discrete",
        curriculum_stage: str = "full",
        campaign_stage: int | None = None
    ):
        super().__init__()
        self.render_mode = render_mode
        self.action_mode = action_mode
        self.curriculum_stage = curriculum_stage
        self.campaign_stage = campaign_stage
        self.camera_alert_cooldown = 0

        if self.campaign_stage is not None:
            stage_curriculums = {
                1: "route",
                2: "sequence",
                3: "sentry",
                4: "patrols",
                5: "full",
                6: "large_easy",
                7: "large_sentry_camera",
                8: "large_proc_full_no_interact",
                9: "large_proc_easy",
                10: "large_proc_cameras",
                11: "large_proc_patrols",
                12: "large_proc_full",
            }
            self.curriculum_stage = stage_curriculums.get(self.campaign_stage, "full")

        if self.action_mode == "discrete":
            self.action_space = spaces.MultiDiscrete([9, 2, 2])
        else:
            self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        self.observation_space = spaces.Box(-1.0, 1.0, shape=(84,), dtype=np.float32)
        self._initial_seed = seed
        self._has_seeded = False
        self.renderer = None
        self.training_stats: dict[str, Any] = {}
        
        # Dynamic boundaries
        self.world_width = 1200.0
        self.world_height = 760.0
        
        # New District Heist entities and state trackers
        self.gates: list[Gate] = []
        self.cameras: list[Camera] = []
        self.loot: list[Loot] = []
        self.security_gates_open = False
        self.visited_districts: set[int] = set()
        self.used_shortcuts: set[int] = set()
        self.current_district = 0
        self.doorway_centers = []
        self.semantic_patrol_nodes = []
        self.procedural_rooms = []
        
        # Room graph state
        self.room_nodes: list[RoomNode] = []
        self.door_edges: list[DoorEdge] = []
        self.room_adjacency: dict[int, list[int]] = {}
        self.visited_rooms: set[int] = set()
        self.exploration_reward_total = 0.0
        self.shock_tiles: list[ShockTile] = []
        self.coolant_spills: list[CoolantSpill] = []
        self.preset_sentry_positions: list[tuple[float, float]] = []
        self.objective_room_reward_claimed: set[tuple[int, int]] = set()
        self.wrong_room_steps = 0
        self.current_room_id = -1
        self.objective_phase = 0
        
        # Diagnostic tracking variables
        self.heat_sum = 0.0
        self.total_wrong_room_steps = 0
        self.objective_room_entries = 0
        self.room_transitions = 0
        self.drone_contacts_count = 0
        self.projectile_hits_count = 0
        self.camera_alerts_count = 0
        self.cone_steps_count = 0
        self.emp_uses_count = 0
        self.emp_effective_uses_count = 0
        self.emp_invalid_presses_count = 0
        self.emp_wasted_uses_count = 0
        self.loot_reward_total = 0.0
        self.loot_items_collected_count = 0
        self.loot_reward_capped_flag = 0.0
        self.same_room_steps = 0
        self.same_room_max_steps = 0
        self.next_doorway_reached_count = 0
        self.objective_room_reached_flag = 0.0
        self.last_doorway_target = None
        self.last_objective_pos = None
        self.episode_rewards: dict[str, float] = {}
        
        template = ARENA_TEMPLATES[0]
        self.layout_name = template.name
        self.layout_id = 0
        self.city_blocks: tuple[Rect, ...] = template.blocks
        self.core_slots: tuple[tuple[float, float], ...] = template.core_slots
        self.spawn_point = np.array(template.spawn_point, dtype=np.float32)
        self.extraction_point = np.array(template.extraction_point, dtype=np.float32)
        self.terminal_point = np.array(template.terminal_point, dtype=np.float32)
        self.drone_routes = template.drone_routes
        self.player = np.zeros(2, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.heading = np.array([1.0, 0.0], dtype=np.float32)
        self.cores: list[np.ndarray] = []
        self.cores_collected = 0
        self.collected_core_indices: set[int] = set()
        self.current_objective_index = -1
        self.extracted = False
        self.alarm_active = False
        self.alarm_steps = 0
        self.heat = 0.0
        self.max_heat = 0.0
        self.terminal_available = True
        self.emp_available = False
        self.emp_timer = 0
        self.shield = 3
        self.dash_cooldown = 0
        self.dash_energy = DASH_ENERGY_MAX
        self.step_count = 0
        self.episode_reward = 0.0
        self.damage_taken = 0
        self.dash_count = 0
        self.boost_active_last_step = False
        self.previous_mission_potential = 0.0
        self.best_mission_potential = 0.0
        self.stalled_steps = 0
        self.hunter_exposure_steps = 0
        self.hunter_lock_count = 0
        self.wall_hits = 0
        self.contact_cooldown = 0
        self.wall_collision_cooldown = 0
        self.seen_lock_triggered = False
        self.consecutive_unseen_steps = 0
        self.cone_escape_reward_count = 0
        self.phi_prev = 0.0
        self.was_in_drone_cone = False
        self.cone_exit_reward_cooldown = 0
        self.last_result: dict[str, Any] | None = None
        self.events: list[tuple[str, tuple[float, float]]] = []
        self.drones: list[Drone] = []
        self.projectiles: list[Projectile] = []
        self._ray_directions = tuple(
            np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
            for angle in np.linspace(0.0, math.tau, RAY_COUNT, endpoint=False)
        )
        
        # Navigation grid settings
        self.grid_size = 25.0
        self.cols = int(math.ceil(self.world_width / self.grid_size))
        self.rows = int(math.ceil(self.world_height / self.grid_size))
        self.grid_passable = np.ones((self.cols, self.rows), dtype=bool)
        self.dist_map = np.full((self.cols, self.rows), -1.0, dtype=np.float32)

    @property
    def objective(self) -> np.ndarray:
        if len(self.collected_core_indices) < len(self.cores) and 0 <= self.current_objective_index < len(self.cores):
            return self.cores[self.current_objective_index]
        return self.extraction_point.copy()

    @property
    def player_current_room(self) -> int:
        curr = getattr(self, "_player_current_room_val", -1)
        if curr == -1:
            curr = self._get_room_id_at_position(self.player)
            self._player_current_room_val = curr
        else:
            # 1. Doorway distance threshold transition (directional path-based)
            if getattr(self, "cores", None) is not None and getattr(self, "door_edges", None) is not None:
                obj_room = self._get_room_id_at_position(self.objective)
                if curr != obj_room:
                    room_path = self._dijkstra_room_path(curr, obj_room)
                    if len(room_path) >= 2:
                        next_room = room_path[1]
                        door_center = None
                        for edge in self.door_edges:
                            if (edge.room_a == curr and edge.room_b == next_room) or \
                               (edge.room_a == next_room and edge.room_b == curr):
                                door_center = np.array(edge.center, dtype=np.float32)
                                break
                        if door_center is not None:
                            if np.linalg.norm(self.player - door_center) < 10.0:
                                self._player_current_room_val = next_room
                                return next_room

            # 2. Geometric containment with 2px margin fallback (oscillations bypassed near connecting door)
            actual_room = self._get_room_id_at_position(self.player)
            if actual_room != curr:
                # Find if there is a door between curr and actual_room
                door_center = None
                for edge in self.door_edges:
                    if (edge.room_a == curr and edge.room_b == actual_room) or \
                       (edge.room_a == actual_room and edge.room_b == curr):
                        door_center = np.array(edge.center, dtype=np.float32)
                        break
                
                # If player is near the door connecting the rooms, do not revert
                if door_center is not None and np.linalg.norm(self.player - door_center) < 12.0:
                    pass
                else:
                    node = self.room_nodes[actual_room]
                    r = node.rect
                    px, py = self.player[0], self.player[1]
                    margin = 2.0
                    if r.left + margin <= px <= r.right - margin and r.top + margin <= py <= r.bottom - margin:
                        self._player_current_room_val = actual_room
                        curr = actual_room
        return curr

    @player_current_room.setter
    def player_current_room(self, val: int) -> None:
        self._player_current_room_val = val

    @property
    def objective_label(self) -> str:
        if len(self.collected_core_indices) < len(self.cores):
            return f"CORE {len(self.collected_core_indices) + 1}/{len(self.cores)}"
        return "EXTRACT"

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        reset_seed = seed
        if reset_seed is None and not self._has_seeded:
            reset_seed = self._initial_seed
        super().reset(seed=reset_seed)
        self._has_seeded = True
        self.shock_tiles = []
        self.coolant_spills = []
        self.preset_sentry_positions = []
        self._cached_target = None
        self._last_curr_room = None
        self._last_terminal_available = None
        self._apply_random_layout()
        
        if "large_proc" not in self.curriculum_stage:
            self.room_nodes = [
                RoomNode(
                    id=0,
                    rect=Rect(0.0, 0.0, self.world_width, self.world_height),
                    role="CONNECT",
                    center=(self.world_width / 2.0, self.world_height / 2.0),
                    col=0,
                    row=0
                )
            ]
            self.door_edges = []
            self.room_adjacency = {0: []}
        
        
        # Dynamic boundaries recalculation for grid passability mapping
        self.cols = int(math.ceil(self.world_width / self.grid_size))
        self.rows = int(math.ceil(self.world_height / self.grid_size))
        self.grid_passable = np.ones((self.cols, self.rows), dtype=bool)
        self.dist_map = np.full((self.cols, self.rows), -1.0, dtype=np.float32)
        
        # Build grid based on block layout
        self._build_grid()
        self._detect_blocked_doorways()
        
        # Determine core count based on curriculum stage
        num_cores = 1 if self.curriculum_stage == "route" else 3
        selected = self.np_random.choice(len(self.core_slots), size=num_cores, replace=False)
        self.cores = [np.array(self.core_slots[int(index)], dtype=np.float32) for index in selected]
        
        self.player = self.spawn_point.copy()
        
        self.visited_rooms = set()
        self.exploration_reward_total = 0.0
        self.objective_room_reward_claimed = set()
        self.wrong_room_steps = 0
        self.current_room_id = self._get_room_id_at_position(self.player)
        self.player_current_room = self.current_room_id
        self.visited_rooms.add(self.current_room_id)
        
        # Reset tracking metrics
        self.heat_sum = 0.0
        self.total_wrong_room_steps = 0
        self.objective_room_entries = 0
        self.room_transitions = 0
        self.drone_contacts_count = 0
        self.projectile_hits_count = 0
        self.camera_alerts_count = 0
        self.cone_steps_count = 0
        self.emp_uses_count = 0
        self.emp_effective_uses_count = 0
        self.emp_invalid_presses_count = 0
        self.emp_wasted_uses_count = 0
        self.loot_reward_total = 0.0
        self.loot_items_collected_count = 0
        self.loot_reward_capped_flag = 0.0
        self.same_room_steps = 0
        self.same_room_max_steps = 0
        self.next_doorway_reached_count = 0
        self.objective_room_reached_flag = 0.0
        self.last_doorway_target = None
        self.last_objective_pos = None

        self.episode_rewards = {
            "extraction": 0.0,
            "core": 0.0,
            "progress": 0.0,
            "room_new": 0.0,
            "room_objective": 0.0,
            "wrong_room": 0.0,
            "time": 0.0,
            "heat": 0.0,
            "cone": 0.0,
            "lock": 0.0,
            "hit": 0.0,
            "projectile": 0.0,
            "emp": 0.0,
            "loot": 0.0,
            "timeout": 0.0,
            "stall": 0.0,
            "proximity": 0.0,
        }
        
        if self.cores:
            self.current_objective_index = int(min(range(len(self.cores)), key=lambda i: math.hypot(self.cores[i][0] - self.player[0], self.cores[i][1] - self.player[1])))
        else:
            self.current_objective_index = -1
        self.velocity = np.zeros(2, dtype=np.float32)
        self.heading = np.array([1.0, 0.0], dtype=np.float32)
        self.cores_collected = 0
        self.collected_core_indices = set()
        self.extracted = False
        self.alarm_active = False
        self.alarm_steps = 0
        self.heat = 0.0
        self.max_heat = 0.0
        self.terminal_available = True
        self.emp_available = False
        self.emp_timer = 0
        self.camera_alert_cooldown = 0
        self.shield = 3
        self.dash_cooldown = 0
        self.dash_energy = DASH_ENERGY_MAX
        self.step_count = 0
        self.episode_reward = 0.0
        self.damage_taken = 0
        self.dash_count = 0
        self.boost_active_last_step = False
        self.stalled_steps = 0
        self.hunter_exposure_steps = 0
        self.hunter_lock_count = 0
        self.wall_hits = 0
        self.contact_cooldown = 0
        self.wall_collision_cooldown = 0
        self.seen_lock_triggered = False
        self.consecutive_unseen_steps = 0
        self.cone_escape_reward_count = 0
        self.was_in_drone_cone = False
        self.cone_exit_reward_cooldown = 0
        self.last_result = None
        self.events = []
        
        # Reset Gate, Camera, and Loot state trackers
        self.security_gates_open = False
        self.visited_districts = set()
        self.used_shortcuts = set()
        self.current_district = self._get_district_id(self.player)
        self.visited_districts.add(self.current_district)
        
        # Initialize gates
        if "large_proc" not in self.curriculum_stage:
            self.gates = []
            if hasattr(self.active_template, "gate_configs"):
                for rect_coords, gate_type in self.active_template.gate_configs:
                    self.gates.append(
                        Gate(
                            rect=Rect(*rect_coords),
                            type=gate_type,
                            is_open=False,
                        )
                    )
                
        # Initialize cameras
        self.cameras = []
        if hasattr(self.active_template, "camera_positions"):
            for cam_pos in self.active_template.camera_positions:
                pos = np.array(cam_pos[:2], dtype=np.float32)
                angle = float(cam_pos[2]) if len(cam_pos) > 2 else float(self.np_random.uniform(0.0, math.tau))
                self.cameras.append(
                    Camera(
                        position=pos,
                        base_angle=angle,
                        sweep_amplitude=math.pi * 0.35,
                        sweep_speed=0.015,
                        fov=math.pi / 3.0,
                        current_angle=0.0,
                        sweep_direction=1,
                    )
                )
                
        # Initialize loot (shards, caches, black box)
        self.loot = []
        if self.world_width > 1300.0:
            loot_types = ["data_shard", "gold_cache", "black_box"]
            for i, lt in enumerate(loot_types):
                unused_slots = [np.array(slot, dtype=np.float32) for slot in self.core_slots if not any(np.allclose(slot, core) for core in self.cores)]
                if i < len(unused_slots):
                    pos = unused_slots[i]
                else:
                    pos = self.spawn_point.copy()
                    for _ in range(50):
                        rx = float(self.np_random.uniform(100.0, self.world_width - 100.0))
                        ry = float(self.np_random.uniform(100.0, self.world_height - 100.0))
                        rp = np.array([rx, ry], dtype=np.float32)
                        if self._point_is_clear(rp, self.city_blocks, PLAYER_RADIUS + 10.0):
                            pos = rp
                            break
                self.loot.append(Loot(position=pos.copy(), type=lt))
                
        # Extract semantic patrol nodes
        self.semantic_patrol_nodes = []
        if "large_proc" in self.curriculum_stage:
            # 1. Room centers
            for r, role in getattr(self, "procedural_rooms", []):
                cx = r.x + r.width / 2.0
                cy = r.y + r.height / 2.0
                pt = np.array([cx, cy], dtype=np.float32)
                self.semantic_patrol_nodes.append(pt)
            # 2. Doorway centers
            for dc in getattr(self, "doorway_centers", []):
                self.semantic_patrol_nodes.append(np.array(dc, dtype=np.float32))
            # 3. Cores
            for core in self.cores:
                self.semantic_patrol_nodes.append(core.copy())
            # 4. Terminal
            self.semantic_patrol_nodes.append(self.terminal_point.copy())
            # 5. Cameras
            for camera in self.cameras:
                self.semantic_patrol_nodes.append(camera.position.copy())
        else:
            seen = set()
            for route in self.drone_routes:
                for wp in route:
                    wp_tup = (float(wp[0]), float(wp[1]))
                    if wp_tup not in seen:
                        seen.add(wp_tup)
                        self.semantic_patrol_nodes.append(np.array(wp, dtype=np.float32))
            if not self.semantic_patrol_nodes:
                self.semantic_patrol_nodes = [self.terminal_point.copy(), self.extraction_point.copy()]
                
        # Deduplicate and filter nodes that are occupyable
        unique_nodes = []
        for node in self.semantic_patrol_nodes:
            if not self._can_occupy(node):
                continue
            if any(length(node - existing) < 20.0 for existing in unique_nodes):
                continue
            unique_nodes.append(node)
        self.semantic_patrol_nodes = unique_nodes

        self.drones = self._create_drones()
        self.projectiles = []
        
        # Compute navigation distance maps
        self._update_navigation()
        
        self.previous_mission_potential = self._route_potential()
        self.best_mission_potential = self.previous_mission_potential

        # Compute initial potential-based reward shaping value
        phase_score = self.cores_collected / 3.0
        path_dist = self._route_distance()
        max_path_dist = math.hypot(self.world_width, self.world_height)
        goal_score = 1.0 - float(np.clip(path_dist / max_path_dist, 0.0, 1.0))
        self.phi_prev = 2.0 * phase_score + goal_score

        # Compute Map Variety Metrics
        if "large_proc" in self.curriculum_stage:
            spawn_to_extract_dist = math.hypot(self.extraction_point[0] - self.spawn_point[0], self.extraction_point[1] - self.spawn_point[1])
            a_star_path = self._find_path_astar(self.spawn_point, self.extraction_point)
            if a_star_path and len(a_star_path) > 1:
                a_star_dist = sum(math.hypot(a_star_path[i][0] - a_star_path[i-1][0], a_star_path[i][1] - a_star_path[i-1][1]) for i in range(1, len(a_star_path)))
                self.metric_route_directness = float(a_star_dist / spawn_to_extract_dist)
            else:
                self.metric_route_directness = 1.0
                
            num_rooms = len(self.room_nodes)
            self.metric_loop_count = len(self.door_edges) - (num_rooms - 1)
            self.metric_locked_edge_count = sum(1 for edge in self.door_edges if getattr(edge, "is_locked", False))
            self.metric_camera_coverage = len(self.cameras)
            
            hazard_rooms = set()
            for tile in getattr(self, "shock_tiles", []):
                r_id = self._get_room_id_at_position((tile.rect.x + 20, tile.rect.y + 20))
                hazard_rooms.add(r_id)
            for spill in getattr(self, "coolant_spills", []):
                r_id = self._get_room_id_at_position((spill.rect.x + 30, spill.rect.y + 30))
                hazard_rooms.add(r_id)
            self.metric_hazard_room_count = len(hazard_rooms)

        return self._observation(), self._info()

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        prev_objective = self.objective.copy()
        if self.action_mode == "discrete":
            act = np.asarray(action)
            move_idx = int(act[0])
            dash_idx = int(act[1])
            emp_idx = int(act[2]) if act.size >= 3 else 0
            wants_interact = False
            
            SQRT2_INV = 0.70710678
            move_mapping = [
                np.array([0.0, 0.0], dtype=np.float32),          # 0: idle
                np.array([0.0, -1.0], dtype=np.float32),         # 1: up
                np.array([SQRT2_INV, -SQRT2_INV], dtype=np.float32), # 2: up-right
                np.array([1.0, 0.0], dtype=np.float32),          # 3: right
                np.array([SQRT2_INV, SQRT2_INV], dtype=np.float32),  # 4: down-right
                np.array([0.0, 1.0], dtype=np.float32),          # 5: down
                np.array([-SQRT2_INV, SQRT2_INV], dtype=np.float32), # 6: down-left
                np.array([-1.0, 0.0], dtype=np.float32),         # 7: left
                np.array([-SQRT2_INV, -SQRT2_INV], dtype=np.float32) # 8: up-left
            ]
            movement = move_mapping[np.clip(move_idx, 0, 8)]
            wants_dash = (dash_idx == 1)
            wants_emp = (emp_idx == 1)
        else:
            action = np.asarray(action, dtype=np.float32)
            movement = clamp_magnitude(action[:2], 1.0)
            wants_dash = float(action[2]) > 0.35
            wants_emp = action.size >= 4 and float(action[3]) > 0.35
            wants_interact = False

        collision = False
        boosting = wants_dash and self.dash_energy > 0.0 and self.dash_cooldown == 0 and length(movement) > 0.05

        if self.boost_active_last_step and not boosting:
            self.dash_cooldown = DASH_COOLDOWN_STEPS

        on_coolant = any(circle_rect_overlap(self.player, PLAYER_RADIUS, spill.rect) for spill in getattr(self, "coolant_spills", []))
        from neon_arena.config import COOLANT_TURN_RATE_MULTIPLIER, COOLANT_FRICTION_MULTIPLIER
        
        if length(movement) > 0.05:
            if on_coolant:
                self.heading = normalized(self.heading * (1.0 - COOLANT_TURN_RATE_MULTIPLIER) + movement * COOLANT_TURN_RATE_MULTIPLIER)
                acceleration = PLAYER_ACCELERATION * (BOOST_ACCELERATION_MULTIPLIER if boosting else 1.0) * COOLANT_TURN_RATE_MULTIPLIER
            else:
                self.heading = normalized(movement)
                acceleration = PLAYER_ACCELERATION * (BOOST_ACCELERATION_MULTIPLIER if boosting else 1.0)
            
            self.velocity += movement * acceleration
            if boosting and not self.boost_active_last_step:
                boost_vel = movement * (DASH_SPEED * 0.48)
                if on_coolant:
                    boost_vel *= COOLANT_TURN_RATE_MULTIPLIER
                self.velocity += boost_vel
        
        friction = PLAYER_FRICTION
        if on_coolant:
            friction = 1.0 - (1.0 - PLAYER_FRICTION) * COOLANT_FRICTION_MULTIPLIER
            
        self.velocity *= friction
        self.velocity = clamp_magnitude(self.velocity, BOOST_MAX_SPEED if boosting else PLAYER_MAX_SPEED)

        if boosting:
            self.dash_energy = max(0.0, self.dash_energy - DASH_ENERGY_DRAIN)
            if not self.boost_active_last_step:
                self.dash_count += 1
            if self.step_count % 8 == 0:
                self.events.append(("dash", tuple(float(value) for value in self.player)))
            if self._drone_within_noise_range():
                self._add_heat(HEAT_DASH_GAIN / 18.0)
        elif self.dash_energy < DASH_ENERGY_MAX:
            self.dash_energy = min(DASH_ENERGY_MAX, self.dash_energy + DASH_ENERGY_RECHARGE)
        self.boost_active_last_step = boosting

        prev_path_dist = self._route_distance()
        phase_score = self.cores_collected / 3.0
        max_path_dist = math.hypot(self.world_width, self.world_height)
        goal_score = 1.0 - float(np.clip(prev_path_dist / max_path_dist, 0.0, 1.0))
        phi_prev = 2.0 * phase_score + goal_score

        if self.camera_alert_cooldown > 0:
            self.camera_alert_cooldown -= 1
        if self.dash_cooldown > 0:
            self.dash_cooldown -= 1
        if self.emp_timer > 0:
            self.emp_timer -= 1
        if self.alarm_active:
            self.alarm_steps += 1
        if self.contact_cooldown > 0:
            self.contact_cooldown -= 1
        if self.cone_exit_reward_cooldown > 0:
            self.cone_exit_reward_cooldown -= 1
        if self.wall_collision_cooldown > 0:
            self.wall_collision_cooldown -= 1

        collision = self._move_player(self.velocity) or collision
        wall_penalty = 0.0
        if collision:
            self.wall_hits += 1
            self._add_heat(HEAT_WALL_GAIN)
            # self.velocity *= 0.32  # Removed speed penalty when hitting a wall
            self.events.append(("impact", tuple(float(value) for value in self.player)))
            if self.wall_collision_cooldown == 0:
                    wall_penalty = -0.3
                    self.wall_collision_cooldown = 10
        # Update gates open conditions
        for gate in self.gates:
            if self.curriculum_stage == "large_easy":
                gate.is_open = True
            else:
                if gate.type == "blue":
                    gate.is_open = (self.cores_collected >= 1)
                elif gate.type == "red":
                    gate.is_open = self.alarm_active
                elif gate.type == "security":
                    if not self.terminal_available:
                        gate.is_open = True

        # Check shortcut gate traversal rewards
        shortcut_reward = 0.0
        for idx, gate in enumerate(self.gates):
            if gate.is_open and idx not in self.used_shortcuts:
                if circle_rect_overlap(self.player, PLAYER_RADIUS, gate.rect):
                    self.used_shortcuts.add(idx)
                    shortcut_reward += 1.5
                    self.events.append(("shortcut", tuple(float(value) for value in self.player)))

        # Update camera sweep angles
        for camera in self.cameras:
            camera.current_angle += camera.sweep_direction * camera.sweep_speed
            if abs(camera.current_angle) > camera.sweep_amplitude:
                camera.sweep_direction *= -1
                camera.current_angle = np.clip(camera.current_angle, -camera.sweep_amplitude, camera.sweep_amplitude)

        camera_detected = False
        spiking_camera = None
        for camera in self.cameras:
            if self._player_seen_by_camera(camera):
                camera_detected = True
                spiking_camera = camera
                break
        if camera_detected:
            self._add_heat(0.05)
            if self.camera_alert_cooldown == 0:
                self._add_heat(10.0)
                self.camera_alert_cooldown = 45
                self.events.append(("camera_detection", tuple(float(value) for value in spiking_camera.position)))
                active_drones = [d for d in self.drones if not self.drone_is_emp_stunned(d) and d.role != "sentry"]
                if active_drones:
                    nearest_drone = min(active_drones, key=lambda d: length(d.position - self.player))
                    nearest_drone.state = "INVESTIGATE"
                    nearest_drone.state_timer = 90
                    nearest_drone.target_pos = self.player.copy()
                    nearest_drone.planned_path = []
                    nearest_drone.lock_steps = HUNTER_LOCK_STEPS if nearest_drone.role == "hunter" else PATROL_LOCK_STEPS


        # Check loot collection
        loot_reward = 0.0
        for item in self.loot:
            if not item.collected:
                if length(item.position - self.player) <= PLAYER_RADIUS + 18.0:
                    item.collected = True
                    self.events.append(("loot_pickup", tuple(float(value) for value in item.position)))
                    self.loot_items_collected_count += 1
                    if self.loot_reward_total < 3.0:
                        added = min(1.5, 3.0 - self.loot_reward_total)
                        self.loot_reward_total += added
                        loot_reward += added
                        if self.loot_reward_total >= 3.0:
                            self.loot_reward_capped_flag = 1.0
                    if item.type == "data_shard":
                        self.shield = min(3, self.shield + 1)
                    elif item.type == "gold_cache":
                        self.heat = max(0.0, self.heat - 30.0)
                    elif item.type == "black_box":
                        self.emp_available = True

        self._update_drones()
        self._update_projectiles()
        drone_exposure = self._total_drone_exposure()
        hunter_exposure = self._total_hunter_exposure()
        if drone_exposure > 0.0:
            self._add_heat(HEAT_EXPOSURE_GAIN * min(2.5, drone_exposure))
        else:
            self._decay_heat()
        if hunter_exposure > 0.0:
            self.hunter_exposure_steps += 1
            
        cone_penalty = 0.0
        lock_penalty = 0.0
        is_seen_or_locked = False
        for drone in self.drones:
            seen = self._drone_can_see_player(drone)
            locked = drone.lock_steps > 0
            if seen or locked:
                is_seen_or_locked = True
            if seen:
                cone_penalty -= 0.015
            if locked:
                lock_penalty -= 0.04
                
        proximity_penalty = 0.0
        for drone in self.drones:
            if not self.drone_is_emp_stunned(drone):
                dist = length(drone.position - self.player)
                if dist < 80.0 and self._has_line_of_sight(self.player, drone.position):
                    clamped_dist = max(30.0, dist)
                    proximity_penalty -= 0.03 * (1.0 - (clamped_dist - 30.0) / 50.0)
                
        heat_penalty = max(-0.06, -0.0008 * self.heat)
        if any(self._drone_can_see_player(d) for d in self.drones):
            self.cone_steps_count += 1

        cone_escape_reward = 0.0
        if is_seen_or_locked:
            self.seen_lock_triggered = True
            self.consecutive_unseen_steps = 0
        else:
            if self.seen_lock_triggered:
                self.consecutive_unseen_steps += 1
                if self.consecutive_unseen_steps >= 20 and self.cone_escape_reward_count < 3:
                    cone_escape_reward += 0.5
                    self.cone_escape_reward_count += 1
                    self.seen_lock_triggered = False
                    self.consecutive_unseen_steps = 0
                    self.events.append(("evade", tuple(float(value) for value in self.player)))

        projectile_penalty = 0.0
        if self._projectile_collision() and self.contact_cooldown <= 0:
            self.contact_cooldown = 28
            self.shield -= 1
            self.damage_taken += 1
            self.projectile_hits_count += 1
            projectile_penalty -= 6.0
            self._add_heat(SENTRY_PROJECTILE_HIT_GAIN)
            self.events.append(("danger", tuple(float(value) for value in self.player)))

        hit_penalty = wall_penalty
        if self._drone_collision() and self.contact_cooldown <= 0:
            self.contact_cooldown = 32
            self.shield -= 1
            self.damage_taken += 1
            self.drone_contacts_count += 1
            self._separate_from_nearest_drone()
            hit_penalty -= 8.0
            self._add_heat(HEAT_DRONE_HIT_GAIN)
            self.events.append(("danger", tuple(float(value) for value in self.player)))

        # Update shock tiles and deal damage
        shock_penalty = 0.0
        for tile in getattr(self, "shock_tiles", []):
            tile.timer += 1
            from neon_arena.config import SHOCK_CYCLE_SAFE, SHOCK_CYCLE_WARMING, SHOCK_CYCLE_ACTIVE
            period = SHOCK_CYCLE_SAFE + SHOCK_CYCLE_WARMING + SHOCK_CYCLE_ACTIVE
            t = tile.timer % period
            if t < SHOCK_CYCLE_SAFE:
                tile.state = "SAFE"
            elif t < SHOCK_CYCLE_SAFE + SHOCK_CYCLE_WARMING:
                tile.state = "WARN"
            else:
                tile.state = "ACTIVE"
                
            if tile.state == "ACTIVE" and self.contact_cooldown <= 0:
                if circle_rect_overlap(self.player, PLAYER_RADIUS, tile.rect):
                    self.contact_cooldown = 30
                    self.shield -= 1
                    self.damage_taken += 1
                    shock_penalty -= 5.0
                    self.events.append(("danger", tuple(float(value) for value in self.player)))

        emp_reward = 0.0
        if self.terminal_available and length(self.terminal_point - self.player) <= PLAYER_RADIUS + 24.0:
            self.terminal_available = False
            self.emp_available = True
            emp_reward += 3.0
            self.events.append(("emp_pickup", tuple(float(value) for value in self.terminal_point)))
            
            # Unlock terminal gates
            for edge in self.door_edges:
                if getattr(edge, "is_locked", False):
                    edge.is_locked = False
                    edge.is_open = True
            for gate in self.gates:
                if gate.type == "terminal":
                    gate.is_open = True
                    
            self._update_navigation()

        if wants_emp:
            if not self.emp_available:
                self.emp_invalid_presses_count += 1
                emp_reward -= 0.05
            elif self.emp_timer <= 0:
                # In procedural stages, only consume the EMP if a threat is actually in range
                is_proc = "large_proc" in self.curriculum_stage
                threats_in_range = not is_proc or any(
                    length(d.position - self.player) <= EMP_RADIUS
                    for d in self.drones
                    if not self.drone_is_emp_stunned(d)
                ) or any(
                    length(c.position - self.player) <= EMP_RADIUS
                    for c in self.cameras
                )
                
                if threats_in_range:
                    self.emp_available = False
                    self.emp_timer = EMP_DURATION_STEPS
                    self.heat = max(0.0, self.heat - HEAT_EMP_DROP)
                    self.events.append(("emp", tuple(float(value) for value in self.player)))
                    self.emp_uses_count += 1
                    
                    drones_stunned = sum(
                        1 for d in self.drones
                        if length(d.position - self.player) <= EMP_RADIUS
                    )
                    cameras_stunned = sum(
                        1 for c in self.cameras
                        if length(c.position - self.player) <= EMP_RADIUS
                    )
                    
                    total_stunned = drones_stunned + cameras_stunned
                    if total_stunned > 0:
                        self.emp_effective_uses_count += 1
                        emp_reward += 1.5 + 0.75 * (total_stunned - 1)
                    else:
                        self.emp_wasted_uses_count += 1
                        emp_reward -= 0.5
                else:
                    # Bypassed invalid press penalty when EMP is available to encourage exploration
                    pass



        core_reward = 0.0
        extraction_reward = 0.0
        
        # Check collision with any uncollected cores
        if len(self.collected_core_indices) < len(self.cores):
            for index, core in enumerate(self.cores):
                if index not in self.collected_core_indices:
                    core_dist = length(core - self.player)
                    if core_dist <= PLAYER_RADIUS + CORE_RADIUS:
                        self.collected_core_indices.add(index)
                        self.cores_collected = len(self.collected_core_indices)
                        
                        uncollected_indices = [i for i in range(len(self.cores)) if i not in self.collected_core_indices]
                        if uncollected_indices:
                            self.current_objective_index = int(min(uncollected_indices, key=lambda i: math.hypot(self.cores[i][0] - self.player[0], self.cores[i][1] - self.player[1])))
                        else:
                            self.current_objective_index = -1
                            
                        if self.cores_collected == 1:
                            core_reward += 12.0
                        elif self.cores_collected == 2:
                            core_reward += 16.0
                        elif self.cores_collected == 3:
                            core_reward += 22.0
                        self.events.append(("core", tuple(float(value) for value in core)))
                        if self.cores_collected >= len(self.cores):
                            self._start_alarm()
                        self._update_navigation()
                        break
        elif length(self.extraction_point - self.player) <= PLAYER_RADIUS + PORTAL_RADIUS:
            extraction_reward += 200.0
            self.extracted = True
            self.events.append(("extract", tuple(float(value) for value in self.player)))

        # Progress and potential shaping based on path distance
        new_path_dist = self._route_distance()
        expected_step_dist = 7.4
        progress = (prev_path_dist - new_path_dist) / expected_step_dist
        progress = np.clip(progress, -1.0, 1.0)
        
        progress_shaping = 0.0
        progress_shaping += 0.03 * progress

        phase_score_next = self.cores_collected / 3.0
        goal_score_next = 1.0 - float(np.clip(new_path_dist / max_path_dist, 0.0, 1.0))
        phi_next = 2.0 * phase_score_next + goal_score_next
        progress_shaping += 0.3 * float(np.clip(0.995 * phi_next - phi_prev, -0.5, 0.5))

        # Check district entry reward
        district_reward = 0.0
        player_district = self._get_district_id(self.player)
        self.current_district = player_district
        
        objective_district = self._get_district_id(self.objective)
        if player_district == objective_district and player_district not in self.visited_districts:
            self.visited_districts.add(player_district)
            district_reward += 1.0
            self.events.append(("district_entry", tuple(float(value) for value in self.player)))

        stalled_out = self._update_stalled()
        stall_penalty = 0.0
        if self.stalled_steps > STALL_GRACE_STEPS:
            stall_penalty -= 0.04


        self.step_count += 1
        terminated = self.extracted or self.shield <= 0
        truncated = self.step_count >= MAX_STEPS or stalled_out

        terminal_penalty = 0.0
        if self.shield <= 0:
            terminal_penalty -= 60.0
        if stalled_out:
            terminal_penalty -= 3.0
            
        hit_penalty += terminal_penalty if self.shield <= 0 else 0.0
        stall_penalty += terminal_penalty if stalled_out else 0.0
            
        timeout_penalty = 0.0
        if self.step_count >= MAX_STEPS and not self.extracted and self.shield > 0:
            missing_cores = max(0, len(self.cores) - self.cores_collected)
            if missing_cores == 0:
                timeout_penalty = -4.0
            else:
                timeout_penalty = -12.0 - 8.0 * missing_cores

        if self.world_width > 1300.0:
            time_cost = -0.005 if self.alarm_active else -0.002
        else:
            time_cost = -0.003 if self.alarm_active else -0.001

        # Exploration and room navigation rewards
        room_new_reward = 0.0
        objective_room_reward = 0.0
        wrong_room_penalty = 0.0
        
        prev_room = self.current_room_id
        curr_room = self.player_current_room
        obj_room = self._get_room_id_at_position(self.objective)
        
        # Track same room consecutive steps
        if prev_room != -1 and curr_room == prev_room:
            self.same_room_steps += 1
            self.same_room_max_steps = max(self.same_room_max_steps, self.same_room_steps)
        else:
            self.same_room_steps = 0
            
        # Room transitions
        if prev_room != -1 and curr_room != prev_room:
            self.room_transitions += 1

        # New room exploration
        if curr_room not in self.visited_rooms:
            self.visited_rooms.add(curr_room)
            if self.exploration_reward_total < 1.5:
                added = min(0.10, 1.5 - self.exploration_reward_total)
                self.exploration_reward_total += added
                room_new_reward += added
                
        # Objective room entry
        if curr_room == obj_room:
            self.objective_room_reached_flag = 1.0
            claim_key = (self.cores_collected, curr_room)
            if claim_key not in self.objective_room_reward_claimed:
                self.objective_room_reward_claimed.add(claim_key)
                objective_room_reward = 0.50
                self.objective_room_entries += 1
                
        # Wrong room penalty resets
        is_locked = any(d.lock_steps > 0 for d in self.drones)
        objective_changed = (self.last_objective_pos is not None and not np.allclose(self.objective, self.last_objective_pos))
        
        if curr_room == obj_room or curr_room != prev_room or objective_changed or is_locked:
            self.wrong_room_steps = 0
        else:
            self.wrong_room_steps += 1
            
        if curr_room != obj_room:
            self.total_wrong_room_steps += 1
            if self.wrong_room_steps > 600:
                wrong_room_penalty = -0.02
            elif self.wrong_room_steps > 400:
                wrong_room_penalty = -0.01

        self.current_room_id = curr_room
        
        # Pack sub-components into shaping logic
        progress_shaping += shortcut_reward
        progress_shaping += district_reward

        # Compute step reward by adding components to log dictionary
        step_reward = 0.0
        step_reward += self._add_reward("extraction", extraction_reward)
        step_reward += self._add_reward("core", core_reward)
        step_reward += self._add_reward("progress", progress_shaping)
        step_reward += self._add_reward("room_new", room_new_reward)
        step_reward += self._add_reward("room_objective", objective_room_reward)
        step_reward += self._add_reward("wrong_room", wrong_room_penalty)
        step_reward += self._add_reward("time", time_cost)
        step_reward += self._add_reward("heat", heat_penalty)
        step_reward += self._add_reward("cone", cone_penalty)
        step_reward += self._add_reward("lock", lock_penalty)
        step_reward += self._add_reward("hit", hit_penalty)
        step_reward += self._add_reward("projectile", projectile_penalty)
        step_reward += self._add_reward("emp", emp_reward)
        step_reward += self._add_reward("loot", loot_reward)
        step_reward += self._add_reward("timeout", timeout_penalty)
        step_reward += self._add_reward("stall", stall_penalty)
        step_reward += self._add_reward("proximity", proximity_penalty)
        
        # Track active target doorway crossings
        curr_doorway_target = self._get_next_route_target()
        if not np.allclose(curr_doorway_target, self.objective):
            if length(self.player - curr_doorway_target) <= PLAYER_RADIUS + 15.0:
                if self.last_doorway_target is None or not np.allclose(curr_doorway_target, self.last_doorway_target):
                    self.next_doorway_reached_count += 1
                    self.last_doorway_target = curr_doorway_target.copy()

        self.last_objective_pos = self.objective.copy()
        self.heat_sum += self.heat
        
        self.episode_reward += step_reward
        if terminated or truncated:
            self.last_result = self._result(truncated=truncated)
        return self._observation(), step_reward, terminated, truncated, self._info(
            collision=collision,
            terminal=terminated or truncated,
        )

    def render(self, *, process_events: bool = True, limit_fps: bool = True) -> bool:
        if self.render_mode != "human":
            return True
        if self.renderer is None:
            from neon_arena.renderer import ArenaRenderer

            self.renderer = ArenaRenderer(self)
        return bool(self.renderer.draw(self.training_stats, process_events=process_events, limit_fps=limit_fps))

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def set_training_stats(self, stats: dict[str, Any]) -> None:
        self.training_stats = dict(stats)

    def pop_events(self) -> list[tuple[str, tuple[float, float]]]:
        events = list(self.events)
        self.events.clear()
        return events

    def set_curriculum_stage(self, stage: str) -> None:
        self.curriculum_stage = stage

    @property
    def heat_ratio(self) -> float:
        return float(np.clip(self.heat / HEAT_MAX, 0.0, 1.0))

    def _add_heat(self, amount: float) -> None:
        self.heat = float(np.clip(self.heat + amount, 0.0, HEAT_MAX))
        self.max_heat = max(self.max_heat, self.heat)

    def _decay_heat(self) -> None:
        decay = HEAT_EMP_DECAY if self.emp_timer > 0 else HEAT_DECAY
        self.heat = max(0.0, self.heat - decay)

    def _drone_within_noise_range(self) -> bool:
        return any(
            not self.drone_is_emp_stunned(drone)
            and length(drone.position - self.player) <= HEAT_DASH_NOISE_RANGE
            for drone in self.drones
        )

    def _generate_procedural_layout(self) -> ArenaTemplate:
        self.doorway_centers = []
        
        # 1. Jitter grid lines
        col_xs = [0.0]
        row_ys = [0.0]
        
        x1 = float(self.np_random.uniform(270.0, 330.0))
        x2 = float(self.np_random.uniform(570.0, 630.0))
        x3 = float(self.np_random.uniform(870.0, 930.0))
        x4 = float(self.np_random.uniform(1170.0, 1230.0))
        col_xs.extend([x1, x2, x3, x4, 1500.0])
        
        y1 = float(self.np_random.uniform(290.0, 340.0))
        y2 = float(self.np_random.uniform(600.0, 650.0))
        row_ys.extend([y1, y2, 950.0])
        
        # 2. Cell merging on 5x3 grid
        cells = [(c, r) for c in range(5) for r in range(3)]
        room_to_cells = {i: {cells[i]} for i in range(15)}
        cell_to_room = {cells[i]: i for i in range(15)}
        
        spawn_cell = (0, 1)
        extract_cell = (4, 1)
        
        for _ in range(80):
            c = self.np_random.integers(0, 5)
            r = self.np_random.integers(0, 3)
            cell1 = (c, r)
            
            dcol, drow = self.np_random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
            nc, nr = c + dcol, r + drow
            if not (0 <= nc < 5 and 0 <= nr < 3):
                continue
            cell2 = (nc, nr)
            
            room1 = cell_to_room[cell1]
            room2 = cell_to_room[cell2]
            
            if room1 == room2:
                continue
                
            unmergable = {(0, 1), (4, 1), (2, 1), (3, 0), (3, 1), (3, 2)}
            if any(cell in unmergable for cell in room_to_cells[room1]):
                continue
            if any(cell in unmergable for cell in room_to_cells[room2]):
                continue
                
            merged_cells = room_to_cells[room1] | room_to_cells[room2]
            cols_in_merge = [cc for cc, rr in merged_cells]
            rows_in_merge = [rr for cc, rr in merged_cells]
            
            min_c, max_c = min(cols_in_merge), max(cols_in_merge)
            min_r, max_r = min(rows_in_merge), max(rows_in_merge)
            
            w = max_c - min_c + 1
            h = max_r - min_r + 1
            
            if w > 2 or h > 2:
                continue
                
            if len(merged_cells) != w * h:
                continue
                
            room_to_cells[room1] = merged_cells
            for cell in room_to_cells[room2]:
                cell_to_room[cell] = room1
            del room_to_cells[room2]
            
        active_rooms = sorted(list(room_to_cells.keys()))
        num_rooms = len(active_rooms)
        
        new_room_to_cells = {}
        new_cell_to_room = {}
        for new_id, old_id in enumerate(active_rooms):
            new_room_to_cells[new_id] = room_to_cells[old_id]
            for cell in room_to_cells[old_id]:
                new_cell_to_room[cell] = new_id
                
        rooms = []
        room_roles = []
        
        for r_id in range(num_rooms):
            cells_in_room = new_room_to_cells[r_id]
            cols_in_room = [cc for cc, rr in cells_in_room]
            rows_in_room = [rr for cc, rr in cells_in_room]
            min_c, max_c = min(cols_in_room), max(cols_in_room)
            min_r, max_r = min(rows_in_room), max(rows_in_room)
            
            rx = col_xs[min_c]
            ry = row_ys[min_r]
            rw = col_xs[max_c + 1] - rx
            rh = row_ys[max_r + 1] - ry
            
            rooms.append(Rect(rx, ry, rw, rh))
            
            c_center = (min_c + max_c) // 2
            if spawn_cell in cells_in_room:
                role = "SPAWN"
            elif extract_cell in cells_in_room:
                role = "EXTRACT"
            elif c_center == 0:
                role = "MARKET"
            elif c_center == 1:
                role = "PLAZA"
            elif c_center == 2:
                if (2, 1) in cells_in_room:
                    role = "SEC-HUB"
                else:
                    role = "SECURITY"
            elif c_center == 3:
                role = "VAULT"
            else:
                role = "CONNECT"
            room_roles.append(role)
            
        adjacencies = []
        for c in range(5):
            for r in range(3):
                curr_room = new_cell_to_room[(c, r)]
                if c < 4:
                    right_room = new_cell_to_room[(c + 1, r)]
                    if curr_room != right_room:
                        adjacencies.append({
                            "rooms": (curr_room, right_room),
                            "is_vertical": True,
                            "col": c,
                            "row": r
                        })
                if r < 2:
                    bottom_room = new_cell_to_room[(c, r + 1)]
                    if curr_room != bottom_room:
                        adjacencies.append({
                            "rooms": (curr_room, bottom_room),
                            "is_vertical": False,
                            "col": c,
                            "row": r
                        })
                        
        parent = list(range(num_rooms))
        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]
            
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j
                return True
            return False
            
        shuffled_adj = list(adjacencies)
        indices = list(range(len(shuffled_adj)))
        self.np_random.shuffle(indices)
        shuffled_adj = [shuffled_adj[idx] for idx in indices]
        
        tree_edges = []
        loop_candidates = []
        for adj in shuffled_adj:
            u, v = adj["rooms"]
            if union(u, v):
                tree_edges.append(adj)
            else:
                loop_candidates.append(adj)
                
        doorway_edges = list(tree_edges)
        
        def get_degrees():
            deg = [0] * num_rooms
            for edge in doorway_edges:
                u, v = edge["rooms"]
                deg[u] += 1
                deg[v] += 1
            return deg
            
        for _ in range(15):
            deg = get_degrees()
            under_deg = [i for i, d in enumerate(deg) if d < 2]
            if not under_deg:
                break
            added = False
            for adj in loop_candidates:
                if adj in doorway_edges:
                    continue
                u, v = adj["rooms"]
                if u in under_deg or v in under_deg:
                    doorway_edges.append(adj)
                    added = True
                    break
            if not added:
                for adj in loop_candidates:
                    if adj not in doorway_edges:
                        doorway_edges.append(adj)
                        added = True
                        break
                if not added:
                    break
                    
        target_doorways = max(num_rooms + 5, min(20, num_rooms * 2))
        for adj in loop_candidates:
            if len(doorway_edges) >= target_doorways:
                break
            if adj not in doorway_edges:
                doorway_edges.append(adj)
                
        doorway_adj = set()
        for adj in doorway_edges:
            doorway_adj.add((adj["col"], adj["row"], adj["is_vertical"]))
            
        blocks = []
        dw = 115.0
        doorway_edges_list = []
        self.gates = []
        
        for c in range(5):
            for r in range(3):
                if c < 4:
                    u = new_cell_to_room[(c, r)]
                    v = new_cell_to_room[(c + 1, r)]
                    if u != v:
                        coord = col_xs[c + 1]
                        start_coord = row_ys[r]
                        end_coord = row_ys[r + 1]
                        is_doorway = (c, r, True) in doorway_adj
                        
                        if is_doorway:
                            mid = (start_coord + end_coord) / 2.0
                            center = (coord, mid)
                            self.doorway_centers.append(center)
                            doorway_edges_list.append(DoorEdge(room_a=u, room_b=v, center=center, width=dw, is_locked=False, is_open=True))
                            
                            p1_len = (mid - dw / 2.0) - start_coord
                            if p1_len > 1.0:
                                blocks.append(Rect(coord - 10.0, start_coord, 20.0, p1_len))
                            p2_len = end_coord - (mid + dw / 2.0)
                            if p2_len > 1.0:
                                blocks.append(Rect(coord - 10.0, mid + dw / 2.0, 20.0, p2_len))
                        else:
                            wall_len = end_coord - start_coord
                            if wall_len > 1.0:
                                blocks.append(Rect(coord - 10.0, start_coord, 20.0, wall_len))
                                
                if r < 2:
                    u = new_cell_to_room[(c, r)]
                    v = new_cell_to_room[(c, r + 1)]
                    if u != v:
                        coord = row_ys[r + 1]
                        start_coord = col_xs[c]
                        end_coord = col_xs[c + 1]
                        is_doorway = (c, r, False) in doorway_adj
                        
                        if is_doorway:
                            mid = (start_coord + end_coord) / 2.0
                            center = (mid, coord)
                            self.doorway_centers.append(center)
                            doorway_edges_list.append(DoorEdge(room_a=u, room_b=v, center=center, width=dw, is_locked=False, is_open=True))
                            
                            p1_len = (mid - dw / 2.0) - start_coord
                            if p1_len > 1.0:
                                blocks.append(Rect(start_coord, coord - 10.0, p1_len, 20.0))
                            p2_len = end_coord - (mid + dw / 2.0)
                            if p2_len > 1.0:
                                blocks.append(Rect(mid + dw / 2.0, coord - 10.0, p2_len, 20.0))
                        else:
                            wall_len = end_coord - start_coord
                            if wall_len > 1.0:
                                blocks.append(Rect(start_coord, coord - 10.0, wall_len, 20.0))
                                
        spawn_idx = next(i for i, role in enumerate(room_roles) if role == "SPAWN")
        extract_idx = next(i for i, role in enumerate(room_roles) if role == "EXTRACT")
        terminal_idx = next(i for i, cells in new_room_to_cells.items() if (2, 1) in cells)
        
        spawn_room = rooms[spawn_idx]
        spawn_point = (spawn_room.x + spawn_room.width / 2.0, spawn_room.y + spawn_room.height / 2.0)
        
        extract_room = rooms[extract_idx]
        extraction_point = (extract_room.x + extract_room.width / 2.0, extract_room.y + extract_room.height / 2.0)
        
        terminal_room = rooms[terminal_idx]
        terminal_point = (terminal_room.x + terminal_room.width / 2.0, terminal_room.y + terminal_room.height / 2.0)
        
        core_slots = []
        for r_idx, role in enumerate(room_roles):
            if role == "VAULT":
                # For each cell of this Vault room that is in column 3, add its cell center to slots
                for cc, rr in new_room_to_cells[r_idx]:
                    if cc == 3:
                        cx = col_xs[3] + (col_xs[4] - col_xs[3]) / 2.0
                        cy = row_ys[rr] + (row_ys[rr+1] - row_ys[rr]) / 2.0
                        core_slots.append((cx, cy))
                        
        while len(core_slots) < 3:
            core_slots.append((col_xs[3] + 150.0, row_ys[1] + 150.0))
        core_slots = tuple(core_slots[:3])
        
        checkpoints = [spawn_point, extraction_point, terminal_point] + list(core_slots)
        
        # Procedural layouts keep all doorways unlocked and open initially
        locked_indices = []
            
        self.preset_sentry_positions = []
        camera_positions = []
        
        from neon_arena.config import ROOM_PRESETS
        
        for r_idx, room in enumerate(rooms):
            role = room_roles[r_idx]
            if role in ("SPAWN", "EXTRACT"):
                continue
            if role == "SEC-HUB" or role == "CONNECT":
                if role == "SEC-HUB":
                    cx = room.x + room.width / 2.0
                    cy = room.y + room.height / 2.0
                    blocks.append(Rect(cx - 50.0, cy - 60.0, 100.0, 20.0))
                    blocks.append(Rect(cx - 50.0, cy + 40.0, 100.0, 20.0))
                    camera_positions.append((room.x + 18.0, room.y + 18.0, math.pi / 4.0))
                continue
                
            presets = ROOM_PRESETS.get(role, [])
            if presets:
                p_idx = self.np_random.integers(0, len(presets))
                preset = presets[p_idx]
                
                for rx, ry, rw, rh in preset.get("blocks", []):
                    bx = room.x + rx * room.width
                    by = room.y + ry * room.height
                    bw = rw * room.width
                    bh = rh * room.height
                    blocks.append(Rect(bx, by, bw, bh))
                    
                for cx_rel, cy_rel in preset.get("cameras", []):
                    cx = room.x + cx_rel * room.width
                    cy = room.y + cy_rel * room.height
                    angle = math.atan2(room.y + room.height / 2.0 - cy, room.x + room.width / 2.0 - cx)
                    camera_positions.append((cx, cy, angle))
                    
                for sx_rel, sy_rel in preset.get("sentries", []):
                    sx = room.x + sx_rel * room.width
                    sy = room.y + sy_rel * room.height
                    self.preset_sentry_positions.append((sx, sy))
                    
        self.shock_tiles = []
        self.coolant_spills = []
                            
        self.procedural_rooms = [(rooms[i], room_roles[i]) for i in range(num_rooms)]
        
        self.room_nodes = []
        for r_idx, room in enumerate(rooms):
            self.room_nodes.append(RoomNode(
                id=r_idx,
                rect=room,
                role=room_roles[r_idx],
                center=(room.x + room.width / 2.0, room.y + room.height / 2.0),
                col=new_room_to_cells[r_idx].copy().pop()[0],
                row=new_room_to_cells[r_idx].copy().pop()[1]
            ))
            
        self.door_edges = doorway_edges_list
        self.room_adjacency = {i: [] for i in range(num_rooms)}
        for edge in self.door_edges:
            self.room_adjacency[edge.room_a].append(edge.room_b)
            self.room_adjacency[edge.room_b].append(edge.room_a)
            
        return ArenaTemplate(
            name="large_proc",
            blocks=tuple(blocks),
            core_slots=core_slots,
            spawn_point=spawn_point,
            extraction_point=extraction_point,
            terminal_point=terminal_point,
            drone_routes=((), (), ()),
            camera_positions=tuple(camera_positions),
            gate_configs=(),
            rooms=tuple(rooms)
        )

    def _apply_random_layout(self) -> None:
        if self.campaign_stage is not None:
            stage_idx = self.campaign_stage
            stage_curriculums = {
                1: ("route", False, 0),
                2: ("sequence", False, 1),
                3: ("sentry", False, 2),
                4: ("patrols", False, 3),
                5: ("full", False, 0),
                6: ("large_easy", True, 0),
                7: ("large_sentry_camera", True, 1),
                8: ("large_proc_full_no_interact", True, 0),
                9: ("large_proc_easy", True, 0),
                10: ("large_proc_cameras", True, 0),
                11: ("large_proc_patrols", True, 0),
                12: ("large_proc_full", True, 0),
            }
            curriculum, is_large, template_index = stage_curriculums.get(stage_idx, ("full", False, 0))
            self.curriculum_stage = curriculum
            
            if "large_proc" not in self.curriculum_stage:
                if is_large:
                    from neon_arena.config import LARGE_TEMPLATES
                    templates = LARGE_TEMPLATES
                    self.world_width = 1500.0
                    self.world_height = 950.0
                else:
                    from neon_arena.config import SMALL_TEMPLATES
                    templates = SMALL_TEMPLATES
                    self.world_width = 1200.0
                    self.world_height = 760.0
                    
                template = templates[template_index]
                self.active_template = template
                
                if self.curriculum_stage in ("route", "sequence"):
                    self._assign_layout(template, template_index, template.blocks, template.core_slots, template.terminal_point)
                else:
                    for _ in range(24):
                        blocks = self._jitter_blocks(template.blocks)
                        core_slots = self._jitter_points(template.core_slots, amount=24.0)
                        
                        terminal_point = template.terminal_point
                        found_terminal = False
                        for _ in range(50):
                            tx = float(self.np_random.uniform(100.0, self.world_width - 100.0))
                            ty = float(self.np_random.uniform(100.0, self.world_height - 100.0))
                            t_point = np.array([tx, ty], dtype=np.float32)
                            dist_spawn = length(t_point - np.array(template.spawn_point))
                            dist_extract = length(t_point - np.array(template.extraction_point))
                            if dist_spawn > 250.0 and dist_extract > 200.0 and self._point_is_clear(t_point, blocks, PLAYER_RADIUS + 18.0):
                                terminal_point = (tx, ty)
                                found_terminal = True
                                break
                        
                        if not found_terminal:
                            terminal_point = self._jitter_point(template.terminal_point, amount=18.0)

                        if self._layout_is_valid(
                            blocks=blocks,
                            core_slots=core_slots,
                            spawn_point=template.spawn_point,
                            extraction_point=template.extraction_point,
                            terminal_point=terminal_point,
                        ):
                            self._assign_layout(template, template_index, blocks, core_slots, terminal_point)
                            return
                    self._assign_layout(template, template_index, template.blocks, template.core_slots, template.terminal_point)
                return

        if "large_proc" in self.curriculum_stage:
            self.world_width = 1500.0
            self.world_height = 950.0
            success = False
            for attempt in range(10):
                template = self._generate_procedural_layout()
                if self._layout_is_valid(
                    blocks=template.blocks,
                    core_slots=template.core_slots,
                    spawn_point=template.spawn_point,
                    extraction_point=template.extraction_point,
                    terminal_point=template.terminal_point,
                ):
                    self._assign_layout(template, 99, template.blocks, template.core_slots, template.terminal_point)
                    success = True
                    break
            if not success:
                from neon_arena.config import LARGE_TEMPLATES
                fallback = LARGE_TEMPLATES[0]
                self.active_template = fallback
                self._assign_layout(fallback, 0, fallback.blocks, fallback.core_slots, fallback.terminal_point)
                import sys
                print("WARNING: Procedural layout generation failed 10 times, fell back to authored layout", file=sys.stderr)
            return

        is_large = "large" in self.curriculum_stage
        
        if is_large:
            from neon_arena.config import LARGE_TEMPLATES
            templates = LARGE_TEMPLATES
            self.world_width = 1500.0
            self.world_height = 950.0
        else:
            from neon_arena.config import SMALL_TEMPLATES
            templates = SMALL_TEMPLATES
            self.world_width = 1200.0
            self.world_height = 760.0
            
        if self.curriculum_stage in ("route", "sequence"):
            template_index = 0
            template = templates[0]
            self.active_template = template
            self._assign_layout(template, template_index, template.blocks, template.core_slots, template.terminal_point)
            return
            
        template_index = int(self.np_random.integers(0, len(templates)))
        template = templates[template_index]
        self.active_template = template
        for _ in range(24):
            blocks = self._jitter_blocks(template.blocks)
            core_slots = self._jitter_points(template.core_slots, amount=24.0)
            
            terminal_point = template.terminal_point
            found_terminal = False
            for _ in range(50):
                tx = float(self.np_random.uniform(100.0, self.world_width - 100.0))
                ty = float(self.np_random.uniform(100.0, self.world_height - 100.0))
                t_point = np.array([tx, ty], dtype=np.float32)
                dist_spawn = length(t_point - np.array(template.spawn_point))
                dist_extract = length(t_point - np.array(template.extraction_point))
                if dist_spawn > 250.0 and dist_extract > 200.0 and self._point_is_clear(t_point, blocks, PLAYER_RADIUS + 18.0):
                    terminal_point = (tx, ty)
                    found_terminal = True
                    break
            
            if not found_terminal:
                terminal_point = self._jitter_point(template.terminal_point, amount=18.0)

            if self._layout_is_valid(
                blocks=blocks,
                core_slots=core_slots,
                spawn_point=template.spawn_point,
                extraction_point=template.extraction_point,
                terminal_point=terminal_point,
            ):
                self._assign_layout(template, template_index, blocks, core_slots, terminal_point)
                return
        self._assign_layout(template, template_index, template.blocks, template.core_slots, template.terminal_point)

    def _assign_layout(
        self,
        template: ArenaTemplate,
        template_index: int,
        blocks: tuple[Rect, ...],
        core_slots: tuple[tuple[float, float], ...],
        terminal_point: tuple[float, float],
    ) -> None:
        self.layout_name = template.name
        self.layout_id = template_index
        self.city_blocks = blocks
        self.core_slots = core_slots
        self.spawn_point = np.array(template.spawn_point, dtype=np.float32)
        self.extraction_point = np.array(template.extraction_point, dtype=np.float32)
        self.terminal_point = np.array(terminal_point, dtype=np.float32)
        self.drone_routes = template.drone_routes
        self.active_template = template

    def _jitter_blocks(self, blocks: tuple[Rect, ...]) -> tuple[Rect, ...]:
        jittered = []
        for block in blocks:
            width = float(np.clip(block.width + self.np_random.uniform(-12.0, 12.0), 105.0, 235.0))
            height = float(np.clip(block.height + self.np_random.uniform(-10.0, 10.0), 82.0, 190.0))
            x = float(np.clip(block.x + self.np_random.uniform(-22.0, 22.0), 54.0, self.world_width - width - 54.0))
            y = float(np.clip(block.y + self.np_random.uniform(-18.0, 18.0), 54.0, self.world_height - height - 54.0))
            jittered.append(Rect(x, y, width, height))
        return tuple(jittered)

    def _jitter_points(
        self,
        points: tuple[tuple[float, float], ...],
        *,
        amount: float,
    ) -> tuple[tuple[float, float], ...]:
        return tuple(self._jitter_point(point, amount=amount) for point in points)

    def _jitter_point(self, point: tuple[float, float], *, amount: float) -> tuple[float, float]:
        x = float(np.clip(point[0] + self.np_random.uniform(-amount, amount), 48.0, self.world_width - 48.0))
        y = float(np.clip(point[1] + self.np_random.uniform(-amount, amount), 48.0, self.world_height - 48.0))
        return x, y

    def _layout_is_valid(
        self,
        *,
        blocks: tuple[Rect, ...],
        core_slots: tuple[tuple[float, float], ...],
        spawn_point: tuple[float, float],
        extraction_point: tuple[float, float],
        terminal_point: tuple[float, float],
    ) -> bool:
        checkpoints = (spawn_point, extraction_point, terminal_point, *core_slots)
        if any(not self._point_is_clear(np.array(point, dtype=np.float32), blocks, PLAYER_RADIUS + 18.0) for point in checkpoints):
            return False
        return self._points_are_reachable(blocks, spawn_point, (extraction_point, terminal_point, *core_slots))

    def _point_is_clear(self, point: np.ndarray, blocks: tuple[Rect, ...], radius: float) -> bool:
        if (
            float(point[0]) < radius
            or float(point[0]) > self.world_width - radius
            or float(point[1]) < radius
            or float(point[1]) > self.world_height - radius
        ):
            return False
        return not any(circle_rect_overlap(point, radius, block) for block in blocks)

    def _points_are_reachable(
        self,
        blocks: tuple[Rect, ...],
        start: tuple[float, float],
        targets: tuple[tuple[float, float], ...],
    ) -> bool:
        grid = 25.0
        columns = int(self.world_width // grid) + 1
        rows = int(self.world_height // grid) + 1

        def center(cell: tuple[int, int]) -> np.ndarray:
            return np.array(
                (
                    min(self.world_width - PLAYER_RADIUS, cell[0] * grid + grid * 0.5),
                    min(self.world_height - PLAYER_RADIUS, cell[1] * grid + grid * 0.5),
                ),
                dtype=np.float32,
            )

        def passable(cell: tuple[int, int]) -> bool:
            x, y = cell
            if x < 0 or y < 0 or x >= columns or y >= rows:
                return False
            return self._point_is_clear(center(cell), blocks, PLAYER_RADIUS + 5.0)

        def nearest_passable(point: tuple[float, float]) -> tuple[int, int] | None:
            base = (int(point[0] // grid), int(point[1] // grid))
            candidates: list[tuple[int, int]] = []
            for radius in range(4):
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if max(abs(dx), abs(dy)) == radius:
                            candidates.append((base[0] + dx, base[1] + dy))
            for candidate in candidates:
                if passable(candidate):
                    return candidate
            return None

        start_cell = nearest_passable(start)
        if start_cell is None:
            return False
        target_cells = {nearest_passable(target) for target in targets}
        if None in target_cells:
            return False

        queue: deque[tuple[int, int]] = deque([start_cell])
        visited = {start_cell}
        while queue:
            cell = queue.popleft()
            if target_cells.issubset(visited):
                return True
            x, y = cell
            for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if neighbor not in visited and passable(neighbor):
                    visited.add(neighbor)
                    queue.append(neighbor)
        return target_cells.issubset(visited)

    def _create_drones(self) -> list[Drone]:
        if self.curriculum_stage in ("route", "sequence", "random-city", "large_easy", "large_proc_easy", "large_proc_cameras"):
            return []
        if self.curriculum_stage in ("sentry", "large_sentry_camera"):
            return [self._create_sentry()]
        drones = []
        is_proc = "large_proc" in self.curriculum_stage
        
        if is_proc:
            roles = ["patrol", "hunter", "patrol"]
            if self.curriculum_stage == "large_proc_patrols":
                roles = ["patrol", "patrol", "patrol"]
            
            for index, role in enumerate(roles):
                num_nodes = self.np_random.integers(3, 6)
                
                # Filter nodes for the start of the route to be away from the player
                start_candidates = [node for node in self.semantic_patrol_nodes if length(node - self.player) > 350.0]
                if not start_candidates:
                    start_candidates = [node for node in self.semantic_patrol_nodes if length(node - self.player) > 150.0]
                if not start_candidates:
                    start_candidates = list(self.semantic_patrol_nodes)
                
                start_node = start_candidates[self.np_random.integers(0, len(start_candidates))]
                
                # Pick other route waypoints from the remaining nodes
                other_candidates = [node for node in self.semantic_patrol_nodes if not np.allclose(node, start_node)]
                if len(other_candidates) >= num_nodes - 1:
                    selected_others = self.np_random.choice(
                        len(other_candidates), size=num_nodes - 1, replace=False
                    )
                    route_pts = [start_node] + [other_candidates[int(i)] for i in selected_others]
                else:
                    route_pts = [start_node] + other_candidates
                    while len(route_pts) < 3:
                        route_pts.append(self.terminal_point.copy())
                
                route = tuple((float(pt[0]), float(pt[1])) for pt in route_pts)
                start_pt = np.array(route[0], dtype=np.float32)
                drones.append(
                    Drone(
                        route=route,
                        position=start_pt,
                        velocity=np.zeros(2, dtype=np.float32),
                        route_index=1 % len(route),
                        base_speed=2.0 + index * 0.22,
                        role=role,
                        state="PATROL",
                    )
                )
        else:
            for index, route in enumerate(self.drone_routes):
                far_indices = [i for i, pt in enumerate(route) if length(np.array(pt, dtype=np.float32) - self.player) > 350.0]
                if not far_indices:
                    far_indices = [i for i, pt in enumerate(route) if length(np.array(pt, dtype=np.float32) - self.player) > 150.0]
                if not far_indices:
                    far_indices = list(range(len(route)))
                
                start_index = int(self.np_random.choice(far_indices))
                start = np.array(route[start_index], dtype=np.float32)
                role = DRONE_ROLES[index]
                if (self.curriculum_stage == "patrols" or self.curriculum_stage == "large_patrols_camera_no_hunters") and role == "hunter":
                    role = "patrol"
                drones.append(
                    Drone(
                        route=route,
                        position=start,
                        velocity=np.zeros(2, dtype=np.float32),
                        route_index=(start_index + 1) % len(route),
                        base_speed=2.0 + index * 0.22,
                        role=role,
                        state="PATROL",
                    )
                )
        drones.append(self._create_sentry())
        return drones

    def _create_sentry(self) -> Drone:
        if getattr(self, "preset_sentry_positions", None):
            s_pos = np.array(self.preset_sentry_positions.pop(0), dtype=np.float32)
            if self._can_occupy(s_pos):
                point = (float(s_pos[0]), float(s_pos[1]))
                return Drone(
                    route=(point,),
                    position=s_pos,
                    velocity=np.zeros(2, dtype=np.float32),
                    route_index=0,
                    base_speed=0.0,
                    role="sentry",
                    facing_angle=float(self.np_random.uniform(0, math.tau)),
                )
        for _ in range(100):
            sx = float(self.np_random.uniform(100.0, self.world_width - 100.0))
            sy = float(self.np_random.uniform(100.0, self.world_height - 100.0))
            s_pos = np.array([sx, sy], dtype=np.float32)
            if length(s_pos - self.spawn_point) > 350.0 and self._can_occupy(s_pos):
                point = (sx, sy)
                return Drone(
                    route=(point,),
                    position=s_pos,
                    velocity=np.zeros(2, dtype=np.float32),
                    route_index=0,
                    base_speed=0.0,
                    role="sentry",
                    facing_angle=float(self.np_random.uniform(0, math.tau)),
                )
        candidates = (
            self.terminal_point + np.array([0.0, -120.0], dtype=np.float32),
            self.terminal_point + np.array([115.0, 0.0], dtype=np.float32),
            self.terminal_point + np.array([-115.0, 0.0], dtype=np.float32),
            np.array((self.world_width * 0.5, self.world_height * 0.5), dtype=np.float32),
        )
        position = next((candidate for candidate in candidates if self._can_occupy(candidate)), self.terminal_point.copy())
        point = (float(position[0]), float(position[1]))
        return Drone(
            route=(point,),
            position=position.copy(),
            velocity=np.zeros(2, dtype=np.float32),
            route_index=0,
            base_speed=0.0,
            role="sentry",
            facing_angle=-math.pi * 0.5,
        )

    def _update_drones(self) -> None:
        for drone in self.drones:
            if self.drone_is_emp_stunned(drone):
                drone.velocity *= 0.72
                drone.state = "PATROL"
                drone.planned_path = []
                continue
                
            if getattr(drone, "path_recalc_cooldown", 0) > 0:
                drone.path_recalc_cooldown -= 1
            if getattr(drone, "state_timer", 0) > 0:
                drone.state_timer -= 1
                
            heat_speed = 1.0 + 0.22 * self.heat_ratio
            speed_multiplier = (1.28 if self.alarm_active else 1.0) * heat_speed
            
            if drone.role == "sentry":
                drone.velocity *= 0.0
                if self._drone_can_see_player(drone):
                    drone.lock_steps = PATROL_LOCK_STEPS
                    delta = self.player - drone.position
                    drone.facing_angle = math.atan2(float(delta[1]), float(delta[0]))
                elif drone.lock_steps > 0:
                    drone.lock_steps = max(0, drone.lock_steps - PATROL_LOST_LOCK_DECAY)
                    delta = self.player - drone.position
                    drone.facing_angle = math.atan2(float(delta[1]), float(delta[0]))
                else:
                    drone.facing_angle = (drone.facing_angle + SENTRY_TURN_RATE * (1.0 + 0.5 * self.heat_ratio)) % math.tau

                if drone.fire_cooldown > 0:
                    drone.fire_cooldown -= 1
                if self._drone_can_see_player(drone) and drone.fire_cooldown <= 0:
                    self._fire_sentry_projectile(drone)
                continue
                
            # Non-sentry drone state machine
            if self._drone_can_see_player(drone):
                if drone.state != "CHASE":
                    self.hunter_lock_count += 1
                drone.state = "CHASE"
                drone.lock_steps = HUNTER_LOCK_STEPS if drone.role == "hunter" else PATROL_LOCK_STEPS
                drone.target_pos = self.player.copy()
            else:
                decay = HUNTER_LOST_LOCK_DECAY if drone.role == "hunter" else PATROL_LOST_LOCK_DECAY
                drone.lock_steps = max(0, drone.lock_steps - decay)
                if drone.state == "CHASE":
                    if drone.lock_steps > 0:
                        # Continue moving to the last-known seen position (do not update to moving player coordinates)
                        pass
                    else:
                        drone.state = "SEARCH"
                        drone.state_timer = 120
                        last_seen = drone.target_pos if drone.target_pos is not None else drone.position.copy()
                        candidates = sorted(self.semantic_patrol_nodes, key=lambda n: length(n - last_seen))
                        search_nodes = candidates[:6]
                        self.np_random.shuffle(search_nodes)
                        drone.search_circuit = [np.array(n) for n in search_nodes[:3]]
                        drone.search_index = 0
                        drone.target_pos = drone.search_circuit[0] if drone.search_circuit else last_seen
                        drone.planned_path = []
                elif drone.state == "SEARCH":
                    if drone.state_timer <= 0:
                        drone.state = "PATROL"
                        drone.planned_path = []
                    else:
                        if drone.target_pos is not None and length(drone.position - drone.target_pos) < 15.0:
                            if hasattr(drone, "search_circuit") and drone.search_circuit:
                                drone.search_index = (drone.search_index + 1) % len(drone.search_circuit)
                                drone.target_pos = drone.search_circuit[drone.search_index]
                                drone.planned_path = []
                elif drone.state == "INVESTIGATE":
                    if drone.state_timer <= 0 or (drone.target_pos is not None and length(drone.position - drone.target_pos) < 15.0):
                        drone.state = "SEARCH"
                        drone.state_timer = 60
                        last_seen = drone.target_pos if drone.target_pos is not None else drone.position.copy()
                        candidates = sorted(self.semantic_patrol_nodes, key=lambda n: length(n - last_seen))
                        search_nodes = candidates[:4]
                        self.np_random.shuffle(search_nodes)
                        drone.search_circuit = [np.array(n) for n in search_nodes[:2]]
                        drone.search_index = 0
                        drone.target_pos = drone.search_circuit[0] if drone.search_circuit else last_seen
                        drone.planned_path = []
                elif drone.state == "SUSPICIOUS":
                    if drone.state_timer <= 0:
                        drone.state = "INVESTIGATE"
                        drone.state_timer = 90
                        drone.planned_path = []
                    else:
                        if drone.target_pos is not None:
                            delta = drone.target_pos - drone.position
                            if length(delta) > 0.001:
                                drone.facing_angle = math.atan2(float(delta[1]), float(delta[0]))
                        drone.velocity *= 0.0
                        continue

            if drone.state == "CHASE":
                speed = (drone.base_speed * 1.58 if drone.role == "hunter" else drone.base_speed * 1.28) * speed_multiplier
                speed = min(PLAYER_MAX_SPEED, speed)
            elif drone.state == "INVESTIGATE":
                speed = drone.base_speed * 1.15 * speed_multiplier
            elif drone.state == "SEARCH":
                speed = drone.base_speed * 0.9 * speed_multiplier
            else:
                speed = drone.base_speed * speed_multiplier
                drone.target_pos = np.array(drone.route[drone.route_index], dtype=np.float32)
                if length(drone.position - drone.target_pos) < 15.0:
                    drone.route_index = (drone.route_index + 1) % len(drone.route)
                    drone.target_pos = np.array(drone.route[drone.route_index], dtype=np.float32)
                    drone.planned_path = []

            if not getattr(drone, "planned_path", None) or drone.path_recalc_cooldown <= 0:
                if drone.target_pos is not None:
                    drone.planned_path = self._find_path_astar(drone.position, drone.target_pos)
                    drone.path_recalc_cooldown = 15

            if drone.planned_path:
                while drone.planned_path and length(drone.position - drone.planned_path[0]) < 12.0:
                    drone.planned_path.pop(0)
                    
                if drone.planned_path:
                    waypoint = drone.planned_path[0]
                    delta = waypoint - drone.position
                    if length(delta) > 0.001:
                        drone.velocity = normalized(delta) * speed
                        drone.position += drone.velocity
                        drone.facing_angle = math.atan2(float(drone.velocity[1]), float(drone.velocity[0]))
                    else:
                        drone.velocity *= 0.0
                else:
                    drone.velocity *= 0.0
            else:
                if drone.target_pos is not None:
                    delta = drone.target_pos - drone.position
                    if length(delta) > 0.001:
                        drone.velocity = normalized(delta) * speed
                        drone.position += drone.velocity
                        drone.facing_angle = math.atan2(float(drone.velocity[1]), float(drone.velocity[0]))
                    else:
                        drone.velocity *= 0.0

    def _fire_sentry_projectile(self, drone: Drone) -> None:
        direction = self._drone_forward(drone)
        spawn = drone.position + direction * (DRONE_RADIUS + SENTRY_PROJECTILE_RADIUS + 4.0)
        self.projectiles.append(
            Projectile(
                position=spawn.astype(np.float32),
                velocity=(direction * SENTRY_PROJECTILE_SPEED).astype(np.float32),
                life=SENTRY_PROJECTILE_LIFETIME,
            )
        )
        drone.fire_cooldown = SENTRY_PROJECTILE_COOLDOWN
        self.events.append(("shot", tuple(float(value) for value in spawn)))

    def _update_projectiles(self) -> None:
        active: list[Projectile] = []
        for projectile in self.projectiles:
            projectile.position = projectile.position + projectile.velocity
            projectile.life -= 1
            if projectile.life <= 0:
                continue
            if not self._projectile_in_bounds(projectile.position):
                continue
            if any(circle_rect_overlap(projectile.position, SENTRY_PROJECTILE_RADIUS, block) for block in self.city_blocks):
                continue
            active.append(projectile)
        self.projectiles = active

    def _projectile_in_bounds(self, position: np.ndarray) -> bool:
        return (
            SENTRY_PROJECTILE_RADIUS <= float(position[0]) <= self.world_width - SENTRY_PROJECTILE_RADIUS
            and SENTRY_PROJECTILE_RADIUS <= float(position[1]) <= self.world_height - SENTRY_PROJECTILE_RADIUS
        )

    def _move_player(self, velocity: np.ndarray) -> bool:
        collided = False
        x_candidate = self.player + np.array([velocity[0], 0.0], dtype=np.float32)
        if self._can_occupy(x_candidate):
            self.player = x_candidate
        else:
            collided = True
        y_candidate = self.player + np.array([0.0, velocity[1]], dtype=np.float32)
        if self._can_occupy(y_candidate):
            self.player = y_candidate
        else:
            collided = True
        return collided

    def _can_occupy(self, position: np.ndarray) -> bool:
        if (
            float(position[0]) < PLAYER_RADIUS
            or float(position[0]) > self.world_width - PLAYER_RADIUS
            or float(position[1]) < PLAYER_RADIUS
            or float(position[1]) > self.world_height - PLAYER_RADIUS
        ):
            return False
        if any(circle_rect_overlap(position, PLAYER_RADIUS, block) for block in self.city_blocks):
            return False
        # Check closed gates
        for gate in self.gates:
            if not gate.is_open:
                if circle_rect_overlap(position, PLAYER_RADIUS, gate.rect):
                    return False
        return True

    def _drone_collision(self) -> bool:
        return any(
            not self.drone_is_emp_stunned(drone)
            and length(drone.position - self.player) < DRONE_RADIUS + PLAYER_RADIUS
            for drone in self.drones
        )

    def _projectile_collision(self) -> bool:
        kept: list[Projectile] = []
        hit = False
        for projectile in self.projectiles:
            if length(projectile.position - self.player) < SENTRY_PROJECTILE_RADIUS + PLAYER_RADIUS:
                hit = True
                continue
            kept.append(projectile)
        self.projectiles = kept
        return hit

    def _separate_from_nearest_drone(self) -> None:
        threats = [
            drone
            for drone in self.drones
            if not self.drone_is_emp_stunned(drone)
            and length(drone.position - self.player) < DRONE_RADIUS + PLAYER_RADIUS + 6.0
        ]
        if not threats:
            self.velocity *= -0.42
            return
        drone = min(threats, key=lambda threat: length(threat.position - self.player))
        away = self.player - drone.position
        if length(away) <= 0.001:
            away = -self.heading
        direction = normalized(away)
        separation = DRONE_RADIUS + PLAYER_RADIUS + 14.0
        target = drone.position + direction * separation
        if self._can_occupy(target):
            self.player = target.astype(np.float32)
        self.velocity = direction * PLAYER_MAX_SPEED
        drone.velocity = -direction * max(1.0, drone.base_speed)
        drone.lock_steps = 0

    def drone_is_emp_stunned(self, drone: Drone) -> bool:
        return self.emp_timer > 0 and length(drone.position - self.player) <= EMP_RADIUS

    def _observation(self) -> np.ndarray:
        objective_delta = self.objective - self.player
        terminal_delta = self.terminal_point - self.player
        rays = [self._ray_distance(direction) / RAY_LENGTH for direction in self._ray_directions]
        ordered_drones = sorted(self.drones, key=lambda drone: length(drone.position - self.player))
        drone_values: list[float] = []
        for drone in ordered_drones[:3]:
            relative = drone.position - self.player
            exposure = min(1.0, self._drone_exposure(drone))
            lock_level = min(1.0, drone.lock_steps / HUNTER_LOCK_STEPS)
            risk = max(exposure, lock_level)
            role_value = 1.0 if drone.role == "hunter" else (0.25 if drone.role == "sentry" else -1.0)
            drone_values.extend(
                (
                    float(np.clip(relative[0] / 400.0, -1.0, 1.0)),
                    float(np.clip(relative[1] / 400.0, -1.0, 1.0)),
                    float(np.clip(drone.velocity[0] / 3.0, -1.0, 1.0)),
                    float(np.clip(drone.velocity[1] / 3.0, -1.0, 1.0)),
                    role_value,
                    float(risk * 2.0 - 1.0),
                )
            )
        while len(drone_values) < 18:
            drone_values.extend((-1.0, -1.0, 0.0, 0.0, -1.0, -1.0))
        ordered_projectiles = sorted(self.projectiles, key=lambda projectile: length(projectile.position - self.player))
        projectile_values: list[float] = []
        for projectile in ordered_projectiles[:2]:
            relative = projectile.position - self.player
            projectile_values.extend(
                (
                    float(np.clip(relative[0] / 200.0, -1.0, 1.0)),
                    float(np.clip(relative[1] / 200.0, -1.0, 1.0)),
                    float(np.clip(projectile.velocity[0] / SENTRY_PROJECTILE_SPEED, -1.0, 1.0)),
                    float(np.clip(projectile.velocity[1] / SENTRY_PROJECTILE_SPEED, -1.0, 1.0)),
                )
            )
        while len(projectile_values) < 8:
            projectile_values.extend((0.0, 0.0, 0.0, 0.0))
        values = [
            float(np.clip(self.velocity[0] / DASH_SPEED, -1.0, 1.0)),
            float(np.clip(self.velocity[1] / DASH_SPEED, -1.0, 1.0)),
            float(self.player[0] / self.world_width * 2.0 - 1.0),
            float(self.player[1] / self.world_height * 2.0 - 1.0),
            float(self.dash_energy / DASH_ENERGY_MAX * 2.0 - 1.0),
            1.0 if self.dash_energy > 0.0 else -1.0,
            float(self.shield / 3.0 * 2.0 - 1.0),
            float(self.heat_ratio * 2.0 - 1.0),
            float(np.clip(objective_delta[0] / self.world_width, -1.0, 1.0)),
            float(np.clip(objective_delta[1] / self.world_height, -1.0, 1.0)),
            float(np.clip(length(objective_delta) / math.hypot(self.world_width, self.world_height), 0.0, 1.0)),
            float(self.cores_collected / len(self.cores) * 2.0 - 1.0),
            1.0 if self.alarm_active else -1.0,
            1.0 if self.terminal_available else -1.0,
            1.0 if self.emp_available else -1.0,
            float(np.clip(terminal_delta[0] / self.world_width, -1.0, 1.0)),
            float(np.clip(terminal_delta[1] / self.world_height, -1.0, 1.0)),
            float(self.emp_timer / EMP_DURATION_STEPS * 2.0 - 1.0),
            *projectile_values,
            *rays,
            *drone_values,
        ]
        
        # Determine target
        if "large_proc" in self.curriculum_stage:
            target = self._get_next_route_target()
        else:
            target = self.objective

        # Check if gates open states have changed since last grid build
        gates_state = tuple(gate.is_open for gate in self.gates)
        if not hasattr(self, "_last_gates_state") or self._last_gates_state != gates_state:
            self._build_grid()
            self._last_gates_state = gates_state
            if hasattr(self, "_dist_map_target"):
                delattr(self, "_dist_map_target")

        # Ensure dist_map matches this target
        if not hasattr(self, "_dist_map_target") or not np.array_equal(self._dist_map_target, target):
            self.dist_map = self._compute_distance_map(target)
            self._dist_map_target = target.copy()

        # Route-aware observations:
        route_dist = self._route_distance()
        route_dist_norm = float(np.clip(route_dist / math.hypot(self.world_width, self.world_height), 0.0, 1.0))
        
        path = self._get_route_waypoints(self.player, target, self.dist_map)
        pc, pr = self._world_to_grid(self.player)
        path_found_val = -1.0 if self.dist_map[pc, pr] < 0 else 1.0
        
        wp1 = path[0] if len(path) >= 1 else target
        wp1_delta = wp1 - self.player
        wp1_dx_norm = float(np.clip(wp1_delta[0] / 200.0, -1.0, 1.0))
        wp1_dy_norm = float(np.clip(wp1_delta[1] / 200.0, -1.0, 1.0))
        
        wp2 = path[1] if len(path) >= 2 else target
        wp2_delta = wp2 - self.player
        wp2_dx_norm = float(np.clip(wp2_delta[0] / 200.0, -1.0, 1.0))
        wp2_dy_norm = float(np.clip(wp2_delta[1] / 200.0, -1.0, 1.0))
        
        los_flag = 1.0 if self._has_line_of_sight(self.player, self.objective) else -1.0
        clearance_flags = self._direction_clearance_flags()
        
        extended_values = values + [
            route_dist_norm,
            wp1_dx_norm,
            wp1_dy_norm,
            wp2_dx_norm,
            wp2_dy_norm,
            los_flag,
            path_found_val
        ] + clearance_flags
        
        # New v4 observations (75-83)
        wp1_dist = length(wp1_delta)
        wp1_dist_norm = float(np.clip(wp1_dist / 200.0, 0.0, 1.0))
        
        closed_gates = [g for g in self.gates if not g.is_open]
        if closed_gates:
            nearest_gate = min(
                closed_gates,
                key=lambda g: length(np.array([g.rect.x + g.rect.width / 2.0, g.rect.y + g.rect.height / 2.0], dtype=np.float32) - self.player)
            )
            g_center = np.array([nearest_gate.rect.x + nearest_gate.rect.width / 2.0, nearest_gate.rect.y + nearest_gate.rect.height / 2.0], dtype=np.float32)
            gate_delta = g_center - self.player
            gate_dist = length(gate_delta)
            gate_dist_norm = float(np.clip(gate_dist / math.hypot(self.world_width, self.world_height), 0.0, 1.0))
            gate_dx = float(np.clip(gate_delta[0] / self.world_width, -1.0, 1.0))
            gate_dy = float(np.clip(gate_delta[1] / self.world_height, -1.0, 1.0))
        else:
            gate_dist_norm = 1.0
            gate_dx = 0.0
            gate_dy = 0.0
            
        exit_delta = self.extraction_point - self.player
        if self.alarm_active:
            exit_dx = float(np.clip(exit_delta[0] / self.world_width, -1.0, 1.0))
            exit_dy = float(np.clip(exit_delta[1] / self.world_height, -1.0, 1.0))
        else:
            exit_dx = 0.0
            exit_dy = 0.0
            
        # Room-level next route target (obs[78-80])
        if "large_proc" in self.curriculum_stage:
            target_obs = target
        else:
            target_obs = wp1
            
        target_delta = target_obs - self.player
        target_dist = length(target_delta)
        target_dist_norm = float(np.clip(target_dist / 400.0, 0.0, 1.0))
        target_dx = float(np.clip(target_delta[0] / 400.0, -1.0, 1.0))
        target_dy = float(np.clip(target_delta[1] / 400.0, -1.0, 1.0))

        # Room role encoding (obs[81])
        if "large_proc" in self.curriculum_stage:
            curr_room_id = self._get_room_id_at_position(self.player)
            curr_room_node = next((n for n in self.room_nodes if n.id == curr_room_id), None)
            if curr_room_node is not None:
                role = curr_room_node.role
                if role in ("SPAWN", "MARKET"):
                    room_role_val = -0.6
                elif role == "PLAZA":
                    room_role_val = -0.2
                elif role in ("SECURITY", "SEC-HUB"):
                    room_role_val = 0.2
                elif role == "VAULT":
                    room_role_val = 0.6
                elif role in ("EXTRACT", "CONNECT"):
                    room_role_val = 1.0
                else:
                    room_role_val = -1.0
            else:
                room_role_val = -1.0
        else:
            room_role_val = -1.0

        final_values = extended_values + [
            wp1_dist_norm,      # obs[75]
            wp1_dx_norm,        # obs[76]
            wp1_dy_norm,        # obs[77]
            target_dist_norm,   # obs[78]
            target_dx,          # obs[79]
            target_dy,          # obs[80]
            room_role_val,      # obs[81]
            exit_dx,            # obs[82]
            exit_dy,            # obs[83]
        ]
        
        return np.asarray(final_values, dtype=np.float32)

    def _start_alarm(self) -> None:
        if self.alarm_active:
            return
        self.alarm_active = True
        self.alarm_steps = 0
        self.dash_cooldown = 0
        self.dash_energy = DASH_ENERGY_MAX
        self.events.append(("alarm", tuple(float(value) for value in self.extraction_point)))
        for drone in self.drones:
            if drone.role != "sentry" and not self.drone_is_emp_stunned(drone):
                drone.state = "INVESTIGATE"
                drone.state_timer = 240
                drone.target_pos = self.player.copy()
                drone.planned_path = []

    @property
    def hunter_lock_range(self) -> float:
        return self._hunter_lock_range()

    def _update_stalled(self) -> bool:
        # Safety Bypasses: Do not accumulate stall steps during active evasion or stealth
        is_locked = any(d.lock_steps > 0 for d in self.drones)
        if is_locked or getattr(self, "emp_timer", 0) > 0:
            self.stalled_steps = 0
            return False

        potential = self._route_potential()
        if potential > self.best_mission_potential + 0.0004:
            self.best_mission_potential = potential
            self.stalled_steps = 0
            return False
        self.stalled_steps += 1
        return self.stalled_steps > STALL_TERMINATE_STEPS

    def _hunter_lock_range(self) -> float:
        range_bonus = self.cores_collected * HUNTER_RANGE_PER_CORE
        if self.alarm_active:
            range_bonus += HUNTER_ALARM_RANGE_BONUS
        range_bonus += 70.0 * self.heat_ratio
        return HUNTER_LOCK_RANGE + range_bonus

    def _hunter_exposure(self, drone: Drone) -> float:
        return self._drone_exposure(drone) if drone.role == "hunter" else 0.0

    def _drone_exposure(self, drone: Drone) -> float:
        if not self._drone_can_see_player(drone):
            return 0.0
        delta = self.player - drone.position
        distance = length(delta)
        lock_range = self._drone_vision_range(drone)
        distance_pressure = 1.0 - distance / max(1.0, lock_range)
        lock_pressure = 0.28 if drone.lock_steps > 0 else 0.0
        return float(np.clip(0.35 + distance_pressure + lock_pressure, 0.0, 1.35))

    def _drone_can_see_player(self, drone: Drone) -> bool:
        if self.drone_is_emp_stunned(drone):
            return False
        delta = self.player - drone.position
        distance = length(delta)
        if distance > self._drone_vision_range(drone):
            return False
        direction = normalized(delta)
        forward = self._drone_forward(drone)
        if float(np.dot(forward, direction)) < self._drone_vision_cosine(drone):
            return False
        return self._has_line_of_sight(drone.position, self.player)

    def _drone_vision_range(self, drone: Drone) -> float:
        if drone.role == "hunter":
            return self._hunter_lock_range()
        if drone.role == "sentry":
            return SENTRY_LOCK_RANGE + 55.0 * self.heat_ratio
        return PATROL_LOCK_RANGE + 35.0 * self.heat_ratio

    def _drone_vision_cosine(self, drone: Drone) -> float:
        if drone.role == "hunter":
            return HUNTER_VISION_COSINE
        if drone.role == "sentry":
            return 0.56
        return 0.72

    def _total_hunter_exposure(self) -> float:
        return float(sum(self._hunter_exposure(drone) for drone in self.drones))

    def _total_drone_exposure(self) -> float:
        return float(sum(self._drone_exposure(drone) for drone in self.drones))

    def _drone_forward(self, drone: Drone) -> np.ndarray:
        if drone.role == "sentry":
            return np.array([math.cos(drone.facing_angle), math.sin(drone.facing_angle)], dtype=np.float32)
        if length(drone.velocity) > 0.001:
            return normalized(drone.velocity)
        target = np.array(drone.route[drone.route_index], dtype=np.float32)
        route_delta = target - drone.position
        if length(route_delta) > 0.001:
            return normalized(route_delta)
        return np.array([1.0, 0.0], dtype=np.float32)

    def _has_line_of_sight(self, origin: np.ndarray, target: np.ndarray) -> bool:
        delta = target - origin
        distance = length(delta)
        if distance <= 1e-6:
            return True
        direction = normalized(delta)
        for block in self.city_blocks:
            hit_distance = ray_rect_distance(origin, direction, block)
            if hit_distance is not None and hit_distance < distance:
                return False
        return True

    def _ray_distance(self, direction: np.ndarray) -> float:
        candidates: list[float] = []
        for block in self.city_blocks:
            distance = ray_rect_distance(self.player, direction, block)
            if distance is not None:
                candidates.append(distance)
                
        # Ray cast against closed gates
        for gate in self.gates:
            if not gate.is_open:
                distance = ray_rect_distance(self.player, direction, gate.rect)
                if distance is not None:
                    candidates.append(distance)
                    
        # Ray cast against active non-stunned drones
        for drone in self.drones:
            if not self.drone_is_emp_stunned(drone):
                w = self.player - drone.position
                b = np.dot(w, direction)
                c = np.dot(w, w) - (DRONE_RADIUS + PLAYER_RADIUS) ** 2
                disc = b ** 2 - c
                if disc >= 0:
                    t1 = -b - math.sqrt(disc)
                    t2 = -b + math.sqrt(disc)
                    if t1 >= 0:
                        candidates.append(t1)
                    elif t2 >= 0:
                        candidates.append(t2)
                        
        if float(direction[0]) > 1e-8:
            candidates.append((self.world_width - float(self.player[0])) / float(direction[0]))
        elif float(direction[0]) < -1e-8:
            candidates.append(-float(self.player[0]) / float(direction[0]))
        if float(direction[1]) > 1e-8:
            candidates.append((self.world_height - float(self.player[1])) / float(direction[1]))
        elif float(direction[1]) < -1e-8:
            candidates.append(-float(self.player[1]) / float(direction[1]))
        positive = [distance for distance in candidates if distance >= 0.0]
        return min(RAY_LENGTH, min(positive, default=RAY_LENGTH))

    def _info(
        self,
        *,
        collision: bool = False,
        terminal: bool = False,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "cores": self.cores_collected,
            "shield": self.shield,
            "dash_ready": self.dash_energy > 0.0,
            "dash_energy": self.dash_energy,
            "objective": self.objective_label,
            "episode_step": self.step_count,
            "episode_reward": self.episode_reward,
            "collision": collision,
            "extracted": self.extracted,
            "layout": self.layout_name,
            "hunter_range": self._hunter_lock_range(),
            "hunter_exposure_steps": self.hunter_exposure_steps,
            "hunter_locks": self.hunter_lock_count,
            "stalled_steps": self.stalled_steps,
            "curriculum_stage": self.curriculum_stage,
            "heat": self.heat,
            "max_heat": self.max_heat,
            "emp_available": self.emp_available,
        }
        if terminal or self.step_count >= MAX_STEPS or self.stalled_steps > STALL_TERMINATE_STEPS:
            info["is_success"] = self.extracted
            info["human_score"] = self._human_score()
            
            if self.extracted:
                info["fail_reason"] = "none"
            elif self.shield <= 0:
                info["fail_reason"] = "courier_down"
            elif self.stalled_steps > STALL_TERMINATE_STEPS:
                info["fail_reason"] = "stalled"
            else:
                info["fail_reason"] = "timeout"
                
            # Export reward components
            for key, val in self.episode_rewards.items():
                info[f"reward_{key}"] = val
                
            # Export behavioral metrics
            info["episode_success"] = float(self.extracted)
            info["episode_cores_collected"] = float(self.cores_collected)
            info["episode_extracted"] = float(self.extracted)
            info["episode_timed_out"] = float(self.step_count >= MAX_STEPS and not self.extracted and self.shield > 0)
            info["episode_shield_depleted"] = float(self.shield <= 0)
            info["episode_final_heat"] = float(self.heat)
            info["episode_max_heat"] = float(self.max_heat)
            info["episode_episode_length"] = float(self.step_count)
            info["episode_final_room"] = float(self.current_room_id)
            info["episode_timeout_with_3_cores"] = float(self.step_count >= MAX_STEPS and not self.extracted and self.cores_collected == len(self.cores))
            info["episode_missing_cores_on_timeout"] = float(max(0, len(self.cores) - self.cores_collected) if (self.step_count >= MAX_STEPS and not self.extracted) else 0)
            
            info["route_wrong_room_steps"] = float(self.total_wrong_room_steps)
            info["route_objective_room_entries"] = float(self.objective_room_entries)
            info["route_objective_room_reached"] = float(self.objective_room_reached_flag)
            info["route_room_transitions"] = float(self.room_transitions)
            info["route_rooms_visited_unique"] = float(len(self.visited_rooms))
            info["route_same_room_max_steps"] = float(self.same_room_max_steps)
            info["route_next_doorway_reached"] = float(self.next_doorway_reached_count)
            
            info["combat_drone_contacts"] = float(self.drone_contacts_count)
            info["combat_projectile_hits"] = float(self.projectile_hits_count)
            
            info["stealth_camera_alerts"] = float(self.camera_alerts_count)
            info["stealth_cone_steps"] = float(self.cone_steps_count)
            
            info["emp_uses"] = float(self.emp_uses_count)
            info["emp_invalid_presses"] = float(self.emp_invalid_presses_count)
            info["emp_wasted_uses"] = float(self.emp_wasted_uses_count)
            info["emp_effective_uses"] = float(self.emp_effective_uses_count)
            
            info["loot_items_collected"] = float(self.loot_items_collected_count)
            info["loot_reward_capped"] = float(self.loot_reward_capped_flag)
            
            # Map metrics
            info["map_route_directness"] = getattr(self, "metric_route_directness", 1.0)
            info["map_loop_count"] = float(getattr(self, "metric_loop_count", 0))
            info["map_locked_edge_count"] = float(getattr(self, "metric_locked_edge_count", 0))
            info["map_camera_coverage"] = float(getattr(self, "metric_camera_coverage", 0))
            info["map_hazard_rooms"] = float(getattr(self, "metric_hazard_room_count", 0))
            
        return info

    def _human_score(self) -> int:
        score = self.cores_collected * 500
        if self.extracted:
            score += 3000
            score += max(0, MAX_STEPS - self.step_count) * 2
        score -= self.damage_taken * 300
        score -= self.dash_count * 3
        
        # Add loot scores
        loot_score = 0
        for item in self.loot:
            if item.collected:
                if item.type == "data_shard":
                    loot_score += 300
                elif item.type == "gold_cache":
                    loot_score += 800
                elif item.type == "black_box":
                    loot_score += 1200
        score += loot_score
        
        return max(0, int(score))

    def _result(self, *, truncated: bool) -> dict[str, Any]:
        if self.extracted:
            status = "EXTRACTED"
        elif truncated:
            if self.stalled_steps > STALL_TERMINATE_STEPS:
                status = "SIGNAL LOST // STALLED"
            else:
                status = "SIGNAL LOST // TIMEOUT"
        else:
            status = "COURIER DOWN"
        return {
            "status": status,
            "score": self._human_score(),
            "cores": self.cores_collected,
            "hits": self.damage_taken,
            "wall_hits": self.wall_hits,
            "hunter_exposure_steps": self.hunter_exposure_steps,
            "dashes": self.dash_count,
            "boosts": self.dash_count,
            "steps": self.step_count,
            "reward": self.episode_reward,
            "heat": self.heat,
            "max_heat": self.max_heat,
            "emp_used": not self.emp_available and not self.terminal_available,
        }

    def _build_grid(self) -> None:
        self.grid_passable = np.ones((self.cols, self.rows), dtype=bool)
        clearance = PLAYER_RADIUS + 5.0
        for c in range(self.cols):
            for r in range(self.rows):
                x = (c + 0.5) * self.grid_size
                y = (r + 0.5) * self.grid_size
                x = min(self.world_width - 1.0, max(1.0, x))
                y = min(self.world_height - 1.0, max(1.0, y))
                point = np.array([x, y], dtype=np.float32)
                # Passable if block is clear and it does not overlap with any closed gates
                block_clear = self._point_is_clear(point, self.city_blocks, clearance)
                gate_clear = not any(circle_rect_overlap(point, clearance, g.rect) for g in self.gates if not g.is_open)
                self.grid_passable[c, r] = block_clear and gate_clear

    def _world_to_grid(self, pos: np.ndarray) -> tuple[int, int]:
        c = int(np.clip(pos[0] // self.grid_size, 0, self.cols - 1))
        r = int(np.clip(pos[1] // self.grid_size, 0, self.rows - 1))
        return c, r

    def _compute_distance_map(self, target_pos: np.ndarray) -> np.ndarray:
        dist = np.full((self.cols, self.rows), -1.0, dtype=np.float32)
        tc, tr = self._world_to_grid(target_pos)
        
        if not self.grid_passable[tc, tr]:
            found = False
            curr_room = self._get_room_id_at_position(self.player)
            for radius in range(1, 5):
                candidates = []
                for dc in range(-radius, radius + 1):
                    for dr in range(-radius, radius + 1):
                        if max(abs(dc), abs(dr)) == radius:
                            nc, nr = tc + dc, tr + dr
                            if 0 <= nc < self.cols and 0 <= nr < self.rows and self.grid_passable[nc, nr]:
                                candidates.append((nc, nr))
                if candidates:
                    room_candidates = [
                        p for p in candidates
                        if self._get_room_id_at_position(np.array([(p[0] + 0.5) * self.grid_size, (p[1] + 0.5) * self.grid_size], dtype=np.float32)) == curr_room
                    ]
                    best_candidates = room_candidates if room_candidates else candidates
                    tc, tr = min(best_candidates, key=lambda p: float(np.sum(((np.array(p) + 0.5) * self.grid_size - target_pos) ** 2)))
                    found = True
                    break
        
        queue = deque([((tc, tr), 0.0)])
        dist[tc, tr] = 0.0
        
        while queue:
            (c, r), d = queue.popleft()
            
            for dc, dr in [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ]:
                nc, nr = c + dc, r + dr
                if 0 <= nc < self.cols and 0 <= nr < self.rows:
                    if self.grid_passable[nc, nr]:
                        if dc != 0 and dr != 0:
                            if not self.grid_passable[c + dc, r] or not self.grid_passable[c, r + dr]:
                                continue
                        step = 1.41421356 if dc != 0 and dr != 0 else 1.0
                        nd = d + step
                        if dist[nc, nr] < 0 or nd < dist[nc, nr]:
                            dist[nc, nr] = nd
                            queue.append(((nc, nr), nd))
        return dist

    def _update_navigation(self) -> None:
        self._build_grid()
        target = self._get_next_route_target() if "large_proc" in self.curriculum_stage else self.objective
        self.dist_map = self._compute_distance_map(target)
        self._dist_map_target = target.copy()

    def _get_route_waypoints(self, start_pos: np.ndarray, target_pos: np.ndarray, dist_map: np.ndarray) -> list[np.ndarray]:
        sc, sr = self._world_to_grid(start_pos)
        tc, tr = self._world_to_grid(target_pos)
        
        if dist_map[sc, sr] < 0:
            found_cell = False
            for radius in range(1, 5):
                candidates = []
                for dc in range(-radius, radius + 1):
                    for dr in range(-radius, radius + 1):
                        if max(abs(dc), abs(dr)) == radius:
                            nc, nr = sc + dc, sr + dr
                            if 0 <= nc < self.cols and 0 <= nr < self.rows and dist_map[nc, nr] >= 0:
                                candidates.append((nc, nr))
                if candidates:
                    sc, sr = min(candidates, key=lambda p: float(np.sum(((np.array(p) + 0.5) * self.grid_size - start_pos) ** 2)))
                    found_cell = True
                    break
            if not found_cell:
                return []
            
        path = []
        curr_c, curr_r = sc, sr
        visited = {(curr_c, curr_r)}
        
        max_len = self.cols * self.rows
        for _ in range(max_len):
            if (curr_c == tc and curr_r == tr) or dist_map[curr_c, curr_r] == 0.0:
                break
                
            best_neighbor = None
            min_d = dist_map[curr_c, curr_r]
            
            for dc, dr in [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ]:
                nc, nr = curr_c + dc, curr_r + dr
                if 0 <= nc < self.cols and 0 <= nr < self.rows:
                    nd = dist_map[nc, nr]
                    if nd >= 0 and nd < min_d and (nc, nr) not in visited:
                        if dc != 0 and dr != 0:
                            if not self.grid_passable[curr_c + dc, curr_r] or not self.grid_passable[curr_c, curr_r + dr]:
                                continue
                        min_d = nd
                        best_neighbor = (nc, nr)
                        
            if best_neighbor is None:
                break
                
            curr_c, curr_r = best_neighbor
            visited.add((curr_c, curr_r))
            path.append(np.array([(curr_c + 0.5) * self.grid_size, (curr_r + 0.5) * self.grid_size], dtype=np.float32))
            
        return path

    def _find_path_astar(self, start_pos: np.ndarray, target_pos: np.ndarray) -> list[np.ndarray]:
        sc, sr = self._world_to_grid(start_pos)
        tc, tr = self._world_to_grid(target_pos)
        
        if not self.grid_passable[tc, tr]:
            found_cell = False
            start_room = self._get_room_id_at_position(start_pos)
            for radius in range(1, 5):
                candidates = []
                for dc in range(-radius, radius + 1):
                    for dr in range(-radius, radius + 1):
                        if max(abs(dc), abs(dr)) == radius:
                            nc, nr = tc + dc, tr + dr
                            if 0 <= nc < self.cols and 0 <= nr < self.rows and self.grid_passable[nc, nr]:
                                candidates.append((nc, nr))
                if candidates:
                    room_candidates = [
                        p for p in candidates
                        if self._get_room_id_at_position(np.array([(p[0] + 0.5) * self.grid_size, (p[1] + 0.5) * self.grid_size], dtype=np.float32)) == start_room
                    ]
                    best_candidates = room_candidates if room_candidates else candidates
                    tc, tr = min(best_candidates, key=lambda p: float(np.sum(((np.array(p) + 0.5) * self.grid_size - target_pos) ** 2)))
                    found_cell = True
                    break
            if not found_cell:
                return [target_pos.copy()]

        if sc == tc and sr == tr:
            return [target_pos.copy()]
            
        import heapq
        
        open_set = []
        heapq.heappush(open_set, (0.0, (sc, sr)))
        
        came_from = {}
        g_score = {(sc, sr): 0.0}
        
        def h(c, r):
            return math.hypot(c - tc, r - tr)
            
        max_iterations = 800
        iterations = 0
        
        found = False
        while open_set and iterations < max_iterations:
            iterations += 1
            _, current = heapq.heappop(open_set)
            
            if current == (tc, tr):
                found = True
                break
                
            cc, cr = current
            current_g = g_score[current]
            
            for dc, dr in [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ]:
                nc, nr = cc + dc, cr + dr
                if 0 <= nc < self.cols and 0 <= nr < self.rows:
                    if not self.grid_passable[nc, nr]:
                        continue
                    if dc != 0 and dr != 0:
                        if not self.grid_passable[cc + dc, cr] or not self.grid_passable[cc, cr + dr]:
                            continue
                    step_cost = 1.41421356 if dc != 0 and dr != 0 else 1.0
                    tentative_g = current_g + step_cost
                    
                    neighbor = (nc, nr)
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + h(nc, nr)
                        heapq.heappush(open_set, (f_score, neighbor))
                        came_from[neighbor] = current
                        
        if not found:
            return [target_pos.copy()]
            
        path_cells = []
        curr = (tc, tr)
        while curr != (sc, sr):
            path_cells.append(curr)
            curr = came_from[curr]
        path_cells.reverse()
        
        world_path = []
        for c, r in path_cells:
            world_path.append(np.array([(c + 0.5) * self.grid_size, (r + 0.5) * self.grid_size], dtype=np.float32))
            
        if world_path:
            world_path[-1] = target_pos.copy()
        return world_path


    def _grid_path_distance(self, p1: np.ndarray | tuple[float, float], p2: np.ndarray | tuple[float, float]) -> float:
        t1 = (round(float(p1[0]), 2), round(float(p1[1]), 2))
        t2 = (round(float(p2[0]), 2), round(float(p2[1]), 2))
        key = (t1, t2) if t1 <= t2 else (t2, t1)
        
        if not hasattr(self, "_door_distance_cache"):
            self._door_distance_cache = {}
            
        if key not in self._door_distance_cache:
            p1_arr = np.array(p1, dtype=np.float32)
            p2_arr = np.array(p2, dtype=np.float32)
            path = self._find_path_astar(p1_arr, p2_arr)
            if not path:
                d = float(length(p2_arr - p1_arr))
            else:
                d = float(length(path[0] - p1_arr))
                for i in range(1, len(path)):
                    d += float(length(path[i] - path[i-1]))
                d += float(length(p2_arr - path[-1]))
            self._door_distance_cache[key] = d
            
        return self._door_distance_cache[key]

    def _route_distance(self) -> float:
        pc, pr = self._world_to_grid(self.player)
        if self.dist_map[pc, pr] < 0:
            return float(length(self.objective - self.player))
            
        path = self._get_route_waypoints(self.player, self._dist_map_target, self.dist_map)
        tc, tr = self._world_to_grid(self._dist_map_target)
        
        if not path and (pc != tc or pr != tr):
            return float(length(self.objective - self.player))
            
        if pc == tc and pr == tr:
            dist_to_route_target = float(length(self._dist_map_target - self.player))
        else:
            wp1 = path[0]
            w1c, w1r = self._world_to_grid(wp1)
            dist_to_route_target = float(self.dist_map[w1c, w1r]) * self.grid_size + float(length(wp1 - self.player))
        
        if "large_proc" in self.curriculum_stage:
            curr_room = self.player_current_room
            if self.terminal_available and not self.emp_available:
                active_target = self.terminal_point.copy()
            else:
                active_target = self.objective.copy()
            obj_room = self._get_room_id_at_position(active_target)
            if curr_room == obj_room:
                return dist_to_route_target
                
            room_path = self._dijkstra_room_path(curr_room, obj_room)
            if len(room_path) >= 2:
                remaining_dist = 0.0
                prev_doorway_center = self._dist_map_target.copy()
                for i in range(1, len(room_path) - 1):
                    r_a = room_path[i]
                    r_b = room_path[i+1]
                    doorway_found = False
                    for edge in self.door_edges:
                        if (edge.room_a == r_a and edge.room_b == r_b) or \
                           (edge.room_a == r_b and edge.room_b == r_a):
                            center = np.array(edge.center, dtype=np.float32)
                            remaining_dist += self._grid_path_distance(prev_doorway_center, center)
                            prev_doorway_center = center
                            doorway_found = True
                            break
                    if not doorway_found:
                        remaining_dist += self._grid_path_distance(prev_doorway_center, active_target)
                        break
                else:
                    remaining_dist += self._grid_path_distance(prev_doorway_center, active_target)
                return dist_to_route_target + remaining_dist
            
        return dist_to_route_target

    def _route_potential(self) -> float:
        if self.extracted:
            return float(len(self.cores) + 1)
        if not self.cores:
            return 0.0
        route_dist = self._route_distance()
        world_diagonal = math.hypot(self.world_width, self.world_height)
        objective_progress = 1.0 - min(1.0, route_dist / world_diagonal)
        return float(self.cores_collected + objective_progress)
        
    def _get_district_id(self, position: np.ndarray) -> int:
        mid_x = self.world_width / 2.0
        mid_y = self.world_height / 2.0
        x, y = position[0], position[1]
        if x < mid_x:
            if y < mid_y:
                return 0  # Top-Left (Market/Maze)
            else:
                return 1  # Bottom-Left (Open Plaza)
        else:
            if y < mid_y:
                return 2  # Top-Right (Security Hub)
            else:
                return 3  # Bottom-Right (Server Vault/Extraction)

    def _get_room_id_at_position(self, pos: np.ndarray | tuple[float, float]) -> int:
        px, py = pos[0], pos[1]
        for node in self.room_nodes:
            r = node.rect
            if r.left <= px <= r.right and r.top <= py <= r.bottom:
                return node.id
        # Fallback to closest room by center distance
        best_id = 0
        best_dist = float('inf')
        for node in self.room_nodes:
            cx, cy = node.center
            d = math.hypot(px - cx, py - cy)
            if d < best_dist:
                best_dist = d
                best_id = node.id
        return best_id

    def _bfs_room_path(self, start_room: int, end_room: int) -> list[int]:
        if start_room == end_room:
            return [start_room]
        queue = deque([[start_room]])
        visited = {start_room}
        while queue:
            path = queue.popleft()
            curr = path[-1]
            if curr == end_room:
                return path
            for neighbor in self.room_adjacency.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return [start_room]

    def _dijkstra_room_path(self, start_room: int, end_room: int) -> list[int]:
        if start_room == end_room:
            return [start_room]
            
        import heapq
        
        num_rooms = len(self.room_nodes)
        dist = {i: float('inf') for i in range(num_rooms)}
        prev = {i: None for i in range(num_rooms)}
        dist[start_room] = 0.0
        
        pq = [(0.0, start_room)]
        
        while pq:
            d, curr = heapq.heappop(pq)
            if d > dist[curr]:
                continue
            if curr == end_room:
                break
                
            for neighbor in self.room_adjacency.get(curr, []):
                edge = None
                for de in self.door_edges:
                    if (de.room_a == curr and de.room_b == neighbor) or (de.room_a == neighbor and de.room_b == curr):
                        edge = de
                        break
                        
                if edge is None:
                    continue
                    
                edge_cost = 1.0
                if getattr(edge, "is_locked", False) or getattr(edge, "is_blocked_by_layout", False):
                    edge_cost += 1e9
                else:
                    n_curr = self.room_nodes[curr]
                    n_neigh = self.room_nodes[neighbor]
                    dist_rooms = math.hypot(n_curr.center[0] - n_neigh.center[0], n_curr.center[1] - n_neigh.center[1])
                    edge_cost += dist_rooms
                    
                nd = dist[curr] + edge_cost
                if nd < dist[neighbor]:
                    dist[neighbor] = nd
                    prev[neighbor] = curr
                    heapq.heappush(pq, (nd, neighbor))
                    
        if dist[end_room] == float('inf'):
            return self._bfs_room_path(start_room, end_room)
            
        path = []
        curr = end_room
        while curr is not None:
            path.append(curr)
            curr = prev[curr]
        path.reverse()
        return path

    def _compute_dijkstra_target(self, curr_room: int, obj_room: int, final_target: np.ndarray) -> np.ndarray:
        if curr_room == obj_room:
            return final_target
            
        if "large_proc" in self.curriculum_stage:
            path = self._dijkstra_room_path(curr_room, obj_room)
        else:
            path = self._bfs_room_path(curr_room, obj_room)
            
        if len(path) >= 2:
            next_room = path[1]
            for edge in self.door_edges:
                if (edge.room_a == curr_room and edge.room_b == next_room) or \
                   (edge.room_a == next_room and edge.room_b == curr_room):
                    return np.array(edge.center, dtype=np.float32)
        return final_target

    def _get_next_route_target(self) -> np.ndarray:
        curr_room = self.player_current_room
        active_core_room = self._get_room_id_at_position(self.objective)

        # If already in the same room as the active core, grab the core
        if curr_room == active_core_room:
            return self.objective.copy()

        # Target EMP terminal first if available and player has no EMP
        if self.terminal_available and not self.emp_available:
            final_target = self.terminal_point.copy()
        else:
            final_target = self.objective.copy()

        obj_room = self._get_room_id_at_position(final_target)

        if curr_room == obj_room:
            return final_target.copy()

        should_recalc = (
            not hasattr(self, "_cached_target") or
            self._cached_target is None or
            self.step_count % 10 == 0 or
            getattr(self, "_last_curr_room", None) != curr_room or
            getattr(self, "_last_terminal_available", None) != self.terminal_available
        )
        if should_recalc:
            self._last_curr_room = curr_room
            self._last_terminal_available = self.terminal_available
            self._cached_target = self._compute_dijkstra_target(curr_room, obj_room, final_target)

        return self._cached_target.copy()


    @property
    def campaign_stage_name(self) -> str:
        if self.campaign_stage is None:
            return ""
        names = {
            1: "Midtown Route",
            2: "Northline Sequence",
            3: "Canal Sentry",
            4: "Market Patrols",
            5: "Midtown Full",
            6: "Large Easy",
            7: "Large Hub",
            8: "Large Heist",
            9: "Procedural Easy",
            10: "Procedural Cameras",
            11: "Procedural Patrols",
            12: "Procedural Full",
        }
        return names.get(self.campaign_stage, "")

    def camera_is_emp_stunned(self, camera: Camera) -> bool:
        return self.emp_timer > 0 and length(camera.position - self.player) <= EMP_RADIUS

    def _player_seen_by_camera(self, camera: Camera) -> bool:
        if self.camera_is_emp_stunned(camera):
            return False
        delta = self.player - camera.position
        dist = length(delta)
        if dist <= 5.0:
            return True
        # Vision range of camera is 280.0
        if dist > 280.0:
            return False
        direction = normalized(delta)
        cam_angle = camera.base_angle + camera.current_angle
        cam_dir = np.array([math.cos(cam_angle), math.sin(cam_angle)], dtype=np.float32)
        # Half FOV check: camera.fov / 2.0
        if float(np.dot(cam_dir, direction)) < math.cos(camera.fov / 2.0):
            return False
        return self._has_line_of_sight(camera.position, self.player)

    def _direction_clearance_flags(self) -> list[float]:
        flags = []
        directions = [
            np.array([0.0, -1.0], dtype=np.float32),
            np.array([0.70710678, -0.70710678], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.70710678, 0.70710678], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([-0.70710678, 0.70710678], dtype=np.float32),
            np.array([-1.0, 0.0], dtype=np.float32),
            np.array([-0.70710678, -0.70710678], dtype=np.float32)
        ]
        for d in directions:
            check_pos = self.player + d * 18.0
            if self._can_occupy(check_pos):
                flags.append(1.0)
            else:
                flags.append(-1.0)
        return flags

    def _add_reward(self, key: str, amount: float) -> float:
        self.episode_rewards[key] = self.episode_rewards.get(key, 0.0) + float(amount)
        return float(amount)

    def _detect_blocked_doorways(self) -> None:
        if "large_proc" not in self.curriculum_stage:
            return
            
        for edge in self.door_edges:
            edge.is_blocked_by_layout = False
            
            dc, dr = self._world_to_grid(edge.center)
            room_a = self.room_nodes[edge.room_a]
            room_b = self.room_nodes[edge.room_b]
            
            if room_a.col != room_b.col:
                # Vertical boundary (horizontal passage)
                c_l, r_l = self._world_to_grid(np.array([edge.center[0] - 25.0, edge.center[1]], dtype=np.float32))
                c_r, r_r = self._world_to_grid(np.array([edge.center[0] + 25.0, edge.center[1]], dtype=np.float32))
                passable = self.grid_passable[dc, dr] and self.grid_passable[c_l, r_l] and self.grid_passable[c_r, r_r]
            else:
                # Horizontal boundary (vertical passage)
                c_a, r_a = self._world_to_grid(np.array([edge.center[0], edge.center[1] - 25.0], dtype=np.float32))
                c_b, r_b = self._world_to_grid(np.array([edge.center[0], edge.center[1] + 25.0], dtype=np.float32))
                passable = self.grid_passable[dc, dr] and self.grid_passable[c_a, r_a] and self.grid_passable[c_b, r_b]
                
            if not passable:
                edge.is_blocked_by_layout = True
