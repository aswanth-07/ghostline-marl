from __future__ import annotations

from dataclasses import dataclass

TILE_SIZE = 32
SIM_HZ = 60
POLICY_REPEAT = 6
PLAYER_RADIUS = 9.0
PLAYER_SPEED = 126.0
PLAYER_PERCEPTION_DISTANCE = 390.0
PLAYER_GUARD_AUDIBLE_DISTANCE = 220.0
PLAYER_FOOTSTEP_AUDIBLE_DISTANCE = 145.0
DASH_SPEED = 226.0
DASH_ENERGY_MAX = 100.0
DASH_DRAIN_PER_SECOND = 34.0
DASH_RECHARGE_PER_SECOND = 23.0
PULSE_RADIUS = 150.0
PULSE_DISABLE_SECONDS = 5.0
HACK_RADIUS = 40.0
DETECTION_GRACE_SECONDS = 0.65
DAMAGE_INVULNERABILITY_SECONDS = 1.75
GUARD_STRIKE_WINDUP_SECONDS = 0.42
DRONE_STRIKE_WINDUP_SECONDS = 0.55
CAMERA_VISION_DISTANCE = 220.0
CAMERA_VISION_COSINE = 0.72
CAMERA_SWEEP_RADIANS = 0.85
GUARD_VISION_BASE_DISTANCE = 205.0
GUARD_VISION_DISTANCE_PER_ALERT = 18.0
GUARD_VISION_COSINE = 0.62
# Standard, Interceptor, and Elite non-chase movement multipliers.
GUARD_GRADE_SPEED_MULTIPLIERS = (1.04, 1.10, 1.16)
# Chase speeds stay readable by grade while the elite operative now matches
# 99% of the runner's undashed speed. Dash remains the decisive escape tool.
GUARD_CHASE_SPEED_RATIOS = (0.95, 0.97, 0.99)
# Standard guards deliberately hold their scan longer; higher grades trade
# that readability for quicker patrol cadence and more persistent searches.
GUARD_PATROL_DWELL_SECONDS = (0.78, 0.52, 0.36)
GUARD_SEARCH_DURATION_MULTIPLIERS = (1.00, 1.18, 1.36)
TRACE_MAX = 100.0
LOCAL_GRID_SIZE = 15
MAX_TARGETS = 5
MAX_ENTITIES = 12
RAY_COUNT = 24


@dataclass(frozen=True)
class TierSpec:
    number: int
    name: str
    room_columns: int
    room_rows: int
    terminal_count: int
    quota: int
    cameras: int
    guards: int
    response_drones: bool
    drone_trace_threshold: float
    pulse_charges: int
    mission_seconds: int
    loop_chance: float


TIERS: dict[int, TierSpec] = {
    1: TierSpec(1, "Orientation", 3, 2, 3, 3, 0, 0, False, 101.0, 0, 115, 0.10),
    2: TierSpec(2, "Surveillance", 3, 2, 4, 4, 2, 0, False, 101.0, 1, 135, 0.24),
    3: TierSpec(3, "Patrol", 4, 2, 4, 5, 2, 3, False, 101.0, 2, 160, 0.32),
    4: TierSpec(4, "Countermeasure", 4, 3, 5, 6, 3, 3, False, 101.0, 3, 180, 0.38),
    5: TierSpec(5, "Lockdown", 5, 3, 5, 7, 4, 3, True, 100.0, 4, 205, 0.46),
    6: TierSpec(6, "Ghostline", 5, 3, 5, 8, 5, 5, True, 72.0, 3, 225, 0.52),
}
MAX_PULSE_CHARGES = max(spec.pulse_charges for spec in TIERS.values())

ROOM_WIDTH_TILES = 11
ROOM_HEIGHT_TILES = 9

ROLE_COLORS = {
    "office": (34, 47, 58),
    "lounge": (48, 40, 55),
    "lab": (31, 49, 52),
    "server": (28, 37, 55),
    "security": (53, 37, 39),
    "vault": (49, 45, 31),
    "utility": (39, 44, 45),
    "corridor": (34, 40, 47),
    "extraction": (30, 49, 43),
}
