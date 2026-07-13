from __future__ import annotations

from dataclasses import dataclass

WORLD_WIDTH = 1200.0
WORLD_HEIGHT = 760.0
MAX_STEPS = 2500
PLAYER_RADIUS = 15.0
PLAYER_ACCELERATION = 0.82
PLAYER_FRICTION = 0.86
PLAYER_MAX_SPEED = 7.4
DASH_SPEED = 15.5
DASH_COOLDOWN_STEPS = 76
DASH_ENERGY_MAX = 100.0
DASH_ENERGY_DRAIN = 1.85
DASH_ENERGY_RECHARGE = 0.82
BOOST_ACCELERATION_MULTIPLIER = 1.65
BOOST_MAX_SPEED = 11.8
DRONE_RADIUS = 15.0
CORE_RADIUS = 16.0
PORTAL_RADIUS = 31.0
RAY_LENGTH = 170.0
RAY_COUNT = 16
EMP_DURATION_STEPS = 130
EMP_RADIUS = 245.0
HUNTER_LOCK_RANGE = 330.0
HUNTER_RANGE_PER_CORE = 52.0
HUNTER_ALARM_RANGE_BONUS = 130.0
HUNTER_VISION_COSINE = 0.68
PATROL_LOCK_RANGE = 205.0
PATROL_LOCK_STEPS = 70
PATROL_LOST_LOCK_DECAY = 5
HUNTER_LOCK_STEPS = 155
HUNTER_LOST_LOCK_DECAY = 6
SENTRY_LOCK_RANGE = 245.0
SENTRY_TURN_RATE = 0.026
SENTRY_PROJECTILE_COOLDOWN = 72
SENTRY_PROJECTILE_SPEED = 8.2
SENTRY_PROJECTILE_RADIUS = 7.0
SENTRY_PROJECTILE_LIFETIME = 120
SENTRY_PROJECTILE_HIT_GAIN = 10.0
HEAT_MAX = 100.0
HEAT_EXPOSURE_GAIN = 0.34
HEAT_DASH_NOISE_RANGE = 230.0
HEAT_DASH_GAIN = 4.5
HEAT_WALL_GAIN = 2.0
HEAT_DRONE_HIT_GAIN = 12.0
HEAT_EMP_DROP = 28.0
HEAT_DECAY = 0.09
HEAT_EMP_DECAY = 0.22
CONE_EXIT_REWARD = 0.04
CONE_EXIT_REWARD_COOLDOWN = 80
STALL_GRACE_STEPS = 400
STALL_TERMINATE_STEPS = 800


# Shock Tile & Coolant Hazards constants
SHOCK_CYCLE_SAFE = 90
SHOCK_CYCLE_WARMING = 60
SHOCK_CYCLE_ACTIVE = 60
COOLANT_FRICTION_MULTIPLIER = 0.52
COOLANT_TURN_RATE_MULTIPLIER = 0.65

# Authored room obstacle presets (relative to room width/height)
ROOM_PRESETS = {
    "VAULT": [
        # Preset 1: vault_ring (central U-shape of server blocks, bottom open)
        {
            "blocks": [
                (0.25, 0.2, 0.08, 0.6),
                (0.67, 0.2, 0.08, 0.6),
                (0.33, 0.2, 0.34, 0.1)
            ],
            "cameras": [(0.5, 0.1)],
            "sentries": [(0.5, 0.5)],
            "loot": [(0.45, 0.45), (0.55, 0.45)]
        },
        # Preset 2: vault_brackets (bracket guard walls)
        {
            "blocks": [
                (0.2, 0.2, 0.1, 0.6),
                (0.7, 0.2, 0.1, 0.6)
            ],
            "cameras": [(0.15, 0.5), (0.85, 0.5)],
            "sentries": [(0.3, 0.5), (0.7, 0.5)],
            "loot": [(0.5, 0.3), (0.5, 0.7)]
        }
    ],
    "PLAZA": [
        # Preset 1: four_pillars
        {
            "blocks": [
                (0.25, 0.25, 0.12, 0.12),
                (0.63, 0.25, 0.12, 0.12),
                (0.25, 0.63, 0.12, 0.12),
                (0.63, 0.63, 0.12, 0.12)
            ],
            "cameras": [(0.5, 0.5)],
            "sentries": [],
            "loot": [(0.5, 0.2), (0.5, 0.8)]
        },
        # Preset 2: open_cross (moved blocks away from walls)
        {
            "blocks": [
                (0.4, 0.2, 0.2, 0.1),
                (0.4, 0.7, 0.2, 0.1),
                (0.2, 0.4, 0.1, 0.2),
                (0.7, 0.4, 0.1, 0.2)
            ],
            "cameras": [(0.1, 0.1), (0.9, 0.9)],
            "sentries": [],
            "loot": [(0.5, 0.5)]
        }
    ],
    "SECURITY": [
        # Preset 1: server_rows_vertical (wider clearance from top/bottom walls)
        {
            "blocks": [
                (0.25, 0.2, 0.08, 0.6),
                (0.5, 0.2, 0.08, 0.6),
                (0.75, 0.2, 0.08, 0.6)
            ],
            "cameras": [(0.15, 0.1), (0.85, 0.9)],
            "sentries": [(0.38, 0.5), (0.62, 0.5)],
            "loot": [(0.15, 0.5), (0.85, 0.5)]
        },
        # Preset 2: server_rows_horizontal (middle horizontal row removed)
        {
            "blocks": [
                (0.15, 0.25, 0.7, 0.08),
                (0.15, 0.67, 0.7, 0.08)
            ],
            "cameras": [(0.5, 0.15), (0.5, 0.85)],
            "sentries": [(0.5, 0.46), (0.5, 0.54)],
            "loot": [(0.5, 0.46), (0.5, 0.54)]
        }
    ],
    "MARKET": [
        # Preset 1: dense_stalls (further from walls)
        {
            "blocks": [
                (0.25, 0.25, 0.12, 0.12),
                (0.44, 0.25, 0.12, 0.12),
                (0.63, 0.25, 0.12, 0.12),
                (0.25, 0.63, 0.12, 0.12),
                (0.44, 0.63, 0.12, 0.12),
                (0.63, 0.63, 0.12, 0.12)
            ],
            "cameras": [(0.5, 0.5)],
            "sentries": [],
            "loot": [(0.3, 0.5), (0.7, 0.5)]
        },
        # Preset 2: zigzag_cover (further from walls)
        {
            "blocks": [
                (0.2, 0.2, 0.35, 0.1),
                (0.45, 0.45, 0.35, 0.1),
                (0.2, 0.7, 0.35, 0.1)
            ],
            "cameras": [(0.1, 0.5), (0.9, 0.5)],
            "sentries": [],
            "loot": [(0.8, 0.2), (0.2, 0.6)]
        }
    ],
    "EXTRACT": [
        # Preset 1: wide_lane
        {
            "blocks": [
                (0.1, 0.3, 0.2, 0.4),
                (0.7, 0.3, 0.2, 0.4)
            ],
            "cameras": [(0.5, 0.15)],
            "sentries": [(0.5, 0.8)],
            "loot": [(0.5, 0.5)]
        },
        # Preset 2: gauntlet_approach (widened gap)
        {
            "blocks": [
                (0.2, 0.1, 0.1, 0.4),
                (0.7, 0.5, 0.1, 0.4)
            ],
            "cameras": [(0.15, 0.5)],
            "sentries": [(0.85, 0.5)],
            "loot": [(0.45, 0.8)]
        }
    ]
}


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height


@dataclass(frozen=True)
class ArenaTemplate:
    name: str
    blocks: tuple[Rect, ...]
    core_slots: tuple[tuple[float, float], ...]
    spawn_point: tuple[float, float]
    extraction_point: tuple[float, float]
    terminal_point: tuple[float, float]
    drone_routes: tuple[tuple[tuple[float, float], ...], ...]
    camera_positions: tuple[tuple[float, float], ...] = ()
    gate_configs: tuple[tuple[tuple[float, float, float, float], str], ...] = ()
    rooms: tuple[Rect, ...] = ()


CITY_BLOCKS = (
    Rect(170.0, 105.0, 180.0, 100.0),
    Rect(485.0, 75.0, 170.0, 125.0),
    Rect(815.0, 120.0, 180.0, 105.0),
    Rect(315.0, 305.0, 205.0, 115.0),
    Rect(700.0, 305.0, 175.0, 120.0),
    Rect(115.0, 555.0, 190.0, 95.0),
    Rect(455.0, 570.0, 200.0, 95.0),
    Rect(885.0, 530.0, 190.0, 105.0),
)

CORE_SLOTS = (
    (405.0, 145.0),
    (740.0, 175.0),
    (1050.0, 290.0),
    (805.0, 500.0),
    (355.0, 515.0),
    (125.0, 365.0),
)

SPAWN_POINT = (72.0, 690.0)
EXTRACTION_POINT = (1125.0, 690.0)
TERMINAL_POINT = (585.0, 492.0)

DRONE_ROUTES = (
    ((395.0, 245.0), (630.0, 245.0), (630.0, 485.0), (395.0, 485.0)),
    ((940.0, 270.0), (1100.0, 270.0), (1100.0, 465.0), (940.0, 465.0)),
    ((65.0, 255.0), (260.0, 255.0), (260.0, 475.0), (65.0, 475.0)),
)

DRONE_ROLES = ("patrol", "hunter", "patrol")

SMALL_TEMPLATES = (
    ArenaTemplate(
        name="midtown",
        blocks=CITY_BLOCKS,
        core_slots=CORE_SLOTS,
        spawn_point=SPAWN_POINT,
        extraction_point=EXTRACTION_POINT,
        terminal_point=TERMINAL_POINT,
        drone_routes=DRONE_ROUTES,
    ),
    ArenaTemplate(
        name="northline",
        blocks=(
            Rect(130.0, 92.0, 190.0, 112.0),
            Rect(415.0, 105.0, 155.0, 145.0),
            Rect(735.0, 80.0, 225.0, 115.0),
            Rect(1015.0, 160.0, 120.0, 190.0),
            Rect(225.0, 292.0, 165.0, 135.0),
            Rect(520.0, 285.0, 210.0, 105.0),
            Rect(850.0, 335.0, 180.0, 100.0),
            Rect(105.0, 520.0, 170.0, 110.0),
            Rect(385.0, 575.0, 190.0, 90.0),
            Rect(720.0, 555.0, 210.0, 100.0),
        ),
        core_slots=(
            (365.0, 165.0),
            (675.0, 175.0),
            (1075.0, 395.0),
            (865.0, 505.0),
            (455.0, 515.0),
            (140.0, 365.0),
            (610.0, 475.0),
        ),
        spawn_point=SPAWN_POINT,
        extraction_point=EXTRACTION_POINT,
        terminal_point=(600.0, 525.0),
        drone_routes=(
            ((330.0, 235.0), (650.0, 235.0), (650.0, 455.0), (330.0, 455.0)),
            ((900.0, 240.0), (1110.0, 240.0), (1110.0, 500.0), (900.0, 500.0)),
            ((80.0, 290.0), (285.0, 290.0), (285.0, 495.0), (80.0, 495.0)),
        ),
    ),
    ArenaTemplate(
        name="canal",
        blocks=(
            Rect(210.0, 85.0, 150.0, 150.0),
            Rect(535.0, 70.0, 180.0, 110.0),
            Rect(900.0, 90.0, 160.0, 150.0),
            Rect(90.0, 275.0, 215.0, 110.0),
            Rect(430.0, 300.0, 160.0, 150.0),
            Rect(735.0, 280.0, 210.0, 125.0),
            Rect(235.0, 530.0, 180.0, 95.0),
            Rect(575.0, 555.0, 145.0, 125.0),
            Rect(925.0, 520.0, 185.0, 95.0),
        ),
        core_slots=(
            (425.0, 135.0),
            (780.0, 145.0),
            (1080.0, 315.0),
            (680.0, 475.0),
            (335.0, 465.0),
            (160.0, 420.0),
            (1000.0, 680.0),
        ),
        spawn_point=SPAWN_POINT,
        extraction_point=EXTRACTION_POINT,
        terminal_point=(620.0, 495.0),
        drone_routes=(
            ((375.0, 230.0), (705.0, 230.0), (705.0, 505.0), (375.0, 505.0)),
            ((850.0, 230.0), (1120.0, 230.0), (1120.0, 475.0), (850.0, 475.0)),
            ((70.0, 225.0), (350.0, 225.0), (350.0, 495.0), (70.0, 495.0)),
        ),
    ),
    ArenaTemplate(
        name="market",
        blocks=(
            Rect(135.0, 130.0, 155.0, 95.0),
            Rect(390.0, 85.0, 145.0, 120.0),
            Rect(640.0, 125.0, 185.0, 95.0),
            Rect(915.0, 95.0, 155.0, 135.0),
            Rect(200.0, 335.0, 150.0, 120.0),
            Rect(480.0, 295.0, 125.0, 95.0),
            Rect(665.0, 330.0, 140.0, 130.0),
            Rect(955.0, 330.0, 150.0, 105.0),
            Rect(135.0, 565.0, 145.0, 90.0),
            Rect(420.0, 535.0, 175.0, 115.0),
            Rect(760.0, 560.0, 170.0, 90.0),
        ),
        core_slots=(
            (330.0, 190.0),
            (610.0, 215.0),
            (1070.0, 255.0),
            (870.0, 500.0),
            (355.0, 505.0),
            (105.0, 385.0),
            (685.0, 520.0),
        ),
        spawn_point=SPAWN_POINT,
        extraction_point=EXTRACTION_POINT,
        terminal_point=(565.0, 465.0),
        drone_routes=(
            ((320.0, 255.0), (640.0, 255.0), (640.0, 490.0), (320.0, 490.0)),
            ((840.0, 245.0), (1130.0, 245.0), (1130.0, 500.0), (840.0, 500.0)),
            ((70.0, 255.0), (300.0, 255.0), (300.0, 525.0), (70.0, 525.0)),
        ),
    ),
)

ARENA_TEMPLATES = SMALL_TEMPLATES

LARGE_TEMPLATES = (
    ArenaTemplate(
        name="large_district_alpha",
        blocks=(
            Rect(150.0, 120.0, 160.0, 140.0),
            Rect(400.0, 80.0, 180.0, 120.0),
            Rect(150.0, 350.0, 180.0, 120.0),
            Rect(455.0, 650.0, 120.0, 120.0),
            Rect(180.0, 600.0, 140.0, 80.0),
            Rect(780.0, 100.0, 200.0, 120.0),
            Rect(1080.0, 120.0, 200.0, 120.0),
            Rect(780.0, 320.0, 180.0, 120.0),
            Rect(1000.0, 580.0, 220.0, 140.0),
            Rect(750.0, 580.0, 150.0, 160.0),
        ),
        core_slots=(
            (300.0, 250.0),
            (950.0, 250.0),
            (1300.0, 700.0),
            (350.0, 520.0),
            (900.0, 500.0),
        ),
        spawn_point=(100.0, 850.0),
        extraction_point=(1400.0, 850.0),
        terminal_point=(900.0, 480.0),
        drone_routes=(
            ((350.0, 270.0), (600.0, 270.0), (600.0, 480.0), (350.0, 480.0)),
            ((850.0, 270.0), (1050.0, 270.0), (1050.0, 480.0), (850.0, 480.0)),
            ((100.0, 250.0), (250.0, 250.0), (250.0, 480.0), (100.0, 480.0)),
        ),
        camera_positions=(
            (750.0, 270.0),
            (1050.0, 270.0),
            (1150.0, 500.0),
        ),
        gate_configs=(),
    ),
    ArenaTemplate(
        name="large_district_beta",
        blocks=(
            Rect(120.0, 150.0, 160.0, 120.0),
            Rect(380.0, 110.0, 160.0, 140.0),
            Rect(180.0, 380.0, 150.0, 130.0),
            Rect(480.0, 680.0, 130.0, 110.0),
            Rect(220.0, 620.0, 120.0, 90.0),
            Rect(800.0, 80.0, 220.0, 110.0),
            Rect(1100.0, 140.0, 180.0, 130.0),
            Rect(820.0, 300.0, 160.0, 140.0),
            Rect(980.0, 600.0, 240.0, 120.0),
            Rect(720.0, 600.0, 160.0, 150.0),
        ),
        core_slots=(
            (280.0, 280.0),
            (980.0, 220.0),
            (1280.0, 720.0),
            (320.0, 550.0),
            (880.0, 480.0),
        ),
        spawn_point=(120.0, 820.0),
        extraction_point=(1380.0, 820.0),
        terminal_point=(920.0, 460.0),
        drone_routes=(
            ((330.0, 290.0), (580.0, 290.0), (580.0, 460.0), (330.0, 460.0)),
            ((880.0, 290.0), (1080.0, 290.0), (1080.0, 460.0), (880.0, 460.0)),
            ((120.0, 280.0), (270.0, 280.0), (270.0, 460.0), (120.0, 460.0)),
        ),
        camera_positions=(
            (730.0, 290.0),
            (1020.0, 290.0),
            (1120.0, 480.0),
        ),
        gate_configs=(),
    ),
)
