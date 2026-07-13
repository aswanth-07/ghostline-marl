from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from ghostline.config import CAMERA_SWEEP_RADIANS


class Tile(IntEnum):
    FLOOR = 0
    WALL = 1
    DOOR = 2


class GuardMode(IntEnum):
    PATROL = 0
    SUSPICIOUS = 1
    INVESTIGATE = 2
    SEARCH = 3
    CHASE = 4
    RETURN = 5


class GuardGrade(IntEnum):
    STANDARD = 0
    INTERCEPTOR = 1
    ELITE = 2


@dataclass(frozen=True)
class Action:
    move: int = 0
    dash: bool = False
    pulse: bool = False

    @classmethod
    def decode(cls, value: int) -> "Action":
        value = int(np.clip(value, 0, 35))
        return cls(move=value % 9, dash=bool((value // 9) % 2), pulse=bool(value // 18))

    def encode(self) -> int:
        return int(self.move) + 9 * int(self.dash) + 18 * int(self.pulse)


@dataclass(frozen=True)
class Room:
    room_id: int
    column: int
    row: int
    role: str
    x: int
    y: int
    width: int
    height: int
    variant: int

    @property
    def center_tile(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2


@dataclass(frozen=True)
class Door:
    tile: tuple[int, int]
    room_a: int
    room_b: int


@dataclass
class Prop:
    kind: str
    tile_x: int
    tile_y: int
    width: int = 1
    height: int = 1
    blocking: bool = True


@dataclass
class Terminal:
    terminal_id: int
    position: np.ndarray
    value: int
    hack_seconds: float
    progress: float = 0.0
    completed: bool = False


@dataclass
class Camera:
    camera_id: int
    position: np.ndarray
    angle: float
    base_angle: float
    sweep: float = CAMERA_SWEEP_RADIANS
    speed: float = 0.72
    disabled_for: float = 0.0
    detected: bool = False
    awareness: float = 0.0


@dataclass
class Guard:
    guard_id: int
    position: np.ndarray
    patrol: list[np.ndarray]
    grade: GuardGrade = GuardGrade.STANDARD
    patrol_index: int = 0
    facing: float = 0.0
    mode: GuardMode = GuardMode.PATROL
    mode_seconds: float = 0.0
    last_known: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    hit_cooldown: float = 0.0
    radio_jammed_for: float = 0.0
    awareness: float = 0.0
    stuck_seconds: float = 0.0
    attack_windup: float = 0.0
    patrol_pause_seconds: float = 0.0
    stimulus: str = "patrol"


@dataclass
class Drone:
    drone_id: int
    position: np.ndarray
    facing: float = 0.0
    disabled_for: float = 0.0
    hit_cooldown: float = 0.0
    attack_windup: float = 0.0


@dataclass
class SecurityIntel:
    """Player-earned knowledge about one security actor.

    The simulation owns this cache so every controller and presentation sees
    the same information.  Positions and motion are snapshots from the last
    unobstructed sighting; they are never updated through walls.
    """

    kind: str
    entity_id: int
    position: np.ndarray
    velocity: np.ndarray
    facing: float
    alert: float
    last_seen_tick: int
    grade: GuardGrade = GuardGrade.STANDARD


@dataclass(frozen=True)
class SimEvent:
    kind: str
    position: tuple[float, float]
    value: float = 0.0


@dataclass
class GeneratedLevel:
    seed: int
    tier: int
    grid: np.ndarray
    rooms: list[Room]
    doors: list[Door]
    props: list[Prop]
    terminals: list[Terminal]
    cameras: list[Camera]
    guards: list[Guard]
    spawn: np.ndarray
    extraction: np.ndarray
    quota: int
    mission_seconds: int
    pulse_charges: int
    response_drones: bool
    drone_trace_threshold: float
    adjacency: dict[int, list[int]]

    @property
    def world_width(self) -> float:
        from ghostline.config import TILE_SIZE

        return float(self.grid.shape[1] * TILE_SIZE)

    @property
    def world_height(self) -> float:
        from ghostline.config import TILE_SIZE

        return float(self.grid.shape[0] * TILE_SIZE)
