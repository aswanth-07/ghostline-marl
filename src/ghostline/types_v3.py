from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class ContractDirective(IntEnum):
    STANDARD = 0
    GHOST = 1
    SPEED = 2
    GREED = 3

    @classmethod
    def parse(cls, value: "ContractDirective | str | int") -> "ContractDirective":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            names = {
                "standard": cls.STANDARD,
                "ghost": cls.GHOST,
                "speed": cls.SPEED,
                "greed": cls.GREED,
            }
            if normalized not in names:
                raise ValueError(f"unknown contract directive: {value}")
            return names[normalized]
        return cls(int(value))


class GuardRole(IntEnum):
    PATROL = 0
    INTERCEPTOR = 1
    SUPPRESSOR = 2


class SecurityIntent(IntEnum):
    PATROL = 0
    INVESTIGATE = 1
    SEARCH = 2
    PURSUE = 3
    INTERCEPT = 4
    FLANK_LEFT = 5
    FLANK_RIGHT = 6
    HOLD = 7


class RadioMessage(IntEnum):
    NONE = 0
    SIGHTING = 1
    SUSPECTED_ROUTE = 2
    REQUEST_INTERCEPT = 3
    REGROUP = 4


@dataclass(frozen=True)
class RunnerActionV3:
    """Env-v3 action: 9 movement x dash x pulse x decoy."""

    move: int = 0
    dash: bool = False
    pulse: bool = False
    decoy: bool = False

    @classmethod
    def decode(cls, value: int) -> "RunnerActionV3":
        value = int(np.clip(value, 0, 71))
        return cls(
            move=value % 9,
            dash=bool((value // 9) % 2),
            pulse=bool((value // 18) % 2),
            decoy=bool((value // 36) % 2),
        )

    def encode(self) -> int:
        return int(self.move) + 9 * int(self.dash) + 18 * int(self.pulse) + 36 * int(self.decoy)


@dataclass(frozen=True)
class SecurityOrder:
    intent: SecurityIntent = SecurityIntent.PATROL
    target: np.ndarray | None = None
    message: RadioMessage = RadioMessage.NONE
    use_ability: bool = False


@dataclass
class OperativeState:
    role: GuardRole = GuardRole.PATROL
    current_order: SecurityOrder = field(default_factory=SecurityOrder)
    heard_position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    heard_confidence: float = 0.0
    weapon_cooldown: float = 0.0
    aim_progress: float = 0.0
    aim_target: np.ndarray | None = None
    radio_assists: int = 0


@dataclass
class SecurityDoor:
    door_id: int
    tile: tuple[int, int]
    warning_remaining: float = 0.0
    lock_remaining: float = 0.0
    forced_open_remaining: float = 0.0

    @property
    def locked(self) -> bool:
        return self.lock_remaining > 0.0 and self.forced_open_remaining <= 0.0


@dataclass
class Decoy:
    decoy_id: int
    position: np.ndarray
    lifetime: float = 2.0
    pulse_cooldown: float = 0.0


@dataclass
class ShockProjectile:
    projectile_id: int
    position: np.ndarray
    velocity: np.ndarray
    source_guard_id: int
    lifetime: float = 1.0


@dataclass(frozen=True)
class RadioTransmission:
    sender_id: int
    message: RadioMessage
    position: np.ndarray
    tick: int
