"""Semantic state and actions for the in-development multi-agent v2 contract."""

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
    # Coordinated closure. PINCER sends operatives to policy-selected public
    # doorway cutoffs so a slower team can cover multiple escape routes without
    # receiving the runner's hidden objective or heading.
    PINCER = 8
    SEAL = 9


class RadioMessage(IntEnum):
    NONE = 0
    SIGHTING = 1
    SUSPECTED_ROUTE = 2
    REQUEST_INTERCEPT = 3
    REGROUP = 4


# 9 movement x dash x pulse x decoy x crouch x interact.
#
# ``interact`` is deliberately context-sensitive rather than one bit per verb.
# Entering a vent and hacking a device are never both legal on the same tile, so
# a single code keeps the action space at 288 instead of 576 and keeps the mask
# doing the disambiguation the simulation already has to do anyway.
RUNNER_ACTION_COUNT_V2 = 288


@dataclass(frozen=True)
class RunnerActionV2:
    """V2 action: 9 movement x dash x pulse x decoy x crouch x interact."""

    move: int = 0
    dash: bool = False
    pulse: bool = False
    decoy: bool = False
    crouch: bool = False
    interact: bool = False

    @classmethod
    def decode(cls, value: int) -> "RunnerActionV2":
        value = max(0, min(int(value), RUNNER_ACTION_COUNT_V2 - 1))
        return cls(
            move=value % 9,
            dash=bool((value // 9) % 2),
            pulse=bool((value // 18) % 2),
            decoy=bool((value // 36) % 2),
            crouch=bool((value // 72) % 2),
            interact=bool((value // 144) % 2),
        )

    def encode(self) -> int:
        return (
            int(self.move)
            + 9 * int(self.dash)
            + 18 * int(self.pulse)
            + 36 * int(self.decoy)
            + 72 * int(self.crouch)
            + 144 * int(self.interact)
        )


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
    lure_remaining: float = 0.0
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


@dataclass
class Vent:
    """A maintenance duct the runner can use and operatives cannot.

    Vents are the counterplay to a sealed chokepoint: they are visible to
    everyone, so using one in sight is a real tell, but only the runner fits.
    """

    vent_id: int
    tile: tuple[int, int]
    exit_tile: tuple[int, int]
    exit_position: np.ndarray


@dataclass
class HackableDevice:
    """A facility system the runner can take over in the field."""

    device_id: int
    kind: str  # "camera" | "door" | "lights"
    tile: tuple[int, int]
    position: np.ndarray
    target_id: int = -1
    # Door panels bind to one authored security-door tile.  ``target_id`` is
    # retained for camera and room-light systems.
    target_tile: tuple[int, int] | None = None
    hacked_for: float = 0.0
    cooldown: float = 0.0


@dataclass
class FieldSensor:
    """A non-lethal operative deployable that reports a crossing."""

    sensor_id: int
    owner_id: int
    position: np.ndarray
    armed_in: float
    lifetime: float
    triggered: bool = False
