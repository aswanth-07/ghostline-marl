"""Dependency-light semantic types shared by security play and training."""

from __future__ import annotations

from enum import IntEnum

from ghostline.config_v2 import SECURITY_TARGET_KINDS


class TargetKind(IntEnum):
    """One code per tactical target slot exposed to an operative policy."""

    PATROL = 0
    CONTACT = 1
    HEARD = 2
    TERMINAL = 3
    EXTRACTION = 4
    DOOR = 5
    FLANK_LEFT = 6
    FLANK_RIGHT = 7
    ESCAPE_ROUTE = 8


assert len(TargetKind) == SECURITY_TARGET_KINDS
