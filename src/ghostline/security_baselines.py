from __future__ import annotations

import numpy as np

from ghostline.types_v3 import GuardRole, RadioMessage, SecurityIntent


def _first_valid(mask: np.ndarray, preferred: int) -> int:
    """Select a preferred semantic factor without bypassing its public mask."""

    if 0 <= preferred < len(mask) and mask[preferred]:
        return int(preferred)
    valid = np.flatnonzero(mask)
    return int(valid[0]) if len(valid) else 0


def tactical_security_action(
    observation: dict[str, np.ndarray],
    guard_id: int,
) -> np.ndarray:
    """Deterministic observation-only baseline used by play and evaluation.

    Keeping this policy outside the presentation controller prevents the
    evaluation command from quietly comparing learned security to a weaker
    heuristic than the one players encounter in Adaptive Contracts.
    """

    role = GuardRole(int(np.argmax(observation["ego"][:3])))
    visible = observation["runner"][5] > 0.0
    confidence = (float(observation["runner"][7]) + 1.0) * 0.5
    quota_met = observation["runner"][11] > 0.0
    intent = SecurityIntent.PATROL
    target = 0
    message = RadioMessage.NONE
    ability = 0
    if visible:
        target = 1
        message = RadioMessage.SIGHTING
        if role == GuardRole.SUPPRESSOR:
            intent = SecurityIntent.HOLD
            ability = int(observation["ability_mask"][1] > 0)
        elif role == GuardRole.INTERCEPTOR:
            intent = SecurityIntent.FLANK_LEFT if guard_id % 2 == 0 else SecurityIntent.FLANK_RIGHT
            target = 6 if intent == SecurityIntent.FLANK_LEFT else 7
        else:
            intent = SecurityIntent.PURSUE
    elif quota_met and observation["target_mask"][4]:
        intent = SecurityIntent.INTERCEPT
        target = 4
        message = RadioMessage.REQUEST_INTERCEPT
    elif confidence > 0.02:
        intent = SecurityIntent.SEARCH if confidence < 0.35 else SecurityIntent.INVESTIGATE
        target = 1 if observation["target_mask"][1] else 2
        message = RadioMessage.SUSPECTED_ROUTE
    return np.asarray(
        (
            _first_valid(observation["intent_mask"], int(intent)),
            _first_valid(observation["target_mask"], target),
            _first_valid(observation["message_mask"], int(message)),
            _first_valid(observation["ability_mask"], ability),
        ),
        dtype=np.int64,
    )
