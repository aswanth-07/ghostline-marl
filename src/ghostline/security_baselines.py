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
            # Interceptors turn current contact into route denial. INTERCEPT
            # still moves toward the public contact target, but also asks the
            # simulation to lock the nearest graph-redundant security door.
            intent = SecurityIntent.INTERCEPT
            message = RadioMessage.REQUEST_INTERCEPT
        else:
            intent = SecurityIntent.PURSUE
    elif quota_met and observation["target_mask"][4]:
        intent = SecurityIntent.INTERCEPT
        target = 4
        message = RadioMessage.REQUEST_INTERCEPT
    elif confidence > 0.02:
        intent = (
            SecurityIntent.INTERCEPT
            if role == GuardRole.INTERCEPTOR and observation["intent_mask"][int(SecurityIntent.INTERCEPT)]
            else SecurityIntent.SEARCH
            if confidence < 0.35
            else SecurityIntent.INVESTIGATE
        )
        target = 1 if observation["target_mask"][1] else 2
        message = RadioMessage.REQUEST_INTERCEPT if intent == SecurityIntent.INTERCEPT else RadioMessage.SUSPECTED_ROUTE
    elif observation["target_mask"][3]:
        # Facility security knows its own unfinished terminals. Proactive
        # terminal coverage is fair, legible, and avoids idle random patrols.
        intent = SecurityIntent.INVESTIGATE
        target = 3
        message = RadioMessage.REGROUP if role == GuardRole.INTERCEPTOR else RadioMessage.NONE
    return np.asarray(
        (
            _first_valid(observation["intent_mask"], int(intent)),
            _first_valid(observation["target_mask"], target),
            _first_valid(observation["message_mask"], int(message)),
            _first_valid(observation["ability_mask"], ability),
        ),
        dtype=np.int64,
    )
