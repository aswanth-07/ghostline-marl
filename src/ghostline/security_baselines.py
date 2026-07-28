from __future__ import annotations

import numpy as np

from ghostline.security_types import TargetKind
from ghostline.types_v2 import GuardRole, RadioMessage, SecurityIntent


def _first_valid(mask: np.ndarray, preferred: int) -> int:
    """Select a preferred semantic factor without bypassing its public mask."""

    if 0 <= preferred < len(mask) and mask[preferred]:
        return int(preferred)
    valid = np.flatnonzero(mask)
    return int(valid[0]) if len(valid) else 0


def _first_target_of_kind(
    observation: dict[str, np.ndarray],
    kind: TargetKind,
    fallback: int,
) -> int:
    for index in np.flatnonzero(observation["target_mask"]):
        encoded_kind = int(np.argmax(observation["targets"][index, 3:]))
        if encoded_kind == int(kind):
            return int(index)
    return _first_valid(observation["target_mask"], fallback)


def _coordinated_route_target(
    observation: dict[str, np.ndarray],
    guard_id: int,
    fallback: int,
) -> int:
    routes = [
        int(index)
        for index in np.flatnonzero(observation["target_mask"])
        if int(np.argmax(observation["targets"][index, 3:]))
        == int(TargetKind.ESCAPE_ROUTE)
    ]
    return routes[int(guard_id) % len(routes)] if routes else fallback


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
            # HOLD demotes the base guard state to suspicious. PURSUE preserves
            # an acquired contact while the role-routed ability still aims and
            # fires the suppressor's telegraphed round.
            intent = SecurityIntent.PURSUE
            ability = int(observation["ability_mask"][1] > 0)
        elif role == GuardRole.INTERCEPTOR:
            if observation["intent_mask"][int(SecurityIntent.SEAL)]:
                intent = SecurityIntent.SEAL
                target = _first_target_of_kind(
                    observation,
                    TargetKind.ESCAPE_ROUTE,
                    _first_target_of_kind(observation, TargetKind.DOOR, target),
                )
            else:
                intent = SecurityIntent.INTERCEPT
            message = RadioMessage.REQUEST_INTERCEPT
        elif observation["intent_mask"][int(SecurityIntent.PINCER)]:
            intent = SecurityIntent.PINCER
            target = _coordinated_route_target(
                observation,
                guard_id,
                target,
            )
        else:
            intent = SecurityIntent.PURSUE
    elif quota_met and observation["target_mask"][4]:
        intent = SecurityIntent.INTERCEPT
        target = 4
        message = RadioMessage.REQUEST_INTERCEPT
    elif confidence > 0.02:
        if (
            confidence >= 0.35
            and observation["intent_mask"][int(SecurityIntent.PINCER)]
        ):
            intent = SecurityIntent.PINCER
            target = _coordinated_route_target(
                observation,
                guard_id,
                1,
            )
        elif (
            role == GuardRole.INTERCEPTOR
            and observation["intent_mask"][int(SecurityIntent.INTERCEPT)]
        ):
            intent = SecurityIntent.INTERCEPT
        else:
            intent = SecurityIntent.SEARCH if confidence < 0.35 else SecurityIntent.INVESTIGATE
        if intent != SecurityIntent.PINCER:
            target = 1 if observation["target_mask"][1] else 2
        message = RadioMessage.REQUEST_INTERCEPT if intent == SecurityIntent.INTERCEPT else RadioMessage.SUSPECTED_ROUTE
    elif observation["target_mask"][3]:
        # Facility security knows its own unfinished terminals. Proactive
        # terminal coverage is fair, legible, and avoids idle random patrols.
        intent = SecurityIntent.INVESTIGATE
        target = 3
        message = RadioMessage.REGROUP if role == GuardRole.INTERCEPTOR else RadioMessage.NONE
        # Non-suppressors have one sensor charge. Spend it only after reaching a
        # known objective lane rather than dropping it at spawn.
        close_to_target = float(observation["targets"][3, 2]) < -0.65
        ability = int(close_to_target and observation["ability_mask"][1] > 0)
    return np.asarray(
        (
            _first_valid(observation["intent_mask"], int(intent)),
            _first_valid(observation["target_mask"], target),
            _first_valid(observation["message_mask"], int(message)),
            _first_valid(observation["ability_mask"], ability),
        ),
        dtype=np.int64,
    )
