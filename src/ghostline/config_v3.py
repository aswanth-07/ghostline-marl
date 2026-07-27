from __future__ import annotations

DECOY_THROW_DISTANCE = 96.0
DECOY_NOISE_RADIUS = 210.0
DECOY_LIFETIME_SECONDS = 2.0
DECOY_PULSE_INTERVAL_SECONDS = 0.25

SECURITY_TACTICAL_HZ = 5
SECURITY_TACTICAL_TICKS = 12
SECURITY_DOOR_LOCK_SECONDS = 3.5
SECURITY_DOOR_WARNING_SECONDS = 0.65
SECURITY_DOOR_TEAM_COOLDOWN_SECONDS = 6.0
SECURITY_DOOR_FORCED_OPEN_SECONDS = 5.0

SUPPRESSOR_AIM_SECONDS = 0.70
SUPPRESSOR_PROJECTILE_SPEED = 260.0
SUPPRESSOR_PROJECTILE_LIFETIME_SECONDS = 1.0
SUPPRESSOR_COOLDOWN_SECONDS = 2.4
SUPPRESSOR_MIN_RANGE = 96.0
SUPPRESSOR_MAX_RANGE = 240.0
SUPPRESSOR_PROJECTILE_RADIUS = 4.0

MAX_SECURITY_TARGETS = 8
MAX_RADIO_MESSAGES = 4
MAX_TEAMMATES = 4

# One target-kind code per tactical slot. Extraction and doors previously
# shared a code, so the policy could not tell an exit from a chokepoint.
SECURITY_TARGET_KINDS = 8
# Three relative geometry values plus the target-kind one-hot.
SECURITY_TARGET_FEATURES = 3 + SECURITY_TARGET_KINDS
# Centralized critic state: mission block (with remaining time, alert tier,
# and live link progress), five operative blocks, three door blocks, and an
# explicit operative presence mask.
SECURITY_CENTRAL_STATE_SIZE = 72

# Stealth states. Crouching trades speed for silence and a smaller visual
# profile; dashing stays fast and now carries a real trace cost, so both a quiet
# and a loud route are viable and they fail in different ways.
CROUCH_SPEED_SCALE = 0.52
CROUCH_FOOTSTEP_RADIUS = 46.0
CROUCH_AWARENESS_SCALE = 0.55
CROUCH_TRACE_DECAY_BONUS = 4.4
WALK_FOOTSTEP_RADIUS = 118.0
DASH_TRACE_COST_PER_SECOND = 7.5
COVER_TRACE_DECAY_BONUS = 2.6
