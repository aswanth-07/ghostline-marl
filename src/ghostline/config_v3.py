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

# Env-v3 stealth economy.
#
# Potential-based shaping (Ng, Harada & Russell 1999) is policy-invariant by
# construction, which is exactly why it is the wrong tool for stealth here: we
# deliberately want to move the optimum away from "race and eat the trace",
# not merely guide search toward the same optimum. Exposure and detection are
# therefore genuine objective terms, while route progress stays potential-based
# because there we do want the original optimum preserved.
#
# A naive potential Phi = -k * trace/TRACE_MAX is also actively wrong: with a
# constant negative potential, gamma*Phi - Phi = Phi*(gamma-1) is *positive*,
# so sitting at maximum trace would pay a small bonus every step.
#
# Budgets for a successful ~370-decision tier-6 run:
#   loud  : -3.7 exposure, -3.6 detection, no quiet bonus
#   quiet : about -0.6 exposure, -0.2 detection, up to +2.8 quiet-data bonus
# against a +32 positive budget (20 extraction + 12 data). The discount supplies
# the speed pressure; these terms supply the stealth pressure.
EXPOSURE_COST_PER_DECISION = 0.010
DETECTION_COST = 0.06
QUIET_DATA_BONUS = 0.35
QUIET_TRACE_CEILING = 45.0

# Vent network. Transit is deliberately slow and the runner is untargetable
# while inside, so a vent is an escape from a sealed route rather than a free
# teleport: entering in sight of an operative is a real tell.
VENT_TRANSIT_SECONDS = 1.15
VENT_PAIRS_PER_TIER = {1: 0, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3}
VENT_MIN_PAIR_DISTANCE_TILES = 8

# Environmental hacking. Charges are shared across device kinds so taking a
# camera down costs the same budget as forcing a sealed door open.
HACK_RANGE = 46.0
HACK_CHARGES_PER_TIER = {1: 0, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3}
HACK_COOLDOWN_SECONDS = 1.6
HACK_CAMERA_DISABLE_SECONDS = 7.0
HACK_DOOR_OVERRIDE_SECONDS = 6.0
HACK_LIGHTS_SECONDS = 9.0
# A darkened room shortens every guard sight cone inside it.
HACK_LIGHTS_VISION_SCALE = 0.55
HACK_DEVICES_PER_TIER = {1: 0, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}

# Upgraded distraction. A lure holds attention at a point instead of emitting
# one pulse, and a crouched throw is quieter but shorter.
DECOY_LURE_SECONDS = 3.4
DECOY_LURE_RADIUS = 168.0
DECOY_CROUCH_THROW_SCALE = 0.62

# Predictive chokepoints. Security seals ahead of the runner's route rather than
# on top of it. Every guarantee from the reactive lock still applies: only
# redundant room-graph edges are eligible, the door warns first, an occupied
# door never closes, and the runner keeps a pulse and a hack override.
CHOKEPOINT_LOOKAHEAD_TILES = 14.0
CHOKEPOINT_MIN_RUNNER_DISTANCE = 72.0
CHOKEPOINT_TEAM_COOLDOWN_SECONDS = 7.5

# Coordinated pincers. Operatives take complementary approach arcs around the
# contact so a strictly slower team can still close from multiple bearings.
PINCER_ARC_RADIANS = 1.15
PINCER_STANDOFF = 132.0

# Non-lethal field tools. Sensors report a crossing; they never damage.
FIELD_SENSOR_ARM_SECONDS = 1.0
FIELD_SENSOR_LIFETIME_SECONDS = 26.0
FIELD_SENSOR_RADIUS = 58.0
FIELD_SENSOR_CHARGES = 1
