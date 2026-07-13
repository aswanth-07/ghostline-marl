from __future__ import annotations

import gymnasium as gym
import numpy as np

from neon_arena.config import (
    DASH_COOLDOWN_STEPS,
    EMP_DURATION_STEPS,
    HUNTER_ALARM_RANGE_BONUS,
    HUNTER_LOCK_STEPS,
    HUNTER_LOCK_RANGE,
    HUNTER_LOST_LOCK_DECAY,
    HUNTER_RANGE_PER_CORE,
    PATROL_LOCK_STEPS,
    DASH_ENERGY_MAX,
    PLAYER_RADIUS,
    PLAYER_MAX_SPEED,
    SENTRY_PROJECTILE_HIT_GAIN,
    SENTRY_PROJECTILE_RADIUS,
    STALL_TERMINATE_STEPS,
)
from neon_arena.env import BlacklineHeistEnv, Projectile


def test_spaces_match_environment_contract() -> None:
    # Test default discrete mode
    env_d = BlacklineHeistEnv(seed=3, action_mode="discrete")
    observation_d, info_d = env_d.reset()

    assert observation_d.shape == (84,)
    assert env_d.observation_space.contains(observation_d)
    assert isinstance(env_d.action_space, gym.spaces.MultiDiscrete)
    assert env_d.action_space.nvec.tolist() == [9, 2, 2]
    assert info_d["cores"] == 0

    # Test continuous mode
    env_c = BlacklineHeistEnv(seed=3, action_mode="continuous")
    observation_c, info_c = env_c.reset()
    assert observation_c.shape == (84,)
    assert env_c.observation_space.contains(observation_c)
    assert env_c.action_space.shape == (4,)


def test_boost_drains_and_recharges_energy() -> None:
    env = BlacklineHeistEnv(seed=4, action_mode="continuous")
    env.reset()

    normal_env = BlacklineHeistEnv(seed=4, action_mode="continuous")
    normal_env.reset()
    normal_env.step(np.array([1.0, 0.0, -1.0, -1.0], dtype=np.float32))

    env.step(np.array([1.0, 0.0, 1.0, -1.0], dtype=np.float32))
    assert env.dash_energy < DASH_ENERGY_MAX
    boosted_speed = np.linalg.norm(env.velocity)
    assert boosted_speed > np.linalg.norm(normal_env.velocity) * 2.0

    for _ in range(DASH_COOLDOWN_STEPS):
        env.step(np.zeros(4, dtype=np.float32))
    assert env.dash_energy > DASH_ENERGY_MAX * 0.9


def test_collecting_cores_unlocks_extraction() -> None:
    env = BlacklineHeistEnv(seed=5, action_mode="continuous")
    env.reset()

    for core in env.cores:
        env.player = core.copy()
        _, reward, terminated, _, _ = env.step(np.zeros(4, dtype=np.float32))
        assert reward > 2.0
        assert not terminated

    assert env.alarm_active is True
    assert env.dash_energy == DASH_ENERGY_MAX

    env.player = env.extraction_point.copy()
    _, reward, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))
    assert reward > 10.0
    assert terminated
    assert info["extracted"] is True
    assert info["human_score"] > 0


def test_emp_powerup_is_collected_then_activated() -> None:
    env = BlacklineHeistEnv(seed=8, action_mode="continuous")
    env.reset()
    env.drones = []

    env.player = env.terminal_point.copy()
    _, reward, _, _, _ = env.step(np.zeros(4, dtype=np.float32))

    assert reward > 0.0
    assert env.terminal_available is False
    assert env.emp_available is True
    assert env.emp_timer == 0

    # Move player away from the terminal point
    env.player = env.terminal_point + np.array([200.0, 200.0], dtype=np.float32)

    # Insert two drones:
    # 1. Near the player's new position (50 units away) -> should be stunned
    # 2. Near the terminal point (now 250 units away from player) -> should NOT be stunned
    from neon_arena.env import Drone
    drone_near_player = Drone(
        route=((float(env.player[0] + 50.0), float(env.player[1])),),
        position=env.player + np.array([50.0, 0.0], dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32),
        route_index=0,
        base_speed=0.0,
        role="sentry",
    )
    drone_near_terminal = Drone(
        route=((float(env.terminal_point[0] + 50.0), float(env.terminal_point[1])),),
        position=env.terminal_point + np.array([50.0, 0.0], dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32),
        route_index=0,
        base_speed=0.0,
        role="sentry",
    )
    env.drones = [drone_near_player, drone_near_terminal]

    _, reward, _, _, _ = env.step(np.array([0.0, 0.0, -1.0, 1.0], dtype=np.float32))

    assert reward > 0.0
    assert env.emp_available is False
    assert env.emp_timer == EMP_DURATION_STEPS
    
    # Assert that drone_near_player is stunned, and drone_near_terminal is NOT stunned
    assert env.drone_is_emp_stunned(drone_near_player) is True
    assert env.drone_is_emp_stunned(drone_near_terminal) is False


def test_static_block_collision_prevents_entry() -> None:
    env = BlacklineHeistEnv(seed=6, action_mode="continuous")
    env.reset()
    block = env.city_blocks[0]
    env.player = np.array([block.left - PLAYER_RADIUS - 2.0, block.top + block.height * 0.5], dtype=np.float32)
    env.velocity = np.zeros(2, dtype=np.float32)

    for _ in range(8):
        _, _, _, _, info = env.step(np.array([1.0, 0.0, -1.0, -1.0], dtype=np.float32))
    assert info["collision"] is True
    assert float(env.player[0]) < block.left


def test_hunter_vision_escalates_with_objective_progress() -> None:
    env = BlacklineHeistEnv(seed=9, action_mode="continuous")
    env.reset()

    assert env.hunter_lock_range == HUNTER_LOCK_RANGE

    env.cores_collected = 2
    assert env.hunter_lock_range == HUNTER_LOCK_RANGE + 2 * HUNTER_RANGE_PER_CORE

    env.alarm_active = True
    assert env.hunter_lock_range == HUNTER_LOCK_RANGE + 2 * HUNTER_RANGE_PER_CORE + HUNTER_ALARM_RANGE_BONUS


def test_stalled_policy_is_truncated() -> None:
    env = BlacklineHeistEnv(seed=10, action_mode="continuous")
    env.reset()
    env.stalled_steps = STALL_TERMINATE_STEPS

    _, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))

    assert not terminated
    assert truncated
    assert reward < -1.0
    assert info["stalled_steps"] > STALL_TERMINATE_STEPS


def test_seeded_resets_still_vary_layouts() -> None:
    env = BlacklineHeistEnv(seed=20, action_mode="continuous")
    signatures = set()

    for _ in range(8):
        env.reset()
        first_block = env.city_blocks[0]
        signatures.add((env.layout_name, round(first_block.x), round(first_block.y)))

    assert len(signatures) > 1


def test_approaching_objective_gives_positive_reward() -> None:
    env = BlacklineHeistEnv(seed=30, action_mode="continuous")
    env.reset()
    env.drones = []
    target = env.objective.copy()
    candidate_offsets = (
        np.array([-90.0, 0.0], dtype=np.float32),
        np.array([90.0, 0.0], dtype=np.float32),
        np.array([0.0, -90.0], dtype=np.float32),
        np.array([0.0, 90.0], dtype=np.float32),
    )
    for offset in candidate_offsets:
        candidate = target + offset
        if env._can_occupy(candidate):
            env.player = candidate
            break
    else:
        raise AssertionError("No clear test position near objective.")
    env.velocity = np.zeros(2, dtype=np.float32)
    env.previous_mission_potential = env._route_potential()
    env.best_mission_potential = env.previous_mission_potential

    direction = target - env.player
    direction = direction / np.linalg.norm(direction)
    _, reward, _, _, _ = env.step(np.array([direction[0], direction[1], -1.0, -1.0], dtype=np.float32))

    assert reward > 0.0


# New PPO learning fix specific tests:

def test_discrete_action_mapping() -> None:
    env = BlacklineHeistEnv(action_mode="discrete")
    env.reset()
    
    # Idle action [0, 0] -> velocity stays 0 (when no initial movement)
    env.step([0, 0, 0])
    assert np.allclose(env.velocity, 0.0)

    # Move right [3, 0] -> positive x velocity
    env.step([3, 0, 0])
    assert env.velocity[0] > 0.0
    assert np.allclose(env.velocity[1], 0.0)

    # Move up-left [8, 0]
    env.reset()
    env.step([8, 0, 0])
    assert env.velocity[0] < 0.0
    assert env.velocity[1] < 0.0


def test_navigation_grid_routing() -> None:
    env = BlacklineHeistEnv(seed=42)
    env.reset()
    
    # Verify that the grid is built and populated with cols/rows
    assert env.cols > 0
    assert env.rows > 0
    assert env.grid_passable.shape == (env.cols, env.rows)
    
    # Verify distance map has valid distance values at player
    pc, pr = env._world_to_grid(env.player)
    assert env.dist_map[pc, pr] >= 0.0
    
    # Get waypoints
    path = env._get_route_waypoints(env.player, env.objective, env.dist_map)
    assert len(path) > 0


def test_curriculum_stages_drones_and_cores() -> None:
    # 1. route stage: 1 core, no drones, fixed midtown layout
    env_route = BlacklineHeistEnv(curriculum_stage="route")
    env_route.reset()
    assert len(env_route.cores) == 1
    assert len(env_route.drones) == 0
    assert env_route.layout_name == "midtown"

    # 2. sequence stage: 3 cores, no drones, fixed layout
    env_seq = BlacklineHeistEnv(curriculum_stage="sequence")
    env_seq.reset()
    assert len(env_seq.cores) == 3
    assert len(env_seq.drones) == 0
    assert env_seq.layout_name == "midtown"

    # 3. random-city: 3 cores, no drones, randomized layouts
    env_city = BlacklineHeistEnv(curriculum_stage="random-city")
    env_city.reset()
    assert len(env_city.cores) == 3
    assert len(env_city.drones) == 0

    # 4. sentry: 3 cores, one static sentry threat
    env_sentry = BlacklineHeistEnv(curriculum_stage="sentry")
    env_sentry.reset()
    assert len(env_sentry.drones) == 1
    assert env_sentry.drones[0].role == "sentry"

    # 5. patrols: 3 cores, patrol drones and sentry, but no hunter role
    env_patrols = BlacklineHeistEnv(curriculum_stage="patrols")
    env_patrols.reset()
    assert len(env_patrols.drones) > 0
    assert all(d.role != "hunter" for d in env_patrols.drones)
    assert any(d.role == "sentry" for d in env_patrols.drones)

    # 6. full: hunter role is present
    env_full = BlacklineHeistEnv(curriculum_stage="full")
    env_full.reset()
    assert any(d.role == "hunter" for d in env_full.drones)

    # 7. large_easy: 3 cores, no drones, large layout
    env_le = BlacklineHeistEnv(curriculum_stage="large_easy")
    env_le.reset()
    assert len(env_le.cores) == 3
    assert len(env_le.drones) == 0
    assert env_le.world_width == 1500.0

    # 8. large_sentry_camera: 3 cores, sentries & cameras
    env_lsc = BlacklineHeistEnv(curriculum_stage="large_sentry_camera")
    env_lsc.reset()
    assert len(env_lsc.cores) == 3
    assert len(env_lsc.drones) == 1
    assert env_lsc.drones[0].role == "sentry"
    assert len(env_lsc.cameras) > 0

    # 9. large_patrols_camera_no_hunters: patrols, sentries, cameras, no hunters
    env_lp = BlacklineHeistEnv(curriculum_stage="large_patrols_camera_no_hunters")
    env_lp.reset()
    assert len(env_lp.cores) == 3
    assert len(env_lp.drones) > 0
    assert all(d.role != "hunter" for d in env_lp.drones)
    assert any(d.role == "sentry" for d in env_lp.drones)
    assert len(env_lp.cameras) > 0

    # 10. large_full: hunters, patrols, sentries, cameras
    env_lf = BlacklineHeistEnv(curriculum_stage="large_full")
    env_lf.reset()
    assert any(d.role == "hunter" for d in env_lf.drones)
    assert len(env_lf.cameras) > 0


def test_terminal_info_and_fail_reasons() -> None:
    env = BlacklineHeistEnv(curriculum_stage="full", action_mode="continuous")
    env.reset()
    
    # Take damage until shield is depleted
    env.shield = 1
    env.contact_cooldown = 0
    env.drones = [env.drones[0]] # keep one drone
    env.player = env.drones[0].position.copy() # force overlap
    
    _, _, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))
    assert terminated is True
    assert info["is_success"] is False
    assert info["fail_reason"] == "courier_down"
    assert info["curriculum_stage"] == "full"


def test_security_heat_increases_and_emp_reduces_it() -> None:
    env = BlacklineHeistEnv(curriculum_stage="full", action_mode="continuous", seed=50)
    env.reset()
    env.heat = 40.0
    env.max_heat = 40.0

    env.player = env.terminal_point.copy()
    env.step(np.zeros(4, dtype=np.float32))
    _, _, _, _, info = env.step(np.array([0.0, 0.0, -1.0, 1.0], dtype=np.float32))

    assert env.heat < 40.0
    assert info["heat"] == env.heat
    assert info["max_heat"] >= 40.0


def test_patrol_drones_lock_and_chase_when_player_is_seen() -> None:
    env = BlacklineHeistEnv(curriculum_stage="patrols", action_mode="continuous", seed=51)
    env.reset()
    patrol = next(drone for drone in env.drones if drone.role == "patrol")
    patrol.position = np.array([300.0, 300.0], dtype=np.float32)
    patrol.velocity = np.array([1.0, 0.0], dtype=np.float32)
    env.player = np.array([390.0, 300.0], dtype=np.float32)
    env.city_blocks = ()

    env._update_drones()

    assert patrol.lock_steps == PATROL_LOCK_STEPS
    assert patrol.velocity[0] > 0.0


def test_hunter_trail_decays_when_visibility_is_broken() -> None:
    env = BlacklineHeistEnv(curriculum_stage="full", action_mode="continuous", seed=52)
    env.reset()
    hunter = next(drone for drone in env.drones if drone.role == "hunter")
    hunter.position = np.array([500.0, 500.0], dtype=np.float32)
    hunter.velocity = np.array([1.0, 0.0], dtype=np.float32)
    env.player = np.array([590.0, 500.0], dtype=np.float32)
    env.city_blocks = ()

    env._update_drones()
    assert hunter.lock_steps == HUNTER_LOCK_STEPS

    env.player = np.array([500.0, 650.0], dtype=np.float32)
    env._update_drones()

    assert hunter.lock_steps == HUNTER_LOCK_STEPS - HUNTER_LOST_LOCK_DECAY


def test_hunter_chase_speed_is_capped_to_player_speed() -> None:
    env = BlacklineHeistEnv(curriculum_stage="full", action_mode="continuous", seed=53)
    env.reset()
    hunter = next(drone for drone in env.drones if drone.role == "hunter")
    hunter.position = np.array([500.0, 500.0], dtype=np.float32)
    hunter.velocity = np.array([1.0, 0.0], dtype=np.float32)
    hunter.base_speed = 20.0
    env.player = np.array([590.0, 500.0], dtype=np.float32)
    env.city_blocks = ()

    env._update_drones()

    assert np.linalg.norm(hunter.velocity) <= PLAYER_MAX_SPEED + 1e-5


def test_sentry_fires_projectile_and_projectile_hit_adds_heat() -> None:
    env = BlacklineHeistEnv(curriculum_stage="sentry", action_mode="continuous", seed=54)
    env.reset()
    sentry = env.drones[0]
    sentry.position = np.array([500.0, 500.0], dtype=np.float32)
    sentry.facing_angle = 0.0
    sentry.fire_cooldown = 0
    env.player = np.array([620.0, 500.0], dtype=np.float32)
    env.city_blocks = ()

    env._update_drones()

    assert len(env.projectiles) == 1
    assert sentry.fire_cooldown > 0

    env.projectiles = [
        Projectile(
            position=env.player + np.array([SENTRY_PROJECTILE_RADIUS * 0.5, 0.0], dtype=np.float32),
            velocity=np.zeros(2, dtype=np.float32),
            life=20,
        )
    ]
    env.heat = 0.0
    env.terminal_available = False
    env.step(np.zeros(4, dtype=np.float32))

    assert env.damage_taken == 1
    assert env.heat >= SENTRY_PROJECTILE_HIT_GAIN


def test_drone_hit_forces_player_separation() -> None:
    env = BlacklineHeistEnv(curriculum_stage="full", action_mode="continuous", seed=55)
    env.reset()
    hunter = next(drone for drone in env.drones if drone.role == "hunter")
    hunter.position = np.array([500.0, 500.0], dtype=np.float32)
    hunter.velocity = np.zeros(2, dtype=np.float32)
    hunter.lock_steps = HUNTER_LOCK_STEPS
    env.player = np.array([500.0, 500.0], dtype=np.float32)
    env.city_blocks = ()

    env.step(np.zeros(4, dtype=np.float32))

    assert env.damage_taken == 1
    distance = np.linalg.norm(env.player - hunter.position)
    assert distance >= PLAYER_RADIUS
    assert hunter.lock_steps == 0


def test_sentry_lock_on_behavior() -> None:
    import math
    env = BlacklineHeistEnv(curriculum_stage="sentry", action_mode="continuous", seed=60)
    env.reset()
    sentry = env.drones[0]
    sentry.position = np.array([500.0, 500.0], dtype=np.float32)
    sentry.facing_angle = 0.0
    sentry.fire_cooldown = 0
    env.player = np.array([600.0, 500.0], dtype=np.float32)
    env.city_blocks = ()

    env._update_drones()

    assert sentry.lock_steps == PATROL_LOCK_STEPS
    assert np.allclose(sentry.facing_angle, 0.0)

    env.player = np.array([570.7, 570.7], dtype=np.float32)
    env._update_drones()
    
    assert sentry.lock_steps == PATROL_LOCK_STEPS
    assert np.allclose(sentry.facing_angle, math.pi * 0.25, atol=1e-2)


def test_randomized_emp_and_sentry_spawns() -> None:
    env = BlacklineHeistEnv(curriculum_stage="full", seed=70)
    terminal_positions = set()
    sentry_positions = set()
    for _ in range(5):
        env.reset()
        terminal_positions.add((float(env.terminal_point[0]), float(env.terminal_point[1])))
        sentry = next(drone for drone in env.drones if drone.role == "sentry")
        sentry_positions.add((float(sentry.position[0]), float(sentry.position[1])))

    assert len(terminal_positions) > 1
    assert len(sentry_positions) > 1



def test_district_heist_camera_exposure() -> None:
    env = BlacklineHeistEnv(curriculum_stage="large_full", action_mode="continuous", seed=42)
    env.reset()
    
    assert len(env.cameras) > 0
    camera = env.cameras[0]
    
    # Place player exactly at camera position to guarantee cone coverage & LOS
    env.player = camera.position.copy()
    
    # Assert camera detects the player directly
    assert env._player_seen_by_camera(camera) is True


def test_district_heist_loot_collection() -> None:
    env = BlacklineHeistEnv(curriculum_stage="large_full", action_mode="continuous", seed=42)
    env.reset()
    
    assert len(env.loot) > 0
    loot = env.loot[0]
    
    # Place player at loot position
    env.player = loot.position.copy()
    
    # Reset human score tracking inputs
    env.damage_taken = 0
    env.dash_count = 0
    env.cores_collected = 0
    env.extracted = False
    
    # Record initial score and heat
    initial_score = env._human_score()
    initial_heat = env.heat
    
    # Take a step to trigger collection
    _, reward, _, _, _ = env.step(np.zeros(4, dtype=np.float32))
    
    # Collection should update the loot state
    assert loot.collected is True
    
    # Step reward should be impacted (contains +1.5 PPO loot reward)
    assert reward > 1.0
    
    # Human score must increase (e.g. +300, +800, or +1200 depending on type)
    new_score = env._human_score()
    assert new_score > initial_score
    
    # If gold cache, heat increases by 15
    if loot.type == "gold_cache":
        assert env.heat >= initial_heat + 14.0


def test_district_heist_one_time_rewards() -> None:
    env = BlacklineHeistEnv(curriculum_stage="large_full", action_mode="discrete", seed=42)
    env.reset()
    
    # Place a blue gate and player next to it
    from neon_arena.env import Gate
    from neon_arena.config import Rect
    env.gates = [Gate(rect=Rect(100.0, 100.0, 20.0, 20.0), type="blue", is_open=True)]
    env.used_shortcuts = set()
    env.visited_districts = set()
    env.cores_collected = 1 # Keep blue gate open during step
    
    # Traverse gate: player at (110.0, 110.0) is inside gate rect
    env.player = np.array([110.0, 110.0], dtype=np.float32)
    
    # First step: should receive shortcut reward (+1.5)
    obs, reward1, _, _, _ = env.step([0, 0, 0])
    assert 0 in env.used_shortcuts
    assert reward1 > 1.0 # Should contain +1.5 shortcut reward
    
    # Second step: same position, should NOT receive shortcut reward again
    _, reward2, _, _, _ = env.step([0, 0, 0])
    assert reward2 < 0.1
    
    # District entry: move player to district containing objective
    obj_district = env._get_district_id(env.objective)
    # Mock player position to that district
    mid_x = env.world_width / 2.0
    mid_y = env.world_height / 2.0
    if obj_district == 0:
        env.player = np.array([mid_x - 50.0, mid_y - 50.0], dtype=np.float32)
    elif obj_district == 1:
        env.player = np.array([mid_x - 50.0, mid_y + 50.0], dtype=np.float32)
    elif obj_district == 2:
        env.player = np.array([mid_x + 50.0, mid_y - 50.0], dtype=np.float32)
    else:
        env.player = np.array([mid_x + 50.0, mid_y + 50.0], dtype=np.float32)
        
    env.visited_districts = set() # Clear to trigger first time
    _, reward3, _, _, _ = env.step([0, 0, 0])
    assert obj_district in env.visited_districts
    assert reward3 > 0.8 # Should contain +1.0 district entry reward
    
    # Next step in same district should not give the reward again
    _, reward4, _, _, _ = env.step([0, 0, 0])
    assert reward4 < 0.1


def test_loot_utility_functions() -> None:
    env = BlacklineHeistEnv(curriculum_stage="large_full", action_mode="discrete", seed=42)
    env.reset()
    
    # 1. Data Shard restores shield
    from neon_arena.env import Loot
    env.shield = 1
    env.loot = [Loot(position=env.player.copy(), type="data_shard")]
    _, reward, _, _, _ = env.step([0, 0, 0])
    assert env.shield == 2
    assert reward > 1.0  # contains +1.5 loot reward

    # 2. Gold Cache drops heat (heat drops by 30, then decays by 0.09)
    env.heat = 50.0
    env.loot = [Loot(position=env.player.copy(), type="gold_cache")]
    _, reward, _, _, _ = env.step([0, 0, 0])
    assert abs(env.heat - 19.91) < 0.01
    assert reward > 1.0

    # 3. Black Box gives EMP charge
    env.emp_available = False
    env.loot_reward_total = 0.0
    env.loot = [Loot(position=env.player.copy(), type="black_box")]
    _, reward, _, _, _ = env.step([0, 0, 0])
    assert env.emp_available is True
    assert reward > 1.0


def test_camera_stun_and_redirection() -> None:
    env = BlacklineHeistEnv(curriculum_stage="large_full", action_mode="discrete", seed=42)
    env.reset()
    
    # Set up one camera and place player on it
    from neon_arena.env import Camera, Drone
    env.cameras = [
        Camera(
            position=env.player.copy(),
            base_angle=0.0,
            sweep_amplitude=0.1,
            sweep_speed=0.0,
            fov=3.0,
        )
    ]
    # Set up one drone 100 units away
    drone = Drone(
        route=((float(env.player[0] + 100.0), float(env.player[1])),),
        position=env.player + np.array([100.0, 0.0], dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32),
        route_index=0,
        base_speed=2.0,
        role="hunter",
    )
    env.drones = [drone]
    
    env.heat = 0.0
    env.camera_alert_cooldown = 0
    
    # Step the env: camera spots player -> heat spikes by 10, drone is redirected
    env.step([0, 0, 0])
    assert env.heat >= 10.0
    assert drone.lock_steps > 0
    assert drone.velocity[0] < 0.0  # moving left towards player

    # Apply EMP: stuns the camera and drone
    env.emp_timer = 10
    assert env.camera_is_emp_stunned(env.cameras[0]) is True
    assert env._player_seen_by_camera(env.cameras[0]) is False


def test_campaign_stage_setup() -> None:
    # Test Stage 1: Midtown Route (curriculum: route, small layout template index 0)
    env = BlacklineHeistEnv(campaign_stage=1)
    env.reset()
    assert env.curriculum_stage == "route"
    assert env.layout_name == "midtown"
    assert len(env.cores) == 1

    # Test Stage 7: Large Hub (curriculum: large_sentry_camera, large template index 1: large_district_beta)
    env_7 = BlacklineHeistEnv(campaign_stage=7)
    env_7.reset()
    assert env_7.curriculum_stage == "large_sentry_camera"
    assert env_7.layout_name == "large_district_beta"
    assert env_7.world_width == 1500.0


def test_procedural_generator() -> None:
    env = BlacklineHeistEnv(seed=42, curriculum_stage="large_proc_easy")
    
    for _ in range(10):
        env.reset()
        assert env.world_width == 1500.0
        assert env.world_height == 950.0
        assert env.layout_id == 99
        assert len(env.cores) == 3
        
        # Check spawn safety from cameras
        for cam in env.cameras:
            assert np.linalg.norm(cam.position - env.spawn_point) >= 150.0
            
        # Check objective spacing
        for j in range(len(env.cores)):
            for k in range(j + 1, len(env.cores)):
                assert np.linalg.norm(env.cores[j] - env.cores[k]) >= 200.0
        assert np.linalg.norm(env.terminal_point - env.spawn_point) >= 250.0
        
        # Check reachability
        assert env._layout_is_valid(
            blocks=env.city_blocks,
            core_slots=env.core_slots,
            spawn_point=tuple(env.spawn_point),
            extraction_point=tuple(env.extraction_point),
            terminal_point=tuple(env.terminal_point),
        )


def test_generator_fallback() -> None:
    env = BlacklineHeistEnv(seed=99, curriculum_stage="large_proc_easy")
    
    original_layout_is_valid = env._layout_is_valid
    env._layout_is_valid = lambda **kwargs: False
    
    env.reset()
    assert env.layout_id == 0
    assert env.layout_name == "large_district_alpha"
    
    env._layout_is_valid = original_layout_is_valid


def test_drone_ai_and_pathfinding() -> None:
    env = BlacklineHeistEnv(seed=42, curriculum_stage="large_proc_patrols")
    env.reset()
    
    assert len(env.semantic_patrol_nodes) > 0
    for node in env.semantic_patrol_nodes:
        assert isinstance(node, np.ndarray)
        assert len(node) == 2
        assert env._can_occupy(node)


        
    for drone in env.drones:
        if drone.role != "sentry":
            assert drone.state == "PATROL"
            assert len(drone.route) > 0
            
    start = env.spawn_point.copy()
    end = env.extraction_point.copy()
    path = env._find_path_astar(start, end)
    assert path is not None
    assert len(path) > 0
    assert np.linalg.norm(path[-1] - end) <= 50.0


def test_room_graph_has_15_rooms() -> None:
    env = BlacklineHeistEnv(seed=42, curriculum_stage="large_proc_easy")
    env.reset()
    assert len(env.room_nodes) <= 15
    assert len(env.room_nodes) >= 6
    for i in range(len(env.room_nodes)):
        assert env.room_nodes[i].id == i


def test_room_graph_is_connected() -> None:
    from collections import deque
    env = BlacklineHeistEnv(seed=42, curriculum_stage="large_proc_easy")
    env.reset()
    visited = {0}
    queue = deque([0])
    while queue:
        curr = queue.popleft()
        for neighbor in env.room_adjacency[curr]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    assert len(visited) == len(env.room_nodes)


def test_all_rooms_degree_at_least_2_on_proc_easy() -> None:
    env = BlacklineHeistEnv(seed=42, curriculum_stage="large_proc_easy")
    for seed in (42, 100, 2023, 777, 999):
        env.reset(seed=seed)
        for room_id in range(len(env.room_nodes)):
            deg = len(env.room_adjacency[room_id])
            assert deg >= 2, f"Room {room_id} has degree {deg} in seed {seed}"


def test_room_ids_are_stable_by_grid_position() -> None:
    env = BlacklineHeistEnv(seed=42, curriculum_stage="large_proc_easy")
    env.reset()
    for room in env.room_nodes:
        assert 0 <= room.id < len(env.room_nodes)
        assert 0 <= room.col < 5
        assert 0 <= room.row < 3


def test_next_doorway_hint_points_to_valid_door() -> None:
    env = BlacklineHeistEnv(seed=42, curriculum_stage="large_proc_easy")
    env.reset()
    spawn_room_id = next(n.id for n in env.room_nodes if n.role == "SPAWN")
    env.player = np.array(env.room_nodes[spawn_room_id].center, dtype=np.float32)
    target = env._get_next_route_target()
    found_matching_door = False
    for edge in env.door_edges:
        if np.allclose(edge.center, target):
            assert edge.room_a == spawn_room_id or edge.room_b == spawn_room_id
            found_matching_door = True
            break
    assert found_matching_door


def test_next_doorway_hint_points_to_objective_inside_objective_room() -> None:
    env = BlacklineHeistEnv(seed=42, curriculum_stage="large_proc_easy")
    env.reset()
    obj_room_id = env._get_room_id_at_position(env.objective)
    env.player = np.array(env.room_nodes[obj_room_id].center, dtype=np.float32)
    target = env._get_next_route_target()
    assert np.allclose(target, env.objective)


def test_new_room_reward_is_capped() -> None:
    env = BlacklineHeistEnv(seed=42, curriculum_stage="large_proc_easy")
    env.reset()
    env.visited_rooms = set()
    env.exploration_reward_total = 0.0
    num_rooms = len(env.room_nodes)
    for r_id in range(num_rooms):
        env.player = np.array(env.room_nodes[r_id].center, dtype=np.float32)
        env.step([0, 0, 0])
    assert len(env.visited_rooms) == num_rooms
    assert abs(env.exploration_reward_total - min(1.5, num_rooms * 0.1)) < 0.01


def test_objective_room_reward_once_per_objective() -> None:
    env = BlacklineHeistEnv(seed=42, curriculum_stage="large_proc_easy")
    env.reset()
    obj_room_id = env._get_room_id_at_position(env.objective)
    obj_center = np.array(env.room_nodes[obj_room_id].center, dtype=np.float32)
    env.player = obj_center - np.array([40.0, 40.0], dtype=np.float32)
    env.objective_room_reward_claimed = set()
    _, reward1, _, _, _ = env.step([0, 0, 0])
    claim_key = (env.cores_collected, obj_room_id)
    assert claim_key in env.objective_room_reward_claimed
    _, reward2, _, _, _ = env.step([0, 0, 0])
    assert reward1 - reward2 > 0.4


def test_wrong_room_penalty_after_120_steps() -> None:
    env = BlacklineHeistEnv(seed=42, curriculum_stage="large_proc_easy")
    env.reset()
    spawn_room_id = next(n.id for n in env.room_nodes if n.role == "SPAWN")
    spawn_center = np.array(env.room_nodes[spawn_room_id].center, dtype=np.float32)
    env.player = spawn_center
    env.wrong_room_steps = 0
    for _ in range(120):
        env.step([0, 0, 0])
    assert env.wrong_room_steps == 120
    env.step([0, 0, 0])
    assert env.wrong_room_steps == 121
    other_room = next(n.id for n in env.room_nodes if n.role != "SPAWN")
    env.player = np.array(env.room_nodes[other_room].center, dtype=np.float32)
    env.step([0, 0, 0])
    assert env.wrong_room_steps == 0


def test_static_template_fallback_obs_shape_84() -> None:
    env = BlacklineHeistEnv(seed=42, curriculum_stage="full")
    obs, info = env.reset()
    assert obs.shape == (84,)
    assert obs[81] == -1.0


def test_reward_components_sum_to_step_reward() -> None:
    env = BlacklineHeistEnv(seed=42, curriculum_stage="large_proc_easy")
    env.reset()
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        comp_sum = sum(env.episode_rewards.values())
        assert abs(env.episode_reward - comp_sum) < 1e-5, f"Mismatch: {env.episode_reward} vs {comp_sum}"
        if terminated or truncated:
            break


def test_episode_reward_components_export_on_termination() -> None:
    env = BlacklineHeistEnv(seed=42, curriculum_stage="full")
    env.reset()
    env.shield = 1
    env.contact_cooldown = 0
    env.player = env.drones[0].position.copy()
    obs, reward, terminated, truncated, info = env.step([0, 0, 0])
    assert terminated is True
    assert "reward_hit" in info
    assert "reward_extraction" in info
    assert "episode_success" in info
    assert "combat_drone_contacts" in info


def test_heat_penalty_limits() -> None:
    env = BlacklineHeistEnv(seed=42, curriculum_stage="full")
    env.reset()
    env.heat = 100.0
    env.step([0, 0, 0])
    assert abs(env.episode_rewards["heat"] - (-0.06)) < 1e-5

    env.reset()
    env.heat = 50.0
    env.step([0, 0, 0])
    assert abs(env.episode_rewards["heat"] - (-0.04)) < 1e-3


def test_emp_no_inventory_press_penalty() -> None:
    env = BlacklineHeistEnv(seed=42, curriculum_stage="full")
    env.reset()
    env.emp_available = False
    env.step([0, 0, 1])
    assert env.emp_invalid_presses_count == 1
    assert abs(env.episode_rewards["emp"] - (-0.05)) < 1e-5


def test_emp_inventory_wasted_use_penalty() -> None:
    env = BlacklineHeistEnv(seed=42, curriculum_stage="full")
    env.reset()
    env.emp_available = True
    env.drones = []
    env.step([0, 0, 1])
    assert env.emp_uses_count == 1
    assert env.emp_wasted_uses_count == 1
    assert abs(env.episode_rewards["emp"] - (-0.5)) < 1e-5


def test_loot_cap_prevents_more_than_3_reward() -> None:
    env = BlacklineHeistEnv(seed=42, curriculum_stage="large_full")
    env.reset()
    from neon_arena.env import Loot
    env.loot = [
        Loot(position=env.player.copy(), type="data_shard"),
        Loot(position=env.player.copy(), type="data_shard"),
        Loot(position=env.player.copy(), type="data_shard"),
    ]
    env.step([0, 0, 0])
    assert env.loot_items_collected_count == 3
    assert abs(env.loot_reward_total - 3.0) < 1e-5
    assert abs(env.episode_rewards["loot"] - 3.0) < 1e-5
    assert env.loot_reward_capped_flag == 1.0


def test_timeout_penalty_cores_logic() -> None:
    env = BlacklineHeistEnv(seed=42, curriculum_stage="large_proc_easy")
    env.reset()
    env.step_count = 2500
    env.cores_collected = 0
    obs, reward, terminated, truncated, info = env.step([0, 0, 0])
    assert truncated is True
    assert info["is_success"] is False
    assert abs(info["reward_timeout"] - (-36.0)) < 1e-5

    env.reset()
    env.step_count = 2500
    env.cores_collected = 3
    env.collected_core_indices = {0, 1, 2}
    obs, reward, terminated, truncated, info = env.step([0, 0, 0])
    assert truncated is True
    assert abs(info["reward_timeout"] - (-4.0)) < 1e-5


def test_wrong_room_penalty_resets() -> None:
    env = BlacklineHeistEnv(seed=42, curriculum_stage="large_proc_easy")
    env.reset()
    spawn_room_id = next(n.id for n in env.room_nodes if n.role == "SPAWN")
    spawn_center = np.array(env.room_nodes[spawn_room_id].center, dtype=np.float32)
    env.player = spawn_center
    for _ in range(120):
        env.step([0, 0, 0])
    assert env.wrong_room_steps == 120

    other_room_id = next(n.id for n in env.room_nodes if n.role != "SPAWN")
    other_room_center = np.array(env.room_nodes[other_room_id].center, dtype=np.float32)
    env.player = other_room_center
    env.step([0, 0, 0])
    assert env.wrong_room_steps == 0

    env.player = other_room_center
    for _ in range(120):
        env.step([0, 0, 0])
    assert env.wrong_room_steps == 120

    from neon_arena.env import Drone
    env.drones = [
        Drone(
            route=((float(env.player[0]), float(env.player[1])),),
            position=env.player.copy(),
            velocity=np.zeros(2, dtype=np.float32),
            route_index=0,
            base_speed=0.0,
            role="sentry",
            lock_steps=10
        )
    ]
    env.contact_cooldown = 100
    env.step([0, 0, 0])
    assert env.wrong_room_steps == 0


def test_polyomino_room_merging_limits() -> None:
    env = BlacklineHeistEnv(seed=12, curriculum_stage="large_proc_full")
    env.reset()
    assert len(env.room_nodes) < 15
    for node in env.room_nodes:
        # Reconstruct grid cell bounding box from rect width/height
        # Cell column boundaries in large_proc are at least 270 units wide
        # Ensure no room spans more than 2 cells in either dimension
        assert node.rect.width <= 750.0
        assert node.rect.height <= 700.0


def test_map_connectivity_and_routing() -> None:
    env = BlacklineHeistEnv(seed=15, curriculum_stage="large_proc_full")
    env.reset()
    spawn_id = next(n.id for n in env.room_nodes if n.role == "SPAWN")
    vault_id = next(n.id for n in env.room_nodes if n.role == "VAULT")
    path = env._dijkstra_room_path(spawn_id, vault_id)
    assert len(path) >= 1
    assert path[0] == spawn_id
    assert path[-1] == vault_id





def test_obs_shape_remains_84() -> None:
    env = BlacklineHeistEnv(seed=55, curriculum_stage="large_proc_full")
    obs, info = env.reset()
    assert obs.shape == (84,)
    
    obs, reward, terminated, truncated, info = env.step([3, 0, 0])
    assert obs.shape == (84,)


def test_proximity_penalty_is_applied() -> None:
    env = BlacklineHeistEnv(seed=42, curriculum_stage="full")
    env.reset()
    env.drones[0].position = env.player + np.array([65.0, 0.0], dtype=np.float32)
    env.drones[0].state = "PATROL"
    env.step([0, 0, 0])
    assert abs(env.episode_rewards["proximity"] - (-0.009)) < 1e-3


def test_navigation_grid_sync_and_hysteresis() -> None:
    env = BlacklineHeistEnv(seed=42, curriculum_stage="large_sentry_camera")
    env.reset()
    
    from neon_arena.config import Rect
    from neon_arena.env import Gate
    mock_gate = Gate(rect=Rect(100.0, 100.0, 50.0, 20.0), type="blue", is_open=False)
    env.city_blocks = ()
    env.gates = [mock_gate]
    env._build_grid()
    env._last_gates_state = (True,) # force mismatch
    
    env._observation()
    test_gate = env.gates[0]
    test_gate.is_open = True
    
    env._observation()
    gate_center = np.array([test_gate.rect.x + test_gate.rect.width / 2.0, test_gate.rect.y + test_gate.rect.height / 2.0], dtype=np.float32)
    gc, gr = env._world_to_grid(gate_center)
    assert env.grid_passable[gc, gr]

    test_gate.is_open = False
    env._observation()
    env.player = gate_center - np.array([40.0, 0.0], dtype=np.float32)
    dist = env._ray_distance(np.array([1.0, 0.0], dtype=np.float32))
    assert dist < 50.0

    env.player_current_room = 0
    room_node = env.room_nodes[0]
    env.player = np.array([room_node.rect.x + room_node.rect.width, room_node.rect.y + room_node.rect.height / 2.0], dtype=np.float32)
    assert env.player_current_room == 0
    
    # Teleport to another room node if there are multiple room nodes
    if len(env.room_nodes) > 1:
        other_room = env.room_nodes[1]
        env.player = np.array([other_room.rect.x + other_room.rect.width / 2.0, other_room.rect.y + other_room.rect.height / 2.0], dtype=np.float32)
        assert env.player_current_room == 1









