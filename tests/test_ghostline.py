from __future__ import annotations

import os

import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from ghostline.curriculum import AdaptiveCurriculum
from ghostline.config import (
    DRONE_STRIKE_WINDUP_SECONDS,
    GUARD_CHASE_SPEED_RATIOS,
    GUARD_GRADE_SPEED_MULTIPLIERS,
    GUARD_PATROL_DWELL_SECONDS,
    GUARD_SEARCH_DURATION_MULTIPLIERS,
    GUARD_STRIKE_WINDUP_SECONDS,
    PLAYER_PERCEPTION_DISTANCE,
    PLAYER_SPEED,
    TIERS,
)
from ghostline.env import GhostlineEnv
from ghostline.generation import MIN_GRAPH_LOOPS, LevelGenerator, tile_center
from ghostline.model import UniversalGhostlinePolicy
from ghostline.progression import load_progression, record_success
from ghostline.simulation import GhostlineSimulation, norm
from ghostline.types import Action, Drone, GuardGrade, GuardMode, Tile


def test_action_contract_round_trips_every_combination() -> None:
    assert {Action.decode(value).encode() for value in range(36)} == set(range(36))
    assert Action.decode(35) == Action(move=8, dash=True, pulse=True)


@pytest.mark.parametrize("tier", range(1, 7))
def test_generator_is_deterministic_connected_and_has_enough_data(tier: int) -> None:
    generator = LevelGenerator()
    first = generator.generate(seed=713, tier=tier)
    second = generator.generate(seed=713, tier=tier)
    assert np.array_equal(first.grid, second.grid)
    assert [prop.kind for prop in first.props] == [prop.kind for prop in second.props]
    assert [terminal.value for terminal in first.terminals] == [terminal.value for terminal in second.terminals]
    assert generator.validate(first)
    assert sum(terminal.value for terminal in first.terminals) >= first.quota


@pytest.mark.parametrize("tier", range(1, 7))
def test_generation_guarantees_route_redundancy_and_fair_link_pockets(tier: int) -> None:
    generator = LevelGenerator()
    spec = TIERS[tier]
    for seed in range(24):
        level = generator.generate(seed=10_000 + seed, tier=tier)
        assert len(level.cameras) == spec.cameras
        assert len(level.guards) == spec.guards
        assert len(level.doors) - (len(level.rooms) - 1) >= MIN_GRAPH_LOOPS[tier]
        for terminal in level.terminals:
            assert len(generator.terminal_approach_tiles(level, terminal)) >= 3
            assert generator.terminal_approach_tiles(level, terminal, camera_safe_only=True)


def test_guard_grades_escalate_without_turning_every_guard_elite() -> None:
    assert {guard.grade for guard in GhostlineSimulation(seed=91, tier=3).level.guards} == {
        GuardGrade.STANDARD
    }
    tier4 = GhostlineSimulation(seed=91, tier=4).level.guards
    assert [guard.grade for guard in tier4].count(GuardGrade.INTERCEPTOR) == 1
    assert {guard.grade for guard in GhostlineSimulation(seed=91, tier=5).level.guards} == set(GuardGrade)
    tier6_grades = [guard.grade for guard in GhostlineSimulation(seed=91, tier=6).level.guards]
    assert tier6_grades.count(GuardGrade.STANDARD) == 3
    assert tier6_grades.count(GuardGrade.INTERCEPTOR) == 1
    assert tier6_grades.count(GuardGrade.ELITE) == 1


def test_guard_grade_speed_curve_is_modest_and_player_escape_remains_possible(monkeypatch) -> None:
    sim = GhostlineSimulation(seed=91, tier=6)
    captured: dict[GuardGrade, float] = {}
    monkeypatch.setattr(sim, "visible", lambda *args, **kwargs: False)

    def capture_speed(guard, target, speed, dt):
        captured[guard.grade] = speed

    monkeypatch.setattr(sim, "_move_agent", capture_speed)
    for guard in sim.level.guards:
        guard.mode = GuardMode.CHASE
        guard.mode_seconds = 2.0
    sim._update_guards(1.0 / 60.0)

    expected = {
        grade: PLAYER_SPEED * GUARD_CHASE_SPEED_RATIOS[int(grade)]
        for grade in GuardGrade
    }
    assert captured == pytest.approx(expected)
    assert captured[GuardGrade.STANDARD] < captured[GuardGrade.INTERCEPTOR]
    assert captured[GuardGrade.INTERCEPTOR] < captured[GuardGrade.ELITE] < PLAYER_SPEED
    assert captured[GuardGrade.ELITE] == pytest.approx(PLAYER_SPEED * 0.99)


def test_guard_grades_have_readable_scan_cadence_and_search_persistence(monkeypatch) -> None:
    sim = GhostlineSimulation(seed=91, tier=6)
    monkeypatch.setattr(sim, "visible", lambda *_args, **_kwargs: False)
    by_grade = {grade: next(guard for guard in sim.level.guards if guard.grade == grade) for grade in GuardGrade}

    for grade, guard in by_grade.items():
        sim.level.guards[:] = [guard]
        guard.mode = GuardMode.PATROL
        guard.position[:] = guard.patrol[guard.patrol_index]
        guard.patrol_pause_seconds = 0.0
        sim._update_guards(1.0 / 60.0)
        assert guard.patrol_pause_seconds == pytest.approx(GUARD_PATROL_DWELL_SECONDS[int(grade)])

        guard.mode = GuardMode.CHASE
        guard.mode_seconds = 0.0
        guard.awareness = 0.0
        sim._update_guards(1.0 / 60.0)
        assert guard.mode == GuardMode.SEARCH
        assert guard.mode_seconds == pytest.approx(3.5 * GUARD_SEARCH_DURATION_MULTIPLIERS[int(grade)])

    assert GUARD_PATROL_DWELL_SECONDS[0] > GUARD_PATROL_DWELL_SECONDS[1] > GUARD_PATROL_DWELL_SECONDS[2]
    assert GUARD_SEARCH_DURATION_MULTIPLIERS[0] < GUARD_SEARCH_DURATION_MULTIPLIERS[1] < GUARD_SEARCH_DURATION_MULTIPLIERS[2]


def test_dash_noise_ring_emits_once_per_continuous_dash() -> None:
    sim = GhostlineSimulation(seed=91, tier=1)
    sim.events.clear()
    sim.advance(Action(move=3, dash=True), ticks=12)
    assert [event.kind for event in sim.events].count("dash_noise") == 1
    sim.advance(Action(move=3, dash=False), ticks=1)
    sim.advance(Action(move=3, dash=True), ticks=1)
    assert [event.kind for event in sim.events].count("dash_noise") == 2


def test_response_drone_has_a_trace_warning_before_deployment() -> None:
    sim = GhostlineSimulation(seed=91, tier=5)
    sim.events.clear()
    sim.trace = sim.level.drone_trace_threshold - 9.0
    sim._update_drones(1.0 / 60.0)
    assert not sim.drones
    assert [event.kind for event in sim.events] == ["drone_warning"]
    sim.trace = sim.level.drone_trace_threshold
    sim._update_drones(1.0 / 60.0)
    assert len(sim.drones) == 1
    assert [event.kind for event in sim.events].count("drone_deployed") == 1


def test_guards_traverse_every_door_without_jamb_stalls() -> None:
    for seed in range(6):
        sim = GhostlineSimulation(seed=20_000 + seed, tier=6)
        guard = sim.level.guards[0]
        sim.level.guards[:] = [guard]
        for door in sim.level.doors:
            room_a, room_b = sim.level.rooms[door.room_a], sim.level.rooms[door.room_b]
            x, y = door.tile
            sides = ((x - 1, y), (x + 1, y)) if room_a.row == room_b.row else ((x, y - 1), (x, y + 1))
            for start, target in (sides, sides[::-1]):
                guard.position[:] = tile_center(start)
                guard.stuck_seconds = 0.0
                sim._guard_waypoints.clear()
                for _ in range(150):
                    sim.elapsed_ticks += 1
                    sim._move_agent(guard, tile_center(target), 54.0, 1.0 / 60.0)
                assert norm(guard.position - tile_center(target)) < 12.0


def test_exploration_reveals_furniture_face_but_not_floor_behind_it() -> None:
    sim = GhostlineSimulation(seed=713, tier=1)
    player_tile = (2, 2)
    furniture_tile = (4, 2)
    hidden_floor_tile = (5, 2)
    for x, y in (player_tile, (3, 2), furniture_tile, hidden_floor_tile):
        sim.level.grid[y, x] = Tile.FLOOR
    sim.player[:] = tile_center(player_tile)
    sim._blocked_tiles = {furniture_tile}
    sim.explored[:] = False

    assert not sim.line_of_sight(sim.player, tile_center(furniture_tile))
    sim._reveal_near_player()

    assert sim.explored[furniture_tile[1], furniture_tile[0]]
    assert not sim.explored[hidden_floor_tile[1], hidden_floor_tile[0]]


def test_environment_contract_and_checker() -> None:
    env = GhostlineEnv(seed=41, tier=6)
    observation, info = env.reset(seed=41)
    assert env.action_space.n == 36
    assert env.observation_space.contains(observation)
    assert observation["ego"].shape == (24,)
    assert observation["objective"].shape == (8,)
    assert observation["local_grid"].shape == (8, 15, 15)
    assert observation["targets"].shape == (5, 10)
    assert observation["entities"].shape == (12, 13)
    assert observation["rays"].shape == (24, 3)
    assert info["seed"] == 41
    check_env(env, skip_render_check=True)
    env.close()


def test_registered_v1_compatibility_and_v2_objective_contract() -> None:
    import ghostline

    legacy = gym.make("GhostlineEnv-v1", seed=5, tier=1)
    modern = gym.make("GhostlineEnv-v2", seed=5, tier=1)
    legacy_observation, _ = legacy.reset(seed=5)
    modern_observation, _ = modern.reset(seed=5)
    assert "objective" not in legacy_observation
    assert modern_observation["objective"].shape == (8,)
    assert modern.observation_space.contains(modern_observation)
    legacy.close(); modern.close()


@pytest.mark.parametrize("lesson,expected_tier", [(1, 1), (2, 1), (3, 1), (4, 2), (5, 3), (6, 4), (7, 6)])
def test_reverse_curriculum_lessons_do_not_change_final_distribution(lesson: int, expected_tier: int) -> None:
    env = GhostlineEnv(seed=91, tier=6)
    observation, _ = env.reset(seed=91, options={"tier": 6, "training_lesson": lesson})
    assert env.tier == expected_tier
    assert env.observation_space.contains(observation)
    if lesson == 1:
        assert env.sim.level.quota == 1 and len(env.sim.level.terminals) == 1
    if lesson == 7:
        reference = GhostlineEnv(seed=91, tier=6)
        reference.reset(seed=91)
        assert len(env.sim.level.guards) == len(reference.sim.level.guards)
        reference.close()
    env.close()


def test_terminal_telemetry_has_efficiency_and_action_accounting() -> None:
    env = GhostlineEnv(seed=3, tier=1)
    env.reset(seed=3, options={"training_lesson": 1})
    for _ in range(500):
        _, _, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break
    assert "telemetry" in info
    telemetry = info["telemetry"]
    assert sum(telemetry["action_counts"]) == telemetry["decision_count"]
    assert 0.0 <= telemetry["path_efficiency"] <= 1.0
    env.close()


def test_v2_shares_live_facility_security_tracking_with_the_player() -> None:
    env = GhostlineEnv(seed=19, tier=3)
    env.reset(seed=19)
    for guard in env.sim.level.guards:
        guard.mode = GuardMode.PATROL
    env.sim.line_of_sight = lambda *args, **kwargs: False  # type: ignore[method-assign]
    observation = env._observation()
    assert observation["entity_mask"].sum() == len(env.sim.level.guards) + len(env.sim.level.cameras)
    assert observation["local_grid"][6].sum() > 0.0
    env.close()


def test_player_and_policy_share_one_live_perception_gate(monkeypatch) -> None:
    env = GhostlineEnv(seed=19, tier=3)
    env.reset(seed=19)
    guard = env.sim.level.guards[0]
    env.sim.level.guards[:] = [guard]
    env.sim.level.cameras.clear()
    env.sim.drones.clear()
    guard.position = env.sim.player + np.asarray((PLAYER_PERCEPTION_DISTANCE - 20.0, 0.0), dtype=np.float32)
    monkeypatch.setattr(env.sim, "line_of_sight", lambda *_args, **_kwargs: True)

    assert env.sim.player_can_see(guard.position)
    observation = env._observation()
    assert observation["entity_mask"].sum() == 1
    assert observation["entities"][0, 11] == pytest.approx(1.0)
    env.close()


def test_facility_intel_tracks_live_guard_through_occlusion(monkeypatch) -> None:
    env = GhostlineEnv(seed=19, tier=3)
    env.reset(seed=19)
    guard = env.sim.level.guards[0]
    env.sim.level.guards[:] = [guard]
    env.sim.level.cameras.clear()
    guard.position = env.sim.player + np.asarray((64.0, 0.0), dtype=np.float32)
    visible = True
    monkeypatch.setattr(env.sim, "line_of_sight", lambda *_args, **_kwargs: visible)
    env.sim._update_player_intel()
    visible = False
    guard.mode = GuardMode.PATROL
    guard.velocity[:] = 0.0
    guard.position += np.asarray((91.0, 57.0), dtype=np.float32)
    env.sim.elapsed_ticks += 180
    env.sim._update_player_intel()
    percept = env._security_percepts()[0]
    assert np.array_equal(percept.position, guard.position)
    observation = env._observation()
    assert observation["entity_mask"].sum() == 1
    assert observation["entities"][0, 11] == pytest.approx(1.0)
    env.close()


def test_live_facility_tracking_takes_precedence_over_audio_estimates(monkeypatch) -> None:
    env = GhostlineEnv(seed=19, tier=3)
    env.reset(seed=19)
    guard = env.sim.level.guards[0]
    env.sim.level.guards[:] = [guard]
    env.sim.level.cameras.clear()
    guard.position = env.sim.player + np.asarray((73.0, 37.0), dtype=np.float32)
    guard.mode = GuardMode.CHASE
    monkeypatch.setattr(env.sim, "line_of_sight", lambda *_args, **_kwargs: False)

    percept = env._security_percepts()[0]
    assert np.allclose(percept.position, guard.position)
    record = env._entities([percept])[0][0]
    assert record[11] == pytest.approx(1.0)
    env.close()


@pytest.mark.parametrize(
    "grade,encoded",
    [
        (GuardGrade.STANDARD, -1.0),
        (GuardGrade.INTERCEPTOR, 0.0),
        (GuardGrade.ELITE, 1.0),
    ],
)
def test_entity_record_exposes_visible_guard_grade(monkeypatch, grade: GuardGrade, encoded: float) -> None:
    env = GhostlineEnv(seed=19, tier=3)
    env.reset(seed=19)
    guard = env.sim.level.guards[0]
    env.sim.level.guards[:] = [guard]
    env.sim.level.cameras.clear()
    guard.grade = grade
    guard.position = env.sim.player + np.asarray((48.0, 0.0), dtype=np.float32)
    monkeypatch.setattr(env.sim, "line_of_sight", lambda *_args, **_kwargs: True)
    entities, mask = env._entities()
    assert mask.sum() == 1
    assert entities[0, 12] == encoded
    env.close()


def test_pulse_observation_preserves_every_charge_count() -> None:
    env = GhostlineEnv(seed=91, tier=5)
    env.reset(seed=91)
    values = []
    for charges in range(5):
        env.sim.pulse_charges = charges
        values.append(float(env._ego()[9]))
    assert values == pytest.approx([-1.0, -0.5, 0.0, 0.5, 1.0])
    env.close()


def test_map_known_geometry_is_not_erased_by_exploration_memory() -> None:
    env = GhostlineEnv(seed=91, tier=5)
    env.reset(seed=91)
    px, py = (int(value // 32) for value in env.sim.player)
    env.sim.explored[:] = False
    env.sim.level.grid[py, px + 1] = Tile.WALL
    env.sim.level.grid[py, px + 2] = Tile.DOOR
    local = env._local_grid([])
    center = local.shape[-1] // 2
    assert local[3].sum() == 0.0
    assert local[1, center, center + 1] == 1.0
    assert local[2, center, center + 2] == 1.0
    env.close()


def test_v2_objective_phase_changes_from_acquire_to_extract() -> None:
    env = GhostlineEnv(seed=12, tier=1)
    observation, _ = env.reset(seed=12, options={"training_lesson": 1})
    assert observation["objective"][0] == -1.0
    terminal = env.sim.level.terminals[0]
    env.sim.player[:] = terminal.position
    env.sim.advance(Action(), ticks=int((terminal.hack_seconds + 0.1) * 60))
    observation = env._observation()
    assert observation["objective"][0] == 1.0
    assert 0.0 <= observation["objective"][6] <= 1.0
    env.close()


def test_acquire_objective_is_stable_until_linking_another_terminal() -> None:
    env = GhostlineEnv(seed=2_300_027, tier=3)
    env.reset(seed=2_300_027)
    selected = env.sim.objective_terminal()
    assert selected is not None
    other = next(terminal for terminal in env.sim.level.terminals if terminal is not selected)

    env.sim.player[:] = other.position + np.asarray((45.0, 0.0))
    env._objective()
    assert env.sim.objective_terminal_id == selected.terminal_id

    env.sim.player[:] = other.position
    env.sim.advance(Action())
    assert env.sim.objective_terminal_id == other.terminal_id
    env.close()


def test_deterministic_replay_from_seed_and_actions() -> None:
    actions = np.random.default_rng(4).integers(0, 36, size=120)
    trajectories = []
    for _ in range(2):
        env = GhostlineEnv(seed=99, tier=4)
        env.reset(seed=99)
        states = []
        for action in actions:
            observation, reward, terminated, truncated, _ = env.step(int(action))
            states.append(
                (
                    env.sim.player.copy(),
                    reward,
                    env.sim.trace,
                    env.sim.data,
                    observation["entities"].copy(),
                    observation["entity_mask"].copy(),
                )
            )
            if terminated or truncated:
                break
        trajectories.append(states)
    assert len(trajectories[0]) == len(trajectories[1])
    for first, second in zip(*trajectories):
        assert np.allclose(first[0], second[0])
        assert first[1:4] == pytest.approx(second[1:4])
        assert np.array_equal(first[4], second[4])
        assert np.array_equal(first[5], second[5])


def test_hack_quota_and_extract_flow() -> None:
    sim = GhostlineSimulation(seed=10, tier=1)
    for terminal in sim.level.terminals:
        if sim.quota_met:
            break
        sim.player[:] = terminal.position
        sim.velocity[:] = 0.0
        sim.advance(Action(), ticks=int((terminal.hack_seconds + 0.1) * 60))
        assert terminal.completed
    assert sim.quota_met
    assert sim.trace_floor > 0.0
    sim.player[:] = sim.level.extraction
    sim.advance(Action())
    assert sim.extracted and sim.terminated


def test_pulse_disables_electronics_and_jams_guards() -> None:
    sim = GhostlineSimulation(seed=22, tier=4)
    camera = sim.level.cameras[0]
    guard = sim.level.guards[0]
    camera.position[:] = sim.player + np.asarray((30.0, 0.0))
    guard.position[:] = sim.player + np.asarray((40.0, 0.0))
    starting_charges = sim.pulse_charges
    sim.advance(Action(pulse=True))
    assert sim.pulse_charges == starting_charges - 1
    assert camera.disabled_for > 0.0
    assert guard.radio_jammed_for > 0.0


def test_guard_detection_raises_trace_and_damage_is_recoverable() -> None:
    sim = GhostlineSimulation(seed=31, tier=3)
    guard = sim.level.guards[0]
    guard.position[:] = sim.player + np.asarray((180.0, 0.0))
    guard.facing = np.pi
    guard.mode = GuardMode.PATROL
    guard.patrol = [guard.position.copy()]
    sim.visible = lambda *args, **kwargs: True  # type: ignore[method-assign]
    sim.advance(Action(), ticks=45)
    assert sim.trace > 0.0
    assert sim.integrity == 3


def test_partial_guard_sighting_enters_readable_suspicious_state() -> None:
    sim = GhostlineSimulation(seed=31, tier=3)
    guard = sim.level.guards[0]
    sim.level.guards[:] = [guard]
    sim.visible = lambda *args, **kwargs: True  # type: ignore[method-assign]
    sim.advance(Action(), ticks=9)
    assert guard.mode == GuardMode.SUSPICIOUS
    assert 0.18 <= guard.awareness < 1.0


def test_damage_breaks_up_nearby_guard_dogpile() -> None:
    sim = GhostlineSimulation(seed=31, tier=6)
    nearby = sim.level.guards[:3]
    for index, guard in enumerate(nearby):
        guard.position[:] = sim.player + np.asarray((8.0 + 8.0 * index, 0.0))
        guard.mode = GuardMode.CHASE
        guard.awareness = 1.0
        guard.attack_windup = 0.3
    sim._damage(nearby[0].position, source_kind="guard")
    assert all(guard.mode == GuardMode.SEARCH for guard in nearby)
    assert all(guard.attack_windup == 0.0 for guard in nearby)
    assert all(guard.hit_cooldown >= 1.75 for guard in nearby)


def test_guard_tackle_has_readable_windup_before_damage() -> None:
    sim = GhostlineSimulation(seed=31, tier=3)
    guard = sim.level.guards[0]
    sim.level.guards[:] = [guard]
    sim.level.cameras.clear()
    guard.position[:] = sim.player
    guard.mode = GuardMode.CHASE
    guard.awareness = 1.0
    sim.visible = lambda *args, **kwargs: True  # type: ignore[method-assign]

    sim.advance(Action(), ticks=20)
    assert sim.integrity == 3
    assert 0.0 < guard.attack_windup < GUARD_STRIKE_WINDUP_SECONDS

    sim.advance(Action(), ticks=12)
    assert sim.integrity == 2
    assert guard.hit_cooldown > 0.0
    assert guard.attack_windup == 0.0
    assert guard.mode == GuardMode.SEARCH


def test_response_drone_strike_has_windup_and_recoil() -> None:
    sim = GhostlineSimulation(seed=31, tier=5)
    sim.level.guards.clear()
    sim.level.cameras.clear()
    drone = Drone(0, sim.player.copy())
    sim.drones[:] = [drone]

    sim.advance(Action(), ticks=25)
    assert sim.integrity == 3
    assert 0.0 < drone.attack_windup < DRONE_STRIKE_WINDUP_SECONDS

    sim.advance(Action(), ticks=12)
    assert sim.integrity == 2
    assert drone.attack_windup == 0.0
    assert drone.disabled_for > 0.0
    assert drone.hit_cooldown > 0.0


def test_guard_radio_alert_is_local_and_limited_to_two_responders() -> None:
    sim = GhostlineSimulation(seed=31, tier=6)
    source, *responders = sim.level.guards
    assert len(responders) >= 3
    source.position[:] = sim.player
    source.last_known[:] = sim.player + np.asarray((12.0, 4.0))
    for index, guard in enumerate(responders):
        guard.position[:] = source.position + np.asarray((50.0 + index * 45.0, 0.0))
        guard.mode = GuardMode.PATROL

    sim._share_alert(source)

    investigating = [guard for guard in responders if guard.mode == GuardMode.INVESTIGATE]
    assert investigating == responders[:2]
    assert all(guard.mode == GuardMode.PATROL for guard in responders[2:])


def test_hacking_continues_while_moving_inside_terminal_zone() -> None:
    sim = GhostlineSimulation(seed=10, tier=1)
    terminal = sim.level.terminals[0]
    sim.player[:] = terminal.position
    sim.advance(Action(move=3), ticks=max(1, int(terminal.hack_seconds * 30)))
    assert terminal.progress > 0.25
    progress = terminal.progress
    sim.player[:] = terminal.position
    sim.advance(Action(move=7), ticks=max(1, int(terminal.hack_seconds * 40)))
    assert terminal.completed or terminal.progress > progress


def test_damage_has_global_grace_and_preserves_hack_progress() -> None:
    sim = GhostlineSimulation(seed=31, tier=3)
    terminal = sim.level.terminals[0]
    terminal.progress = terminal.hack_seconds * 0.5
    source = sim.player + np.asarray((5.0, 0.0))
    sim._damage(source)
    sim._damage(source)
    assert sim.integrity == 2
    assert terminal.progress == pytest.approx(terminal.hack_seconds * 0.5)


def test_damage_source_accounting_distinguishes_guards_and_drones() -> None:
    sim = GhostlineSimulation(seed=31, tier=5)
    sim._damage(sim.player + np.asarray((5.0, 0.0)), source_kind="guard")
    sim.damage_cooldown = 0.0
    sim._damage(sim.player + np.asarray((5.0, 0.0)), source_kind="drone")
    info = sim.terminal_info()
    assert info["damage_by_guard"] == 1
    assert info["damage_by_drone"] == 1


def test_vectorized_recurrent_sequence_matches_stepwise_policy() -> None:
    import torch

    env = GhostlineEnv(seed=4, tier=1)
    observation, _ = env.reset(seed=4)
    observations = []
    for _ in range(3):
        observations.append({key: torch.as_tensor(value).unsqueeze(0) for key, value in observation.items()})
        observation, _, _, _, _ = env.step(0)
    sequence = {key: torch.stack([item[key][0] for item in observations]).unsqueeze(1) for key in observations[0]}
    policy = UniversalGhostlinePolicy().eval()
    sequence_logits, sequence_values, _ = policy.forward_sequence(sequence)
    hidden = None
    for tick, item in enumerate(observations):
        logits, values, hidden = policy(item, hidden)
        assert torch.allclose(sequence_logits[tick], logits, atol=1e-5)
        assert torch.allclose(sequence_values[tick], values, atol=1e-5)


def test_reward_components_sum_exactly() -> None:
    env = GhostlineEnv(seed=8, tier=1)
    env.reset(seed=8)
    reward_total = 0.0
    for _ in range(50):
        _, reward, terminated, truncated, info = env.step(0)
        reward_total += reward
        if terminated or truncated:
            break
    assert reward_total == pytest.approx(sum(env.reward_components.values()))


def test_potential_progress_reward_cannot_profit_from_a_closed_cycle() -> None:
    from ghostline.env import REWARD_DISCOUNT, potential_progress_reward

    first, second = 0.8, 1.4
    outward = potential_progress_reward(first, second)
    returning = potential_progress_reward(second, first)
    discounted_cycle = outward + REWARD_DISCOUNT * returning
    assert discounted_cycle <= 0.0
    assert discounted_cycle == pytest.approx(0.35 * (REWARD_DISCOUNT**2 - 1.0) * first)


def test_invalid_actions_are_masked_and_safe() -> None:
    env = GhostlineEnv(seed=7, tier=1)
    observation, _ = env.reset(seed=7)
    assert observation["action_mask"][Action(move=0, dash=True).encode()] == 0
    _, reward, _, _, _ = env.step(Action(move=0, dash=True, pulse=True).encode())
    assert np.isfinite(reward)


def test_curriculum_promotes_after_two_passes_and_retains_earlier_tiers() -> None:
    curriculum = AdaptiveCurriculum()
    assert not curriculum.observe_validation(1, 0.96)
    assert curriculum.observe_validation(1, 0.97)
    assert curriculum.current_tier == 2
    rng = np.random.default_rng(5)
    samples = [curriculum.sample_tier(rng) for _ in range(500)]
    assert 1 in samples and 2 in samples
    assert samples.count(2) > samples.count(1)


def test_universal_model_forward_contract() -> None:
    import torch

    env = GhostlineEnv(seed=2, tier=6)
    observation, _ = env.reset(seed=2)
    policy = UniversalGhostlinePolicy()
    tensors = {key: torch.as_tensor(value).unsqueeze(0) for key, value in observation.items()}
    logits, value, hidden = policy(tensors)
    assert logits.shape == (1, 36)
    assert value.shape == (1,)
    assert hidden is not None and hidden.shape == (1, 1, policy.recurrent_size)
    assert torch.isfinite(logits[tensors["action_mask"].bool()]).all()


def test_progression_is_versioned_and_persistent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert load_progression()["highest_unlocked_tier"] == 1
    record_success(tier=1, score=1234)
    data = load_progression()
    assert data["highest_unlocked_tier"] == 2
    assert data["best_scores"]["1"] == 1234


def test_headless_renderer_returns_logical_rgb(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from ghostline.presentation import GhostlineRenderer

    sim = GhostlineSimulation(seed=42, tier=6)
    renderer = GhostlineRenderer(sim, visible=False)
    frame = renderer.draw(return_array=True)
    renderer.close()
    assert frame.shape == (360, 640, 3)
    assert frame.dtype == np.uint8
    assert float(frame.mean()) > 4.0
