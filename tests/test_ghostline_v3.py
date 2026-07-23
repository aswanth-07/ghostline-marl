from __future__ import annotations

import json
import numpy as np
import pytest
import torch
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from pettingzoo.test import parallel_api_test

import ghostline
from ghostline.cli import build_parser
from ghostline.env_v3 import GhostlineEnvV3
from ghostline.marl_train import (
    _adaptive_tier_probabilities,
    _batched_security_actions,
    _selection_key,
    evaluate_security_checkpoint,
    train_security,
)
from ghostline.security_env import GhostlineSecurityParallelEnv, _capped_radio_credit
from ghostline.security_baselines import tactical_security_action
from ghostline.security_controller import AdaptiveSecurityController
from ghostline.security_model import SharedSecurityActorCritic, load_security_policy, save_security_policy
from ghostline.simulation import norm
from ghostline.simulation_v3 import GhostlineSimulationV3
from ghostline.types import Action, GuardMode
from ghostline.types_v3 import (
    ContractDirective,
    GuardRole,
    RadioMessage,
    RunnerActionV3,
    SecurityIntent,
    SecurityOrder,
)


def test_v3_action_contract_round_trips_all_72_semantic_combinations() -> None:
    assert {RunnerActionV3.decode(value).encode() for value in range(72)} == set(range(72))
    assert RunnerActionV3.decode(71) == RunnerActionV3(move=8, dash=True, pulse=True, decoy=True)


def test_adaptive_cli_defaults_bind_training_to_the_frozen_runner() -> None:
    train = build_parser().parse_args(["train-security", "--dry-run"])
    evaluate = build_parser().parse_args(["evaluate-security"])
    play = build_parser().parse_args(["play", "--adaptive", "--directive", "ghost"])
    assert train.envs == 8
    assert train.runner_model.as_posix() == "models/ghostline-policy.pt"
    assert not train.scripted_runner
    assert evaluate.seed_start == 12_000_000
    assert play.adaptive and play.directive == "ghost"


def test_v2_contract_remains_immutable_while_v3_is_registered() -> None:
    classic = gym.make("GhostlineEnv-v2", seed=11, tier=6)
    adaptive = gym.make("GhostlineEnv-v3", seed=11, tier=6, directive="ghost")
    classic_observation, classic_info = classic.reset(seed=11)
    adaptive_observation, adaptive_info = adaptive.reset(seed=11)
    assert classic.action_space.n == 36
    assert classic_observation["ego"].shape == (24,)
    assert classic_observation["entities"].shape == (12, 13)
    assert "directive" not in classic_info
    assert adaptive.action_space.n == 72
    assert adaptive_observation["ego"].shape == (27,)
    assert adaptive_observation["entities"].shape == (12, 16)
    assert adaptive_observation["directive"].shape == (6,)
    assert adaptive_info["contract"] == "GhostlineEnv-v3"
    classic.close()
    adaptive.close()


def test_v3_environment_checker_and_directive_observation() -> None:
    env = GhostlineEnvV3(seed=31, tier=6, directive=ContractDirective.GREED)
    observation, info = env.reset(seed=31)
    assert env.observation_space.contains(observation)
    assert observation["local_grid"].shape == (11, 15, 15)
    assert observation["rays"].shape == (24, 4)
    assert observation["action_mask"].shape == (72,)
    assert observation["directive"][2] == 1.0
    assert info["directive"] == "greed"
    check_env(env, skip_render_check=True)
    env.close()


def test_v3_replay_is_deterministic_for_seed_tier_directive_and_actions() -> None:
    actions = [RunnerActionV3.decode(value) for value in (1, 10, 46, 3, 21, 0, 71, 8) * 4]
    first = GhostlineSimulationV3(seed=701, tier=6, directive="speed")
    second = GhostlineSimulationV3(seed=701, tier=6, directive="speed")
    for action in actions:
        first.advance(action, ticks=6)
        second.advance(action, ticks=6)
    assert np.array_equal(first.player, second.player)
    assert np.array_equal(first.velocity, second.velocity)
    assert first.trace == pytest.approx(second.trace)
    assert first.integrity == second.integrity
    assert [(door.tile, door.locked) for door in first.security_doors] == [
        (door.tile, door.locked) for door in second.security_doors
    ]
    assert [tuple(guard.position) for guard in first.level.guards] == [
        tuple(guard.position) for guard in second.level.guards
    ]


def test_noise_decoy_is_latched_limited_and_attracts_operatives() -> None:
    sim = GhostlineSimulationV3(seed=51, tier=5)
    sim.level.guards[0].position = sim.player + np.asarray((32.0, 0.0), dtype=np.float32)
    sim.events.clear()
    before = sim.decoy_charges
    sim.advance(RunnerActionV3(move=1, decoy=True), ticks=6)
    assert sim.decoy_charges == before - 1
    assert sim.decoys_used == 1
    assert len(sim.decoys) == 1
    assert [event.kind for event in sim.events].count("decoy_deployed") == 1
    assert any(state.heard_confidence > 0.0 for state in sim.operative_states.values())
    sim.advance(RunnerActionV3(move=1, decoy=True), ticks=6)
    assert sim.decoys_used == 1


def test_security_doors_only_use_redundant_edges_and_are_telegraphed() -> None:
    sim = GhostlineSimulationV3(seed=93, tier=6)
    assert len(sim.security_doors) == 3
    by_tile = {door.tile: door for door in sim.level.doors}
    for security_door in sim.security_doors:
        source = by_tile[security_door.tile]
        assert sim._door_edge_is_redundant(source.room_a, source.room_b)
    door = sim.security_doors[0]
    sim.player[:] = sim.level.spawn
    for guard in sim.level.guards:
        guard.position[:] = sim.level.spawn
    assert sim._request_nearest_door_lock(np.asarray((10_000.0, 10_000.0), dtype=np.float32))
    assert any(candidate.warning_remaining > 0.0 for candidate in sim.security_doors)
    warned = next(candidate for candidate in sim.security_doors if candidate.warning_remaining > 0.0)
    sim._update_security_doors(1.0)
    assert warned.locked
    assert warned.tile in sim._blocked_tiles


def test_pulse_jams_radio_and_forces_nearby_security_door_open() -> None:
    sim = GhostlineSimulationV3(seed=94, tier=6)
    door = sim.security_doors[0]
    door.lock_remaining = 3.0
    sim._refresh_navigation_blocks()
    sim.player[:] = np.asarray(((door.tile[0] + 0.5) * 32.0, (door.tile[1] + 0.5) * 32.0), dtype=np.float32)
    guard = sim.level.guards[0]
    guard.position[:] = sim.player
    sim._activate_pulse()
    assert guard.radio_jammed_for > 0.0
    assert not door.locked
    assert door.tile not in sim._blocked_tiles


def test_suppressor_projectile_has_aim_telegraph_and_friendly_fire_gate(monkeypatch) -> None:
    sim = GhostlineSimulationV3(seed=95, tier=6, external_security=True)
    suppressor = next(
        guard for guard in sim.level.guards if sim.operative_states[guard.guard_id].role == GuardRole.SUPPRESSOR
    )
    suppressor.position = sim.player + np.asarray((120.0, 0.0), dtype=np.float32)
    suppressor.facing = np.pi
    state = sim.operative_states[suppressor.guard_id]
    state.current_order = SecurityOrder(SecurityIntent.PURSUE, sim.player.copy(), RadioMessage.NONE, True)
    monkeypatch.setattr(sim, "visible", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(sim, "_shot_clear", lambda *_args, **_kwargs: True)
    sim.events.clear()
    sim._update_suppressors(0.35)
    assert state.aim_progress == pytest.approx(0.35)
    assert not sim.projectiles
    assert any(event.kind == "suppressor_aim" for event in sim.events)
    sim._update_suppressors(0.36)
    assert len(sim.projectiles) == 1
    assert any(event.kind == "suppressor_fire" for event in sim.events)


def test_external_security_waypoints_are_projected_inside_navigation_bounds() -> None:
    sim = GhostlineSimulationV3(seed=96, tier=6, external_security=True)
    guard = sim.level.guards[0]
    sim.set_security_orders(
        {
            guard.guard_id: SecurityOrder(
                SecurityIntent.FLANK_LEFT,
                np.asarray((sim.level.world_width + 96.0, -64.0), dtype=np.float32),
            )
        }
    )
    sim.advance(RunnerActionV3(), ticks=24)
    target = sim.operative_states[guard.guard_id].current_order.target
    assert target is not None
    tx, ty = (int(target[0] // 32), int(target[1] // 32))
    assert 0 <= tx < sim.level.grid.shape[1]
    assert 0 <= ty < sim.level.grid.shape[0]


def test_security_observation_hides_unperceived_runner_state(monkeypatch) -> None:
    env = GhostlineSecurityParallelEnv(tier=6, seed=20_000_031)
    observations, _ = env.reset(seed=20_000_031)
    guard = env.sim.level.guards[0]
    guard.mode = GuardMode.PATROL
    guard.velocity[:] = 0.0
    state = env.sim.operative_states[guard.guard_id]
    state.heard_confidence = 0.0
    monkeypatch.setattr(env.sim, "visible", lambda *_args, **_kwargs: False)
    env.sim.velocity[:] = 0.0
    observation = env._observation("guard_0")
    assert observation["runner"][0] == 0.0
    assert observation["runner"][1] == 0.0
    assert observation["runner"][5] == -1.0
    assert observation["runner"][6] == -1.0
    assert observation["runner"][7] == -1.0
    assert env.observation_space("guard_0").contains(observation)
    assert observations.keys() == {f"guard_{index}" for index in range(5)}
    env.close()


def test_security_parallel_api_and_parameter_shared_recurrent_policy(tmp_path) -> None:
    parallel_api_test(GhostlineSecurityParallelEnv(tier=6, seed=20_000_041), num_cycles=25)
    env = GhostlineSecurityParallelEnv(tier=6, seed=20_000_041)
    observations, _ = env.reset(seed=20_000_041)
    policy = SharedSecurityActorCritic(recurrent_size=256)
    action, hidden = policy.act(observations["guard_0"])
    assert action.shape == (4,)
    assert hidden.shape == (1, 1, 256)
    assert env.action_space("guard_0").contains(action)
    checkpoint = tmp_path / "security.pt"
    save_security_policy(policy, checkpoint, purpose="test")
    restored = load_security_policy(checkpoint)
    restored_action, _ = restored.act(observations["guard_0"])
    assert np.array_equal(action, restored_action)
    env.close()


def test_batched_security_evaluation_matches_individual_deterministic_actions() -> None:
    env = GhostlineSecurityParallelEnv(tier=6, seed=20_000_044)
    observations, _ = env.reset(seed=20_000_044)
    policy = SharedSecurityActorCritic(recurrent_size=256)
    individual = {
        agent: policy.act(observation, deterministic=True)[0]
        for agent, observation in observations.items()
    }
    batched, hidden = _batched_security_actions(
        policy,
        observations,
        None,
        deterministic=True,
        device=torch.device("cpu"),
    )
    assert hidden.shape == (1, len(observations), 256)
    assert all(np.array_equal(individual[agent], batched[agent]) for agent in observations)
    env.close()


def test_human_game_security_controller_batches_the_shared_policy() -> None:
    env = GhostlineSecurityParallelEnv(tier=6, seed=20_000_045)
    observations, _ = env.reset(seed=20_000_045)
    policy = SharedSecurityActorCritic(recurrent_size=256)
    individual = {
        agent: policy.act(observation, deterministic=True)[0]
        for agent, observation in observations.items()
    }
    controller = object.__new__(AdaptiveSecurityController)
    controller.policy = policy
    controller._batch_hidden = None
    controller._batch_agents = ()

    batched = controller._policy_actions(observations)

    assert controller._batch_hidden.shape == (1, len(observations), 256)
    assert all(np.array_equal(individual[agent], batched[agent]) for agent in observations)
    env.close()


def test_security_reward_components_sum_exactly() -> None:
    env = GhostlineSecurityParallelEnv(tier=6, seed=20_000_042)
    observations, _ = env.reset(seed=20_000_042)
    actions = {
        agent: tactical_security_action(observation, env.agent_name_mapping[agent])
        for agent, observation in observations.items()
    }
    _, rewards, _, _, infos = env.step(actions)
    for agent, reward in rewards.items():
        components = infos[agent]["reward_components"]
        assert components["total"] == pytest.approx(reward)
        assert sum(value for key, value in components.items() if key != "total") == pytest.approx(reward)
    env.close()


def test_security_radio_shaping_cannot_be_farmed_by_repeated_messages() -> None:
    assert _capped_radio_credit(0, 4, 5) == pytest.approx(0.02)
    assert _capped_radio_credit(4, 40, 5) == 0.0
    rewards = [_capped_radio_credit(before, after, 5) for before, after in ((0, 4), (4, 8), (8, 12))]
    assert sum(rewards) == pytest.approx(0.02)


@pytest.mark.parametrize(
    ("argument", "value"),
    (
        ("bc_warmup_steps", -1),
        ("bc_warmup_epochs", 0),
        ("bc_warmup_entropy", -0.1),
        ("scripted_opponent_fraction", -0.1),
        ("scripted_opponent_fraction", 1.1),
    ),
)
def test_security_training_rejects_invalid_warmup_arguments(tmp_path, argument, value) -> None:
    kwargs = {argument: value}
    with pytest.raises(ValueError):
        train_security(
            output=tmp_path / argument,
            hours=0.01,
            max_steps=20,
            env_count=1,
            rollout=3,
            epochs=1,
            tiers="6",
            recurrent_size=256,
            validation_interval=0,
            resume=False,
            device="cpu",
            **kwargs,
        )


def test_security_curriculum_targets_weakest_tiers_without_forgetting_replay() -> None:
    report = {
        "tiers": {
            "3": {"security_stop_rate": 0.2},
            "4": {"security_stop_rate": 0.0},
            "5": {"security_stop_rate": 0.0},
            "6": {"security_stop_rate": 0.1},
        }
    }
    probabilities = _adaptive_tier_probabilities(report, (3, 4, 5, 6))
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities.tolist() == pytest.approx([0.075, 0.425, 0.425, 0.075])


def test_security_selection_preserves_real_stops_before_damage_tiebreak() -> None:
    def report(rates, damages):
        tiers = {
            str(tier): {
                "security_stop_rate": rate,
                "mean_damage": damage,
                "mean_detections": 1.0,
                "mean_duration_seconds": 10.0,
            }
            for tier, rate, damage in zip((3, 4, 5, 6), rates, damages, strict=True)
        }
        return {"tiers": tiers, "worst_tier_security_stop_rate": min(rates)}

    useful = report((0.1, 0.1, 0.0, 0.0), (0.1, 0.1, 0.1, 0.1))
    noisy = report((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0))
    assert _selection_key(useful) > _selection_key(noisy)


def test_tactical_security_baseline_is_masked_and_shared_with_game_controller() -> None:
    env = GhostlineSecurityParallelEnv(tier=6, seed=20_000_043)
    observations, _ = env.reset(seed=20_000_043)
    for agent, observation in observations.items():
        action = tactical_security_action(observation, env.agent_name_mapping[agent])
        assert env.action_space(agent).contains(action)
        for value, key in zip(
            action,
            ("intent_mask", "target_mask", "message_mask", "ability_mask"),
            strict=True,
        ):
            assert observation[key][value] == 1
    env.close()


def test_security_mappo_cpu_smoke_run(tmp_path) -> None:
    champion = train_security(
        output=tmp_path / "security-smoke",
        hours=0.01,
        max_steps=20,
        env_count=1,
        rollout=3,
        epochs=1,
        tiers="6",
        recurrent_size=256,
        validation_interval=0,
        resume=False,
        device="cpu",
        bc_warmup_steps=8,
        bc_warmup_epochs=1,
    )
    assert champion.exists()
    assert load_security_policy(champion).recurrent_size == 256
    assert (tmp_path / "security-smoke" / "behavior-warmup.json").is_file()


def test_security_mappo_can_start_from_compatible_policy(tmp_path) -> None:
    initialization = tmp_path / "initial-security.pt"
    save_security_policy(SharedSecurityActorCritic(recurrent_size=256), initialization, purpose="test-init")
    champion = train_security(
        output=tmp_path / "initialized-smoke",
        hours=0.01,
        max_steps=12,
        env_count=1,
        rollout=3,
        epochs=1,
        tiers="6",
        recurrent_size=256,
        validation_interval=0,
        resume=False,
        device="cpu",
        init_checkpoint=initialization,
    )
    assert champion.exists()
    payload = torch.load(tmp_path / "initialized-smoke" / "latest.pt", map_location="cpu", weights_only=False)
    assert str(initialization) in payload["training_args"]["init_checkpoint"]


def test_security_resume_fails_closed_when_opponent_contract_changes(tmp_path) -> None:
    output = tmp_path / "security-resume"
    train_security(
        output=output,
        hours=0.01,
        max_steps=20,
        env_count=1,
        rollout=3,
        epochs=1,
        tiers="6",
        recurrent_size=256,
        validation_interval=0,
        resume=False,
        device="cpu",
    )
    payload = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
    payload["training_args"]["runner_opponent"] = "different-opponent"
    torch.save(payload, output / "latest.pt")
    with pytest.raises(RuntimeError, match="runner opponent"):
        train_security(
            output=output,
            hours=0.01,
            max_steps=40,
            env_count=1,
            rollout=3,
            epochs=1,
            tiers="6",
            recurrent_size=256,
            validation_interval=0,
            resume=True,
            device="cpu",
        )


def test_security_evaluation_writes_json_and_both_csv_views(tmp_path) -> None:
    output = tmp_path / "security-evaluation.json"
    evaluate_security_checkpoint(
        model=None,
        output=output,
        tiers="3",
        episodes_per_tier=1,
        seed_start=11_900_000,
        device="cpu",
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    summary = report["tiers"]["3"]
    assert 0.0 <= summary["security_stop_ci95_low"] <= summary["security_stop_ci95_high"] <= 1.0
    assert output.with_suffix(".csv").is_file()
    assert output.with_name("security-evaluation.episodes.csv").is_file()
