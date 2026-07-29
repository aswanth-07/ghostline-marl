from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from pettingzoo.test import parallel_api_test

import ghostline
from ghostline.cli import build_parser
from ghostline.env_v2 import GhostlineEnvV2
from ghostline.marl_train import (
    SECURITY_EXPERIMENT_MANIFEST_CONTRACT,
    _adaptive_tier_probabilities,
    _batched_security_actions,
    _selection_key,
    evaluate_security_checkpoint,
    train_security,
)
from ghostline.security_env import GhostlineSecurityParallelEnv, _capped_radio_credit
from ghostline.security_baselines import tactical_security_action
from ghostline.security_controller import AdaptiveSecurityController
from ghostline.security_model import (
    SharedSecurityActorCritic,
    _canonical_security_source_digest,
    load_security_policy,
    save_security_policy,
    security_environment_fingerprint,
)
from ghostline.simulation import norm
from ghostline.simulation_v2 import GhostlineSimulationV2
from ghostline.types import Action, GuardMode
from ghostline.types_v2 import (
    ContractDirective,
    GuardRole,
    RadioMessage,
    RunnerActionV2,
    SecurityIntent,
    SecurityOrder,
)


def test_v2_action_contract_round_trips_all_288_semantic_combinations() -> None:
    from ghostline.types_v2 import RUNNER_ACTION_COUNT_V2

    assert RUNNER_ACTION_COUNT_V2 == 288
    values = range(RUNNER_ACTION_COUNT_V2)
    assert {RunnerActionV2.decode(value).encode() for value in values} == set(values)
    assert RunnerActionV2.decode(71) == RunnerActionV2(move=8, dash=True, pulse=True, decoy=True)
    assert RunnerActionV2.decode(287) == RunnerActionV2(
        move=8, dash=True, pulse=True, decoy=True, crouch=True, interact=True
    )
    # Both new bits are additive: every original code keeps its meaning.
    assert all(not RunnerActionV2.decode(value).crouch for value in range(72))
    assert all(not RunnerActionV2.decode(value).interact for value in range(144))


def test_adaptive_cli_defaults_bind_training_to_the_frozen_runner() -> None:
    train = build_parser().parse_args(["train-security", "--dry-run"])
    evaluate = build_parser().parse_args(
        ["evaluate-security", "--model", "artifacts/security-v2/champion.pt"]
    )
    play = build_parser().parse_args(["play", "--adaptive", "--directive", "ghost"])
    assert train.envs == 8
    assert train.runner_model.as_posix() == "models/ghostline-policy.pt"
    assert not train.scripted_runner
    assert evaluate.seed_start == 14_000_000
    assert evaluate.episodes_per_tier == 500
    assert evaluate.slice_manifest == Path(
        "benchmarks/security/v2-final-test-slices.json"
    )
    slice_ledger = json.loads(
        (
            Path(__file__).resolve().parents[1] / evaluate.slice_manifest
        ).read_text(encoding="utf-8")
    )
    assert slice_ledger["environment_fingerprint"] == (
        security_environment_fingerprint()
    )
    assert slice_ledger["slices"][0]["status"] == "reserved_unopened"
    assert play.adaptive and play.directive == "ghost"


def test_published_v1_remains_immutable_while_multi_agent_v2_is_registered() -> None:
    published = gym.make("GhostlineEnv-v1", seed=11, tier=6)
    multi_agent = gym.make("GhostlineEnv-v2", seed=11, tier=6, directive="ghost")
    published_observation, published_info = published.reset(seed=11)
    multi_agent_observation, multi_agent_info = multi_agent.reset(seed=11)
    assert published.action_space.n == 36
    assert published_observation["ego"].shape == (24,)
    assert published_observation["entities"].shape == (12, 13)
    assert "directive" not in published_info
    assert published_info["contract"] == "GhostlineEnv-v1"
    assert published_info["historical_internal_contract"] == "GhostlineEnv-v2"
    assert multi_agent.action_space.n == 288
    assert multi_agent_observation["ego"].shape == (27,)
    assert multi_agent_observation["entities"].shape == (12, 16)
    assert multi_agent_observation["directive"].shape == (6,)
    assert multi_agent_observation["field_targets"].shape == (16, 13)
    assert multi_agent_info["contract"] == "GhostlineEnv-v2"
    published.close()
    multi_agent.close()


def test_v2_environment_checker_and_directive_observation() -> None:
    env = GhostlineEnvV2(seed=31, tier=6, directive=ContractDirective.GREED)
    observation, info = env.reset(seed=31)
    assert env.observation_space.contains(observation)
    assert observation["local_grid"].shape == (15, 15, 15)
    assert observation["field_targets"].shape == (16, 13)
    assert observation["field_target_mask"].shape == (16,)
    assert observation["rays"].shape == (24, 4)
    assert observation["action_mask"].shape == (288,)
    assert observation["directive"][2] == 1.0
    assert info["directive"] == "greed"
    check_env(env, skip_render_check=True)
    env.close()


def test_greed_keeps_objective_and_extraction_locked_until_every_terminal() -> None:
    env = GhostlineEnvV2(
        seed=32,
        tier=6,
        directive=ContractDirective.GREED,
    )
    observation, _ = env.reset(seed=32)
    assert len(env.sim.level.terminals) > 1

    first = env.sim.level.terminals[0]
    first.completed = True
    first.progress = first.hack_seconds
    env.sim.data = env.sim.level.quota
    env.sim.optional_data = 0
    selected = env.sim.objective_terminal()
    assert selected is not None and not selected.completed
    assert not env.sim.quota_met
    assert env._objective()[0] == -1.0
    assert not env._acquisition_complete()
    assert "LINK EVERY DATA NODE" in env.sim.context_hint

    env.sim.player = env.sim.level.extraction.copy()
    env.sim._check_extraction()
    assert not env.sim.extracted and not env.sim.terminated

    for terminal in env.sim.level.terminals:
        terminal.completed = True
        terminal.progress = terminal.hack_seconds
    env.sim.data = sum(
        terminal.value
        for terminal in env.sim.level.terminals
    )
    assert env.sim.objective_terminal() is None
    assert env.sim.quota_met
    assert env._acquisition_complete()
    assert env._objective()[0] == 1.0
    env.sim._check_extraction()
    assert env.sim.extracted and env.sim.terminated
    env.close()


def test_v2_success_and_dominant_reward_require_directive_completion() -> None:
    env = GhostlineEnvV2(
        seed=33,
        tier=1,
        directive=ContractDirective.GHOST,
    )
    env.reset(seed=33)
    env.sim.level.guards = []
    env.sim.level.cameras = []
    env.sim.drones = []
    env.sim.data = env.sim.level.quota
    env.sim.max_trace = 80.0
    env.sim.trace = 80.0
    env.sim.player = env.sim.level.extraction.copy()

    _, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated and env.sim.extracted
    assert not info["directive_success"]
    assert not info["is_success"]
    assert info["fail_reason"] == "directive_incomplete"
    assert info["reward_extraction"] == pytest.approx(4.0)
    assert info["reward_failure"] == pytest.approx(-2.0)
    assert info["reward_total"] == pytest.approx(
        sum(info["reward_components"].values())
    )
    assert reward == pytest.approx(info["reward_total"])
    env.close()


def test_v2_replay_is_deterministic_for_seed_tier_directive_and_actions() -> None:
    actions = [RunnerActionV2.decode(value) for value in (1, 10, 46, 3, 21, 0, 71, 8) * 4]
    first = GhostlineSimulationV2(seed=701, tier=6, directive="speed")
    second = GhostlineSimulationV2(seed=701, tier=6, directive="speed")
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
    sim = GhostlineSimulationV2(seed=51, tier=5)
    sim.level.guards[0].position = sim.player + np.asarray((32.0, 0.0), dtype=np.float32)
    sim.events.clear()
    before = sim.decoy_charges
    sim.advance(RunnerActionV2(move=1, decoy=True), ticks=6)
    assert sim.decoy_charges == before - 1
    assert sim.decoys_used == 1
    assert len(sim.decoys) == 1
    assert [event.kind for event in sim.events].count("decoy_deployed") == 1
    assert any(state.heard_confidence > 0.0 for state in sim.operative_states.values())
    sim.advance(RunnerActionV2(move=1, decoy=True), ticks=6)
    assert sim.decoys_used == 1


def test_security_doors_only_use_redundant_edges_and_are_telegraphed() -> None:
    sim = GhostlineSimulationV2(seed=93, tier=6)
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
    sim = GhostlineSimulationV2(seed=94, tier=6)
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
    sim = GhostlineSimulationV2(seed=95, tier=6, external_security=True)
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
    sim = GhostlineSimulationV2(seed=96, tier=6, external_security=True)
    guard = sim.level.guards[0]
    sim.set_security_orders(
        {
            guard.guard_id: SecurityOrder(
                SecurityIntent.FLANK_LEFT,
                np.asarray((sim.level.world_width + 96.0, -64.0), dtype=np.float32),
            )
        }
    )
    sim.advance(RunnerActionV2(), ticks=24)
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


def test_security_fingerprint_payload_is_checkout_line_ending_invariant(tmp_path) -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "ghostline"
    lf_root = tmp_path / "lf"
    crlf_root = tmp_path / "crlf"
    lf_root.mkdir()
    crlf_root.mkdir()
    for name in (
        "config_v2.py",
        "types_v2.py",
        "simulation_v2.py",
        "security_baselines.py",
        "security_env.py",
        "security_types.py",
    ):
        payload = (source / name).read_bytes().replace(b"\r\n", b"\n")
        (lf_root / name).write_bytes(payload)
        (crlf_root / name).write_bytes(payload.replace(b"\n", b"\r\n"))

    assert _canonical_security_source_digest(lf_root) == _canonical_security_source_digest(crlf_root)


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
    shaping_values = []
    for agent, reward in rewards.items():
        components = infos[agent]["reward_components"]
        shared = components["total"]
        shaping = infos[agent]["agent_shaping"]
        shaping_values.append(shaping)
        # The shared team components still account for themselves exactly, and
        # each operative's reward is that shared total plus only its own
        # containment shaping. Nothing else may leak into the per-agent reward.
        assert sum(value for key, value in components.items() if key != "total") == pytest.approx(shared)
        assert shared + shaping == pytest.approx(reward)
    # Credit assignment is the point: operatives in different positions must not
    # all receive the same number.
    assert len(set(round(value, 9) for value in shaping_values)) > 1
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
    selected = train_security(
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
    assert selected.name == "last-policy.pt"
    assert selected.exists()
    assert not (tmp_path / "security-smoke" / "champion.pt").exists()
    assert load_security_policy(selected).recurrent_size == 256
    assert (tmp_path / "security-smoke" / "behavior-warmup.json").is_file()


def test_security_dry_run_validates_opponent_and_freezes_manifest(
    tmp_path,
) -> None:
    output = tmp_path / "security-preflight"
    published = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "ghostline-policy.pt"
    )
    manifest_path = train_security(
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
        dry_run=True,
        device="cpu",
        runner_checkpoint=published,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_path.name == "experiment-manifest.json"
    assert manifest["manifest_contract"] == (
        SECURITY_EXPERIMENT_MANIFEST_CONTRACT
    )
    assert manifest["status"] == "preflight-passed"
    assert manifest["observation_contract"] == (
        "GhostlineSecurityParallel-v2"
    )
    assert len(manifest["environment_fingerprint"]) == 64
    assert manifest["runner_opponents"][0]["kind"] == "published-v1"
    assert len(manifest["runner_opponents"][0]["sha256"]) == 64
    assert manifest["seed_namespaces"][
        "final_test_not_consumed_by_training"
    ]
    assert not (output / "latest.pt").exists()
    assert not (output / "champion.pt").exists()


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


def test_security_final_slice_is_consumed_exactly_once(tmp_path) -> None:
    model = tmp_path / "security.pt"
    save_security_policy(
        SharedSecurityActorCritic(recurrent_size=256),
        model,
        purpose="one-way-final-slice-test",
    )
    manifest = tmp_path / "slices.json"
    fingerprint = security_environment_fingerprint()
    manifest.write_text(
        json.dumps(
            {
                "manifest_contract": "ghostline-final-test-slices-v1",
                "observation_contract": "GhostlineSecurityParallel-v2",
                "environment_fingerprint": fingerprint,
                "slices": [
                    {
                        "environment_fingerprint": fingerprint,
                        "episodes_per_tier": 1,
                        "policy_kind": "security-v2-neural-vs-fair-scripted",
                        "seed_start": 14_000_000,
                        "status": "reserved_unopened",
                        "tiers": [3],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "final.json"
    evaluate_security_checkpoint(
        model=model,
        output=output,
        tiers="3",
        episodes_per_tier=1,
        seed_start=14_000_000,
        device="cpu",
        slice_manifest=manifest,
    )
    consumed = json.loads(manifest.read_text(encoding="utf-8"))["slices"][0]
    assert consumed["status"] == "consumed"
    assert consumed["result"]["meets_acceptance_thresholds"] is None
    assert len(consumed["result"]["outputs"]) == 3
    with pytest.raises(RuntimeError, match="reserved_unopened"):
        evaluate_security_checkpoint(
            model=model,
            output=tmp_path / "second.json",
            tiers="3",
            episodes_per_tier=1,
            seed_start=14_000_000,
            device="cpu",
            slice_manifest=manifest,
        )


def test_security_sight_matches_the_simulation_detection_envelope(monkeypatch) -> None:
    """The observation's ``visible`` flag must agree with actual detection.

    A hardcoded 245 px / cos 0.45 cone reported the runner visible when the
    guard could not detect it, and the error inverted at high alert where the
    real envelope grows past 245 px. ``intent_mask[PURSUE]`` is gated on this
    flag, so the legal action set has to track the detection model.
    """

    from ghostline.config import (
        GUARD_VISION_BASE_DISTANCE,
        GUARD_VISION_COSINE,
        GUARD_VISION_DISTANCE_PER_ALERT,
    )

    env = GhostlineSecurityParallelEnv(tier=6, seed=20_000_101)
    env.reset(seed=20_000_101)
    guard = env.sim.level.guards[0]
    captured: list[dict[str, float]] = []
    real_visible = env.sim.visible

    def tracked(origin, facing, target, **kwargs):
        captured.append(dict(kwargs))
        return real_visible(origin, facing, target, **kwargs)

    env.sim.visible = tracked
    for alert in (0, 4):
        captured.clear()
        monkeypatch.setattr(type(env.sim), "alert_tier", property(lambda _self, value=alert: value))
        env._runner_contact(guard)
        assert captured, "no sight query issued"
        contract = captured[0]
        assert contract["cosine"] == GUARD_VISION_COSINE
        assert contract["distance"] == GUARD_VISION_BASE_DISTANCE + GUARD_VISION_DISTANCE_PER_ALERT * alert
    env.close()


def test_security_audio_estimate_is_coarse_and_not_invertible() -> None:
    """Heard contacts must not resolve to the exact runner position."""

    import numpy as np

    from ghostline.config import TILE_SIZE

    env = GhostlineSecurityParallelEnv(tier=6, seed=20_000_102)
    env.reset(seed=20_000_102)
    guard = env.sim.level.guards[0]

    offsets = set()
    exact_hits = 0
    for step in range(24):
        env.sim.player = guard.position + np.asarray((90.0 + step * 3.0, 40.0 - step * 2.0), dtype=np.float32)
        estimate = env._quantized_audio_estimate(guard)
        delta = estimate - env.sim.player
        offsets.add((round(float(delta[0]), 3), round(float(delta[1]), 3)))
        if float(np.linalg.norm(delta)) < 1e-6:
            exact_hits += 1

    # A constant per-guard offset would produce exactly one distinct delta and
    # would be trivially invertible back to the true position.
    assert len(offsets) > 1
    assert exact_hits == 0
    # Quantisation stays bounded: it is a coarse cue, not random noise.
    worst = max(max(abs(dx), abs(dy)) for dx, dy in offsets)
    assert worst <= TILE_SIZE * 4
    env.close()


def test_security_targets_distinguish_extraction_from_doors() -> None:
    """Every tactical slot carries its own kind code."""

    import numpy as np

    from ghostline.config_v2 import SECURITY_TARGET_FEATURES, SECURITY_TARGET_KINDS
    from ghostline.security_env import TargetKind
    from ghostline.security_types import TargetKind as RuntimeTargetKind

    env = GhostlineSecurityParallelEnv(tier=6, seed=20_000_103)
    observations, _ = env.reset(seed=20_000_103)
    observation = observations[next(iter(observations))]
    targets = observation["targets"]

    assert targets.shape == (10, SECURITY_TARGET_FEATURES)
    assert len(TargetKind) == SECURITY_TARGET_KINDS
    assert [(item.name, item.value) for item in RuntimeTargetKind] == [
        (item.name, item.value) for item in TargetKind
    ]
    kinds = [int(np.argmax(row[3:])) for row in targets]
    assert len(set(kinds[:8])) == 8, "fixed semantic slots share a target-kind code"
    # Escape routes keep their semantic identity even before an operative has
    # a credible contact. Their legality mask, rather than a fake contact,
    # controls whether PINCER/SEAL may select them.
    route_slots = [
        kinds[index]
        for index in range(8, len(kinds))
        if np.any(targets[index, 3:])
    ]
    assert route_slots
    assert route_slots == [int(TargetKind.ESCAPE_ROUTE)] * len(route_slots)
    assert kinds[int(TargetKind.EXTRACTION)] != kinds[int(TargetKind.DOOR)]
    env.close()


def test_central_critic_state_carries_the_mission_clock() -> None:
    """Timer expiry is a scoring outcome, so the critic must see the clock."""

    import numpy as np

    from ghostline.config_v2 import SECURITY_CENTRAL_STATE_SIZE

    env = GhostlineSecurityParallelEnv(tier=6, seed=20_000_104)
    env.reset(seed=20_000_104)
    early = env.state()
    assert early.shape == (SECURITY_CENTRAL_STATE_SIZE,)

    env.sim.elapsed_ticks += 60 * 100
    late = env.state()
    assert not np.array_equal(early, late), "state is blind to elapsed mission time"

    # Missing operatives are marked by an explicit presence mask rather than a
    # -1.0 sentinel that a real operative at the world origin could produce.
    present = env.state()[-5:]
    live = {guard.guard_id for guard in env.sim.level.guards}
    for guard_id in range(5):
        assert (present[guard_id] > 0.0) == (guard_id in live)
    env.close()


def test_interception_beats_tailing_in_the_shaping_potential() -> None:
    """Being ahead of the runner must score above trailing it.

    Guards move at 95-99% of runner speed, so a tail chase can never close.
    The old potential rewarded raw proximity, which trained exactly that.
    """

    import numpy as np

    env = GhostlineSecurityParallelEnv(tier=6, seed=20_000_105)
    env.reset(seed=20_000_105)
    guard = env.sim.level.guards[0]
    cutoff = env.sim.escape_route_cutoffs(env.sim.player, limit=1)[0]
    direction = cutoff - env.sim.player
    direction = direction / max(1e-6, float(np.linalg.norm(direction)))

    guard.position = (env.sim.player - direction * 40.0).astype(np.float32)
    trailing = env._route_score(guard, cutoff)
    guard.position = (env.sim.player + direction * 120.0).astype(np.float32)
    intercepting = env._route_score(guard, cutoff)

    assert intercepting > trailing
    env.close()


def test_crouch_trades_speed_for_silence_and_a_smaller_profile() -> None:
    """Quiet play must be a real option, not a slower version of the same run."""

    from ghostline.config_v2 import (
        CROUCH_AWARENESS_SCALE,
        CROUCH_FOOTSTEP_RADIUS,
        CROUCH_SPEED_SCALE,
        WALK_FOOTSTEP_RADIUS,
    )

    def travel(crouch: bool) -> float:
        sim = GhostlineSimulationV2(seed=4242, tier=6)
        start = sim.player.copy()
        for _ in range(30):
            sim.advance(RunnerActionV2(move=3, crouch=crouch), ticks=1)
        return float(norm(sim.player - start))

    walked, crouched = travel(False), travel(True)
    assert crouched == pytest.approx(walked * CROUCH_SPEED_SCALE, rel=0.05)

    def footstep_radii(crouch: bool) -> set[float]:
        sim = GhostlineSimulationV2(seed=4242, tier=6)
        seen: set[float] = set()
        real = GhostlineSimulationV2._broadcast_noise

        def capture(**kwargs):
            seen.add(kwargs["radius"])
            return real(sim, **kwargs)

        sim._broadcast_noise = capture
        for _ in range(120):
            sim.advance(RunnerActionV2(move=3, crouch=crouch), ticks=1)
        return seen

    assert footstep_radii(False) == {WALK_FOOTSTEP_RADIUS}
    assert footstep_radii(True) == {CROUCH_FOOTSTEP_RADIUS}
    assert CROUCH_FOOTSTEP_RADIUS < WALK_FOOTSTEP_RADIUS
    assert 0.0 < CROUCH_AWARENESS_SCALE < 1.0


def test_crouch_slows_awareness_without_making_the_runner_invisible(monkeypatch) -> None:
    """Patience buys safety, but the guard still accumulates awareness."""

    from ghostline.config_v2 import CROUCH_AWARENESS_SCALE

    def awareness_after(crouch: bool) -> float:
        sim = GhostlineSimulationV2(seed=2_000_004, tier=6)
        for guard in sim.level.guards:
            guard.awareness = 0.0
            guard.position[:] = sim.player + np.asarray((32.0, 0.0), dtype=np.float32)
            guard.facing = np.pi
        sim.crouching = crouch
        monkeypatch.setattr(sim, "visible", lambda *_args, **_kwargs: True)
        sim._update_guards(1.0 / 60.0)
        return float(sim.level.guards[0].awareness)

    loud, quiet = awareness_after(False), awareness_after(True)
    assert quiet == pytest.approx(loud * CROUCH_AWARENESS_SCALE)
    assert quiet < loud, "crouching must slow how fast a guard fills its meter"
    assert quiet > 0.0, "crouching must never make the runner undetectable"


def test_dash_costs_trace_so_loud_routes_are_expensive() -> None:
    """Loud stays viable and fast, but it escalates the network."""

    def trace_after(dash: bool, crouch: bool = False) -> float:
        sim = GhostlineSimulationV2(seed=4242, tier=6)
        for _ in range(120):
            sim.advance(RunnerActionV2(move=3, dash=dash, crouch=crouch), ticks=1)
        return float(sim.trace)

    assert trace_after(True) > trace_after(False)
    # Unseen and quiet, the network actively cools back toward its floor.
    sim = GhostlineSimulationV2(seed=4242, tier=6)
    sim.trace = 60.0
    for _ in range(120):
        sim.advance(RunnerActionV2(move=3, crouch=True), ticks=1)
    assert sim.trace < 60.0


def test_crouch_cannot_be_combined_with_dash_in_the_action_mask() -> None:
    """Crouch is the quiet state; it may not silence a dash."""

    sim = GhostlineSimulationV2(seed=4242, tier=6)
    mask = sim.action_mask()
    for value, legal in enumerate(mask):
        action = RunnerActionV2.decode(value)
        if action.crouch and action.dash:
            assert legal == 0

    # Even if an illegal pair is forced through, the dash wins and stays loud.
    sim.advance(RunnerActionV2(move=3, dash=True, crouch=True), ticks=6)
    assert sim.crouching is False


def test_stealth_economy_makes_a_loud_mission_materially_expensive() -> None:
    """Exposure and detection must carry a real budget, not a rounding error.

    The previous gain-only trace term gave the entire stealth budget about 4.7%
    of a successful run, so playing loud for a whole mission cost less than
    spending 75 extra seconds. These are objective terms rather than
    potential-based shaping because the intent is to move the optimum, not to
    guide search toward the same one.
    """

    from ghostline.config import TRACE_MAX
    from ghostline.config_v2 import (
        DETECTION_COST,
        EXPOSURE_COST_PER_DECISION,
        QUIET_DATA_BONUS,
        QUIET_TRACE_CEILING,
    )

    decisions = 370
    loud_exposure = EXPOSURE_COST_PER_DECISION * decisions
    quiet_exposure = EXPOSURE_COST_PER_DECISION * decisions * (QUIET_TRACE_CEILING / TRACE_MAX) * 0.3
    positive_budget = 20.0 + 12.0

    # A fully hot mission must cost a meaningful slice of a successful run.
    assert loud_exposure / positive_budget > 0.08
    assert loud_exposure > quiet_exposure * 3

    # Being seen repeatedly has to hurt on its own, independent of trace level.
    assert DETECTION_COST * 60 > 3.0
    # ...but never so much that a successful loud extraction becomes pointless.
    assert loud_exposure + DETECTION_COST * 60 < positive_budget * 0.5
    assert QUIET_DATA_BONUS > 0.0


def test_exposure_scales_with_trace_and_quiet_data_earns_a_bonus() -> None:
    """The live terms track simulation state exactly."""

    from ghostline.config import TRACE_MAX
    from ghostline.config_v2 import EXPOSURE_COST_PER_DECISION, QUIET_TRACE_CEILING

    env = GhostlineEnvV2(seed=3_000_021, tier=6, directive="standard")
    env.reset(seed=3_000_021)
    env.sim.trace = TRACE_MAX
    env.step(RunnerActionV2(move=0).encode())
    hot = env.reward_components["exposure"]
    assert hot == pytest.approx(-EXPOSURE_COST_PER_DECISION, rel=0.02)

    env.reset(seed=3_000_021)
    env.sim.trace = 0.0
    env.step(RunnerActionV2(move=0).encode())
    cold = env.reward_components["exposure"]
    assert cold > hot, "a cold network must cost less than a hot one"
    assert QUIET_TRACE_CEILING < TRACE_MAX
    env.close()


def test_ghost_potential_exposes_irreversible_trace_and_damage_budgets() -> None:
    """Stealth shaping is immediate, bounded, and unavailable to other directives."""

    ghost = GhostlineEnvV2(seed=3_000_023, tier=3, directive="ghost")
    ghost.reset(seed=3_000_023)
    quiet = ghost._mission_potential()
    ghost.sim.max_trace = 75.0
    exhausted_trace = ghost._mission_potential()
    ghost.sim.integrity = 2
    damaged = ghost._mission_potential()

    assert quiet - exhausted_trace == pytest.approx(8.0)
    assert exhausted_trace - damaged == pytest.approx(2.0 / 3.0)

    standard = GhostlineEnvV2(
        seed=3_000_023,
        tier=3,
        directive="standard",
    )
    standard.reset(seed=3_000_023)
    before = standard._mission_potential()
    standard.sim.max_trace = 75.0
    standard.sim.integrity = 2
    assert standard._mission_potential() == pytest.approx(before)
    ghost.close()
    standard.close()


def test_ghost_behavior_cost_is_dense_and_cannot_be_farmed() -> None:
    """Dash and rising awareness cost reward; awareness recovery never pays."""

    from ghostline.config_v2 import (
        GHOST_AWARENESS_GAIN_COST,
        GHOST_DASH_COST_PER_DECISION,
    )
    from ghostline.env_v2 import ghost_stealth_behavior_cost

    cost = ghost_stealth_behavior_cost(
        dash=True,
        awareness_before=0.2,
        awareness_after=0.7,
    )
    assert cost == pytest.approx(
        -GHOST_DASH_COST_PER_DECISION - 0.5 * GHOST_AWARENESS_GAIN_COST
    )
    assert ghost_stealth_behavior_cost(
        dash=False,
        awareness_before=0.8,
        awareness_after=0.2,
    ) == pytest.approx(0.0)


def test_holding_cover_while_crouched_is_not_punished_as_idling() -> None:
    """Waiting out a patrol is a tactic, not stalling."""

    from ghostline.generation import tile_center
    from ghostline.types import Tile

    def cover_position(sim):
        """Find a walkable tile that actually sits against geometry."""
        grid = sim.level.grid
        for ty in range(1, grid.shape[0] - 1):
            for tx in range(1, grid.shape[1] - 1):
                if grid[ty, tx] == Tile.WALL or (tx, ty) in sim._blocked_tiles:
                    continue
                for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if grid[ty + oy, tx + ox] == Tile.WALL:
                        return tile_center((tx, ty))
        raise AssertionError("no cover tile in this facility")

    env = GhostlineEnvV2(seed=3_000_022, tier=6, directive="standard")
    env.reset(seed=3_000_022)
    spot = cover_position(env.sim)
    env.sim.player[:] = spot
    assert env.sim.in_cover

    env.step(RunnerActionV2(move=0, crouch=True).encode())
    crouched_idle = env.reward_components["idle"]

    env.reset(seed=3_000_022)
    env.sim.player[:] = spot
    env.step(RunnerActionV2(move=0, crouch=False).encode())
    standing_idle = env.reward_components["idle"]

    assert crouched_idle == 0.0
    assert standing_idle < 0.0
    # The time cost still applies, so cover cannot be farmed indefinitely.
    assert env.reward_components["time"] < 0.0
    env.close()


def test_v2_generator_reshapes_every_facility_and_stays_valid() -> None:
    """The multi-agent track gets varied layouts that still pass validation."""

    from ghostline.generation import LevelGenerator
    from ghostline.generation_v2 import FacilityLayoutV2

    base, shaped = LevelGenerator(), FacilityLayoutV2()
    reshaped = 0
    for seed in range(4_100_000, 4_100_030):
        tier = 3 + seed % 4
        original = base.generate(seed=seed, tier=tier)
        variant = shaped.generate(seed=seed, tier=tier)
        # Reuses the original validator: reachability, route redundancy, door
        # throats and every security clearance still hold.
        assert shaped.validate(variant), f"seed {seed} produced an invalid facility"
        assert len(variant.props) > len(original.props), "no interior structure added"
        if not np.array_equal(original.grid, variant.grid):
            reshaped += 1
    assert reshaped >= 25, f"only {reshaped}/30 facilities were reshaped"


def test_v2_generator_is_deterministic_for_a_seed() -> None:
    """Replay and evaluation both depend on this."""

    from ghostline.generation_v2 import FacilityLayoutV2

    first = FacilityLayoutV2().generate(seed=4_100_777, tier=6)
    second = FacilityLayoutV2().generate(seed=4_100_777, tier=6)
    assert np.array_equal(first.grid, second.grid)
    assert [(p.kind, p.tile_x, p.tile_y) for p in first.props] == [
        (p.kind, p.tile_x, p.tile_y) for p in second.props
    ]


def test_v2_simulation_uses_the_reshaped_generator_and_v2_does_not() -> None:
    """The frozen single-agent track must keep its original facilities."""

    from ghostline.generation import LevelGenerator
    from ghostline.generation_v2 import FacilityLayoutV2
    from ghostline.simulation import GhostlineSimulation

    classic = GhostlineSimulation(seed=4_100_042, tier=6)
    adaptive = GhostlineSimulationV2(seed=4_100_042, tier=6)
    assert type(classic.generator) is LevelGenerator
    assert isinstance(adaptive.generator, FacilityLayoutV2)
    assert not np.array_equal(classic.level.grid, adaptive.level.grid)

    # The swap survives a reset, which is the path training actually uses.
    adaptive.reset(seed=4_100_043, tier=5)
    assert isinstance(adaptive.generator, FacilityLayoutV2)


def test_v2_interior_structure_creates_cover_without_sealing_routes() -> None:
    """Aisles must produce cover and still leave the facility walkable."""

    from ghostline.generation import flood_fill, world_to_tile
    from ghostline.generation_v2 import FacilityLayoutV2

    level = FacilityLayoutV2().generate(seed=4_100_099, tier=6)
    blocked = {
        (p.tile_x + dx, p.tile_y + dy)
        for p in level.props
        if p.blocking
        for dx in range(p.width)
        for dy in range(p.height)
    }
    reachable = flood_fill(level.grid, world_to_tile(level.spawn), blocked)
    for terminal in level.terminals:
        assert world_to_tile(terminal.position) in reachable
    assert world_to_tile(level.extraction) in reachable

    # Structure exists in the room bodies, not only against the walls.
    structural = [p for p in level.props if p.kind in ("pillar", "partition")]
    assert structural, "no sightline-breaking structure was placed"


def test_v2_runner_policy_starts_uniform_over_legal_actions() -> None:
    """Orthogonal init with a near-zero policy head is the point of the change.

    The project previously used PyTorch defaults everywhere, so the policy began
    training already committed to whatever the random draw favoured.
    """

    import torch

    from ghostline.env_v2 import GhostlineEnvV2
    from ghostline.model_v2 import RunnerPolicyV2

    env = GhostlineEnvV2(seed=4_300_001, tier=6, directive="ghost")
    observation, _ = env.reset(seed=4_300_001)
    policy = RunnerPolicyV2(recurrent_size=256)
    tensors = {key: torch.as_tensor(value).unsqueeze(0) for key, value in observation.items()}
    logits, value, _hidden = policy.forward(tensors, None)

    mask = torch.as_tensor(observation["action_mask"]).bool()
    probabilities = torch.softmax(logits[0], dim=-1)
    assert float(probabilities[~mask].sum()) == pytest.approx(0.0, abs=1e-9)
    entropy = float(torch.distributions.Categorical(logits=logits[0]).entropy())
    assert entropy == pytest.approx(float(np.log(int(mask.sum()))), rel=0.02)
    assert value.shape == (1,)
    env.close()


def test_v2_runner_policy_respects_the_directive_and_sequence_resets() -> None:
    """FiLM conditioning must actually change the representation."""

    import torch

    from ghostline.env_v2 import GhostlineEnvV2
    from ghostline.model_v2 import RunnerPolicyV2

    env = GhostlineEnvV2(seed=4_300_002, tier=6, directive="ghost")
    observation, _ = env.reset(seed=4_300_002)
    policy = RunnerPolicyV2(recurrent_size=256)
    tensors = {key: torch.as_tensor(value).unsqueeze(0) for key, value in observation.items()}
    ghost = policy.encode(tensors)

    greed = dict(tensors)
    directive = torch.zeros_like(tensors["directive"])
    directive[0, 3] = 1.0
    greed["directive"] = directive
    assert not torch.allclose(ghost, policy.encode(greed)), "directive does not reach the representation"

    steps, batch = 3, 2
    sequence = {
        key: torch.as_tensor(np.stack([[value] * batch] * steps)) for key, value in observation.items()
    }
    resets = torch.zeros(steps, batch, dtype=torch.bool)
    resets[1, 0] = True
    logits, values, _hidden = policy.forward_sequence(sequence, None, resets)
    assert logits.shape == (steps, batch, 288)
    assert values.shape == (steps, batch)
    env.close()


def test_v2_runner_checkpoint_round_trips_and_rejects_wrong_contracts(tmp_path) -> None:
    from ghostline.model_v2 import RunnerPolicyV2, load_runner_v2, save_runner_v2

    path = tmp_path / "runner-v2.pt"
    policy = RunnerPolicyV2(recurrent_size=256)
    save_runner_v2(policy, path, note="unit")
    restored = load_runner_v2(path)
    assert restored.recurrent_size == 256

    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["action_count"] = 72
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="action count"):
        load_runner_v2(path)


def test_running_return_scale_tracks_target_magnitude() -> None:
    """Value normalisation is what stopped the critic chasing its own scale."""

    from ghostline.marl_train import RunningReturnScale

    scale = RunningReturnScale()
    scale.update(np.full(512, 20.0, dtype=np.float32))
    assert scale.sigma == pytest.approx(20.0, rel=0.05)

    small = RunningReturnScale()
    small.update(np.full(512, 0.5, dtype=np.float32))
    assert small.sigma < scale.sigma
    # Never zero, so the division in the value loss is always safe.
    assert RunningReturnScale().sigma > 0.0


def test_vents_move_the_runner_and_operatives_cannot_follow() -> None:
    """A vent is the answer to a sealed route, and a committed one."""

    from ghostline.config_v2 import VENT_TRANSIT_SECONDS
    from ghostline.generation import tile_center

    sim = GhostlineSimulationV2(seed=4_400_002, tier=6)
    assert sim.vents, "tier 6 should place a vent network"
    vent = sim.vents[0]
    sim.player[:] = tile_center(vent.tile)
    origin = sim.player.copy()

    assert sim.can_interact()
    sim.advance(RunnerActionV2(move=0, interact=True), ticks=1)
    assert sim.vent_transit == pytest.approx(VENT_TRANSIT_SECONDS, rel=0.05)

    # Mid-transit the runner is frozen and cannot be steered.
    sim.advance(RunnerActionV2(move=3), ticks=6)
    assert float(norm(sim.velocity)) == 0.0
    assert float(norm(sim.player - origin)) == pytest.approx(0.0, abs=1e-6)

    for _ in range(90):
        sim.advance(RunnerActionV2(move=0), ticks=1)
    assert float(norm(sim.player - vent.exit_position)) == pytest.approx(0.0, abs=1e-3)
    assert sim.vent_uses == 1

    # Operatives have no vent verb at all: the network is runner-only.
    assert not hasattr(sim.level.guards[0], "vent")


def test_hacking_disables_a_camera_and_spends_a_charge() -> None:
    sim = GhostlineSimulationV2(seed=4_400_003, tier=6)
    device = next(item for item in sim.hackable if item.kind == "camera")
    camera = next(item for item in sim.level.cameras if int(item.camera_id) == device.target_id)
    sim.player[:] = device.position
    charges = sim.hack_charges

    sim.advance(RunnerActionV2(move=0, interact=True), ticks=1)
    assert sim.hack_charges == charges - 1
    assert camera.disabled_for > 0.0
    assert sim.hacks_used == 1
    # A spent device goes on cooldown so one panel is not an infinite supply.
    assert device.cooldown > 0.0


def test_hacking_the_lights_shortens_sight_inside_that_room_only(monkeypatch) -> None:
    """Darkness is applied through the shared visibility predicate."""

    from ghostline.config_v2 import HACK_LIGHTS_VISION_SCALE

    sim = GhostlineSimulationV2(seed=4_400_003, tier=6)
    panel = next(item for item in sim.hackable if item.kind == "lights")
    sim.player[:] = panel.position
    sim.advance(RunnerActionV2(move=0, interact=True), ticks=1)
    assert panel.target_id in sim.darkened_rooms

    guard = sim.level.guards[0]
    guard.position[:] = panel.position
    guard.facing = 0.0
    monkeypatch.setattr(sim, "line_of_sight", lambda *_args, **_kwargs: True)
    assert sim.guard_vision_scale(guard) == pytest.approx(HACK_LIGHTS_VISION_SCALE)

    far = guard.position + np.asarray((190.0, 0.0), dtype=np.float32)
    assert not sim.visible(guard.position, guard.facing, far, distance=205.0, cosine=0.62)
    sim.darkened_rooms.clear()
    assert sim.visible(guard.position, guard.facing, far, distance=205.0, cosine=0.62)


def test_crouched_decoy_throw_is_shorter_than_a_standing_throw() -> None:
    """The quiet route trades reach for concealment here too."""

    from ghostline.config_v2 import DECOY_CROUCH_THROW_SCALE

    def throw(crouch: bool) -> float:
        sim = GhostlineSimulationV2(seed=4_400_010, tier=6)
        sim.decoy_charges = 3
        origin = sim.player.copy()
        sim.advance(RunnerActionV2(move=3, decoy=True, crouch=crouch), ticks=1)
        assert sim.decoys, "decoy was not deployed"
        return float(norm(sim.decoys[0].position - origin))

    standing, crouched = throw(False), throw(True)
    assert crouched <= standing
    assert 0.0 < DECOY_CROUCH_THROW_SCALE < 1.0


def test_predictive_seal_uses_an_explicit_public_cutoff() -> None:
    """The controller must select from public contact-derived cutoff targets."""

    from ghostline.config_v2 import CHOKEPOINT_MIN_RUNNER_DISTANCE
    from ghostline.generation import tile_center

    sim = GhostlineSimulationV2(seed=4_400_004, tier=6)
    cutoffs = sim.escape_route_cutoffs(sim.player.copy())
    assert cutoffs
    assert sim.request_predictive_seal(cutoffs[0])
    warned = [door for door in sim.security_doors if door.warning_remaining > 0.0 or door.locked]
    assert warned, "no door was asked to close"
    # It telegraphs before closing and is never on top of the runner.
    for door in warned:
        assert float(norm(tile_center(door.tile) - sim.player)) >= CHOKEPOINT_MIN_RUNNER_DISTANCE

    # Only redundant room-graph edges are ever eligible, so a seal can never
    # remove the last route between two rooms.
    by_tile = {door.tile: door for door in sim.level.doors}
    for security_door in sim.security_doors:
        source = by_tile[security_door.tile]
        assert sim._door_edge_is_redundant(source.room_a, source.room_b)


def test_hacking_a_door_forces_a_seal_back_open() -> None:
    """The runner always keeps an answer to a chokepoint."""

    sim = GhostlineSimulationV2(seed=4_400_004, tier=6)
    panel = next(item for item in sim.hackable if item.kind == "door")
    door = next(
        item
        for item in sim.security_doors
        if item.door_id == panel.target_id and item.tile == panel.target_tile
    )
    door.lock_remaining = 3.0
    sim._refresh_navigation_blocks()
    assert door.locked

    sim.player[:] = panel.position
    sim.advance(RunnerActionV2(move=0, interact=True), ticks=1)
    assert not door.locked
    assert door.tile not in sim._blocked_tiles


def test_pincer_assigns_distinct_approach_arcs_across_the_team() -> None:
    """The policy can split a team across distinct public escape cutoffs."""

    from ghostline.types_v2 import SecurityIntent, SecurityOrder

    sim = GhostlineSimulationV2(seed=4_400_004, tier=6, external_security=True)
    contact = sim.player.copy()
    cutoffs = sim.escape_route_cutoffs(contact, limit=len(sim.level.guards))
    assert len(cutoffs) >= 2
    sim.set_security_orders(
        {
            guard.guard_id: SecurityOrder(
                SecurityIntent.PINCER,
                cutoffs[min(index, len(cutoffs) - 1)].copy(),
            )
            for index, guard in enumerate(sim.level.guards)
        }
    )
    sim.advance(RunnerActionV2(move=0), ticks=1)

    stations = [sim.operative_states[guard.guard_id].current_order.target for guard in sim.level.guards]
    separations = [
        float(norm(first - second))
        for index, first in enumerate(stations)
        for second in stations[index + 1 :]
    ]
    assert len(stations) >= 3
    distinct = {
        (round(float(station[0]), 3), round(float(station[1]), 3))
        for station in stations
    }
    assert len(distinct) >= 2, "all operatives stacked on one cutoff"
    assert max(separations) > 20.0


def test_field_sensors_report_a_crossing_and_never_damage() -> None:
    """Non-lethal by construction: a sensor is information, not a weapon."""

    sim = GhostlineSimulationV2(seed=4_400_003, tier=6)
    guard = sim.level.guards[0]
    assert sim.deploy_field_sensor(guard)
    assert not sim.deploy_field_sensor(guard), "sensor charges are limited"

    integrity = sim.integrity
    sim.player[:] = guard.position
    for _ in range(90):
        sim.advance(RunnerActionV2(move=0), ticks=1)

    assert any(sensor.triggered for sensor in sim.field_sensors)
    assert sim.integrity == integrity, "a sensor must never cost integrity"
    assert any(state.heard_confidence > 0.5 for state in sim.operative_states.values())


def test_interact_is_masked_out_when_there_is_nothing_to_use() -> None:
    """The context-sensitive verb must never be legal in empty space."""

    from ghostline.generation import tile_center

    sim = GhostlineSimulationV2(seed=4_400_002, tier=6)
    # Somewhere with no vent and no panel in range.
    far = next(
        tile_center((x, y))
        for y in range(2, sim.level.grid.shape[0] - 2)
        for x in range(2, sim.level.grid.shape[1] - 2)
        if sim._can_occupy(tile_center((x, y)), 6.0)
        and all(norm(tile_center((x, y)) - d.position) > 200.0 for d in sim.hackable)
        and all((x, y) != v.tile for v in sim.vents)
    )
    sim.player[:] = far
    assert not sim.can_interact()
    mask = sim.action_mask()
    assert all(mask[value] == 0 for value in range(len(mask)) if RunnerActionV2.decode(value).interact)

    vent = sim.vents[0]
    sim.player[:] = tile_center(vent.tile)
    mask = sim.action_mask()
    assert any(mask[value] == 1 for value in range(len(mask)) if RunnerActionV2.decode(value).interact)
