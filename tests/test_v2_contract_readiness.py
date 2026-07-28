from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import ghostline
from ghostline.env import GhostlineEnv
from ghostline.env_v2 import GhostlineEnvV2
from ghostline.generation import tile_center
from ghostline.imitation import training_environment_fingerprint
from ghostline.model import load_policy
from ghostline.model_v2 import (
    RunnerPolicyV2,
    initialize_runner_v2_from_published_v1,
    save_runner_v2,
)
from ghostline.runner_opponents import (
    FrozenPublishedV1RunnerOpponent,
    FrozenRunnerV2Opponent,
    load_runner_opponent_policy,
    make_frozen_runner_opponent,
)
from ghostline.simulation_v2 import GhostlineSimulationV2
from ghostline.types import GuardMode
from ghostline.types_v2 import (
    RUNNER_ACTION_COUNT_V2,
    FieldSensor,
    RunnerActionV2,
    SecurityIntent,
    SecurityOrder,
)


PUBLISHED_V1_FINGERPRINT = (
    "521c449a8bd9a540977a918f5b094dd3aeff44cc579a55f75e22a74bab20e129"
)


def test_public_version_registry_and_published_fingerprint() -> None:
    import gymnasium as gym

    assert ghostline.GhostlineEnv.__name__ == "PublishedGhostlineEnvV1"
    assert ghostline.GhostlineEnvV2 is GhostlineEnvV2
    assert gym.spec("GhostlineEnv-v1").entry_point == (
        "ghostline.env_v1:PublishedGhostlineEnvV1"
    )
    assert gym.spec("GhostlineEnv-v2").entry_point == (
        "ghostline.env_v2:GhostlineEnvV2"
    )
    assert gym.spec("GhostlineLegacyEnv-v0").entry_point == (
        "ghostline.env:GhostlineEnvV1"
    )
    assert "GhostlineEnv-v3" not in gym.registry
    assert training_environment_fingerprint() == PUBLISHED_V1_FINGERPRINT

    published = gym.make("GhostlineEnv-v1", seed=7, tier=1)
    observation, info = published.reset(seed=7)
    assert published.action_space.n == 36
    assert observation["ego"].shape == (24,)
    assert info["contract"] == "GhostlineEnv-v1"
    published.close()


def test_vectorized_v2_geometry_observations_preserve_published_semantics() -> None:
    """The optimized v2 prefix must remain player-equivalent to frozen v1."""

    env = GhostlineEnvV2(seed=20_261_017, tier=6, directive="ghost")
    observation, _ = env.reset(seed=20_261_017)
    for _ in range(12):
        percepts = env._security_percepts()
        visible_positions = env._perceived_entity_positions(percepts)
        optimized_grid = env._local_grid(visible_positions)
        optimized_rays = env._rays(visible_positions)
        reference_grid = GhostlineEnv._local_grid(env, visible_positions)
        reference_rays = GhostlineEnv._rays(env, visible_positions)

        assert np.array_equal(optimized_grid[:8], reference_grid)
        assert np.allclose(
            optimized_rays[:, :3],
            reference_rays,
            atol=1e-7,
            rtol=0.0,
        )

        legal = np.flatnonzero(observation["action_mask"])
        action = int(legal[(_ * 17 + 3) % len(legal)])
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()


def test_v2_terminal_info_exposes_an_exact_named_reward_ledger() -> None:
    env = GhostlineEnvV2(seed=20_261_018, tier=1)
    env.reset(seed=20_261_018)
    env.sim.truncated = True
    env.sim.fail_reason = "clock_expired"

    _, _, terminated, truncated, info = env.step(0)

    assert not terminated and truncated
    assert set(info["reward_components"]) == set(env.reward_components)
    assert info["reward_total"] == pytest.approx(
        sum(info["reward_components"].values())
    )
    env.close()


def test_wrapper_executes_every_legal_v2_action_without_aliasing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = GhostlineEnvV2(seed=81, tier=6)
    env.reset(seed=81)
    all_legal = np.ones(RUNNER_ACTION_COUNT_V2, dtype=np.int8)
    executed: list[RunnerActionV2] = []
    monkeypatch.setattr(env.sim, "action_mask", lambda: all_legal.copy())
    monkeypatch.setattr(
        env.sim,
        "advance",
        lambda action, *, ticks: executed.append(action),
    )
    monkeypatch.setattr(env, "_observation", lambda: {})

    for value in range(RUNNER_ACTION_COUNT_V2):
        _observation, _reward, terminated, truncated, _info = env.step(value)
        assert not terminated and not truncated
        assert executed[-1].encode() == value
        assert env.reward_components["invalid"] == pytest.approx(0.0)
    env.close()


def test_v2_runner_can_warm_start_from_immutable_published_v1() -> None:
    checkpoint = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "ghostline-policy.pt"
    )
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy = RunnerPolicyV2(recurrent_size=384)
    report = initialize_runner_v2_from_published_v1(policy, checkpoint)

    assert report["source_environment_fingerprint"] == PUBLISHED_V1_FINGERPRINT
    assert report["target_observation_contract"] == "GhostlineEnv-v2"
    assert torch.equal(
        policy.local_encoder[0].weight[:, :8],
        source["model"]["local_encoder.0.weight"],
    )
    assert torch.count_nonzero(policy.local_encoder[0].weight[:, 8:]) == 0
    assert torch.equal(
        policy.base_action_head.weight[0],
        source["model"]["action_head.weight"][0],
    )
    assert torch.count_nonzero(policy.decoy_head.weight) == 0
    assert float(
        (policy.decoy_head.bias[1] - policy.decoy_head.bias[0]).detach()
    ) == pytest.approx(-1.25)

    env = GhostlineEnvV2(seed=12_345, tier=6, directive="ghost")
    observation, _ = env.reset(seed=12_345)
    published_observation = {
        "ego": observation["ego"][:24],
        "objective": observation["objective"],
        "local_grid": observation["local_grid"][:8],
        "targets": observation["targets"],
        "target_mask": observation["target_mask"],
        "entities": observation["entities"][:, :13],
        "entity_mask": observation["entity_mask"],
        "rays": observation["rays"][:, :3],
        "action_mask": observation["action_mask"][:36],
    }
    published = load_policy(checkpoint)
    with torch.no_grad():
        published_logits, published_value, published_hidden = published(
            {
                key: torch.as_tensor(value).unsqueeze(0)
                for key, value in published_observation.items()
            }
        )
        v2_logits, v2_value, v2_hidden = policy(
            {
                key: torch.as_tensor(value).unsqueeze(0)
                for key, value in observation.items()
            }
        )
    assert torch.allclose(v2_logits[:, :36], published_logits, atol=1e-5)
    assert torch.allclose(v2_value, published_value * 0.05, atol=1e-5)
    assert torch.allclose(v2_hidden, published_hidden, atol=1e-5)
    env.close()

    incompatible = RunnerPolicyV2(recurrent_size=256)
    with pytest.raises(RuntimeError, match="recurrent widths"):
        initialize_runner_v2_from_published_v1(incompatible, checkpoint)


def test_v2_runner_action_head_shares_semantics_but_keeps_full_residual_capacity() -> None:
    policy = RunnerPolicyV2(recurrent_size=256)
    latent = torch.zeros(2, 256)
    mask = torch.ones(2, RUNNER_ACTION_COUNT_V2, dtype=torch.int8)
    with torch.no_grad():
        policy.action_residual_head.weight.zero_()
        policy.action_residual_head.bias.zero_()
        policy.decoy_head.weight.zero_()
        policy.decoy_head.bias.copy_(torch.tensor((0.0, 0.75)))
        decoded = policy.policy_decoder(latent)
        logits = policy.action_logits(latent, mask)
    assert torch.allclose(
        logits[:, 36] - logits[:, 0],
        torch.full((2,), 0.75),
    )
    assert decoded.shape == (2, 256)


def test_v2_runner_potential_is_discount_matched_and_zero_at_terminal() -> None:
    from ghostline.env import PROGRESS_POTENTIAL_SCALE
    from ghostline.env_v2 import runner_potential_progress_reward

    previous, current, gamma = 0.8, 1.4, 0.999
    ordinary = runner_potential_progress_reward(
        previous,
        current,
        gamma=gamma,
    )
    terminal = runner_potential_progress_reward(
        previous,
        current,
        gamma=gamma,
        terminal=True,
    )
    assert ordinary == pytest.approx(
        PROGRESS_POTENTIAL_SCALE * (gamma * current - previous)
    )
    assert terminal == pytest.approx(-PROGRESS_POTENTIAL_SCALE * previous)


def test_security_can_use_a_native_v2_runner_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runner-v2.pt"
    save_runner_v2(RunnerPolicyV2(recurrent_size=256), checkpoint)
    policy, kind = load_runner_opponent_policy(checkpoint)
    assert kind == "runner-v2"

    sim = GhostlineSimulationV2(
        seed=44_001,
        tier=6,
        directive="ghost",
        external_security=True,
    )
    opponent = FrozenRunnerV2Opponent(policy)
    action = opponent(sim)
    assert 0 <= action < RUNNER_ACTION_COUNT_V2
    assert opponent.hidden is not None
    assert opponent.env is not None
    assert opponent.env.sim is sim
    opponent.close()


def test_security_can_reuse_the_published_v1_runner_in_an_opponent_pool() -> None:
    checkpoint = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "ghostline-policy.pt"
    )
    policy, kind = load_runner_opponent_policy(checkpoint)
    assert kind == "published-v1"
    opponent = make_frozen_runner_opponent(
        policy,
        opponent_id="published-v1:test",
    )
    assert isinstance(opponent, FrozenPublishedV1RunnerOpponent)
    assert opponent.opponent_id == "published-v1:test"
    action = opponent(
        GhostlineSimulationV2(
            seed=44_002,
            tier=6,
            external_security=True,
        )
    )
    assert 0 <= action < 36
    opponent.close()


def test_interact_action_reaches_a_vent_through_the_gym_wrapper() -> None:
    env = GhostlineEnvV2(seed=93, tier=2)
    env.reset(seed=93)
    vent = env.sim.vents[0]
    env.sim.player[:] = tile_center(vent.tile)
    action = RunnerActionV2(interact=True).encode()
    assert env.sim.action_mask()[action] == 1

    _observation, _reward, terminated, truncated, _info = env.step(action)

    assert not terminated and not truncated
    assert env._action_history[-1] == action
    assert env.sim.vent_transit > 0.0
    assert env.sim.vent_uses == 1
    env.close()


def test_vent_transit_advances_the_whole_world_and_blocks_damage() -> None:
    sim = GhostlineSimulationV2(seed=114, tier=6)
    vent = sim.vents[0]
    sim.player[:] = tile_center(vent.tile)
    sim._activate_interact()
    assert sim.vent_transit > 0.0
    before_ticks = sim.elapsed_ticks
    before_angle = sim.level.cameras[0].angle
    before_integrity = sim.integrity

    sim.advance(RunnerActionV2(), ticks=30)

    assert sim.elapsed_ticks == before_ticks + 30
    assert sim.level.cameras[0].angle != pytest.approx(before_angle)
    sim._damage(sim.player.copy(), source_kind="guard")
    assert sim.integrity == before_integrity


def test_crouch_awareness_is_scaled_before_detection_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim = GhostlineSimulationV2(seed=151, tier=3)
    guard = sim.level.guards[0]
    guard.awareness = 0.97
    guard.mode = GuardMode.SUSPICIOUS
    sim.crouching = True
    before = sim.detections
    monkeypatch.setattr(sim, "visible", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(sim, "_move_agent", lambda *_args, **_kwargs: None)

    sim._update_guards(1.0 / 60.0)

    assert guard.awareness < 1.0
    assert sim.detections == before
    assert guard.mode != GuardMode.CHASE


def test_hold_order_cannot_farm_repeat_detections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sim = GhostlineSimulationV2(seed=152, tier=6, external_security=True)
    guard = sim.level.guards[0]
    guard.mode = GuardMode.CHASE
    guard.awareness = 1.0
    monkeypatch.setattr(sim, "visible", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(sim, "_move_agent", lambda *_args, **_kwargs: None)
    before = sim.detections

    for _ in range(5):
        sim.set_security_orders(
            {
                guard.guard_id: SecurityOrder(
                    SecurityIntent.HOLD,
                    sim.player.copy(),
                )
            }
        )
        sim._update_guards(1.0 / 60.0)

    assert sim.detections == before
    assert guard.mode == GuardMode.CHASE


def test_field_sensor_requires_line_of_sight_and_is_not_globally_observed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = GhostlineEnvV2(seed=190, tier=6)
    env.reset(seed=190)
    sensor = FieldSensor(
        sensor_id=999,
        owner_id=0,
        position=env.sim.player + np.asarray((16.0, 0.0), dtype=np.float32),
        armed_in=0.0,
        lifetime=10.0,
    )
    env.sim.field_sensors.append(sensor)
    monkeypatch.setattr(env.sim, "line_of_sight", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(env.sim, "player_can_see", lambda *_args, **_kwargs: False)

    env.sim._update_field_sensors(1.0 / 60.0)
    observation = env._observation()

    assert not sensor.triggered
    assert not np.any(observation["local_grid"][13])
    assert not np.any(
        observation["field_targets"][observation["field_target_mask"] > 0, 4] > 0
    )
    env.close()


def test_door_hack_opens_only_its_bound_security_door() -> None:
    sim = GhostlineSimulationV2(seed=245, tier=6)
    panel = next(device for device in sim.hackable if device.kind == "door")
    target = next(
        door
        for door in sim.security_doors
        if door.door_id == panel.target_id and door.tile == panel.target_tile
    )
    for door in sim.security_doors:
        door.lock_remaining = 3.0
    sim._refresh_navigation_blocks()
    sim.player[:] = panel.position
    sim._activate_interact()

    assert not target.locked
    assert target.forced_open_remaining > 0.0
    assert all(
        door.locked
        for door in sim.security_doors
        if door.door_id != target.door_id
    )
