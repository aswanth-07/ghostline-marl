"""Long-run safety regressions for the multi-agent v2 security track.

These tests deliberately exercise the public PettingZoo contract and the small
pure helpers used by MAPPO.  They are not campaign-quality checks; their job is
to stop a long run from starting with an unreachable semantic action, a
farmable reward, padded-agent statistics, or an incomplete resume checkpoint.
"""

from __future__ import annotations

import copy
from pathlib import Path
import random
import shutil

import numpy as np
import pytest
import torch

from ghostline import marl_train
from ghostline.config_v2 import (
    MAX_SECURITY_TARGETS,
    SECURITY_INTENT_COUNT,
)
from ghostline.security_env import GhostlineSecurityParallelEnv, TargetKind
from ghostline.security_model import (
    SECURITY_ACTION_SIZES,
    _canonical_security_source_digest,
    SharedSecurityActorCritic,
    factorized_log_prob,
    select_factorized_actions,
)
from ghostline.types import GuardMode
from ghostline.types_v2 import (
    GuardRole,
    RadioMessage,
    SecurityIntent,
)


@pytest.fixture
def preserve_process_rng_state():
    """Keep this resume-contract test isolated from the rest of the suite."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _stationary_runner(_simulation) -> int:
    return 0


def _legal_action(
    observation: dict[str, np.ndarray],
    *,
    intent: SecurityIntent = SecurityIntent.PATROL,
    ability: int = 0,
    target_kind: TargetKind | None = None,
) -> np.ndarray:
    """Construct one legal semantic action without assuming a target layout."""

    targets = np.flatnonzero(observation["target_mask"])
    assert len(targets), "every operative observation needs a fallback target"
    if target_kind is not None:
        targets = np.asarray(
            [
                index
                for index in targets
                if int(np.argmax(observation["targets"][index, 3:]))
                == int(target_kind)
            ],
            dtype=np.int64,
        )
        assert len(targets), f"no legal {target_kind.name.lower()} target row"
    return np.asarray(
        (
            int(intent),
            int(targets[0]),
            int(RadioMessage.NONE),
            int(ability),
        ),
        dtype=np.int64,
    )


def _refresh_observations(
    env: GhostlineSecurityParallelEnv,
) -> dict[str, dict[str, np.ndarray]]:
    observations = {agent: env._observation(agent) for agent in env.agents}
    env._current_observations = observations
    return observations


def test_security_contract_exposes_all_ten_intents_and_coordinated_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PINCER and SEAL must be learnable, not simulation-only enum values."""

    assert [int(intent) for intent in SecurityIntent] == list(range(10))
    assert SECURITY_INTENT_COUNT == len(SecurityIntent)
    assert SECURITY_ACTION_SIZES == (
        len(SecurityIntent),
        MAX_SECURITY_TARGETS,
        len(RadioMessage),
        2,
    )

    env = GhostlineSecurityParallelEnv(
        tier=6,
        seed=20_600_011,
        runner=_stationary_runner,
    )
    observations, _ = env.reset(seed=20_600_011)
    agent = env.agents[0]
    assert tuple(env.action_space(agent).nvec) == SECURITY_ACTION_SIZES
    assert observations[agent]["intent_mask"].shape == (len(SecurityIntent),)

    # Make the policy-visible contact deterministic.  This avoids depending on
    # the procedural spawn orientation while still exercising the real mask and
    # semantic-action decoder.
    contact = env.sim.player.copy()
    monkeypatch.setattr(
        env,
        "_runner_contact",
        lambda _guard: (
            contact.copy(),
            True,
            False,
            1.0,
            contact.copy(),
        ),
    )
    env.sim.security_door_cooldown = 0.0
    observations = _refresh_observations(env)
    mask = observations[agent]["intent_mask"]
    assert mask[int(SecurityIntent.PINCER)] == 1
    assert mask[int(SecurityIntent.SEAL)] == int(bool(env.sim.security_doors))

    action = _legal_action(
        observations[agent],
        intent=SecurityIntent.PINCER,
        target_kind=TargetKind.ESCAPE_ROUTE,
    )
    orders, invalid = env.orders_from_actions(
        {agent: action},
        observations=observations,
    )
    guard_id = env.agent_name_mapping[agent]
    assert invalid == 0
    assert orders[guard_id].intent is SecurityIntent.PINCER
    env.close()


def test_non_suppressor_sensor_ability_is_legal_and_consumes_its_charge() -> None:
    """The shared ability factor must not be dead for patrol/interceptor roles."""

    env = GhostlineSecurityParallelEnv(
        tier=6,
        seed=20_600_013,
        runner=_stationary_runner,
    )
    observations, _ = env.reset(seed=20_600_013)
    agent = next(
        name
        for name in env.agents
        if env.sim.operative_states[env.agent_name_mapping[name]].role
        is not GuardRole.SUPPRESSOR
    )
    guard_id = env.agent_name_mapping[agent]
    before_charges = env.sim.sensor_charges[guard_id]
    assert before_charges > 0
    assert observations[agent]["ability_mask"][1] == 1

    actions = {
        name: _legal_action(
            observation,
            ability=int(name == agent),
        )
        for name, observation in observations.items()
    }
    next_observations, _rewards, _terminated, _truncated, infos = env.step(actions)
    assert any(sensor.owner_id == guard_id for sensor in env.sim.field_sensors)
    assert env.sim.sensor_charges[guard_id] == before_charges - 1
    assert infos[agent]["invalid_actions"] == 0
    if env.agents:
        assert next_observations[agent]["ability_mask"][1] == 0
    env.close()


def test_intent_target_sampling_and_log_probability_use_the_same_joint_mask() -> None:
    """PINCER cannot sample a patrol row and execute a substituted cutoff."""

    env = GhostlineSecurityParallelEnv(
        tier=6,
        seed=20_600_015,
        runner=_stationary_runner,
    )
    observations, _ = env.reset(seed=20_600_015)
    agent = env.agents[0]
    guard_id = env.agent_name_mapping[agent]
    env.sim.operative_states[guard_id].heard_position = env.sim.player.copy()
    env.sim.operative_states[guard_id].heard_confidence = 1.0
    observation = env._observation(agent)
    route_targets = np.flatnonzero(
        observation["intent_target_mask"][int(SecurityIntent.PINCER)]
    )
    assert len(route_targets)

    logits = (
        torch.full((1, len(SecurityIntent)), -20.0),
        torch.full((1, MAX_SECURITY_TARGETS), -20.0),
        torch.zeros((1, len(RadioMessage))),
        torch.zeros((1, 2)),
    )
    logits[0][0, int(SecurityIntent.PINCER)] = 20.0
    logits[1][0, 0] = 40.0  # Globally preferred, but illegal for PINCER.
    logits[1][0, int(route_targets[0])] = 1.0
    joint_mask = torch.as_tensor(
        observation["intent_target_mask"][None, ...]
    )
    action = select_factorized_actions(
        logits,
        joint_mask,
        deterministic=True,
    )
    assert action[0, 0].item() == int(SecurityIntent.PINCER)
    assert action[0, 1].item() in route_targets
    log_probability, entropy = factorized_log_prob(
        logits,
        action,
        joint_mask,
    )
    assert torch.isfinite(log_probability).all()
    assert torch.isfinite(entropy).all()
    env.close()


def test_security_reset_boundary_chunking_matches_stepwise_gru() -> None:
    env = GhostlineSecurityParallelEnv(
        tier=6,
        seed=20_600_016,
        runner=_stationary_runner,
    )
    observations, _ = env.reset(seed=20_600_016)
    agents = env.agents[:2]
    policy = SharedSecurityActorCritic(recurrent_size=256)
    time_steps = 7
    sequence = {
        key: torch.as_tensor(
            np.stack(
                [
                    np.stack([observations[agent][key] for agent in agents])
                    for _ in range(time_steps)
                ]
            )
        )
        for key in marl_train.ACTOR_OBS_KEYS
    }
    resets = torch.zeros(time_steps, len(agents), dtype=torch.bool)
    resets[0] = True
    resets[3, 1] = True
    initial = torch.zeros(1, len(agents), 256)
    with torch.no_grad():
        sequence_logits, sequence_hidden = policy.forward_actor_sequence(
            sequence,
            initial,
            resets,
        )
        hidden = initial
        step_heads: list[list[torch.Tensor]] = [[], [], [], []]
        for index in range(time_steps):
            if resets[index].any():
                hidden = hidden.clone()
                hidden[:, resets[index], :] = 0.0
            heads, hidden = policy.forward_actor(
                {key: value[index] for key, value in sequence.items()},
                hidden,
            )
            for factor, head in enumerate(heads):
                step_heads[factor].append(head)
    for factor, head in enumerate(sequence_logits):
        assert torch.allclose(
            head,
            torch.stack(step_heads[factor]),
            atol=1e-6,
        )
    assert torch.allclose(sequence_hidden, hidden, atol=1e-6)
    env.close()


def test_repeated_hold_under_continuous_contact_cannot_farm_detection_reward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Orders may not repeatedly demote CHASE and re-trigger one sighting."""

    env = GhostlineSecurityParallelEnv(
        tier=6,
        seed=20_600_017,
        runner=_stationary_runner,
    )
    env.reset(seed=20_600_017)
    monkeypatch.setattr(env.sim, "visible", lambda *_args, **_kwargs: True)
    for guard in env.sim.level.guards:
        guard.awareness = 1.0
        guard.mode = GuardMode.CHASE
        guard.mode_seconds = 2.0

    detection_credits: list[float] = []
    for _ in range(4):
        observations = _refresh_observations(env)
        actions = {
            agent: _legal_action(observation, intent=SecurityIntent.HOLD)
            for agent, observation in observations.items()
        }
        _observations, _rewards, terminated, truncated, infos = env.step(actions)
        components = next(iter(infos.values()))["reward_components"]
        detection_credits.append(float(components["contact_acquisition"]))
        if any(terminated.values()) or any(truncated.values()):
            break

    assert len(detection_credits) >= 3
    # A transition tracker may award the first acquisition. Continuous contact
    # after that is one event, irrespective of how often HOLD is submitted.
    assert np.count_nonzero(np.asarray(detection_credits) > 0.0) <= 1
    assert detection_credits[1:] == pytest.approx([0.0] * (len(detection_credits) - 1))
    env.close()


def test_reward_ledger_is_exact_and_formation_cost_is_bounded() -> None:
    """Five stacked agents must not overwhelm the +/-20 terminal objective."""

    env = GhostlineSecurityParallelEnv(
        tier=6,
        seed=20_600_019,
        runner=_stationary_runner,
    )
    observations, _ = env.reset(seed=20_600_019)
    anchor = env.sim.level.guards[0].position.copy()
    for guard in env.sim.level.guards:
        guard.position[:] = anchor

    actions = {
        agent: _legal_action(observation)
        for agent, observation in observations.items()
    }
    _observations, rewards, _terminated, _truncated, infos = env.step(actions)
    components = next(iter(infos.values()))["reward_components"]
    assert components["formation"] >= -0.05
    assert components["total"] == pytest.approx(
        sum(value for name, value in components.items() if name != "total"),
        abs=1e-9,
    )
    for agent, reward in rewards.items():
        agent_components = infos[agent]["agent_reward_components"]
        assert agent_components["total"] == pytest.approx(
            agent_components["potential"]
            + agent_components["contact_credit"]
        )
        assert reward == pytest.approx(
            components["total"] + agent_components["total"],
            abs=1e-9,
        )
    env.close()


def test_potential_shaping_uses_the_same_discount_as_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PBRS is only policy invariant when its gamma matches PPO's gamma."""

    gamma = 0.987
    env = GhostlineSecurityParallelEnv(
        tier=6,
        seed=20_600_023,
        runner=_stationary_runner,
        reward_gamma=gamma,
    )
    observations, _ = env.reset(seed=20_600_023)
    before, after = 0.75, -0.25
    potentials = iter((before, after))
    monkeypatch.setattr(env, "_security_potential", lambda: next(potentials))
    monkeypatch.setattr(
        env,
        "_agent_potentials",
        lambda: {env.agent_name_mapping[agent]: 0.0 for agent in env.agents},
    )
    monkeypatch.setattr(env, "_formation_penalty", lambda: 0.0)
    actions = {
        agent: _legal_action(observation)
        for agent, observation in observations.items()
    }
    _observations, _rewards, _terminated, _truncated, infos = env.step(actions)
    components = next(iter(infos.values()))["reward_components"]
    assert components["potential"] == pytest.approx(gamma * after - before)
    env.close()


def test_active_advantage_normalization_ignores_and_zeros_padded_agents() -> None:
    """Inactive slots may not move the mean or variance seen by active actors."""

    normalize = getattr(marl_train, "_normalize_active_advantages", None)
    assert callable(normalize), "MAPPO needs an active-only advantage helper"

    advantages = np.asarray(
        [
            [[1.0, 1_000_000.0, -1_000_000.0, 50.0, -50.0]],
            [[3.0, -8_000_000.0, 8_000_000.0, -25.0, 25.0]],
        ],
        dtype=np.float32,
    )
    active = np.zeros_like(advantages, dtype=np.float32)
    active[:, :, 0] = 1.0
    normalized = normalize(advantages, active)

    assert normalized.shape == advantages.shape
    assert normalized[:, :, 0].reshape(-1) == pytest.approx((-1.0, 1.0))
    assert np.array_equal(normalized[active == 0], np.zeros(np.count_nonzero(active == 0)))

    changed_padding = advantages.copy()
    changed_padding[active == 0] *= 1000.0
    np.testing.assert_allclose(
        normalize(changed_padding, active)[active > 0],
        normalized[active > 0],
        atol=1e-7,
    )


def test_masked_value_loss_ignores_padded_agents_and_has_zero_padding_gradient() -> None:
    """A padded critic row must contribute neither loss nor gradient."""

    value_loss = getattr(marl_train, "_masked_value_loss", None)
    assert callable(value_loss), "MAPPO needs a masked clipped-value-loss helper"

    predicted = torch.tensor((0.5, -0.5, 999.0), requires_grad=True)
    old = torch.zeros(3)
    returns = torch.tensor((1.0, -1.0, -999.0))
    active = torch.tensor((1.0, 1.0, 0.0))
    loss = value_loss(predicted, old, returns, active, 0.2)
    # Both active rows clip from +/-0.5 to +/-0.2, giving max squared error
    # 0.8^2. PPO's value loss applies the conventional 0.5 multiplier.
    assert float(loss.detach()) == pytest.approx(0.5 * 0.8**2)
    loss.backward()
    assert predicted.grad is not None
    assert predicted.grad[2].item() == pytest.approx(0.0, abs=1e-12)

    changed = predicted.detach().clone()
    changed[2] = -1e9
    changed_returns = returns.clone()
    changed_returns[2] = 1e9
    assert float(value_loss(changed, old, changed_returns, active, 0.2)) == pytest.approx(
        float(loss.detach())
    )


def test_security_fingerprint_covers_inherited_and_v2_contract_sources(
    tmp_path: Path,
) -> None:
    """A mechanics/model edit must fail closed against stale checkpoints."""

    source = Path(__file__).resolve().parents[1] / "src" / "ghostline"
    copied = tmp_path / "ghostline"
    shutil.copytree(source, copied)
    required_sources = (
        "config.py",
        "config_v2.py",
        "types.py",
        "types_v2.py",
        "generation.py",
        "generation_v2.py",
        "simulation.py",
        "simulation_v2.py",
        "security_baselines.py",
        "security_env.py",
        "security_model.py",
    )
    baseline = _canonical_security_source_digest(copied)
    for name in required_sources:
        path = copied / name
        original = path.read_bytes()
        path.write_bytes(original + b"\n# fingerprint-contract-regression\n")
        assert _canonical_security_source_digest(copied) != baseline, (
            f"{name} can change without invalidating a security checkpoint"
        )
        path.write_bytes(original)
        assert _canonical_security_source_digest(copied) == baseline


def test_training_checkpoint_persists_exact_resume_boundary(
    tmp_path: Path,
    preserve_process_rng_state,
) -> None:
    """Resume must continue the same curriculum, validation, and RNG streams."""

    policy = SharedSecurityActorCritic(recurrent_size=256)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    rng = np.random.default_rng(7831)
    # Advance every stream so this does not accidentally pass with seed-only
    # reconstruction.
    rng.random(7)
    np.random.seed(9182)
    np.random.random(5)
    random.seed(417)
    [random.random() for _ in range(3)]
    torch.manual_seed(6281)
    torch.rand(11)

    expected_numpy_state = copy.deepcopy(rng.bit_generator.state)
    expected_global_numpy_state = copy.deepcopy(np.random.get_state())
    expected_python_state = copy.deepcopy(random.getstate())
    expected_torch_state = torch.get_rng_state().clone()
    expected_numpy_draw = rng.random(4)
    expected_global_numpy_draw = np.random.random(4)
    expected_python_draw = [random.random() for _ in range(4)]
    expected_torch_draw = torch.rand(4)

    # Put each stream back on the exact boundary captured above.
    rng.bit_generator.state = expected_numpy_state
    np.random.set_state(expected_global_numpy_state)
    random.setstate(expected_python_state)
    torch.set_rng_state(expected_torch_state)

    checkpoint = tmp_path / "resume.pt"
    marl_train._training_checkpoint(
        policy,
        optimizer,
        checkpoint,
        steps=12_345,
        updates=17,
        seed_cursor=91,
        best_worst_tier=0.42,
        best_selection_key=(0.42, 0.51, 0.49, 1.0, 2.0, 80.0),
        tiers=(3, 4, 5, 6),
        tier_probabilities=np.asarray((0.1, 0.2, 0.3, 0.4)),
        args={"gamma": 0.987, "gae_lambda": 0.95, "reward_contract": "security-v2"},
        rng=rng,
        next_validation=15_000,
        validation_cursor=700,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key in (
        "environment_fingerprint",
        "runtime",
        "training_args",
        "rng_state",
        "numpy_global_rng_state",
        "python_rng_state",
        "torch_rng_state",
        "next_validation",
        "validation_cursor",
        "resume_state",
    ):
        assert key in payload
    assert payload["next_validation"] == 15_000
    assert payload["validation_cursor"] == 700
    assert payload["training_args"]["gamma"] == pytest.approx(0.987)

    restored_rng = np.random.default_rng()
    restored_rng.bit_generator.state = payload["rng_state"]
    np.random.set_state(payload["numpy_global_rng_state"])
    random.setstate(payload["python_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    np.testing.assert_allclose(restored_rng.random(4), expected_numpy_draw)
    np.testing.assert_allclose(np.random.random(4), expected_global_numpy_draw)
    assert [random.random() for _ in range(4)] == pytest.approx(expected_python_draw)
    assert torch.rand(4) == pytest.approx(expected_torch_draw)

    boundary = payload["resume_state"]
    assert boundary["steps"] == 12_345
    assert boundary["updates"] == 17
    assert boundary["seed_cursor"] == 91
    assert boundary["next_validation"] == 15_000
    assert boundary["validation_cursor"] == 700


def test_interrupted_training_matches_the_same_continuous_boundary(
    tmp_path: Path,
) -> None:
    """A rollout-boundary restart must be bit-exact on CPU."""

    common = {
        "hours": 0.01,
        "env_count": 1,
        "rollout": 3,
        "epochs": 1,
        "tiers": "6",
        "recurrent_size": 256,
        "validation_interval": 0,
        "device": "cpu",
        "seed": 31415,
    }
    resumed = tmp_path / "resumed"
    continuous = tmp_path / "continuous"
    marl_train.train_security(
        output=resumed,
        max_steps=12,
        resume=False,
        **common,
    )
    marl_train.train_security(
        output=resumed,
        max_steps=24,
        resume=True,
        **common,
    )
    marl_train.train_security(
        output=continuous,
        max_steps=24,
        resume=False,
        **common,
    )
    resumed_payload = torch.load(
        resumed / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    continuous_payload = torch.load(
        continuous / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert resumed_payload["steps"] == continuous_payload["steps"]
    assert resumed_payload["updates"] == continuous_payload["updates"]
    assert resumed_payload["rng_state"] == continuous_payload["rng_state"]
    assert torch.equal(
        resumed_payload["torch_rng_state"],
        continuous_payload["torch_rng_state"],
    )
    assert all(
        torch.equal(resumed_payload["model"][name], continuous_payload["model"][name])
        for name in resumed_payload["model"]
    )
