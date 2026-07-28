"""Frozen-security integration checks for v2 runner training."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from ghostline.security_model import (
    SECURITY_OBSERVATION_CONTRACT,
    SharedSecurityActorCritic,
    save_security_policy,
)
from ghostline.security_opponents import (
    FrozenSecurityOpponentPool,
    FrozenSecurityRunnerEnvV2,
    load_frozen_security_policy,
)
from ghostline.runner_train_v2 import ScheduledRunnerEnv, observation_digest


def _checkpoint(path: Path, *, seed: int) -> Path:
    previous = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        save_security_policy(
            SharedSecurityActorCritic(recurrent_size=256),
            path,
            fixture="frozen-security-opponent",
        )
    finally:
        torch.set_rng_state(previous)
    return path


def _rollout(
    checkpoint: Path,
    *,
    seed: int,
    actions: tuple[int, ...],
) -> tuple[list[tuple[object, ...]], dict[str, np.ndarray]]:
    env = FrozenSecurityRunnerEnvV2(
        security_checkpoints=(checkpoint,),
        seed=seed,
        tier=6,
    )
    observation, info = env.reset(seed=seed, options={"tier": 6})
    assert info["security_opponent"]["observation_contract"] == (
        SECURITY_OBSERVATION_CONTRACT
    )
    records: list[tuple[object, ...]] = []
    for action in actions:
        observation, reward, terminated, truncated, info = env.step(action)
        records.append(
            (
                env.sim.elapsed_ticks,
                env.sim.player.copy(),
                tuple(
                    (
                        guard.guard_id,
                        guard.position.copy(),
                        int(
                            env.sim.operative_states[
                                guard.guard_id
                            ].current_order.intent
                        ),
                    )
                    for guard in env.sim.level.guards
                ),
                float(reward),
                bool(terminated),
                bool(truncated),
                int(info["security_decisions"]),
            )
        )
        if terminated or truncated:
            break
    hidden = env.security_controller.hidden
    assert hidden is not None
    hidden_copy = hidden.detach().cpu().numpy().copy()
    env.close()
    assert env.security_controller.adapter is None
    return records, {"hidden": hidden_copy, **observation}


def test_frozen_security_runner_reset_steps_and_replays_exactly(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path / "security.pt", seed=17)
    actions = (0, 1, 2, 0, 4, 0)
    first_records, first_observation = _rollout(
        checkpoint,
        seed=20_800_011,
        actions=actions,
    )
    second_records, second_observation = _rollout(
        checkpoint,
        seed=20_800_011,
        actions=actions,
    )

    assert len(first_records) == len(actions)
    for first, second in zip(first_records, second_records, strict=True):
        assert first[0] == second[0]
        assert np.array_equal(first[1], second[1])
        assert first[3:] == pytest.approx(second[3:])
        for first_guard, second_guard in zip(first[2], second[2], strict=True):
            assert first_guard[0] == second_guard[0]
            assert np.array_equal(first_guard[1], second_guard[1])
            assert first_guard[2] == second_guard[2]
    assert first_records[-1][-1] >= 3
    for key in first_observation:
        assert np.array_equal(first_observation[key], second_observation[key])


def test_reset_clears_recurrent_state_and_pool_selection_is_seeded(
    tmp_path: Path,
) -> None:
    first = _checkpoint(tmp_path / "security-a.pt", seed=31)
    second = _checkpoint(tmp_path / "security-b.pt", seed=37)
    pool = FrozenSecurityOpponentPool((first, second), selection_salt=41)
    sequence = [
        pool.select(seed=seed).provenance.opponent_id
        for seed in range(20_800_100, 20_800_116)
    ]
    replay = [
        pool.select(seed=seed).provenance.opponent_id
        for seed in range(20_800_100, 20_800_116)
    ]
    assert sequence == replay
    assert len(set(sequence)) == 2
    pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.select(seed=20_800_100)

    env = FrozenSecurityRunnerEnvV2(
        security_checkpoints=(first,),
        seed=20_800_117,
        tier=6,
    )
    env.reset(seed=20_800_117)
    env.step(0)
    assert env.security_controller.hidden is not None
    old_hidden = env.security_controller.hidden.detach().clone()
    env.reset(seed=20_800_117)
    assert env.security_controller.decisions == 1
    assert env.security_controller.hidden is not None
    assert torch.equal(env.security_controller.hidden, old_hidden)
    # A reset replays the first observation, rather than carrying memory from
    # the previous episode. Continuing once must therefore reproduce it.
    first_after_reset = env.security_controller.hidden.detach().clone()
    env.step(0)
    env.reset(seed=20_800_117)
    assert torch.equal(
        env.security_controller.hidden,
        first_after_reset,
    )
    env.close()


def test_checkpoint_loading_fails_closed_and_records_hash(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path / "security.pt", seed=43)
    frozen = load_frozen_security_policy(checkpoint)
    assert len(frozen.provenance.checkpoint_sha256) == 64
    assert frozen.provenance.checkpoint_path == str(checkpoint.resolve())
    assert frozen.provenance.deterministic

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["environment_fingerprint"] = "stale"
    stale = tmp_path / "stale.pt"
    torch.save(payload, stale)
    with pytest.raises(RuntimeError, match="stale security environment"):
        load_frozen_security_policy(stale)

    payload["observation_contract"] = "GhostlineSecurityParallel-v0"
    wrong = tmp_path / "wrong.pt"
    torch.save(payload, wrong)
    with pytest.raises(RuntimeError, match="not a GhostlineSecurityParallel-v2"):
        load_frozen_security_policy(wrong)


def test_bridge_uses_local_observation_contract_not_centralized_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actor inference may not inspect critic state or derive private targets."""

    checkpoint = _checkpoint(tmp_path / "security.pt", seed=47)
    env = FrozenSecurityRunnerEnvV2(
        security_checkpoints=(checkpoint,),
        seed=20_800_119,
        tier=6,
    )
    env.reset(seed=20_800_119)
    adapter = env.security_controller.adapter
    assert adapter is not None
    monkeypatch.setattr(
        adapter,
        "state",
        lambda: (_ for _ in ()).throw(
            AssertionError("centralized critic state reached actor inference")
        ),
    )
    assert env.security_controller.update(force=True)

    # Target positions flow through the security observation and its action
    # decoder. The bridge contains no independent player/route target oracle.
    source = inspect.getsource(type(env.security_controller).update)
    assert ".state(" not in source
    assert ".player" not in source
    assert "escape_route_cutoffs" not in source
    assert "_targets(" not in source
    assert env.security_controller.last_actions
    env.close()


def test_scheduled_runner_replays_a_frozen_security_episode_exactly(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path / "security.pt", seed=53)
    frozen = load_frozen_security_policy(checkpoint)
    arguments = {
        "rank": 0,
        "env_count": 1,
        "training_seed_start": 0,
        "tiers": (1, 2, 3, 4, 5, 6),
        "directives": (0, 1, 2, 3),
        "schedule_salt": 61,
        "adaptive_curriculum": True,
        "initial_curriculum_tier": 6,
        "security_opponent_paths": (str(checkpoint.resolve()),),
        "security_opponent_sha256": (
            frozen.provenance.checkpoint_sha256,
        ),
        "security_pool_salt": 67,
    }
    original = ScheduledRunnerEnv(**arguments)
    restored = ScheduledRunnerEnv(**arguments)
    try:
        observation, _ = original.reset()
        action = int(np.flatnonzero(observation["action_mask"])[0])
        observation, _, terminated, truncated, _ = original.step(action)
        assert not terminated and not truncated
        state = original.checkpoint_state()
        replay = restored.restore_state((state,))
        assert observation_digest(replay) == observation_digest(observation)
        assert isinstance(original.env, FrozenSecurityRunnerEnvV2)
        assert isinstance(restored.env, FrozenSecurityRunnerEnvV2)
        assert (
            original.env.security_controller.decisions
            == restored.env.security_controller.decisions
        )
        assert torch.equal(
            original.env.security_controller.hidden,
            restored.env.security_controller.hidden,
        )
    finally:
        original.close()
        restored.close()
