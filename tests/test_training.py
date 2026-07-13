from __future__ import annotations

import pytest

pytest.importorskip("stable_baselines3", reason="legacy baseline requires its frozen compatibility environment")

from neon_arena.training import (
    POLICY_NET_ARCH,
    EpisodeProgressCallback,
    mlp_parameter_count,
)


def test_policy_architecture_is_documented_size() -> None:
    assert POLICY_NET_ARCH == (384, 256)
    assert mlp_parameter_count() == 398_094

    # Dynamically verify parameter count matches mlp_parameter_count()
    from stable_baselines3.common.vec_env import DummyVecEnv
    from sb3_contrib import RecurrentPPO
    from neon_arena.env import BlacklineHeistEnv

    env = DummyVecEnv([lambda: BlacklineHeistEnv(curriculum_stage="full")])
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        policy_kwargs={
            "net_arch": dict(pi=list(POLICY_NET_ARCH), vf=list(POLICY_NET_ARCH)),
            "lstm_hidden_size": 256,
            "n_lstm_layers": 1,
            "shared_lstm": False,
            "enable_critic_lstm": True,
        }
    )
    actor_params = sum(p.numel() for p in model.policy.mlp_extractor.policy_net.parameters())
    actor_params += sum(p.numel() for p in model.policy.action_net.parameters())
    critic_params = sum(p.numel() for p in model.policy.mlp_extractor.value_net.parameters())
    critic_params += sum(p.numel() for p in model.policy.value_net.parameters())
    assert mlp_parameter_count() == actor_params + critic_params


def test_episode_progress_tracks_final_reward_postfix() -> None:
    callback = EpisodeProgressCallback(target_episodes=5, enabled=False)
    callback.locals = {
        "infos": [
            {"episode": {"r": 4.0, "l": 610}},
            {},
            {"episode": {"r": -2.0, "l": 770}},
        ]
    }
    callback.num_timesteps = 2048

    assert callback._on_step() is True

    assert callback.completed_episodes == 2
    assert callback.latest_return == -2.0
    assert callback.latest_length == 770
    assert list(callback.recent_returns) == [4.0, -2.0]
    assert callback._postfix() == {
        "steps": 2048,
        "completed": 2,
        "stage": "full",
        "success_100": "0.0%",
        "cores_100": "0.00",
        "fail_reason": "unknown",
        "final_reward": "-2.00",
        "mean_reward_50": "1.00",
        "last_len": 770,
    }


def test_episode_progress_counts_final_vectorized_overshoot() -> None:
    class ProgressProbe:
        def __init__(self) -> None:
            self.updated = 0
            self.postfix = None

        def update(self, value: int) -> None:
            self.updated += value

        def set_postfix(self, value: dict[str, int | str]) -> None:
            self.postfix = value

    callback = EpisodeProgressCallback(target_episodes=3, enabled=False)
    probe = ProgressProbe()
    callback.progress = probe  # type: ignore[assignment]
    callback.locals = {
        "infos": [
            {"episode": {"r": 1.0, "l": 10}},
            {"episode": {"r": 2.0, "l": 11}},
        ]
    }
    assert callback._on_step() is True
    callback.locals = {
        "infos": [
            {"episode": {"r": 3.0, "l": 12}},
            {"episode": {"r": 4.0, "l": 13}},
        ]
    }

    assert callback._on_step() is False
    assert callback.completed_episodes == 4
    assert probe.updated == 3
    assert probe.postfix is not None
    assert probe.postfix["completed"] == 3
