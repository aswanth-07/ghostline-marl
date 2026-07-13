"""Random Network Distillation for decaying, player-equivalent exploration reward."""
from __future__ import annotations

from typing import Mapping

import torch
from torch import nn


# v2 compact projection, including 12 player-intel records x 13 features.
RND_INPUT_SIZE = 527


def rnd_features(observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Compact raw-observation projection that is independent from policy weights."""
    leading = observation["ego"].shape[:-1]
    local = observation["local_grid"].float()[..., ::3, ::3].reshape(*leading, -1)
    values = (
        observation["ego"].float(),
        observation["objective"].float(),
        observation["rays"].float().reshape(*leading, -1),
        local,
        observation["targets"].float().reshape(*leading, -1),
        observation["target_mask"].float().reshape(*leading, -1),
        observation["entities"].float().reshape(*leading, -1),
        observation["entity_mask"].float().reshape(*leading, -1),
    )
    features = torch.cat(values, dim=-1)
    if features.shape[-1] != RND_INPUT_SIZE:
        raise RuntimeError(f"RND feature contract changed: expected {RND_INPUT_SIZE}, got {features.shape[-1]}")
    return features


def _network() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(RND_INPUT_SIZE, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, 128),
    )


class RandomNetworkDistillation(nn.Module):
    """Fixed target and trainable predictor with normalized novelty bonuses."""

    def __init__(self, *, normalization_decay: float = 0.999) -> None:
        super().__init__()
        self.target = _network()
        self.predictor = _network()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.normalization_decay = float(normalization_decay)
        self.register_buffer("error_square_mean", torch.ones(()))
        self.register_buffer("normalizer_updates", torch.zeros((), dtype=torch.long))

    def prediction_error(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        features = rnd_features(observation)
        with torch.no_grad():
            target = self.target(features)
        predicted = self.predictor(features)
        return (predicted - target).square().mean(dim=-1)

    @torch.no_grad()
    def intrinsic_reward(self, observation: Mapping[str, torch.Tensor], *, update_stats: bool = True) -> torch.Tensor:
        error = self.prediction_error(observation)
        if update_stats:
            square_mean = error.square().mean()
            if int(self.normalizer_updates.item()) == 0:
                self.error_square_mean.copy_(square_mean.clamp_min(1e-8))
            else:
                self.error_square_mean.mul_(self.normalization_decay).add_(
                    square_mean * (1.0 - self.normalization_decay)
                )
            self.normalizer_updates.add_(1)
        return (error / self.error_square_mean.sqrt().clamp_min(1e-6)).clamp(0.0, 5.0)

    def predictor_loss(
        self,
        observation: Mapping[str, torch.Tensor],
        *,
        update_fraction: float = 0.25,
    ) -> torch.Tensor:
        error = self.prediction_error(observation).flatten()
        if update_fraction >= 1.0:
            return error.mean()
        selected = torch.rand_like(error) < update_fraction
        if not selected.any():
            selected[torch.randint(0, len(selected), ())] = True
        return error[selected].mean()


def decaying_rnd_coefficient(
    steps: int,
    *,
    initial: float,
    decay_steps: int,
    final_fraction: float = 0.05,
) -> float:
    if initial <= 0.0:
        return 0.0
    progress = min(1.0, max(0.0, steps / max(1, decay_steps)))
    return float(initial * max(final_fraction, 1.0 - progress))
