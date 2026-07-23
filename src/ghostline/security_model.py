from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn


SECURITY_OBSERVATION_CONTRACT = "GhostlineSecurityParallel-v0"
SECURITY_ACTION_FACTORS = ("intent", "target", "message", "ability")
SECURITY_ACTION_SIZES = (8, 8, 5, 2)
SECURITY_MASK_KEYS = ("intent_mask", "target_mask", "message_mask", "ability_mask")


def security_environment_fingerprint() -> str:
    """Fingerprint the exact public security-training contract.

    The v2 player checkpoint has its own immutable fingerprint.  Keeping this
    one separate makes it impossible to accidentally load a runner policy as a
    security controller, or to resume MAPPO after a silent mechanics change.
    """

    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in (
        "config_v3.py",
        "types_v3.py",
        "simulation_v3.py",
        "security_baselines.py",
        "security_env.py",
        "security_model.py",
    ):
        path = root / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


class MaskedSecuritySetEncoder(nn.Module):
    def __init__(self, inputs: int, hidden: int = 64):
        super().__init__()
        self.item = nn.Sequential(
            nn.Linear(inputs, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
        )
        self.score = nn.Linear(hidden, 1)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.item(values.float())
        valid = mask > 0
        logits = self.score(encoded).squeeze(-1).masked_fill(~valid, -1e9)
        weights = torch.softmax(logits, dim=-1) * valid.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.sum(encoded * weights.unsqueeze(-1), dim=-2)


class SharedSecurityActorCritic(nn.Module):
    """Parameter-shared recurrent operative actor with a CTDE team critic."""

    def __init__(self, *, recurrent_size: int = 256):
        super().__init__()
        if recurrent_size not in (256, 384):
            raise ValueError("recurrent_size must be 256 or 384")
        self.recurrent_size = int(recurrent_size)
        self.local_encoder = nn.Sequential(
            nn.Conv2d(8, 24, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(24, 40, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(40, 48, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(48 * 4 * 4, 96),
            nn.ELU(),
        )
        self.ego_encoder = nn.Sequential(nn.Linear(18, 64), nn.ELU(), nn.Linear(64, 48), nn.ELU())
        self.runner_encoder = nn.Sequential(nn.Linear(12, 48), nn.ELU(), nn.Linear(48, 48), nn.ELU())
        self.teammate_encoder = MaskedSecuritySetEncoder(12, 48)
        self.target_encoder = MaskedSecuritySetEncoder(8, 48)
        self.radio_encoder = MaskedSecuritySetEncoder(8, 48)
        self.actor_fusion = nn.Sequential(
            nn.Linear(336, 320),
            nn.ELU(),
            nn.LayerNorm(320),
        )
        self.actor_core = nn.GRU(320, self.recurrent_size, batch_first=True)
        self.actor_decoder = nn.Sequential(nn.Linear(self.recurrent_size, 192), nn.ELU())
        self.intent_head = nn.Linear(192, SECURITY_ACTION_SIZES[0])
        self.target_head = nn.Linear(192, SECURITY_ACTION_SIZES[1])
        self.message_head = nn.Linear(192, SECURITY_ACTION_SIZES[2])
        self.ability_head = nn.Linear(192, SECURITY_ACTION_SIZES[3])

        # Centralized training-only critic.  No actor method reads this state.
        self.critic = nn.Sequential(
            nn.Linear(64, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )

    def encode_actor(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        local = self.local_encoder(observation["local_grid"].float())
        ego = self.ego_encoder(observation["ego"].float())
        runner = self.runner_encoder(observation["runner"].float())
        teammates = self.teammate_encoder(observation["teammates"], observation["teammate_mask"])
        targets = self.target_encoder(observation["targets"], observation["target_mask"])
        radio = self.radio_encoder(observation["radio"], observation["radio_mask"])
        return self.actor_fusion(torch.cat((local, ego, runner, teammates, targets, radio), dim=-1))

    @staticmethod
    def _mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask > 0
        # Every environment observation guarantees a fallback action, but this
        # guard keeps padded inactive agents numerically safe during training.
        empty = ~valid.any(dim=-1)
        if empty.any():
            valid = valid.clone()
            valid[empty, 0] = True
        return logits.masked_fill(~valid, -1e9)

    def _heads(
        self,
        latent: torch.Tensor,
        observation: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        decoded = self.actor_decoder(latent)
        raw = (
            self.intent_head(decoded),
            self.target_head(decoded),
            self.message_head(decoded),
            self.ability_head(decoded),
        )
        return tuple(
            self._mask_logits(logits, observation[mask_key])
            for logits, mask_key in zip(raw, SECURITY_MASK_KEYS, strict=True)
        )

    def forward_actor(
        self,
        observation: Mapping[str, torch.Tensor],
        hidden: torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        encoded = self.encode_actor(observation)
        sequence, next_hidden = self.actor_core(encoded.unsqueeze(1), hidden)
        return self._heads(sequence[:, -1], observation), next_hidden

    def forward_actor_sequence(
        self,
        observation: Mapping[str, torch.Tensor],
        hidden: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Run time-major observations shaped ``[time, batch, ...]``."""

        time_steps, batch = observation["ego"].shape[:2]
        flattened = {key: value.flatten(0, 1) for key, value in observation.items()}
        encoded = self.encode_actor(flattened).reshape(time_steps, batch, -1).transpose(0, 1)
        if reset_mask is None:
            sequence, next_hidden = self.actor_core(encoded, hidden)
        else:
            outputs: list[torch.Tensor] = []
            next_hidden = hidden
            for index in range(time_steps):
                reset = reset_mask[index].bool()
                if reset.any() and next_hidden is not None:
                    next_hidden = next_hidden.clone()
                    next_hidden[:, reset, :] = 0.0
                output, next_hidden = self.actor_core(encoded[:, index : index + 1], next_hidden)
                outputs.append(output)
            sequence = torch.cat(outputs, dim=1)
        latent = sequence.transpose(0, 1)
        return self._heads(latent, observation), next_hidden

    def value(self, central_state: torch.Tensor) -> torch.Tensor:
        return self.critic(central_state.float()).squeeze(-1)

    @torch.no_grad()
    def act(
        self,
        observation: Mapping[str, np.ndarray],
        hidden: torch.Tensor | None = None,
        *,
        deterministic: bool = True,
        device: str | torch.device = "cpu",
    ) -> tuple[np.ndarray, torch.Tensor]:
        tensors = {
            key: torch.as_tensor(value, device=device).unsqueeze(0)
            for key, value in observation.items()
        }
        logits, next_hidden = self.forward_actor(tensors, hidden)
        if deterministic:
            action = [int(torch.argmax(head, dim=-1).item()) for head in logits]
        else:
            action = [int(torch.distributions.Categorical(logits=head).sample().item()) for head in logits]
        return np.asarray(action, dtype=np.int64), next_hidden


def factorized_log_prob(
    logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return summed log probability and entropy for the semantic action."""

    log_probability = torch.zeros(actions.shape[:-1], dtype=torch.float32, device=actions.device)
    entropy = torch.zeros_like(log_probability)
    for index, head in enumerate(logits):
        distribution = torch.distributions.Categorical(logits=head)
        log_probability = log_probability + distribution.log_prob(actions[..., index])
        entropy = entropy + distribution.entropy()
    return log_probability, entropy


def save_security_policy(policy: SharedSecurityActorCritic, path: Path, **metadata: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = security_environment_fingerprint()
    torch.save(
        {
            "model": policy.state_dict(),
            "recurrent_size": policy.recurrent_size,
            "observation_contract": SECURITY_OBSERVATION_CONTRACT,
            "environment_fingerprint": fingerprint,
            "metadata": metadata,
        },
        path,
    )


def load_security_policy(
    path: Path,
    *,
    device: str | torch.device = "cpu",
) -> SharedSecurityActorCritic:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("observation_contract") != SECURITY_OBSERVATION_CONTRACT:
        raise RuntimeError(f"{path} is not a {SECURITY_OBSERVATION_CONTRACT} checkpoint")
    expected = security_environment_fingerprint()
    if payload.get("environment_fingerprint") != expected:
        raise RuntimeError(f"{path} was produced by a stale security environment contract")
    policy = SharedSecurityActorCritic(recurrent_size=int(payload.get("recurrent_size", 256))).to(device)
    policy.load_state_dict(payload["model"], strict=True)
    policy.eval()
    return policy
