"""Runner policy for the multi-agent (Env-v3) track.

`model.py` is the frozen Env-v2 network and is not touched. This module owns the
Env-v3 runner, which sees a wider observation than Env-v2 -- a 27-value ego
vector, an 11-channel local grid, 16-feature entity rows, and a directive
record -- and chooses from 144 masked actions rather than 36.

Architecture decisions and why:

* **Orthogonal initialisation with per-layer gains.** The project had no
  explicit init anywhere, so every network relied on PyTorch's default uniform
  scheme. Orthogonal init with `sqrt(2)` on hidden layers, `0.01` on the policy
  head and `1.0` on value heads is the standard PPO recipe and mainly matters
  at the policy head: a near-zero final layer starts the agent close to uniform
  over legal actions instead of committing hard to whatever the random init
  preferred, which is worth real sample efficiency early in training.
* **Directive conditioning via FiLM.** Ghost, Speed and Greed ask for different
  behaviour from the same weights. Concatenating a six-value directive into a
  384-wide fusion lets the network mostly ignore it. A feature-wise linear
  modulation instead scales and shifts every fused feature, so the directive can
  actually re-purpose the shared representation.
* **Separate actor and critic trunks after the recurrent core.** Sharing a trunk
  couples policy and value gradients, and the value loss is typically much
  larger; keeping the decoders separate is cheap and avoids that interference.
* **Auxiliary heads retained.** Objective bearing and danger prediction give the
  recurrent core a dense learning signal that survives sparse-reward stretches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn

from ghostline.types_v3 import RUNNER_ACTION_COUNT_V3

OBSERVATION_CONTRACT_V3 = "GhostlineEnv-v3"


def orthogonal_(module: nn.Module, gain: float) -> nn.Module:
    """Orthogonal weights with zero bias, the standard PPO initialisation."""

    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    return module


class MaskedSetEncoderV3(nn.Module):
    """Attention-pooled encoder over a masked, variable-length entity set."""

    def __init__(self, inputs: int, hidden: int = 64):
        super().__init__()
        self.item = nn.Sequential(
            orthogonal_(nn.Linear(inputs, hidden), np.sqrt(2)),
            nn.ELU(),
            orthogonal_(nn.Linear(hidden, hidden), np.sqrt(2)),
            nn.ELU(),
        )
        self.score = orthogonal_(nn.Linear(hidden, 1), 1.0)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.item(values.float())
        valid = mask > 0
        logits = self.score(encoded).squeeze(-1).masked_fill(~valid, -1e9)
        weights = torch.softmax(logits, dim=-1) * valid.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.sum(encoded * weights.unsqueeze(-1), dim=-2)


class RunnerPolicyV3(nn.Module):
    """Recurrent, directive-conditioned actor-critic for the Env-v3 runner."""

    def __init__(self, *, recurrent_size: int = 384):
        super().__init__()
        if recurrent_size not in (256, 384, 512):
            raise ValueError("recurrent_size must be 256, 384 or 512")
        self.recurrent_size = int(recurrent_size)
        self.local_encoder = nn.Sequential(
            orthogonal_(nn.Conv2d(11, 32, 3, padding=1), np.sqrt(2)),
            nn.ELU(),
            orthogonal_(nn.Conv2d(32, 48, 3, stride=2, padding=1), np.sqrt(2)),
            nn.ELU(),
            orthogonal_(nn.Conv2d(48, 64, 3, stride=2, padding=1), np.sqrt(2)),
            nn.ELU(),
            nn.Flatten(),
            orthogonal_(nn.Linear(64 * 4 * 4, 128), np.sqrt(2)),
            nn.ELU(),
        )
        self.ego_encoder = self._mlp(27, 96, 64)
        self.objective_encoder = self._mlp(8, 48, 64)
        self.ray_encoder = self._mlp(24 * 4, 96, 64)
        self.target_encoder = MaskedSetEncoderV3(10, 64)
        self.entity_encoder = MaskedSetEncoderV3(16, 64)

        fused = 128 + 64 + 64 + 64 + 64 + 64
        self.fusion = nn.Sequential(
            orthogonal_(nn.Linear(fused, 384), np.sqrt(2)),
            nn.ELU(),
            nn.LayerNorm(384),
        )
        # FiLM conditioning: the directive scales and shifts the fused features.
        self.directive_film = orthogonal_(nn.Linear(6, 384 * 2), 1.0)
        self.core = nn.GRU(384, self.recurrent_size, batch_first=True)
        for name, parameter in self.core.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(parameter, gain=1.0)
            else:
                nn.init.constant_(parameter, 0.0)

        self.policy_decoder = nn.Sequential(
            orthogonal_(nn.Linear(self.recurrent_size, 256), np.sqrt(2)), nn.ELU()
        )
        self.value_decoder = nn.Sequential(
            orthogonal_(nn.Linear(self.recurrent_size, 256), np.sqrt(2)), nn.ELU()
        )
        # A near-zero policy head starts close to uniform over legal actions.
        self.action_head = orthogonal_(nn.Linear(256, RUNNER_ACTION_COUNT_V3), 0.01)
        self.value_head = orthogonal_(nn.Linear(256, 1), 1.0)
        self.objective_head = orthogonal_(nn.Linear(self.recurrent_size, 2), 1.0)
        self.danger_head = orthogonal_(nn.Linear(self.recurrent_size, 1), 1.0)

    @staticmethod
    def _mlp(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
        return nn.Sequential(
            orthogonal_(nn.Linear(inputs, hidden), np.sqrt(2)),
            nn.ELU(),
            orthogonal_(nn.Linear(hidden, outputs), np.sqrt(2)),
            nn.ELU(),
        )

    def encode(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        local = self.local_encoder(observation["local_grid"].float())
        ego = self.ego_encoder(observation["ego"].float())
        objective = self.objective_encoder(observation["objective"].float())
        rays = self.ray_encoder(observation["rays"].float().flatten(-2, -1))
        targets = self.target_encoder(observation["targets"], observation["target_mask"])
        entities = self.entity_encoder(observation["entities"], observation["entity_mask"])
        fused = self.fusion(torch.cat((local, ego, objective, rays, targets, entities), dim=-1))
        scale, shift = self.directive_film(observation["directive"].float()).chunk(2, dim=-1)
        # 1 + scale keeps the identity transform available at initialisation.
        return fused * (1.0 + scale) + shift

    @staticmethod
    def _masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask > 0
        empty = ~valid.any(dim=-1)
        if empty.any():
            valid = valid.clone()
            valid[empty, 0] = True
        return logits.masked_fill(~valid, -1e9)

    def forward(
        self,
        observation: Mapping[str, torch.Tensor],
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encode(observation)
        sequence, next_hidden = self.core(encoded.unsqueeze(1), hidden)
        latent = sequence[:, -1]
        logits = self._masked_logits(self.action_head(self.policy_decoder(latent)), observation["action_mask"])
        value = self.value_head(self.value_decoder(latent)).squeeze(-1)
        return logits, value, next_hidden

    def forward_sequence(
        self,
        observation: Mapping[str, torch.Tensor],
        hidden: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run time-major observations shaped ``[time, batch, ...]``."""

        time_steps, batch = observation["ego"].shape[:2]
        flat = {key: value.flatten(0, 1) for key, value in observation.items()}
        encoded = self.encode(flat).reshape(time_steps, batch, -1).transpose(0, 1)
        if reset_mask is None:
            sequence, next_hidden = self.core(encoded, hidden)
        else:
            outputs = []
            next_hidden = hidden
            for index in range(time_steps):
                reset = reset_mask[index].bool()
                if reset.any() and next_hidden is not None:
                    next_hidden = next_hidden.clone()
                    next_hidden[:, reset, :] = 0.0
                output, next_hidden = self.core(encoded[:, index : index + 1], next_hidden)
                outputs.append(output)
            sequence = torch.cat(outputs, dim=1)
        latent = sequence.transpose(0, 1)
        logits = self._masked_logits(self.action_head(self.policy_decoder(latent)), observation["action_mask"])
        value = self.value_head(self.value_decoder(latent)).squeeze(-1)
        return logits, value, next_hidden

    def auxiliary(self, observation: Mapping[str, torch.Tensor], hidden: torch.Tensor | None = None):
        """Objective bearing and danger, used as dense auxiliary targets."""

        encoded = self.encode(observation)
        sequence, next_hidden = self.core(encoded.unsqueeze(1), hidden)
        latent = sequence[:, -1]
        return torch.tanh(self.objective_head(latent)), torch.sigmoid(self.danger_head(latent).squeeze(-1)), next_hidden

    @torch.no_grad()
    def act(
        self,
        observation: Mapping[str, np.ndarray],
        hidden: torch.Tensor | None = None,
        *,
        deterministic: bool = True,
        device: str | torch.device = "cpu",
    ) -> tuple[int, torch.Tensor]:
        tensors = {
            key: torch.as_tensor(value, device=device).unsqueeze(0) for key, value in observation.items()
        }
        logits, _value, next_hidden = self.forward(tensors, hidden)
        if deterministic:
            action = int(torch.argmax(logits, dim=-1).item())
        else:
            action = int(torch.distributions.Categorical(logits=logits).sample().item())
        return action, next_hidden


def save_runner_v3(policy: RunnerPolicyV3, path: Path, **metadata: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": policy.state_dict(),
            "recurrent_size": policy.recurrent_size,
            "observation_contract": OBSERVATION_CONTRACT_V3,
            "action_count": RUNNER_ACTION_COUNT_V3,
            "metadata": metadata,
        },
        path,
    )


def load_runner_v3(path: Path, *, device: str | torch.device = "cpu") -> RunnerPolicyV3:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("observation_contract") != OBSERVATION_CONTRACT_V3:
        raise RuntimeError(f"{path} is not a {OBSERVATION_CONTRACT_V3} checkpoint")
    if int(payload.get("action_count", 0)) != RUNNER_ACTION_COUNT_V3:
        raise RuntimeError(f"{path} was exported against a different Env-v3 action count")
    policy = RunnerPolicyV3(recurrent_size=int(payload.get("recurrent_size", 384))).to(device)
    policy.load_state_dict(payload["model"], strict=True)
    policy.eval()
    return policy
