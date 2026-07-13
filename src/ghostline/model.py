from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn


OBSERVATION_CONTRACT = "GhostlineEnv-v2"


def current_environment_fingerprint() -> str:
    """Return the frozen player/simulation contract fingerprint.

    The import is intentionally delayed: ``imitation`` owns the source-file
    fingerprint and imports this module for the network implementation.
    """

    from ghostline.imitation import training_environment_fingerprint

    return training_environment_fingerprint()


def checkpoint_environment_fingerprint(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    direct = payload.get("environment_fingerprint")
    if direct:
        return str(direct)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and metadata.get("environment_fingerprint"):
        return str(metadata["environment_fingerprint"])
    return None


def require_current_checkpoint(payload: object, *, path: Path | None = None) -> str:
    """Fail closed when a policy/training state predates the current contract."""

    expected = current_environment_fingerprint()
    actual = checkpoint_environment_fingerprint(payload)
    label = str(path) if path is not None else "checkpoint"
    if actual is None:
        raise RuntimeError(
            f"Ghostline checkpoint {label} has no environment fingerprint; "
            "legacy checkpoints are audit evidence only"
        )
    if actual != expected:
        raise RuntimeError(
            f"Ghostline checkpoint {label} was produced by a stale environment "
            f"fingerprint ({actual[:12]} != {expected[:12]})"
        )
    return actual


class MaskedSetEncoder(nn.Module):
    def __init__(self, inputs: int, hidden: int = 64):
        super().__init__()
        self.item = nn.Sequential(nn.Linear(inputs, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU())
        self.score = nn.Linear(hidden, 1)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.item(values)
        logits = self.score(encoded).squeeze(-1).masked_fill(mask <= 0, -1e9)
        weights = torch.softmax(logits, dim=-1) * mask.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.sum(encoded * weights.unsqueeze(-1), dim=-2)


class UniversalGhostlinePolicy(nn.Module):
    """Entity-aware recurrent actor-critic used by training and packaged inference."""

    def __init__(self, *, recurrent: bool = True, recurrent_size: int = 384):
        super().__init__()
        if recurrent_size not in (256, 384, 512):
            raise ValueError("recurrent_size must be 256 or 384 (512 is retained for legacy checkpoints)")
        self.recurrent = recurrent
        self.recurrent_size = int(recurrent_size)
        self.local_encoder = nn.Sequential(
            nn.Conv2d(8, 32, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ELU(),
        )
        self.ego_encoder = nn.Sequential(nn.Linear(24, 96), nn.ELU(), nn.Linear(96, 64), nn.ELU())
        self.objective_encoder = nn.Sequential(nn.Linear(8, 48), nn.ELU(), nn.Linear(48, 64), nn.ELU())
        self.ray_encoder = nn.Sequential(nn.Linear(24 * 3, 96), nn.ELU(), nn.Linear(96, 64), nn.ELU())
        self.target_encoder = MaskedSetEncoder(10, 64)
        self.entity_encoder = MaskedSetEncoder(13, 64)
        self.fusion = nn.Sequential(nn.Linear(384, 384), nn.ELU(), nn.LayerNorm(384))
        self.core = nn.GRU(384, self.recurrent_size, batch_first=True) if recurrent else None
        core_size = self.recurrent_size if recurrent else 384
        self.policy_decoder = nn.Sequential(nn.Linear(core_size, 256), nn.ELU())
        self.value_decoder = nn.Sequential(nn.Linear(core_size, 256), nn.ELU())
        self.action_head = nn.Linear(256, 36)
        self.value_head = nn.Linear(256, 1)
        self.objective_head = nn.Linear(core_size, 2)
        self.danger_head = nn.Linear(core_size, 1)

    def encode(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        local = self.local_encoder(obs["local_grid"].float())
        ego = self.ego_encoder(obs["ego"].float())
        objective = obs.get("objective")
        if objective is None:
            objective = torch.zeros((*obs["ego"].shape[:-1], 8), device=obs["ego"].device)
        ego = ego + self.objective_encoder(objective.float())
        rays = self.ray_encoder(obs["rays"].float().flatten(1))
        targets = self.target_encoder(obs["targets"].float(), obs["target_mask"])
        entities = self.entity_encoder(obs["entities"].float(), obs["entity_mask"])
        return self.fusion(torch.cat((local, ego, rays, targets, entities), dim=-1))

    def forward(
        self,
        obs: Mapping[str, torch.Tensor],
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        encoded = self.encode(obs)
        if self.core is not None:
            output, next_hidden = self.core(encoded.unsqueeze(1), hidden)
            latent = output[:, -1]
        else:
            latent, next_hidden = encoded, None
        logits = self.action_head(self.policy_decoder(latent))
        if "action_mask" in obs:
            logits = logits.masked_fill(obs["action_mask"] <= 0, -1e9)
        value = self.value_head(self.value_decoder(latent)).squeeze(-1)
        return logits, value, next_hidden

    def auxiliary(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict goal direction and visible-danger pressure for imitation auxiliaries."""
        bearing = torch.tanh(self.objective_head(latent))
        danger = torch.sigmoid(self.danger_head(latent).squeeze(-1))
        return bearing, danger

    def _decode(self, latent: torch.Tensor, obs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.action_head(self.policy_decoder(latent))
        if "action_mask" in obs:
            logits = logits.masked_fill(obs["action_mask"] <= 0, -1e9)
        values = self.value_head(self.value_decoder(latent)).squeeze(-1)
        return logits, values

    def forward_sequence(
        self,
        obs: Mapping[str, torch.Tensor],
        hidden: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Vectorized [time, batch, ...] recurrent pass used by PPO updates."""
        time_steps, batch = obs["ego"].shape[:2]
        flattened = {key: value.flatten(0, 1) for key, value in obs.items()}
        encoded = self.encode(flattened).reshape(time_steps, batch, -1).transpose(0, 1)
        if self.core is not None and reset_mask is not None:
            outputs = []
            next_hidden = hidden
            for index in range(time_steps):
                if index > 0:
                    reset = reset_mask[index].to(encoded.device).bool()
                    if reset.any() and next_hidden is not None:
                        next_hidden = next_hidden.clone()
                        next_hidden[:, reset, :] = 0.0
                output, next_hidden = self.core(encoded[:, index : index + 1], next_hidden)
                outputs.append(output)
            latent = torch.cat(outputs, dim=1)
        elif self.core is not None:
            latent, next_hidden = self.core(encoded, hidden)
        else:
            latent, next_hidden = encoded, None
        latent = latent.transpose(0, 1)
        logits, values = self._decode(latent, obs)
        return logits, values, next_hidden

    def forward_sequence_aux(
        self,
        obs: Mapping[str, torch.Tensor],
        hidden: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Sequence pass returning policy/value plus player-equivalent auxiliary predictions."""
        time_steps, batch = obs["ego"].shape[:2]
        flattened = {key: value.flatten(0, 1) for key, value in obs.items()}
        encoded = self.encode(flattened).reshape(time_steps, batch, -1).transpose(0, 1)
        if self.core is None:
            latent, next_hidden = encoded, None
        elif reset_mask is None:
            latent, next_hidden = self.core(encoded, hidden)
        else:
            outputs = []
            next_hidden = hidden
            for index in range(time_steps):
                if index > 0 and next_hidden is not None:
                    reset = reset_mask[index].to(encoded.device).bool()
                    if reset.any():
                        next_hidden = next_hidden.clone()
                        next_hidden[:, reset, :] = 0.0
                output, next_hidden = self.core(encoded[:, index : index + 1], next_hidden)
                outputs.append(output)
            latent = torch.cat(outputs, dim=1)
        latent = latent.transpose(0, 1)
        logits, values = self._decode(latent, obs)
        bearing, danger = self.auxiliary(latent)
        return logits, values, bearing, danger, next_hidden

    @torch.no_grad()
    def act(
        self,
        observation: Mapping[str, np.ndarray],
        hidden: torch.Tensor | None = None,
        *,
        deterministic: bool = True,
        device: str | torch.device = "cpu",
    ) -> tuple[int, torch.Tensor | None]:
        tensors = {key: torch.as_tensor(value, device=device).unsqueeze(0) for key, value in observation.items()}
        logits, _, next_hidden = self(tensors, hidden)
        if deterministic:
            action = int(torch.argmax(logits, dim=-1).item())
        else:
            action = int(torch.distributions.Categorical(logits=logits).sample().item())
        return action, next_hidden


def load_policy(path: Path, *, device: str | torch.device = "cpu") -> UniversalGhostlinePolicy:
    payload = torch.load(path, map_location=device, weights_only=False)
    require_current_checkpoint(payload, path=path)
    recurrent = bool(payload.get("recurrent", True)) if isinstance(payload, dict) else True
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    if isinstance(payload, dict) and "recurrent_size" in payload:
        recurrent_size = int(payload["recurrent_size"])
    elif recurrent and "core.weight_hh_l0" in state:
        recurrent_size = int(state["core.weight_hh_l0"].shape[1])
    else:
        recurrent_size = 384
    policy = UniversalGhostlinePolicy(recurrent=recurrent, recurrent_size=recurrent_size).to(device)
    missing, unexpected = policy.load_state_dict(state, strict=False)
    incompatible = [name for name in missing if not name.startswith(("objective_encoder", "objective_head", "danger_head"))]
    if incompatible or unexpected:
        raise RuntimeError(f"Incompatible Ghostline checkpoint; missing={incompatible}, unexpected={unexpected}")
    policy.eval()
    return policy


def save_policy(policy: UniversalGhostlinePolicy, path: Path, **metadata: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = current_environment_fingerprint()
    metadata["environment_fingerprint"] = fingerprint
    torch.save(
        {
            "model": policy.state_dict(),
            "recurrent": policy.recurrent,
            "recurrent_size": policy.recurrent_size,
            "observation_version": 2,
            "observation_contract": OBSERVATION_CONTRACT,
            "environment_fingerprint": fingerprint,
            "metadata": metadata,
        },
        path,
    )
