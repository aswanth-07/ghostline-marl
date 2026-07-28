from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn

from ghostline.config_v2 import (
    MAX_SECURITY_TARGETS,
    SECURITY_CENTRAL_STATE_SIZE,
    SECURITY_TARGET_FEATURES,
)
from ghostline.types_v2 import RadioMessage, SecurityIntent


SECURITY_OBSERVATION_CONTRACT = "GhostlineSecurityParallel-v2"
SECURITY_ACTION_FACTORS = ("intent", "target", "message", "ability")
SECURITY_ACTION_SIZES = (
    len(SecurityIntent),
    MAX_SECURITY_TARGETS,
    len(RadioMessage),
    2,
)
SECURITY_MASK_KEYS = ("intent_mask", "target_mask", "message_mask", "ability_mask")
SECURITY_CONDITIONAL_MASK_KEY = "intent_target_mask"
SECURITY_MODEL_CONTRACT_VERSION = "shared-security-actor-critic-v4"
SECURITY_FINGERPRINT_FILES = (
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
    "security_types.py",
)


def _canonical_security_source_digest(root: Path | None = None) -> str:
    """Hash the mechanics contract with checkout-independent line endings."""

    root = Path(__file__).resolve().parent if root is None else Path(root)
    digest = hashlib.sha256()
    for name in SECURITY_FINGERPRINT_FILES:
        path = root / name
        digest.update(name.encode("utf-8"))
        payload = path.read_bytes().replace(b"\r\n", b"\n") if path.is_file() else b"<missing>"
        digest.update(payload)
    digest.update(f"security-model-contract:{SECURITY_MODEL_CONTRACT_VERSION}".encode("utf-8"))
    return digest.hexdigest()


def security_environment_fingerprint() -> str:
    """Hash inherited and v2 mechanics, generation, reward, and model sources."""

    return _canonical_security_source_digest()


def orthogonal_(module: nn.Module, gain: float) -> nn.Module:
    """Orthogonal weights with zero bias, the standard PPO initialisation.

    The project previously used PyTorch defaults everywhere. The gain schedule
    matters most at the action heads: initialising them near zero starts every
    operative close to uniform over its legal actions instead of committing to
    whatever the random draw happened to favour, which is worth real sample
    efficiency in the first few thousand updates.
    """

    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    return module


class MaskedSecuritySetEncoder(nn.Module):
    def __init__(self, inputs: int, hidden: int = 64):
        super().__init__()
        self.item = nn.Sequential(
            orthogonal_(nn.Linear(inputs, hidden), 2.0 ** 0.5),
            nn.ELU(),
            orthogonal_(nn.Linear(hidden, hidden), 2.0 ** 0.5),
            nn.ELU(),
        )
        self.score = orthogonal_(nn.Linear(hidden, 1), 1.0)

    def encode_items(self, values: torch.Tensor) -> torch.Tensor:
        return self.item(values.float())

    def pool(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask > 0
        logits = self.score(encoded).squeeze(-1).masked_fill(~valid, -1e9)
        weights = torch.softmax(logits, dim=-1) * valid.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.sum(encoded * weights.unsqueeze(-1), dim=-2)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.pool(self.encode_items(values), mask)


class SharedSecurityActorCritic(nn.Module):
    """Decentralized recurrent actor with an agent-specific CTDE critic."""

    def __init__(self, *, recurrent_size: int = 256):
        super().__init__()
        if recurrent_size not in (256, 384):
            raise ValueError("recurrent_size must be 256 or 384")
        self.recurrent_size = int(recurrent_size)
        self.local_encoder = nn.Sequential(
            orthogonal_(nn.Conv2d(8, 24, 3, padding=1), 2.0 ** 0.5),
            nn.ELU(),
            orthogonal_(nn.Conv2d(24, 40, 3, stride=2, padding=1), 2.0 ** 0.5),
            nn.ELU(),
            orthogonal_(nn.Conv2d(40, 48, 3, stride=2, padding=1), 2.0 ** 0.5),
            nn.ELU(),
            nn.Flatten(),
            orthogonal_(nn.Linear(48 * 4 * 4, 96), 2.0 ** 0.5),
            nn.ELU(),
        )
        self.ego_encoder = nn.Sequential(
            orthogonal_(nn.Linear(18, 64), 2.0 ** 0.5), nn.ELU(),
            orthogonal_(nn.Linear(64, 48), 2.0 ** 0.5), nn.ELU(),
        )
        self.runner_encoder = nn.Sequential(
            orthogonal_(nn.Linear(12, 48), 2.0 ** 0.5), nn.ELU(),
            orthogonal_(nn.Linear(48, 48), 2.0 ** 0.5), nn.ELU(),
        )
        self.teammate_encoder = MaskedSecuritySetEncoder(12, 48)
        self.target_encoder = MaskedSecuritySetEncoder(SECURITY_TARGET_FEATURES, 48)
        self.radio_encoder = MaskedSecuritySetEncoder(8, 48)
        self.actor_fusion = nn.Sequential(
            orthogonal_(nn.Linear(336, 320), 2.0 ** 0.5),
            nn.ELU(),
            nn.LayerNorm(320),
        )
        self.actor_core = nn.GRU(320, self.recurrent_size, batch_first=True)
        for name, parameter in self.actor_core.named_parameters():
            if "weight" in name:
                # PyTorch concatenates reset/update/new gates on axis zero.
                # Initialize each gate independently; applying one orthogonal
                # transform to the concatenated matrix couples their bases.
                for gate in parameter.chunk(3, dim=0):
                    nn.init.orthogonal_(gate, gain=1.0)
            else:
                nn.init.constant_(parameter, 0.0)
        self.actor_decoder = nn.Sequential(orthogonal_(nn.Linear(self.recurrent_size, 192), 2.0 ** 0.5), nn.ELU())
        # Near-zero action heads start each operative close to uniform over
        # its legal factorised actions.
        self.intent_head = orthogonal_(nn.Linear(192, SECURITY_ACTION_SIZES[0]), 0.01)
        # Targets are a variable semantic set. A learned pointer preserves each
        # slot's geometry and kind; pooling them and predicting a fixed index
        # loses the very row the target factor is meant to select.
        self.target_query = orthogonal_(nn.Linear(192, 48), 0.01)
        self.message_head = orthogonal_(nn.Linear(192, SECURITY_ACTION_SIZES[2]), 0.01)
        self.ability_head = orthogonal_(nn.Linear(192, SECURITY_ACTION_SIZES[3]), 0.01)

        # Centralized training-only critic. Operative blocks use a shared
        # encoder and masked pooling, making the joint representation
        # permutation equivariant. Each V_i receives its own encoded block,
        # team context, and the unordered facility context. Actor methods never
        # read this privileged state.
        self.critic_agent = nn.Sequential(
            orthogonal_(nn.Linear(8, 64), 2.0 ** 0.5),
            nn.ELU(),
            orthogonal_(nn.Linear(64, 64), 2.0 ** 0.5),
            nn.ELU(),
        )
        critic_global_size = SECURITY_CENTRAL_STATE_SIZE - 5 * 8 - 5
        self.critic_global = nn.Sequential(
            orthogonal_(nn.Linear(critic_global_size, 128), 2.0 ** 0.5),
            nn.ELU(),
            nn.LayerNorm(128),
        )
        self.critic_head = nn.Sequential(
            orthogonal_(nn.Linear(128 + 64 + 64, 192), 2.0 ** 0.5),
            nn.ELU(),
            orthogonal_(nn.Linear(192, 1), 1.0),
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
            fallback = torch.zeros_like(valid)
            fallback[..., 0] = True
            valid = torch.where(empty.unsqueeze(-1), fallback, valid)
        return logits.masked_fill(~valid, -1e9)

    def _heads(
        self,
        latent: torch.Tensor,
        observation: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        decoded = self.actor_decoder(latent)
        target_items = self.target_encoder.encode_items(observation["targets"])
        target_query = self.target_query(decoded)
        target_logits = torch.sum(
            target_items * target_query.unsqueeze(-2),
            dim=-1,
        ) / float(target_items.shape[-1]) ** 0.5
        raw = (
            self.intent_head(decoded),
            target_logits,
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
        """Return one centralized value per operative slot.

        ``central_state`` may be ``[batch, state]`` or time-major
        ``[time, batch, state]``. The result appends an operative dimension of
        length five. Inactive values are harmless and are excluded by the
        trainer's explicit active mask.
        """

        state = central_state.float()
        if state.shape[-1] != SECURITY_CENTRAL_STATE_SIZE:
            raise ValueError(
                f"expected central state width {SECURITY_CENTRAL_STATE_SIZE}, "
                f"got {state.shape[-1]}"
            )
        mission = state[..., :12]
        agents = state[..., 12 : 12 + 5 * 8].reshape(*state.shape[:-1], 5, 8)
        facility = state[..., 12 + 5 * 8 : -5]
        presence = state[..., -5:] > 0.0
        encoded_agents = self.critic_agent(agents)
        weights = presence.float().unsqueeze(-1)
        team = (encoded_agents * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)
        global_context = self.critic_global(torch.cat((mission, facility), dim=-1))
        global_context = global_context.unsqueeze(-2).expand(*encoded_agents.shape[:-1], 128)
        team = team.unsqueeze(-2).expand_as(encoded_agents)
        return self.critic_head(
            torch.cat((global_context, team, encoded_agents), dim=-1)
        ).squeeze(-1)

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
        action = select_factorized_actions(
            logits,
            tensors[SECURITY_CONDITIONAL_MASK_KEY],
            deterministic=deterministic,
        )
        return action[0].cpu().numpy().astype(np.int64), next_hidden


def _conditional_target_logits(
    target_logits: torch.Tensor,
    intents: torch.Tensor,
    intent_target_mask: torch.Tensor,
) -> torch.Tensor:
    target_count = target_logits.shape[-1]
    selected_mask = torch.gather(
        intent_target_mask,
        dim=-2,
        index=intents.unsqueeze(-1).unsqueeze(-1).expand(
            *intents.shape,
            1,
            target_count,
        ),
    ).squeeze(-2)
    valid = selected_mask > 0
    empty = ~valid.any(dim=-1)
    if empty.any():
        fallback = torch.zeros_like(valid)
        fallback[..., 0] = True
        valid = torch.where(empty.unsqueeze(-1), fallback, valid)
    return target_logits.masked_fill(~valid, -1e9)


def select_factorized_actions(
    logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    intent_target_mask: torch.Tensor,
    *,
    deterministic: bool,
) -> torch.Tensor:
    """Sample intent first, then a target legal for that intent."""

    intent_distribution = torch.distributions.Categorical(logits=logits[0])
    intents = (
        torch.argmax(logits[0], dim=-1)
        if deterministic
        else intent_distribution.sample()
    )
    conditional_targets = _conditional_target_logits(
        logits[1],
        intents,
        intent_target_mask,
    )
    target_distribution = torch.distributions.Categorical(logits=conditional_targets)
    targets = (
        torch.argmax(conditional_targets, dim=-1)
        if deterministic
        else target_distribution.sample()
    )
    remaining = [
        torch.argmax(head, dim=-1)
        if deterministic
        else torch.distributions.Categorical(logits=head).sample()
        for head in logits[2:]
    ]
    return torch.stack((intents, targets, *remaining), dim=-1)


def factorized_log_prob(
    logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    actions: torch.Tensor,
    intent_target_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the autoregressive semantic log probability and entropy."""

    log_probability = torch.zeros(actions.shape[:-1], dtype=torch.float32, device=actions.device)
    entropy = torch.zeros_like(log_probability)
    intent_distribution = torch.distributions.Categorical(logits=logits[0])
    log_probability = log_probability + intent_distribution.log_prob(actions[..., 0])
    entropy = entropy + intent_distribution.entropy()
    target_logits = logits[1]
    if intent_target_mask is not None:
        target_logits = _conditional_target_logits(
            target_logits,
            actions[..., 0],
            intent_target_mask,
        )
    target_distribution = torch.distributions.Categorical(logits=target_logits)
    log_probability = log_probability + target_distribution.log_prob(actions[..., 1])
    entropy = entropy + target_distribution.entropy()
    for index, head in enumerate(logits[2:], start=2):
        distribution = torch.distributions.Categorical(logits=head)
        log_probability = log_probability + distribution.log_prob(actions[..., index])
        entropy = entropy + distribution.entropy()
    return log_probability, entropy


def save_security_policy(policy: SharedSecurityActorCritic, path: Path, **metadata: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = security_environment_fingerprint()
    payload = {
            "model": policy.state_dict(),
            "recurrent_size": policy.recurrent_size,
            "observation_contract": SECURITY_OBSERVATION_CONTRACT,
            "environment_fingerprint": fingerprint,
            "metadata": metadata,
        }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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
