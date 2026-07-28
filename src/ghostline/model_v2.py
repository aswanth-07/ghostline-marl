"""Runner policy for the multi-agent (Env-v2) track.

`model.py` is the frozen published-v1 network and is not touched. This module
owns the developmental v2 runner, which sees a wider observation, public field
targets, a 15-channel local grid, role-labelled entity rows, and chooses from
288 masked semantic actions.

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

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn

from ghostline.types_v2 import RUNNER_ACTION_COUNT_V2

OBSERVATION_CONTRACT_V2 = "GhostlineEnv-v2"
RUNNER_MODEL_CONTRACT_V2 = "runner-recurrent-field-policy-v2"


def runner_model_fingerprint(root: Path | None = None) -> str:
    """Bind checkpoints to the exact v2 network semantics and dimensions."""

    root = Path(__file__).resolve().parent if root is None else Path(root)
    model_source = (root / "model_v2.py").read_bytes().replace(b"\r\n", b"\n")
    payload = {
        "model_source": hashlib.sha256(model_source).hexdigest(),
        "model_contract": RUNNER_MODEL_CONTRACT_V2,
        "observation_contract": OBSERVATION_CONTRACT_V2,
        "action_count": RUNNER_ACTION_COUNT_V2,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def multi_agent_environment_fingerprint(root: Path | None = None) -> str:
    """Hash every source file that can alter a v2 transition or observation."""

    root = Path(__file__).resolve().parent if root is None else Path(root)
    digest = hashlib.sha256()
    for name in (
        "config.py",
        "types.py",
        "generation.py",
        "simulation.py",
        "env.py",
        "config_v2.py",
        "types_v2.py",
        "generation_v2.py",
        "simulation_v2.py",
        "env_v2.py",
    ):
        path = root / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    digest.update(f"runner-model-contract:{RUNNER_MODEL_CONTRACT_V2}".encode("utf-8"))
    return digest.hexdigest()


def orthogonal_(module: nn.Module, gain: float) -> nn.Module:
    """Orthogonal weights with zero bias, the standard PPO initialisation."""

    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    return module


class MaskedSetEncoderV2(nn.Module):
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


class RunnerPolicyV2(nn.Module):
    """Recurrent, directive-conditioned actor-critic for the v2 runner."""

    def __init__(self, *, recurrent_size: int = 384):
        super().__init__()
        if recurrent_size not in (256, 384, 512):
            raise ValueError("recurrent_size must be 256, 384 or 512")
        self.recurrent_size = int(recurrent_size)
        self.local_encoder = nn.Sequential(
            orthogonal_(nn.Conv2d(15, 32, 3, padding=1), np.sqrt(2)),
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
        # Runner field systems: charges, reach, transit, darkness.
        self.field_encoder = self._mlp(8, 48, 48)
        self.field_target_encoder = MaskedSetEncoderV2(13, 64)
        self.target_encoder = MaskedSetEncoderV2(10, 64)
        self.entity_encoder = MaskedSetEncoderV2(16, 64)

        fused = 128 + 64 + 64 + 64 + 64 + 64 + 64 + 48
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
        self.action_head = orthogonal_(nn.Linear(256, RUNNER_ACTION_COUNT_V2), 0.01)
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
        field = self.field_encoder(observation["field"].float())
        field_targets = self.field_target_encoder(
            observation["field_targets"], observation["field_target_mask"]
        )
        fused = self.fusion(
            torch.cat(
                (local, ego, objective, rays, targets, entities, field, field_targets),
                dim=-1,
            )
        )
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


def save_runner_v2(policy: RunnerPolicyV2, path: Path, **metadata: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": policy.state_dict(),
            "recurrent_size": policy.recurrent_size,
            "observation_contract": OBSERVATION_CONTRACT_V2,
            "action_count": RUNNER_ACTION_COUNT_V2,
            "environment_fingerprint": multi_agent_environment_fingerprint(),
            "model_fingerprint": runner_model_fingerprint(),
            "metadata": metadata,
        },
        path,
    )


def load_runner_v2(path: Path, *, device: str | torch.device = "cpu") -> RunnerPolicyV2:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("observation_contract") != OBSERVATION_CONTRACT_V2:
        raise RuntimeError(f"{path} is not a {OBSERVATION_CONTRACT_V2} checkpoint")
    if int(payload.get("action_count", 0)) != RUNNER_ACTION_COUNT_V2:
        raise RuntimeError(f"{path} was exported against a different v2 action count")
    if payload.get("environment_fingerprint") != multi_agent_environment_fingerprint():
        raise RuntimeError(f"{path} was produced by a stale v2 environment contract")
    if payload.get("model_fingerprint") != runner_model_fingerprint():
        raise RuntimeError(f"{path} was produced by a stale v2 model contract")
    policy = RunnerPolicyV2(recurrent_size=int(payload.get("recurrent_size", 384))).to(device)
    policy.load_state_dict(payload["model"], strict=True)
    policy.eval()
    return policy


def initialize_runner_v2_from_published_v1(
    policy: RunnerPolicyV2,
    path: Path,
) -> dict[str, object]:
    """Warm-start v2 from the immutable published-v1 recurrent policy.

    This is an explicit weight transplant, not checkpoint compatibility.  The
    published model continues to validate against its historical internal
    ``GhostlineEnv-v2`` metadata and frozen source fingerprint.  Shared visual,
    objective, recurrent, value, and movement parameters are copied into their
    v2 counterparts; new field channels stay neutral until v2 training.

    All 288 v2 actions inherit the logits of their corresponding 36-action
    movement/dash/pulse base.  A small semantic prior discourages immediately
    spamming an untrained decoy/crouch/interact bit without making those actions
    unreachable to PPO.
    """

    from ghostline.model import require_current_checkpoint

    source_path = Path(path)
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    require_current_checkpoint(payload, path=source_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise RuntimeError(f"{source_path} is not a published-v1 policy payload")
    if not bool(payload.get("recurrent", True)):
        raise RuntimeError("v2 warm-start requires the published recurrent policy")
    source_size = int(payload.get("recurrent_size", 0))
    if source_size != policy.recurrent_size:
        raise RuntimeError(
            "published-v1 and v2 recurrent widths must match for a safe warm-start "
            f"({source_size} != {policy.recurrent_size})"
        )
    source = payload["model"]

    def copy_exact(target_name: str, source_name: str | None = None) -> None:
        source_name = target_name if source_name is None else source_name
        target_tensor = policy.state_dict()[target_name]
        source_tensor = source.get(source_name)
        if source_tensor is None or tuple(source_tensor.shape) != tuple(target_tensor.shape):
            raise RuntimeError(
                f"cannot transplant {source_name} into {target_name}: "
                f"{None if source_tensor is None else tuple(source_tensor.shape)} "
                f"!= {tuple(target_tensor.shape)}"
            )
        target_tensor.copy_(source_tensor)

    with torch.no_grad():
        for name in (
            "local_encoder.0.bias",
            "local_encoder.2.weight",
            "local_encoder.2.bias",
            "local_encoder.4.weight",
            "local_encoder.4.bias",
            "local_encoder.7.weight",
            "local_encoder.7.bias",
            "ego_encoder.0.bias",
            "ego_encoder.2.weight",
            "ego_encoder.2.bias",
            "objective_encoder.0.weight",
            "objective_encoder.0.bias",
            "objective_encoder.2.weight",
            "objective_encoder.2.bias",
            "ray_encoder.0.bias",
            "ray_encoder.2.weight",
            "ray_encoder.2.bias",
            "target_encoder.item.0.weight",
            "target_encoder.item.0.bias",
            "target_encoder.item.2.weight",
            "target_encoder.item.2.bias",
            "target_encoder.score.weight",
            "target_encoder.score.bias",
            "entity_encoder.item.0.bias",
            "entity_encoder.item.2.weight",
            "entity_encoder.item.2.bias",
            "entity_encoder.score.weight",
            "entity_encoder.score.bias",
            "fusion.0.bias",
            "fusion.2.weight",
            "fusion.2.bias",
            "core.weight_ih_l0",
            "core.weight_hh_l0",
            "core.bias_ih_l0",
            "core.bias_hh_l0",
            "policy_decoder.0.weight",
            "policy_decoder.0.bias",
            "value_decoder.0.weight",
            "value_decoder.0.bias",
            "value_head.weight",
            "value_head.bias",
            "objective_head.weight",
            "objective_head.bias",
            "danger_head.weight",
            "danger_head.bias",
        ):
            copy_exact(name)

        # Existing spatial channels occupy the first eight planes.
        policy.local_encoder[0].weight.zero_()
        policy.local_encoder[0].weight[:, :8].copy_(source["local_encoder.0.weight"])

        # Existing ego features occupy the first 24 values.
        policy.ego_encoder[0].weight.zero_()
        policy.ego_encoder[0].weight[:, :24].copy_(source["ego_encoder.0.weight"])

        # V2 appends one projectile feature to each of the 24 ray records.
        policy.ray_encoder[0].weight.zero_()
        for ray in range(24):
            policy.ray_encoder[0].weight[:, ray * 4 : ray * 4 + 3].copy_(
                source["ray_encoder.0.weight"][:, ray * 3 : ray * 3 + 3]
            )

        # Existing entity columns retain their order; role labels are appended.
        policy.entity_encoder.item[0].weight.zero_()
        policy.entity_encoder.item[0].weight[:, :13].copy_(
            source["entity_encoder.item.0.weight"]
        )

        # The published encoder added objective features into the ego vector
        # before fusion. V2 separates them, so both blocks receive the same old
        # fusion projection. New field blocks are intentionally zeroed.
        old_fusion = source["fusion.0.weight"]
        new_fusion = policy.fusion[0].weight
        new_fusion.zero_()
        old_offsets = (0, 128, 192, 256, 320, 384)
        new_offsets = (0, 128, 192, 256, 320, 384, 448, 496, 560)
        new_fusion[:, new_offsets[0] : new_offsets[1]].copy_(
            old_fusion[:, old_offsets[0] : old_offsets[1]]
        )
        old_ego = old_fusion[:, old_offsets[1] : old_offsets[2]]
        new_fusion[:, new_offsets[1] : new_offsets[2]].copy_(old_ego)
        new_fusion[:, new_offsets[2] : new_offsets[3]].copy_(old_ego)
        for old_index, new_index in zip((2, 3, 4), (3, 4, 5), strict=True):
            new_fusion[:, new_offsets[new_index] : new_offsets[new_index + 1]].copy_(
                old_fusion[:, old_offsets[old_index] : old_offsets[old_index + 1]]
            )

        # A zero FiLM layer is the identity transform for every directive.
        policy.directive_film.weight.zero_()
        policy.directive_film.bias.zero_()

        old_action_weight = source["action_head.weight"]
        old_action_bias = source["action_head.bias"]
        for value in range(RUNNER_ACTION_COUNT_V2):
            base = value % 36
            semantic_bits = (
                int((value // 36) % 2)
                + int((value // 72) % 2)
                + int((value // 144) % 2)
            )
            policy.action_head.weight[value].copy_(old_action_weight[base])
            policy.action_head.bias[value].copy_(
                old_action_bias[base] - 1.25 * semantic_bits
            )

    return {
        "source": str(source_path),
        "source_environment_fingerprint": str(payload.get("environment_fingerprint", "")),
        "source_observation_contract": str(payload.get("observation_contract", "")),
        "source_recurrent_size": source_size,
        "target_observation_contract": OBSERVATION_CONTRACT_V2,
        "target_action_count": RUNNER_ACTION_COUNT_V2,
        "method": "published-v1-overlap-transplant-v1",
    }
