"""Frozen v2 security opponents for runner training and evaluation.

The security policy is allowed to consume only the decentralized observation
dictionary produced for each operative by :class:`GhostlineSecurityParallelEnv`.
This adapter deliberately never reads the centralized critic state and never
derives a target from live simulation fields.  Semantic actions are converted
back into orders by the security environment itself, so inference and MAPPO
training share the same conditional intent-target legality contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ghostline.config_v2 import SECURITY_TACTICAL_TICKS
from ghostline.env_v2 import GhostlineEnvV2
from ghostline.security_env import GhostlineSecurityParallelEnv
from ghostline.security_model import (
    SECURITY_MODEL_CONTRACT_VERSION,
    SECURITY_OBSERVATION_CONTRACT,
    SharedSecurityActorCritic,
    load_security_policy,
    security_environment_fingerprint,
    select_factorized_actions,
)
from ghostline.simulation_v2 import GhostlineSimulationV2


FROZEN_SECURITY_OPPONENT_CONTRACT = "ghostline-frozen-security-opponent-v2.1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_mix(value: int) -> int:
    """SplitMix64 finalizer for deterministic checkpoint selection."""

    mask = (1 << 64) - 1
    value = (int(value) + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


@dataclass(frozen=True)
class FrozenSecurityProvenance:
    """Immutable identity recorded with runner trajectories/checkpoints."""

    checkpoint_path: str
    checkpoint_sha256: str
    observation_contract: str
    environment_fingerprint: str
    model_contract: str
    recurrent_size: int
    deterministic: bool = True
    opponent_contract: str = FROZEN_SECURITY_OPPONENT_CONTRACT

    @property
    def opponent_id(self) -> str:
        return f"security-v2:{self.checkpoint_sha256}"

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["opponent_id"] = self.opponent_id
        return result


@dataclass(frozen=True)
class FrozenSecurityPolicy:
    """A validated checkpoint plus its immutable provenance."""

    policy: SharedSecurityActorCritic
    provenance: FrozenSecurityProvenance


def load_frozen_security_policy(
    checkpoint: Path,
    *,
    device: str | torch.device = "cpu",
) -> FrozenSecurityPolicy:
    """Load one exact-contract policy, failing closed on any drift."""

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"security opponent checkpoint is missing: {path}")
    digest = _sha256(path)
    # ``load_security_policy`` verifies the observation contract, current
    # environment fingerprint, state-dict shape and recurrent width.
    policy = load_security_policy(path, device=device)
    provenance = FrozenSecurityProvenance(
        checkpoint_path=str(path),
        checkpoint_sha256=digest,
        observation_contract=SECURITY_OBSERVATION_CONTRACT,
        environment_fingerprint=security_environment_fingerprint(),
        model_contract=SECURITY_MODEL_CONTRACT_VERSION,
        recurrent_size=policy.recurrent_size,
    )
    return FrozenSecurityPolicy(policy=policy, provenance=provenance)


class FrozenSecurityOpponentPool:
    """Validated policies with deterministic per-episode selection."""

    def __init__(
        self,
        checkpoints: Sequence[Path],
        *,
        device: str | torch.device = "cpu",
        selection_salt: int = 0,
    ):
        paths = tuple(Path(path) for path in checkpoints)
        if not paths:
            raise ValueError("a frozen security pool requires at least one checkpoint")
        loaded = tuple(load_frozen_security_policy(path, device=device) for path in paths)
        hashes = [item.provenance.checkpoint_sha256 for item in loaded]
        if len(set(hashes)) != len(hashes):
            raise ValueError("a frozen security pool may not contain duplicate checkpoints")
        self._policies = loaded
        self.selection_salt = int(selection_salt)
        self._closed = False
        self.pool_id = hashlib.sha256(
            (
                FROZEN_SECURITY_OPPONENT_CONTRACT
                + ":"
                + ":".join(hashes)
                + f":{self.selection_salt}"
            ).encode("utf-8")
        ).hexdigest()

    @property
    def policies(self) -> tuple[FrozenSecurityPolicy, ...]:
        if self._closed:
            raise RuntimeError("frozen security pool is closed")
        return self._policies

    @property
    def provenance(self) -> tuple[dict[str, Any], ...]:
        return tuple(item.provenance.as_dict() for item in self.policies)

    def select(self, *, seed: int) -> FrozenSecurityPolicy:
        policies = self.policies
        index = _stable_mix(int(seed) ^ self.selection_salt) % len(policies)
        return policies[index]

    def close(self) -> None:
        self._policies = ()
        self._closed = True


class FrozenSecurityController:
    """Run a frozen recurrent security actor against one shared v2 simulation."""

    def __init__(self, frozen_policy: FrozenSecurityPolicy):
        self._frozen_policy = frozen_policy
        self._adapter: GhostlineSecurityParallelEnv | None = None
        self._hidden: torch.Tensor | None = None
        self._batch_agents: tuple[str, ...] = ()
        self._last_decision_tick: int | None = None
        self._closed = False
        self.decisions = 0
        self.last_actions: dict[str, np.ndarray] = {}

    @property
    def provenance(self) -> FrozenSecurityProvenance:
        return self._frozen_policy.provenance

    @property
    def hidden(self) -> torch.Tensor | None:
        return self._hidden

    @property
    def adapter(self) -> GhostlineSecurityParallelEnv | None:
        return self._adapter

    def set_policy(self, frozen_policy: FrozenSecurityPolicy) -> None:
        if self._closed:
            raise RuntimeError("frozen security controller is closed")
        self._frozen_policy = frozen_policy
        self._hidden = None
        self._batch_agents = ()
        self._last_decision_tick = None

    def reset(self, sim: GhostlineSimulationV2) -> None:
        """Bind to an episode and clear every recurrent/adapter state."""

        if self._closed:
            raise RuntimeError("frozen security controller is closed")
        if not isinstance(sim, GhostlineSimulationV2):
            raise TypeError("frozen security control requires GhostlineSimulationV2")
        sim.external_security = True
        if self._adapter is None:
            # The adapter's own simulation is immediately replaced. Its
            # ParallelEnv observation/action contracts remain the sole policy
            # boundary; its runner callback is never invoked.
            self._adapter = GhostlineSecurityParallelEnv(
                tier=sim.tier,
                seed=sim.seed,
                runner=lambda _sim: 0,
            )
        adapter = self._adapter
        adapter.sim = sim
        adapter.tier = sim.tier
        adapter.agents = [
            f"guard_{guard.guard_id}"
            for guard in sorted(sim.level.guards, key=lambda item: item.guard_id)
        ]
        adapter._target_cache.clear()
        adapter._current_observations = {}
        adapter._plane_signature = None
        adapter._plane_cache = None
        adapter._invalid_actions = 0
        self._hidden = None
        self._batch_agents = ()
        self._last_decision_tick = None
        self.decisions = 0
        self.last_actions = {}

    def update(self, *, force: bool = False) -> bool:
        """Apply one 5 Hz team decision; return whether inference ran."""

        if self._closed:
            raise RuntimeError("frozen security controller is closed")
        if self._adapter is None:
            raise RuntimeError("reset must bind a simulation before inference")
        sim = self._adapter.sim
        tick = int(sim.elapsed_ticks)
        if self._last_decision_tick == tick and not force:
            return False
        if not force and tick % SECURITY_TACTICAL_TICKS != 0:
            return False

        agents = tuple(
            f"guard_{guard.guard_id}"
            for guard in sorted(sim.level.guards, key=lambda item: item.guard_id)
        )
        self._adapter.agents = list(agents)
        if not agents:
            self._last_decision_tick = tick
            return False
        if agents != self._batch_agents:
            # Operatives are not respawned within a contract. Fail-safe reset
            # is still preferable to assigning one operative another's memory
            # if a future tier changes the active roster.
            self._hidden = None
            self._batch_agents = agents

        observations = {
            agent: self._adapter._observation(agent)
            for agent in agents
        }
        self._adapter._current_observations = observations
        device = next(self._frozen_policy.policy.parameters()).device
        tensors = {
            key: torch.as_tensor(
                np.stack([observations[agent][key] for agent in agents]),
                device=device,
            )
            for key in observations[agents[0]]
        }
        with torch.no_grad():
            logits, next_hidden = self._frozen_policy.policy.forward_actor(
                tensors,
                self._hidden,
            )
            selected = select_factorized_actions(
                logits,
                tensors["intent_target_mask"],
                deterministic=True,
            )
        actions_array = selected.cpu().numpy().astype(np.int64)
        actions = {
            agent: actions_array[index]
            for index, agent in enumerate(agents)
        }
        orders, invalid = self._adapter.orders_from_actions(
            actions,
            observations=observations,
        )
        if invalid:
            raise RuntimeError(
                "frozen security policy produced an action outside its conditional mask"
            )
        sim.set_security_orders(orders)
        self._hidden = next_hidden.detach()
        self._last_decision_tick = tick
        self.decisions += 1
        self.last_actions = {
            agent: action.copy()
            for agent, action in actions.items()
        }
        return True

    def close(self) -> None:
        if self._closed:
            return
        if self._adapter is not None:
            self._adapter.close()
            self._adapter = None
        self._hidden = None
        self._batch_agents = ()
        self.last_actions = {}
        self._closed = True


class FrozenSecurityRunnerEnvV2(GhostlineEnvV2):
    """Drop-in runner environment backed by a sampled frozen security pool.

    Integration with ``ScheduledRunnerEnv`` requires only replacing its
    ``GhostlineEnvV2(...)`` construction with this class and forwarding the
    declared checkpoint list and pool salt. Episode seeds deterministically
    select the opponent, so action replay also reconstructs recurrent security
    state exactly.
    """

    def __init__(
        self,
        *,
        security_checkpoints: Sequence[Path] = (),
        security_pool: FrozenSecurityOpponentPool | None = None,
        security_device: str | torch.device = "cpu",
        security_pool_salt: int = 0,
        **kwargs: Any,
    ):
        kwargs["external_security"] = True
        super().__init__(**kwargs)
        if security_pool is not None and security_checkpoints:
            raise ValueError(
                "pass either a shared security_pool or security_checkpoints"
            )
        if security_pool is None:
            self.security_pool = FrozenSecurityOpponentPool(
                security_checkpoints,
                device=security_device,
                selection_salt=security_pool_salt,
            )
            self._owns_security_pool = True
        else:
            self.security_pool = security_pool
            self._owns_security_pool = False
        selected = self.security_pool.select(seed=self.sim.seed)
        self.security_controller = FrozenSecurityController(selected)
        self.security_controller.reset(self.sim)
        self.security_controller.update(force=True)
        self._closed_security = False

    @property
    def security_opponent_provenance(self) -> Mapping[str, Any]:
        return self.security_controller.provenance.as_dict()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        requested = dict(options or {})
        requested["external_security"] = True
        observation, info = super().reset(seed=seed, options=requested)
        selected = self.security_pool.select(seed=self.sim.seed)
        self.security_controller.set_policy(selected)
        self.security_controller.reset(self.sim)
        self.security_controller.update(force=True)
        # The first order does not advance the simulation, so the runner's
        # reset observation remains valid.
        info = dict(info)
        info["security_opponent"] = selected.provenance.as_dict()
        info["security_pool_id"] = self.security_pool.pool_id
        return observation, info

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self.security_controller.update()
        observation, reward, terminated, truncated, info = super().step(action)
        info = dict(info)
        info["security_opponent_id"] = (
            self.security_controller.provenance.opponent_id
        )
        info["security_decisions"] = self.security_controller.decisions
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        if not self._closed_security:
            self.security_controller.close()
            if self._owns_security_pool:
                self.security_pool.close()
            self._closed_security = True
        super().close()
