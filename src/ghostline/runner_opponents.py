from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ghostline.env import GhostlineEnv
from ghostline.env_v2 import GhostlineEnvV2
from ghostline.simulation_v2 import GhostlineSimulationV2


class FrozenPublishedV1RunnerOpponent:
    """Run the published-v1 policy inside a developmental-v2 security match.

    V2 is additive: the frozen runner deliberately retains its original 36
    actions and cannot use the new decoy.  It does, however, receive the same
    public live geometry with temporary locks represented as blocked cells.
    This provides a stable, provenance-bound opponent for security training.
    """

    def __init__(
        self,
        policy: Any,
        *,
        device: str = "cpu",
        opponent_id: str | None = None,
    ):
        self.policy = policy
        self.device = device
        self.opponent_id = opponent_id
        self.hidden = None
        self.env: GhostlineEnv | None = None
        self._sim_identity: int | None = None
        self._topology_signature: tuple[tuple[tuple[int, int], bool], ...] = ()

    def reset(self, sim: GhostlineSimulationV2) -> None:
        if self.env is not None:
            self.env.close()
        self.env = GhostlineEnv(seed=sim.seed, tier=sim.tier)
        self.env.sim = sim
        self.env.tier = sim.tier
        self.env._distance_cache.clear()
        self.hidden = None
        self._sim_identity = id(sim)
        self._topology_signature = self._topology(sim)

    @staticmethod
    def _topology(sim: GhostlineSimulationV2) -> tuple[tuple[tuple[int, int], bool], ...]:
        return tuple((door.tile, bool(door.locked)) for door in sim.security_doors)

    def __call__(self, sim: GhostlineSimulationV2) -> int:
        if self.env is None or self._sim_identity != id(sim):
            self.reset(sim)
        assert self.env is not None
        topology = self._topology(sim)
        if topology != self._topology_signature:
            self.env._distance_cache.clear()
            self._topology_signature = topology
        observation = self.env._observation()
        observation["action_mask"] = np.asarray(observation["action_mask"][:36], dtype=np.int8)
        action, self.hidden = self.policy.act(
            observation,
            self.hidden,
            deterministic=True,
            device=self.device,
        )
        return int(action)

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None


class FrozenRunnerV2Opponent:
    """Run a frozen 288-action v2 runner against adaptive security."""

    def __init__(
        self,
        policy: Any,
        *,
        device: str = "cpu",
        opponent_id: str | None = None,
    ):
        self.policy = policy
        self.device = device
        self.opponent_id = opponent_id
        self.hidden = None
        self.env: GhostlineEnvV2 | None = None
        self._sim_identity: int | None = None
        self._topology_signature: tuple[tuple[tuple[int, int], bool], ...] = ()

    def reset(self, sim: GhostlineSimulationV2) -> None:
        if self.env is not None:
            self.env.close()
        self.env = GhostlineEnvV2(
            seed=sim.seed,
            tier=sim.tier,
            directive=sim.directive,
            external_security=True,
        )
        self.env.sim = sim
        self.env.tier = sim.tier
        self.env.directive = sim.directive
        self.env.external_security = True
        self.env._reset_episode_metrics()
        self.hidden = None
        self._sim_identity = id(sim)
        self._topology_signature = self._topology(sim)

    @staticmethod
    def _topology(
        sim: GhostlineSimulationV2,
    ) -> tuple[tuple[tuple[int, int], bool], ...]:
        return tuple(
            (door.tile, bool(door.locked))
            for door in sim.security_doors
        )

    def __call__(self, sim: GhostlineSimulationV2) -> int:
        if self.env is None or self._sim_identity != id(sim):
            self.reset(sim)
        assert self.env is not None
        topology = self._topology(sim)
        if topology != self._topology_signature:
            self.env._distance_cache.clear()
            self._topology_signature = topology
        observation = self.env._observation()
        action, self.hidden = self.policy.act(
            observation,
            self.hidden,
            deterministic=True,
            device=self.device,
        )
        return int(action)

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None


# Historical internal name retained for old imports and audit fixtures. Public
# documentation calls this the published-v1 opponent.
FrozenV2RunnerOpponent = FrozenPublishedV1RunnerOpponent


def runner_opponent_kind(policy: Any) -> str:
    """Identify a loaded runner policy without relying on old metadata names."""

    action_count = int(getattr(getattr(policy, "action_head", None), "out_features", 0))
    if action_count == 288:
        return "runner-v2"
    if action_count == 36:
        return "published-v1"
    raise RuntimeError(f"unsupported runner opponent action count: {action_count}")


def make_frozen_runner_opponent(
    policy: Any,
    *,
    opponent_id: str | None = None,
) -> Any:
    kind = runner_opponent_kind(policy)
    if kind == "runner-v2":
        return FrozenRunnerV2Opponent(policy, opponent_id=opponent_id)
    return FrozenPublishedV1RunnerOpponent(policy, opponent_id=opponent_id)


def load_runner_opponent_policy(
    path: Path,
    *,
    device: str = "cpu",
) -> tuple[Any, str]:
    """Load either immutable published-v1 or native v2 runner policy."""

    import torch

    source = Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    action_count = int(payload.get("action_count", 36)) if isinstance(payload, dict) else 36
    if action_count == 288:
        from ghostline.model_v2 import load_runner_v2

        policy = load_runner_v2(source, device=device)
    elif action_count == 36:
        from ghostline.model import load_policy

        policy = load_policy(source, device=device)
    else:
        raise RuntimeError(
            f"{source} declares unsupported runner action count {action_count}"
        )
    return policy, runner_opponent_kind(policy)
