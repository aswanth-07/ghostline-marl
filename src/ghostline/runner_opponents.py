from __future__ import annotations

from typing import Any

import numpy as np

from ghostline.env import GhostlineEnv
from ghostline.simulation_v3 import GhostlineSimulationV3


class FrozenV2RunnerOpponent:
    """Run the published Env-v2 policy inside an Env-v3 security match.

    Env-v3 is additive: the frozen runner deliberately retains its original 36
    actions and cannot use the new decoy.  It does, however, receive the same
    public live geometry with temporary locks represented as blocked cells.
    This provides a stable, provenance-bound opponent for security training.
    """

    def __init__(self, policy: Any, *, device: str = "cpu"):
        self.policy = policy
        self.device = device
        self.hidden = None
        self.env: GhostlineEnv | None = None
        self._sim_identity: int | None = None
        self._topology_signature: tuple[tuple[tuple[int, int], bool], ...] = ()

    def reset(self, sim: GhostlineSimulationV3) -> None:
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
    def _topology(sim: GhostlineSimulationV3) -> tuple[tuple[tuple[int, int], bool], ...]:
        return tuple((door.tile, bool(door.locked)) for door in sim.security_doors)

    def __call__(self, sim: GhostlineSimulationV3) -> int:
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
