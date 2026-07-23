from __future__ import annotations

import importlib
import time
from typing import Any

import numpy as np

from ghostline.config_v3 import SECURITY_TACTICAL_TICKS
from ghostline.resources import runtime_asset_path
from ghostline.security_baselines import tactical_security_action
from ghostline.simulation import norm
from ghostline.simulation_v3 import GhostlineSimulationV3
from ghostline.types import GuardMode
from ghostline.types_v3 import GuardRole, RadioMessage, SecurityIntent, SecurityOrder


class AdaptiveSecurityController:
    """5 Hz semantic team controller used by human Adaptive Contracts.

    A trained shared policy is loaded when present.  The deterministic fallback
    consumes the exact same local observations and masks, keeping the game mode
    available in lightweight builds that intentionally omit PyTorch.
    """

    def __init__(self, sim: GhostlineSimulationV3):
        self.policy = None
        self.adapter = None
        self.hidden: dict[str, Any] = {}
        self.policy_name = "TACTICAL RULE TEAM"
        self.last_latency_ms = 0.0
        self.decisions = 0
        self.last_orders: dict[int, SecurityOrder] = {}
        try:
            environment_type = getattr(
                importlib.import_module("ghostline.security_env"),
                "GhostlineSecurityParallelEnv",
            )
            self.adapter = environment_type(tier=sim.tier, seed=sim.seed)
        except Exception:
            self.adapter = None
        with runtime_asset_path("models/ghostline-security.pt") as checkpoint:
            if checkpoint is not None:
                try:
                    loader = getattr(importlib.import_module("ghostline.security_model"), "load_security_policy")
                    self.policy = loader(checkpoint)
                    self.policy_name = "RECURRENT MAPPO SECURITY"
                except Exception:
                    self.policy = None
        self.reset(sim)

    def reset(self, sim: GhostlineSimulationV3) -> None:
        self.sim = sim
        self.sim.external_security = True
        self.hidden = {f"guard_{guard.guard_id}": None for guard in sim.level.guards}
        self.last_orders = {}
        self.last_latency_ms = 0.0
        self.decisions = 0
        if self.adapter is not None:
            self.adapter.sim = sim
            self.adapter.tier = sim.tier
            self.adapter.agents = list(self.hidden)
            self.adapter._target_cache.clear()
            self.adapter._invalid_actions = 0

    def update(self, *, force: bool = False) -> None:
        if not force and self.sim.elapsed_ticks % SECURITY_TACTICAL_TICKS != 0:
            return
        started = time.perf_counter()
        if self.adapter is None:
            orders = self._direct_fallback_orders()
        else:
            self.adapter.sim = self.sim
            self.adapter.agents = [f"guard_{guard.guard_id}" for guard in self.sim.level.guards]
            observations = {
                agent: self.adapter._observation(agent)
                for agent in self.adapter.agents
            }
            actions: dict[str, np.ndarray] = {}
            for agent, observation in observations.items():
                if self.policy is None:
                    actions[agent] = self._rule_action(observation, int(agent.rsplit("_", 1)[1]))
                else:
                    actions[agent], self.hidden[agent] = self.policy.act(
                        observation,
                        self.hidden.get(agent),
                        deterministic=True,
                    )
            orders, _ = self.adapter.orders_from_actions(actions, observations=observations)
        self.last_orders = orders
        self.sim.set_security_orders(orders)
        self.decisions += 1
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0

    def _rule_action(self, observation: dict[str, np.ndarray], guard_id: int) -> np.ndarray:
        return tactical_security_action(observation, guard_id)

    def _direct_fallback_orders(self) -> dict[int, SecurityOrder]:
        orders: dict[int, SecurityOrder] = {}
        for guard in self.sim.level.guards:
            state = self.sim.operative_states[guard.guard_id]
            visible = self.sim.visible(guard.position, guard.facing, self.sim.player, distance=245.0, cosine=0.45)
            if visible:
                target = self.sim.player.copy()
                message = RadioMessage.SIGHTING
                if state.role == GuardRole.SUPPRESSOR:
                    intent = SecurityIntent.HOLD
                    ability = True
                elif state.role == GuardRole.INTERCEPTOR:
                    intent = SecurityIntent.FLANK_LEFT if guard.guard_id % 2 == 0 else SecurityIntent.FLANK_RIGHT
                    delta = target - guard.position
                    direction = delta / max(1e-6, norm(delta))
                    perpendicular = np.asarray((-direction[1], direction[0]), dtype=np.float32) * 64.0
                    target = target + (perpendicular if intent == SecurityIntent.FLANK_LEFT else -perpendicular)
                    ability = False
                else:
                    intent, ability = SecurityIntent.PURSUE, False
            elif self.sim.quota_met:
                intent, target = SecurityIntent.INTERCEPT, self.sim.level.extraction.copy()
                message, ability = RadioMessage.REQUEST_INTERCEPT, False
            elif state.heard_confidence > 0.02:
                intent, target = SecurityIntent.INVESTIGATE, state.heard_position.copy()
                message, ability = RadioMessage.SUSPECTED_ROUTE, False
            elif guard.mode in (GuardMode.INVESTIGATE, GuardMode.SEARCH, GuardMode.CHASE):
                intent, target = SecurityIntent.SEARCH, guard.last_known.copy()
                message, ability = RadioMessage.NONE, False
            else:
                intent, target = SecurityIntent.PATROL, guard.patrol[guard.patrol_index].copy()
                message, ability = RadioMessage.NONE, False
            orders[guard.guard_id] = SecurityOrder(intent, target.astype(np.float32), message, ability)
        return orders

    def telemetry(self) -> dict[str, Any]:
        return {
            "security_policy": self.policy_name,
            "security_decisions": self.decisions,
            "security_latency_ms": self.last_latency_ms,
            "security_intents": {
                str(guard_id): order.intent.name.lower()
                for guard_id, order in self.last_orders.items()
            },
        }

    def close(self) -> None:
        if self.adapter is not None:
            self.adapter.close()
