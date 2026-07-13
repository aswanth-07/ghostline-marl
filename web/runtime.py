from __future__ import annotations

import asyncio
import inspect
import json
import platform
from pathlib import Path
import time
from typing import Any, Mapping


OBSERVATION_KEYS = (
    "ego",
    "objective",
    "local_grid",
    "targets",
    "target_mask",
    "entities",
    "entity_mask",
    "rays",
    "action_mask",
)
PROGRESSION_STORAGE_KEY = "ghostline.progression-v1"


def hydrate_progression(host: Any, path: Path) -> bool:
    """Restore the normal progression JSON from browser localStorage."""
    try:
        raw = host.localStorage.getItem(PROGRESSION_STORAGE_KEY)
        if raw is None:
            return False
        text = str(raw)
        if len(text.encode("utf-8")) > 1_000_000:
            return False
        parsed = json.loads(text)
        if not isinstance(parsed, dict) or int(parsed.get("version", 0)) != 1:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(parsed, separators=(",", ":")), encoding="utf-8")
        return True
    except Exception:
        return False


def persist_progression(host: Any, path: Path) -> bool:
    """Mirror progression to localStorage; storage denial never blocks play."""
    try:
        if not path.is_file():
            return False
        text = path.read_text(encoding="utf-8")
        parsed = json.loads(text)
        if not isinstance(parsed, dict) or int(parsed.get("version", 0)) != 1:
            return False
        host.localStorage.setItem(PROGRESSION_STORAGE_KEY, json.dumps(parsed, separators=(",", ":")))
        return True
    except Exception:
        return False


def observation_json(observation: Mapping[str, Any]) -> str:
    """Serialize the player-equivalent observation without leaking simulation state."""
    missing = [key for key in OBSERVATION_KEYS if key not in observation]
    if missing:
        raise KeyError(f"Web policy observation is missing: {', '.join(missing)}")
    payload = {
        key: value.tolist() if hasattr(value, "tolist") else value
        for key, value in observation.items()
        if key in OBSERVATION_KEYS
    }
    return json.dumps(payload, separators=(",", ":"), allow_nan=False)


async def _await_js(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


class BrowserOnnxPolicy:
    """Synchronous facade over the bridge's coalescing asynchronous inference queue."""

    def __init__(self, host: Any):
        self.host = host
        self.prefetched = False

    def prefetch(self, observation: Mapping[str, Any]) -> None:
        self.host.ghostlinePolicy.step(observation_json(observation))
        self.prefetched = True

    def act(
        self,
        observation: Mapping[str, Any],
        hidden: Any = None,
        *,
        deterministic: bool = True,
        device: str = "cpu",
    ) -> tuple[int, None]:
        del hidden, deterministic, device
        if self.prefetched:
            action = self.host.ghostlinePolicy.currentAction()
            self.prefetched = False
        else:
            action = self.host.ghostlinePolicy.step(observation_json(observation))
        return int(action), None


class GhostlineWebRuntime:
    """Connect the static web shell to an otherwise unchanged ``GameApp``."""

    def __init__(self, app: Any, *, host: Any | None = None):
        self.app = app
        self.host = host if host is not None else platform.window
        self.policy = BrowserOnnxPolicy(self.host)
        self.control_mode = "human"
        self.run_mode = "human"
        self._last_metrics = 0.0
        self._last_progression_snapshot = ""
        self._last_prefetch_tick = -1

    async def run(self) -> int:
        self.host.ghostlineShell.markGameReady()
        game = asyncio.create_task(self.app.run_async())
        controls = asyncio.create_task(self._control_loop())
        try:
            return int(await game)
        finally:
            controls.cancel()
            try:
                await controls
            except asyncio.CancelledError:
                pass

    async def _control_loop(self) -> None:
        while self.app.running:
            command = self._next_command()
            if command:
                await self._handle(command)
            self._prefetch_agent_action()
            now = time.monotonic()
            if now - self._last_metrics >= 0.2:
                self._publish_metrics()
                self._persist_if_changed()
                self._last_metrics = now
            await asyncio.sleep(0)

    def _next_command(self) -> dict[str, Any] | None:
        raw = self.host.ghostlineShell.consumeCommand()
        if raw is None:
            return None
        text = str(raw)
        if not text or text in {"null", "undefined", "None"}:
            return None
        try:
            command = json.loads(text)
        except (TypeError, ValueError):
            self.host.ghostlineShell.showNotice("The browser sent an invalid control command.", "error")
            return None
        return command if isinstance(command, dict) else None

    async def _handle(self, command: Mapping[str, Any]) -> None:
        kind = str(command.get("type", ""))
        tier = max(1, min(6, int(command.get("tier", self.app.selected_tier))))
        seed_value = command.get("seed")
        seed = int(seed_value) if seed_value not in (None, "") else None
        if seed is not None:
            seed = max(0, min(2_147_483_647, seed))

        if kind == "launch-human":
            self.app.selected_tier = tier
            self.app.seed = seed
            self.control_mode = "human"
            self.run_mode = "human"
            self.app._start_mission(agent=False)
            self.host.ghostlineShell.setControlMode("human")
        elif kind == "agent":
            await self._enable_agent(tier=tier, seed=seed)
        elif kind == "human":
            active_mission = self.app.state in {"play", "pause", "lab_play"} and not self.app.sim.terminated and not self.app.sim.truncated
            if active_mission and self.run_mode == "agent":
                self._mark_hybrid_run()
            self.control_mode = "human"
            if self.app.state == "lab_play" and not self.app.sim.terminated and not self.app.sim.truncated:
                self.app.state = "play"
            self._release_agent_environment()
            self.host.ghostlineShell.setControlMode("human")
            self.host.ghostlineShell.showNotice("Manual control restored.", "success")
        elif kind == "policy-failed":
            self._restore_human_after_policy_failure()
        elif kind in {"pause-hidden", "pause-focus"}:
            # Agent showcase runs do not depend on keyboard focus. Human play
            # must never spend mission time while its iframe/tab has no input.
            if self.app.state == "play" and not self.app.sim.terminated and not self.app.sim.truncated:
                self.app.state = "pause"
                self.app.selection = 0
                self.host.ghostlineShell.showNotice("Mission paused after keyboard focus was lost.", "info")
        elif kind == "reset-policy":
            self.host.ghostlinePolicy.reset()
        elif kind == "focus":
            self.host.document.getElementById("canvas").focus()

    async def _enable_agent(self, *, tier: int, seed: int | None) -> None:
        self.host.ghostlineShell.setPolicyState("loading", "Loading recurrent policy…")
        loaded = bool(await _await_js(self.host.ghostlinePolicy.load()))
        if not loaded:
            self.host.ghostlineShell.setPolicyState("unavailable", "Agent unavailable — human play still works")
            self.host.ghostlineShell.showNotice("The policy could not load. Continuing in human mode.", "error")
            return

        self.host.ghostlinePolicy.reset()
        self.policy.prefetched = False
        self.app.learned_policy = self.policy
        self.app.selected_tier = tier
        self.app.seed = seed
        active_mission = self.app.state in {"play", "pause", "lab_play"} and not self.app.sim.terminated and not self.app.sim.truncated
        if active_mission:
            if self.run_mode != "agent":
                self._mark_hybrid_run()
            self._release_agent_environment()
            environment_type = getattr(__import__("ghostline.env", fromlist=["GhostlineEnv"]), "GhostlineEnv")
            environment = environment_type(seed=self.app.sim.seed, tier=self.app.sim.tier)
            environment.sim = self.app.sim
            environment._distance_cache.clear()
            self.app.agent_env = environment
            self.app.agent_observation = environment._observation()
            self.app.agent_hidden = None
            self.app._agent_action = 0
            self.app._agent_tick = 0
            self.app.state = "lab_play"
        else:
            self.app._start_mission(agent=True)
            self.run_mode = "agent"
        self.control_mode = "agent"
        self.host.ghostlineShell.setControlMode("agent")
        self.host.ghostlineShell.setPolicyState("ready", "Agent online")
        self.host.ghostlineShell.showNotice("Agent takeover engaged. Use TAKE CONTROL at any time.", "success")

    def _mark_hybrid_run(self) -> None:
        """Exclude mixed-control runs from pure human/agent comparisons."""

        self.run_mode = "hybrid"
        telemetry = getattr(self.app, "_telemetry", None)
        if isinstance(telemetry, dict) and telemetry:
            telemetry["controller"] = "hybrid"
            telemetry["policy"] = "human + recurrent policy"
            telemetry.setdefault("takeover_elapsed_seconds", round(float(self.app.sim.elapsed_seconds), 3))
            telemetry.setdefault("takeover_data", int(self.app.sim.data))

    def _restore_human_after_policy_failure(self) -> None:
        """Fail safely from a live browser policy without replaying its last action."""

        if self.control_mode != "agent":
            return
        active_mission = (
            self.app.state in {"play", "pause", "lab_play"}
            and not self.app.sim.terminated
            and not self.app.sim.truncated
        )
        # A backend failure must never leave a result eligible as a pure agent
        # run, including the race where the neutral fallback tick terminates
        # the mission before this command reaches Python.
        if self.run_mode in {"agent", "hybrid"}:
            self._mark_hybrid_run()
        if self.app.state == "lab_play" and active_mission:
            self.app.state = "play"
        self.control_mode = "human"
        self.policy.prefetched = False
        self._last_prefetch_tick = -1
        self.app._agent_action = 0
        self.app.learned_policy = None
        self._release_agent_environment()
        self.host.ghostlinePolicy.reset()
        self.host.ghostlineShell.setControlMode("human")
        self.host.ghostlineShell.setPolicyState(
            "unavailable",
            "Agent offline — manual control restored",
        )
        self.host.ghostlineShell.showNotice(
            "Agent inference stopped. This run is now hybrid and manual control is active.",
            "error",
        )

    def _release_agent_environment(self) -> None:
        environment = getattr(self.app, "agent_env", None)
        if environment is not None:
            environment.close()
        self.app.agent_env = None
        self.app.agent_observation = None
        self.app.agent_hidden = None

    def _prefetch_agent_action(self) -> None:
        """Use the spare frame before each 10 Hz decision to hide async browser latency."""
        if self.app.state != "lab_play" or self.app.agent_env is None:
            self._last_prefetch_tick = -1
            return
        tick = int(self.app._agent_tick)
        if tick <= 0 or tick % 6 or tick == self._last_prefetch_tick:
            return
        self.policy.prefetch(self.app.agent_env._observation())
        self._last_prefetch_tick = tick

    def _publish_metrics(self) -> None:
        sim = self.app.sim
        payload = {
            "mode": self.run_mode,
            "controller": self.control_mode,
            "tier": int(sim.tier),
            "seed": int(sim.seed),
            "status": "success" if sim.extracted else ("failed" if sim.terminated or sim.truncated else "active"),
            "data": int(sim.data),
            "quota": int(sim.level.quota),
            "time": round(float(sim.elapsed_seconds), 2),
            "trace": round(float(sim.trace), 2),
            "damage": int(sim.damage_taken),
            "detections": int(sim.detections),
            "distance": round(float(sim.distance_travelled), 1),
        }
        self.host.ghostlineShell.updateMetrics(json.dumps(payload, separators=(",", ":")))

    def _persist_if_changed(self) -> None:
        from ghostline.progression import progression_path

        path = progression_path()
        try:
            snapshot = path.read_text(encoding="utf-8") if path.is_file() else ""
        except OSError:
            return
        if snapshot and snapshot != self._last_progression_snapshot and persist_progression(self.host, path):
            self._last_progression_snapshot = snapshot
