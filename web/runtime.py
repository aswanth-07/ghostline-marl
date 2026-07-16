from __future__ import annotations

import asyncio
import json
import platform
from pathlib import Path
import sys
import time
from types import ModuleType
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


def browser_prefers_touch(host: Any) -> bool:
    """Detect a touch-first browser without making touch a web dependency."""

    touch_points = 0
    try:
        navigator = host.navigator
        touch_points = int(getattr(navigator, "maxTouchPoints", 0) or 0)
    except Exception:
        pass
    try:
        coarse = bool(host.matchMedia("(pointer: coarse)").matches)
        fine = bool(host.matchMedia("(pointer: fine)").matches)
        constrained = bool(
            host.matchMedia("(max-width: 980px), (max-height: 600px)").matches
        )
        # A hybrid Windows laptop commonly reports touch points while retaining
        # a fine primary pointer. Do not permanently cover its desktop canvas
        # with phone controls; real finger events can still opt in later.
        return coarse or (touch_points > 0 and constrained and not fine)
    except Exception:
        try:
            constrained = int(host.innerWidth) <= 980 or int(host.innerHeight) <= 600
        except Exception:
            constrained = False
        return touch_points > 0 and constrained


def _browser_gymnasium_shim() -> ModuleType:
    """Return the tiny Gymnasium surface used by the browser policy adapter.

    Pygbag's import rewriter eagerly follows Gymnasium's unused vector package
    into ``multiprocessing.sharedctypes``, which CPython/WASM does not ship.
    Ghostline's web agent needs only ``Env.reset`` seeding and declarative space
    records, so keeping that compatibility layer here avoids changing the real
    desktop/training environment or pretending multiprocessing exists.
    """
    import numpy as np

    module = ModuleType("gymnasium")
    spaces_module = ModuleType("gymnasium.spaces")

    class Env:
        @classmethod
        def __class_getitem__(cls, _item: Any) -> type[Env]:
            return cls

        def reset(self, *, seed: int | None = None, options: Any = None) -> None:
            del options
            if seed is not None or not hasattr(self, "np_random"):
                self.np_random = np.random.default_rng(seed)

    class Discrete:
        def __init__(self, n: int):
            self.n = int(n)

    class Box:
        def __init__(self, low: Any, high: Any, *, shape: tuple[int, ...], dtype: Any):
            self.low = low
            self.high = high
            self.shape = tuple(shape)
            self.dtype = np.dtype(dtype)

    class DictSpace:
        def __init__(self, spaces: Mapping[str, Any]):
            self.spaces = dict(spaces)

    Env.__module__ = module.__name__
    Discrete.__module__ = spaces_module.__name__
    Box.__module__ = spaces_module.__name__
    DictSpace.__module__ = spaces_module.__name__
    spaces_module.Discrete = Discrete
    spaces_module.Box = Box
    spaces_module.Dict = DictSpace
    module.Env = Env
    module.spaces = spaces_module
    return module


def _install_browser_gymnasium_shim() -> None:
    if "gymnasium" in sys.modules:
        return
    module = _browser_gymnasium_shim()
    sys.modules["gymnasium"] = module
    sys.modules["gymnasium.spaces"] = module.spaces


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


class BrowserOnnxPolicy:
    """Synchronous facade over the bridge's coalescing asynchronous inference queue."""

    def __init__(self, host: Any):
        self.host = host
        self.prefetched = False
        self._prefetch_inference_count = -1
        self.waiting_for_action = False

    def prefetch(self, observation: Mapping[str, Any]) -> None:
        if self.prefetched:
            return
        self._prefetch_inference_count = int(self.host.ghostlinePolicy.inferenceCount)
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
        if not self.prefetched:
            # Browser inference is asynchronous. Queue this observation and
            # fail closed for the current decision instead of pretending that
            # the bridge's immediate neutral return is the finished result.
            self.prefetch(observation)
            self.waiting_for_action = True
            return 0, None
        if not bool(
            self.host.ghostlinePolicy.hasCompletedAction(
                self._prefetch_inference_count
            )
        ):
            # Keep the outstanding generation alive. Clearing it here caused
            # every 60 ms WebGPU result to be replaced before Python consumed
            # it, leaving the production agent permanently on HOLD.
            self.waiting_for_action = True
            return 0, None
        action = self.host.ghostlinePolicy.currentAction()
        self.prefetched = False
        self._prefetch_inference_count = -1
        self.waiting_for_action = False
        return int(action), None


class GhostlineWebRuntime:
    """Connect the static web shell to an otherwise unchanged ``GameApp``."""

    def __init__(self, app: Any, *, host: Any | None = None):
        self.app = app
        self.host = host if host is not None else platform.window
        self.app.touch_controls_enabled = browser_prefers_touch(self.host)
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
        elif kind == "agent-ready":
            await self._enable_agent(tier=tier, seed=seed, fresh=bool(command.get("fresh", False)))
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
            self.app.audio.set_focus_active(False)
            # Agent showcase runs do not depend on keyboard focus. Human play
            # must never spend mission time while its iframe/tab has no input.
            if self.app.state == "play" and not self.app.sim.terminated and not self.app.sim.truncated:
                self.app.state = "pause"
                self.app.selection = 0
                self.host.ghostlineShell.showNotice("Mission paused after keyboard focus was lost.", "info")
        elif kind == "reset-policy":
            self.host.ghostlinePolicy.reset()
        elif kind == "focus":
            self.app.audio.set_focus_active(True)
            self.host.document.getElementById("canvas").focus()

    async def _enable_agent(self, *, tier: int, seed: int | None, fresh: bool = False) -> None:
        self.host.ghostlineShell.setPolicyState("loading", "Loading recurrent policy…")
        if str(self.host.ghostlinePolicy.state) != "ready":
            self.host.ghostlineShell.setPolicyState("unavailable", "Agent unavailable — human play still works")
            self.host.ghostlineShell.showNotice("The policy could not load. Continuing in human mode.", "error")
            return

        self.host.ghostlinePolicy.reset()
        self.policy.prefetched = False
        self.policy._prefetch_inference_count = -1
        self.policy.waiting_for_action = False
        self.app.learned_policy = self.policy
        self.app.selected_tier = tier
        self.app.seed = seed
        _install_browser_gymnasium_shim()
        active_mission = (
            not fresh
            and self.app.state in {"play", "pause", "lab_play"}
            and not self.app.sim.terminated
            and not self.app.sim.truncated
        )
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
        # Do not claim visible control until the browser has produced a real
        # first decision. Cold WebGPU/WASM initialization can take long enough
        # that an immediate "AGENT CONTROL" label looks like a dead button.
        self.host.ghostlineShell.setControlMode("handoff")
        self.host.ghostlineShell.setPolicyState("loading", "Agent acquiring first action…")
        self.host.ghostlineShell.showNotice(
            "Agent connected. Acquiring its first policy decision…",
            "info",
        )
        # The first game decision queues its own observation. Subsequent
        # decisions are phase-advanced by the control loop below.
        self._last_prefetch_tick = -1

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
        self.policy._prefetch_inference_count = -1
        self.policy.waiting_for_action = False
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
        """Queue each decision from its exact 10 Hz simulation boundary."""
        if self.app.state != "lab_play" or self.app.agent_env is None:
            self._last_prefetch_tick = -1
            return
        tick = int(self.app._agent_tick)
        if (
            tick < 0
            or tick % 6 != 0
            or tick == self._last_prefetch_tick
            or self.policy.prefetched
        ):
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
